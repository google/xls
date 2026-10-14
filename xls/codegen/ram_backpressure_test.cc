// Copyright 2026 The XLS Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Cycle-level tests of RAM channels under backpressure, run against both
// codegen 1.0 and codegen 1.5.
//
// Each test generates a block for a proc that uses a RAM (including the
// RamRewritePass), then simulates it with the block interpreter, driving
// random input validity & output backpressure, and modeling the RAM with a
// fixed latency of 1 cycle.

#include <cstdint>
#include <deque>
#include <memory>
#include <optional>
#include <random>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "gtest/gtest.h"
#include "absl/container/flat_hash_map.h"
#include "absl/random/distributions.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "xls/codegen/codegen_result.h"
#include "xls/codegen/module_signature.pb.h"
#include "xls/common/status/matchers.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/interpreter/block_evaluator.h"
#include "xls/interpreter/block_interpreter.h"
#include "xls/ir/bits.h"
#include "xls/ir/block.h"
#include "xls/ir/events.h"
#include "xls/ir/ir_parser.h"
#include "xls/ir/nodes.h"
#include "xls/ir/package.h"
#include "xls/ir/value.h"
#include "xls/scheduling/scheduling_result.h"
#include "xls/tools/codegen.h"
#include "xls/tools/codegen_flags.pb.h"
#include "xls/tools/scheduling_options_flags.pb.h"

namespace xls {
namespace {

// A line-buffer-like proc using a 1R1W RAM:
//
//   stage 0: x = recv(in); [optionally send(side, x)]; send(rd_req, cnt)
//   stage 1: old = recv(rd_resp); send(wr_req, (cnt, x)); send(out, old + x)
//   stage 2: recv(wr_comp)
//
// so out[k] = x[k] + x[k - 16] (or x[k] for k < 16).
//
// Backpressure on `out` stalls stage 1, the consumer of the read response, so
// the read for the next activation may be issued before the current response
// is consumed.
std::string Ram1R1WProcIr(bool with_side_output) {
  const std::string side_channel =
      with_side_output ? "chan side(bits[32], id=6, kind=streaming, "
                         "ops=send_only, flow_control=ready_valid)\n"
                       : "";
  const std::string side_send =
      with_side_output
          ? "  send_side: token = send(rcv_in_tok, x, channel=side)\n"
          : "";
  const std::string rd_req_token =
      with_side_output ? "send_side" : "rcv_in_tok";
  return absl::StrCat(R"(package ram_backpressure

chan in(bits[32], id=0, kind=streaming, ops=receive_only, flow_control=ready_valid)
chan out(bits[32], id=1, kind=streaming, ops=send_only, flow_control=ready_valid)
chan rd_req((bits[4], ()), id=2, kind=streaming, ops=send_only, flow_control=ready_valid)
chan rd_resp((bits[32]), id=3, kind=streaming, ops=receive_only, flow_control=ready_valid)
chan wr_req((bits[4], bits[32], ()), id=4, kind=streaming, ops=send_only, flow_control=ready_valid)
chan wr_comp((), id=5, kind=streaming, ops=receive_only, flow_control=ready_valid)
)",
                      side_channel, R"(
proc ram_proc(cnt: bits[4], init={0}) {
  tkn: token = literal(value=token)
  rcv_in: (token, bits[32]) = receive(tkn, channel=in)
  rcv_in_tok: token = tuple_index(rcv_in, index=0)
  x: bits[32] = tuple_index(rcv_in, index=1)
  empty: () = literal(value=())
)",
                      side_send,
                      R"(  rd_req_data: (bits[4], ()) = tuple(cnt, empty)
  send_rd: token = send()",
                      rd_req_token, R"(, rd_req_data, channel=rd_req)
  rcv_rd: (token, (bits[32])) = receive(send_rd, channel=rd_resp)
  rcv_rd_tok: token = tuple_index(rcv_rd, index=0)
  rd_tuple: (bits[32]) = tuple_index(rcv_rd, index=1)
  old: bits[32] = tuple_index(rd_tuple, index=0)
  wr_req_data: (bits[4], bits[32], ()) = tuple(cnt, x, empty)
  send_wr: token = send(rcv_rd_tok, wr_req_data, channel=wr_req)
  rcv_wc: (token, ()) = receive(send_wr, channel=wr_comp)
  sum: bits[32] = add(old, x)
  send_out: token = send(rcv_rd_tok, sum, channel=out)
  one: bits[4] = literal(value=1)
  next_cnt: bits[4] = add(cnt, one)
  next_value.200: () = next_value(param=cnt, value=next_cnt)
}
)");
}

// The same line-buffer pattern, using a single-port (1RW) RAM: each activation
// issues one request that reads (the old value) & writes (x) the same address.
// With read-before-write semantics, out[k] = x[k] + x[k - 16].
constexpr std::string_view kRam1RWProcIr = R"(package ram1rw_backpressure

chan in(bits[32], id=0, kind=streaming, ops=receive_only, flow_control=ready_valid)
chan out(bits[32], id=1, kind=streaming, ops=send_only, flow_control=ready_valid)
chan req((bits[4], bits[32], (), (), bits[1], bits[1]), id=2, kind=streaming, ops=send_only, flow_control=ready_valid)
chan resp((bits[32]), id=3, kind=streaming, ops=receive_only, flow_control=ready_valid)
chan wr_comp((), id=4, kind=streaming, ops=receive_only, flow_control=ready_valid)

proc ram_proc(cnt: bits[4], init={0}) {
  tkn: token = literal(value=token)
  rcv_in: (token, bits[32]) = receive(tkn, channel=in)
  rcv_in_tok: token = tuple_index(rcv_in, index=0)
  x: bits[32] = tuple_index(rcv_in, index=1)
  empty: () = literal(value=())
  one1: bits[1] = literal(value=1)
  req_data: (bits[4], bits[32], (), (), bits[1], bits[1]) = tuple(cnt, x, empty, empty, one1, one1)
  send_req: token = send(rcv_in_tok, req_data, channel=req)
  rcv_resp: (token, (bits[32])) = receive(send_req, channel=resp)
  rcv_resp_tok: token = tuple_index(rcv_resp, index=0)
  resp_tuple: (bits[32]) = tuple_index(rcv_resp, index=1)
  old: bits[32] = tuple_index(resp_tuple, index=0)
  rcv_wc: (token, ()) = receive(send_req, channel=wr_comp)
  sum: bits[32] = add(old, x)
  send_out: token = send(rcv_resp_tok, sum, channel=out)
  one: bits[4] = literal(value=1)
  next_cnt: bits[4] = add(cnt, one)
  next_value.200: () = next_value(param=cnt, value=next_cnt)
}
)";

// A proc whose RAM accesses & output are predicated on mutually exclusive
// phases. Activations cycle through 4 phases:
//
//   phase 0: read mem[addr] (stage 0), receive it (stage 1), and save
//            `last = old + y`, where y comes from `in2` in stage 1
//   phase 1: send(out, last)
//   phase 2: write mem[addr] = x
//   phase 3: nothing
//
// Every activation receives from `in` (stage 0) & `in2` (stage 1); `in2` can
// stall the stage that consumes the read response.
constexpr std::string_view kRamPredicatedProcIr = R"(package ram_predicated

chan in(bits[32], id=0, kind=streaming, ops=receive_only, flow_control=ready_valid)
chan in2(bits[32], id=1, kind=streaming, ops=receive_only, flow_control=ready_valid)
chan out(bits[32], id=2, kind=streaming, ops=send_only, flow_control=ready_valid)
chan rd_req((bits[4], ()), id=3, kind=streaming, ops=send_only, flow_control=ready_valid)
chan rd_resp((bits[32]), id=4, kind=streaming, ops=receive_only, flow_control=ready_valid)
chan wr_req((bits[4], bits[32], ()), id=5, kind=streaming, ops=send_only, flow_control=ready_valid)
chan wr_comp((), id=6, kind=streaming, ops=receive_only, flow_control=ready_valid)

proc ram_proc(cnt: bits[6], last: bits[32], init={0, 0}) {
  tkn: token = literal(value=token)
  phase: bits[2] = bit_slice(cnt, start=0, width=2)
  addr: bits[4] = bit_slice(cnt, start=2, width=4)
  zero2: bits[2] = literal(value=0)
  one2: bits[2] = literal(value=1)
  two2: bits[2] = literal(value=2)
  do_read: bits[1] = eq(phase, zero2)
  do_out: bits[1] = eq(phase, one2)
  do_write: bits[1] = eq(phase, two2)
  rcv_in: (token, bits[32]) = receive(tkn, channel=in)
  rcv_in_tok: token = tuple_index(rcv_in, index=0)
  x: bits[32] = tuple_index(rcv_in, index=1)
  empty: () = literal(value=())
  rd_req_data: (bits[4], ()) = tuple(addr, empty)
  send_rd: token = send(rcv_in_tok, rd_req_data, predicate=do_read, channel=rd_req)
  rcv_rd: (token, (bits[32])) = receive(send_rd, predicate=do_read, channel=rd_resp)
  rcv_rd_tok: token = tuple_index(rcv_rd, index=0)
  rd_tuple: (bits[32]) = tuple_index(rcv_rd, index=1)
  old: bits[32] = tuple_index(rd_tuple, index=0)
  rcv_in2: (token, bits[32]) = receive(rcv_rd_tok, channel=in2)
  rcv_in2_tok: token = tuple_index(rcv_in2, index=0)
  y: bits[32] = tuple_index(rcv_in2, index=1)
  sum: bits[32] = add(old, y)
  send_out: token = send(rcv_in2_tok, last, predicate=do_out, channel=out)
  wr_req_data: (bits[4], bits[32], ()) = tuple(addr, x, empty)
  send_wr: token = send(rcv_in_tok, wr_req_data, predicate=do_write, channel=wr_req)
  rcv_wc: (token, ()) = receive(send_wr, predicate=do_write, channel=wr_comp)
  one6: bits[6] = literal(value=1)
  next_cnt: bits[6] = add(cnt, one6)
  next_value.300: () = next_value(param=cnt, value=next_cnt)
  next_value.301: () = next_value(param=last, value=sum, predicate=do_read)
}
)";

constexpr std::string_view kRam1R1WConfig =
    "ram:1R1W:rd_req:rd_resp:wr_req:wr_comp";
constexpr std::string_view kRam1RWConfig = "ram:1RW:req:resp:wr_comp";

// A proc whose only streaming send is a request to a single-port (1RW) RAM, so
// one-shot logic isn't needed to keep multiple sends from repeating. Each
// activation k reads the old value at address k % 16 (asserting that it's the
// value written 16 activations earlier, or 0) & writes k there. Each activation
// then receives from `in`, which can stall the stage that consumes the read
// response.
constexpr std::string_view kRamOnlySendProcIr = R"(package ram_only_send

chan in(bits[32], id=0, kind=streaming, ops=receive_only, flow_control=ready_valid)
chan req((bits[4], bits[32], (), (), bits[1], bits[1]), id=1, kind=streaming, ops=send_only, flow_control=ready_valid)
chan resp((bits[32]), id=2, kind=streaming, ops=receive_only, flow_control=ready_valid)
chan wr_comp((), id=3, kind=streaming, ops=receive_only, flow_control=ready_valid)

proc ram_proc(cnt: bits[32], init={0}) {
  tkn: token = literal(value=token)
  addr: bits[4] = bit_slice(cnt, start=0, width=4)
  empty: () = literal(value=())
  one1: bits[1] = literal(value=1)
  req_data: (bits[4], bits[32], (), (), bits[1], bits[1]) = tuple(addr, cnt, empty, empty, one1, one1)
  send_req: token = send(tkn, req_data, channel=req)
  rcv_resp: (token, (bits[32])) = receive(send_req, channel=resp)
  rcv_resp_tok: token = tuple_index(rcv_resp, index=0)
  resp_tuple: (bits[32]) = tuple_index(rcv_resp, index=1)
  old: bits[32] = tuple_index(resp_tuple, index=0)
  rcv_wc: (token, ()) = receive(send_req, channel=wr_comp)
  rcv_in: (token, bits[32]) = receive(rcv_resp_tok, channel=in)
  rcv_in_tok: token = tuple_index(rcv_in, index=0)
  sixteen: bits[32] = literal(value=16)
  zero: bits[32] = literal(value=0)
  is_early: bits[1] = ult(cnt, sixteen)
  cnt_minus_16: bits[32] = sub(cnt, sixteen)
  expected_old: bits[32] = sel(is_early, cases=[cnt_minus_16, zero])
  read_ok: bits[1] = eq(old, expected_old)
  assert_read_ok: token = assert(rcv_in_tok, read_ok, message="RAM read returned the wrong value", label="read_ok")
  one: bits[32] = literal(value=1)
  next_cnt: bits[32] = add(cnt, one)
  next_value.100: () = next_value(param=cnt, value=next_cnt)
}
)";

struct CodegenConfig {
  CodegenVersionProto version;
  int64_t pipeline_stages = 3;
  // If set, all (non-RAM) inputs & outputs are flopped with this kind of
  // register.
  std::optional<IOKindProto> io_kind;

  std::string ToString() const {
    std::string io = "NoFlops";
    if (io_kind.has_value()) {
      switch (*io_kind) {
        case IO_KIND_FLOP:
          io = "Flops";
          break;
        case IO_KIND_SKID_BUFFER:
          io = "SkidBuffers";
          break;
        case IO_KIND_ZERO_LATENCY_BUFFER:
          io = "ZeroLatencyBuffers";
          break;
        default:
          io = "UnknownFlops";
          break;
      }
    }
    return absl::StrFormat(
        "Codegen%s_%s_%dStages",
        version == CODEGEN_VERSION_ONE_DOT_ZERO ? "1_0" : "1_5", io,
        pipeline_stages);
  }
};

struct GeneratedBlock {
  std::unique_ptr<Package> package;
  Block* block;
  verilog::ModuleSignatureProto signature;
};

absl::StatusOr<GeneratedBlock> Generate(std::string_view ir,
                                        const CodegenConfig& config,
                                        std::string_view ram_config) {
  XLS_ASSIGN_OR_RETURN(std::unique_ptr<Package> package,
                       Parser::ParsePackage(ir));
  SchedulingOptionsFlagsProto scheduling_options;
  scheduling_options.set_delay_model("unit");
  scheduling_options.set_pipeline_stages(config.pipeline_stages);

  CodegenFlagsProto codegen_options;
  codegen_options.set_top("ram_proc");
  codegen_options.set_generator(GENERATOR_KIND_PIPELINE);
  codegen_options.set_reset("rst");
  codegen_options.set_reset_active_low(false);
  codegen_options.set_reset_asynchronous(false);
  codegen_options.set_reset_data_path(true);
  codegen_options.set_flop_inputs(config.io_kind.has_value());
  codegen_options.set_flop_outputs(config.io_kind.has_value());
  if (config.io_kind.has_value()) {
    codegen_options.set_flop_inputs_kind(*config.io_kind);
    codegen_options.set_flop_outputs_kind(*config.io_kind);
  }
  codegen_options.set_streaming_channel_data_suffix("");
  codegen_options.set_streaming_channel_valid_suffix("_vld");
  codegen_options.set_streaming_channel_ready_suffix("_rdy");
  codegen_options.set_codegen_version(config.version);
  codegen_options.add_ram_configurations(std::string(ram_config));

  XLS_ASSIGN_OR_RETURN(
      (std::pair<SchedulingResult, verilog::CodegenResult> result),
      ScheduleAndCodegen(package.get(), scheduling_options, codegen_options,
                         /*with_delay_model=*/true));

  XLS_ASSIGN_OR_RETURN(Block * block, package->GetBlock("ram_proc"));
  return GeneratedBlock{.package = std::move(package),
                        .block = block,
                        .signature = result.second.signature.proto()};
}

// The ports of a RAM with one read port & one write port. (A 1RW RAM's single
// port plays both roles.)
struct RamPorts {
  std::string rd_addr;
  std::string rd_en;
  std::string rd_data;
  std::string wr_addr;
  std::string wr_en;
  std::string wr_data;
};

absl::StatusOr<RamPorts> GetRamPorts(
    const verilog::ModuleSignatureProto& signature) {
  XLS_RET_CHECK_EQ(signature.rams_size(), 1);
  const verilog::RamProto& ram = signature.rams(0);
  if (ram.has_ram_1rw()) {
    const auto& rw = ram.ram_1rw().rw_port();
    return RamPorts{.rd_addr = rw.request().address().name(),
                    .rd_en = rw.request().read_enable().name(),
                    .rd_data = rw.response().read_data().name(),
                    .wr_addr = rw.request().address().name(),
                    .wr_en = rw.request().write_enable().name(),
                    .wr_data = rw.request().write_data().name()};
  }
  XLS_RET_CHECK(ram.has_ram_1r1w());
  const auto& r = ram.ram_1r1w().r_port();
  const auto& w = ram.ram_1r1w().w_port();
  return RamPorts{.rd_addr = r.request().address().name(),
                  .rd_en = r.request().enable().name(),
                  .rd_data = r.response().data().name(),
                  .wr_addr = w.request().address().name(),
                  .wr_en = w.request().enable().name(),
                  .wr_data = w.request().data().name()};
}

struct SimSpec {
  // Input channel name -> values to feed.
  std::vector<std::pair<std::string, std::vector<uint64_t>>> inputs;
  // Output channel name -> number of values to collect before stopping.
  std::vector<std::pair<std::string, int64_t>> outputs;
  double input_valid_probability = 0.9;
  double output_ready_probability = 0.5;
  uint64_t seed = 0;
  int64_t max_cycles = 20000;
  // Declare a deadlock if no channel transfer happens for this many cycles.
  int64_t stall_limit = 500;
  // Once all inputs are consumed & all outputs are collected, keep simulating
  // for this many cycles (e.g., to let the last activations finish).
  int64_t drain_cycles = 0;
  int64_t ram_size = 16;
};

struct SimResult {
  absl::flat_hash_map<std::string, std::vector<uint64_t>> outputs;
  int64_t cycles = 0;
  bool deadlocked = false;
  std::vector<std::string> assertion_failures;

  std::string ToString() const {
    std::vector<std::string> parts;
    for (const auto& [name, values] : outputs) {
      parts.push_back(
          absl::StrFormat("%s: [%s]", name, absl::StrJoin(values, ", ")));
    }
    return absl::StrFormat(
        "cycles=%d deadlocked=%v assertion_failures=[%s] outputs={%s}", cycles,
        deadlocked, absl::StrJoin(assertion_failures, "; "),
        absl::StrJoin(parts, "; "));
  }
};

absl::StatusOr<SimResult> Simulate(const GeneratedBlock& generated,
                                   const SimSpec& spec) {
  Block* block = generated.block;
  XLS_ASSIGN_OR_RETURN(RamPorts ram, GetRamPorts(generated.signature));
  auto port_width = [&](std::string_view name) -> absl::StatusOr<int64_t> {
    XLS_ASSIGN_OR_RETURN(InputPort * port, block->GetInputPort(name));
    return port->GetType()->GetFlatBitCount();
  };
  auto to_u64 = [](const Value& v) { return v.bits().ToUint64().value(); };

  std::mt19937_64 rng(spec.seed);
  XLS_ASSIGN_OR_RETURN(std::unique_ptr<BlockContinuation> continuation,
                       kInterpreterBlockEvaluator.NewContinuation(block));

  absl::flat_hash_map<std::string, std::deque<uint64_t>> queues;
  absl::flat_hash_map<std::string, bool> held_valid;
  for (const auto& [name, values] : spec.inputs) {
    queues[name] = std::deque<uint64_t>(values.begin(), values.end());
    held_valid[name] = false;
  }
  std::vector<uint64_t> memory(spec.ram_size, 0);
  std::optional<uint64_t> pending_read;

  SimResult result;
  for (const auto& [name, _] : spec.outputs) {
    result.outputs[name] = {};
  }
  int64_t last_progress = 0;
  std::optional<int64_t> done_cycle;
  constexpr int64_t kResetCycles = 2;
  for (int64_t cycle = 0; cycle < spec.max_cycles; ++cycle) {
    const bool resetting = cycle < kResetCycles;
    absl::flat_hash_map<std::string, Value> inputs;
    inputs["rst"] = Value(UBits(resetting ? 1 : 0, 1));
    absl::flat_hash_map<std::string, bool> driven_valid;
    for (const auto& [name, _] : spec.inputs) {
      const std::deque<uint64_t>& queue = queues[name];
      // Once asserted, valid stays asserted until the transfer completes.
      const bool valid = !resetting && !queue.empty() &&
                         (held_valid[name] ||
                          absl::Bernoulli(rng, spec.input_valid_probability));
      driven_valid[name] = valid;
      inputs[absl::StrCat(name, "_vld")] = Value(UBits(valid ? 1 : 0, 1));
      XLS_ASSIGN_OR_RETURN(int64_t width, port_width(name));
      inputs[name] = Value(UBits(queue.empty() ? 0 : queue.front(), width));
    }
    absl::flat_hash_map<std::string, bool> driven_ready;
    for (const auto& [name, _] : spec.outputs) {
      const bool ready = absl::Bernoulli(rng, spec.output_ready_probability);
      driven_ready[name] = ready;
      inputs[absl::StrCat(name, "_rdy")] = Value(UBits(ready ? 1 : 0, 1));
    }
    {
      XLS_ASSIGN_OR_RETURN(int64_t width, port_width(ram.rd_data));
      // Garbage, unless a read was issued in the previous cycle.
      inputs[ram.rd_data] =
          Value(UBits(pending_read.value_or(0xBAD0BAD0), width));
    }

    XLS_RETURN_IF_ERROR(continuation->RunOneCycle(inputs));
    const absl::flat_hash_map<std::string, Value>& outputs =
        continuation->output_ports();
    if (resetting) {
      pending_read = std::nullopt;
      continue;
    }
    for (const std::string& message :
         continuation->events().GetAssertMessages()) {
      result.assertion_failures.push_back(
          absl::StrFormat("cycle %d: %s", cycle, message));
    }

    for (const auto& [name, _] : spec.inputs) {
      const bool transferred =
          driven_valid[name] &&
          to_u64(outputs.at(absl::StrCat(name, "_rdy"))) != 0;
      if (transferred) {
        queues[name].pop_front();
        last_progress = cycle;
      }
      held_valid[name] = driven_valid[name] && !transferred;
    }
    for (const auto& [name, _] : spec.outputs) {
      if (driven_ready[name] &&
          to_u64(outputs.at(absl::StrCat(name, "_vld"))) != 0) {
        result.outputs[name].push_back(to_u64(outputs.at(name)));
        last_progress = cycle;
      }
    }

    // The RAM reads before it writes, and returns read data in the next cycle.
    pending_read = std::nullopt;
    if (to_u64(outputs.at(ram.rd_en)) != 0) {
      pending_read = memory[to_u64(outputs.at(ram.rd_addr)) % spec.ram_size];
    }
    if (to_u64(outputs.at(ram.wr_en)) != 0) {
      memory[to_u64(outputs.at(ram.wr_addr)) % spec.ram_size] =
          to_u64(outputs.at(ram.wr_data));
    }

    result.cycles = cycle + 1;
    bool done = true;
    for (const auto& [_, queue] : queues) {
      if (!queue.empty()) {
        done = false;
      }
    }
    for (const auto& [name, count] : spec.outputs) {
      if (static_cast<int64_t>(result.outputs[name].size()) < count) {
        done = false;
      }
    }
    if (done && !done_cycle.has_value()) {
      done_cycle = cycle;
    }
    if (done_cycle.has_value() && cycle - *done_cycle >= spec.drain_cycles) {
      break;
    }
    if (cycle - last_progress > spec.stall_limit) {
      result.deadlocked = true;
      break;
    }
  }
  return result;
}

constexpr int64_t kSeeds = 10;

// Simulates `generated` with the given spec over several random seeds, and
// checks that each run delivers exactly `expected_outputs` without deadlocking
// or failing any assertions.
void ExpectCorrectOutputs(
    const GeneratedBlock& generated, SimSpec spec,
    const absl::flat_hash_map<std::string, std::vector<uint64_t>>&
        expected_outputs) {
  int64_t failures = 0;
  for (uint64_t seed = 0; seed < kSeeds; ++seed) {
    spec.seed = seed;
    XLS_ASSERT_OK_AND_ASSIGN(SimResult result, Simulate(generated, spec));
    if (result.deadlocked || !result.assertion_failures.empty() ||
        result.outputs != expected_outputs) {
      ++failures;
      if (failures <= 2) {
        ADD_FAILURE() << "seed " << seed << ": " << result.ToString();
      }
    }
  }
  EXPECT_EQ(failures, 0) << "seeds failed out of " << kSeeds;
}

// Returns pseudorandom 16-bit inputs.
std::vector<uint64_t> Inputs(int64_t count, uint64_t salt) {
  std::vector<uint64_t> inputs;
  inputs.reserve(count);
  for (int64_t k = 0; k < count; ++k) {
    inputs.push_back((k * 2654435761ull + salt) & 0xffff);
  }
  return inputs;
}

// For the line-buffer procs: out[k] = x[k] + x[k - 16].
std::vector<uint64_t> LineBufferOutputs(absl::Span<const uint64_t> xs) {
  std::vector<uint64_t> outputs;
  outputs.reserve(xs.size());
  for (int64_t k = 0; k < xs.size(); ++k) {
    outputs.push_back(xs[k] + (k >= 16 ? xs[k - 16] : 0));
  }
  return outputs;
}

class RamBackpressureTest : public ::testing::TestWithParam<CodegenConfig> {};

TEST_P(RamBackpressureTest, ReadWhileConsumerStalled) {
  XLS_ASSERT_OK_AND_ASSIGN(GeneratedBlock generated,
                           Generate(Ram1R1WProcIr(/*with_side_output=*/false),
                                    GetParam(), kRam1R1WConfig));
  std::vector<uint64_t> xs = Inputs(200, /*salt=*/12345);
  ExpectCorrectOutputs(
      generated,
      SimSpec{.inputs = {{"in", xs}}, .outputs = {{"out", xs.size()}}},
      {{"out", LineBufferOutputs(xs)}});
}

TEST_P(RamBackpressureTest, ReadWithSameStageSideOutput) {
  XLS_ASSERT_OK_AND_ASSIGN(GeneratedBlock generated,
                           Generate(Ram1R1WProcIr(/*with_side_output=*/true),
                                    GetParam(), kRam1R1WConfig));
  std::vector<uint64_t> xs = Inputs(200, /*salt=*/12345);
  ExpectCorrectOutputs(
      generated,
      SimSpec{.inputs = {{"in", xs}},
              .outputs = {{"out", xs.size()}, {"side", xs.size()}}},
      {{"out", LineBufferOutputs(xs)}, {"side", xs}});
}

TEST_P(RamBackpressureTest, ReadWithoutBackpressure) {
  XLS_ASSERT_OK_AND_ASSIGN(GeneratedBlock generated,
                           Generate(Ram1R1WProcIr(/*with_side_output=*/false),
                                    GetParam(), kRam1R1WConfig));
  std::vector<uint64_t> xs = Inputs(200, /*salt=*/12345);
  ExpectCorrectOutputs(generated,
                       SimSpec{.inputs = {{"in", xs}},
                               .outputs = {{"out", xs.size()}},
                               .output_ready_probability = 1.0},
                       {{"out", LineBufferOutputs(xs)}});
}

TEST_P(RamBackpressureTest, SinglePortRamWhileConsumerStalled) {
  XLS_ASSERT_OK_AND_ASSIGN(GeneratedBlock generated,
                           Generate(kRam1RWProcIr, GetParam(), kRam1RWConfig));
  std::vector<uint64_t> xs = Inputs(200, /*salt=*/4242);
  ExpectCorrectOutputs(
      generated,
      SimSpec{.inputs = {{"in", xs}}, .outputs = {{"out", xs.size()}}},
      {{"out", LineBufferOutputs(xs)}});
}

TEST_P(RamBackpressureTest, PredicatedReadWhileConsumerStalled) {
  XLS_ASSERT_OK_AND_ASSIGN(
      GeneratedBlock generated,
      Generate(kRamPredicatedProcIr, GetParam(), kRam1R1WConfig));
  constexpr int64_t kGroups = 50;
  std::vector<uint64_t> xs = Inputs(4 * kGroups, /*salt=*/777);
  std::vector<uint64_t> ys = Inputs(4 * kGroups, /*salt=*/31);
  // In phase 0, group j reads the value written by group j - 16 (in its phase
  // 2) & adds the `in2` value it received; in phase 1, it sends the result.
  std::vector<uint64_t> expected;
  for (int64_t j = 0; j < kGroups; ++j) {
    const uint64_t old = j >= 16 ? xs[4 * (j - 16) + 2] : 0;
    expected.push_back(old + ys[4 * j]);
  }
  ExpectCorrectOutputs(generated,
                       SimSpec{.inputs = {{"in", xs}, {"in2", ys}},
                               .outputs = {{"out", kGroups}},
                               .input_valid_probability = 0.5},
                       {{"out", expected}});
}

TEST_P(RamBackpressureTest, RamRequestIsOnlyStreamingSend) {
  XLS_ASSERT_OK_AND_ASSIGN(
      GeneratedBlock generated,
      Generate(kRamOnlySendProcIr, GetParam(), kRam1RWConfig));
  // The proc has no outputs; it checks its own reads with an assertion.
  ExpectCorrectOutputs(
      generated,
      SimSpec{.inputs = {{"in", std::vector<uint64_t>(200, 0)}},
              .input_valid_probability = 0.5,
              .drain_cycles = 10},
      /*expected_outputs=*/{});
}

std::vector<CodegenConfig> AllConfigs() {
  std::vector<CodegenConfig> configs;
  for (CodegenVersionProto version :
       {CODEGEN_VERSION_ONE_DOT_ZERO, CODEGEN_VERSION_ONE_DOT_FIVE}) {
    configs.push_back({.version = version});
    configs.push_back({.version = version, .pipeline_stages = 5});
    for (IOKindProto io_kind :
         {IO_KIND_FLOP, IO_KIND_SKID_BUFFER, IO_KIND_ZERO_LATENCY_BUFFER}) {
      configs.push_back({.version = version, .io_kind = io_kind});
    }
  }
  return configs;
}

INSTANTIATE_TEST_SUITE_P(
    RamBackpressureTestInstantiation, RamBackpressureTest,
    ::testing::ValuesIn(AllConfigs()),
    [](const ::testing::TestParamInfo<CodegenConfig>& info) {
      return info.param.ToString();
    });

}  // namespace
}  // namespace xls
