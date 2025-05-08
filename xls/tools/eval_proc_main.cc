// Copyright 2020 The XLS Authors
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

// Tool to evaluate the behavior of a Proc network.

#include <algorithm>
#include <cstdint>
#include <deque>
#include <iostream>
#include <iterator>
#include <memory>
#include <numeric>
#include <optional>
#include <random>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/btree_map.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/flags/flag.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/random/distributions.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/escaping.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "xls/codegen/module_signature.pb.h"
#include "xls/common/exit_status.h"
#include "xls/common/file/filesystem.h"
#include "xls/common/init_xls.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/dev_tools/tool_timeout.h"
#include "xls/interpreter/block_evaluator.h"
#include "xls/interpreter/block_interpreter.h"
#include "xls/interpreter/channel_queue.h"
#include "xls/interpreter/evaluator_options.h"
#include "xls/interpreter/interpreter_proc_runtime.h"
#include "xls/interpreter/serial_proc_runtime.h"
#include "xls/ir/bits.h"
#include "xls/ir/block.h"
#include "xls/ir/block_elaboration.h"
#include "xls/ir/channel.h"
#include "xls/ir/events.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/ir_parser.h"
#include "xls/ir/nodes.h"
#include "xls/ir/package.h"
#include "xls/ir/proc.h"
#include "xls/ir/ram_rewrite.pb.h"
#include "xls/ir/register.h"
#include "xls/ir/value.h"
#include "xls/ir/value_utils.h"
#include "xls/jit/block_jit.h"
#include "xls/jit/jit_proc_runtime.h"
#include "xls/jit/jit_runtime.h"
#include "xls/tools/eval_utils.h"
#include "xls/tools/memory_models.h"
#include "xls/tools/node_coverage_utils.h"

static constexpr std::string_view kUsage = R"(
Evaluates an IR file containing Procs, or a Block generated from them.
The Proc network will be ticked a fixed number of times
(specified on the command line) and the final state
value of each proc will be printed to the terminal upon completion.

Initial states are set according to their declarations inside the IR itself.
)";

ABSL_FLAG(
    std::optional<std::string>, top, std::nullopt,
    "If present the top construct to simulate. Must be an exact match to "
    "the name of an appropriate proc/block. Until new-style-procs are "
    "available this is mostly just to support module-name for block "
    "simulations as the specified top must be the actual top of the design.");
ABSL_FLAG(std::vector<std::string>, ticks, {},
          "Can be a comma-separated list of runs. "
          "Number of clock ticks to execute for each, with proc state "
          "resetting per run. <0 runs until all outputs are verified.");
ABSL_FLAG(std::string, backend, "serial_jit",
          "Backend to use for evaluation. Valid options are:\n"
          " * serial_jit: JIT-backed single-stepping runtime.\n"
          " * ir_interpreter: Interpreter at the IR level.\n"
          " * block_interpreter: Interpret a block generated from a proc.\n"
          " * block_jit: JIT-backed block execution generated from a proc.");
ABSL_FLAG(std::string, block_signature_proto, "",
          "Path to textproto file containing signature from codegen");
ABSL_FLAG(int64_t, max_cycles_no_output, 100,
          "For block simulation, stop after this many cycles without output.");

ABSL_FLAG(
    std::vector<std::string>, inputs_for_channels, {},
    "Comma separated list of channel=filename pairs, for example: ch_a=foo.ir. "
    "Files contain one XLS Value in human-readable form per line. Either "
    "'inputs_for_channels' or 'testvector_textproto' can be defined.");
ABSL_FLAG(
    std::vector<std::string>, expected_outputs_for_channels, {},
    "Comma separated list of channel=filename pairs, for example: ch_a=foo.ir. "
    "Files contain one XLS Value in human-readable form per line. Either "
    "'expected_outputs_for_channels' or 'expected_outputs_for_all_channels' "
    "can be defined.\n"
    "For procs, when 'expected_outputs_for_channels' or "
    "'expected_outputs_for_all_channels' are not specified the values of all "
    "the channel are displayed on stdout.");

ABSL_FLAG(std::string, testvector_textproto, "",
          "A textproto file containing proc channel test vectors.");

// TODO(google/xls#1645) Remove in favor of --testvector_textproto
ABSL_FLAG(
    std::string, inputs_for_all_channels, "",
    "Path to file containing inputs for all channels.\n"
    "The file format is:\n"
    "CHANNEL_NAME : {\n"
    "  VALUE\n"
    "}\n"
    "where CHANNEL_NAME is the name of the channel and VALUE is one XLS Value "
    "in human-readable form. There is one VALUE per line. There may be zero or "
    "more occurrences of VALUE for a channel. The file may contain one or more "
    "channels. One of 'inputs_for_channels', 'inputs_for_all_channels', or "
    "'proto_inputs_for_all_channels' can be defined.");
ABSL_FLAG(
    std::string, expected_outputs_for_all_channels, "",
    "Path to file containing outputs for all channels.\n"
    "The file format is:\n"
    "CHANNEL_NAME : {\n"
    "  VALUE\n"
    "}\n"
    "where CHANNEL_NAME is the name of the channel and VALUE is one XLS Value "
    "in human-readable form. There is one VALUE per line. There may be zero or "
    "more occurrences of VALUE for a channel. The file may contain one or more "
    "channels. Either 'expected_outputs_for_channels' or "
    "'expected_outputs_for_all_channels' can be defined.\n"
    "For procs, when 'expected_outputs_for_channels', "
    "'expected_outputs_for_all_channels' or "
    "'expected_proto_outputs_for_all_channels' are not specified the values of "
    "all the channel are displayed on stdout.");

// TODO(google/xls#1645) Also probably remove in favor of --testvector_textproto
ABSL_FLAG(
    std::string, proto_inputs_for_all_channels, "",
    "Path to ProcChannelValuesProto binary proto containing inputs for all "
    "channels.");
ABSL_FLAG(
    std::string, expected_proto_outputs_for_all_channels, "",
    "Path to file containing ProcChannelValuesProto binary proto of outputs "
    "for all channels.");

ABSL_FLAG(int64_t, random_seed, 42, "Random seed");
ABSL_FLAG(double, prob_input_valid_assert, 1.0,
          "Single-cycle probability of asserting valid with more input ready.");
ABSL_FLAG(bool, show_trace, false, "Whether or not to print trace messages.");
ABSL_FLAG(bool, trace_channels, false,
          "If true, values sent and received on channels are recorded as trace "
          "messages");
ABSL_FLAG(int64_t, max_trace_verbosity, 0,
          "Maximum verbosity for traces. Traces with higher verbosity are "
          "stripped from codegen output. 0 by default.");
ABSL_FLAG(int64_t, trace_per_ticks, 100, "Print a trace every N ticks.");
ABSL_FLAG(std::string, output_stats_path, "", "File to output statistics to.");
ABSL_FLAG(bool, fail_on_assert, false,
          "When set to true, the simulation fails on the activation or cycle "
          "in which an assertion fires.");
ABSL_FLAG(std::optional<std::string>, output_node_coverage_stats_proto,
          std::nullopt,
          "File to write a (binary) NodeCoverageStatsProto showing which bits "
          "in the run were actually set for each node.");
ABSL_FLAG(std::optional<std::string>, output_node_coverage_stats_textproto,
          std::nullopt,
          "File to write a (text) NodeCoverageStatsProto showing which bits "
          "in the run were actually set for each node.");
ABSL_FLAG(bool, abstract_ram_model, false,
          "Whether or not to use an abstract RAM model, as opposed to a "
          "rewritten RAM model, for proc memory.\n");
ABSL_FLAG(std::string, ram_rewrites_textproto, "",
          "Path to ram rewrites textproto, which is used to create memory "
          "models. Blank is default, in which case no memory models are added "
          "to the simulation.");

namespace xls {

namespace {

static absl::Status LogInterpreterEvents(
    std::string_view entity_name, const InterpreterEvents& events,
    std::optional<int> cycle = std::nullopt) {
  const std::string cycle_str =
      cycle.has_value() ? absl::StrFormat("Cycle[%d]: ", cycle.value()) : "";

  if (absl::GetFlag(FLAGS_show_trace)) {
    for (const auto& msg : events.trace_msgs) {
      if (msg.verbosity <= absl::GetFlag(FLAGS_max_trace_verbosity)) {
        std::string unescaped_msg;
        XLS_RET_CHECK(absl::CUnescape(msg.message, &unescaped_msg));
        LOG(INFO) << cycle_str << "Proc " << entity_name
                  << " trace: " << unescaped_msg << "\n";
      }
    }
  }
  for (const auto& msg : events.assert_msgs) {
    std::string unescaped_msg;
    XLS_RET_CHECK(absl::CUnescape(msg, &unescaped_msg));
    LOG(INFO) << cycle_str << "Proc " << entity_name
              << " assert: " << unescaped_msg << "\n";
  }
  return absl::OkStatus();
}

struct EvaluateProcsOptions {
  bool use_jit = false;
  bool fail_on_assert = false;
  std::vector<int64_t> ticks = {-1};
  std::optional<std::string> top = std::nullopt;
};

static absl::Status EvaluateProcs(
    Package* package,
    const absl::btree_map<std::string, std::vector<Value>>& inputs_for_channels,
    absl::btree_map<std::string, std::vector<Value>>&
        expected_outputs_for_channels,
    const RamRewritesProto& ram_rewrites,
    const EvaluateProcsOptions& options = {}) {
  std::unique_ptr<SerialProcRuntime> runtime;
  std::optional<JitRuntime*> jit;
  EvaluatorOptions evaluator_options;
  evaluator_options.set_trace_channels(absl::GetFlag(FLAGS_trace_channels));
  bool uses_observers =
      absl::GetFlag(FLAGS_output_node_coverage_stats_proto).has_value() ||
      absl::GetFlag(FLAGS_output_node_coverage_stats_textproto).has_value();
  if (options.top) {
    XLS_ASSIGN_OR_RETURN(Proc * proc, package->GetProc(*options.top));
    if (proc != package->GetTop()) {
      return absl::UnimplementedError(
          "Simulating subsets of the proc network is not implemented yet.");
    }
  }
  evaluator_options.set_support_observers(uses_observers);
  if (options.use_jit) {
    XLS_ASSIGN_OR_RETURN(
        runtime, CreateJitSerialProcRuntime(package, evaluator_options));
    XLS_ASSIGN_OR_RETURN(auto jit_queue, runtime->GetJitChannelQueueManager());
    jit = &jit_queue->runtime();
  } else {
    XLS_ASSIGN_OR_RETURN(runtime, CreateInterpreterSerialProcRuntime(
                                      package, evaluator_options));
  }
  ScopedRecordNodeCoverage cov(
      absl::GetFlag(FLAGS_output_node_coverage_stats_proto),
      absl::GetFlag(FLAGS_output_node_coverage_stats_textproto), jit);
  if (cov.observer()) {
    XLS_RETURN_IF_ERROR(runtime->SetObserver(*cov.observer()));
    LOG(ERROR) << "Set observer!";
  }

  ChannelQueueManager& queue_manager = runtime->queue_manager();

  std::vector<std::unique_ptr<memory_model::ProcMemoryModel>> memory_models;

  const bool abstract_ram_model = absl::GetFlag(FLAGS_abstract_ram_model);

  for (const RamRewriteProto& ram_rewrite : ram_rewrites.rewrites()) {
    XLS_RET_CHECK(ram_rewrite.has_to_config());

    XLS_RET_CHECK_EQ(ram_rewrite.from_config().depth(),
                     ram_rewrite.to_config().depth());

    std::unique_ptr<memory_model::ProcMemoryModel> memory_model;

    if (abstract_ram_model) {
      XLS_ASSIGN_OR_RETURN(memory_model,
                           memory_model::CreateAbstractProcMemoryModel(
                               ram_rewrite, queue_manager));
    } else if (ram_rewrite.to_config().kind() == RamKindProto::RAM_1RW) {
      XLS_ASSIGN_OR_RETURN(memory_model,
                           memory_model::CreateRewrittenProcMemoryModel(
                               ram_rewrite, queue_manager));
    } else {
      return absl::UnimplementedError(absl::StrFormat(
          "Don't know what memory model to use with RamKind %s from rewrites "
          "proto",
          xls::RamKindProto_Name(ram_rewrite.to_config().kind())));
    }

    memory_models.push_back(std::move(memory_model));
  }

  for (const auto& [channel_name, values] : inputs_for_channels) {
    XLS_ASSIGN_OR_RETURN(ChannelQueue * in_queue,
                         queue_manager.GetQueueByName(channel_name));
    for (const Value& value : values) {
      XLS_RETURN_IF_ERROR(in_queue->Write(value));
    }
    if (absl::GetFlag(FLAGS_show_trace)) {
      LOG(INFO) << "Channel " << channel_name << " has " << values.size()
                << " inputs";
    }
  }
  if (absl::GetFlag(FLAGS_show_trace)) {
    for (const auto& [channel_name, values] : expected_outputs_for_channels) {
      LOG(INFO) << "Channel " << channel_name << " has " << values.size()
                << " outputs";
    }
  }

  absl::Time start_time = absl::Now();

  const int64_t trace_per_ticks = absl::GetFlag(FLAGS_trace_per_ticks);

  for (int64_t this_ticks : options.ticks) {
    if (absl::GetFlag(FLAGS_show_trace)) {
      LOG(INFO) << "Resetting proc state";
    }
    runtime->ResetState();

    for (int i = 0; this_ticks < 0 || i < this_ticks; i++) {
      if (absl::GetFlag(FLAGS_show_trace) &&
          (i < trace_per_ticks || i % trace_per_ticks == 0)) {
        std::ostringstream ostr;
        for (const auto& [channel_name, values] :
             expected_outputs_for_channels) {
          XLS_ASSIGN_OR_RETURN(ChannelQueue * out_queue,
                               queue_manager.GetQueueByName(channel_name));
          ostr << channel_name << "[" << out_queue->GetSize() << "] " << " ";
        }
        for (const auto& [channel_name, values] : inputs_for_channels) {
          XLS_ASSIGN_OR_RETURN(ChannelQueue * in_queue,
                               queue_manager.GetQueueByName(channel_name));
          ostr << channel_name << "[" << in_queue->GetSize() << "] " << " ";
        }
        LOG(INFO) << "Tick " << i << ": " << ostr.str();
      }
      // Don't double print events (traces, assertions, etc)
      runtime->ClearInterpreterEvents();
      absl::Status tick_ret = runtime->Tick();

      if (!tick_ret.ok()) {
        for (const auto& [channel_name, values] :
             expected_outputs_for_channels) {
          XLS_ASSIGN_OR_RETURN(ChannelQueue * out_queue,
                               queue_manager.GetQueueByName(channel_name));

          LOG(INFO) << absl::StreamFormat(
              "out_queue[%s]: size %li, reference values %li", channel_name,
              out_queue->GetSize(), values.size());
        }
        for (const auto& [channel_name, values] : inputs_for_channels) {
          XLS_ASSIGN_OR_RETURN(ChannelQueue * in_queue,
                               queue_manager.GetQueueByName(channel_name));

          LOG(INFO) << absl::StreamFormat(
              "in_queue[%s]: size %li, reference values %li", channel_name,
              in_queue->GetSize(), values.size());
        }
        return tick_ret;
      }

      for (std::unique_ptr<memory_model::ProcMemoryModel>& memory :
           memory_models) {
        XLS_RETURN_IF_ERROR(memory->Tick());
      }

      // Sort the keys for stable print order.
      absl::flat_hash_map<Proc*, std::vector<Value>> states;
      std::vector<Proc*> sorted_procs;
      for (const auto& proc : package->procs()) {
        sorted_procs.push_back(proc.get());
        states[proc.get()] = runtime->ResolveState(proc.get());
      }

      std::sort(sorted_procs.begin(), sorted_procs.end(),
                [](Proc* a, Proc* b) { return a->name() < b->name(); });

      std::vector<std::string> asserts;

      XLS_RETURN_IF_ERROR(
          LogInterpreterEvents("[global]", runtime->GetGlobalEvents()));
      for (Proc* proc : sorted_procs) {
        const xls::InterpreterEvents& events =
            runtime->GetInterpreterEvents(proc);
        XLS_RETURN_IF_ERROR(LogInterpreterEvents(proc->name(), events));
        if (options.fail_on_assert) {
          for (const std::string& assert : events.assert_msgs) {
            asserts.push_back(
                absl::StrFormat("Proc %s: %s", proc->name(), assert));
          }
        }
      }

      for (const auto& proc : package->procs()) {
        const auto& state = states.at(proc.get());
        VLOG(1) << "Proc " << proc->name() << " : "
                << absl::StrFormat("{%s}",
                                   absl::StrJoin(state, ", ", ValueFormatter));
      }

      if (!asserts.empty()) {
        return absl::UnknownError(absl::StrFormat(
            "Assert(s) fired:\n\n%s", absl::StrJoin(asserts, "\n")));
      }

      // --ticks 0 stops when all outputs are verified
      if (this_ticks < 0) {
        bool all_outputs_produced = true;
        for (const auto& [channel_name, values] :
             expected_outputs_for_channels) {
          XLS_ASSIGN_OR_RETURN(ChannelQueue * out_queue,
                               queue_manager.GetQueueByName(channel_name));
          if (out_queue->GetSize() < values.size()) {
            all_outputs_produced = false;
          }
        }
        if (all_outputs_produced) {
          absl::btree_map<std::string, std::vector<Value>> unconsumed_inputs;
          for (const auto& [channel_name, _] : inputs_for_channels) {
            XLS_ASSIGN_OR_RETURN(ChannelQueue * in_queue,
                                 queue_manager.GetQueueByName(channel_name));
            // Ignore single value channels in this check
            if (in_queue->channel()->kind() == ChannelKind::kSingleValue) {
              continue;
            }
            while (!in_queue->IsEmpty()) {
              unconsumed_inputs[channel_name].push_back(*in_queue->Read());
            }
          }
          if (!unconsumed_inputs.empty()) {
            LOG(WARNING)
                << "Warning: Not all inputs were consumed by the time all "
                   "expected outputs were produced. Remaining inputs:\n"
                << ChannelValuesToString(unconsumed_inputs);
          }
          break;
        }
      }
    }
  }
  absl::Duration elapsed_time = absl::Now() - start_time;
  LOG(INFO) << "Elapsed time: " << elapsed_time;
  bool checked_any_output = false;
  std::vector<std::string> errors;
  for (const auto& [channel_name, values] : expected_outputs_for_channels) {
    XLS_ASSIGN_OR_RETURN(ChannelQueue * out_queue,
                         queue_manager.GetQueueByName(channel_name));
    uint64_t processed_count = 0;
    for (const Value& value : values) {
      std::optional<Value> out_val = out_queue->Read();
      if (!out_val.has_value()) {
        errors.push_back(absl::StrFormat(
            "Channel %s didn't consume %d expected values (processed %d)",
            channel_name, values.size() - processed_count, processed_count));
        break;
      }
      if (value != *out_val) {
        errors.push_back(absl::StrFormat(
            "Mismatched (channel=%s) after %d outputs (%s != %s)", channel_name,
            processed_count, value.ToString(), out_val->ToString()));
        break;
      }
      if (absl::GetFlag(FLAGS_show_trace)) {
        LOG(INFO) << absl::StreamFormat("Matched (channel=%s) after %d outputs",
                                        channel_name, processed_count);
      }
      checked_any_output = true;
      ++processed_count;
    }
  }
  if (!errors.empty()) {
    return absl::UnknownError(
        absl::StrFormat("Outputs did not match expectations:\n\n%s",
                        absl::StrJoin(errors, "\n")));
  }
  if (!checked_any_output && !expected_outputs_for_channels.empty()) {
    return absl::UnknownError("No output verified (empty expected values?)");
  }

  if (expected_outputs_for_channels.empty()) {
    for (const Channel* channel : package->channels()) {
      if (!channel->CanSend()) {
        continue;
      }
      XLS_ASSIGN_OR_RETURN(ChannelQueue * out_queue,
                           queue_manager.GetQueueByName(channel->name()));
      std::vector<Value> channel_values(out_queue->GetSize());
      int64_t index = 0;
      while (!out_queue->IsEmpty()) {
        std::optional<Value> out_val = out_queue->Read();
        channel_values[index++] = out_val.value();
      }
      expected_outputs_for_channels.insert(
          {std::string{channel->name()}, channel_values});
    }
    std::cout << ChannelValuesToString(expected_outputs_for_channels);
  }
  return absl::OkStatus();
}

struct ChannelInfo {
  int64_t width = -1;
  bool port_input = false;
  // Is this ready-valid?
  bool ready_valid = false;

  // Precalculated channel names
  std::string channel_ready;
  std::string channel_valid;
  std::string channel_data;
};

absl::StatusOr<absl::flat_hash_map<std::string, ChannelInfo>>
InterpretBlockSignature(
    const verilog::ModuleSignatureProto& signature,
    const absl::btree_map<std::string, std::vector<Value>>& inputs_for_channels,
    const absl::btree_map<std::string, std::vector<Value>>&
        expected_outputs_for_channels,
    const RamRewritesProto& ram_rewrites) {
  absl::flat_hash_map<std::string, ChannelInfo> channel_info;
  // Pull the information out of the channel_protos
  for (const verilog::ChannelInterfaceProto& channel :
       signature.channel_interfaces()) {
    const auto& data_port_it = absl::c_find_if(
        signature.data_ports(), [&](const verilog::PortProto& port) {
          return port.name() == channel.data_port_name();
        });
    if (data_port_it == signature.data_ports().cend()) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "Channel '%s' names its data port as '%s' but no such port exists.",
          channel.channel_name(), channel.data_port_name()));
    }
    ChannelInfo info{
        .width = data_port_it->width(),
        .ready_valid =
            channel.flow_control() == verilog::CHANNEL_FLOW_CONTROL_READY_VALID,
        .channel_data = channel.data_port_name(),
    };
    if (channel.direction() == verilog::CHANNEL_DIRECTION_SEND) {
      // Output channel
      info.port_input = false;
    } else if (channel.direction() == verilog::CHANNEL_DIRECTION_RECEIVE) {
      // Input channel
      info.port_input = true;
    } else {
      XLS_RET_CHECK_FAIL() << "Internal/send&recv channel '"
                           << channel.DebugString()
                           << "' ended up in block signature.";
    }
    if (info.ready_valid) {
      if (!channel.has_ready_port_name()) {
        return absl::InvalidArgumentError(
            absl::StrFormat("Ready/valid channel '%s' has no ready port.",
                            channel.channel_name()));
      }
      if (!channel.has_valid_port_name()) {
        return absl::InvalidArgumentError(
            absl::StrFormat("Ready/valid channel '%s' has no valid port.",
                            channel.channel_name()));
      }
      info.channel_valid = channel.valid_port_name();
      info.channel_ready = channel.ready_port_name();
    }
    channel_info[channel.channel_name()] = std::move(info);
  }

  // If channels aren't around we are interpreting a 'fn' so need to get the
  // inputs directly from the data ports. Luckily we don't need to worry about
  // R/V signaling for fns.
  if (channel_info.empty()) {
    for (const verilog::PortProto& port : signature.data_ports()) {
      channel_info[port.name()] = ChannelInfo{
          .width = port.width(),
          .port_input = port.direction() == verilog::PORT_DIRECTION_INPUT,
          .ready_valid = false,
          .channel_data = port.name()};
    }
  }

  for (auto& [name, info] : channel_info) {
    if (info.port_input) {
      XLS_RET_CHECK(inputs_for_channels.contains(name))
          << "missing port " << name;
    } else {
      XLS_RET_CHECK(expected_outputs_for_channels.contains(name))
          << "Missing port " << name;
    }
  }

  for (const auto& [name, _] : inputs_for_channels) {
    if (!channel_info.contains(name)) {
      return absl::InvalidArgumentError(
          absl::StrFormat("Channel %s should not be in channel inputs file, as "
                          "there are no corresponding ports",
                          name));
    }
  }
  for (const auto& [name, _] : expected_outputs_for_channels) {
    if (!channel_info.contains(name)) {
      return absl::InvalidArgumentError(
          absl::StrFormat("Channel %s should not be in channel outputs file, "
                          "as there are no corresponding ports",
                          name));
    }
  }

  return channel_info;
}

static xls::Type* GetOutputPortSampleTimeOrNull(Block* block,
                                                std::string_view port_name) {
  for (const InputPort* port : block->GetInputPorts()) {
    if (port->name() == port_name) {
      return port->GetType();
    }
  }
  return nullptr;
}

struct RunBlockOptions {
  bool use_jit = false;
  std::vector<int64_t> ticks = {-1};
  int64_t max_cycles_no_output = 100;
  std::optional<std::string> top;
  int random_seed;
  double prob_input_valid_assert;
  bool show_trace;
  bool fail_on_assert;
};

// Helper to hold various commonly needed port names for a particular ram.
struct StandardRamInfo {
  std::string_view rd_addr;
  std::string_view rd_en;
  std::string_view rd_data;
  std::string_view wr_addr;
  std::string_view wr_en;
  std::string_view wr_data;
};

constexpr BlockEvaluator::OutputPortSampleTime kClocked =
    BlockEvaluator::OutputPortSampleTime::kAtLastPosEdgeClock;

absl::StatusOr<absl::flat_hash_map<std::string, StandardRamInfo>> GetRamInfoMap(
    const verilog::ModuleSignatureProto& sig) {
  absl::flat_hash_map<std::string, StandardRamInfo> all_infos;
  all_infos.reserve(sig.rams_size());
  for (const verilog::RamProto& ram_info : sig.rams()) {
    StandardRamInfo info;
    switch (ram_info.ram_oneof_case()) {
      case xls::verilog::RamProto::RamOneofCase::kRam1Rw:
        info.rd_addr = ram_info.ram_1rw().rw_port().request().address().name();
        info.rd_en =
            ram_info.ram_1rw().rw_port().request().read_enable().name();
        info.rd_data =
            ram_info.ram_1rw().rw_port().response().read_data().name();
        info.wr_addr = ram_info.ram_1rw().rw_port().request().address().name();
        info.wr_data =
            ram_info.ram_1rw().rw_port().request().write_data().name();
        info.wr_en =
            ram_info.ram_1rw().rw_port().request().write_enable().name();
        break;
      case xls::verilog::RamProto::RamOneofCase::kRam1R1W:
        info.wr_addr = ram_info.ram_1r1w().w_port().request().address().name();
        info.wr_data = ram_info.ram_1r1w().w_port().request().data().name();
        info.wr_en = ram_info.ram_1r1w().w_port().request().enable().name();
        info.rd_addr = ram_info.ram_1r1w().r_port().request().address().name();
        info.rd_data = ram_info.ram_1r1w().r_port().response().data().name();
        info.rd_en = ram_info.ram_1r1w().r_port().request().enable().name();
        break;
      case xls::verilog::RamProto::RamOneofCase::RAM_ONEOF_NOT_SET:
        XLS_RET_CHECK_FAIL() << "Ram request '" << ram_info.name()
                             << "' does not include read/write info";
    }
    all_infos[ram_info.name()] = std::move(info);
  }
  return all_infos;
}

static absl::Status RunBlock(
    Package* package, const verilog::ModuleSignatureProto& signature,
    const absl::btree_map<std::string, std::vector<Value>>& inputs_for_channels,
    absl::btree_map<std::string, std::vector<Value>>&
        expected_outputs_for_channels,
    const RamRewritesProto& ram_rewrites, std::string_view output_stats_path,
    const RunBlockOptions& options = {}) {
  Block* block;
  if (options.top) {
    XLS_ASSIGN_OR_RETURN(block, package->GetBlock(*options.top));
  } else if (package->HasTop()) {
    if (package->GetTop().value()->IsBlock()) {
      XLS_ASSIGN_OR_RETURN(block, package->GetTopAsBlock());
    } else if (package->blocks().size() == 1) {
      block = package->blocks().front().get();
    } else {
      // This is result of codegen-ing a proc so use the block for the top proc
      // as top.
      XLS_ASSIGN_OR_RETURN(Proc * top_proc, package->GetTopAsProc());
      XLS_ASSIGN_OR_RETURN(
          block, package->GetBlock(top_proc->name()),
          _ << "Unable to determine top. Pass --top to select one manually.");
    }
  } else if (package->blocks().size() == 1) {
    block = package->blocks().front().get();
  } else {
    return absl::InvalidArgumentError(
        "Input IR should contain exactly one block or a top");
  }

  std::mt19937_64 bit_gen(options.random_seed);

  // TODO: Support multiple resets
  CHECK_EQ(options.ticks.size(), 1);

  XLS_ASSIGN_OR_RETURN(
      (absl::flat_hash_map<std::string, ChannelInfo> channel_info),
      InterpretBlockSignature(signature, inputs_for_channels,
                              expected_outputs_for_channels, ram_rewrites),
      _ << "signature was: " << signature.DebugString());
  XLS_ASSIGN_OR_RETURN(
      (absl::flat_hash_map<std::string, StandardRamInfo> ram_info),
      GetRamInfoMap(signature));

  // Prepare values in queue format
  absl::flat_hash_map<std::string, std::deque<Value>> channel_value_queues;
  for (const auto& [name, values] : inputs_for_channels) {
    XLS_RET_CHECK(!channel_value_queues.contains(name));
    absl::c_copy(values, std::back_inserter(channel_value_queues[name]));
  }
  for (const auto& [name, values] : expected_outputs_for_channels) {
    XLS_RET_CHECK(!channel_value_queues.contains(name));
    absl::c_copy(values, std::back_inserter(channel_value_queues[name]));
  }

  absl::flat_hash_map<std::string,
                      std::unique_ptr<memory_model::BlockMemoryModel>>
      model_memories;

  for (const RamRewriteProto& rewrite : ram_rewrites.rewrites()) {
    const std::string& name = rewrite.to_name_prefix();
    const std::string_view& wr_data = ram_info.at(name).wr_data;
    XLS_ASSIGN_OR_RETURN(const OutputPort* port, block->GetOutputPort(wr_data));

    Type* element_type = port->operand(0)->GetType();

    model_memories[name] = std::make_unique<memory_model::BlockMemoryModel>(
        name, /*size=*/rewrite.from_config().depth(),
        /*initial_value=*/memory_model::XsOfType(element_type),
        /*read_disabled_value=*/memory_model::XsOfType(element_type),
        options.show_trace);
  }

  absl::flat_hash_map<std::string, Value> reg_state;
  {
    XLS_ASSIGN_OR_RETURN(BlockElaboration elab,
                         BlockElaboration::Elaborate(block));
    for (BlockInstance* inst : elab.instances()) {
      if (!inst->block()) {
        // Actually a fifo or something without real registers.
        continue;
      }
      for (Register* reg : (*inst->block())->GetRegisters()) {
        // Initial register state is one for all registers.
        // Ideally this would be randomized, but at least 1s are more likely to
        //  expose bad behavior than 0s.
        reg_state[absl::StrCat(inst->RegisterPrefix(), reg->name())] =
            memory_model::XsOfType(reg->type());
      }
    }
  }

  bool needs_observer =
      absl::GetFlag(FLAGS_output_node_coverage_stats_proto).has_value() ||
      absl::GetFlag(FLAGS_output_node_coverage_stats_textproto).has_value();
  const BlockEvaluator& continuation_factory =
      options.use_jit
          ? reinterpret_cast<const BlockEvaluator&>(
                needs_observer ? kObservableJitBlockEvaluator
                               : kJitBlockEvaluator)
          : reinterpret_cast<const BlockEvaluator&>(kInterpreterBlockEvaluator);
  XLS_ASSIGN_OR_RETURN(auto continuation, continuation_factory.NewContinuation(
                                              block, reg_state, kClocked));
  std::optional<JitRuntime*> jit;
  if (options.use_jit) {
    XLS_ASSIGN_OR_RETURN(jit,
                         kJitBlockEvaluator.GetRuntime(continuation.get()));
  }
  ScopedRecordNodeCoverage cov(
      absl::GetFlag(FLAGS_output_node_coverage_stats_proto),
      absl::GetFlag(FLAGS_output_node_coverage_stats_textproto), jit);

  if (cov.observer()) {
    XLS_RETURN_IF_ERROR(continuation->SetObserver(*cov.observer()));
  }

  int64_t last_output_cycle = 0;
  int64_t matched_outputs = 0;
  bool checked_any_output = false;
  absl::Time start_time = absl::Now();
  if (signature.reset().name().empty()) {
    LOG(WARNING) << "No reset found in signature!";
  }
  for (int64_t cycle = 0;; ++cycle) {
    // Idealized reset behavior
    const bool resetting = (cycle == 0);
    // We don't want the cycle where we are initially resetting the registers to
    // be counted in coverage since its unlikely to be valuable.
    cov.SetPaused(resetting);

    if (options.show_trace && ((cycle < 30) || (cycle % 100 == 0))) {
      LOG(INFO) << "Cycle[" << cycle << "]: resetting? " << resetting
                << " matched outputs " << matched_outputs;
    }

    absl::flat_hash_set<std::string> asserted_valids;
    absl::flat_hash_map<std::string, Value> input_set;

    if (!signature.reset().name().empty()) {
      input_set[signature.reset().name()] = Value(
          xls::UBits((resetting ^ signature.reset().active_low()) ? 1 : 0, 1));
    }

    for (const auto& [name, _] : inputs_for_channels) {
      const ChannelInfo& info = channel_info.at(name);
      const std::deque<Value>& queue = channel_value_queues.at(name);
      if (info.ready_valid) {
        // Don't bring valid low without a transaction
        const bool asserted_valid = asserted_valids.contains(name);
        const bool random_go_head =
            absl::Bernoulli(bit_gen, options.prob_input_valid_assert);
        const bool this_valid =
            asserted_valid || (random_go_head && !queue.empty());
        if (this_valid) {
          asserted_valids.insert(name);
        }
        input_set[info.channel_valid] =
            Value(xls::UBits(this_valid ? 1 : 0, 1));
        // Channels without data port will return nullptr
        xls::Type* port_type = info.width != 0 ? GetOutputPortSampleTimeOrNull(
                                                     block, info.channel_data)
                                               : nullptr;

        if (port_type != nullptr) {
          input_set[info.channel_data] =
              queue.empty() ? memory_model::XsOfType(port_type) : queue.front();
        }
      } else {
        // Just take the first value for the single value channels
        XLS_RET_CHECK(!queue.empty());
        input_set[name] = queue.front();
      }
    }
    for (const auto& [name, model] : model_memories) {
      XLS_RET_CHECK(ram_info.contains(name));
      std::string_view rd_data = ram_info.at(name).rd_data;
      input_set[rd_data] = model->GetValueReadLastTick();
    }
    for (const auto& [name, _] : expected_outputs_for_channels) {
      const ChannelInfo& info = channel_info.at(name);
      // TODO(allight): Support simulating fns which aren't ready-valid.
      XLS_RET_CHECK(info.ready_valid);
      input_set[info.channel_ready] = Value(xls::UBits(1, 1));
    }
    XLS_RETURN_IF_ERROR(continuation->RunOneCycle(input_set));
    const absl::flat_hash_map<std::string, Value>& outputs =
        continuation->output_ports();

    // Output trace messages
    const xls::InterpreterEvents& events = continuation->events();
    XLS_RETURN_IF_ERROR(LogInterpreterEvents(block->name(), events, cycle));

    if (!events.assert_msgs.empty() && options.fail_on_assert) {
      return absl::UnknownError(absl::StrFormat(
          "Assert(s) fired:\n\n%s", absl::StrJoin(events.assert_msgs, "\n")));
    }

    if (resetting) {
      last_output_cycle = cycle;
      continue;
    }

    // Channel output checks
    for (const auto& [name, _] : inputs_for_channels) {
      const ChannelInfo& info = channel_info.at(name);

      if (!info.ready_valid) {
        continue;
      }

      const bool vld_value = input_set.at(info.channel_valid).bits().Get(0);
      const bool rdy_value = outputs.at(info.channel_ready).bits().Get(0);

      std::deque<Value>& queue = channel_value_queues.at(name);
      if (vld_value && rdy_value) {
        if (options.show_trace) {
          LOG(INFO) << "Cycle[" << cycle
                    << "]: Channel Model: Consuming input for " << name << ": "
                    << queue.front().ToString();
        }
        queue.pop_front();
        asserted_valids.erase(name);
      }
    }

    std::vector<std::string> errors;
    for (const auto& [name, _] : expected_outputs_for_channels) {
      const ChannelInfo& info = channel_info.at(name);

      const bool vld_value = outputs.at(info.channel_valid).bits().Get(0);
      const bool rdy_value = input_set.at(info.channel_ready).bits().Get(0);

      std::deque<Value>& queue = channel_value_queues.at(name);

      if (rdy_value && vld_value) {
        if (queue.empty()) {
          errors.push_back(
              absl::StrFormat("Block wrote past the end of the expected values "
                              "list for channel %s: %s",
                              name, outputs.at(info.channel_data).ToString()));
          continue;
        }
        checked_any_output = true;
        if (info.width != 0) {
          const Value& data_value = outputs.at(info.channel_data);
          const Value& match_value = queue.front();
          if (options.show_trace) {
            LOG(INFO) << "Cycle[" << cycle
                      << "]: Channel Model: Consuming output for " << name
                      << ": " << data_value << ", remaining " << queue.size();
          }
          if (match_value != data_value) {
            errors.push_back(absl::StrFormat(
                "Cycle[%d]: Output mismatched for channel %s: expected %s, "
                "block "
                "outputted "
                "%s",
                cycle, name, match_value.ToString(), data_value.ToString()));
            continue;
          }
        } else if (queue.front().GetFlatBitCount() != 0) {
          // TODO(allight): Actually check the types match up too.
          errors.push_back(
              absl::StrFormat("Cycle[%d]: Output mismatched for channel %s: "
                              "expected %s, block outputted "
                              "zero-len data",
                              cycle, name, queue.front().ToString()));
          continue;
        }
        ++matched_outputs;
        queue.pop_front();
        last_output_cycle = cycle;
      }
    }
    if (!errors.empty()) {
      return absl::UnknownError(absl::StrFormat(
          "Cycle[%d]: Outputs did not match expectations:\n\n%s", cycle,
          absl::StrJoin(errors, "\n")));
    }

    // Memory model outputs
    for (const auto& [name, model] : model_memories) {
      XLS_RET_CHECK(ram_info.contains(name));
      const StandardRamInfo info = ram_info.at(name);
      // Write handling
      {
        const Value wr_en_val = outputs.at(info.wr_en);
        XLS_RET_CHECK(wr_en_val.IsBits());
        if (wr_en_val.IsAllOnes()) {
          const Value wr_addr_val = outputs.at(info.wr_addr);
          const Value wr_data_val = outputs.at(info.wr_data);
          XLS_RET_CHECK(wr_addr_val.IsBits());
          XLS_RET_CHECK(wr_data_val.IsBits());
          XLS_ASSIGN_OR_RETURN(uint64_t addr, wr_addr_val.bits().ToUint64());
          XLS_RETURN_IF_ERROR(model->Write(addr, wr_data_val));
        }
      }
      // Read handling
      {
        const Value rd_en_val = outputs.at(info.rd_en);
        XLS_RET_CHECK(rd_en_val.IsBits());
        if (rd_en_val.IsAllOnes()) {
          const Value rd_addr_val = outputs.at(info.rd_addr);
          XLS_RET_CHECK(rd_addr_val.IsBits());
          XLS_ASSIGN_OR_RETURN(uint64_t addr, rd_addr_val.bits().ToUint64());
          XLS_RETURN_IF_ERROR(model->Read(addr));
        }
      }
    }

    bool all_output_queues_empty = true;
    for (const auto& [name, _] : expected_outputs_for_channels) {
      // Ignore single value channels in this check
      const ChannelInfo& info = channel_info.at(name);
      if (!info.ready_valid) {
        continue;
      }

      const std::deque<Value>& queue = channel_value_queues.at(name);
      if (!queue.empty()) {
        all_output_queues_empty = false;
      }
    }
    if (all_output_queues_empty) {
      if (absl::GetFlag(FLAGS_show_trace)) {
        LOG(INFO) << "Finished at cycle " << cycle;
      }
      break;
    }

    // Break on no output for too long
    if ((cycle - last_output_cycle) > options.max_cycles_no_output) {
      return absl::OutOfRangeError(
          absl::StrFormat("Block didn't produce output for %i cycles",
                          options.max_cycles_no_output));
    }

    for (const auto& [_, model] : model_memories) {
      XLS_RETURN_IF_ERROR(model->Tick());
    }
  }

  absl::Duration elapsed_time = absl::Now() - start_time;
  LOG(INFO) << "Elapsed time: " << elapsed_time;

  absl::btree_map<std::string, std::vector<Value>> unconsumed_inputs;
  for (const auto& [channel_name, _] : inputs_for_channels) {
    // Ignore single value channels in this check
    const ChannelInfo& info = channel_info.at(channel_name);
    if (!info.ready_valid) {
      continue;
    }

    std::deque<Value>& queue = channel_value_queues.at(channel_name);
    if (!queue.empty()) {
      absl::c_copy(queue, std::back_inserter(unconsumed_inputs[channel_name]));
    }
  }
  if (!unconsumed_inputs.empty()) {
    LOG(WARNING) << "Warning: Not all inputs were consumed by the time all "
                    "expected outputs were produced. Remaining inputs:\n"
                 << ChannelValuesToString(unconsumed_inputs);
  }
  if (!checked_any_output) {
    return absl::UnknownError("No output verified (empty expected values?)");
  }
  if (!output_stats_path.empty()) {
    XLS_RETURN_IF_ERROR(xls::SetFileContents(
        output_stats_path, absl::StrFormat("%i", last_output_cycle)));
  }

  return absl::OkStatus();
}

static absl::StatusOr<absl::flat_hash_map<std::string, std::string>>
ParseChannelFilenames(absl::Span<const std::string> files_raw) {
  absl::flat_hash_map<std::string, std::string> ret;
  for (const std::string& file : files_raw) {
    std::vector<std::string> split = absl::StrSplit(file, '=');
    if (split.size() != 2) {
      return absl::InvalidArgumentError(
          "Format of argument should be channel=file");
    }

    ret[split[0]] = split[1];
  }
  return ret;
}

static absl::StatusOr<absl::btree_map<std::string, std::vector<Value>>>
GetValuesForEachChannels(
    absl::Span<const std::string> filenames_for_each_channel,
    const int64_t total_ticks) {
  absl::flat_hash_map<std::string, std::string> channel_filenames;
  XLS_ASSIGN_OR_RETURN(channel_filenames,
                       ParseChannelFilenames(filenames_for_each_channel));
  absl::btree_map<std::string, std::vector<Value>> values_for_channels;

  for (const auto& [channel_name, filename] : channel_filenames) {
    XLS_ASSIGN_OR_RETURN(std::vector<Value> values,
                         ParseValuesFile(filename, total_ticks));
    values_for_channels[channel_name] = values;
  }
  return values_for_channels;
}

static absl::Status RealMain(
    std::string_view ir_file, std::string_view backend,
    std::string_view block_signature_proto, std::vector<int64_t> ticks,
    const int64_t max_cycles_no_output,
    const std::vector<std::string>& inputs_for_channels_text,
    const std::vector<std::string>& expected_outputs_for_channels_text,
    const std::string& ram_rewrites_textproto_path,
    const std::string& inputs_for_all_channels_text,
    const std::string& expected_outputs_for_all_channels_text,
    const std::string& proto_inputs_for_all_channels,
    const std::string& testvector_proto,
    const std::string& expected_proto_outputs_for_all_channels,
    const int random_seed, const double prob_input_valid_assert,
    bool show_trace, std::string_view output_stats_path, bool fail_on_assert) {
  auto timeout = StartTimeoutTimer();
  // Don't waste time and memory parsing more input than can possibly be
  // consumed.
  const int64_t total_ticks =
      std::accumulate(ticks.begin(), ticks.end(), static_cast<int64_t>(0));

  absl::btree_map<std::string, std::vector<Value>> inputs_for_channels;
  if (!inputs_for_channels_text.empty()) {
    XLS_ASSIGN_OR_RETURN(
        inputs_for_channels,
        GetValuesForEachChannels(inputs_for_channels_text, total_ticks));
  } else if (!inputs_for_all_channels_text.empty()) {
    XLS_ASSIGN_OR_RETURN(
        inputs_for_channels,
        ParseChannelValuesFromFile(inputs_for_all_channels_text, total_ticks));
  } else if (!proto_inputs_for_all_channels.empty()) {
    XLS_ASSIGN_OR_RETURN(inputs_for_channels,
                         ParseChannelValuesFromProtoFile(
                             proto_inputs_for_all_channels, total_ticks));
  } else if (!testvector_proto.empty()) {
    XLS_ASSIGN_OR_RETURN(
        inputs_for_channels,
        ParseChannelValuesFromTestVectorFile(testvector_proto, total_ticks));
  }

  absl::btree_map<std::string, std::vector<Value>>
      expected_outputs_for_channels;
  if (!expected_outputs_for_channels_text.empty()) {
    XLS_ASSIGN_OR_RETURN(expected_outputs_for_channels,
                         GetValuesForEachChannels(
                             expected_outputs_for_channels_text, total_ticks));
  } else if (!expected_outputs_for_all_channels_text.empty()) {
    XLS_ASSIGN_OR_RETURN(
        expected_outputs_for_channels,
        ParseChannelValuesFromFile(expected_outputs_for_all_channels_text,
                                   total_ticks));
  } else if (!expected_proto_outputs_for_all_channels.empty()) {
    XLS_ASSIGN_OR_RETURN(
        expected_outputs_for_channels,
        ParseChannelValuesFromProtoFile(expected_proto_outputs_for_all_channels,
                                        total_ticks));
  }

  RamRewritesProto ram_rewrites;

  if (!ram_rewrites_textproto_path.empty()) {
    XLS_RETURN_IF_ERROR(
        xls::ParseTextProtoFile(ram_rewrites_textproto_path, &ram_rewrites));
  }

  XLS_ASSIGN_OR_RETURN(std::string ir_text, GetFileContents(ir_file));
  XLS_ASSIGN_OR_RETURN(auto package, Parser::ParsePackage(ir_text));

  if (backend.starts_with("block")) {
    RunBlockOptions block_options = {
        .ticks = ticks,
        .max_cycles_no_output = max_cycles_no_output,
        .top = absl::GetFlag(FLAGS_top),
        .random_seed = random_seed,
        .prob_input_valid_assert = prob_input_valid_assert,
        .show_trace = show_trace,
        .fail_on_assert = fail_on_assert};
    if (backend == "block_jit") {
      block_options.use_jit = true;
    } else if (backend == "block_interpreter") {
      block_options.use_jit = false;
    } else {
      LOG(QFATAL) << "Unknown backend type";
    }
    verilog::ModuleSignatureProto proto;
    CHECK_OK(ParseTextProtoFile(block_signature_proto, &proto));
    return RunBlock(package.get(), proto, inputs_for_channels,
                    expected_outputs_for_channels, ram_rewrites,
                    output_stats_path, block_options);
  }

  // Not block sim
  EvaluateProcsOptions evaluate_procs_options = {
      .fail_on_assert = fail_on_assert,
      .ticks = ticks,
      .top = absl::GetFlag(FLAGS_top),
  };

  if (backend == "serial_jit") {
    evaluate_procs_options.use_jit = true;
  } else if (backend == "ir_interpreter") {
    evaluate_procs_options.use_jit = false;
  } else {
    LOG(QFATAL) << "Unknown backend type";
  }
  return EvaluateProcs(package.get(), inputs_for_channels,
                       expected_outputs_for_channels, ram_rewrites,
                       evaluate_procs_options);
}

}  // namespace
}  // namespace xls

int main(int argc, char* argv[]) {
  std::vector<std::string_view> positional_args =
      xls::InitXls(kUsage, argc, argv);
  if (positional_args.size() != 1) {
    LOG(QFATAL) << "One (and only one) IR file must be given.";
  }

  std::string backend = absl::GetFlag(FLAGS_backend);
  if (backend != "serial_jit" && backend != "ir_interpreter" &&
      backend != "block_interpreter" && backend != "block_jit") {
    LOG(QFATAL) << "Unrecognized backend choice.";
  }

  if ((backend == "block_interpreter" || backend == "block_jit") &&
      absl::GetFlag(FLAGS_block_signature_proto).empty()) {
    LOG(QFATAL) << "Block evaluation requires --block_signature_proto.";
  }

  std::vector<int64_t> ticks;
  for (const std::string& run_str : absl::GetFlag(FLAGS_ticks)) {
    int ticks_int;
    if (!absl::SimpleAtoi(run_str, &ticks_int)) {
      LOG(QFATAL) << "Couldn't parse run description in --ticks: " << run_str;
    }
    ticks.push_back(ticks_int);
  }
  if (ticks.empty()) {
    LOG(QFATAL) << "--ticks must be specified.";
  }

  if (absl::c_count(
          absl::Span<const bool>{
              !absl::GetFlag(FLAGS_inputs_for_channels).empty(),
              !absl::GetFlag(FLAGS_inputs_for_all_channels).empty(),
              !absl::GetFlag(FLAGS_proto_inputs_for_all_channels).empty()},
          true) > 1) {
    LOG(QFATAL) << "Only one of --inputs_for_channels, "
                   "--inputs_for_all_channels, and "
                   "--proto_inputs_for_all_channels must be set.";
  }

  if (absl::c_count(
          absl::Span<const bool>{
              !absl::GetFlag(FLAGS_expected_outputs_for_channels).empty(),
              !absl::GetFlag(FLAGS_expected_outputs_for_all_channels).empty(),
              !absl::GetFlag(FLAGS_expected_proto_outputs_for_all_channels)
                   .empty()},
          true) > 1) {
    LOG(QFATAL) << "Only one of --expected_outputs_for_channels, "
                   "--expected_outputs_for_all_channels, and "
                   "--expected_proto_outputs_for_all_channels must be set.";
  }

  return xls::ExitStatus(xls::RealMain(
      positional_args[0], backend, absl::GetFlag(FLAGS_block_signature_proto),
      ticks, absl::GetFlag(FLAGS_max_cycles_no_output),
      absl::GetFlag(FLAGS_inputs_for_channels),
      absl::GetFlag(FLAGS_expected_outputs_for_channels),
      absl::GetFlag(FLAGS_ram_rewrites_textproto),
      absl::GetFlag(FLAGS_inputs_for_all_channels),
      absl::GetFlag(FLAGS_expected_outputs_for_all_channels),
      absl::GetFlag(FLAGS_proto_inputs_for_all_channels),
      absl::GetFlag(FLAGS_testvector_textproto),
      absl::GetFlag(FLAGS_expected_proto_outputs_for_all_channels),
      absl::GetFlag(FLAGS_random_seed),
      absl::GetFlag(FLAGS_prob_input_valid_assert),
      absl::GetFlag(FLAGS_show_trace), absl::GetFlag(FLAGS_output_stats_path),
      absl::GetFlag(FLAGS_fail_on_assert)));
}
