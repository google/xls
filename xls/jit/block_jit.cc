// Copyright 2023 The XLS Authors
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

#include "xls/jit/block_jit.h"

#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
#include <random>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/interpreter/block_evaluator.h"
#include "xls/ir/block.h"
#include "xls/ir/events.h"
#include "xls/ir/nodes.h"
#include "xls/ir/register.h"
#include "xls/ir/type.h"
#include "xls/ir/value.h"
#include "xls/ir/value_utils.h"
#include "xls/jit/function_base_jit.h"
#include "xls/jit/jit_runtime.h"
#include "xls/jit/orc_jit.h"

namespace xls {

absl::StatusOr<std::unique_ptr<BlockJit>> BlockJit::Create(
    Block* block, JitRuntime* runtime) {
  XLS_ASSIGN_OR_RETURN(std::unique_ptr<OrcJit> orc_jit, OrcJit::Create());
  XLS_ASSIGN_OR_RETURN(auto function,
                       JittedFunctionBase::Build(block, *orc_jit));
  if (!block->GetInstantiations().empty()) {
    return absl::UnimplementedError(
        "Jitting of blocks with instantiations is not yet supported.");
  }
  return std::unique_ptr<BlockJit>(
      new BlockJit(block, runtime, std::move(orc_jit), std::move(function)));
}

std::unique_ptr<BlockJitContinuation> BlockJit::NewContinuation() {
  return std::unique_ptr<BlockJitContinuation>(
      new BlockJitContinuation(block_, this, runtime_, function_));
}

absl::Status BlockJit::RunOneCycle(BlockJitContinuation& continuation) {
  function_.RunJittedFunction(
      continuation.input_buffers_.current(),
      continuation.output_buffers_.current(), continuation.temp_buffer_,
      &continuation.GetEvents(), /*instance_context=*/nullptr, runtime_,
      /*continuation_point=*/0);
  continuation.SwapRegisters();
  return absl::OkStatus();
}

absl::StatusOr<JitArgumentSet> BlockJitContinuation::CombineBuffers(
    const JittedFunctionBase& jit_func, const JitArgumentSet& left,
    int64_t left_count, const JitArgumentSet& rest, int64_t rest_start,
    bool is_inputs) {
  XLS_RET_CHECK_EQ(left.source(), &jit_func);
  XLS_RET_CHECK_EQ(rest.source(), &jit_func);
  const auto& final_sizes = is_inputs ? jit_func.input_buffer_sizes()
                                      : jit_func.output_buffer_sizes();
  const auto& final_aligns =
      is_inputs ? jit_func.input_buffer_preferred_alignments()
                : jit_func.output_buffer_preferred_alignments();
  const absl::Span<int64_t const> left_sizes =
      left.is_inputs() ? left.source()->input_buffer_sizes()
                       : left.source()->output_buffer_sizes();
  const absl::Span<int64_t const> left_aligns =
      left.is_inputs() ? left.source()->input_buffer_preferred_alignments()
                       : left.source()->output_buffer_preferred_alignments();
  const absl::Span<int64_t const> rest_sizes =
      rest.is_inputs() ? rest.source()->input_buffer_sizes()
                       : rest.source()->output_buffer_sizes();
  const absl::Span<int64_t const> rest_aligns =
      rest.is_inputs() ? rest.source()->input_buffer_preferred_alignments()
                       : rest.source()->output_buffer_preferred_alignments();
  std::vector<uint8_t*> final_ptrs;
  XLS_RET_CHECK_LE(left_count, final_sizes.size());
  XLS_RET_CHECK_LE(left_count, left.pointers().size());
  final_ptrs.reserve(final_sizes.size());
  for (int64_t i = 0; i < left_count; ++i) {
    XLS_RET_CHECK_EQ(final_sizes[i], left_sizes[i]) << i;
    XLS_RET_CHECK_EQ(final_aligns[i], left_aligns[i]) << i;
    final_ptrs.push_back(left.pointers()[i]);
  }
  for (int64_t i = left_count; i < final_sizes.size(); ++i) {
    XLS_RET_CHECK_EQ(final_sizes[i], rest_sizes[rest_start])
        << i << " rest: " << rest_start;
    XLS_RET_CHECK_EQ(final_aligns[i], rest_aligns[rest_start])
        << i << " rest: " << rest_start;
    final_ptrs.push_back(rest.pointers()[rest_start++]);
  }
  return JitArgumentSet(&jit_func, /*data=*/nullptr, std::move(final_ptrs),
                        /*is_inputs=*/is_inputs, /*is_outputs=*/!is_inputs);
}

BlockJitContinuation::IOSpace BlockJitContinuation::MakeCombinedBuffers(
    const JittedFunctionBase& jit_func, const Block* block,
    const JitArgumentSet& ports, const BlockJitContinuation::BufferPair& regs,
    bool input) {
  int64_t num_ports =
      input ? block->GetInputPorts().size() : block->GetOutputPorts().size();
  // Registers use the input port offsets.
  int64_t num_input_ports = block->GetInputPorts().size();
  return IOSpace(CombineBuffers(jit_func, ports, num_ports, regs[0],
                                num_input_ports, input)
                     .value(),
                 CombineBuffers(jit_func, ports, num_ports, regs[1],
                                num_input_ports, input)
                     .value());
}

BlockJitContinuation::BlockJitContinuation(Block* block, BlockJit* jit,
                                           JitRuntime* runtime,
                                           const JittedFunctionBase& jit_func)
    : block_(block),
      block_jit_(jit),
      runtime_(runtime),
      register_buffers_memory_{jit_func.CreateInputBuffer(),
                               jit_func.CreateInputBuffer()},
      input_port_buffers_memory_(jit_func.CreateInputBuffer()),
      output_port_buffers_memory_(jit_func.CreateOutputBuffer()),
      input_buffers_(MakeCombinedBuffers(jit_func, block_,
                                         input_port_buffers_memory_,
                                         register_buffers_memory_,
                                         /*input=*/true)),
      output_buffers_(MakeCombinedBuffers(jit_func, block_,
                                          output_port_buffers_memory_,
                                          register_buffers_memory_,
                                          /*input=*/false)),
      temp_buffer_(jit_func.CreateTempBuffer()) {
  // since input and output share the same register pointers they need to use
  // different sides at all times.
  input_buffers_.SetActive(IOSpace::RegisterSpace::kLeft);
  output_buffers_.SetActive(IOSpace::RegisterSpace::kRight);
}

absl::Status BlockJitContinuation::SetInputPorts(
    absl::Span<const Value> values) {
  XLS_RET_CHECK_EQ(block_->GetInputPorts().size(), values.size());
  std::vector<Type*> types;
  types.reserve(values.size());
  auto it = values.cbegin();
  for (auto ip : block_->GetInputPorts()) {
    types.push_back(ip->GetType());
    XLS_RET_CHECK(ValueConformsToType(*it, ip->GetType()))
        << "input port " << ip->name() << " cannot be set to value of " << *it
        << " due to type mismatch with input port type of "
        << ip->GetType()->ToString();
    ++it;
  }
  return runtime_->PackArgs(values, types, input_port_pointers());
}

absl::Status BlockJitContinuation::SetInputPorts(
    absl::Span<const uint8_t* const> inputs) {
  XLS_RET_CHECK_EQ(block_->GetInputPorts().size(), inputs.size());
  // TODO(allight): This is a lot of copying. We could do this more efficiently
  for (int i = 0; i < inputs.size(); ++i) {
    memcpy(input_port_pointers()[i], inputs[i],
           block_jit_->input_port_sizes()[i]);
  }
  return absl::OkStatus();
}

absl::Status BlockJitContinuation::SetInputPorts(
    const absl::flat_hash_map<std::string, Value>& inputs) {
  std::vector<Value> values(block_->GetInputPorts().size());
  auto input_indices = GetInputPortIndices();
  for (const auto& [name, value] : inputs) {
    if (!input_indices.contains(name)) {
      return absl::InvalidArgumentError(
          absl::StrFormat("Block has no input port '%s'", name));
    }
    values[input_indices.at(name)] = value;
  }
  if (block_->GetInputPorts().size() != inputs.size()) {
    std::ostringstream oss;
    for (auto p : block_->GetInputPorts()) {
      if (!inputs.contains(p->name())) {
        oss << "\n\tMissing input for port '" << p->name() << "'";
      }
    }
    return absl::InvalidArgumentError(
        absl::StrFormat("Expected %d input port values but only got %d:%s",
                        values.size(), inputs.size(), oss.str()));
  }
  return SetInputPorts(values);
}

absl::Status BlockJitContinuation::SetRegisters(
    absl::Span<const Value> values) {
  XLS_RET_CHECK_EQ(block_->GetRegisters().size(), values.size());
  std::vector<Type*> types;
  types.reserve(values.size());
  auto it = values.cbegin();
  for (auto reg : block_->GetRegisters()) {
    types.push_back(reg->type());
    XLS_RET_CHECK(ValueConformsToType(*it, reg->type()))
        << "register " << reg->name() << " cannot be set to value of " << *it
        << " due to type mismatch with register type of "
        << reg->type()->ToString();
    ++it;
  }
  return runtime_->PackArgs(values, types, register_pointers());
}

absl::Status BlockJitContinuation::SetRegisters(
    absl::Span<const uint8_t* const> regs) {
  XLS_RET_CHECK_EQ(block_->GetRegisters().size(), regs.size());
  // TODO(allight): This is a lot of copying. We could do this more efficiently
  for (int i = 0; i < regs.size(); ++i) {
    memcpy(register_pointers()[i], regs[i], block_jit_->register_sizes()[i]);
  }
  return absl::OkStatus();
}

absl::Status BlockJitContinuation::SetRegisters(
    const absl::flat_hash_map<std::string, Value>& regs) {
  auto reg_indices = GetRegisterIndices();
  std::vector<Value> values(reg_indices.size());
  for (const auto& [name, value] : regs) {
    if (!reg_indices.contains(name)) {
      return absl::InvalidArgumentError(
          absl::StrFormat("Block has no register '%s'", name));
    }
    values[reg_indices.at(name)] = value;
  }

  if (block_->GetRegisters().size() != regs.size()) {
    std::ostringstream oss;
    for (auto p : block_->GetRegisters()) {
      if (!regs.contains(p->name())) {
        oss << "\n\tMissing value for port '" << p->name() << "'";
      }
    }
    return absl::InvalidArgumentError(
        absl::StrFormat("Expected %d register values but only got %d:%s",
                        reg_indices.size(), regs.size(), oss.str()));
  }
  return SetRegisters(values);
}

std::vector<Value> BlockJitContinuation::GetOutputPorts() const {
  std::vector<Value> result;
  result.reserve(output_port_pointers().size());
  int i = 0;
  for (auto ptr : output_port_pointers()) {
    result.push_back(runtime_->UnpackBuffer(
        ptr, block_->GetOutputPorts()[i++]->operand(0)->GetType()));
  }
  return result;
}

absl::flat_hash_map<std::string, int64_t>
BlockJitContinuation::GetInputPortIndices() const {
  absl::flat_hash_map<std::string, int64_t> ret;
  int i = 0;
  for (auto v : block_->GetInputPorts()) {
    ret[v->name()] = i++;
  }
  return ret;
}

absl::flat_hash_map<std::string, int64_t>
BlockJitContinuation::GetOutputPortIndices() const {
  absl::flat_hash_map<std::string, int64_t> ret;
  int i = 0;
  for (auto v : block_->GetOutputPorts()) {
    ret[v->name()] = i++;
  }
  return ret;
}

absl::flat_hash_map<std::string, int64_t>
BlockJitContinuation::GetRegisterIndices() const {
  absl::flat_hash_map<std::string, int64_t> ret;
  int i = 0;
  for (auto v : block_->GetRegisters()) {
    ret[v->name()] = i++;
  }
  return ret;
}

absl::flat_hash_map<std::string, Value>
BlockJitContinuation::GetOutputPortsMap() const {
  absl::flat_hash_map<std::string, Value> result;
  result.reserve(output_port_pointers().size());
  auto regs = GetOutputPorts();
  for (const auto& [name, off] : GetOutputPortIndices()) {
    result[name] = regs[off];
  }
  return result;
}

std::vector<Value> BlockJitContinuation::GetRegisters() const {
  std::vector<Value> result;
  result.reserve(register_pointers().size());
  int i = 0;
  for (auto ptr : register_pointers()) {
    result.push_back(
        runtime_->UnpackBuffer(ptr, block_->GetRegisters()[i++]->type()));
  }
  return result;
}

absl::flat_hash_map<std::string, Value> BlockJitContinuation::GetRegistersMap()
    const {
  absl::flat_hash_map<std::string, Value> result;
  result.reserve(register_pointers().size());
  auto regs = GetRegisters();
  for (const auto& [name, off] : GetRegisterIndices()) {
    result[name] = regs[off];
  }
  return result;
}

absl::StatusOr<BlockRunResult> JitBlockEvaluator::EvaluateBlock(
    const absl::flat_hash_map<std::string, Value>& inputs,
    const absl::flat_hash_map<std::string, Value>& reg_state,
    Block* block) const {
  XLS_ASSIGN_OR_RETURN(auto runtime, JitRuntime::Create());
  XLS_ASSIGN_OR_RETURN(auto jit, BlockJit::Create(block, runtime.get()));
  auto continuation = jit->NewContinuation();
  XLS_RETURN_IF_ERROR(continuation->SetInputPorts(inputs));
  XLS_RETURN_IF_ERROR(continuation->SetRegisters(reg_state));
  XLS_RETURN_IF_ERROR(jit->RunOneCycle(*continuation));
  return BlockRunResult{
      .outputs = continuation->GetOutputPortsMap(),
      .reg_state = continuation->GetRegistersMap(),
      .interpreter_events = continuation->GetEvents(),
  };
}

absl::StatusOr<std::vector<absl::flat_hash_map<std::string, Value>>>
StreamingJitBlockEvaluator::EvaluateSequentialBlock(
    Block* block,
    absl::Span<const absl::flat_hash_map<std::string, Value>> inputs) const {
  // Initial register state is zero for all registers.
  absl::flat_hash_map<std::string, Value> reg_state;
  for (Register* reg : block->GetRegisters()) {
    reg_state[reg->name()] = ZeroOfType(reg->type());
  }
  XLS_ASSIGN_OR_RETURN(auto runtime, JitRuntime::Create());
  XLS_ASSIGN_OR_RETURN(auto jit, BlockJit::Create(block, runtime.get()));
  auto continuation = jit->NewContinuation();
  XLS_RETURN_IF_ERROR(continuation->SetRegisters(reg_state));

  std::vector<absl::flat_hash_map<std::string, Value>> outputs;
  for (const absl::flat_hash_map<std::string, Value>& input_set : inputs) {
    XLS_RETURN_IF_ERROR(continuation->SetInputPorts(input_set));
    XLS_RETURN_IF_ERROR(jit->RunOneCycle(*continuation));
    outputs.push_back(continuation->GetOutputPortsMap());
  }
  return std::move(outputs);
}

absl::StatusOr<BlockIOResults>
StreamingJitBlockEvaluator::EvaluateChannelizedSequentialBlock(
    Block* block, absl::Span<ChannelSource> channel_sources,
    absl::Span<ChannelSink> channel_sinks,
    absl::Span<const absl::flat_hash_map<std::string, Value>> inputs,
    const std::optional<verilog::ResetProto>& reset, int64_t seed) const {
  std::minstd_rand random_engine;
  random_engine.seed(seed);

  // Initial register state is zero for all registers.
  absl::flat_hash_map<std::string, Value> reg_state;
  for (Register* reg : block->GetRegisters()) {
    reg_state[reg->name()] = ZeroOfType(reg->type());
  }

  XLS_ASSIGN_OR_RETURN(auto runtime, JitRuntime::Create());
  XLS_ASSIGN_OR_RETURN(auto jit, BlockJit::Create(block, runtime.get()));
  auto continuation = jit->NewContinuation();
  XLS_RETURN_IF_ERROR(continuation->SetRegisters(reg_state));

  int64_t max_cycle_count = inputs.size();

  BlockIOResults block_io_results;
  for (int64_t cycle = 0; cycle < max_cycle_count; ++cycle) {
    absl::flat_hash_map<std::string, Value> input_set = inputs.at(cycle);

    // Sources set data/valid
    for (ChannelSource& src : channel_sources) {
      XLS_RETURN_IF_ERROR(
          src.SetBlockInputs(cycle, input_set, random_engine, reset));
    }

    // Sinks set ready
    for (ChannelSink& sink : channel_sinks) {
      XLS_RETURN_IF_ERROR(
          sink.SetBlockInputs(cycle, input_set, random_engine, reset));
    }

    if (VLOG_IS_ON(3)) {
      VLOG(3) << absl::StrFormat("Inputs Cycle %d", cycle);
      for (const auto& [name, val] : input_set) {
        VLOG(3) << absl::StrFormat("%s: %s", name, val.ToString());
      }
    }

    XLS_RETURN_IF_ERROR(continuation->SetInputPorts(input_set));
    XLS_RETURN_IF_ERROR(jit->RunOneCycle(*continuation));
    auto outputs = continuation->GetOutputPortsMap();

    // Sources get ready
    for (ChannelSource& src : channel_sources) {
      XLS_RETURN_IF_ERROR(src.GetBlockOutputs(cycle, outputs));
    }

    // Sinks get data/valid
    for (ChannelSink& sink : channel_sinks) {
      XLS_RETURN_IF_ERROR(sink.GetBlockOutputs(cycle, outputs));
    }

    if (VLOG_IS_ON(3)) {
      VLOG(3) << absl::StrFormat("Outputs Cycle %d", cycle);
      for (const auto& [name, val] : outputs) {
        VLOG(3) << absl::StrFormat("%s: %s", name, val.ToString());
      }
    }

    block_io_results.inputs.push_back(std::move(input_set));
    block_io_results.outputs.push_back(std::move(outputs));
  }

  return block_io_results;
}

namespace {
// Helper adapter to implement the interpreter-focused block-continuation api
// used by eval_proc_main. This holds live all the values needed to run the
// block-jit.
class BlockContinuationJitWrapper final : public BlockContinuation {
 public:
  BlockContinuationJitWrapper(std::unique_ptr<BlockJitContinuation>&& cont,
                              std::unique_ptr<BlockJit>&& jit,
                              std::unique_ptr<JitRuntime>&& runtime)
      : continuation_(std::move(cont)),
        jit_(std::move(jit)),
        runtime_(std::move(runtime)) {}
  const absl::flat_hash_map<std::string, Value>& output_ports() final {
    if (!temporary_outputs_) {
      temporary_outputs_.emplace(continuation_->GetOutputPortsMap());
    }
    return *temporary_outputs_;
  }
  const absl::flat_hash_map<std::string, Value>& registers() final {
    if (!temporary_regs_) {
      temporary_regs_.emplace(continuation_->GetRegistersMap());
    }
    return *temporary_regs_;
  }
  const InterpreterEvents& events() final { return continuation_->GetEvents(); }
  absl::Status RunOneCycle(
      const absl::flat_hash_map<std::string, Value>& inputs) final {
    temporary_outputs_.reset();
    temporary_regs_.reset();
    continuation_->ClearEvents();
    XLS_RETURN_IF_ERROR(continuation_->SetInputPorts(inputs));
    return jit_->RunOneCycle(*continuation_);
  }
  absl::Status SetRegisters(
      const absl::flat_hash_map<std::string, Value>& regs) final {
    return continuation_->SetRegisters(regs);
  }

 private:
  std::unique_ptr<BlockJitContinuation> continuation_;
  std::unique_ptr<BlockJit> jit_;
  std::unique_ptr<JitRuntime> runtime_;
  // Holder for the data we return out of output_ports so that we can reduce
  // copying.
  std::optional<absl::flat_hash_map<std::string, Value>> temporary_outputs_;
  // Holder for the data we return out of registers so that we can reduce
  // copying.
  std::optional<absl::flat_hash_map<std::string, Value>> temporary_regs_;
};
}  // namespace

absl::StatusOr<std::unique_ptr<BlockContinuation>>
StreamingJitBlockEvaluator::NewContinuation(
    Block* block,
    const absl::flat_hash_map<std::string, Value>& initial_registers) const {
  XLS_ASSIGN_OR_RETURN(auto runtime, JitRuntime::Create());
  XLS_ASSIGN_OR_RETURN(auto jit, BlockJit::Create(block, runtime.get()));
  auto jit_cont = jit->NewContinuation();
  XLS_RETURN_IF_ERROR(jit_cont->SetRegisters(initial_registers));
  return std::make_unique<BlockContinuationJitWrapper>(
      std::move(jit_cont), std::move(jit), std::move(runtime));
}

}  // namespace xls
