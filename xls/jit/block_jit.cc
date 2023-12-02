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

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "xls/common/logging/logging.h"
#include "xls/common/logging/vlog_is_on.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/interpreter/block_evaluator.h"
#include "xls/ir/block.h"
#include "xls/ir/events.h"
#include "xls/ir/nodes.h"
#include "xls/ir/register.h"
#include "xls/ir/type.h"
#include "xls/ir/value.h"
#include "xls/ir/value_helpers.h"
#include "xls/jit/function_base_jit.h"
#include "xls/jit/jit_runtime.h"
#include "xls/jit/orc_jit.h"

namespace xls {

absl::StatusOr<std::unique_ptr<BlockJit>> BlockJit::Create(
    Block* block, JitRuntime* runtime) {
  XLS_ASSIGN_OR_RETURN(std::unique_ptr<OrcJit> orc_jit, OrcJit::Create());
  XLS_ASSIGN_OR_RETURN(auto function, BuildBlockFunction(block, *orc_jit));
  if (!block->GetInstantiations().empty()) {
    return absl::UnimplementedError(
        "Jitting of blocks with instantiations is not yet supported.");
  }
  return std::unique_ptr<BlockJit>(
      new BlockJit(block, runtime, std::move(orc_jit), std::move(function)));
}

std::unique_ptr<BlockJitContinuation> BlockJit::NewContinuation() {
  return std::unique_ptr<BlockJitContinuation>(new BlockJitContinuation(
      block_, this, runtime_, function_.temp_buffer_size,
      /*register_sizes=*/
      absl::MakeSpan(function_.input_buffer_sizes)
          .subspan(block_->GetInputPorts().size()),
      /*register_alignments=*/
      absl::MakeSpan(function_.input_buffer_prefered_alignments)
          .subspan(block_->GetInputPorts().size()),
      /*output_port_sizes=*/
      absl::MakeSpan(function_.output_buffer_sizes)
          .subspan(0, block_->GetOutputPorts().size()),
      /*output_port_alignments=*/
      absl::MakeSpan(function_.output_buffer_prefered_alignments)
          .subspan(0, block_->GetOutputPorts().size()),
      /*input_port_sizes=*/
      absl::MakeSpan(function_.input_buffer_sizes)
          .subspan(0, block_->GetInputPorts().size()),
      /*input_port_alignments=*/
      absl::MakeSpan(function_.input_buffer_prefered_alignments)
          .subspan(0, block_->GetInputPorts().size())));
}

absl::Status BlockJit::RunOneCycle(BlockJitContinuation& continuation) {
  function_.RunJittedFunction(
      continuation.function_inputs().data(),
      continuation.function_outputs().data(),
      runtime_->AsStack(continuation.temp_buffer()).data(),
      &continuation.GetEvents(), /*user_data=*/nullptr, runtime_,
      /*continuation_point=*/0);
  continuation.SwapRegisters();
  return absl::OkStatus();
}

namespace {
// Determine how much memory is needed to hold all the arguments of the given
// sizes. This is larger than just the sum to allow for space for all of them to
// be properly aligned.
int64_t FindFullBufferSize(JitRuntime* runtime, absl::Span<int64_t const> sizes,
                           absl::Span<int64_t const> alignments) {
  XLS_CHECK_EQ(sizes.size(), alignments.size());
  int64_t total = 0;
  for (int64_t i = 0; i < sizes.size(); ++i) {
    total += runtime->ShouldAllocateForAlignment(sizes[i], alignments[i]);
  }
  return total;
}

std::vector<uint8_t*> CombineLists(absl::Span<uint8_t* const> a,
                                   absl::Span<uint8_t* const> b) {
  std::vector<uint8_t*> res;
  res.reserve(a.size() + b.size());
  for (auto i : a) {
    res.push_back(i);
  }
  for (auto i : b) {
    res.push_back(i);
  }
  return res;
}

// Find the start-pointer of each argument in the argument arena. The size of
// each argument is given by the sizes span.
std::vector<uint8_t*> CalculatePointers(JitRuntime* runtime,
                                        absl::Span<uint8_t> base_buffer,
                                        absl::Span<const int64_t> sizes,
                                        absl::Span<const int64_t> alignments) {
  XLS_CHECK_EQ(sizes.size(), alignments.size());
  std::vector<uint8_t*> out;
  out.reserve(sizes.size());
  for (int64_t i = 0; i < sizes.size(); ++i) {
    auto aligned_span = runtime->AsAligned(base_buffer, alignments[i]);
    XLS_CHECK_GE(aligned_span.size(), sizes[i])
        << "Buffer for " << i << " element not large enough!";
    out.push_back(aligned_span.data());
    base_buffer = aligned_span.subspan(sizes[i]);
  }
  return out;
}
}  // namespace

BlockJitContinuation::BlockJitContinuation(
    Block* block, BlockJit* jit, JitRuntime* runtime, size_t temp_size,
    absl::Span<const int64_t> register_sizes,
    absl::Span<const int64_t> register_alignments,
    absl::Span<const int64_t> output_port_sizes,
    absl::Span<const int64_t> output_port_alignments,
    absl::Span<const int64_t> input_port_sizes,
    absl::Span<const int64_t> input_port_alignments)
    : block_(block),
      block_jit_(jit),
      runtime_(runtime),
      register_arena_left_(
          FindFullBufferSize(runtime, register_sizes, register_alignments), 0),
      register_arena_right_(register_arena_left_.size(), 0xff),
      output_port_arena_(FindFullBufferSize(runtime, output_port_sizes,
                                            output_port_alignments),
                         0xff),
      input_port_arena_(
          FindFullBufferSize(runtime, input_port_sizes, input_port_alignments),
          0xff),
      register_pointers_(BlockJitContinuation::IOSpace(
          CalculatePointers(runtime, absl::MakeSpan(register_arena_left_),
                            register_sizes, register_alignments),
          CalculatePointers(runtime, absl::MakeSpan(register_arena_right_),
                            register_sizes, register_alignments),
          BlockJitContinuation::IOSpace::RegisterSpace::kLeft)),
      output_port_pointers_(
          CalculatePointers(runtime, absl::MakeSpan(output_port_arena_),
                            output_port_sizes, output_port_alignments)),
      input_port_pointers_(
          CalculatePointers(runtime, absl::MakeSpan(input_port_arena_),
                            input_port_sizes, input_port_alignments)),
      full_input_pointer_set_(BlockJitContinuation::IOSpace(
          CombineLists(input_port_pointers_, register_pointers_.left()),
          CombineLists(input_port_pointers_, register_pointers_.right()),
          BlockJitContinuation::IOSpace::RegisterSpace::kLeft)),
      full_output_pointer_set_(BlockJitContinuation::IOSpace(
          CombineLists(output_port_pointers_, register_pointers_.left()),
          CombineLists(output_port_pointers_, register_pointers_.right()),
          BlockJitContinuation::IOSpace::RegisterSpace::kRight)),
      temp_data_arena_(temp_size) {}

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
  return runtime_->PackArgs(values, types, input_port_pointers_);
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

    if (XLS_VLOG_IS_ON(3)) {
      XLS_VLOG(3) << absl::StrFormat("Inputs Cycle %d", cycle);
      for (auto [name, val] : input_set) {
        XLS_VLOG(3) << absl::StrFormat("%s: %s", name, val.ToString());
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

    if (XLS_VLOG_IS_ON(3)) {
      XLS_VLOG(3) << absl::StrFormat("Outputs Cycle %d", cycle);
      for (auto [name, val] : outputs) {
        XLS_VLOG(3) << absl::StrFormat("%s: %s", name, val.ToString());
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
