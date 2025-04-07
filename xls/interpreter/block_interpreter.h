// Copyright 2021 The XLS Authors
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

#ifndef XLS_INTERPRETER_BLOCK_INTERPRETER_H_
#define XLS_INTERPRETER_BLOCK_INTERPRETER_H_

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xls/codegen/module_signature.pb.h"
#include "xls/interpreter/block_evaluator.h"
#include "xls/ir/block.h"
#include "xls/ir/block_elaboration.h"
#include "xls/ir/value.h"

namespace xls {

class InterpreterBlockEvaluator final : public BlockEvaluator {
 public:
  constexpr InterpreterBlockEvaluator() : BlockEvaluator("Interpreter") {}

 protected:
  absl::StatusOr<std::unique_ptr<BlockContinuation>> MakeNewContinuation(
      BlockElaboration&& elaboration,
      const absl::flat_hash_map<std::string, Value>& initial_registers,
      OutputPortSampleTime sample_time) const override;
};

// Runs the interpreter on a combinational block. `inputs` must contain a
// value for each input port in the block. The returned map contains a value
// for each output port of the block.
inline absl::StatusOr<absl::flat_hash_map<std::string, Value>>
InterpretCombinationalBlock(
    Block* block, const absl::flat_hash_map<std::string, Value>& inputs,
    BlockEvaluator::OutputPortSampleTime sample_time =
        BlockEvaluator::OutputPortSampleTime::kAtLastPosEdgeClock) {
  return InterpreterBlockEvaluator().EvaluateCombinationalBlock(block, inputs,
                                                                sample_time);
}

// Overload which accepts and returns uint64_t values instead of xls::Values.
inline absl::StatusOr<absl::flat_hash_map<std::string, uint64_t>>
InterpretCombinationalBlock(
    Block* block, const absl::flat_hash_map<std::string, uint64_t>& inputs,
    BlockEvaluator::OutputPortSampleTime sample_time =
        BlockEvaluator::OutputPortSampleTime::kAtLastPosEdgeClock) {
  return InterpreterBlockEvaluator().EvaluateCombinationalBlock(block, inputs,
                                                                sample_time);
}

// Runs the interpreter on a block feeding a sequence of values to input ports
// and returning the resulting sequence of values from the output
// ports. Registers are clocked between each set of inputs fed to the block.
// Initial register state is zero for all registers.
inline absl::StatusOr<std::vector<absl::flat_hash_map<std::string, Value>>>
InterpretSequentialBlock(
    Block* block,
    absl::Span<const absl::flat_hash_map<std::string, Value>> inputs,
    BlockEvaluator::OutputPortSampleTime sample_time =
        BlockEvaluator::OutputPortSampleTime::kAtLastPosEdgeClock) {
  return InterpreterBlockEvaluator().EvaluateSequentialBlock(block, inputs,
                                                             sample_time);
}

// Overload which accepts and returns uint64_t values instead of xls::Values.
inline absl::StatusOr<std::vector<absl::flat_hash_map<std::string, uint64_t>>>
InterpretSequentialBlock(
    Block* block,
    absl::Span<const absl::flat_hash_map<std::string, uint64_t>> inputs,
    BlockEvaluator::OutputPortSampleTime sample_time =
        BlockEvaluator::OutputPortSampleTime::kAtLastPosEdgeClock) {
  return InterpreterBlockEvaluator().EvaluateSequentialBlock(block, inputs,
                                                             sample_time);
}

// Runs the interpreter on a block.  Each input port in the block
// should be given a sequence of data values to drive the block.
//
// This sequence of data values can be provided either by
//  1. Providing a ChannelSource in channel_sources that will drive the
//     data or valid port.
//  2. Providing a ChannelSink in channel_sinks that will drive the ready port.
//  3. Providing the sequence of Values as part of inputs, for any
//     port (like rst) that is not part of a Channel definition.
//
// Registers are clocked between each set of inputs fed to the block.
// Initial register state is zero for all registers.
inline absl::StatusOr<BlockIOResults> InterpretChannelizedSequentialBlock(
    Block* block, absl::Span<ChannelSource> channel_sources,
    absl::Span<ChannelSink> channel_sinks,
    absl::Span<const absl::flat_hash_map<std::string, Value>> inputs,
    const std::optional<verilog::ResetProto>& reset = std::nullopt,
    int64_t seed = 0) {
  return InterpreterBlockEvaluator().EvaluateChannelizedSequentialBlock(
      block, channel_sources, channel_sinks, inputs, reset, seed);
}

// Variant which accepts and returns uint64_t values instead of xls::Values.
inline absl::StatusOr<BlockIOResultsAsUint64>
InterpretChannelizedSequentialBlockWithUint64(
    Block* block, absl::Span<ChannelSource> channel_sources,
    absl::Span<ChannelSink> channel_sinks,
    absl::Span<const absl::flat_hash_map<std::string, uint64_t>> inputs,
    const std::optional<verilog::ResetProto>& reset = std::nullopt,
    int64_t seed = 0) {
  return InterpreterBlockEvaluator()
      .EvaluateChannelizedSequentialBlockWithUint64(
          block, channel_sources, channel_sinks, inputs, reset, seed);
}

// A single evaluator which uses the interpreter.
inline constexpr InterpreterBlockEvaluator kInterpreterBlockEvaluator;

}  // namespace xls

#endif  // XLS_INTERPRETER_BLOCK_INTERPRETER_H_
