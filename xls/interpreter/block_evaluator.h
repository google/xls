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

#ifndef XLS_INTERPRETER_BLOCK_EVALUATOR_H_
#define XLS_INTERPRETER_BLOCK_EVALUATOR_H_

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/random/bit_gen_ref.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xls/codegen/module_signature.pb.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/block.h"
#include "xls/ir/block_elaboration.h"
#include "xls/ir/events.h"
#include "xls/ir/value.h"

namespace xls {

struct BlockRunResult {
  absl::flat_hash_map<std::string, Value> outputs;
  absl::flat_hash_map<std::string, Value> reg_state;
  InterpreterEvents interpreter_events;
};

// Drives input channel simulation for testing blocks.
//
// For each successive input, new data is driven after a randomized delay.
class ChannelSource {
 public:
  enum class BehaviorDuringReset : uint8_t {
    kIgnoreReady,  // Ignore ready signal during reset
    kAttendReady,  // Send on ready regardless of reset
  };

  // Construct a ChannelSource given port names for data/ready/valid
  // for the given block.
  //
  // lambda is the probability that the next value in the input sequence
  // (set by SetDataSequence), will be put on an the channel and valid
  // asserted (if the channel is not otherwise occupied by a in-progress
  // transaction).  Once valid is asserted, it will remain asserted until
  // the transaction completes via ready being asserted.
  ChannelSource(
      std::string_view data_name, std::string_view valid_name,
      std::string_view ready_name, double lambda, Block* block,
      BehaviorDuringReset reset_behavior = BehaviorDuringReset::kIgnoreReady)
      : data_name_(data_name),
        valid_name_(valid_name),
        ready_name_(ready_name),
        lambda_(lambda),
        block_(block),
        reset_behavior_(reset_behavior) {}

  // Sets sequence of data that this source will send to its block.
  absl::Status SetDataSequence(std::vector<Value> data);
  absl::Status SetDataSequence(absl::Span<const uint64_t> data);

  // For each cycle, SetBlockInputs() is called to provide the block
  // this channel's inputs for the cycle.
  //
  // inputs is a map of the block's input port names to their driven values
  // for this_cycle. SetBlockInputs will set inputs[data_name] and
  // inputs[valid_name] to the Value they should be driven for this_cycle.
  absl::Status SetBlockInputs(int64_t this_cycle,
                              absl::flat_hash_map<std::string, Value>& inputs,
                              absl::BitGenRef random_engine,
                              std::optional<verilog::ResetProto> reset);

  // For each cycle, GetBlockOutputs() is called to provide this channel
  // the block's outputs for the cycle.
  //
  // outputs is a map of output port names to their Values.
  absl::Status GetBlockOutputs(
      int64_t this_cycle,
      const absl::flat_hash_map<std::string, Value>& outputs);

  // Source has transferred all data to the block.
  bool AllDataSent() const { return !HasMoreData() && !is_valid_; }

 private:
  // This source has more data to be sent.
  bool HasMoreData() const {
    return current_index_ + 1 < data_sequence_.size();
  }

  std::string data_name_;
  std::string valid_name_;
  std::string ready_name_;

  double lambda_ = 1.0;  // For geometric inter-arrival times.

  Block* block_ = nullptr;

  BehaviorDuringReset reset_behavior_;

  // Data sequence to be sent.
  // Only one of data_sequence_ and data_sequence_as_uint64_ is used,
  // depending on which constructor was called.
  std::vector<Value> data_sequence_;

  int64_t current_index_ = -1;  // Cycle next data will be sent on.
  bool is_valid_ = false;       // Valid signal is asserted.
};

// Drives output channel simulation for testing blocks.
//
// Each successive output is received with a fixed probability.
class ChannelSink {
 public:
  enum class BehaviorDuringReset : uint8_t {
    kIgnoreValid,  // Ignore valid signal during reset
    kAttendValid,  // Receive on valid regardless of reset
  };

  // Construct a ChannelSource given port names for data/ready/valid
  // for the given block.
  //
  // lambda is the probability that for a given cycle, the sink will assert
  // ready.
  ChannelSink(
      std::string_view data_name, std::string_view valid_name,
      std::string_view ready_name, double lambda, Block* block,
      BehaviorDuringReset reset_behavior = BehaviorDuringReset::kAttendValid)
      : data_name_(data_name),
        valid_name_(valid_name),
        ready_name_(ready_name),
        lambda_(lambda),
        block_(block),
        reset_behavior_(reset_behavior) {}

  // For each cycle, SetBlockInputs() is called to provide the block
  // this channel's inputs for the cycle.
  //
  // inputs is a map of the block's input port names to their driven values
  // for this_cycle. SetBlockInputs will set inputs[ready_name] to the Value
  // it should be driven for this_cycle.
  absl::Status SetBlockInputs(int64_t this_cycle,
                              absl::flat_hash_map<std::string, Value>& inputs,
                              absl::BitGenRef random_engine,
                              std::optional<verilog::ResetProto> reset);

  // For each cycle, GetBlockOutputs() is called to provide this channel
  // the block's outputs for the cycle.
  //
  // outputs is a map of output port names to their Values.
  absl::Status GetBlockOutputs(
      int64_t this_cycle,
      const absl::flat_hash_map<std::string, Value>& outputs);

  // Returns the sequence of values read from the block.
  absl::StatusOr<std::vector<uint64_t>> GetOutputSequenceAsUint64() const;
  absl::Span<const Value> GetOutputSequence() const { return data_sequence_; }

  // Returns the sequence of values read from the block each cycle.
  absl::StatusOr<std::vector<std::optional<uint64_t>>>
  GetOutputCycleSequenceAsUint64() const;
  absl::Span<const std::optional<Value>> GetOutputCycleSequence() const {
    return data_per_cycle_;
  }

 private:
  std::string data_name_;
  std::string valid_name_;
  std::string ready_name_;

  double lambda_ = 1.0;  // Receive with probability lambda.
  Block* block_ = nullptr;

  BehaviorDuringReset reset_behavior_;

  bool is_ready_ = false;             // Ready is asserted.
  std::vector<Value> data_sequence_;  // Data sequence received.
  std::vector<std::optional<Value>>
      data_per_cycle_;  // Data received each cycle.
};

struct BlockIOResults {
  std::vector<absl::flat_hash_map<std::string, Value>> inputs;
  std::vector<absl::flat_hash_map<std::string, Value>> outputs;
  InterpreterEvents interpreter_events;
};

struct BlockIOResultsAsUint64 {
  std::vector<absl::flat_hash_map<std::string, uint64_t>> inputs;
  std::vector<absl::flat_hash_map<std::string, uint64_t>> outputs;
  InterpreterEvents interpreter_events;
};

class BlockContinuation;
class BlockEvaluator {
 public:
  explicit constexpr BlockEvaluator(std::string_view name) : name_(name) {}
  virtual ~BlockEvaluator() = default;

  // Create a new block continuation with all registers initialized to the given
  // values. This continuation can be used to feed input values in
  // cycle-by-cycle.
  absl::StatusOr<std::unique_ptr<BlockContinuation>> NewContinuation(
      Block* block,
      const absl::flat_hash_map<std::string, Value>& initial_registers) const;

  // Create a new block continuation with all registers initialized to zero
  // values. This continuation can be used to feed input values in
  // cycle-by-cycle.
  absl::StatusOr<std::unique_ptr<BlockContinuation>> NewContinuation(
      Block* block) const;

  // The name of this evaluator for debug purposes.
  std::string_view name() const { return name_; }

  // Runs the evaluator on a combinational block. `inputs` must contain a
  // value for each input port in the block. The returned map contains a value
  // for each output port of the block.
  virtual absl::StatusOr<absl::flat_hash_map<std::string, Value>>
  EvaluateCombinationalBlock(
      Block* block,
      const absl::flat_hash_map<std::string, Value>& inputs) const;

  // Overload which accepts and returns uint64_t values instead of xls::Values.
  virtual absl::StatusOr<absl::flat_hash_map<std::string, uint64_t>>
  EvaluateCombinationalBlock(
      Block* block,
      const absl::flat_hash_map<std::string, uint64_t>& inputs) const;

  // Runs the evaluator on a block feeding a sequence of values to input ports
  // and returning the resulting sequence of values from the output
  // ports. Registers are clocked between each set of inputs fed to the block.
  // Initial register state is zero for all registers.
  virtual absl::StatusOr<std::vector<absl::flat_hash_map<std::string, Value>>>
  EvaluateSequentialBlock(
      Block* block,
      absl::Span<const absl::flat_hash_map<std::string, Value>> inputs) const;

  // Overload which accepts and returns uint64_t values instead of xls::Values.
  virtual absl::StatusOr<
      std::vector<absl::flat_hash_map<std::string, uint64_t>>>
  EvaluateSequentialBlock(
      Block* block,
      absl::Span<const absl::flat_hash_map<std::string, uint64_t>> inputs)
      const;

  // Runs the evaluator on a block.  Each input port in the block
  // should be given a sequence of data values to drive the block.
  //
  // This sequence of data values can be provided either by
  //  1. Providing a ChannelSource in channel_sources that will drive the
  //     data or valid port.
  //  2. Providing a ChannelSink in channel_sinks that will drive the ready
  //     port.
  //  3. Providing the sequence of Values as part of inputs, for any
  //     port (like rst) that is not part of a Channel definition.
  //
  // Registers are clocked between each set of inputs fed to the block.
  // Initial register state is zero for all registers.
  virtual absl::StatusOr<BlockIOResults> EvaluateChannelizedSequentialBlock(
      Block* block, absl::Span<ChannelSource> channel_sources,
      absl::Span<ChannelSink> channel_sinks,
      absl::Span<const absl::flat_hash_map<std::string, Value>> inputs,
      const std::optional<verilog::ResetProto>& reset, int64_t seed) const;

  // Runs the evaluator on a block.  Each input port in the block
  // should be given a sequence of data values to drive the block.
  //
  // This sequence of data values can be provided either by
  //  1. Providing a ChannelSource in channel_sources that will drive the
  //     data or valid port.
  //  2. Providing a ChannelSink in channel_sinks that will drive the ready
  //     port.
  //  3. Providing the sequence of Values as part of inputs, for any
  //     port (like rst) that is not part of a Channel definition.
  //
  // Registers are clocked between each set of inputs fed to the block.
  // Initial register state is zero for all registers.
  absl::StatusOr<BlockIOResults> EvaluateChannelizedSequentialBlock(
      Block* block, absl::Span<ChannelSource> channel_sources,
      absl::Span<ChannelSink> channel_sinks,
      absl::Span<const absl::flat_hash_map<std::string, Value>> inputs,
      const std::optional<verilog::ResetProto>& reset) const {
    return EvaluateChannelizedSequentialBlock(
        block, channel_sources, channel_sinks, inputs, reset, /*seed=*/0);
  }

  // Runs the evaluator on a block.  Each input port in the block
  // should be given a sequence of data values to drive the block.
  //
  // This sequence of data values can be provided either by
  //  1. Providing a ChannelSource in channel_sources that will drive the
  //     data or valid port.
  //  2. Providing a ChannelSink in channel_sinks that will drive the ready
  //     port.
  //  3. Providing the sequence of Values as part of inputs, for any
  //     port (like rst) that is not part of a Channel definition.
  //
  // Registers are clocked between each set of inputs fed to the block.
  // Initial register state is zero for all registers.
  absl::StatusOr<BlockIOResults> EvaluateChannelizedSequentialBlock(
      Block* block, absl::Span<ChannelSource> channel_sources,
      absl::Span<ChannelSink> channel_sinks,
      absl::Span<const absl::flat_hash_map<std::string, Value>> inputs) const {
    return EvaluateChannelizedSequentialBlock(
        block, channel_sources, channel_sinks, inputs, /*reset=*/std::nullopt);
  }

  // Variant which accepts and returns uint64_t values instead of xls::Values.
  virtual absl::StatusOr<BlockIOResultsAsUint64>
  EvaluateChannelizedSequentialBlockWithUint64(
      Block* block, absl::Span<ChannelSource> channel_sources,
      absl::Span<ChannelSink> channel_sinks,
      absl::Span<const absl::flat_hash_map<std::string, uint64_t>> inputs,
      const std::optional<verilog::ResetProto>& reset, int64_t seed) const;

  // Variant which accepts and returns uint64_t values instead of xls::Values.
  absl::StatusOr<BlockIOResultsAsUint64>
  EvaluateChannelizedSequentialBlockWithUint64(
      Block* block, absl::Span<ChannelSource> channel_sources,
      absl::Span<ChannelSink> channel_sinks,
      absl::Span<const absl::flat_hash_map<std::string, uint64_t>> inputs,
      const std::optional<verilog::ResetProto>& reset) const {
    return EvaluateChannelizedSequentialBlockWithUint64(
        block, channel_sources, channel_sinks, inputs, reset, /*seed=*/0);
  }

  // Variant which accepts and returns uint64_t values instead of xls::Values.
  absl::StatusOr<BlockIOResultsAsUint64>
  EvaluateChannelizedSequentialBlockWithUint64(
      Block* block, absl::Span<ChannelSource> channel_sources,
      absl::Span<ChannelSink> channel_sinks,
      absl::Span<const absl::flat_hash_map<std::string, uint64_t>> inputs)
      const {
    return EvaluateChannelizedSequentialBlockWithUint64(
        block, channel_sources, channel_sinks, inputs, /*reset=*/std::nullopt);
  }

  template <typename Sink>
  friend void AbslStringify(Sink& sink, const BlockEvaluator& b) {
    absl::Format(&sink, "%s", b.name());
  }

 protected:
  virtual absl::StatusOr<std::unique_ptr<BlockContinuation>>
  MakeNewContinuation(BlockElaboration&& elaboration,
                      const absl::flat_hash_map<std::string, Value>&
                          initial_registers) const = 0;

  std::string_view name_;
};

// A sequence of block runs with preserved registers.
class BlockContinuation {
 public:
  virtual ~BlockContinuation() = default;

  // Get the output-ports as they exist on the current cycle. This is only valid
  // until the next call to 'RunOneCycle'. The contents are undefined before the
  // first call to RunOneCycle.
  virtual const absl::flat_hash_map<std::string, Value>& output_ports() = 0;
  // Get the registers as they exist on the current cycle. The reference is only
  // valid until the next call to 'RunOneCycle'.
  virtual const absl::flat_hash_map<std::string, Value>& registers() = 0;
  // Get the interpreter events for the last cycle.
  virtual const InterpreterEvents& events() = 0;
  // Run a single cycle of the block on the given inputs using the current
  // register state.
  virtual absl::Status RunOneCycle(
      const absl::flat_hash_map<std::string, Value>& inputs) = 0;
  // Update the registers to the give values.
  virtual absl::Status SetRegisters(
      const absl::flat_hash_map<std::string, Value>& regs) = 0;
};

}  // namespace xls

#endif  // XLS_INTERPRETER_BLOCK_EVALUATOR_H_
