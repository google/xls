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

#include <random>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xls/codegen/module_signature.pb.h"
#include "xls/interpreter/ir_interpreter.h"
#include "xls/ir/block.h"
#include "xls/ir/value.h"

namespace xls {

struct BlockRunResult {
  absl::flat_hash_map<std::string, Value> outputs;
  absl::flat_hash_map<std::string, Value> reg_state;
};

// Runs a single cycle of a block with the given register values and input
// values. Returns the value sent to the output port and the next register
// state.
absl::StatusOr<BlockRunResult> BlockRun(
    const absl::flat_hash_map<std::string, Value>& inputs,
    const absl::flat_hash_map<std::string, Value>& reg_state, Block* block);

// Runs the interpreter on a combinational block. `inputs` must contain a
// value for each input port in the block. The returned map contains a value
// for each output port of the block.
absl::StatusOr<absl::flat_hash_map<std::string, Value>>
InterpretCombinationalBlock(
    Block* block, const absl::flat_hash_map<std::string, Value>& inputs);

// Overload which accepts and returns uint64_t values instead of xls::Values.
absl::StatusOr<absl::flat_hash_map<std::string, uint64_t>>
InterpretCombinationalBlock(
    Block* block, const absl::flat_hash_map<std::string, uint64_t>& inputs);

// Runs the interpreter on a block feeding a sequence of values to input ports
// and returning the resulting sequence of values from the output
// ports. Registers are clocked between each set of inputs fed to the block.
// Initial register state is zero for all registers.
absl::StatusOr<std::vector<absl::flat_hash_map<std::string, Value>>>
InterpretSequentialBlock(
    Block* block,
    absl::Span<const absl::flat_hash_map<std::string, Value>> inputs);

// Overload which accepts and returns uint64_t values instead of xls::Values.
absl::StatusOr<std::vector<absl::flat_hash_map<std::string, uint64_t>>>
InterpretSequentialBlock(
    Block* block,
    absl::Span<const absl::flat_hash_map<std::string, uint64_t>> inputs);

// Drives input channel simulation for testing blocks.
//
// For each successive input, new data is driven after a randomized delay.
class ChannelSource {
 public:
  // Construct a ChannelSource given port names for data/ready/valid
  // for the given block.
  //
  // lambda is the probability that the next value in the input sequence
  // (set by SetDataSequence), will be put on an the channel and valid
  // asserted (if the channel is not otherwise occupied by a in-progress
  // transaction).  Once valid is asserted, it will remain asserted until
  // the transaction completes via ready being asserted.
  ChannelSource(std::string_view data_name, std::string_view valid_name,
                std::string_view ready_name, double lambda, Block* block)
      : data_name_(data_name),
        valid_name_(valid_name),
        ready_name_(ready_name),
        lambda_(lambda),
        block_(block) {}

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
                              std::minstd_rand& random_engine,
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
  // Construct a ChannelSource given port names for data/ready/valid
  // for the given block.
  //
  // lambda is the probability that for a given cycle, the sink will assert
  // ready.
  ChannelSink(std::string_view data_name, std::string_view valid_name,
              std::string_view ready_name, double lambda, Block* block)
      : data_name_(data_name),
        valid_name_(valid_name),
        ready_name_(ready_name),
        lambda_(lambda),
        block_(block) {}

  // For each cycle, SetBlockInputs() is called to provide the block
  // this channel's inputs for the cycle.
  //
  // inputs is a map of the block's input port names to their driven values
  // for this_cycle. SetBlockInputs will set inputs[ready_name] to the Value
  // it should be driven for this_cycle.
  absl::Status SetBlockInputs(int64_t this_cycle,
                              absl::flat_hash_map<std::string, Value>& inputs,
                              std::minstd_rand& random_engine);

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

 private:
  std::string data_name_;
  std::string valid_name_;
  std::string ready_name_;

  double lambda_ = 1.0;  // Receive with probability lambda.
  Block* block_ = nullptr;

  bool is_ready_ = false;             // Ready is asserted.
  std::vector<Value> data_sequence_;  // Data sequence received.
};

struct BlockIOResults {
  std::vector<absl::flat_hash_map<std::string, Value>> inputs;
  std::vector<absl::flat_hash_map<std::string, Value>> outputs;
};

struct BlockIOResultsAsUint64 {
  std::vector<absl::flat_hash_map<std::string, uint64_t>> inputs;
  std::vector<absl::flat_hash_map<std::string, uint64_t>> outputs;
};

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
absl::StatusOr<BlockIOResults> InterpretChannelizedSequentialBlock(
    Block* block, absl::Span<ChannelSource> channel_sources,
    absl::Span<ChannelSink> channel_sinks,
    absl::Span<const absl::flat_hash_map<std::string, Value>> inputs,
    std::optional<verilog::ResetProto> reset = std::nullopt, int64_t seed = 0);

// Variant which accepts and returns uint64_t values instead of xls::Values.
absl::StatusOr<BlockIOResultsAsUint64>
InterpretChannelizedSequentialBlockWithUint64(
    Block* block, absl::Span<ChannelSource> channel_sources,
    absl::Span<ChannelSink> channel_sinks,
    absl::Span<const absl::flat_hash_map<std::string, uint64_t>> inputs,
    std::optional<verilog::ResetProto> reset = std::nullopt, int64_t seed = 0);

}  // namespace xls

#endif  // XLS_INTERPRETER_BLOCK_INTERPRETER_H_
