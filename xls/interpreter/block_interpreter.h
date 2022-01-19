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

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
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

}  // namespace xls

#endif  // XLS_INTERPRETER_BLOCK_INTERPRETER_H_
