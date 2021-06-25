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

// An interpreter for XLS blocks.
class BlockInterpreter : public IrInterpreter {
 public:
  BlockInterpreter(const absl::flat_hash_map<std::string, Value>& inputs)
      : IrInterpreter(/*args=*/{}), inputs_(inputs) {}

  // Runs the interpreter on a combinational block. `inputs` must contain a
  // value for each input port in the block. The returned map contains a value
  // for each output port of the block.
  static absl::StatusOr<absl::flat_hash_map<std::string, Value>>
  RunCombinational(Block* block,
                   const absl::flat_hash_map<std::string, Value>& inputs);

  // Overload which accepts and returns uint64_t values instead of xls::Values.
  static absl::StatusOr<absl::flat_hash_map<std::string, uint64_t>>
  RunCombinational(Block* block,
                   const absl::flat_hash_map<std::string, uint64_t>& inputs);

  absl::Status HandleInputPort(InputPort* input_port) override;
  absl::Status HandleOutputPort(OutputPort* output_port) override;

 private:
  absl::flat_hash_map<std::string, Value> inputs_;
};

}  // namespace xls

#endif  // XLS_INTERPRETER_BLOCK_INTERPRETER_H_
