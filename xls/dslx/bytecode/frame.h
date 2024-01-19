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

#ifndef XLS_DSLX_BYTECODE_FRAME_H_
#define XLS_DSLX_BYTECODE_FRAME_H_

#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

#include "xls/dslx/bytecode/bytecode.h"
#include "xls/dslx/interp_value.h"
#include "xls/dslx/type_system/parametric_env.h"
#include "xls/dslx/type_system/type_info.h"

namespace xls::dslx {

// Represents a frame on the function stack: holds the program counter, local
// storage, and instructions to execute.
class Frame {
 public:
  // Args:
  //  bindings: holds the bindings used to instantiate the BytecodeFunction's
  //    source Function, if it is parametric.
  //  bf_holder: stores the pointer to ephemeral functions, e.g., those
  //    generated on-the-fly from interpreting the `map` operation. For other
  //    cases, the BytecodeCache will own BytecodeFunction storage.
  //  initial_args: holds the set of args used to initially construct the frame
  //    (i.e., the arguments to this function). This is necessary for comparing
  //    results to a reference, e.g., the JIT via BytecodeInterpreter's
  //    post_fn_eval_hook.
  Frame(BytecodeFunction* bf, std::vector<InterpValue> args,
        const TypeInfo* type_info, const std::optional<ParametricEnv>& bindings,
        std::vector<InterpValue> initial_args,
        std::unique_ptr<BytecodeFunction> bf_holder = nullptr);

  int64_t pc() const { return pc_; }
  void set_pc(int64_t pc) { pc_ = pc; }
  void IncrementPc() { pc_++; }
  std::vector<InterpValue>& slots() { return slots_; }
  BytecodeFunction* bf() const { return bf_; }
  const TypeInfo* type_info() const { return type_info_; }
  const std::optional<ParametricEnv>& bindings() const { return bindings_; }
  const std::vector<InterpValue>& initial_args() { return initial_args_; }

  void StoreSlot(Bytecode::SlotIndex slot_index, InterpValue value);

 private:
  int64_t pc_;
  std::vector<InterpValue> slots_;
  BytecodeFunction* bf_;
  const TypeInfo* type_info_;
  std::optional<ParametricEnv> bindings_;
  std::vector<InterpValue> initial_args_;

  std::unique_ptr<BytecodeFunction> bf_holder_;
};

}  // namespace xls::dslx

#endif  // XLS_DSLX_BYTECODE_FRAME_H_
