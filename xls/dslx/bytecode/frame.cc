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

#include "xls/dslx/bytecode/frame.h"

#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "xls/common/logging/logging.h"
#include "xls/dslx/bytecode/bytecode.h"
#include "xls/dslx/interp_value.h"
#include "xls/dslx/type_system/parametric_env.h"
#include "xls/dslx/type_system/type_info.h"

namespace xls::dslx {

Frame::Frame(BytecodeFunction* bf, std::vector<InterpValue> args,
             const TypeInfo* type_info,
             const std::optional<ParametricEnv>& bindings,
             std::vector<InterpValue> initial_args,
             std::unique_ptr<BytecodeFunction> bf_holder)
    : pc_(0),
      slots_(std::move(args)),
      bf_(bf),
      type_info_(type_info),
      bindings_(bindings),
      initial_args_(std::move(initial_args)),
      bf_holder_(std::move(bf_holder)) {
  // Note: bf->owner() can apparently be null for "helper" bytecode sequences we
  // generate, like for map() operations.
  if (bf != nullptr && bf->owner() != nullptr && type_info != nullptr) {
    CHECK_EQ(bf->owner(), type_info->module());
  }
}

void Frame::StoreSlot(Bytecode::SlotIndex slot, InterpValue value) {
  // Slots are usually encountered in order of use (and assignment), except for
  // those declared inside conditional branches, which may never be seen,
  // so we may have to add more than one slot at a time in such cases.
  while (slots_.size() <= slot.value()) {
    slots_.push_back(InterpValue::MakeToken());
  }

  slots_.at(slot.value()) = std::move(value);
}

}  // namespace xls::dslx
