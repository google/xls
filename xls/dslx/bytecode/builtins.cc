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

#include "xls/dslx/bytecode/builtins.h"

#include <cstdint>
#include <functional>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xls/common/logging/logging.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/dslx/bytecode/bytecode.h"
#include "xls/dslx/bytecode/interpreter_stack.h"
#include "xls/dslx/interp_value.h"
#include "xls/ir/bits.h"
#include "xls/ir/bits_ops.h"

namespace xls::dslx {
namespace {

absl::Status RunTernaryBuiltin(
    const std::function<absl::StatusOr<InterpValue>(
        const InterpValue& a, const InterpValue& b, const InterpValue& c)>& fn,
    InterpreterStack& stack) {
  XLS_RET_CHECK_GE(stack.size(), 3);
  XLS_ASSIGN_OR_RETURN(InterpValue c, stack.Pop());
  XLS_ASSIGN_OR_RETURN(InterpValue b, stack.Pop());
  XLS_ASSIGN_OR_RETURN(InterpValue a, stack.Pop());
  XLS_ASSIGN_OR_RETURN(InterpValue result, fn(a, b, c));
  stack.Push(result);
  return absl::OkStatus();
}

}  // namespace

absl::Status RunBuiltinSlice(const Bytecode& bytecode,
                             InterpreterStack& stack) {
  return RunTernaryBuiltin(
      [](const InterpValue& basis, const InterpValue& start,
         const InterpValue& type_value) {
        return basis.Slice(start, type_value);
      },
      stack);
}

absl::Status RunBuiltinUpdate(const Bytecode& bytecode,
                              InterpreterStack& stack) {
  return RunTernaryBuiltin(
      [](const InterpValue& array, const InterpValue& index,
         const InterpValue& new_value) {
        return array.Update(index, new_value);
      },
      stack);
}

absl::Status RunBuiltinBitSlice(const Bytecode& bytecode,
                                InterpreterStack& stack) {
  XLS_VLOG(3) << "Executing builtin BitSlice.";
  return RunTernaryBuiltin(
      [](const InterpValue& subject, const InterpValue& start,
         const InterpValue& width) -> absl::StatusOr<InterpValue> {
        XLS_ASSIGN_OR_RETURN(Bits subject_bits, subject.GetBits());
        XLS_ASSIGN_OR_RETURN(int64_t start_index, start.GetBitValueInt64());
        if (start_index >= subject_bits.bit_count()) {
          start_index = subject_bits.bit_count();
        }
        XLS_ASSIGN_OR_RETURN(int64_t bit_count, width.GetBitCount());
        return InterpValue::MakeBits(
            /*is_signed=*/false, subject_bits.Slice(start_index, bit_count));
      },
      stack);

  return absl::OkStatus();
}

absl::Status RunBuiltinBitSliceUpdate(const Bytecode& bytecode,
                                      InterpreterStack& stack) {
  XLS_VLOG(3) << "Executing builtin BitSliceUpdate.";

  return RunTernaryBuiltin(
      [](const InterpValue& subject, const InterpValue& start,
         const InterpValue& update_value) -> absl::StatusOr<InterpValue> {
        XLS_ASSIGN_OR_RETURN(Bits subject_bits, subject.GetBits());
        XLS_ASSIGN_OR_RETURN(Bits start_bits, start.GetBits());
        XLS_ASSIGN_OR_RETURN(Bits update_value_bits, update_value.GetBits());

        if (bits_ops::UGreaterThanOrEqual(start_bits,
                                          subject_bits.bit_count())) {
          // Update is entirely out of bounds, so no bits of the subject are
          // updated.
          return InterpValue::MakeBits(InterpValueTag::kUBits, subject_bits);
        }

        XLS_ASSIGN_OR_RETURN(int64_t start_index, start_bits.ToUint64());
        return InterpValue::MakeBits(
            InterpValueTag::kUBits,
            bits_ops::BitSliceUpdate(subject_bits, start_index,
                                     update_value_bits));
      },
      stack);
}

}  // namespace xls::dslx
