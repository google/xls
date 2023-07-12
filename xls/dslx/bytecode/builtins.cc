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
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xls/common/logging/logging.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/dslx/bytecode/bytecode.h"
#include "xls/dslx/bytecode/interpreter_stack.h"
#include "xls/dslx/interp_value.h"
#include "xls/dslx/interp_value_helpers.h"
#include "xls/ir/bits.h"
#include "xls/ir/bits_ops.h"

namespace xls::dslx {
namespace {

absl::Status RunBinaryBuiltin(
    const std::function<absl::StatusOr<InterpValue>(const InterpValue& a,
                                                    const InterpValue& b)>& fn,
    InterpreterStack& stack) {
  XLS_RET_CHECK_GE(stack.size(), 2);
  XLS_ASSIGN_OR_RETURN(InterpValue b, stack.Pop());
  XLS_ASSIGN_OR_RETURN(InterpValue a, stack.Pop());
  XLS_ASSIGN_OR_RETURN(InterpValue result, fn(a, b));
  stack.Push(result);
  return absl::OkStatus();
}

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

absl::Status BuiltinRangeInternal(InterpreterStack& stack) {
  return RunBinaryBuiltin(
      [](const InterpValue& start,
         const InterpValue& end) -> absl::StatusOr<InterpValue> {
        XLS_RET_CHECK(start.IsBits());
        XLS_RET_CHECK(end.IsBits());
        XLS_ASSIGN_OR_RETURN(InterpValue start_ge_end, start.Ge(end));
        if (start_ge_end.IsTrue()) {
          return InterpValue::MakeArray({});
        }

        std::vector<InterpValue> elements;
        InterpValue cur = start;
        XLS_ASSIGN_OR_RETURN(InterpValue done, cur.Ge(end));
        XLS_ASSIGN_OR_RETURN(int64_t cur_bits, cur.GetBitCount());
        InterpValue one(cur.IsSigned() ? InterpValue::MakeSBits(cur_bits, 1)
                                       : InterpValue::MakeUBits(cur_bits, 1));
        while (done.IsFalse()) {
          elements.push_back(cur);
          XLS_ASSIGN_OR_RETURN(cur, cur.Add(one));
          XLS_ASSIGN_OR_RETURN(done, cur.Ge(end));
        }
        return InterpValue::MakeArray(elements);
      },
      stack);
}

absl::Status RunBuiltinGate(const Bytecode& bytecode, InterpreterStack& stack) {
  return RunBinaryBuiltin(
      [](const InterpValue& pass_value,
         const InterpValue& value) -> absl::StatusOr<InterpValue> {
        if (pass_value.IsTrue()) {
          return value;
        }

        return CreateZeroValue(value);
      },
      stack);
}

absl::Status RunBuiltinOneHot(const Bytecode& bytecode,
                              InterpreterStack& stack) {
  return RunBinaryBuiltin(
      [](const InterpValue& input,
         const InterpValue& lsb_is_prio) -> absl::StatusOr<InterpValue> {
        return input.OneHot(lsb_is_prio.IsTrue());
      },
      stack);
}

absl::Status RunBuiltinOneHotSel(const Bytecode& bytecode,
                                 InterpreterStack& stack) {
  return RunBinaryBuiltin(
      [](const InterpValue& selector,
         const InterpValue& cases_array) -> absl::StatusOr<InterpValue> {
        XLS_ASSIGN_OR_RETURN(Bits selector_bits, selector.GetBits());
        XLS_ASSIGN_OR_RETURN(const std::vector<InterpValue>* cases,
                             cases_array.GetValues());
        if (cases->empty()) {
          return absl::InternalError(
              "At least one case must be specified for one_hot_sel.");
        }
        XLS_ASSIGN_OR_RETURN(int64_t result_bit_count,
                             cases->at(0).GetBitCount());
        Bits result(result_bit_count);
        for (int i = 0; i < cases->size(); i++) {
          if (!selector_bits.Get(i)) {
            continue;
          }

          XLS_ASSIGN_OR_RETURN(Bits case_bits, cases->at(i).GetBits());
          result = bits_ops::Or(result, case_bits);
        }

        return InterpValue::MakeBits(cases->at(0).tag(), result);
      },
      stack);
}

absl::Status RunBuiltinPrioritySel(const Bytecode& bytecode,
                                   InterpreterStack& stack) {
  return RunBinaryBuiltin(
      [](const InterpValue& selector,
         const InterpValue& cases_array) -> absl::StatusOr<InterpValue> {
        XLS_ASSIGN_OR_RETURN(Bits selector_bits, selector.GetBits());
        XLS_ASSIGN_OR_RETURN(const std::vector<InterpValue>* cases,
                             cases_array.GetValues());
        if (cases->empty()) {
          return absl::InternalError(
              "At least one case must be specified for priority_sel.");
        }
        XLS_ASSIGN_OR_RETURN(int64_t result_bit_count,
                             cases->at(0).GetBitCount());
        for (int i = 0; i < cases->size(); i++) {
          if (selector_bits.Get(i)) {
            XLS_ASSIGN_OR_RETURN(Bits case_bits, cases->at(i).GetBits());
            return InterpValue::MakeBits(cases->at(0).tag(), case_bits);
          }
        }

        Bits empty_result(result_bit_count);
        return InterpValue::MakeBits(cases->at(0).tag(), empty_result);
      },
      stack);
}

absl::Status RunBuiltinSignex(const Bytecode& bytecode,
                              InterpreterStack& stack) {
  return RunBinaryBuiltin(
      [](const InterpValue& value,
         const InterpValue& type_value) -> absl::StatusOr<InterpValue> {
        XLS_ASSIGN_OR_RETURN(int64_t old_bit_count, value.GetBitCount());
        XLS_ASSIGN_OR_RETURN(int64_t new_bit_count, type_value.GetBitCount());
        XLS_ASSIGN_OR_RETURN(Bits bits, value.GetBits());
        if (new_bit_count < old_bit_count) {
          return InterpValue::MakeBits(
              type_value.IsSigned(),
              bits.Slice(/*start=*/0, /*width=*/new_bit_count));
        }
        return InterpValue::MakeBits(type_value.IsSigned(),
                                     bits_ops::SignExtend(bits, new_bit_count));
      },
      stack);
}

absl::Status RunBuiltinSMulp(const Bytecode& bytecode,
                             InterpreterStack& stack) {
  return RunBinaryBuiltin(
      [](const InterpValue& lhs,
         const InterpValue& rhs) -> absl::StatusOr<InterpValue> {
        XLS_ASSIGN_OR_RETURN(int64_t lhs_bitwidth, lhs.GetBitCount());
        XLS_ASSIGN_OR_RETURN(int64_t rhs_bitwidth, lhs.GetBitCount());
        XLS_CHECK_EQ(lhs_bitwidth, rhs_bitwidth);
        int64_t product_bitwidth = lhs_bitwidth;
        std::vector<InterpValue> outputs;
        InterpValue offset = InterpValue::MakeUnsigned(
            MulpOffsetForSimulation(product_bitwidth, /*shift_size=*/1));
        XLS_ASSIGN_OR_RETURN(InterpValue product, lhs.Mul(rhs));
        // Return unsigned partial product.
        XLS_ASSIGN_OR_RETURN(Bits product_raw_bits, product.GetBits());
        product = InterpValue::MakeUnsigned(product_raw_bits);
        XLS_ASSIGN_OR_RETURN(InterpValue product_minus_offset,
                             product.Sub(offset));
        outputs.push_back(offset);
        outputs.push_back(product_minus_offset);
        return InterpValue::MakeTuple(outputs);
      },
      stack);
}

absl::Status RunBuiltinUMulp(const Bytecode& bytecode,
                             InterpreterStack& stack) {
  return RunBinaryBuiltin(
      [](const InterpValue& lhs,
         const InterpValue& rhs) -> absl::StatusOr<InterpValue> {
        XLS_ASSIGN_OR_RETURN(int64_t lhs_bitwidth, lhs.GetBitCount());
        XLS_ASSIGN_OR_RETURN(int64_t rhs_bitwidth, lhs.GetBitCount());
        XLS_CHECK_EQ(lhs_bitwidth, rhs_bitwidth);
        int64_t product_bitwidth = lhs_bitwidth;
        std::vector<InterpValue> outputs;
        InterpValue offset = InterpValue::MakeUnsigned(
            MulpOffsetForSimulation(product_bitwidth, /*shift_size=*/1));
        XLS_ASSIGN_OR_RETURN(InterpValue product, lhs.Mul(rhs));
        XLS_ASSIGN_OR_RETURN(InterpValue product_minus_offset,
                             product.Sub(offset));
        outputs.push_back(offset);
        outputs.push_back(product_minus_offset);
        return InterpValue::MakeTuple(outputs);
      },
      stack);
}

}  // namespace xls::dslx
