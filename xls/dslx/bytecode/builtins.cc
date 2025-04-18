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

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iterator>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_split.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/dslx/bytecode/bytecode.h"
#include "xls/dslx/bytecode/bytecode_interpreter_options.h"
#include "xls/dslx/bytecode/frame.h"
#include "xls/dslx/bytecode/interpreter_stack.h"
#include "xls/dslx/errors.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/frontend/proc_id.h"
#include "xls/dslx/interp_value.h"
#include "xls/dslx/interp_value_utils.h"
#include "xls/dslx/make_value_format_descriptor.h"
#include "xls/dslx/type_system/type.h"
#include "xls/dslx/type_system/type_info.h"
#include "xls/dslx/value_format_descriptor.h"
#include "xls/ir/bits.h"
#include "xls/ir/bits_ops.h"
#include "xls/ir/format_preference.h"

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

absl::Status RunBuiltinArraySlice(const Bytecode& bytecode,
                                  InterpreterStack& stack) {
  return RunTernaryBuiltin(
      [](const InterpValue& basis, const InterpValue& start,
         const InterpValue& type_value) -> absl::StatusOr<InterpValue> {
        XLS_RET_CHECK(basis.IsArray());
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
  VLOG(3) << "Executing builtin BitSlice.";
  return RunTernaryBuiltin(
      [](const InterpValue& subject, const InterpValue& start,
         const InterpValue& width) -> absl::StatusOr<InterpValue> {
        XLS_ASSIGN_OR_RETURN(Bits subject_bits, subject.GetBits());
        XLS_ASSIGN_OR_RETURN(int64_t start_index, start.GetBitValueSigned());
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
  VLOG(3) << "Executing builtin BitSliceUpdate.";

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

absl::Status RunBuiltinEncode(const Bytecode& bytecode,
                              InterpreterStack& stack) {
  VLOG(3) << "Executing builtin encode.";
  XLS_RET_CHECK(!stack.empty());

  XLS_ASSIGN_OR_RETURN(InterpValue input, stack.Pop());
  XLS_ASSIGN_OR_RETURN(InterpValue output, input.Encode());
  stack.Push(std::move(output));

  return absl::OkStatus();
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
        for (int64_t i = 0; i < cases->size(); i++) {
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
  return RunTernaryBuiltin(
      [](const InterpValue& selector, const InterpValue& cases_array,
         const InterpValue& default_value) -> absl::StatusOr<InterpValue> {
        XLS_ASSIGN_OR_RETURN(Bits selector_bits, selector.GetBits());
        XLS_ASSIGN_OR_RETURN(const std::vector<InterpValue>* cases,
                             cases_array.GetValues());
        if (cases->empty()) {
          return absl::InternalError(
              "At least one case must be specified for priority_sel.");
        }
        for (int64_t i = 0; i < cases->size(); i++) {
          if (selector_bits.Get(i)) {
            XLS_ASSIGN_OR_RETURN(Bits case_bits, cases->at(i).GetBits());
            return InterpValue::MakeBits(cases->at(0).tag(), case_bits);
          }
        }

        XLS_ASSIGN_OR_RETURN(Bits default_bits, default_value.GetBits());
        return InterpValue::MakeBits(cases->at(0).tag(), default_bits);
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
        CHECK_EQ(lhs_bitwidth, rhs_bitwidth);
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
        CHECK_EQ(lhs_bitwidth, rhs_bitwidth);
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

absl::Status RunBuiltinAndReduce(const Bytecode& bytecode,
                                 InterpreterStack& stack) {
  VLOG(3) << "Executing builtin AndReduce.";
  XLS_RET_CHECK(!stack.empty());
  XLS_ASSIGN_OR_RETURN(InterpValue value, stack.Pop());
  XLS_ASSIGN_OR_RETURN(Bits bits, value.GetBits());
  bits = bits_ops::AndReduce(bits);
  stack.Push(InterpValue::MakeBool(bits.IsOne()));
  return absl::OkStatus();
}

// Returns appropriately formatted lhs and rhs values for use within an
// assert_eq (lt, etc) message. Assert failure messages preserve the value
// formatting (hex, binary) of literal numbers which appear in the assert
// call. For example, in `assert_eq(x, u32:0b10101)`, values will appear in
// binary format in the failure message.
static absl::StatusOr<std::pair<std::string, std::string>>
GetAssertFormattedStrings(const InterpreterStack::FormattedInterpValue& lhs,
                          const InterpreterStack::FormattedInterpValue& rhs,
                          const Bytecode& bytecode, const Frame& frame,
                          const BytecodeInterpreterOptions& options) {
  auto formatted_pair = [&](const ValueFormatDescriptor& fmt)
      -> absl::StatusOr<std::pair<std::string, std::string>> {
    XLS_ASSIGN_OR_RETURN(
        std::string lhs_string,
        lhs.value.ToFormattedString(fmt, /*include_type_prefix=*/true));
    XLS_ASSIGN_OR_RETURN(
        std::string rhs_string,
        rhs.value.ToFormattedString(fmt, /*include_type_prefix=*/true));
    return std::make_pair(lhs_string, rhs_string);
  };

  if (options.format_preference() == FormatPreference::kDefault &&
      (lhs.format_descriptor.has_value() ||
       rhs.format_descriptor.has_value())) {
    return formatted_pair(lhs.format_descriptor.has_value()
                              ? *lhs.format_descriptor
                              : *rhs.format_descriptor);
  }
  const TypeInfo* type_info = frame.type_info();
  XLS_ASSIGN_OR_RETURN(Bytecode::InvocationData invocation_data,
                       bytecode.invocation_data());
  XLS_ASSIGN_OR_RETURN(
      Type * lhs_type,
      type_info->GetItemOrError(invocation_data.invocation()->args()[0]));
  XLS_ASSIGN_OR_RETURN(
      Type * rhs_type,
      type_info->GetItemOrError(invocation_data.invocation()->args()[1]));
  XLS_RET_CHECK_EQ(*lhs_type, *rhs_type) << absl::StreamFormat(
      "Assert arguments are not the same type: `%s` vs `%s`",
      lhs_type->ToString(), rhs_type->ToString());
  XLS_ASSIGN_OR_RETURN(
      ValueFormatDescriptor fmt,
      MakeValueFormatDescriptor(*lhs_type, options.format_preference()));
  return formatted_pair(fmt);
}

absl::Status RunBuiltinAssertEq(const Bytecode& bytecode,
                                InterpreterStack& stack, const Frame& frame,
                                const BytecodeInterpreterOptions& options,
                                const std::optional<ProcId>& caller_proc_id) {
  VLOG(3) << "Executing builtin AssertEq.";
  XLS_RET_CHECK_GE(stack.size(), 2);

  XLS_ASSIGN_OR_RETURN(InterpreterStack::FormattedInterpValue formatted_rhs,
                       stack.PopFormattedValue());
  XLS_ASSIGN_OR_RETURN(InterpreterStack::FormattedInterpValue formatted_lhs,
                       stack.PopFormattedValue());
  const InterpValue& lhs = formatted_lhs.value;
  const InterpValue& rhs = formatted_rhs.value;
  stack.Push(InterpValue::MakeUnit());
  bool eq = lhs.Eq(rhs);
  if (!eq) {
    XLS_ASSIGN_OR_RETURN((const auto& [lhs_string, rhs_string]),
                         GetAssertFormattedStrings(formatted_lhs, formatted_rhs,
                                                   bytecode, frame, options));
    std::string message =
        absl::StrContains(lhs_string, '\n')
            ? absl::StrCat(
                  "\n  lhs and rhs were not equal:\n",
                  HighlightLineByLineDifferences(lhs_string, rhs_string))
            : absl::StrFormat("\n  lhs: %s\n  rhs: %s\n  were not equal",
                              lhs_string, rhs_string);
    if (lhs.IsArray() && rhs.IsArray()) {
      XLS_ASSIGN_OR_RETURN(
          std::optional<int64_t> i,
          FindFirstDifferingIndex(lhs.GetValuesOrDie(), rhs.GetValuesOrDie()));
      XLS_RET_CHECK(i.has_value());
      const auto& lhs_values = lhs.GetValuesOrDie();
      const auto& rhs_values = rhs.GetValuesOrDie();
      message += absl::StrFormat("; first differing index: %d :: %s vs %s", *i,
                                 lhs_values[*i].ToHumanString(),
                                 rhs_values[*i].ToHumanString());
    }
    if (caller_proc_id.has_value()) {
      message += absl::StrFormat(" (called from %s)",
                                 caller_proc_id.value().ToString());
    }
    return FailureErrorStatus(bytecode.source_span(), message,
                              stack.file_table());
  }

  return absl::OkStatus();
}

absl::Status RunBuiltinAssertLt(const Bytecode& bytecode,
                                InterpreterStack& stack, const Frame& frame,
                                const BytecodeInterpreterOptions& options,
                                const std::optional<ProcId>& caller_proc_id) {
  VLOG(3) << "Executing builtin AssertLt.";
  XLS_RET_CHECK_GE(stack.size(), 2);

  XLS_ASSIGN_OR_RETURN(InterpreterStack::FormattedInterpValue formatted_rhs,
                       stack.PopFormattedValue());
  XLS_ASSIGN_OR_RETURN(InterpreterStack::FormattedInterpValue formatted_lhs,
                       stack.PopFormattedValue());
  const InterpValue& lhs = formatted_lhs.value;
  const InterpValue& rhs = formatted_rhs.value;
  stack.Push(InterpValue::MakeUnit());
  XLS_ASSIGN_OR_RETURN(InterpValue lt_value, lhs.Lt(rhs));
  bool lt = lt_value.IsTrue();
  if (!lt) {
    XLS_ASSIGN_OR_RETURN((const auto& [lhs_string, rhs_string]),
                         GetAssertFormattedStrings(formatted_lhs, formatted_rhs,
                                                   bytecode, frame, options));
    std::string message = absl::StrFormat(
        "\n  lhs: %s\n was not less than rhs: %s", lhs_string, rhs_string);
    if (caller_proc_id.has_value()) {
      message += absl::StrFormat(" (called from %s)",
                                 caller_proc_id.value().ToString());
    }
    return FailureErrorStatus(bytecode.source_span(), message,
                              stack.file_table());
  }

  return absl::OkStatus();
}

absl::Status RunBuiltinCeilLog2(const Bytecode& bytecode,
                                InterpreterStack& stack) {
  VLOG(3) << "Executing builtin ceillog2.";
  XLS_RET_CHECK(!stack.empty());

  XLS_ASSIGN_OR_RETURN(InterpValue input, stack.Pop());
  XLS_ASSIGN_OR_RETURN(InterpValue result, input.CeilOfLog2());
  stack.Push(result);

  return absl::OkStatus();
}

absl::Status RunBuiltinClz(const Bytecode& bytecode, InterpreterStack& stack) {
  VLOG(3) << "Executing builtin clz.";
  XLS_RET_CHECK(!stack.empty());

  XLS_ASSIGN_OR_RETURN(InterpValue input, stack.Pop());
  XLS_ASSIGN_OR_RETURN(Bits bits, input.GetBits());
  stack.Push(
      InterpValue::MakeUBits(bits.bit_count(), bits.CountLeadingZeros()));

  return absl::OkStatus();
}

absl::Status RunBuiltinCover(const Bytecode& bytecode,
                             InterpreterStack& stack) {
  VLOG(3) << "Executing builtin `cover!`";
  XLS_RET_CHECK_GE(stack.size(), 2);

  XLS_ASSIGN_OR_RETURN(InterpValue string, stack.Pop());
  XLS_ASSIGN_OR_RETURN(InterpValue predicate, stack.Pop());
  stack.Push(InterpValue::MakeToken());

  return absl::OkStatus();
}

absl::Status RunBuiltinCtz(const Bytecode& bytecode, InterpreterStack& stack) {
  VLOG(3) << "Executing builtin ctz.";
  XLS_RET_CHECK(!stack.empty());

  XLS_ASSIGN_OR_RETURN(InterpValue input, stack.Pop());
  XLS_ASSIGN_OR_RETURN(Bits bits, input.GetBits());
  stack.Push(
      InterpValue::MakeUBits(bits.bit_count(), bits.CountTrailingZeros()));

  return absl::OkStatus();
}

absl::Status RunBuiltinEnumerate(const Bytecode& bytecode,
                                 InterpreterStack& stack) {
  XLS_RET_CHECK(!stack.empty());
  XLS_ASSIGN_OR_RETURN(InterpValue input, stack.Pop());
  XLS_ASSIGN_OR_RETURN(const std::vector<InterpValue>* values,
                       input.GetValues());

  std::vector<InterpValue> elements;
  elements.reserve(values->size());
  for (int32_t i = 0; i < values->size(); i++) {
    elements.push_back(
        InterpValue::MakeTuple({InterpValue::MakeU32(i), values->at(i)}));
  }
  XLS_ASSIGN_OR_RETURN(InterpValue result, InterpValue::MakeArray(elements));
  stack.Push(result);
  return absl::OkStatus();
}

absl::Status RunBuiltinOrReduce(const Bytecode& bytecode,
                                InterpreterStack& stack) {
  VLOG(3) << "Executing builtin OrReduce.";
  XLS_RET_CHECK(!stack.empty());
  XLS_ASSIGN_OR_RETURN(InterpValue value, stack.Pop());
  XLS_ASSIGN_OR_RETURN(Bits bits, value.GetBits());
  bits = bits_ops::OrReduce(bits);
  stack.Push(InterpValue::MakeBool(bits.IsOne()));
  return absl::OkStatus();
}

absl::Status RunBuiltinRange(const Bytecode& bytecode,
                             InterpreterStack& stack) {
  return BuiltinRangeInternal(stack);
}

absl::Status RunBuiltinArrayRev(const Bytecode& bytecode,
                                InterpreterStack& stack) {
  XLS_RET_CHECK(!stack.empty());
  XLS_ASSIGN_OR_RETURN(InterpValue value, stack.Pop());
  if (!value.IsArray()) {
    return absl::InvalidArgumentError(
        "Argument to `array_rev` builtin must be an array.");
  }
  std::vector<InterpValue> reversed;
  const auto& values = value.GetValuesOrDie();
  std::reverse_copy(values.begin(), values.end(), std::back_inserter(reversed));
  stack.Push(InterpValue::MakeArray(std::move(reversed)).value());
  return absl::OkStatus();
}

absl::Status RunBuiltinArraySize(const Bytecode& bytecode,
                                 InterpreterStack& stack) {
  XLS_RET_CHECK(!stack.empty());
  XLS_ASSIGN_OR_RETURN(InterpValue value, stack.Pop());
  if (!value.IsArray()) {
    return absl::InvalidArgumentError(
        "Argument to `array_rev` builtin must be an array.");
  }
  int64_t length = value.GetLength().value();
  auto length_u32 = static_cast<uint32_t>(length);
  XLS_RET_CHECK_EQ(length, length_u32);
  stack.Push(InterpValue::MakeU32(length_u32));
  return absl::OkStatus();
}

absl::Status RunBuiltinRev(const Bytecode& bytecode, InterpreterStack& stack) {
  XLS_RET_CHECK(!stack.empty());
  XLS_ASSIGN_OR_RETURN(InterpValue value, stack.Pop());
  if (!value.IsBits() || value.IsSigned()) {
    return absl::InvalidArgumentError(
        "Argument to `rev` builtin must be an unsigned bits-typed value.");
  }
  XLS_ASSIGN_OR_RETURN(Bits bits, value.GetBits());
  stack.Push(
      InterpValue::MakeBits(/*is_signed=*/false, bits_ops::Reverse(bits)));
  return absl::OkStatus();
}

absl::Status RunBuiltinZip(const Bytecode& bytecode, InterpreterStack& stack) {
  return RunBinaryBuiltin(
      [](const InterpValue& lhs,
         const InterpValue& rhs) -> absl::StatusOr<InterpValue> {
        // Should have been checked by type inference.
        XLS_RET_CHECK(lhs.IsArray() && rhs.IsArray());

        XLS_ASSIGN_OR_RETURN(size_t lhs_length, lhs.GetLength());
        XLS_ASSIGN_OR_RETURN(size_t rhs_length, rhs.GetLength());

        // Should have been checked by type inference.
        XLS_RET_CHECK_EQ(rhs_length, rhs_length);

        std::vector<InterpValue> zipped;
        zipped.reserve(lhs_length);
        for (size_t i = 0; i < lhs_length; ++i) {
          zipped.push_back(InterpValue::MakeTuple(
              {lhs.GetValuesOrDie()[i], rhs.GetValuesOrDie()[i]}));
        }

        return InterpValue::MakeArray(std::move(zipped));
      },
      stack);
}

absl::Status RunBuiltinXorReduce(const Bytecode& bytecode,
                                 InterpreterStack& stack) {
  VLOG(3) << "Executing builtin XorReduce.";
  XLS_RET_CHECK(!stack.empty());
  XLS_ASSIGN_OR_RETURN(InterpValue value, stack.Pop());
  XLS_ASSIGN_OR_RETURN(Bits bits, value.GetBits());
  bits = bits_ops::XorReduce(bits);
  stack.Push(InterpValue::MakeBool(bits.IsOne()));
  return absl::OkStatus();
}

std::string HighlightLineByLineDifferences(std::string_view lhs,
                                           std::string_view rhs) {
  std::string result;
  result.reserve(rhs.size() + lhs.size());  // Just a guess.
  std::vector<std::string_view> rhs_split = absl::StrSplit(rhs, '\n');
  std::vector<std::string_view> lhs_split = absl::StrSplit(lhs, '\n');
  int64_t i = 0;
  for (; i < lhs_split.size() && i < rhs_split.size(); ++i) {
    if (lhs_split[i] != rhs_split[i]) {
      absl::StrAppend(&result, "< ", lhs_split[i],
                      "\n"
                      "> ",
                      rhs_split[i], "\n");
    } else {
      absl::StrAppend(&result, "  ", lhs_split[i], "\n");
    }
  }
  // I don't think trailers are possible right now -- but if they ever
  // show up, let's handle them in the obvious way.
  // (Only one of these two loops ever runs.)
  for (; i < lhs_split.size(); ++i) {
    absl::StrAppend(&result, "< ", lhs_split[i], "\n");
  }
  for (; i < rhs_split.size(); ++i) {
    absl::StrAppend(&result, "> ", rhs_split[i], "\n");
  }
  return result;
}

}  // namespace xls::dslx
