// Copyright 2020 The XLS Authors
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

#include "xls/dslx/builtins.h"

#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/bits_ops.h"

namespace xls::dslx {
namespace {

// Simple fluent object for checking args up front and creating error statuses
// on precondition violations.
class ArgChecker {
 public:
  ArgChecker(std::string name, absl::Span<const InterpValue> args)
      : name_(std::move(name)), args_(args) {}

  ArgChecker& size(int64 target) {
    if (args_.size() != target) {
      status_.Update(absl::InvalidArgumentError(
          absl::StrFormat("Expect %d argument(s) to %s(); got %d", target,
                          name_, args_.size())));
    }
    return *this;
  }

  ArgChecker& size_ge(int64 target) {
    if (args_.size() < target) {
      status_.Update(absl::InvalidArgumentError(
          absl::StrFormat("Expect >= %d argument(s) to %s(); got %d", target,
                          name_, args_.size())));
    }
    return *this;
  }

  ArgChecker& array(int64 argno) {
    if (!args_[argno].IsArray()) {
      status_.Update(absl::InvalidArgumentError(
          absl::StrFormat("Expect argument %d to %s to be an array; got: %s",
                          argno, name_, TagToString(args_[argno].tag()))));
    }
    return *this;
  }

  ArgChecker& bits(int64 argno) {
    if (!args_[argno].IsBits()) {
      status_.Update(absl::InvalidArgumentError(
          absl::StrFormat("Expect argument %d to %s to be bits; got: %s", argno,
                          name_, TagToString(args_[argno].tag()))));
    }
    return *this;
  }

  const absl::Status& status() const { return status_; }

 private:
  std::string name_;
  absl::Span<const InterpValue> args_;
  absl::Status status_;
};

// Helper that creates a stylized error status that represents a FailureError --
// when it propagates to the pybind11 boundary it should be thrown as an
// exception.
absl::Status FailureError(const Span& span, absl::string_view message) {
  return absl::InternalError(absl::StrFormat(
      "FailureError: %s The program being interpreted failed! %s",
      span.ToString(), message));
}

// Helper that returns a failure error if pred (predicated) is false.
absl::StatusOr<InterpValue> FailUnless(const InterpValue& pred,
                                       std::string message, const Span& span) {
  if (pred.IsFalse()) {
    return FailureError(span, message);
  }
  return InterpValue::MakeNil();
}

// Helper that finds the first differing index among value spans.
//
// Precondition: lhs and rhs must be same size.
absl::optional<int64> FindFirstDifferingIndex(
    absl::Span<const InterpValue> lhs, absl::Span<const InterpValue> rhs) {
  XLS_CHECK_EQ(lhs.size(), rhs.size());
  for (int64 i = 0; i < lhs.size(); ++i) {
    if (lhs[i].Ne(rhs[i])) {
      return i;
    }
  }
  return absl::nullopt;
}

}  // namespace

std::string SignedCmpToString(SignedCmp cmp) {
  switch (cmp) {
    case SignedCmp::kLt:
      return "slt";
    case SignedCmp::kLe:
      return "sle";
    case SignedCmp::kGt:
      return "sgt";
    case SignedCmp::kGe:
      return "sge";
  }
  return absl::StrFormat("<invalid SignedCmp(%d)>", static_cast<int64>(cmp));
}

absl::StatusOr<InterpValue> BuiltinScmp(SignedCmp cmp,
                                        absl::Span<const InterpValue> args,
                                        const Span& span, Invocation* expr,
                                        SymbolicBindings* symbolic_bindings) {
  XLS_RETURN_IF_ERROR(ArgChecker(SignedCmpToString(cmp), args)
                          .size(2)
                          .bits(0)
                          .bits(1)
                          .status());
  const Bits& lhs = args[0].GetBitsOrDie();
  const Bits& rhs = args[1].GetBitsOrDie();
  bool result = false;
  switch (cmp) {
    case SignedCmp::kLt:
      result = bits_ops::SLessThan(lhs, rhs);
      break;
    case SignedCmp::kLe:
      result = bits_ops::SLessThanOrEqual(lhs, rhs);
      break;
    case SignedCmp::kGt:
      result = bits_ops::SGreaterThan(lhs, rhs);
      break;
    case SignedCmp::kGe:
      result = bits_ops::SGreaterThanOrEqual(lhs, rhs);
      break;
  }
  return InterpValue::MakeBool(result);
}

absl::StatusOr<InterpValue> BuiltinFail(absl::Span<const InterpValue> args,
                                        const Span& span, Invocation* expr,
                                        SymbolicBindings* symbolic_bindings) {
  XLS_RETURN_IF_ERROR(ArgChecker("fail!", args).size(1).status());
  return FailureError(span, args[0].ToString());
}

absl::StatusOr<InterpValue> BuiltinAssertEq(
    absl::Span<const InterpValue> args, const Span& span, Invocation* expr,
    SymbolicBindings* symbolic_bindings) {
  XLS_RETURN_IF_ERROR(ArgChecker("assert_eq", args).size(2).status());
  const InterpValue& lhs = args[0];
  const InterpValue& rhs = args[1];
  if (lhs.Eq(rhs)) {
    return InterpValue::MakeNil();
  }

  std::string message =
      absl::StrFormat("\n  lhs: %s\n  rhs: %s\n  were not equal",
                      lhs.ToHumanString(), rhs.ToHumanString());

  if (lhs.IsArray() && rhs.IsArray()) {
    absl::optional<int64> i =
        FindFirstDifferingIndex(lhs.GetValuesOrDie(), rhs.GetValuesOrDie());
    XLS_RET_CHECK(i.has_value());
    const auto& lhs_values = lhs.GetValuesOrDie();
    const auto& rhs_values = rhs.GetValuesOrDie();
    message += absl::StrFormat("; first differing index: %d :: %s vs %s", *i,
                               lhs_values[*i].ToHumanString(),
                               rhs_values[*i].ToHumanString());
  }

  return FailureError(span, message);
}

absl::StatusOr<InterpValue> BuiltinAssertLt(
    absl::Span<const InterpValue> args, const Span& span, Invocation* expr,
    SymbolicBindings* symbolic_bindings) {
  XLS_RETURN_IF_ERROR(ArgChecker("assert_lt", args).size(2).status());
  const InterpValue& lhs = args[0];
  const InterpValue& rhs = args[1];
  XLS_ASSIGN_OR_RETURN(InterpValue pred, lhs.Lt(rhs));

  std::string message = absl::StrFormat(
      "\n  want: %s < %s", lhs.ToHumanString(), rhs.ToHumanString());
  return FailUnless(pred, message, span);
}

absl::StatusOr<InterpValue> BuiltinAndReduce(
    absl::Span<const InterpValue> args, const Span& span, Invocation* expr,
    SymbolicBindings* symbolic_bindings) {
  XLS_RETURN_IF_ERROR(ArgChecker("and_reduce", args).size(1).status());
  XLS_ASSIGN_OR_RETURN(Bits bits, args[0].GetBits());
  return InterpValue::MakeBits(InterpValueTag::kUBits,
                               bits_ops::AndReduce(bits));
}

absl::StatusOr<InterpValue> BuiltinOrReduce(
    absl::Span<const InterpValue> args, const Span& span, Invocation* expr,
    SymbolicBindings* symbolic_bindings) {
  XLS_RETURN_IF_ERROR(ArgChecker("or_reduce", args).size(1).status());
  XLS_ASSIGN_OR_RETURN(Bits bits, args[0].GetBits());
  return InterpValue::MakeBits(InterpValueTag::kUBits,
                               bits_ops::OrReduce(bits));
}

absl::StatusOr<InterpValue> BuiltinXorReduce(
    absl::Span<const InterpValue> args, const Span& span, Invocation* expr,
    SymbolicBindings* symbolic_bindings) {
  XLS_RETURN_IF_ERROR(ArgChecker("xor_reduce", args).size(1).status());
  XLS_ASSIGN_OR_RETURN(Bits bits, args[0].GetBits());
  return InterpValue::MakeBits(InterpValueTag::kUBits,
                               bits_ops::XorReduce(bits));
}

absl::StatusOr<InterpValue> BuiltinRev(absl::Span<const InterpValue> args,
                                       const Span& span, Invocation* expr,
                                       SymbolicBindings* symbolic_bindings) {
  XLS_RETURN_IF_ERROR(ArgChecker("rev", args).size(1).status());
  XLS_ASSIGN_OR_RETURN(Bits bits, args[0].GetBits());
  return InterpValue::MakeBits(InterpValueTag::kUBits, bits_ops::Reverse(bits));
}

absl::StatusOr<InterpValue> BuiltinEnumerate(
    absl::Span<const InterpValue> args, const Span& span, Invocation* expr,
    SymbolicBindings* symbolic_bindings) {
  XLS_RETURN_IF_ERROR(ArgChecker("enumerate", args).size(1).array(0).status());
  auto& values = args[0].GetValuesOrDie();
  std::vector<InterpValue> results;
  for (int64 i = 0; i < values.size(); ++i) {
    auto ordinal = InterpValue::MakeUBits(/*bit_count=*/32, /*value=*/i);
    auto tuple = InterpValue::MakeTuple({ordinal, values[i]});
    results.push_back(tuple);
  }
  return InterpValue::MakeArray(results);
}

absl::StatusOr<InterpValue> BuiltinRange(absl::Span<const InterpValue> args,
                                         const Span& span, Invocation* expr,
                                         SymbolicBindings* symbolic_bindings) {
  XLS_RETURN_IF_ERROR(ArgChecker("range", args).size_ge(1).bits(0).status());
  absl::optional<InterpValue> start;
  absl::optional<InterpValue> limit;
  if (args.size() == 1) {
    limit = args[0];
    start = InterpValue::MakeUBits(/*bit_count=*/limit->GetBitCount().value(),
                                   /*value=*/0);
  } else if (args.size() == 2) {
    start = args[0];
    limit = args[1];
  } else {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Expected 1 or 2 arguments to range; got: %d", args.size()));
  }

  std::vector<InterpValue> elements;
  while (start->Lt(*limit).value().IsTrue()) {
    elements.push_back(*start);
    start = start
                ->Add(InterpValue::MakeUBits(
                    /*bit_count=*/limit->GetBitCount().value(), /*value=*/1))
                .value();
  }
  return InterpValue::MakeArray(std::move(elements));
}

absl::StatusOr<InterpValue> BuiltinBitSlice(
    absl::Span<const InterpValue> args, const Span& span, Invocation* expr,
    SymbolicBindings* symbolic_bindings) {
  XLS_RETURN_IF_ERROR(ArgChecker("bit_slice", args).size(3).status());
  const InterpValue& subject = args[0];
  const InterpValue& start = args[1];
  const InterpValue& width = args[2];
  XLS_ASSIGN_OR_RETURN(Bits subject_bits, subject.GetBits());
  XLS_ASSIGN_OR_RETURN(Bits start_bits, start.GetBits());
  XLS_ASSIGN_OR_RETURN(uint64 start_index, start_bits.ToUint64());
  if (start_index >= subject_bits.bit_count()) {
    start_index = subject_bits.bit_count();
  }
  // Note: output size has to be derived from the types of the input argument,
  // so the bitwidth of the "width" argument is what determines the output size,
  // not the value. This is the "forcing a known-const to be available via the
  // type system" kind of trick before we can have explicit parametrics for
  // builtins.
  XLS_ASSIGN_OR_RETURN(int64 bit_count, width.GetBitCount());
  return InterpValue::MakeBits(InterpValueTag::kUBits,
                               subject_bits.Slice(start_index, bit_count));
}

absl::StatusOr<InterpValue> BuiltinSlice(absl::Span<const InterpValue> args,
                                         const Span& span, Invocation* expr,
                                         SymbolicBindings* symbolic_bindings) {
  XLS_RETURN_IF_ERROR(ArgChecker("slice", args).size(3).status());
  const InterpValue& array = args[0];
  const InterpValue& start = args[1];
  const InterpValue& length = args[2];
  return array.Slice(start, length);
}

absl::StatusOr<InterpValue> BuiltinAddWithCarry(
    absl::Span<const InterpValue> args, const Span& span, Invocation* expr,
    SymbolicBindings* symbolic_bindings) {
  XLS_RETURN_IF_ERROR(ArgChecker("add_with_carry", args).size(2).status());
  const InterpValue& lhs = args[0];
  const InterpValue& rhs = args[1];
  return lhs.AddWithCarry(rhs);
}

absl::StatusOr<InterpValue> BuiltinClz(absl::Span<const InterpValue> args,
                                       const Span& span, Invocation* expr,
                                       SymbolicBindings* symbolic_bindings) {
  XLS_RETURN_IF_ERROR(ArgChecker("clz", args).size(1).status());
  const InterpValue& arg = args[0];
  XLS_ASSIGN_OR_RETURN(Bits bits, arg.GetBits());
  return InterpValue::MakeUBits(/*bit_count=*/bits.bit_count(),
                                /*value=*/bits.CountLeadingZeros());
}

absl::StatusOr<InterpValue> BuiltinCtz(absl::Span<const InterpValue> args,
                                       const Span& span, Invocation* expr,
                                       SymbolicBindings* symbolic_bindings) {
  XLS_RETURN_IF_ERROR(ArgChecker("ctz", args).size(1).status());
  const InterpValue& arg = args[0];
  XLS_ASSIGN_OR_RETURN(Bits bits, arg.GetBits());
  return InterpValue::MakeUBits(/*bit_count=*/bits.bit_count(),
                                /*value=*/bits.CountTrailingZeros());
}

absl::StatusOr<InterpValue> BuiltinOneHot(absl::Span<const InterpValue> args,
                                          const Span& span, Invocation* expr,
                                          SymbolicBindings* symbolic_bindings) {
  XLS_RETURN_IF_ERROR(ArgChecker("one_hot", args).size(2).status());
  const InterpValue& arg = args[0];
  const InterpValue& lsb_prio = args[1];
  XLS_ASSIGN_OR_RETURN(Bits bits, arg.GetBits());
  XLS_ASSIGN_OR_RETURN(Bits lsb_prio_bits, lsb_prio.GetBits());
  auto f = lsb_prio_bits.IsAllOnes() ? bits_ops::OneHotLsbToMsb
                                     : bits_ops::OneHotMsbToLsb;
  Bits result = f(bits);
  return InterpValue::MakeBits(InterpValueTag::kUBits, result);
}

absl::StatusOr<InterpValue> BuiltinOneHotSel(
    absl::Span<const InterpValue> args, const Span& span, Invocation* expr,
    SymbolicBindings* symbolic_bindings) {
  XLS_RETURN_IF_ERROR(ArgChecker("one_hot_sel", args).size(2).status());
  const InterpValue& selector = args[0];
  const InterpValue& cases = args[1];
  XLS_ASSIGN_OR_RETURN(Bits selector_bits, selector.GetBits());
  XLS_ASSIGN_OR_RETURN(const std::vector<InterpValue>* values,
                       cases.GetValues());
  if (values->empty()) {
    return absl::InvalidArgumentError(
        "At least one value to select is required.");
  }
  XLS_ASSIGN_OR_RETURN(int64 result_bit_count, (*values)[0].GetBitCount());
  Bits accum(result_bit_count);
  for (int64 i = 0; i < values->size(); ++i) {
    if (!selector_bits.Get(i)) {
      continue;
    }
    XLS_ASSIGN_OR_RETURN(Bits i_bits, (*values)[i].GetBits());
    accum = bits_ops::Or(i_bits, accum);
  }
  return InterpValue::MakeBits((*values)[0].tag(), accum);
}

absl::StatusOr<InterpValue> BuiltinSignex(absl::Span<const InterpValue> args,
                                          const Span& span, Invocation* expr,
                                          SymbolicBindings* symbolic_bindings) {
  XLS_RETURN_IF_ERROR(ArgChecker("signex", args).size(2).status());
  const InterpValue& lhs = args[0];
  const InterpValue& rhs = args[1];
  XLS_ASSIGN_OR_RETURN(int64 new_bit_count, rhs.GetBitCount());
  XLS_ASSIGN_OR_RETURN(Bits lhs_bits, lhs.GetBits());
  return InterpValue::MakeBits(rhs.tag(),
                               bits_ops::SignExtend(lhs_bits, new_bit_count));
}

}  // namespace xls::dslx
