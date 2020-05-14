// Copyright 2020 Google LLC
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

#include "xls/ir/function_builder.h"

#include "absl/strings/str_format.h"
#include "xls/common/logging/logging.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/symbolized_stacktrace.h"
#include "xls/ir/package.h"

namespace xls {

using ::absl::StrFormat;

BValue BValue::operator>>(BValue rhs) { return builder()->Shrl(*this, rhs); }
BValue BValue::operator<<(BValue rhs) { return builder()->Shll(*this, rhs); }
BValue BValue::operator|(BValue rhs) { return builder()->Or(*this, rhs); }
BValue BValue::operator^(BValue rhs) { return builder()->Xor(*this, rhs); }
BValue BValue::operator*(BValue rhs) { return builder()->UMul(*this, rhs); }
BValue BValue::operator-(BValue rhs) { return builder()->Subtract(*this, rhs); }
BValue BValue::operator+(BValue rhs) { return builder()->Add(*this, rhs); }
BValue BValue::operator-() { return builder()->Negate(*this); }

BValue FunctionBuilder::SetError(std::string msg,
                                 absl::optional<SourceLocation> loc) {
  error_pending_ = true;
  error_msg_ = msg;
  if (loc.has_value()) {
    error_loc_.emplace(loc.value());
  }
  error_stacktrace_ = GetSymbolizedStackTraceAsString();
  return BValue();
}

BValue FunctionBuilder::Param(absl::string_view name, Type* type,
                              absl::optional<SourceLocation> loc) {
  if (ErrorPending()) {
    return BValue();
  }
  return AddNode<xls::Param>(loc, name, type);
}

BValue FunctionBuilder::Literal(Value value,
                                absl::optional<SourceLocation> loc) {
  if (ErrorPending()) {
    return BValue();
  }
  return AddNode<xls::Literal>(loc, value);
}

BValue FunctionBuilder::Negate(BValue x, absl::optional<SourceLocation> loc) {
  if (ErrorPending()) {
    return BValue();
  }
  return AddUnOp(Op::kNeg, x, loc);
}

BValue FunctionBuilder::Not(BValue x, absl::optional<SourceLocation> loc) {
  if (ErrorPending()) {
    return BValue();
  }

  return AddUnOp(Op::kNot, x, loc);
}

BValue FunctionBuilder::Select(BValue selector, absl::Span<const BValue> cases,
                               absl::optional<BValue> default_value,
                               absl::optional<SourceLocation> loc) {
  if (ErrorPending()) {
    return BValue();
  }
  std::vector<Node*> cases_nodes;
  for (const BValue& bvalue : cases) {
    XLS_CHECK_EQ(selector.builder(), bvalue.builder());
    cases_nodes.push_back(bvalue.node());
  }
  absl::optional<Node*> default_node = absl::nullopt;
  if (default_value.has_value()) {
    default_node = default_value->node();
  }
  return AddNode<xls::Select>(loc, selector.node(), cases_nodes, default_node);
}

BValue FunctionBuilder::Select(BValue selector, BValue on_true, BValue on_false,
                               absl::optional<SourceLocation> loc) {
  return Select(selector, {on_false, on_true}, /*default_value=*/absl::nullopt,
                loc);
}

BValue FunctionBuilder::OneHot(BValue input, LsbOrMsb priority,
                               absl::optional<SourceLocation> loc) {
  if (ErrorPending()) {
    return BValue();
  }
  return AddNode<xls::OneHot>(loc, input.node(), priority);
}

BValue FunctionBuilder::OneHotSelect(BValue selector,
                                     absl::Span<const BValue> cases,
                                     absl::optional<SourceLocation> loc) {
  if (ErrorPending()) {
    return BValue();
  }
  std::vector<Node*> cases_nodes;
  for (const BValue& bvalue : cases) {
    XLS_CHECK_EQ(selector.builder(), bvalue.builder());
    cases_nodes.push_back(bvalue.node());
  }
  return AddNode<xls::OneHotSelect>(loc, selector.node(), cases_nodes);
}

BValue FunctionBuilder::Clz(BValue x, absl::optional<SourceLocation> loc) {
  if (ErrorPending()) {
    return BValue();
  }
  if (!x.GetType()->IsBits()) {
    return SetError(
        StrFormat("Count-leading-zeros argument must be of Bits type; is: %s",
                  x.GetType()->ToString()),
        loc);
  }
  return ZeroExtend(
      Encode(OneHot(Reverse(x, loc), /*priority=*/LsbOrMsb::kLsb, loc)),
      x.BitCountOrDie());
}

BValue FunctionBuilder::Ctz(BValue x, absl::optional<SourceLocation> loc) {
  if (ErrorPending()) {
    return BValue();
  }
  if (!x.GetType()->IsBits()) {
    return SetError(
        StrFormat("Count-leading-zeros argument must be of Bits type; is: %s",
                  x.GetType()->ToString()),
        loc);
  }
  return ZeroExtend(Encode(OneHot(x, /*priority=*/LsbOrMsb::kLsb, loc)),
                    x.BitCountOrDie());
}

BValue FunctionBuilder::Match(BValue condition, absl::Span<const Case> cases,
                              BValue default_value,
                              absl::optional<SourceLocation> loc) {
  if (ErrorPending()) {
    return BValue();
  }
  std::vector<Case> boolean_cases;
  for (const Case& cas : cases) {
    boolean_cases.push_back(Case{Eq(condition, cas.clause, loc), cas.value});
  }
  return MatchTrue(boolean_cases, default_value, loc);
}

BValue FunctionBuilder::MatchTrue(absl::Span<const BValue> case_clauses,
                                  absl::Span<const BValue> case_values,
                                  BValue default_value,
                                  absl::optional<SourceLocation> loc) {
  if (case_clauses.size() != case_values.size()) {
    return SetError(
        StrFormat(
            "Number of case clauses %d does not equal number of values (%d)",
            case_clauses.size(), case_values.size()),
        loc);
  }
  std::vector<Case> cases;
  for (int64 i = 0; i < case_clauses.size(); ++i) {
    cases.push_back(Case{case_clauses[i], case_values[i]});
  }
  return MatchTrue(cases, default_value, loc);
}

BValue FunctionBuilder::MatchTrue(absl::Span<const Case> cases,
                                  BValue default_value,
                                  absl::optional<SourceLocation> loc) {
  if (ErrorPending()) {
    return BValue();
  }
  std::vector<BValue> selector_bits;
  std::vector<BValue> case_values;
  for (int64 i = 0; i < cases.size(); ++i) {
    XLS_CHECK_EQ(cases[i].clause.builder(), default_value.builder());
    XLS_CHECK_EQ(cases[i].value.builder(), default_value.builder());
    if (GetType(cases[i].clause) != package()->GetBitsType(1)) {
      return SetError(
          StrFormat("Selector %d must be a single-bit Bits type, is: %s", i,
                    GetType(cases[i].clause)->ToString()),
          loc);
    }
    selector_bits.push_back(cases[i].clause);
    case_values.push_back(cases[i].value);
  }
  case_values.push_back(default_value);

  // Reverse the order of the bits because bit index and indexing of concat
  // elements are reversed. That is, the zero-th operand of concat becomes the
  // most-significant part (highest index) of the result.
  std::reverse(selector_bits.begin(), selector_bits.end());

  BValue concat = Concat(selector_bits, loc);
  BValue one_hot = OneHot(concat, /*priority=*/LsbOrMsb::kLsb, loc);

  return OneHotSelect(one_hot, case_values, loc);
}

BValue FunctionBuilder::Tuple(absl::Span<const BValue> elements,
                              absl::optional<SourceLocation> loc) {
  if (ErrorPending()) {
    return BValue();
  }
  std::vector<Node*> nodes;
  nodes.reserve(elements.size());
  for (const BValue& value : elements) {
    nodes.push_back(value.node());
  }
  return AddNode<xls::Tuple>(loc, nodes);
}

BValue FunctionBuilder::Array(absl::Span<const BValue> elements,
                              Type* element_type,
                              absl::optional<SourceLocation> loc) {
  if (ErrorPending()) {
    return BValue();
  }
  std::vector<Node*> nodes;
  nodes.reserve(elements.size());
  for (const BValue& value : elements) {
    nodes.push_back(value.node());
    if (GetType(value) != element_type) {
      return SetError(
          StrFormat("Element type %s does not match expected type: %s",
                    GetType(value)->ToString(), element_type->ToString()),
          loc);
    }
  }

  return AddNode<xls::Array>(loc, nodes, element_type);
}

BValue FunctionBuilder::TupleIndex(BValue arg, int64 idx,
                                   absl::optional<SourceLocation> loc) {
  if (ErrorPending()) {
    return BValue();
  }
  return AddNode<xls::TupleIndex>(loc, arg.node(), idx);
}

BValue FunctionBuilder::CountedFor(BValue init_value, int64 trip_count,
                                   int64 stride, Function* body,
                                   absl::Span<const BValue> invariant_args,
                                   absl::optional<SourceLocation> loc) {
  if (ErrorPending()) {
    return BValue();
  }
  std::vector<Node*> invariant_arg_nodes;
  for (const BValue& arg : invariant_args) {
    invariant_arg_nodes.push_back(arg.node());
  }
  return AddNode<xls::CountedFor>(loc, init_value.node(), invariant_arg_nodes,
                                  trip_count, stride, body);
}

BValue FunctionBuilder::Map(BValue operand, Function* to_apply,
                            absl::optional<SourceLocation> loc) {
  if (ErrorPending()) {
    return BValue();
  }
  return AddNode<xls::Map>(loc, operand.node(), to_apply);
}

BValue FunctionBuilder::Invoke(absl::Span<const BValue> args,
                               Function* to_apply,
                               absl::optional<SourceLocation> loc) {
  if (ErrorPending()) {
    return BValue();
  }
  std::vector<Node*> arg_nodes;
  arg_nodes.reserve(args.size());
  for (const BValue& value : args) {
    arg_nodes.push_back(value.node());
  }
  return AddNode<xls::Invoke>(loc, arg_nodes, to_apply);
}

BValue FunctionBuilder::ArrayIndex(BValue arg, BValue idx,
                                   absl::optional<SourceLocation> loc) {
  if (ErrorPending()) {
    return BValue();
  }
  if (!arg.node()->GetType()->IsArray()) {
    return SetError(
        absl::StrFormat(
            "Cannot array-index the node %s because it has non-array type %s",
            arg.node()->ToString(), arg.node()->GetType()->ToString()),
        loc);
  }
  return AddNode<xls::ArrayIndex>(loc, arg.node(), idx.node());
}

BValue FunctionBuilder::Reverse(BValue arg,
                                absl::optional<SourceLocation> loc) {
  if (ErrorPending()) {
    return BValue();
  }
  return AddUnOp(Op::kReverse, arg, loc);
}

BValue FunctionBuilder::Identity(BValue var,
                                 absl::optional<SourceLocation> loc) {
  if (ErrorPending()) {
    return BValue();
  }
  return AddUnOp(Op::kIdentity, var, loc);
}

BValue FunctionBuilder::SignExtend(BValue arg, int64 new_bit_count,
                                   absl::optional<SourceLocation> loc) {
  if (ErrorPending()) {
    return BValue();
  }
  return AddNode<xls::ExtendOp>(loc, arg.node(), new_bit_count, Op::kSignExt);
}

BValue FunctionBuilder::ZeroExtend(BValue arg, int64 new_bit_count,
                                   absl::optional<SourceLocation> loc) {
  if (ErrorPending()) {
    return BValue();
  }
  return AddNode<xls::ExtendOp>(loc, arg.node(), new_bit_count, Op::kZeroExt);
}

BValue FunctionBuilder::BitSlice(BValue arg, int64 start, int64 width,
                                 absl::optional<SourceLocation> loc) {
  if (ErrorPending()) {
    return BValue();
  }
  return AddNode<xls::BitSlice>(loc, arg.node(), start, width);
}

BValue FunctionBuilder::Encode(BValue arg, absl::optional<SourceLocation> loc) {
  if (ErrorPending()) {
    return BValue();
  }
  return AddNode<xls::Encode>(loc, arg.node());
}

BValue FunctionBuilder::Decode(BValue arg, absl::optional<int64> width,
                               absl::optional<SourceLocation> loc) {
  if (ErrorPending()) {
    return BValue();
  }
  if (!arg.GetType()->IsBits()) {
    return SetError(StrFormat("Decode argument must be of Bits type; is: %s",
                              arg.GetType()->ToString()),
                    loc);
  }
  // The full output width ('width' not given) is an exponential function of the
  // argument width. Set a limit of 16 bits on the argument width.
  const int64 arg_width = arg.GetType()->AsBitsOrDie()->bit_count();
  if (!width.has_value() && arg_width > 16) {
    return SetError(
        StrFormat(
            "Decode argument width be no greater than 32-bits; is %d bits",
            arg_width),
        loc);
  }
  return AddNode<xls::Decode>(loc, arg.node(), /*width=*/width.has_value()
                                                   ? *width
                                                   : (1LL << arg_width));
}

BValue FunctionBuilder::Shra(BValue operand, BValue amount,
                             absl::optional<SourceLocation> loc) {
  if (ErrorPending()) {
    return BValue();
  }
  return AddBinOp(Op::kShra, operand, amount, loc);
}
BValue FunctionBuilder::Shrl(BValue operand, BValue amount,
                             absl::optional<SourceLocation> loc) {
  if (ErrorPending()) {
    return BValue();
  }
  return AddBinOp(Op::kShrl, operand, amount, loc);
}
BValue FunctionBuilder::Shll(BValue operand, BValue amount,
                             absl::optional<SourceLocation> loc) {
  if (ErrorPending()) {
    return BValue();
  }
  return AddBinOp(Op::kShll, operand, amount, loc);
}
BValue FunctionBuilder::Subtract(BValue lhs, BValue rhs,
                                 absl::optional<SourceLocation> loc) {
  if (ErrorPending()) {
    return BValue();
  }
  return AddBinOp(Op::kSub, lhs, rhs, loc);
}
BValue FunctionBuilder::Add(BValue lhs, BValue rhs,
                            absl::optional<SourceLocation> loc) {
  if (ErrorPending()) {
    return BValue();
  }
  return AddBinOp(Op::kAdd, lhs, rhs, loc);
}
BValue FunctionBuilder::Or(BValue lhs, BValue rhs,
                           absl::optional<SourceLocation> loc) {
  if (ErrorPending()) {
    return BValue();
  }
  return AddNaryOp(Op::kOr, std::vector<BValue>{lhs, rhs}, loc);
}
BValue FunctionBuilder::Or(absl::Span<const BValue> operands,
                           absl::optional<SourceLocation> loc) {
  if (ErrorPending()) {
    return BValue();
  }
  return AddNaryOp(Op::kOr, operands, loc);
}
BValue FunctionBuilder::Xor(BValue lhs, BValue rhs,
                            absl::optional<SourceLocation> loc) {
  if (ErrorPending()) {
    return BValue();
  }
  return AddNaryOp(Op::kXor, std::vector<BValue>{lhs, rhs}, loc);
}
BValue FunctionBuilder::And(BValue lhs, BValue rhs,
                            absl::optional<SourceLocation> loc) {
  if (ErrorPending()) {
    return BValue();
  }
  return AddNaryOp(Op::kAnd, std::vector<BValue>{lhs, rhs}, loc);
}

BValue FunctionBuilder::AndReduce(BValue operand,
                                  absl::optional<SourceLocation> loc) {
  if (ErrorPending()) {
    return BValue();
  }
  return AddBitwiseReductionOp(Op::kAndReduce, operand);
}

BValue FunctionBuilder::OrReduce(BValue operand,
                                 absl::optional<SourceLocation> loc) {
  if (ErrorPending()) {
    return BValue();
  }
  return AddBitwiseReductionOp(Op::kOrReduce, operand);
}

BValue FunctionBuilder::XorReduce(BValue operand,
                                  absl::optional<SourceLocation> loc) {
  if (ErrorPending()) {
    return BValue();
  }
  return AddBitwiseReductionOp(Op::kXorReduce, operand);
}

BValue FunctionBuilder::SMul(BValue lhs, BValue rhs,
                             absl::optional<SourceLocation> loc) {
  if (ErrorPending()) {
    return BValue();
  }
  return AddArithOp(Op::kSMul, lhs, rhs, /*result_width=*/absl::nullopt, loc);
}
BValue FunctionBuilder::UMul(BValue lhs, BValue rhs,
                             absl::optional<SourceLocation> loc) {
  if (ErrorPending()) {
    return BValue();
  }
  return AddArithOp(Op::kUMul, lhs, rhs, /*result_width=*/absl::nullopt, loc);
}
BValue FunctionBuilder::SMul(BValue lhs, BValue rhs, int64 result_width,
                             absl::optional<SourceLocation> loc) {
  if (ErrorPending()) {
    return BValue();
  }
  return AddArithOp(Op::kSMul, lhs, rhs, result_width, loc);
}
BValue FunctionBuilder::UMul(BValue lhs, BValue rhs, int64 result_width,
                             absl::optional<SourceLocation> loc) {
  if (ErrorPending()) {
    return BValue();
  }
  return AddArithOp(Op::kUMul, lhs, rhs, result_width, loc);
}
BValue FunctionBuilder::UDiv(BValue lhs, BValue rhs,
                             absl::optional<SourceLocation> loc) {
  if (ErrorPending()) {
    return BValue();
  }
  return AddBinOp(Op::kUDiv, lhs, rhs, loc);
}
BValue FunctionBuilder::Eq(BValue lhs, BValue rhs,
                           absl::optional<SourceLocation> loc) {
  if (ErrorPending()) {
    return BValue();
  }
  return AddCompareOp(Op::kEq, lhs, rhs, loc);
}
BValue FunctionBuilder::Ne(BValue lhs, BValue rhs,
                           absl::optional<SourceLocation> loc) {
  if (ErrorPending()) {
    return BValue();
  }
  return AddCompareOp(Op::kNe, lhs, rhs, loc);
}
BValue FunctionBuilder::UGe(BValue lhs, BValue rhs,
                            absl::optional<SourceLocation> loc) {
  if (ErrorPending()) {
    return BValue();
  }
  return AddCompareOp(Op::kUGe, lhs, rhs, loc);
}
BValue FunctionBuilder::UGt(BValue lhs, BValue rhs,
                            absl::optional<SourceLocation> loc) {
  if (ErrorPending()) {
    return BValue();
  }
  return AddCompareOp(Op::kUGt, lhs, rhs, loc);
}
BValue FunctionBuilder::ULe(BValue lhs, BValue rhs,
                            absl::optional<SourceLocation> loc) {
  if (ErrorPending()) {
    return BValue();
  }
  return AddCompareOp(Op::kULe, lhs, rhs, loc);
}
BValue FunctionBuilder::ULt(BValue lhs, BValue rhs,
                            absl::optional<SourceLocation> loc) {
  if (ErrorPending()) {
    return BValue();
  }
  return AddCompareOp(Op::kULt, lhs, rhs, loc);
}
BValue FunctionBuilder::SLt(BValue lhs, BValue rhs,
                            absl::optional<SourceLocation> loc) {
  if (ErrorPending()) {
    return BValue();
  }
  return AddCompareOp(Op::kSLt, lhs, rhs, loc);
}
BValue FunctionBuilder::SLe(BValue lhs, BValue rhs,
                            absl::optional<SourceLocation> loc) {
  if (ErrorPending()) {
    return BValue();
  }
  return AddCompareOp(Op::kSLe, lhs, rhs, loc);
}
BValue FunctionBuilder::SGe(BValue lhs, BValue rhs,
                            absl::optional<SourceLocation> loc) {
  if (ErrorPending()) {
    return BValue();
  }
  return AddCompareOp(Op::kSGe, lhs, rhs, loc);
}
BValue FunctionBuilder::SGt(BValue lhs, BValue rhs,
                            absl::optional<SourceLocation> loc) {
  if (ErrorPending()) {
    return BValue();
  }
  return AddCompareOp(Op::kSGt, lhs, rhs, loc);
}

BValue FunctionBuilder::Concat(absl::Span<const BValue> operands,
                               absl::optional<SourceLocation> loc) {
  if (ErrorPending()) {
    return BValue();
  }
  std::vector<Node*> node_operands;
  for (const BValue operand : operands) {
    node_operands.push_back(operand.node());
    if (!operand.node()->GetType()->IsBits()) {
      return SetError(
          absl::StrFormat(
              "Cannot concatenate node %s because it has non-bits type %s",
              operand.node()->ToString(),
              operand.node()->GetType()->ToString()),
          loc);
    }
  }
  return AddNode<xls::Concat>(loc, node_operands);
}

xabsl::StatusOr<Function*> FunctionBuilder::Build() {
  if (function_ == nullptr) {
    return absl::FailedPreconditionError(
        "Cannot build function multiple times");
  }
  if (function_->node_count() == 0) {
    return absl::InvalidArgumentError("Function cannot be empty");
  }
  XLS_RET_CHECK(last_node_ != nullptr);
  return BuildWithReturnValue(BValue(last_node_, this));
}

xabsl::StatusOr<Function*> FunctionBuilder::BuildWithReturnValue(
    BValue return_value) {
  if (ErrorPending()) {
    std::string msg = error_msg_ + " ";
    if (error_loc_.has_value()) {
      absl::StrAppendFormat(
          &msg, "File: %d, Line: %d, Col: %d", error_loc_->fileno().value(),
          error_loc_->lineno().value(), error_loc_->colno().value());
    }
    absl::StrAppend(&msg, "\nStack Trace:\n" + error_stacktrace_);
    return absl::InvalidArgumentError("Could not build IR: " + msg);
  }
  XLS_RET_CHECK_EQ(return_value.builder(), this);
  function_->set_return_value(return_value.node());
  return package()->AddFunction(std::move(function_));
}

BValue FunctionBuilder::AddArithOp(Op op, BValue lhs, BValue rhs,
                                   absl::optional<int64> result_width,
                                   absl::optional<SourceLocation> loc) {
  XLS_CHECK_EQ(lhs.builder(), rhs.builder());
  if (ErrorPending()) {
    return BValue();
  }
  if (!lhs.GetType()->IsBits() || !rhs.GetType()->IsBits()) {
    return SetError(
        StrFormat("Arithmetic arguments must be of Bits type; is: %s and %s",
                  lhs.GetType()->ToString(), rhs.GetType()->ToString()),
        loc);
  }
  int64 width;
  if (result_width.has_value()) {
    width = *result_width;
  } else {
    if (lhs.BitCountOrDie() != rhs.BitCountOrDie()) {
      return SetError(
          StrFormat("Arguments of arithmetic operation must be same width if "
                    "result width is not specified; is: %s and %s",
                    lhs.GetType()->ToString(), rhs.GetType()->ToString()),
          loc);
    }
    width = lhs.BitCountOrDie();
  }
  return AddNode<ArithOp>(loc, lhs.node(), rhs.node(), width, op);
}

BValue FunctionBuilder::AddUnOp(Op op, BValue x,
                                absl::optional<SourceLocation> loc) {
  XLS_CHECK_EQ(this, x.builder());
  if (ErrorPending()) {
    return BValue();
  }
  return AddNode<UnOp>(loc, x.node(), op);
}

BValue FunctionBuilder::AddBinOp(Op op, BValue lhs, BValue rhs,
                                 absl::optional<SourceLocation> loc) {
  XLS_CHECK_EQ(lhs.builder(), rhs.builder());
  if (ErrorPending()) {
    return BValue();
  }
  return AddNode<BinOp>(loc, lhs.node(), rhs.node(), op);
}

BValue FunctionBuilder::AddCompareOp(Op op, BValue lhs, BValue rhs,
                                     absl::optional<SourceLocation> loc) {
  XLS_CHECK_EQ(lhs.builder(), rhs.builder());
  if (ErrorPending()) {
    return BValue();
  }
  return AddNode<CompareOp>(loc, lhs.node(), rhs.node(), op);
}

BValue FunctionBuilder::AddNaryOp(Op op, absl::Span<const BValue> args,
                                  absl::optional<SourceLocation> loc) {
  if (ErrorPending()) {
    return BValue();
  }
  std::vector<Node*> nodes;
  for (const BValue& bvalue : args) {
    nodes.push_back(bvalue.node());
  }
  return AddNode<NaryOp>(loc, nodes, op);
}

BValue FunctionBuilder::AddBitwiseReductionOp(
    Op op, BValue arg, absl::optional<SourceLocation> loc) {
  XLS_CHECK_EQ(this, arg.builder());
  if (ErrorPending()) {
    return BValue();
  }
  return AddNode<BitwiseReductionOp>(loc, arg.node(), op);
}

}  // namespace xls
