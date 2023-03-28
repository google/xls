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

#include "xls/ir/function_builder.h"

#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "xls/common/logging/logging.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/symbolized_stacktrace.h"
#include "xls/ir/channel.h"
#include "xls/ir/function.h"
#include "xls/ir/function_base.h"
#include "xls/ir/nodes.h"
#include "xls/ir/package.h"
#include "xls/ir/proc.h"
#include "xls/ir/verifier.h"

namespace xls {

Type* BValue::GetType() const { return node()->GetType(); }

int64_t BValue::BitCountOrDie() const { return node()->BitCountOrDie(); }

const SourceInfo& BValue::loc() const { return node_->loc(); }

std::string BValue::ToString() const {
  return node_ == nullptr ? std::string("<null BValue>") : node_->ToString();
}

std::string BValue::SetName(std::string_view name) {
  if (node_ != nullptr) {
    node_->SetName(name);
  }
  return "";
}

std::string BValue::GetName() const {
  if (node_ != nullptr) {
    return node_->GetName();
  }
  return "";
}

bool BValue::HasAssignedName() const {
  if (node_ != nullptr) {
    return node_->HasAssignedName();
  }
  return false;
}

BValue BValue::operator>>(BValue rhs) { return builder()->Shrl(*this, rhs); }
BValue BValue::operator<<(BValue rhs) { return builder()->Shll(*this, rhs); }
BValue BValue::operator|(BValue rhs) { return builder()->Or(*this, rhs); }
BValue BValue::operator^(BValue rhs) { return builder()->Xor(*this, rhs); }
BValue BValue::operator*(BValue rhs) { return builder()->UMul(*this, rhs); }
BValue BValue::operator-(BValue rhs) { return builder()->Subtract(*this, rhs); }
BValue BValue::operator+(BValue rhs) { return builder()->Add(*this, rhs); }
BValue BValue::operator-() { return builder()->Negate(*this); }

BValue BuilderBase::CreateBValue(Node* node, const SourceInfo& loc) {
  last_node_ = node;
  if (should_verify_) {
    absl::Status verify_status = VerifyNode(last_node_);
    if (!verify_status.ok()) {
      return SetError(verify_status.message(), loc);
    }
  }
  return BValue(last_node_, this);
}

template <typename NodeT, typename... Args>
BValue BuilderBase::AddNode(const SourceInfo& loc, Args&&... args) {
  last_node_ = function_->AddNode<NodeT>(std::make_unique<NodeT>(
      loc, std::forward<Args>(args)..., function_.get()));
  return CreateBValue(last_node_, loc);
}

const std::string& BuilderBase::name() const { return function_->name(); }

absl::Status BuilderBase::SetAsTop() {
  if (package() == nullptr) {
    return absl::InternalError(
        absl::StrFormat("Package is not set for builder base: %s.", name()));
  }
  return package()->SetTop(function_.get());
}

BValue BuilderBase::SetError(std::string_view msg, const SourceInfo& loc) {
  XLS_VLOG(3) << absl::StreamFormat("BuilderBase::SetError; msg: %s; loc: %s",
                                    msg, loc.ToString());
  error_pending_ = true;
  error_msg_ = std::string(msg);
  error_loc_ = loc;
  error_stacktrace_ = GetSymbolizedStackTraceAsString();
  return BValue();
}

BValue BuilderBase::Literal(Value value, const SourceInfo& loc,
                            std::string_view name) {
  if (ErrorPending()) {
    return BValue();
  }
  return AddNode<xls::Literal>(loc, value, name);
}

BValue BuilderBase::Negate(BValue x, const SourceInfo& loc,
                           std::string_view name) {
  if (ErrorPending()) {
    return BValue();
  }
  return AddUnOp(Op::kNeg, x, loc, name);
}

BValue BuilderBase::Not(BValue x, const SourceInfo& loc,
                        std::string_view name) {
  if (ErrorPending()) {
    return BValue();
  }

  return AddUnOp(Op::kNot, x, loc, name);
}

BValue BuilderBase::Select(BValue selector, absl::Span<const BValue> cases,
                           std::optional<BValue> default_value,
                           const SourceInfo& loc, std::string_view name) {
  if (ErrorPending()) {
    return BValue();
  }
  std::vector<Node*> cases_nodes;
  for (const BValue& bvalue : cases) {
    XLS_CHECK_EQ(selector.builder(), bvalue.builder());
    cases_nodes.push_back(bvalue.node());
  }
  std::optional<Node*> default_node = std::nullopt;
  if (default_value.has_value()) {
    default_node = default_value->node();
  }
  return AddNode<xls::Select>(loc, selector.node(), cases_nodes, default_node,
                              name);
}

BValue BuilderBase::Select(BValue selector, BValue on_true, BValue on_false,
                           const SourceInfo& loc, std::string_view name) {
  return Select(selector, {on_false, on_true}, /*default_value=*/std::nullopt,
                loc, name);
}

BValue BuilderBase::OneHot(BValue input, LsbOrMsb priority,
                           const SourceInfo& loc, std::string_view name) {
  if (ErrorPending()) {
    return BValue();
  }
  return AddNode<xls::OneHot>(loc, input.node(), priority, name);
}

BValue BuilderBase::OneHotSelect(BValue selector,
                                 absl::Span<const BValue> cases,
                                 const SourceInfo& loc,
                                 std::string_view name) {
  if (ErrorPending()) {
    return BValue();
  }
  std::vector<Node*> cases_nodes;
  for (const BValue& bvalue : cases) {
    XLS_CHECK_EQ(selector.builder(), bvalue.builder());
    cases_nodes.push_back(bvalue.node());
  }
  return AddNode<xls::OneHotSelect>(loc, selector.node(), cases_nodes, name);
}

BValue BuilderBase::PrioritySelect(BValue selector,
                                   absl::Span<const BValue> cases,
                                   const SourceInfo& loc,
                                   std::string_view name) {
  if (ErrorPending()) {
    return BValue();
  }
  std::vector<Node*> cases_nodes;
  for (const BValue& bvalue : cases) {
    XLS_CHECK_EQ(selector.builder(), bvalue.builder());
    cases_nodes.push_back(bvalue.node());
  }
  return AddNode<xls::PrioritySelect>(loc, selector.node(), cases_nodes, name);
}

BValue BuilderBase::Clz(BValue x, const SourceInfo& loc,
                        std::string_view name) {
  if (ErrorPending()) {
    return BValue();
  }
  if (!x.GetType()->IsBits()) {
    return SetError(
        absl::StrFormat(
            "Count-leading-zeros argument must be of Bits type; is: %s",
            x.GetType()->ToString()),
        loc);
  }
  return ZeroExtend(
      Encode(OneHot(Reverse(x, loc), /*priority=*/LsbOrMsb::kLsb, loc), loc),
      x.BitCountOrDie(), loc, name);
}

BValue BuilderBase::Ctz(BValue x, const SourceInfo& loc,
                        std::string_view name) {
  if (ErrorPending()) {
    return BValue();
  }
  if (!x.GetType()->IsBits()) {
    return SetError(
        absl::StrFormat(
            "Count-leading-zeros argument must be of Bits type; is: %s",
            x.GetType()->ToString()),
        loc);
  }
  return ZeroExtend(Encode(OneHot(x, /*priority=*/LsbOrMsb::kLsb, loc)),
                    x.BitCountOrDie(), loc, name);
}

BValue BuilderBase::Match(BValue condition, absl::Span<const Case> cases,
                          BValue default_value, const SourceInfo& loc,
                          std::string_view name) {
  if (ErrorPending()) {
    return BValue();
  }
  std::vector<Case> boolean_cases;
  for (const Case& cas : cases) {
    boolean_cases.push_back(Case{Eq(condition, cas.clause, loc), cas.value});
  }
  return MatchTrue(boolean_cases, default_value, loc, name);
}

BValue BuilderBase::MatchTrue(absl::Span<const BValue> case_clauses,
                              absl::Span<const BValue> case_values,
                              BValue default_value, const SourceInfo& loc,
                              std::string_view name) {
  if (case_clauses.size() != case_values.size()) {
    return SetError(
        absl::StrFormat(
            "Number of case clauses %d does not equal number of values (%d)",
            case_clauses.size(), case_values.size()),
        loc);
  }
  std::vector<Case> cases;
  for (int64_t i = 0; i < case_clauses.size(); ++i) {
    cases.push_back(Case{case_clauses[i], case_values[i]});
  }
  return MatchTrue(cases, default_value, loc, name);
}

BValue BuilderBase::MatchTrue(absl::Span<const Case> cases,
                              BValue default_value, const SourceInfo& loc,
                              std::string_view name) {
  if (ErrorPending()) {
    return BValue();
  }
  std::vector<BValue> selector_bits;
  std::vector<BValue> case_values;
  for (int64_t i = 0; i < cases.size(); ++i) {
    XLS_CHECK_EQ(cases[i].clause.builder(), default_value.builder());
    XLS_CHECK_EQ(cases[i].value.builder(), default_value.builder());
    if (GetType(cases[i].clause) != package()->GetBitsType(1)) {
      return SetError(
          absl::StrFormat("Selector %d must be a single-bit Bits type, is: %s",
                          i, GetType(cases[i].clause)->ToString()),
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

  return OneHotSelect(one_hot, case_values, loc, name);
}

BValue BuilderBase::AfterAll(absl::Span<const BValue> dependencies,
                             const SourceInfo& loc, std::string_view name) {
  if (ErrorPending()) {
    return BValue();
  }
  std::vector<Node*> nodes;
  nodes.reserve(dependencies.size());
  for (const BValue& value : dependencies) {
    nodes.push_back(value.node());
    if (!GetType(value)->IsToken()) {
      return SetError(absl::StrFormat("Dependency type %s is not a token.",
                                      GetType(value)->ToString()),
                      loc);
    }
  }
  return AddNode<xls::AfterAll>(loc, nodes, name);
}

BValue BuilderBase::Tuple(absl::Span<const BValue> elements,
                          const SourceInfo& loc, std::string_view name) {
  if (ErrorPending()) {
    return BValue();
  }
  std::vector<Node*> nodes;
  nodes.reserve(elements.size());
  for (const BValue& value : elements) {
    nodes.push_back(value.node());
  }
  return AddNode<xls::Tuple>(loc, nodes, name);
}

BValue BuilderBase::Array(absl::Span<const BValue> elements, Type* element_type,
                          const SourceInfo& loc, std::string_view name) {
  if (ErrorPending()) {
    return BValue();
  }
  std::vector<Node*> nodes;
  nodes.reserve(elements.size());
  for (const BValue& value : elements) {
    nodes.push_back(value.node());
    if (GetType(value) != element_type) {
      return SetError(
          absl::StrFormat("Element type %s does not match expected type: %s",
                          GetType(value)->ToString(), element_type->ToString()),
          loc);
    }
  }

  return AddNode<xls::Array>(loc, nodes, element_type, name);
}

BValue BuilderBase::TupleIndex(BValue arg, int64_t idx, const SourceInfo& loc,
                               std::string_view name) {
  if (ErrorPending()) {
    return BValue();
  }
  if (!GetType(arg)->IsTuple()) {
    return SetError(
        absl::StrFormat(
            "Operand of tuple-index must be tuple-typed, is type: %s",
            GetType(arg)->ToString()),
        loc);
  }
  return AddNode<xls::TupleIndex>(loc, arg.node(), idx, name);
}

BValue BuilderBase::CountedFor(BValue init_value, int64_t trip_count,
                               int64_t stride, Function* body,
                               absl::Span<const BValue> invariant_args,
                               const SourceInfo& loc, std::string_view name) {
  if (ErrorPending()) {
    return BValue();
  }
  std::vector<Node*> invariant_arg_nodes;
  for (const BValue& arg : invariant_args) {
    invariant_arg_nodes.push_back(arg.node());
  }
  return AddNode<xls::CountedFor>(loc, init_value.node(), invariant_arg_nodes,
                                  trip_count, stride, body, name);
}

BValue BuilderBase::DynamicCountedFor(BValue init_value, BValue trip_count,
                                      BValue stride, Function* body,
                                      absl::Span<const BValue> invariant_args,
                                      const SourceInfo& loc,
                                      std::string_view name) {
  if (ErrorPending()) {
    return BValue();
  }
  std::vector<Node*> invariant_arg_nodes;
  for (const BValue& arg : invariant_args) {
    invariant_arg_nodes.push_back(arg.node());
  }
  return AddNode<xls::DynamicCountedFor>(loc, init_value.node(),
                                         trip_count.node(), stride.node(),
                                         invariant_arg_nodes, body, name);
}

BValue BuilderBase::Map(BValue operand, Function* to_apply,
                        const SourceInfo& loc, std::string_view name) {
  if (ErrorPending()) {
    return BValue();
  }
  return AddNode<xls::Map>(loc, operand.node(), to_apply, name);
}

BValue BuilderBase::Invoke(absl::Span<const BValue> args, Function* to_apply,
                           const SourceInfo& loc, std::string_view name) {
  if (ErrorPending()) {
    return BValue();
  }
  std::vector<Node*> arg_nodes;
  arg_nodes.reserve(args.size());
  for (const BValue& value : args) {
    arg_nodes.push_back(value.node());
  }
  return AddNode<xls::Invoke>(loc, arg_nodes, to_apply, name);
}

BValue BuilderBase::ArrayIndex(BValue arg, absl::Span<const BValue> indices,
                               const SourceInfo& loc, std::string_view name) {
  if (ErrorPending()) {
    return BValue();
  }
  for (int64_t i = 0; i < indices.size(); ++i) {
    const BValue& index = indices[i];
    if (!index.node()->GetType()->IsBits()) {
      return SetError(absl::StrFormat("Indices to multi-array index operation "
                                      "must all be bits types, index %d is: %s",
                                      i, index.node()->GetType()->ToString()),
                      loc);
    }
  }
  if (indices.size() > GetArrayDimensionCount(arg.node()->GetType())) {
    return SetError(
        absl::StrFormat("Too many indices (%d) to index into array of type %s",
                        indices.size(), arg.node()->GetType()->ToString()),
        loc);
  }

  std::vector<Node*> index_operands;
  for (const BValue& index : indices) {
    index_operands.push_back(index.node());
  }
  return AddNode<xls::ArrayIndex>(loc, arg.node(), index_operands, name);
}

BValue BuilderBase::ArraySlice(BValue array, BValue start, int64_t width,
                               const SourceInfo& loc, std::string_view name) {
  if (ErrorPending()) {
    return BValue();
  }
  if (width == 0) {
    return SetError("Array slice operation must have width > 0", loc);
  }
  if (!array.node()->GetType()->IsArray()) {
    return SetError(absl::StrFormat("Argument of array slice operation must "
                                    "be an array type, array is: %s",
                                    array.node()->GetType()->ToString()),
                    loc);
  }
  if (!start.node()->GetType()->IsBits()) {
    return SetError(absl::StrFormat("Indices to array slice operation must "
                                    "be bits types, start is: %s",
                                    start.node()->GetType()->ToString()),
                    loc);
  }

  return AddNode<xls::ArraySlice>(loc, array.node(), start.node(), width, name);
}

BValue BuilderBase::ArrayUpdate(BValue arg, BValue update_value,
                                absl::Span<const BValue> indices,
                                const SourceInfo& loc, std::string_view name) {
  if (ErrorPending()) {
    return BValue();
  }
  for (int64_t i = 0; i < indices.size(); ++i) {
    const BValue& index = indices[i];
    if (!index.node()->GetType()->IsBits()) {
      return SetError(absl::StrFormat("Indices to multi-array update operation "
                                      "must all be bits types, index %d is: %s",
                                      i, index.node()->GetType()->ToString()),
                      loc);
    }
  }
  if (indices.size() > GetArrayDimensionCount(arg.node()->GetType())) {
    return SetError(
        absl::StrFormat("Too many indices (%d) to index into array of type %s",
                        indices.size(), arg.node()->GetType()->ToString()),
        loc);
  }
  absl::StatusOr<Type*> indexed_type_status =
      GetIndexedElementType(arg.node()->GetType(), indices.size());
  // With the check above against the array dimension count, this should never
  // fail but check for it anyway.
  if (!indexed_type_status.ok()) {
    return SetError(
        absl::StrFormat("Unable to determing indexed element type; indexing %s "
                        "with %d indices",
                        arg.node()->GetType()->ToString(), indices.size()),
        loc);
  }
  Type* indexed_type = indexed_type_status.value();
  if (update_value.node()->GetType() != indexed_type) {
    return SetError(
        absl::StrFormat("Expected update value to have type %s; has type %s",
                        indexed_type->ToString(),
                        update_value.node()->GetType()->ToString()),
        loc);
  }
  std::vector<Node*> index_operands;
  for (const BValue& index : indices) {
    index_operands.push_back(index.node());
  }
  return AddNode<xls::ArrayUpdate>(loc, arg.node(), update_value.node(),
                                   index_operands, name);
}

BValue BuilderBase::ArrayConcat(absl::Span<const BValue> operands,
                                const SourceInfo& loc, std::string_view name) {
  if (ErrorPending()) {
    return BValue();
  }

  std::vector<Node*> node_operands;
  Type* zeroth_element_type = nullptr;

  for (const BValue operand : operands) {
    node_operands.push_back(operand.node());
    if (!operand.node()->GetType()->IsArray()) {
      return SetError(
          absl::StrFormat(
              "Cannot array-concat node %s because it has non-array type %s",
              operand.node()->GetName(), operand.node()->GetType()->ToString()),
          loc);
    }

    ArrayType* operand_type = operand.node()->GetType()->AsArrayOrDie();
    Type* element_type = operand_type->element_type();

    if (!zeroth_element_type) {
      zeroth_element_type = element_type;
    } else if (zeroth_element_type != element_type) {
      return SetError(
          absl::StrFormat(
              "Cannot array-concat node %s because it has element type %s"
              " but expected %s",
              operand.node()->GetName(), element_type->ToString(),
              zeroth_element_type->ToString()),
          loc);
    }
  }

  return AddNode<xls::ArrayConcat>(loc, node_operands, name);
}

BValue BuilderBase::Reverse(BValue arg, const SourceInfo& loc,
                            std::string_view name) {
  if (ErrorPending()) {
    return BValue();
  }
  return AddUnOp(Op::kReverse, arg, loc);
}

BValue BuilderBase::Identity(BValue var, const SourceInfo& loc,
                             std::string_view name) {
  if (ErrorPending()) {
    return BValue();
  }
  return AddUnOp(Op::kIdentity, var, loc);
}

BValue BuilderBase::SignExtend(BValue arg, int64_t new_bit_count,
                               const SourceInfo& loc, std::string_view name) {
  if (ErrorPending()) {
    return BValue();
  }
  return AddNode<xls::ExtendOp>(loc, arg.node(), new_bit_count, Op::kSignExt,
                                name);
}

BValue BuilderBase::ZeroExtend(BValue arg, int64_t new_bit_count,
                               const SourceInfo& loc, std::string_view name) {
  if (ErrorPending()) {
    return BValue();
  }
  return AddNode<xls::ExtendOp>(loc, arg.node(), new_bit_count, Op::kZeroExt,
                                name);
}

BValue BuilderBase::BitSlice(BValue arg, int64_t start, int64_t width,
                             const SourceInfo& loc, std::string_view name) {
  if (ErrorPending()) {
    return BValue();
  }
  return AddNode<xls::BitSlice>(loc, arg.node(), start, width, name);
}

BValue BuilderBase::BitSliceUpdate(BValue arg, BValue start,
                                   BValue update_value, const SourceInfo& loc,
                                   std::string_view name) {
  if (ErrorPending()) {
    return BValue();
  }
  return AddNode<xls::BitSliceUpdate>(loc, arg.node(), start.node(),
                                      update_value.node(), name);
}

BValue BuilderBase::DynamicBitSlice(BValue arg, BValue start, int64_t width,
                                    const SourceInfo& loc,
                                    std::string_view name) {
  if (ErrorPending()) {
    return BValue();
  }
  return AddNode<xls::DynamicBitSlice>(loc, arg.node(), start.node(), width,
                                       name);
}

BValue BuilderBase::Encode(BValue arg, const SourceInfo& loc,
                           std::string_view name) {
  if (ErrorPending()) {
    return BValue();
  }
  return AddNode<xls::Encode>(loc, arg.node(), name);
}

BValue BuilderBase::Decode(BValue arg, std::optional<int64_t> width,
                           const SourceInfo& loc, std::string_view name) {
  if (ErrorPending()) {
    return BValue();
  }
  if (!arg.GetType()->IsBits()) {
    return SetError(
        absl::StrFormat("Decode argument must be of Bits type; is: %s",
                        arg.GetType()->ToString()),
        loc);
  }
  // The full output width ('width' not given) is an exponential function of the
  // argument width. Set a limit of 16 bits on the argument width.
  const int64_t arg_width = arg.GetType()->AsBitsOrDie()->bit_count();
  if (!width.has_value() && arg_width > 16) {
    return SetError(
        absl::StrFormat(
            "Decode argument width be no greater than 32-bits; is %d bits",
            arg_width),
        loc);
  }
  return AddNode<xls::Decode>(
      loc, arg.node(),
      /*width=*/width.has_value() ? *width : (1LL << arg_width), name);
}

BValue BuilderBase::Shra(BValue operand, BValue amount, const SourceInfo& loc,
                         std::string_view name) {
  if (ErrorPending()) {
    return BValue();
  }
  return AddBinOp(Op::kShra, operand, amount, loc, name);
}
BValue BuilderBase::Shrl(BValue operand, BValue amount, const SourceInfo& loc,
                         std::string_view name) {
  if (ErrorPending()) {
    return BValue();
  }
  return AddBinOp(Op::kShrl, operand, amount, loc, name);
}
BValue BuilderBase::Shll(BValue operand, BValue amount, const SourceInfo& loc,
                         std::string_view name) {
  if (ErrorPending()) {
    return BValue();
  }
  return AddBinOp(Op::kShll, operand, amount, loc, name);
}
BValue BuilderBase::Subtract(BValue lhs, BValue rhs, const SourceInfo& loc,
                             std::string_view name) {
  if (ErrorPending()) {
    return BValue();
  }
  return AddBinOp(Op::kSub, lhs, rhs, loc, name);
}
BValue BuilderBase::Add(BValue lhs, BValue rhs, const SourceInfo& loc,
                        std::string_view name) {
  if (ErrorPending()) {
    return BValue();
  }
  return AddBinOp(Op::kAdd, lhs, rhs, loc, name);
}
BValue BuilderBase::Or(absl::Span<const BValue> operands, const SourceInfo& loc,
                       std::string_view name) {
  if (ErrorPending()) {
    return BValue();
  }
  return AddNaryOp(Op::kOr, operands, loc, name);
}
BValue BuilderBase::Or(BValue lhs, BValue rhs, const SourceInfo& loc,
                       std::string_view name) {
  return Or({lhs, rhs}, loc, name);
}
BValue BuilderBase::Xor(absl::Span<const BValue> operands,
                        const SourceInfo& loc, std::string_view name) {
  if (ErrorPending()) {
    return BValue();
  }
  return AddNaryOp(Op::kXor, operands, loc, name);
}
BValue BuilderBase::Xor(BValue lhs, BValue rhs, const SourceInfo& loc,
                        std::string_view name) {
  return Xor({lhs, rhs}, loc, name);
}
BValue BuilderBase::And(absl::Span<const BValue> operands,
                        const SourceInfo& loc, std::string_view name) {
  if (ErrorPending()) {
    return BValue();
  }
  return AddNaryOp(Op::kAnd, operands, loc, name);
}
BValue BuilderBase::And(BValue lhs, BValue rhs, const SourceInfo& loc,
                        std::string_view name) {
  return And({lhs, rhs}, loc, name);
}

BValue BuilderBase::AndReduce(BValue operand, const SourceInfo& loc,
                              std::string_view name) {
  if (ErrorPending()) {
    return BValue();
  }
  return AddBitwiseReductionOp(Op::kAndReduce, operand);
}

BValue BuilderBase::OrReduce(BValue operand, const SourceInfo& loc,
                             std::string_view name) {
  if (ErrorPending()) {
    return BValue();
  }
  return AddBitwiseReductionOp(Op::kOrReduce, operand);
}

BValue BuilderBase::XorReduce(BValue operand, const SourceInfo& loc,
                              std::string_view name) {
  if (ErrorPending()) {
    return BValue();
  }
  return AddBitwiseReductionOp(Op::kXorReduce, operand);
}

BValue BuilderBase::SMul(BValue lhs, BValue rhs, const SourceInfo& loc,
                         std::string_view name) {
  if (ErrorPending()) {
    return BValue();
  }
  return AddArithOp(Op::kSMul, lhs, rhs, /*result_width=*/std::nullopt, loc,
                    name);
}
BValue BuilderBase::UMul(BValue lhs, BValue rhs, const SourceInfo& loc,
                         std::string_view name) {
  if (ErrorPending()) {
    return BValue();
  }
  return AddArithOp(Op::kUMul, lhs, rhs, /*result_width=*/std::nullopt, loc,
                    name);
}
BValue BuilderBase::SMul(BValue lhs, BValue rhs, int64_t result_width,
                         const SourceInfo& loc, std::string_view name) {
  if (ErrorPending()) {
    return BValue();
  }
  return AddArithOp(Op::kSMul, lhs, rhs, result_width, loc, name);
}
BValue BuilderBase::UMul(BValue lhs, BValue rhs, int64_t result_width,
                         const SourceInfo& loc, std::string_view name) {
  if (ErrorPending()) {
    return BValue();
  }
  return AddArithOp(Op::kUMul, lhs, rhs, result_width, loc, name);
}
BValue BuilderBase::SMulp(BValue lhs, BValue rhs, const SourceInfo& loc,
                          std::string_view name) {
  if (ErrorPending()) {
    return BValue();
  }
  return AddPartialProductOp(Op::kSMulp, lhs, rhs,
                             /*result_width=*/std::nullopt, loc, name);
}
BValue BuilderBase::UMulp(BValue lhs, BValue rhs, const SourceInfo& loc,
                          std::string_view name) {
  if (ErrorPending()) {
    return BValue();
  }
  return AddPartialProductOp(Op::kUMulp, lhs, rhs,
                             /*result_width=*/std::nullopt, loc, name);
}
BValue BuilderBase::SMulp(BValue lhs, BValue rhs, int64_t result_width,
                          const SourceInfo& loc, std::string_view name) {
  if (ErrorPending()) {
    return BValue();
  }
  return AddPartialProductOp(Op::kSMulp, lhs, rhs, result_width, loc, name);
}
BValue BuilderBase::UMulp(BValue lhs, BValue rhs, int64_t result_width,
                          const SourceInfo& loc, std::string_view name) {
  if (ErrorPending()) {
    return BValue();
  }
  return AddPartialProductOp(Op::kUMulp, lhs, rhs, result_width, loc, name);
}
BValue BuilderBase::UDiv(BValue lhs, BValue rhs, const SourceInfo& loc,
                         std::string_view name) {
  if (ErrorPending()) {
    return BValue();
  }
  return AddBinOp(Op::kUDiv, lhs, rhs, loc, name);
}
BValue BuilderBase::SDiv(BValue lhs, BValue rhs, const SourceInfo& loc,
                         std::string_view name) {
  if (ErrorPending()) {
    return BValue();
  }
  return AddBinOp(Op::kSDiv, lhs, rhs, loc, name);
}
BValue BuilderBase::Eq(BValue lhs, BValue rhs, const SourceInfo& loc,
                       std::string_view name) {
  if (ErrorPending()) {
    return BValue();
  }
  return AddCompareOp(Op::kEq, lhs, rhs, loc, name);
}
BValue BuilderBase::Ne(BValue lhs, BValue rhs, const SourceInfo& loc,
                       std::string_view name) {
  if (ErrorPending()) {
    return BValue();
  }
  return AddCompareOp(Op::kNe, lhs, rhs, loc, name);
}
BValue BuilderBase::UGe(BValue lhs, BValue rhs, const SourceInfo& loc,
                        std::string_view name) {
  if (ErrorPending()) {
    return BValue();
  }
  return AddCompareOp(Op::kUGe, lhs, rhs, loc, name);
}
BValue BuilderBase::UGt(BValue lhs, BValue rhs, const SourceInfo& loc,
                        std::string_view name) {
  if (ErrorPending()) {
    return BValue();
  }
  return AddCompareOp(Op::kUGt, lhs, rhs, loc, name);
}
BValue BuilderBase::ULe(BValue lhs, BValue rhs, const SourceInfo& loc,
                        std::string_view name) {
  if (ErrorPending()) {
    return BValue();
  }
  return AddCompareOp(Op::kULe, lhs, rhs, loc, name);
}
BValue BuilderBase::ULt(BValue lhs, BValue rhs, const SourceInfo& loc,
                        std::string_view name) {
  if (ErrorPending()) {
    return BValue();
  }
  return AddCompareOp(Op::kULt, lhs, rhs, loc, name);
}
BValue BuilderBase::SLt(BValue lhs, BValue rhs, const SourceInfo& loc,
                        std::string_view name) {
  if (ErrorPending()) {
    return BValue();
  }
  return AddCompareOp(Op::kSLt, lhs, rhs, loc, name);
}
BValue BuilderBase::SLe(BValue lhs, BValue rhs, const SourceInfo& loc,
                        std::string_view name) {
  if (ErrorPending()) {
    return BValue();
  }
  return AddCompareOp(Op::kSLe, lhs, rhs, loc, name);
}
BValue BuilderBase::SGe(BValue lhs, BValue rhs, const SourceInfo& loc,
                        std::string_view name) {
  if (ErrorPending()) {
    return BValue();
  }
  return AddCompareOp(Op::kSGe, lhs, rhs, loc, name);
}
BValue BuilderBase::SGt(BValue lhs, BValue rhs, const SourceInfo& loc,
                        std::string_view name) {
  if (ErrorPending()) {
    return BValue();
  }
  return AddCompareOp(Op::kSGt, lhs, rhs, loc, name);
}

BValue BuilderBase::Concat(absl::Span<const BValue> operands,
                           const SourceInfo& loc, std::string_view name) {
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
  return AddNode<xls::Concat>(loc, node_operands, name);
}

FunctionBuilder::FunctionBuilder(std::string_view name, Package* package,
                                 bool should_verify)
    : BuilderBase(std::make_unique<Function>(std::string(name), package),
                  should_verify) {}

BValue FunctionBuilder::Param(std::string_view name, Type* type,
                              const SourceInfo& loc) {
  if (ErrorPending()) {
    return BValue();
  }
  for (xls::Param* param : function()->params()) {
    if (name == param->GetName()) {
      return SetError(
          absl::StrFormat("Parameter named \"%s\" already exists", name), loc);
    }
  }
  return AddNode<xls::Param>(loc, name, type);
}

absl::StatusOr<Function*> FunctionBuilder::Build() {
  if (function_ == nullptr) {
    return absl::FailedPreconditionError(
        "Cannot build function multiple times");
  }
  if (function_->node_count() == 0) {
    return absl::InvalidArgumentError("Function cannot be empty");
  }
  XLS_ASSIGN_OR_RETURN(BValue last_value, GetLastValue());
  return BuildWithReturnValue(last_value);
}

absl::StatusOr<Function*> FunctionBuilder::BuildWithReturnValue(
    BValue return_value) {
  if (ErrorPending()) {
    std::string msg = absl::StrCat(error_msg_, "\n");
    for (const SourceLocation& loc : error_loc_.locations) {
      absl::StrAppendFormat(&msg, "  File: %d, Line: %d, Col: %d\n",
                            loc.fileno().value(), loc.lineno().value(),
                            loc.colno().value());
    }
    absl::StrAppend(&msg, "Stack Trace:\n" + error_stacktrace_);
    return absl::InvalidArgumentError("Could not build IR: " + msg);
  }
  XLS_RET_CHECK_EQ(return_value.builder(), this);
  // down_cast the FunctionBase* to Function*. We know this is safe because
  // FunctionBuilder constructs and passes a Function to BuilderBase
  // constructor so function_ is always a Function.
  Function* f = package()->AddFunction(
      absl::WrapUnique(down_cast<Function*>(function_.release())));
  XLS_RETURN_IF_ERROR(f->set_return_value(return_value.node()));
  if (should_verify_) {
    XLS_RETURN_IF_ERROR(VerifyFunction(f));
  }
  return f;
}

ProcBuilder::ProcBuilder(std::string_view name, std::string_view token_name,
                         Package* package, bool should_verify)
    : BuilderBase(std::make_unique<Proc>(name, token_name, package),
                  should_verify),
      token_param_(proc()->TokenParam(), this) {}

Proc* ProcBuilder::proc() const { return down_cast<Proc*>(function()); }

absl::StatusOr<Proc*> ProcBuilder::Build(BValue token,
                                         absl::Span<const BValue> next_state) {
  if (ErrorPending()) {
    std::string msg = absl::StrCat(error_msg_, "\n");
    for (const SourceLocation& loc : error_loc_.locations) {
      absl::StrAppendFormat(&msg, "  File: %d, Line: %d, Col: %d\n",
                            loc.fileno().value(), loc.lineno().value(),
                            loc.colno().value());
    }
    absl::StrAppend(&msg, "Stack Trace:\n" + error_stacktrace_);
    return absl::InvalidArgumentError("Could not build IR: " + msg);
  }
  if (!GetType(token)->IsToken()) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Recurrent token of proc must be token type, is: %s.",
                        GetType(token)->ToString()));
  }
  if (next_state.size() != state_params_.size()) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Number of recurrent state elements given (%d) does "
                        "equal the number of state elements in the proc (%d)",
                        next_state.size(), state_params_.size()));
  }
  for (int64_t i = 0; i < state_params_.size(); ++i) {
    if (GetType(next_state[i]) != GetType(GetStateParam(i))) {
      return absl::InvalidArgumentError(
          absl::StrFormat("Recurrent state type %s does not match proc "
                          "parameter state type %s for element %d.",
                          GetType(GetStateParam(i))->ToString(),
                          GetType(next_state[i])->ToString(), i));
    }
  }

  // down_cast the FunctionBase* to Proc*. We know this is safe because
  // ProcBuilder constructs and passes a Proc to BuilderBase constructor so
  // function_ is always a Proc.
  Proc* proc = package()->AddProc(
      absl::WrapUnique(down_cast<Proc*>(function_.release())));
  XLS_RETURN_IF_ERROR(proc->SetNextToken(token.node()));
  for (int64_t i = 0; i < next_state.size(); ++i) {
    XLS_RETURN_IF_ERROR(proc->SetNextStateElement(i, next_state[i].node()));
  }
  if (should_verify_) {
    XLS_RETURN_IF_ERROR(VerifyProc(proc));
  }
  return proc;
}

BValue ProcBuilder::StateElement(std::string_view name,
                                 const Value initial_value,
                                 const SourceInfo& loc) {
  absl::StatusOr<xls::Param*> param_or =
      proc()->AppendStateElement(name, initial_value);
  if (!param_or.ok()) {
    return SetError(absl::StrFormat("Unable to add state element: %s",
                                    param_or.status().message()),
                    loc);
  }
  state_params_.push_back(CreateBValue(param_or.value(), loc));
  return state_params_.back();
}

BValue ProcBuilder::Param(std::string_view name, Type* type,
                          const SourceInfo& loc) {
  if (ErrorPending()) {
    return BValue();
  }
  return SetError("Use StateElement to add state parameters to procs", loc);
}

BuilderBase::BuilderBase(std::unique_ptr<FunctionBase> function,
                         bool should_verify)
    : function_(std::move(function)),
      error_pending_(false),
      should_verify_(should_verify) {}

BuilderBase::~BuilderBase() = default;

Package* BuilderBase::package() const { return function_->package(); }

BValue BuilderBase::AddArithOp(Op op, BValue lhs, BValue rhs,
                               std::optional<int64_t> result_width,
                               const SourceInfo& loc, std::string_view name) {
  if (ErrorPending()) {
    return BValue();
  }
  XLS_CHECK_EQ(lhs.builder(), rhs.builder());
  if (!lhs.GetType()->IsBits() || !rhs.GetType()->IsBits()) {
    return SetError(
        absl::StrFormat(
            "Arithmetic arguments must be of Bits type; is: %s and %s",
            lhs.GetType()->ToString(), rhs.GetType()->ToString()),
        loc);
  }
  int64_t width;
  if (result_width.has_value()) {
    width = *result_width;
  } else {
    if (lhs.BitCountOrDie() != rhs.BitCountOrDie()) {
      return SetError(
          absl::StrFormat(
              "Arguments of arithmetic operation must be same width if "
              "result width is not specified; is: %s and %s",
              lhs.GetType()->ToString(), rhs.GetType()->ToString()),
          loc);
    }
    width = lhs.BitCountOrDie();
  }
  return AddNode<ArithOp>(loc, lhs.node(), rhs.node(), width, op, name);
}

BValue BuilderBase::AddPartialProductOp(Op op, BValue lhs, BValue rhs,
                                        std::optional<int64_t> result_width,
                                        const SourceInfo& loc,
                                        std::string_view name) {
  if (ErrorPending()) {
    return BValue();
  }
  XLS_CHECK_EQ(lhs.builder(), rhs.builder());
  if (!lhs.GetType()->IsBits() || !rhs.GetType()->IsBits()) {
    return SetError(
        absl::StrFormat(
            "Arithmetic arguments must be of Bits type; is: %s and %s",
            lhs.GetType()->ToString(), rhs.GetType()->ToString()),
        loc);
  }
  int64_t width;
  if (result_width.has_value()) {
    width = *result_width;
  } else {
    if (lhs.BitCountOrDie() != rhs.BitCountOrDie()) {
      return SetError(
          absl::StrFormat(
              "Arguments of arithmetic operation must be same width if "
              "result width is not specified; is: %s and %s",
              lhs.GetType()->ToString(), rhs.GetType()->ToString()),
          loc);
    }
    width = lhs.BitCountOrDie();
  }
  return AddNode<PartialProductOp>(loc, lhs.node(), rhs.node(), width, op,
                                   name);
}

BValue BuilderBase::AddUnOp(Op op, BValue x, const SourceInfo& loc,
                            std::string_view name) {
  if (ErrorPending()) {
    return BValue();
  }
  XLS_CHECK_EQ(this, x.builder());
  if (!IsOpClass<UnOp>(op)) {
    return SetError(absl::StrFormat("Op %s is not a operation of class UnOp",
                                    OpToString(op)),
                    loc);
  }
  return AddNode<UnOp>(loc, x.node(), op, name);
}

BValue BuilderBase::AddBinOp(Op op, BValue lhs, BValue rhs,
                             const SourceInfo& loc, std::string_view name) {
  if (ErrorPending()) {
    return BValue();
  }
  XLS_CHECK_EQ(lhs.builder(), rhs.builder());
  if (!IsOpClass<BinOp>(op)) {
    return SetError(absl::StrFormat("Op %s is not a operation of class BinOp",
                                    OpToString(op)),
                    loc);
  }
  return AddNode<BinOp>(loc, lhs.node(), rhs.node(), op, name);
}

BValue BuilderBase::AddCompareOp(Op op, BValue lhs, BValue rhs,
                                 const SourceInfo& loc,
                                 std::string_view name) {
  if (ErrorPending()) {
    return BValue();
  }
  XLS_CHECK_EQ(lhs.builder(), rhs.builder());
  if (!IsOpClass<CompareOp>(op)) {
    return SetError(
        absl::StrFormat("Op %s is not a operation of class CompareOp",
                        OpToString(op)),
        loc);
  }
  return AddNode<CompareOp>(loc, lhs.node(), rhs.node(), op, name);
}

BValue BuilderBase::AddNaryOp(Op op, absl::Span<const BValue> args,
                              const SourceInfo& loc, std::string_view name) {
  if (ErrorPending()) {
    return BValue();
  }
  if (!IsOpClass<NaryOp>(op)) {
    return SetError(absl::StrFormat("Op %s is not a operation of class NaryOp",
                                    OpToString(op)),
                    loc);
  }
  std::vector<Node*> nodes;
  for (const BValue& bvalue : args) {
    nodes.push_back(bvalue.node());
  }
  return AddNode<NaryOp>(loc, nodes, op, name);
}

BValue BuilderBase::AddBitwiseReductionOp(Op op, BValue arg,
                                          const SourceInfo& loc,
                                          std::string_view name) {
  if (ErrorPending()) {
    return BValue();
  }
  XLS_CHECK_EQ(this, arg.builder());
  return AddNode<BitwiseReductionOp>(loc, arg.node(), op, name);
}

BValue BuilderBase::Assert(BValue token, BValue condition,
                           std::string_view message,
                           std::optional<std::string> label,
                           const SourceInfo& loc, std::string_view name) {
  if (ErrorPending()) {
    return BValue();
  }
  if (!token.GetType()->IsToken()) {
    return SetError(
        absl::StrFormat("First operand of assert must be of token type; is: %s",
                        token.GetType()->ToString()),
        loc);
  }
  if (!condition.GetType()->IsBits() ||
      condition.GetType()->AsBitsOrDie()->bit_count() != 1) {
    return SetError(
        absl::StrFormat("Condition operand of assert must be of bits "
                        "type of width 1; is: %s",
                        condition.GetType()->ToString()),
        loc);
  }
  return AddNode<xls::Assert>(loc, token.node(), condition.node(), message,
                              label, name);
}

BValue BuilderBase::Trace(BValue token, BValue condition,
                          absl::Span<const BValue> args,
                          absl::Span<const FormatStep> format,
                          const SourceInfo& loc, std::string_view name) {
  if (ErrorPending()) {
    return BValue();
  }
  if (!token.GetType()->IsToken()) {
    return SetError(
        absl::StrFormat("First operand of trace must be of token type; is: %s",
                        token.GetType()->ToString()),
        loc);
  }
  if (!condition.GetType()->IsBits() ||
      condition.GetType()->AsBitsOrDie()->bit_count() != 1) {
    return SetError(
        absl::StrFormat("Condition operand of trace must be of bits "
                        "type of width 1; is: %s",
                        condition.GetType()->ToString()),
        loc);
  }

  int64_t expected_operands = OperandsExpectedByFormat(format);
  if (args.size() != expected_operands) {
    return SetError(
        absl::StrFormat(
            "Trace node expects %d data operands, but %d were supplied",
            expected_operands, args.size()),
        loc);
  }

  std::vector<Node*> arg_nodes;
  for (const BValue& arg : args) {
    if (!arg.GetType()->IsBits()) {
      return SetError(
          absl::StrFormat("Trace arguments must be of bits type; is: %s",
                          arg.GetType()->ToString()),
          loc);
    }
    arg_nodes.push_back(arg.node());
  }

  return AddNode<xls::Trace>(loc, token.node(), condition.node(), arg_nodes,
                             format, name);
}

BValue BuilderBase::Trace(BValue token, BValue condition,
                          absl::Span<const BValue> args,
                          std::string_view format_string,
                          const SourceInfo& loc, std::string_view name) {
  auto parse_status = ParseFormatString(format_string);

  if (!parse_status.ok()) {
    return SetError(parse_status.status().message(), loc);
  }

  return Trace(token, condition, args, parse_status.value(), loc, name);
}

BValue BuilderBase::Cover(BValue token, BValue condition,
                          std::string_view label, const SourceInfo& loc,
                          std::string_view name) {
  if (ErrorPending()) {
    return BValue();
  }
  if (!token.GetType()->IsToken()) {
    return SetError(
        absl::StrFormat("First operand of cover must be of token type; is: %s",
                        token.GetType()->ToString()),
        loc);
  }
  if (!condition.GetType()->IsBits() ||
      condition.GetType()->AsBitsOrDie()->bit_count() != 1) {
    return SetError(
        absl::StrFormat("Condition operand of cover must be of bits "
                        "type of width 1; is: %s",
                        condition.GetType()->ToString()),
        loc);
  }
  if (label.empty()) {
    return SetError("The label of a cover node cannot be empty.", loc);
  }
  return AddNode<xls::Cover>(loc, token.node(), condition.node(), label, name);
}

BValue BuilderBase::Gate(BValue condition, BValue data, const SourceInfo& loc,
                         std::string_view name) {
  if (ErrorPending()) {
    return BValue();
  }
  if (!condition.GetType()->IsBits() ||
      condition.GetType()->AsBitsOrDie()->bit_count() != 1) {
    return SetError(
        absl::StrFormat(
            "Condition operand of gate must be of bits type of width 1; is: %s",
            condition.GetType()->ToString()),
        loc);
  }
  return AddNode<xls::Gate>(loc, condition.node(), data.node(), name);
}

BValue ProcBuilder::Receive(Channel* channel, BValue token,
                            const SourceInfo& loc, std::string_view name) {
  if (ErrorPending()) {
    return BValue();
  }
  if (!token.GetType()->IsToken()) {
    return SetError(
        absl::StrFormat(
            "Token operand of receive must be of token type; is: %s",
            token.GetType()->ToString()),
        loc);
  }
  return AddNode<xls::Receive>(loc, token.node(), /*predicate=*/std::nullopt,
                               channel->id(), /*is_blocking=*/true, name);
}

BValue ProcBuilder::ReceiveNonBlocking(Channel* channel, BValue token,
                                       const SourceInfo& loc,
                                       std::string_view name) {
  if (ErrorPending()) {
    return BValue();
  }
  if (!token.GetType()->IsToken()) {
    return SetError(
        absl::StrFormat(
            "Token operand of receive must be of token type; is: %s",
            token.GetType()->ToString()),
        loc);
  }
  return AddNode<xls::Receive>(loc, token.node(), /*predicate=*/std::nullopt,
                               channel->id(), /*is_blocking=*/false, name);
}

BValue ProcBuilder::ReceiveIf(Channel* channel, BValue token, BValue pred,
                              const SourceInfo& loc, std::string_view name) {
  if (ErrorPending()) {
    return BValue();
  }
  if (!token.GetType()->IsToken()) {
    return SetError(
        absl::StrFormat(
            "Token operand of receive must be of token type; is: %s",
            token.GetType()->ToString()),
        loc);
  }
  if (!pred.GetType()->IsBits() ||
      pred.GetType()->AsBitsOrDie()->bit_count() != 1) {
    return SetError(
        absl::StrFormat("Predicate operand of receive_if must be of bits "
                        "type of width 1; is: %s",
                        pred.GetType()->ToString()),
        loc);
  }
  return AddNode<xls::Receive>(loc, token.node(), pred.node(), channel->id(),
                               /*is_blocking=*/true, name);
}

BValue ProcBuilder::ReceiveIfNonBlocking(Channel* channel, BValue token,
                                         BValue pred, const SourceInfo& loc,
                                         std::string_view name) {
  if (ErrorPending()) {
    return BValue();
  }
  if (!token.GetType()->IsToken()) {
    return SetError(
        absl::StrFormat(
            "Token operand of receive must be of token type; is: %s",
            token.GetType()->ToString()),
        loc);
  }
  if (!pred.GetType()->IsBits() ||
      pred.GetType()->AsBitsOrDie()->bit_count() != 1) {
    return SetError(
        absl::StrFormat("Predicate operand of receive_if must be of bits "
                        "type of width 1; is: %s",
                        pred.GetType()->ToString()),
        loc);
  }
  return AddNode<xls::Receive>(loc, token.node(), pred.node(), channel->id(),
                               /*is_blocking=*/false, name);
}

BValue ProcBuilder::Send(Channel* channel, BValue token, BValue data,
                         const SourceInfo& loc, std::string_view name) {
  if (ErrorPending()) {
    return BValue();
  }
  if (!token.GetType()->IsToken()) {
    return SetError(
        absl::StrFormat("Token operand of send must be of token type; is: %s",
                        token.GetType()->ToString()),
        loc);
  }
  return AddNode<xls::Send>(loc, token.node(), data.node(),
                            /*predicate=*/std::nullopt, channel->id(), name);
}

BValue ProcBuilder::SendIf(Channel* channel, BValue token, BValue pred,
                           BValue data, const SourceInfo& loc,
                           std::string_view name) {
  if (ErrorPending()) {
    return BValue();
  }
  if (!token.GetType()->IsToken()) {
    return SetError(
        absl::StrFormat("Token operand of send must be of token type; is: %s",
                        token.GetType()->ToString()),
        loc);
  }
  if (!pred.GetType()->IsBits() ||
      pred.GetType()->AsBitsOrDie()->bit_count() != 1) {
    return SetError(absl::StrFormat("Predicate operand of send must be of bits "
                                    "type of width 1; is: %s",
                                    pred.GetType()->ToString()),
                    loc);
  }
  return AddNode<xls::Send>(loc, token.node(), data.node(), pred.node(),
                            channel->id(), name);
}

BValue TokenlessProcBuilder::Receive(Channel* channel, const SourceInfo& loc,
                                     std::string_view name) {
  BValue rcv = ProcBuilder::Receive(channel, last_token_, loc, name);
  last_token_ = TupleIndex(rcv, 0);
  return TupleIndex(rcv, 1);
}

std::pair<BValue, BValue> TokenlessProcBuilder::ReceiveNonBlocking(
    Channel* channel, const SourceInfo& loc, std::string_view name) {
  BValue rcv = ProcBuilder::ReceiveNonBlocking(channel, last_token_, loc, name);
  last_token_ = TupleIndex(rcv, 0, loc);
  return {TupleIndex(rcv, 1), TupleIndex(rcv, 2)};
}

BValue TokenlessProcBuilder::ReceiveIf(Channel* channel, BValue pred,
                                       const SourceInfo& loc,
                                       std::string_view name) {
  BValue rcv_if = ProcBuilder::ReceiveIf(channel, last_token_, pred, loc, name);
  last_token_ = TupleIndex(rcv_if, 0);
  return TupleIndex(rcv_if, 1);
}

std::pair<BValue, BValue> TokenlessProcBuilder::ReceiveIfNonBlocking(
    Channel* channel, BValue pred, const SourceInfo& loc,
    std::string_view name) {
  BValue rcv =
      ProcBuilder::ReceiveIfNonBlocking(channel, last_token_, pred, loc, name);
  last_token_ = TupleIndex(rcv, 0, loc);
  return {TupleIndex(rcv, 1), TupleIndex(rcv, 2)};
}

BValue TokenlessProcBuilder::Send(Channel* channel, BValue data,
                                  const SourceInfo& loc,
                                  std::string_view name) {
  last_token_ = ProcBuilder::Send(channel, last_token_, data, loc, name);
  return last_token_;
}

BValue TokenlessProcBuilder::SendIf(Channel* channel, BValue pred, BValue data,
                                    const SourceInfo& loc,
                                    std::string_view name) {
  last_token_ =
      ProcBuilder::SendIf(channel, last_token_, pred, data, loc, name);
  return last_token_;
}

BValue TokenlessProcBuilder::Assert(BValue condition, std::string_view message,
                                    std::optional<std::string> label,
                                    const SourceInfo& loc,
                                    std::string_view name) {
  last_token_ =
      BuilderBase::Assert(last_token_, condition, message, label, loc, name);
  return last_token_;
}

absl::StatusOr<Proc*> TokenlessProcBuilder::Build(
    absl::Span<const BValue> next_state) {
  return ProcBuilder::Build(last_token_, next_state);
}

BValue BlockBuilder::Param(std::string_view name, Type* type,
                           const SourceInfo& loc) {
  if (ErrorPending()) {
    return BValue();
  }
  return SetError("Cannot add parameters to blocks", loc);
}

BValue BlockBuilder::InputPort(std::string_view name, Type* type,
                               const SourceInfo& loc) {
  if (ErrorPending()) {
    return BValue();
  }
  absl::StatusOr<xls::InputPort*> port_status =
      block()->AddInputPort(name, type, loc);
  if (!port_status.ok()) {
    return SetError(absl::StrFormat("Unable to add port to block: %s",
                                    port_status.status().message()),
                    loc);
  }
  return CreateBValue(port_status.value(), loc);
}

BValue BlockBuilder::OutputPort(std::string_view name, BValue operand,
                                const SourceInfo& loc) {
  if (ErrorPending()) {
    return BValue();
  }
  absl::StatusOr<xls::OutputPort*> port_status =
      block()->AddOutputPort(name, operand.node(), loc);
  if (!port_status.ok()) {
    return SetError(absl::StrFormat("Unable to add port to block: %s",
                                    port_status.status().message()),
                    loc);
  }
  return CreateBValue(port_status.value(), loc);
}

BValue BlockBuilder::RegisterRead(Register* reg, const SourceInfo& loc,
                                  std::string_view name) {
  if (ErrorPending()) {
    return BValue();
  }
  if (!block()->IsOwned(reg)) {
    return SetError("Register is defined in different block", loc);
  }
  return AddNode<xls::RegisterRead>(loc, reg, name);
}

BValue BlockBuilder::RegisterWrite(Register* reg, BValue data,
                                   std::optional<BValue> load_enable,
                                   std::optional<BValue> reset,
                                   const SourceInfo& loc,
                                   std::string_view name) {
  if (ErrorPending()) {
    return BValue();
  }
  if (!block()->IsOwned(reg)) {
    return SetError("Register is defined in different block", loc);
  }
  return AddNode<xls::RegisterWrite>(
      loc, data.node(),
      load_enable.has_value() ? std::optional<Node*>(load_enable->node())
                              : std::nullopt,
      reset.has_value() ? std::optional<Node*>(reset->node()) : std::nullopt,
      reg, name);
}

BValue BlockBuilder::InsertRegister(std::string_view name, BValue data,
                                    std::optional<BValue> load_enable,
                                    const SourceInfo& loc) {
  if (ErrorPending()) {
    return BValue();
  }
  absl::StatusOr<Register*> reg_status =
      block()->AddRegister(name, data.GetType());
  if (!reg_status.ok()) {
    return SetError(absl::StrFormat("Cannot add register: %s",
                                    reg_status.status().message()),
                    loc);
  }
  Register* reg = reg_status.value();
  RegisterWrite(reg, data, load_enable, /*reset=*/std::nullopt, loc,
                absl::StrFormat("%s_write", reg->name()));
  return RegisterRead(reg, loc, reg->name());
}

BValue BlockBuilder::InsertRegister(std::string_view name, BValue data,
                                    BValue reset_signal, Reset reset,
                                    std::optional<BValue> load_enable,
                                    const SourceInfo& loc) {
  if (ErrorPending()) {
    return BValue();
  }
  absl::StatusOr<Register*> reg_status =
      block()->AddRegister(name, data.GetType(), reset);
  if (!reg_status.ok()) {
    return SetError(absl::StrFormat("Cannot add register: %s",
                                    reg_status.status().message()),
                    loc);
  }
  Register* reg = reg_status.value();
  RegisterWrite(reg, data, load_enable, reset_signal, loc,
                absl::StrFormat("%s_write", reg->name()));
  return RegisterRead(reg, loc, reg->name());
}

absl::StatusOr<Block*> BlockBuilder::Build() {
  if (ErrorPending()) {
    std::string msg = absl::StrCat(error_msg_, "\n");
    for (const SourceLocation& loc : error_loc_.locations) {
      absl::StrAppendFormat(&msg, "  File: %d, Line: %d, Col: %d\n",
                            loc.fileno().value(), loc.lineno().value(),
                            loc.colno().value());
    }
    absl::StrAppend(&msg, "Stack Trace:\n" + error_stacktrace_);
    return absl::InvalidArgumentError("Could not build IR: " + msg);
  }

  // down_cast the FunctionBase* to Block*. We know this is safe because
  // BlockBuilder constructs and passes a Block to BuilderBase constructor so
  // function_ is always a Block.
  Block* block = package()->AddBlock(
      absl::WrapUnique(down_cast<Block*>(function_.release())));
  if (should_verify_) {
    XLS_RETURN_IF_ERROR(VerifyBlock(block));
  }
  return block;
}

BValue BlockBuilder::InstantiationInput(Instantiation* instantiation,
                                        std::string_view port_name,
                                        BValue data, const SourceInfo& loc,
                                        std::string_view name) {
  if (ErrorPending()) {
    return BValue();
  }
  if (!block()->IsOwned(instantiation)) {
    return SetError(
        absl::StrFormat("Instantiation `%s` (%p) is defined in different block",
                        instantiation->name(), instantiation),
        loc);
  }
  return AddNode<xls::InstantiationInput>(loc, data.node(), instantiation,
                                          port_name, name);
}

BValue BlockBuilder::InstantiationOutput(Instantiation* instantiation,
                                         std::string_view port_name,
                                         const SourceInfo& loc,
                                         std::string_view name) {
  if (ErrorPending()) {
    return BValue();
  }
  if (!block()->IsOwned(instantiation)) {
    return SetError(
        absl::StrFormat("Instantiation `%s` (%p) is defined in different block",
                        instantiation->name(), instantiation),
        loc);
  }
  return AddNode<xls::InstantiationOutput>(loc, instantiation, port_name, name);
}

}  // namespace xls
