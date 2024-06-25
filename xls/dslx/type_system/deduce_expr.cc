// Copyright 2024 The XLS Authors
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

#include "xls/dslx/type_system/deduce_expr.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/cleanup/cleanup.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/dslx/constexpr_evaluator.h"
#include "xls/dslx/errors.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/frontend/ast_node.h"
#include "xls/dslx/interp_value.h"
#include "xls/dslx/type_system/deduce_ctx.h"
#include "xls/dslx/type_system/deduce_utils.h"
#include "xls/dslx/type_system/parametric_env.h"
#include "xls/dslx/type_system/type.h"
#include "xls/dslx/type_system/unwrap_meta_type.h"
#include "xls/dslx/warning_kind.h"
#include "xls/ir/bits.h"

namespace xls::dslx {

// Forward declaration from sister implementation file as we are recursively
// bound to the central deduce-and-resolve routine in our node deduction
// routines.
//
// TODO(cdleary): 2024-01-16 We can break this circular resolution with a
// virtual function on DeduceCtx when we get things refactored nicely.
extern absl::StatusOr<std::unique_ptr<Type>> DeduceAndResolve(
    const AstNode* node, DeduceCtx* ctx);
extern absl::StatusOr<std::unique_ptr<Type>> Resolve(const Type& type,
                                                     DeduceCtx* ctx);

absl::StatusOr<std::unique_ptr<Type>> DeduceArray(const Array* node,
                                                  DeduceCtx* ctx) {
  VLOG(5) << "DeduceArray; node: " << node->ToString();

  std::vector<std::unique_ptr<Type>> member_types;
  for (Expr* member : node->members()) {
    XLS_ASSIGN_OR_RETURN(std::unique_ptr<Type> member_type,
                         DeduceAndResolve(member, ctx));
    member_types.push_back(std::move(member_type));
  }

  for (int64_t i = 1; i < member_types.size(); ++i) {
    if (*member_types[0] != *member_types[i]) {
      return ctx->TypeMismatchError(
          node->span(), nullptr, *member_types[0], nullptr, *member_types[i],
          "Array member did not have same type as other members.");
    }
  }

  if (!member_types.empty() && member_types[0]->HasToken()) {
    return TypeInferenceErrorStatus(
        node->span(), member_types[0].get(),
        "Types with tokens cannot be placed in arrays.");
  }

  if (node->has_ellipsis() && node->members().empty()) {
    return TypeInferenceErrorStatus(
        node->span(), nullptr,
        "Array cannot have an ellipsis without an element to repeat; please "
        "add at least one element");
  }

  auto member_types_dim =
      TypeDim::CreateU32(static_cast<uint32_t>(member_types.size()));

  // Try to infer the array type from the first member.
  std::unique_ptr<ArrayType> inferred;
  if (!member_types.empty()) {
    inferred = std::make_unique<ArrayType>(member_types[0]->CloneToUnique(),
                                           member_types_dim);
  }

  if (node->type_annotation() == nullptr) {
    if (inferred != nullptr) {
      return inferred;
    }

    return TypeInferenceErrorStatus(
        node->span(), nullptr, "Cannot deduce the type of an empty array.");
  }

  XLS_ASSIGN_OR_RETURN(std::unique_ptr<Type> annotated,
                       ctx->Deduce(node->type_annotation()));
  XLS_ASSIGN_OR_RETURN(annotated,
                       UnwrapMetaType(std::move(annotated), node->span(),
                                      "array type-prefix position"));
  auto* array_type = dynamic_cast<ArrayType*>(annotated.get());
  if (array_type == nullptr) {
    return TypeInferenceErrorStatus(
        node->span(), annotated.get(),
        "Array was not annotated with an array type.");
  }

  if (array_type->HasParametricDims()) {
    return TypeInferenceErrorStatus(
        node->type_annotation()->span(), array_type,
        absl::StrFormat("Annotated type for array "
                        "literal must be constexpr; type has dimensions that "
                        "cannot be resolved."));
  }

  // If we were presented with the wrong number of elements (vs what the
  // annotated type expected), flag an error.
  if (array_type->size() != member_types_dim && !node->has_ellipsis()) {
    std::string message = absl::StrFormat(
        "Annotated array size %s does not match inferred array size %d.",
        array_type->size().ToString(), member_types.size());
    if (inferred == nullptr) {
      // No type to compare our expectation to, as there was no member to infer
      // the type from.
      return TypeInferenceErrorStatus(node->span(), array_type, message);
    }
    return ctx->TypeMismatchError(node->span(), nullptr, *array_type, nullptr,
                                  *inferred, message);
  }

  // Implementation note: we can only do this after we've checked that the size
  // is correct (zero elements provided and zero elements expected).
  if (member_types.empty()) {
    return annotated;
  }

  XLS_ASSIGN_OR_RETURN(std::unique_ptr<Type> resolved_element_type,
                       Resolve(array_type->element_type(), ctx));
  if (*resolved_element_type != *member_types[0]) {
    return ctx->TypeMismatchError(
        node->members().at(0)->span(), nullptr, *resolved_element_type, nullptr,
        *member_types[0],
        "Annotated element type did not match inferred "
        "element type.");
  }

  if (node->has_ellipsis()) {
    // Need to constexpr evaluate here - while we have the concrete type - or
    // else we'd infer the wrong array size.
    XLS_RETURN_IF_ERROR(ConstexprEvaluator::Evaluate(
        ctx->import_data(), ctx->type_info(), ctx->warnings(),
        ctx->GetCurrentParametricEnv(), node, array_type));
    return annotated;
  }

  return inferred;
}

absl::StatusOr<std::unique_ptr<Type>> DeduceConstantArray(
    const ConstantArray* node, DeduceCtx* ctx) {
  if (node->type_annotation() == nullptr) {
    return DeduceArray(node, ctx);
  }

  XLS_ASSIGN_OR_RETURN(std::unique_ptr<Type> type,
                       ctx->Deduce(node->type_annotation()));
  XLS_ASSIGN_OR_RETURN(
      type, UnwrapMetaType(std::move(type), node->type_annotation()->span(),
                           "array type-prefix position"));

  auto* array_type = dynamic_cast<ArrayType*>(type.get());
  if (array_type == nullptr) {
    return TypeInferenceErrorStatus(
        node->type_annotation()->span(), type.get(),
        absl::StrFormat("Annotated type for array "
                        "literal must be an array type; got %s %s",
                        type->GetDebugTypeName(),
                        node->type_annotation()->ToString()));
  }

  const Type& element_type = array_type->element_type();
  for (Expr* member : node->members()) {
    XLS_RET_CHECK(IsConstant(member));
    if (Number* number = dynamic_cast<Number*>(member);
        number != nullptr && number->type_annotation() == nullptr) {
      ctx->type_info()->SetItem(member, element_type);

      if (std::optional<BitsLikeProperties> bits_like =
              GetBitsLike(element_type);
          bits_like.has_value()) {
        XLS_RETURN_IF_ERROR(
            TryEnsureFitsInType(*number, bits_like.value(), element_type));
      } else {
        return TypeInferenceErrorStatus(
            number->span(), &element_type,
            absl::StrFormat("Annotated element type for array cannot be "
                            "applied to a literal number"));
      }
    }
  }

  XLS_RETURN_IF_ERROR(DeduceArray(node, ctx).status());
  return type;
}

absl::StatusOr<std::unique_ptr<Type>> DeduceNumber(const Number* node,
                                                   DeduceCtx* ctx) {
  auto note_constexpr_value = [&](const Type& type) -> absl::Status {
    if (type.HasParametricDims()) {
      return absl::OkStatus();
    }
    XLS_ASSIGN_OR_RETURN(InterpValue value, EvaluateNumber(*node, type));
    ctx->type_info()->NoteConstExpr(node, value);
    return absl::OkStatus();
  };

  std::unique_ptr<Type> type;

  ParametricEnv bindings = ctx->GetCurrentParametricEnv();
  if (node->type_annotation() == nullptr) {
    switch (node->number_kind()) {
      case NumberKind::kBool: {
        auto type = BitsType::MakeU1();
        ctx->type_info()->SetItem(node, *type);
        XLS_RETURN_IF_ERROR(note_constexpr_value(*type));
        return type;
      }
      case NumberKind::kCharacter: {
        auto type = BitsType::MakeU8();
        ctx->type_info()->SetItem(node, *type);
        XLS_RETURN_IF_ERROR(note_constexpr_value(*type));
        return type;
      }
      default:
        break;
    }

    if (ctx->in_typeless_number_ctx()) {
      type = BitsType::MakeU32();
    } else {
      return TypeInferenceErrorStatus(node->span(), nullptr,
                                      "Could not infer a type for "
                                      "this number, please annotate a type.");
    }
  } else {
    XLS_ASSIGN_OR_RETURN(type, ctx->Deduce(node->type_annotation()));
    XLS_ASSIGN_OR_RETURN(
        type, UnwrapMetaType(std::move(type), node->type_annotation()->span(),
                             "numeric literal type-prefix"));
  }

  CHECK(type != nullptr);
  XLS_ASSIGN_OR_RETURN(type, Resolve(*type, ctx));
  XLS_RET_CHECK(!type->IsMeta());

  if (std::optional<BitsLikeProperties> bits_like = GetBitsLike(*type);
      bits_like.has_value()) {
    XLS_RETURN_IF_ERROR(TryEnsureFitsInType(*node, bits_like.value(), *type));
  } else {
    return TypeInferenceErrorStatus(
        node->span(), type.get(),
        "Non-bits type used to define a numeric literal.");
  }
  ctx->type_info()->SetItem(node, *type);
  XLS_RETURN_IF_ERROR(note_constexpr_value(*type));
  return type;
}

absl::StatusOr<std::unique_ptr<Type>> DeduceString(const String* string,
                                                   DeduceCtx* ctx) {
  auto dim = TypeDim::CreateU32(static_cast<uint32_t>(string->text().size()));
  return std::make_unique<ArrayType>(BitsType::MakeU8(), std::move(dim));
}

static bool IsBlockJustUnitTuple(const StatementBlock& block) {
  if (block.empty()) {
    return true;
  }
  if (block.size() != 1) {
    return false;
  }
  const Statement* statement = block.statements().front();
  const Statement::Wrapped& wrapped = statement->wrapped();
  if (!std::holds_alternative<Expr*>(wrapped)) {
    return false;
  }
  const auto* expr = std::get<Expr*>(wrapped);
  auto* tuple = dynamic_cast<const XlsTuple*>(expr);
  if (tuple == nullptr) {
    return false;
  }
  return tuple->empty();
}

static bool IsBlockWithOneFailStmt(const StatementBlock& block) {
  if (block.size() != 1) {
    return false;
  }
  const Statement* statement = block.statements().front();
  const Statement::Wrapped& wrapped = statement->wrapped();
  if (!std::holds_alternative<Expr*>(wrapped)) {
    return false;
  }
  const auto* expr = std::get<Expr*>(wrapped);
  auto* invocation = dynamic_cast<const Invocation*>(expr);
  if (invocation == nullptr) {
    return false;
  }
  Expr* callee = invocation->callee();
  auto* name_ref = dynamic_cast<const NameRef*>(callee);
  if (name_ref == nullptr) {
    return false;
  }
  AnyNameDef any_name_def = name_ref->name_def();
  if (!std::holds_alternative<BuiltinNameDef*>(any_name_def)) {
    return false;
  }
  auto* bnd = std::get<BuiltinNameDef*>(any_name_def);
  return bnd->identifier() == "fail!";
}

static void WarnOnConditionalContainingJustFailStatement(
    const Conditional& node, DeduceCtx* ctx) {
  const StatementBlock* consequent = node.consequent();
  std::variant<StatementBlock*, Conditional*> alternate_ast_node =
      node.alternate();
  if (!std::holds_alternative<StatementBlock*>(alternate_ast_node)) {
    return;
  }
  const StatementBlock* alternate =
      std::get<StatementBlock*>(alternate_ast_node);

  if (IsBlockWithOneFailStmt(*consequent) && IsBlockJustUnitTuple(*alternate)) {
    std::string message = absl::StrFormat(
        "`if test { fail!(...) } else { () }` pattern should be replaced with "
        "`assert!(test, ...)`");
    ctx->warnings()->Add(node.span(), WarningKind::kShouldUseAssert,
                         std::move(message));
  }
}

absl::StatusOr<std::unique_ptr<Type>> DeduceConditional(const Conditional* node,
                                                        DeduceCtx* ctx) {
  XLS_ASSIGN_OR_RETURN(std::unique_ptr<Type> test_type,
                       ctx->Deduce(node->test()));
  XLS_ASSIGN_OR_RETURN(test_type, Resolve(*test_type, ctx));
  auto test_want = BitsType::MakeU1();
  if (*test_type != *test_want) {
    return ctx->TypeMismatchError(node->span(), node->test(), *test_type,
                                  nullptr, *test_want,
                                  "Test type for conditional expression is not "
                                  "\"bool\"");
  }

  XLS_ASSIGN_OR_RETURN(std::unique_ptr<Type> consequent_type,
                       ctx->Deduce(node->consequent()));
  XLS_ASSIGN_OR_RETURN(consequent_type, Resolve(*consequent_type, ctx));
  XLS_ASSIGN_OR_RETURN(std::unique_ptr<Type> alternate_type,
                       ctx->Deduce(ToAstNode(node->alternate())));
  XLS_ASSIGN_OR_RETURN(alternate_type, Resolve(*alternate_type, ctx));

  if (*consequent_type != *alternate_type) {
    return ctx->TypeMismatchError(
        node->span(), node->consequent(), *consequent_type,
        ToAstNode(node->alternate()), *alternate_type,
        "Conditional consequent type (in the 'then' clause) "
        "did not match alternative type (in the 'else' clause)");
  }

  WarnOnConditionalContainingJustFailStatement(*node, ctx);

  return consequent_type;
}

absl::StatusOr<std::unique_ptr<Type>> DeduceUnop(const Unop* node,
                                                 DeduceCtx* ctx) {
  XLS_ASSIGN_OR_RETURN(std::unique_ptr<Type> operand_type,
                       ctx->Deduce(node->operand()));

  if (dynamic_cast<BitsType*>(operand_type.get()) == nullptr) {
    return TypeInferenceErrorStatus(
        node->span(), operand_type.get(),
        absl::StrFormat(
            "Unary operation `%s` can only be applied to bits-typed operands.",
            UnopKindToString(node->unop_kind())));
  }

  return operand_type;
}

// Returns a set of the kinds of binary operations that it's ok to use on an
// enum value.
static const absl::flat_hash_set<BinopKind>& GetEnumOkKinds() {
  static const auto* set = []() {
    return new absl::flat_hash_set<BinopKind>{
        BinopKind::kEq,
        BinopKind::kNe,
    };
  }();
  return *set;
}

// Shift operations are binary operations that require bits types as their
// operands.
static absl::StatusOr<std::unique_ptr<Type>> DeduceShift(const Binop* node,
                                                         DeduceCtx* ctx) {
  XLS_ASSIGN_OR_RETURN(std::unique_ptr<Type> lhs,
                       DeduceAndResolve(node->lhs(), ctx));

  std::optional<uint64_t> number_value;
  if (auto* number = dynamic_cast<Number*>(node->rhs());
      number != nullptr && number->type_annotation() == nullptr) {
    // Infer RHS node as bit type and retrieve bit width.
    const std::string& number_str = number->text();
    XLS_RET_CHECK(!number_str.empty()) << "Number literal empty.";
    if (number_str[0] == '-') {
      return TypeInferenceErrorStatus(
          number->span(), nullptr,
          absl::StrFormat("Negative literal values cannot be used as shift "
                          "amounts; got: %s",
                          number_str));
    }
    XLS_ASSIGN_OR_RETURN(number_value, number->GetAsUint64());
    ctx->type_info()->SetItem(
        number, BitsType(/*is_signed=*/false,
                         Bits::MinBitCountUnsigned(number_value.value())));
  }

  XLS_ASSIGN_OR_RETURN(std::unique_ptr<Type> rhs,
                       DeduceAndResolve(node->rhs(), ctx));

  // Validate bits type for lhs and rhs.
  BitsType* lhs_bit_type = dynamic_cast<BitsType*>(lhs.get());
  if (lhs_bit_type == nullptr) {
    return TypeInferenceErrorStatus(
        node->lhs()->span(), lhs.get(),
        "Shift operations can only be applied to bits-typed operands.");
  }
  BitsType* rhs_bit_type = dynamic_cast<BitsType*>(rhs.get());
  if (rhs_bit_type == nullptr) {
    return TypeInferenceErrorStatus(
        node->rhs()->span(), rhs.get(),
        "Shift operations can only be applied to bits-typed operands.");
  }

  if (rhs_bit_type->is_signed()) {
    return TypeInferenceErrorStatus(node->rhs()->span(), rhs.get(),
                                    "Shift amount must be unsigned.");
  }

  if (number_value.has_value()) {
    const TypeDim& lhs_size = lhs_bit_type->size();
    CHECK(!lhs_size.IsParametric()) << "Shift amount type not inferred.";
    XLS_ASSIGN_OR_RETURN(int64_t lhs_bit_count, lhs_size.GetAsInt64());
    if (lhs_bit_count < number_value.value()) {
      return TypeInferenceErrorStatus(
          node->rhs()->span(), rhs.get(),
          absl::StrFormat(
              "Shift amount is larger than shift value bit width of %d.",
              lhs_bit_count));
    }
  }

  return lhs;
}

static absl::StatusOr<std::unique_ptr<Type>> DeduceConcat(const Binop* node,
                                                          DeduceCtx* ctx) {
  XLS_ASSIGN_OR_RETURN(std::unique_ptr<Type> lhs,
                       DeduceAndResolve(node->lhs(), ctx));
  XLS_ASSIGN_OR_RETURN(std::unique_ptr<Type> rhs,
                       DeduceAndResolve(node->rhs(), ctx));

  auto* lhs_array = dynamic_cast<ArrayType*>(lhs.get());
  auto* rhs_array = dynamic_cast<ArrayType*>(rhs.get());
  bool lhs_is_array = lhs_array != nullptr;
  bool rhs_is_array = rhs_array != nullptr;

  if (lhs_is_array != rhs_is_array) {
    return ctx->TypeMismatchError(node->span(), node->lhs(), *lhs, node->rhs(),
                                  *rhs,
                                  "Attempting to concatenate array/non-array "
                                  "values together.");
  }

  if (lhs_is_array && lhs_array->element_type() != rhs_array->element_type()) {
    return ctx->TypeMismatchError(
        node->span(), nullptr, *lhs, nullptr, *rhs,
        "Array concatenation requires element types to be the same.");
  }

  if (lhs_is_array) {
    XLS_ASSIGN_OR_RETURN(TypeDim new_size,
                         lhs_array->size().Add(rhs_array->size()));
    return std::make_unique<ArrayType>(
        lhs_array->element_type().CloneToUnique(), new_size);
  }

  auto* lhs_bits = dynamic_cast<BitsType*>(lhs.get());
  auto* rhs_bits = dynamic_cast<BitsType*>(rhs.get());
  bool lhs_is_bits = lhs_bits != nullptr;
  bool rhs_is_bits = rhs_bits != nullptr;
  if (!lhs_is_bits || !rhs_is_bits) {
    if (lhs->HasEnum() || rhs->HasEnum()) {
      return ctx->TypeMismatchError(
          node->span(), node->lhs(), *lhs, node->rhs(), *rhs,
          "Enum values must be cast to unsigned bits before concatenation.");
    }
    return ctx->TypeMismatchError(node->span(), node->lhs(), *lhs, node->rhs(),
                                  *rhs,
                                  "Concatenation requires operand types to be "
                                  "either both-arrays or both-bits");
  }

  if (lhs_bits->is_signed() || rhs_bits->is_signed()) {
    return ctx->TypeMismatchError(
        node->span(), node->lhs(), *lhs, node->rhs(), *rhs,
        "Concatenation requires operand types to both be "
        "unsigned bits");
  }

  XLS_RET_CHECK(lhs_bits != nullptr);
  XLS_RET_CHECK(rhs_bits != nullptr);
  XLS_ASSIGN_OR_RETURN(TypeDim new_size,
                       lhs_bits->size().Add(rhs_bits->size()));
  return std::make_unique<BitsType>(/*signed=*/false, /*size=*/new_size);
}

absl::StatusOr<std::unique_ptr<Type>> DeduceBinop(const Binop* node,
                                                  DeduceCtx* ctx) {
  if (node->binop_kind() == BinopKind::kConcat) {
    return DeduceConcat(node, ctx);
  }

  if (GetBinopShifts().contains(node->binop_kind())) {
    return DeduceShift(node, ctx);
  }

  XLS_ASSIGN_OR_RETURN(std::unique_ptr<Type> lhs,
                       DeduceAndResolve(node->lhs(), ctx));
  XLS_ASSIGN_OR_RETURN(std::unique_ptr<Type> rhs,
                       DeduceAndResolve(node->rhs(), ctx));

  if (*lhs != *rhs) {
    return ctx->TypeMismatchError(
        node->span(), node->lhs(), *lhs, node->rhs(), *rhs,
        absl::StrFormat("Could not deduce type for "
                        "binary operation '%s'",
                        BinopKindFormat(node->binop_kind())));
  }

  if (auto* enum_type = dynamic_cast<EnumType*>(lhs.get());
      enum_type != nullptr && !GetEnumOkKinds().contains(node->binop_kind())) {
    return TypeInferenceErrorStatus(
        node->span(), nullptr,
        absl::StrFormat("Cannot use '%s' on values with enum type %s.",
                        BinopKindFormat(node->binop_kind()),
                        enum_type->nominal_type().identifier()));
  }

  if (GetBinopComparisonKinds().contains(node->binop_kind())) {
    return BitsType::MakeU1();
  }

  if (GetBinopLogicalKinds().contains(node->binop_kind())) {
    auto u1 = BitsType::MakeU1();
    if (*lhs != *u1) {
      return ctx->TypeMismatchError(
          node->span(), node->lhs(), *lhs, nullptr, *u1,
          absl::StrFormat("Logical operation `%s` can only be applied to "
                          "`bool`/`u1` operands",
                          BinopKindFormat(node->binop_kind())));
    }
    return u1;
  }

  if (!IsBitsLike(*lhs)) {
    return TypeInferenceErrorStatus(
        node->span(), lhs.get(),
        "Binary operations can only be applied to bits-typed operands.");
  }

  return lhs;
}

absl::StatusOr<std::unique_ptr<Type>> DeduceTupleIndex(const TupleIndex* node,
                                                       DeduceCtx* ctx) {
  XLS_ASSIGN_OR_RETURN(std::unique_ptr<Type> lhs_type,
                       ctx->Deduce(node->lhs()));
  TupleType* tuple_type = dynamic_cast<TupleType*>(lhs_type.get());
  if (tuple_type == nullptr) {
    return TypeInferenceErrorStatus(
        node->span(), lhs_type.get(),
        absl::StrCat("Attempted to use tuple indexing on a non-tuple: ",
                     node->ToString()));
  }

  ctx->set_in_typeless_number_ctx(true);
  absl::Cleanup cleanup = [ctx]() { ctx->set_in_typeless_number_ctx(false); };
  XLS_ASSIGN_OR_RETURN(std::unique_ptr<Type> index_type,
                       ctx->Deduce(node->index()));
  std::move(cleanup).Cancel();

  // TupleIndex RHSs are always constexpr numbers.
  XLS_ASSIGN_OR_RETURN(
      InterpValue index_value,
      ConstexprEvaluator::EvaluateToValue(
          ctx->import_data(), ctx->type_info(), ctx->warnings(),
          ctx->GetCurrentParametricEnv(), node->index(), index_type.get()));
  XLS_ASSIGN_OR_RETURN(int64_t index, index_value.GetBitValueViaSign());
  if (index >= tuple_type->size()) {
    return TypeInferenceErrorStatus(
        node->span(), tuple_type,
        absl::StrCat("Out-of-bounds tuple index specified: ",
                     node->index()->ToString()));
  }

  return tuple_type->GetMemberType(index).CloneToUnique();
}

absl::StatusOr<std::unique_ptr<Type>> DeduceXlsTuple(const XlsTuple* node,
                                                     DeduceCtx* ctx) {
  // Give a warning if the tuple is on a single line, is more than one element,
  // but has a trailing comma.
  //
  // Note: warning diagnostics and type checking are currently fused together,
  // but this is a pure post-parsing warning -- currently type checking the pass
  // that has a warning collector available.
  if (node->span().start().lineno() == node->span().limit().lineno() &&
      node->members().size() > 1 && node->has_trailing_comma()) {
    std::string message = absl::StrFormat(
        "Tuple expression (with >1 element) is on a single "
        "line, but has a trailing comma.");
    ctx->warnings()->Add(node->span(),
                         WarningKind::kSingleLineTupleTrailingComma,
                         std::move(message));
  }

  std::vector<std::unique_ptr<Type>> members;
  for (Expr* e : node->members()) {
    XLS_ASSIGN_OR_RETURN(std::unique_ptr<Type> m, ctx->Deduce(e));
    members.push_back(std::move(m));
  }
  return std::make_unique<TupleType>(std::move(members));
}

}  // namespace xls::dslx
