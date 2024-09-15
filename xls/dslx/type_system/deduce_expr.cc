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
#include "xls/dslx/frontend/pos.h"
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

absl::StatusOr<std::unique_ptr<Type>> DeduceEmptyArray(const Array* node,
                                                       DeduceCtx* ctx) {
  // We cannot have an array that is just an ellipsis, ellipsis indicates we
  // should replicate the last member.
  if (node->has_ellipsis()) {
    return TypeInferenceErrorStatus(
        node->span(), nullptr,
        "Array cannot have an ellipsis without an element to repeat; please "
        "add at least one element",
        ctx->file_table());
  }

  if (node->type_annotation() == nullptr) {
    return TypeInferenceErrorStatus(
        node->span(), nullptr,
        "Empty array must have a type annotation; please add one",
        ctx->file_table());
  }

  // We need the type annotation because we don't have an element that allows
  // us to deduce the type of the empty array.
  XLS_ASSIGN_OR_RETURN(std::unique_ptr<Type> annotated,
                       ctx->Deduce(node->type_annotation()));
  XLS_ASSIGN_OR_RETURN(
      annotated,
      UnwrapMetaType(std::move(annotated), node->span(),
                     "array type-prefix position", ctx->file_table()));

  // Check that it's an array type of size zero.
  auto* array_type = dynamic_cast<ArrayType*>(annotated.get());
  if (array_type == nullptr) {
    return TypeInferenceErrorStatus(
        node->span(), annotated.get(),
        "Array was not annotated with an array type.", ctx->file_table());
  }
  const TypeDim& annotation_size = array_type->size();
  XLS_ASSIGN_OR_RETURN(int64_t annotation_size_i64,
                       annotation_size.GetAsInt64());
  if (annotation_size_i64 != 0) {
    return TypeInferenceErrorStatus(
        node->span(), array_type,
        absl::StrFormat(
            "Array has zero elements but type annotation size is %d",
            annotation_size_i64),
        ctx->file_table());
  }

  return annotated;
}

absl::StatusOr<std::unique_ptr<Type>> DeduceArray(const Array* node,
                                                  DeduceCtx* ctx) {
  VLOG(5) << "DeduceArray; node: " << node->ToString();

  if (node->members().empty()) {
    return DeduceEmptyArray(node, ctx);
  }

  std::vector<std::unique_ptr<Type>> member_types;
  for (Expr* member : node->members()) {
    XLS_ASSIGN_OR_RETURN(std::unique_ptr<Type> member_type,
                         DeduceAndResolve(member, ctx));
    member_types.push_back(std::move(member_type));
  }

  // Check that all subsequent member types are the same as the original member
  // type.
  for (int64_t i = 1; i < member_types.size(); ++i) {
    if (*member_types[0] != *member_types[i]) {
      return ctx->TypeMismatchError(
          node->span(), nullptr, *member_types[0], nullptr, *member_types[i],
          "Array member did not have same type as other members.");
    }
  }

  // Check that we're not making a token array, as tokens must not alias, and
  // arrays obscure their provenance as they are aggregate types.
  if (member_types[0]->HasToken()) {
    return TypeInferenceErrorStatus(
        node->span(), member_types[0].get(),
        "Types with tokens cannot be placed in arrays.", ctx->file_table());
  }

  auto member_types_dim =
      TypeDim::CreateU32(static_cast<uint32_t>(member_types.size()));

  // Try to infer the array type from the first member.
  std::unique_ptr<ArrayType> inferred = std::make_unique<ArrayType>(
      member_types[0]->CloneToUnique(), member_types_dim);

  if (node->type_annotation() == nullptr) {
    return inferred;
  }

  // The type annotation is present, see what it is.
  XLS_ASSIGN_OR_RETURN(std::unique_ptr<Type> annotated,
                       ctx->Deduce(node->type_annotation()));
  XLS_ASSIGN_OR_RETURN(
      annotated,
      UnwrapMetaType(std::move(annotated), node->span(),
                     "array type-prefix position", ctx->file_table()));
  VLOG(5) << absl::StreamFormat(
      "DeduceArray; inferred type annotation `%s` to be `%s`",
      node->type_annotation()->ToString(), annotated->ToString());

  // It must be an array!
  auto* array_type = dynamic_cast<ArrayType*>(annotated.get());
  if (array_type == nullptr) {
    return TypeInferenceErrorStatus(
        node->span(), annotated.get(),
        "Array was not annotated with an array type.", ctx->file_table());
  }

  if (array_type->HasParametricDims()) {
    return TypeInferenceErrorStatus(
        node->type_annotation()->span(), array_type,
        absl::StrFormat("Annotated type for array "
                        "literal must be constexpr; type has dimensions that "
                        "cannot be resolved."),
        ctx->file_table());
  }

  const TypeDim& annotation_size = array_type->size();

  // If we were presented with the wrong number of elements (vs what the
  // annotated type expected), flag an error.
  if (node->has_ellipsis()) {
    XLS_ASSIGN_OR_RETURN(int64_t annotation_size_i64,
                         annotation_size.GetAsInt64());
    // Check that there are <= members vs the annotated type if it's present.
    if (annotation_size_i64 < member_types.size()) {
      std::string message = absl::StrFormat(
          "Annotated array size %s is too small for observed array member "
          "count %d",
          array_type->size().ToString(), member_types.size());
      return ctx->TypeMismatchError(node->span(), nullptr, *array_type, nullptr,
                                    *inferred, message);
    }
  } else {  // No ellipsis.
    if (annotation_size != member_types_dim) {
      std::string message = absl::StrFormat(
          "Annotated array size %s does not match inferred array size %d.",
          array_type->size().ToString(), member_types.size());
      if (inferred == nullptr) {
        // No type to compare our expectation to, as there was no member to
        // infer the type from.
        return TypeInferenceErrorStatus(node->span(), array_type, message,
                                        ctx->file_table());
      }
      return ctx->TypeMismatchError(node->span(), nullptr, *array_type, nullptr,
                                    *inferred, message);
    }
  }

  // Check the element type of the annotation is the same as the inferred type
  // from the element(s).
  XLS_ASSIGN_OR_RETURN(std::unique_ptr<Type> resolved_element_type,
                       Resolve(array_type->element_type(), ctx));
  if (*resolved_element_type != *member_types[0]) {
    return ctx->TypeMismatchError(
        node->members().at(0)->span(), nullptr, *resolved_element_type, nullptr,
        *member_types[0],
        "Annotated element type did not match inferred "
        "element type.");
  }

  // In case of an ellipsis, the annotated type wins out.
  return annotated;
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
                           "array type-prefix position", ctx->file_table()));

  auto* array_type = dynamic_cast<ArrayType*>(type.get());
  if (array_type == nullptr) {
    return TypeInferenceErrorStatus(
        node->type_annotation()->span(), type.get(),
        absl::StrFormat("Annotated type for array "
                        "literal must be an array type; got %s %s",
                        type->GetDebugTypeName(),
                        node->type_annotation()->ToString()),
        ctx->file_table());
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
                            "applied to a literal number"),
            ctx->file_table());
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
                                      "this number, please annotate a type.",
                                      ctx->file_table());
    }
  } else {
    XLS_ASSIGN_OR_RETURN(type, ctx->Deduce(node->type_annotation()));
    XLS_ASSIGN_OR_RETURN(
        type, UnwrapMetaType(std::move(type), node->type_annotation()->span(),
                             "numeric literal type-prefix", ctx->file_table()));
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
        "Non-bits type used to define a numeric literal.", ctx->file_table());
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
            UnopKindToString(node->unop_kind())),
        ctx->file_table());
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
                          number_str),
          ctx->file_table());
    }
    const FileTable& file_table = ctx->file_table();
    XLS_ASSIGN_OR_RETURN(number_value, number->GetAsUint64(file_table));
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
        "Shift operations can only be applied to bits-typed operands.",
        ctx->file_table());
  }
  BitsType* rhs_bit_type = dynamic_cast<BitsType*>(rhs.get());
  if (rhs_bit_type == nullptr) {
    return TypeInferenceErrorStatus(
        node->rhs()->span(), rhs.get(),
        "Shift operations can only be applied to bits-typed operands.",
        ctx->file_table());
  }

  if (rhs_bit_type->is_signed()) {
    return TypeInferenceErrorStatus(node->rhs()->span(), rhs.get(),
                                    "Shift amount must be unsigned.",
                                    ctx->file_table());
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
              lhs_bit_count),
          ctx->file_table());
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

  std::optional<BitsLikeProperties> lhs_bits_like = GetBitsLike(*lhs);
  std::optional<BitsLikeProperties> rhs_bits_like = GetBitsLike(*rhs);
  if (lhs_bits_like.has_value() && rhs_bits_like.has_value()) {
    XLS_ASSIGN_OR_RETURN(bool lhs_is_signed, lhs_bits_like->is_signed.GetAsBool());
    XLS_ASSIGN_OR_RETURN(bool rhs_is_signed, rhs_bits_like->is_signed.GetAsBool());
    if (lhs_is_signed ||
        rhs_is_signed) {
      return TypeInferenceErrorStatus(
          node->span(), nullptr,
          absl::StrFormat("Concatenation requires operand types to both be "
          "unsigned bits; got lhs: `%s`; rhs: `%s`", lhs->ToString(), rhs->ToString()),
          ctx->file_table() );
    }
    XLS_ASSIGN_OR_RETURN(TypeDim new_size,
                         lhs_bits_like->size.Add(rhs_bits_like->size));
    return std::make_unique<BitsType>(/*signed=*/false, /*size=*/new_size);
  }

  auto* lhs_array = dynamic_cast<ArrayType*>(lhs.get());
  auto* rhs_array = dynamic_cast<ArrayType*>(rhs.get());
  bool lhs_is_array = lhs_array != nullptr && !lhs_bits_like.has_value();
  bool rhs_is_array = rhs_array != nullptr && !rhs_bits_like.has_value();

  if (lhs_is_array != rhs_is_array) {
    return TypeInferenceErrorStatus(node->span(), nullptr, absl::StrFormat(
                                  "Attempting to concatenate array/non-array "
                                  "values together; got lhs: `%s`; rhs: `%s`.",
          lhs->ToString(), rhs->ToString()), ctx->file_table());
  }

  if (lhs_is_array && lhs_array->element_type() != rhs_array->element_type()) {
    return ctx->TypeMismatchError(
        node->span(), nullptr, lhs_array->element_type(), nullptr, rhs_array->element_type(),
        "Array concatenation requires element types to be the same.");
  }

  if (lhs_is_array) {
    XLS_ASSIGN_OR_RETURN(TypeDim new_size,
                         lhs_array->size().Add(rhs_array->size()));
    return std::make_unique<ArrayType>(
        lhs_array->element_type().CloneToUnique(), new_size);
  }

  if (lhs->HasEnum() || rhs->HasEnum()) {
    return TypeInferenceErrorStatus(
        node->span(), nullptr,
        absl::StrFormat("Enum values must be cast to unsigned bits before concatenation; got lhs: `%s`; rhs: `%s`", lhs->ToString(), rhs->ToString()),
        ctx->file_table());
  }
  return TypeInferenceErrorStatus(node->span(), nullptr,
                                absl::StrFormat("Concatenation requires operand types to be "
                                "either both-arrays or both-bits; got lhs: `%s`; rhs: `%s`", lhs->ToString(), rhs->ToString()),
                        ctx->file_table());
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
                        enum_type->nominal_type().identifier()),
        ctx->file_table());
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
        "Binary operations can only be applied to bits-typed operands.",
        ctx->file_table());
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
                     node->ToString()),
        ctx->file_table());
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
                     node->index()->ToString()),
        ctx->file_table());
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
