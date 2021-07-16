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

#include "xls/dslx/deduce.h"

#include "absl/container/btree_set.h"
#include "absl/status/status.h"
#include "absl/strings/match.h"
#include "xls/dslx/ast.h"
#include "xls/dslx/interpreter.h"
#include "xls/dslx/scanner.h"

namespace xls::dslx {
namespace {

// Resolves "ref" to an AST function.
absl::StatusOr<Function*> ResolveColonRefToFn(ColonRef* ref, DeduceCtx* ctx) {
  absl::optional<Import*> import = ref->ResolveImportSubject();
  XLS_RET_CHECK(import.has_value())
      << "ColonRef did not refer to an import: " << ref->ToString();
  absl::optional<const ImportedInfo*> imported_info =
      ctx->type_info()->GetImported(*import);
  return imported_info.value()->module->GetFunctionOrError(ref->attr());
}

// If the width is known for "type", checks that "number" fits in that type.
absl::Status TryEnsureFitsInType(const Number& number,
                                 const ConcreteType& type) {
  XLS_ASSIGN_OR_RETURN(ConcreteTypeDim bits_dim, type.GetTotalBitCount());
  if (!absl::holds_alternative<InterpValue>(bits_dim.value())) {
    // We have to wait for the dimension to be fully resolved before we can
    // check that the number is compliant.
    return absl::OkStatus();
  }
  XLS_ASSIGN_OR_RETURN(int64_t bit_count, bits_dim.GetAsInt64());
  absl::StatusOr<Bits> bits = number.GetBits(bit_count);
  if (!bits.ok()) {
    return TypeInferenceErrorStatus(
        number.span(), &type,
        absl::StrFormat("Value '%s' does not fit in "
                        "the bitwidth of a %s (%d)",
                        number.text(), type.ToString(), bit_count));
  }
  return absl::OkStatus();
}

absl::StatusOr<std::unique_ptr<ConcreteType>> DeduceUnop(Unop* node,
                                                         DeduceCtx* ctx) {
  return ctx->Deduce(node->operand());
}

absl::StatusOr<std::unique_ptr<ConcreteType>> DeduceParam(Param* node,
                                                          DeduceCtx* ctx) {
  return ctx->Deduce(node->type_annotation());
}

absl::StatusOr<std::unique_ptr<ConcreteType>> DeduceConstantDef(
    ConstantDef* node, DeduceCtx* ctx) {
  XLS_VLOG(5) << "Noting constant: " << node->ToString();
  ctx->type_info()->NoteConstant(node->name_def(), node);
  XLS_ASSIGN_OR_RETURN(std::unique_ptr<ConcreteType> result,
                       ctx->Deduce(node->value()));
  // Right now we only know how to do integral expressions as constexpr values.
  if (dynamic_cast<BitsType*>(result.get()) != nullptr) {
    absl::flat_hash_map<std::string, InterpValue> env = MakeConstexprEnv(
        node->value(), ctx->fn_stack().back().symbolic_bindings(),
        ctx->type_info());
    const FnStackEntry& peek_entry = ctx->fn_stack().back();
    absl::optional<FnCtx> fn_ctx;
    if (peek_entry.function() != nullptr) {
      fn_ctx.emplace(FnCtx{peek_entry.module()->name(),
                           peek_entry.function()->identifier(),
                           peek_entry.symbolic_bindings()});
    }
    absl::StatusOr<InterpValue> constexpr_value = Interpreter::InterpretExpr(
        node->owner(), ctx->type_info(), ctx->typecheck_module(),
        ctx->additional_search_paths(), ctx->import_data(), env, node->value(),
        fn_ctx ? &*fn_ctx : nullptr);
    if (constexpr_value.ok()) {
      XLS_VLOG(5) << "Noting constexpr: " << node->value() << " in "
                  << ctx->type_info();
      ctx->type_info()->NoteConstExpr(node->value(), constexpr_value.value());
    }
  }
  ctx->type_info()->SetItem(node->name_def(), *result);
  return result;
}

absl::StatusOr<std::unique_ptr<ConcreteType>> DeduceTypeRef(TypeRef* node,
                                                            DeduceCtx* ctx) {
  return ctx->Deduce(ToAstNode(node->type_definition()));
}

absl::StatusOr<std::unique_ptr<ConcreteType>> DeduceTypeDef(TypeDef* node,
                                                            DeduceCtx* ctx) {
  XLS_ASSIGN_OR_RETURN(std::unique_ptr<ConcreteType> type,
                       ctx->Deduce(node->type_annotation()));
  ctx->type_info()->SetItem(node->name_def(), *type);
  return type;
}

absl::StatusOr<std::unique_ptr<ConcreteType>> DeduceXlsTuple(XlsTuple* node,
                                                             DeduceCtx* ctx) {
  std::vector<std::unique_ptr<ConcreteType>> members;
  for (Expr* e : node->members()) {
    XLS_ASSIGN_OR_RETURN(std::unique_ptr<ConcreteType> m, ctx->Deduce(e));
    members.push_back(std::move(m));
  }
  return absl::make_unique<TupleType>(std::move(members));
}

absl::StatusOr<std::unique_ptr<ConcreteType>> DeduceNumber(Number* node,
                                                           DeduceCtx* ctx) {
  if (node->type_annotation() == nullptr) {
    switch (node->kind()) {
      case NumberKind::kBool:
        return BitsType::MakeU1();
      case NumberKind::kCharacter:
        return BitsType::MakeU8();
      default:
        break;
    }
    return TypeInferenceErrorStatus(node->span(), nullptr,
                                    "Could not infer a type for "
                                    "this number, please annotate a type.");
  }

  XLS_ASSIGN_OR_RETURN(std::unique_ptr<ConcreteType> concrete_type,
                       ctx->Deduce(node->type_annotation()));
  XLS_ASSIGN_OR_RETURN(concrete_type, Resolve(*concrete_type, ctx));
  if (dynamic_cast<BitsType*>(concrete_type.get()) == nullptr) {
    return TypeInferenceErrorStatus(
        node->span(), concrete_type.get(),
        "Non-bits type used to define a numeric literal.");
  }
  XLS_RETURN_IF_ERROR(TryEnsureFitsInType(*node, *concrete_type));
  return concrete_type;
}

absl::StatusOr<std::unique_ptr<ConcreteType>> DeduceString(String* string,
                                                           DeduceCtx* ctx) {
  auto dim =
      ConcreteTypeDim::CreateU32(static_cast<int64_t>(string->text().size()));
  return absl::make_unique<ArrayType>(BitsType::MakeU8(), std::move(dim));
}

absl::StatusOr<std::unique_ptr<ConcreteType>> DeduceTernary(Ternary* node,
                                                            DeduceCtx* ctx) {
  XLS_ASSIGN_OR_RETURN(std::unique_ptr<ConcreteType> test_type,
                       ctx->Deduce(node->test()));
  XLS_ASSIGN_OR_RETURN(test_type, Resolve(*test_type, ctx));
  auto test_want = BitsType::MakeU1();
  if (*test_type != *test_want) {
    return XlsTypeErrorStatus(node->span(), *test_type, *test_want,
                              "Test type for conditional expression is not "
                              "\"bool\"");
  }

  XLS_ASSIGN_OR_RETURN(std::unique_ptr<ConcreteType> consequent_type,
                       ctx->Deduce(node->consequent()));
  XLS_ASSIGN_OR_RETURN(consequent_type, Resolve(*consequent_type, ctx));
  XLS_ASSIGN_OR_RETURN(std::unique_ptr<ConcreteType> alternate_type,
                       ctx->Deduce(node->alternate()));
  XLS_ASSIGN_OR_RETURN(alternate_type, Resolve(*alternate_type, ctx));

  if (*consequent_type != *alternate_type) {
    return XlsTypeErrorStatus(
        node->span(), *consequent_type, *alternate_type,
        "Ternary consequent type (in the 'then' clause) "
        "did not match alternative type (in the 'else' clause)");
  }
  return consequent_type;
}

static absl::StatusOr<std::unique_ptr<ConcreteType>> DeduceConcat(
    Binop* node, DeduceCtx* ctx) {
  XLS_ASSIGN_OR_RETURN(std::unique_ptr<ConcreteType> lhs,
                       DeduceAndResolve(node->lhs(), ctx));
  XLS_ASSIGN_OR_RETURN(std::unique_ptr<ConcreteType> rhs,
                       DeduceAndResolve(node->rhs(), ctx));

  auto* lhs_array = dynamic_cast<ArrayType*>(lhs.get());
  auto* rhs_array = dynamic_cast<ArrayType*>(rhs.get());
  bool lhs_is_array = lhs_array != nullptr;
  bool rhs_is_array = rhs_array != nullptr;

  if (lhs_is_array != rhs_is_array) {
    return XlsTypeErrorStatus(node->span(), *lhs, *rhs,
                              "Attempting to concatenate array/non-array "
                              "values together.");
  }

  if (lhs_is_array && lhs_array->element_type() != rhs_array->element_type()) {
    return XlsTypeErrorStatus(
        node->span(), *lhs, *rhs,
        "Array concatenation requires element types to be the same.");
  }

  if (lhs_is_array) {
    XLS_ASSIGN_OR_RETURN(ConcreteTypeDim new_size,
                         lhs_array->size().Add(rhs_array->size()));
    return absl::make_unique<ArrayType>(
        lhs_array->element_type().CloneToUnique(), new_size);
  }

  auto* lhs_bits = dynamic_cast<BitsType*>(lhs.get());
  auto* rhs_bits = dynamic_cast<BitsType*>(rhs.get());
  XLS_ASSIGN_OR_RETURN(ConcreteTypeDim new_size,
                       lhs_bits->size().Add(rhs_bits->size()));
  return absl::make_unique<BitsType>(/*signed=*/false, /*size=*/new_size);
}

// Shift operations are binary operations that require bits types as their
// operands.
static absl::StatusOr<std::unique_ptr<ConcreteType>> DeduceShift(
    Binop* node, DeduceCtx* ctx) {
  XLS_ASSIGN_OR_RETURN(std::unique_ptr<ConcreteType> lhs,
                       DeduceAndResolve(node->lhs(), ctx));

  absl::optional<uint64_t> number_value;
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

  XLS_ASSIGN_OR_RETURN(std::unique_ptr<ConcreteType> rhs,
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
    const ConcreteTypeDim& lhs_size = lhs_bit_type->size();
    XLS_CHECK(!lhs_size.IsParametric()) << "Shift amount type not inferred.";
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

// Returns a set of the kinds of binary operations that are comparisons; that
// is, they are `(T, T) -> bool` typed.
static const absl::flat_hash_set<BinopKind>& GetBinopComparisonKinds() {
  static const auto* set = [] {
    return new absl::flat_hash_set<BinopKind>{
        BinopKind::kEq, BinopKind::kNe, BinopKind::kGt,
        BinopKind::kGe, BinopKind::kLt, BinopKind::kLe,
    };
  }();
  return *set;
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

absl::StatusOr<std::unique_ptr<ConcreteType>> DeduceBinop(Binop* node,
                                                          DeduceCtx* ctx) {
  if (node->kind() == BinopKind::kConcat) {
    return DeduceConcat(node, ctx);
  }

  if (GetBinopShifts().contains(node->kind())) {
    return DeduceShift(node, ctx);
  }

  XLS_ASSIGN_OR_RETURN(std::unique_ptr<ConcreteType> lhs,
                       DeduceAndResolve(node->lhs(), ctx));
  XLS_ASSIGN_OR_RETURN(std::unique_ptr<ConcreteType> rhs,
                       DeduceAndResolve(node->rhs(), ctx));

  if (*lhs != *rhs) {
    return XlsTypeErrorStatus(node->span(), *lhs, *rhs,
                              absl::StrFormat("Could not deduce type for "
                                              "binary operation '%s'",
                                              BinopKindFormat(node->kind())));
  }

  if (auto* enum_type = dynamic_cast<EnumType*>(lhs.get());
      enum_type != nullptr && !GetEnumOkKinds().contains(node->kind())) {
    return TypeInferenceErrorStatus(
        node->span(), nullptr,
        absl::StrFormat("Cannot use '%s' on values with enum type %s.",
                        BinopKindFormat(node->kind()),
                        enum_type->nominal_type()->identifier()));
  }

  if (GetBinopComparisonKinds().contains(node->kind())) {
    return BitsType::MakeU1();
  }

  if (dynamic_cast<BitsType*>(lhs.get()) == nullptr) {
    return TypeInferenceErrorStatus(
        node->span(), lhs.get(),
        "Binary operations can only be applied to bits-typed operands.");
  }

  return lhs;
}

absl::StatusOr<std::unique_ptr<ConcreteType>> DeduceEnumDef(EnumDef* node,
                                                            DeduceCtx* ctx) {
  XLS_ASSIGN_OR_RETURN(std::unique_ptr<ConcreteType> type,
                       DeduceAndResolve(node->type_annotation(), ctx));
  auto* bits_type = dynamic_cast<BitsType*>(type.get());
  if (bits_type == nullptr) {
    return TypeInferenceErrorStatus(node->span(), bits_type,
                                    "Underlying type for an enum "
                                    "must be a bits type.");
  }

  // Grab the bit count of the Enum's underlying type.
  const ConcreteTypeDim& bit_count = bits_type->size();
  node->set_signedness(bits_type->is_signed());

  auto result = absl::make_unique<EnumType>(node, bit_count);
  for (const EnumMember& member : node->values()) {
    // Note: the parser places the type from the enum on the value when it is a
    // number, so this deduction flags inappropriate numbers.
    XLS_RETURN_IF_ERROR(ctx->Deduce(member.value).status());
    ctx->type_info()->SetItem(ToAstNode(member.value), *result);
    ctx->type_info()->SetItem(member.name_def, *result);
  }
  ctx->type_info()->SetItem(node->name_def(), *result);
  ctx->type_info()->SetItem(node, *result);
  return result;
}

// Typechecks the name def tree items against type, putting the corresponding
// type information for the AST nodes within the name_def_tree as corresponding
// to the types within "type" (recursively).
//
// For example:
//
//    (a, (b, c))  vs (u8, (u4, u2))
//
// Will put a correspondence of {a: u8, b: u4, c: u2} into the mapping in ctx.
static absl::Status BindNames(NameDefTree* name_def_tree,
                              const ConcreteType& type, DeduceCtx* ctx) {
  if (name_def_tree->is_leaf()) {
    AstNode* name_def = ToAstNode(name_def_tree->leaf());
    ctx->type_info()->SetItem(name_def, type);
    return absl::OkStatus();
  }

  auto* tuple_type = dynamic_cast<const TupleType*>(&type);
  if (tuple_type == nullptr) {
    return TypeInferenceErrorStatus(
        name_def_tree->span(), &type,
        absl::StrFormat("Expected a tuple type for these names, but "
                        "got %s.",
                        type.ToString()));
  }

  if (name_def_tree->nodes().size() != tuple_type->size()) {
    return TypeInferenceErrorStatus(
        name_def_tree->span(), &type,
        absl::StrFormat("Could not bind names, names are mismatched "
                        "in number vs type; at this level of the tuple: %d "
                        "names, %d types.",
                        name_def_tree->nodes().size(), tuple_type->size()));
  }

  for (int64_t i = 0; i < name_def_tree->nodes().size(); ++i) {
    NameDefTree* subtree = name_def_tree->nodes()[i];
    const ConcreteType& subtype = tuple_type->GetMemberType(i);
    ctx->type_info()->SetItem(subtree, subtype);
    XLS_RETURN_IF_ERROR(BindNames(subtree, subtype, ctx));
  }

  return absl::OkStatus();
}

absl::StatusOr<std::unique_ptr<ConcreteType>> DeduceLet(Let* node,
                                                        DeduceCtx* ctx) {
  XLS_ASSIGN_OR_RETURN(std::unique_ptr<ConcreteType> rhs,
                       DeduceAndResolve(node->rhs(), ctx));

  if (node->type_annotation() != nullptr) {
    XLS_ASSIGN_OR_RETURN(std::unique_ptr<ConcreteType> annotated,
                         DeduceAndResolve(node->type_annotation(), ctx));
    if (*rhs != *annotated) {
      return XlsTypeErrorStatus(node->type_annotation()->span(), *annotated,
                                *rhs,
                                "Annotated type did not match inferred type "
                                "of right hand side expression.");
    }
  }

  XLS_RETURN_IF_ERROR(BindNames(node->name_def_tree(), *rhs, ctx));

  if (node->constant_def() != nullptr) {
    XLS_RETURN_IF_ERROR(ctx->Deduce(node->constant_def()).status());
  }

  return ctx->Deduce(node->body());
}

absl::StatusOr<std::unique_ptr<ConcreteType>> DeduceFor(For* node,
                                                        DeduceCtx* ctx) {
  // Type of the init value to the for loop (also the accumulator type).
  XLS_ASSIGN_OR_RETURN(std::unique_ptr<ConcreteType> init_type,
                       DeduceAndResolve(node->init(), ctx));

  // Type of the iterable (whose elements are being used as the induction
  // variable in the for loop).
  XLS_ASSIGN_OR_RETURN(std::unique_ptr<ConcreteType> iterable_type,
                       DeduceAndResolve(node->iterable(), ctx));
  auto* iterable_array_type = dynamic_cast<ArrayType*>(iterable_type.get());
  if (iterable_array_type == nullptr) {
    return TypeInferenceErrorStatus(node->iterable()->span(),
                                    iterable_type.get(),
                                    "For loop iterable value is not an array.");
  }
  const ConcreteType& iterable_element_type =
      iterable_array_type->element_type();

  std::vector<std::unique_ptr<ConcreteType>> target_annotated_type_elems;
  target_annotated_type_elems.push_back(iterable_element_type.CloneToUnique());
  target_annotated_type_elems.push_back(init_type->CloneToUnique());
  auto target_annotated_type =
      absl::make_unique<TupleType>(std::move(target_annotated_type_elems));

  // If there was an explicitly annotated type, ensure it matches our inferred
  // one.
  if (node->type_annotation() != nullptr) {
    XLS_ASSIGN_OR_RETURN(std::unique_ptr<ConcreteType> annotated_type,
                         DeduceAndResolve(node->type_annotation(), ctx));

    if (*target_annotated_type != *annotated_type) {
      return XlsTypeErrorStatus(
          node->span(), *annotated_type, *target_annotated_type,
          "For-loop annotated type did not match inferred type.");
    }
  }

  // Bind the names to their associated types for use in the body.
  XLS_RETURN_IF_ERROR(BindNames(node->names(), *target_annotated_type, ctx));

  // Now we can deduce the body.
  XLS_ASSIGN_OR_RETURN(std::unique_ptr<ConcreteType> body_type,
                       DeduceAndResolve(node->body(), ctx));

  if (*init_type != *body_type) {
    return XlsTypeErrorStatus(node->span(), *init_type, *body_type,
                              "For-loop init value type did not match "
                              "for-loop body's result type.");
  }

  return init_type;
}

// Returns true if the cast-conversion from "from" to "to" is acceptable (i.e.
// should not cause a type error to occur).
static bool IsAcceptableCast(const ConcreteType& from, const ConcreteType& to) {
  auto is_array = [](const ConcreteType& ct) -> bool {
    return dynamic_cast<const ArrayType*>(&ct) != nullptr;
  };
  auto is_bits = [](const ConcreteType& ct) -> bool {
    return dynamic_cast<const BitsType*>(&ct) != nullptr;
  };
  auto is_enum = [](const ConcreteType& ct) -> bool {
    return dynamic_cast<const EnumType*>(&ct) != nullptr;
  };
  if ((is_array(from) && is_bits(to)) || (is_bits(from) && is_array(to))) {
    return from.GetTotalBitCount() == to.GetTotalBitCount();
  }
  if ((is_bits(from) || is_enum(from)) && is_bits(to)) {
    return true;
  }
  if (is_bits(from) && is_enum(to)) {
    return true;
  }
  return false;
}

absl::StatusOr<std::unique_ptr<ConcreteType>> DeduceCast(Cast* node,
                                                         DeduceCtx* ctx) {
  XLS_ASSIGN_OR_RETURN(std::unique_ptr<ConcreteType> type,
                       DeduceAndResolve(node->type_annotation(), ctx));
  XLS_ASSIGN_OR_RETURN(std::unique_ptr<ConcreteType> expr,
                       DeduceAndResolve(node->expr(), ctx));

  if (!IsAcceptableCast(/*from=*/*expr, /*to=*/*type)) {
    return XlsTypeErrorStatus(
        node->span(), *expr, *type,
        absl::StrFormat("Cannot cast from expression type %s to %s.",
                        expr->ToErrorString(), type->ToErrorString()));
  }
  return type;
}

absl::StatusOr<std::unique_ptr<ConcreteType>> DeduceStructDef(StructDef* node,
                                                              DeduceCtx* ctx) {
  for (const ParametricBinding* parametric : node->parametric_bindings()) {
    XLS_ASSIGN_OR_RETURN(std::unique_ptr<ConcreteType> parametric_binding_type,
                         ctx->Deduce(parametric->type_annotation()));
    if (parametric->expr() != nullptr) {
      XLS_ASSIGN_OR_RETURN(std::unique_ptr<ConcreteType> expr_type,
                           ctx->Deduce(parametric->expr()));
      if (*expr_type != *parametric_binding_type) {
        return XlsTypeErrorStatus(
            node->span(), *expr_type, *parametric_binding_type,
            "Annotated type of "
            "parametric value did not match inferred type.");
      }
    }
    ctx->type_info()->SetItem(parametric->name_def(), *parametric_binding_type);
  }

  std::vector<std::unique_ptr<ConcreteType>> members;
  for (auto [name_def, type] : node->members()) {
    XLS_ASSIGN_OR_RETURN(std::unique_ptr<ConcreteType> concrete,
                         DeduceAndResolve(type, ctx));
    members.push_back(std::move(concrete));
  }
  auto result = absl::make_unique<StructType>(std::move(members), node);
  ctx->type_info()->SetItem(node->name_def(), *result);
  XLS_VLOG(5) << absl::StreamFormat("Deduced type for struct %s => %s",
                                    node->ToString(), result->ToString());
  return result;
}

absl::StatusOr<std::unique_ptr<ConcreteType>> DeduceArray(Array* node,
                                                          DeduceCtx* ctx) {
  std::vector<std::unique_ptr<ConcreteType>> member_types;
  for (Expr* member : node->members()) {
    XLS_ASSIGN_OR_RETURN(std::unique_ptr<ConcreteType> member_type,
                         DeduceAndResolve(member, ctx));
    member_types.push_back(std::move(member_type));
  }

  for (int64_t i = 1; i < member_types.size(); ++i) {
    if (*member_types[0] != *member_types[i]) {
      return XlsTypeErrorStatus(
          node->span(), *member_types[0], *member_types[i],
          "Array member did not have same type as other members.");
    }
  }

  auto dim = ConcreteTypeDim::CreateU32(member_types.size());

  std::unique_ptr<ArrayType> inferred;
  if (!member_types.empty()) {
    inferred =
        absl::make_unique<ArrayType>(member_types[0]->CloneToUnique(), dim);
  }

  if (node->type_annotation() == nullptr) {
    if (inferred != nullptr) {
      return inferred;
    } else {
      return TypeInferenceErrorStatus(
          node->span(), nullptr, "Cannot deduce the type of an empty array.");
    }
  }

  XLS_ASSIGN_OR_RETURN(std::unique_ptr<ConcreteType> annotated,
                       ctx->Deduce(node->type_annotation()));
  auto* array_type = dynamic_cast<ArrayType*>(annotated.get());
  if (array_type == nullptr) {
    return TypeInferenceErrorStatus(
        node->span(), annotated.get(),
        "Array was not annotated with an array type.");
  }

  if (member_types.empty()) {
    return annotated;
  }

  XLS_ASSIGN_OR_RETURN(std::unique_ptr<ConcreteType> resolved_element_type,
                       Resolve(array_type->element_type(), ctx));
  if (*resolved_element_type != *member_types[0]) {
    return XlsTypeErrorStatus(node->span(), *resolved_element_type,
                              *member_types[0],
                              "Annotated element type did not match inferred "
                              "element type.");
  }

  if (node->has_ellipsis()) {
    return annotated;
  }

  if (array_type->size() != dim) {
    return XlsTypeErrorStatus(
        node->span(), *array_type, *inferred,
        absl::StrFormat("Annotated array size %s does not match "
                        "inferred array size %d.",
                        array_type->size().ToString(), member_types.size()));
  }

  return inferred;
}

absl::StatusOr<std::unique_ptr<ConcreteType>> DeduceAttr(Attr* node,
                                                         DeduceCtx* ctx) {
  XLS_ASSIGN_OR_RETURN(std::unique_ptr<ConcreteType> type,
                       ctx->Deduce(node->lhs()));
  auto* struct_type = dynamic_cast<StructType*>(type.get());
  if (struct_type == nullptr) {
    return TypeInferenceErrorStatus(node->span(), type.get(),
                                    absl::StrFormat("Expected a struct for "
                                                    "attribute access; got %s",
                                                    type->ToString()));
  }

  const std::string& attr_name = node->attr()->identifier();
  if (!struct_type->HasNamedMember(attr_name)) {
    return TypeInferenceErrorStatus(
        node->span(), nullptr,
        absl::StrFormat("Struct '%s' does not have a "
                        "member with name "
                        "'%s'",
                        struct_type->nominal_type().identifier(), attr_name));
  }

  absl::optional<const ConcreteType*> result =
      struct_type->GetMemberTypeByName(attr_name);
  XLS_RET_CHECK(result.has_value());  // We checked above we had named member.
  return result.value()->CloneToUnique();
}

absl::StatusOr<std::unique_ptr<ConcreteType>> DeduceConstantArray(
    ConstantArray* node, DeduceCtx* ctx) {
  if (node->type_annotation() == nullptr) {
    return DeduceArray(node, ctx);
  }

  XLS_ASSIGN_OR_RETURN(std::unique_ptr<ConcreteType> type,
                       ctx->Deduce(node->type_annotation()));
  auto* array_type = dynamic_cast<ArrayType*>(type.get());
  if (array_type == nullptr) {
    return TypeInferenceErrorStatus(
        node->type_annotation()->span(), type.get(),
        absl::StrFormat("Annotated type for array "
                        "literal must be an array type; got %s %s",
                        type->GetDebugTypeName(),
                        node->type_annotation()->ToString()));
  }

  const ConcreteType& element_type = array_type->element_type();
  for (Expr* member : node->members()) {
    XLS_RET_CHECK(IsConstant(member));
    if (Number* number = dynamic_cast<Number*>(member);
        number != nullptr && number->type_annotation() == nullptr) {
      ctx->type_info()->SetItem(member, element_type);
      XLS_RETURN_IF_ERROR(TryEnsureFitsInType(*number, element_type));
    }
  }

  XLS_RETURN_IF_ERROR(DeduceArray(node, ctx).status());
  return type;
}

static bool IsPublic(const ModuleMember& member) {
  if (absl::holds_alternative<Function*>(member)) {
    return absl::get<Function*>(member)->is_public();
  }
  if (absl::holds_alternative<TypeDef*>(member)) {
    return absl::get<TypeDef*>(member)->is_public();
  }
  if (absl::holds_alternative<StructDef*>(member)) {
    return absl::get<StructDef*>(member)->is_public();
  }
  if (absl::holds_alternative<ConstantDef*>(member)) {
    return absl::get<ConstantDef*>(member)->is_public();
  }
  if (absl::holds_alternative<EnumDef*>(member)) {
    return absl::get<EnumDef*>(member)->is_public();
  }
  if (absl::holds_alternative<TestFunction*>(member)) {
    return false;
  }
  if (absl::holds_alternative<QuickCheck*>(member)) {
    return false;
  }
  if (absl::holds_alternative<Import*>(member)) {
    return false;
  }
  XLS_LOG(FATAL) << "Unhandled ModuleMember variant.";
}

// Deduces a colon-ref in the particular case when the subject is known to be an
// import.
static absl::StatusOr<std::unique_ptr<ConcreteType>> DeduceColonRefImport(
    ColonRef* node, Import* import, DeduceCtx* ctx) {
  // Referring to something within an (imported) module.
  absl::optional<const ImportedInfo*> imported =
      ctx->type_info()->GetImported(import);
  XLS_RET_CHECK(imported.has_value());

  // Find the member being referred to within the imported module.
  Module* imported_module = (*imported)->module;
  absl::optional<ModuleMember*> elem =
      imported_module->FindMemberWithName(node->attr());
  if (!elem.has_value()) {
    return TypeInferenceErrorStatus(
        node->span(), nullptr,
        absl::StrFormat("Attempted to refer to module %s member '%s' "
                        "which does not exist.",
                        imported_module->name(), node->attr()));
  }
  if (!IsPublic(*elem.value())) {
    return TypeInferenceErrorStatus(
        node->span(), nullptr,
        absl::StrFormat("Attempted to refer to module member %s that "
                        "is not public.",
                        ToAstNode(*elem.value())->ToString()));
  }

  TypeInfo* imported_type_info = (*imported)->type_info;
  if (absl::holds_alternative<Function*>(*elem.value())) {
    auto* function = absl::get<Function*>(*elem.value());
    if (!imported_type_info->Contains(function->name_def())) {
      XLS_VLOG(2) << "Function name not in imported_type_info; indicates it is "
                     "parametric.";
      XLS_RET_CHECK(function->IsParametric());
      // We don't type check parametric functions until invocations.
      // Let's typecheck this imported parametric function with respect to its
      // module (this will only get the type signature, the body gets
      // typechecked after parametric instantiation).
      std::unique_ptr<DeduceCtx> imported_ctx =
          ctx->MakeCtx(imported_type_info, imported_module);
      const FnStackEntry& peek_entry = ctx->fn_stack().back();
      imported_ctx->fn_stack().push_back(peek_entry);
      XLS_RETURN_IF_ERROR(
          ctx->typecheck_function()(function, imported_ctx.get()));
      imported_type_info = imported_ctx->type_info();
    }
  }

  AstNode* member_node = ToAstNode(*elem.value());
  absl::optional<ConcreteType*> type = imported_type_info->GetItem(member_node);
  XLS_RET_CHECK(type.has_value()) << member_node->ToString();
  return type.value()->CloneToUnique();
}

absl::StatusOr<std::unique_ptr<ConcreteType>> DeduceColonRef(ColonRef* node,
                                                             DeduceCtx* ctx) {
  XLS_VLOG(5) << "Deducing type for ColonRef @ " << node->span().ToString();
  bool subject_is_name_ref = absl::holds_alternative<NameRef*>(node->subject());
  if (subject_is_name_ref) {
    NameRef* name_ref = absl::get<NameRef*>(node->subject());
    if (absl::holds_alternative<BuiltinNameDef*>(name_ref->name_def())) {
      auto* builtin_name_def = absl::get<BuiltinNameDef*>(name_ref->name_def());
      return TypeInferenceErrorStatus(
          node->span(), nullptr,
          absl::StrFormat("Builtin '%s' has no attributes.",
                          builtin_name_def->identifier()));
    }
    NameDef* name_def = absl::get<NameDef*>(name_ref->name_def());
    Import* import = dynamic_cast<Import*>(name_def->definer());
    if (import != nullptr) {
      return DeduceColonRefImport(node, import, ctx);
    }
  }

  XLS_ASSIGN_OR_RETURN(std::unique_ptr<ConcreteType> subject_type,
                       ctx->Deduce(ToAstNode(node->subject())));
  auto* enum_type = dynamic_cast<EnumType*>(subject_type.get());
  if (enum_type == nullptr) {
    return TypeInferenceErrorStatus(
        node->span(), subject_type.get(),
        absl::StrFormat("Cannot use '::' on '%s', it is not an enum or module",
                        ToAstNode(node->subject())->ToString()));
  }
  EnumDef* enum_def = enum_type->nominal_type();
  if (!enum_def->HasValue(node->attr())) {
    return TypeInferenceErrorStatus(
        node->span(), nullptr,
        absl::StrFormat("Name '%s' is not defined by the enum %s.",
                        node->attr(), enum_def->identifier()));
  }
  return subject_type;
}

// Deduces the concrete type for a tuple indexing operation.
static absl::StatusOr<std::unique_ptr<ConcreteType>> DeduceTupleIndex(
    Index* node, DeduceCtx* ctx, const TupleType& lhs_type) {
  IndexRhs rhs = node->rhs();
  auto* expr = absl::get<Expr*>(rhs);

  // TODO(leary): 2020-11-09 When we add unifying type inference this will also
  // be able to be a ConstRef.
  auto* number = dynamic_cast<Number*>(expr);
  if (number == nullptr) {
    return TypeInferenceErrorStatus(
        node->span(), &lhs_type,
        "Tuple index is not a literal number or named constant.");
  }

  if (number->type_annotation() == nullptr) {
    ctx->type_info()->SetItem(number, *BitsType::MakeU32());
  } else {
    // If the number has an annotated type, flag it as unnecessary.
    XLS_RETURN_IF_ERROR(ctx->Deduce(number).status());
    XLS_LOG(WARNING) << absl::StreamFormat(
        "Warning: type annotation for tuple index is unnecessary @ %s: %s",
        node->span().ToString(), node->ToString());
  }

  XLS_ASSIGN_OR_RETURN(int64_t value, number->GetAsUint64());
  if (value >= lhs_type.size()) {
    return TypeInferenceErrorStatus(
        node->span(), &lhs_type,
        absl::StrFormat("Tuple index %d is out of "
                        "range for this tuple type.",
                        value));
  }
  return lhs_type.GetMemberType(value).CloneToUnique();
}

// Returns (start, width), resolving indices via DSLX bit slice semantics.
static absl::StatusOr<StartAndWidth> ResolveBitSliceIndices(
    int64_t bit_count, absl::optional<int64_t> start_opt,
    absl::optional<int64_t> limit_opt) {
  XLS_RET_CHECK_GE(bit_count, 0);
  int64_t start = 0;
  int64_t limit = bit_count;

  if (start_opt.has_value()) {
    start = *start_opt;
  }
  if (limit_opt.has_value()) {
    limit = *limit_opt;
  }

  if (start < 0) {
    start += bit_count;
  }
  if (limit < 0) {
    limit += bit_count;
  }

  limit = std::min(std::max(limit, int64_t{0}), bit_count);
  start = std::min(std::max(start, int64_t{0}), limit);
  XLS_RET_CHECK_GE(start, 0);
  XLS_RET_CHECK_GE(limit, start);
  return StartAndWidth{start, limit - start};
}

static absl::StatusOr<std::unique_ptr<ConcreteType>> DeduceWidthSliceType(
    Index* node, const BitsType& subject_type, const WidthSlice& width_slice,
    DeduceCtx* ctx) {
  // Start expression; e.g. in `x[a+:u4]` this is `a`.
  Expr* start = width_slice.start();

  // Determined type of the start expression (must be bits kind).
  std::unique_ptr<ConcreteType> start_type_owned;
  BitsType* start_type;

  if (Number* start_number = dynamic_cast<Number*>(start);
      start_number != nullptr && start_number->type_annotation() == nullptr) {
    // A literal number with no annotated type as the slice start.
    //
    // By default, we use the "subject" type (converted to unsigned) as the type
    // for the slice start.
    start_type_owned = subject_type.ToUBits();
    start_type = dynamic_cast<BitsType*>(start_type_owned.get());

    // Get the start number as an integral value, after we make sure it fits.
    XLS_ASSIGN_OR_RETURN(Bits start_bits, start_number->GetBits(64));
    XLS_ASSIGN_OR_RETURN(int64_t start_int, start_bits.ToInt64());

    XLS_ASSIGN_OR_RETURN(std::unique_ptr<ConcreteType> resolved_start_type,
                         Resolve(*start_type, ctx));
    XLS_ASSIGN_OR_RETURN(ConcreteTypeDim bit_count_ctd,
                         resolved_start_type->GetTotalBitCount());
    XLS_ASSIGN_OR_RETURN(int64_t bit_count, bit_count_ctd.GetAsInt64());

    // Make sure the start_int literal fits in the type we determined.
    absl::Status fits_status = SBitsWithStatus(start_int, bit_count).status();
    if (!fits_status.ok()) {
      return TypeInferenceErrorStatus(
          node->span(), resolved_start_type.get(),
          absl::StrFormat("Cannot fit slice start %d in %d bits (width "
                          "inferred from slice subject).",
                          start_int, bit_count));
    }
    ctx->type_info()->SetItem(start, *start_type);
  } else {
    // Aside from a bare literal (with no type) we should be able to deduce the
    // start expression's type.
    XLS_ASSIGN_OR_RETURN(start_type_owned, ctx->Deduce(start));
    start_type = dynamic_cast<BitsType*>(start_type_owned.get());
    if (start_type == nullptr) {
      return TypeInferenceErrorStatus(
          start->span(), start_type,
          "Start expression for width slice must be bits typed.");
    }
  }

  // Validate that the start is unsigned.
  if (start_type->is_signed()) {
    return TypeInferenceErrorStatus(
        node->span(), start_type,
        "Start index for width-based slice must be unsigned.");
  }

  // If the width of the width_type is bigger than the subject, we flag an
  // error (prevent requesting over-slicing at compile time).
  XLS_ASSIGN_OR_RETURN(std::unique_ptr<ConcreteType> width_type,
                       ctx->Deduce(width_slice.width()));
  XLS_ASSIGN_OR_RETURN(ConcreteTypeDim width_ctd,
                       width_type->GetTotalBitCount());
  ConcreteTypeDim subject_ctd = subject_type.size();
  if (absl::holds_alternative<InterpValue>(width_ctd.value()) &&
      absl::holds_alternative<InterpValue>(subject_ctd.value())) {
    XLS_ASSIGN_OR_RETURN(int64_t width_bits, width_ctd.GetAsInt64());
    XLS_ASSIGN_OR_RETURN(int64_t subject_bits, subject_ctd.GetAsInt64());
    if (width_bits > subject_bits) {
      return XlsTypeErrorStatus(
          start->span(), subject_type, *width_type,
          absl::StrFormat("Slice type must have <= original number of bits; "
                          "attempted slice from %d to %d bits.",
                          subject_bits, width_bits));
    }
  }

  // Validate that the width type is bits-based (e.g. no enums, since sliced
  // value could be out of range of the valid enum values).
  if (dynamic_cast<BitsType*>(width_type.get()) == nullptr) {
    return TypeInferenceErrorStatus(
        node->span(), width_type.get(),
        "A bits type is required for a width-based slice.");
  }

  // The width type is the thing returned from the width-slice.
  return width_type;
}

// Attempts to resolve one of the bounds (start or limit) of slice into a
// DSLX-compile-time constant.
static absl::StatusOr<absl::optional<int64_t>> TryResolveBound(
    Slice* slice, Expr* bound, absl::string_view bound_name, ConcreteType* s32,
    const absl::flat_hash_map<std::string, InterpValue>& env, DeduceCtx* ctx) {
  if (bound == nullptr) {
    return absl::nullopt;
  }
  absl::StatusOr<InterpValue> bound_or = Interpreter::InterpretExpr(
      slice->owner(), ctx->type_info(), ctx->typecheck_module(),
      ctx->additional_search_paths(), ctx->import_data(), env, bound,
      /*fn_ctx=*/nullptr, s32);
  if (!bound_or.ok()) {
    const absl::Status& status = bound_or.status();
    if (absl::StrContains(status.message(), "Could not find bindings entry")) {
      return TypeInferenceErrorStatus(
          bound->span(), nullptr,
          absl::StrFormat(
              "Unable to resolve slice %s to a compile-time constant.",
              bound_name));
    }
  }
  const InterpValue& value = bound_or.value();
  if (value.tag() != InterpValueTag::kSBits) {  // Error if bound is not signed.
    std::string error_suffix = ".";
    if (value.tag() == InterpValueTag::kUBits) {
      error_suffix = " -- consider casting to a signed value?";
    }
    return TypeInferenceErrorStatus(
        bound->span(), nullptr,
        absl::StrFormat(
            "Slice %s must be a signed compile-time-constant value%s",
            bound_name, error_suffix));
  }

  XLS_ASSIGN_OR_RETURN(int64_t as_64b, value.GetBitValueInt64());
  XLS_VLOG(3) << absl::StreamFormat("Slice %s bound @ %s has value: %d",
                                    bound_name, bound->span().ToString(),
                                    as_64b);
  return as_64b;
}

// Deduces the concrete type for an Index AST node with a slice spec.
//
// Precondition: node->rhs() is either a Slice or a WidthSlice.
static absl::StatusOr<std::unique_ptr<ConcreteType>> DeduceSliceType(
    Index* node, DeduceCtx* ctx, std::unique_ptr<ConcreteType> lhs_type) {
  auto* bits_type = dynamic_cast<BitsType*>(lhs_type.get());
  if (bits_type == nullptr) {
    // TODO(leary): 2019-10-28 Only slicing bits types for now, and only with
    // Number AST nodes, generalize to arrays and constant expressions.
    return TypeInferenceErrorStatus(node->span(), lhs_type.get(),
                                    "Value to slice is not of 'bits' type.");
  }

  if (absl::holds_alternative<WidthSlice*>(node->rhs())) {
    auto* width_slice = absl::get<WidthSlice*>(node->rhs());
    return DeduceWidthSliceType(node, *bits_type, *width_slice, ctx);
  }

  const absl::flat_hash_map<std::string, InterpValue> env = MakeConstexprEnv(
      node, ctx->fn_stack().back().symbolic_bindings(), ctx->type_info());

  std::unique_ptr<ConcreteType> s32 = BitsType::MakeS32();
  auto* slice = absl::get<Slice*>(node->rhs());

  XLS_ASSIGN_OR_RETURN(
      absl::optional<int64_t> start,
      TryResolveBound(slice, slice->start(), "start", s32.get(), env, ctx));
  XLS_ASSIGN_OR_RETURN(
      absl::optional<int64_t> limit,
      TryResolveBound(slice, slice->limit(), "limit", s32.get(), env, ctx));

  const SymbolicBindings& fn_symbolic_bindings =
      ctx->fn_stack().back().symbolic_bindings();
  XLS_ASSIGN_OR_RETURN(ConcreteTypeDim lhs_bit_count_ctd,
                       lhs_type->GetTotalBitCount());
  int64_t bit_count;
  if (absl::holds_alternative<ConcreteTypeDim::OwnedParametric>(
          lhs_bit_count_ctd.value())) {
    auto& owned_parametric =
        absl::get<ConcreteTypeDim::OwnedParametric>(lhs_bit_count_ctd.value());
    ParametricExpression::Evaluated evaluated =
        owned_parametric->Evaluate(ToParametricEnv(fn_symbolic_bindings));
    InterpValue v = absl::get<InterpValue>(evaluated);
    bit_count = v.IsSigned() ? v.GetBitValueInt64().value()
                             : v.GetBitValueUint64().value();
  } else {
    XLS_ASSIGN_OR_RETURN(bit_count, lhs_bit_count_ctd.GetAsInt64());
  }
  XLS_ASSIGN_OR_RETURN(StartAndWidth saw,
                       ResolveBitSliceIndices(bit_count, start, limit));
  ctx->type_info()->AddSliceStartAndWidth(slice, fn_symbolic_bindings, saw);
  return absl::make_unique<BitsType>(/*signed=*/false, saw.width);
}

absl::StatusOr<std::unique_ptr<ConcreteType>> DeduceIndex(Index* node,
                                                          DeduceCtx* ctx) {
  XLS_ASSIGN_OR_RETURN(std::unique_ptr<ConcreteType> lhs_type,
                       ctx->Deduce(node->lhs()));

  if (absl::holds_alternative<Slice*>(node->rhs()) ||
      absl::holds_alternative<WidthSlice*>(node->rhs())) {
    return DeduceSliceType(node, ctx, std::move(lhs_type));
  }

  if (auto* tuple_type = dynamic_cast<TupleType*>(lhs_type.get())) {
    return DeduceTupleIndex(node, ctx, *tuple_type);
  }

  auto* array_type = dynamic_cast<ArrayType*>(lhs_type.get());
  if (array_type == nullptr) {
    return TypeInferenceErrorStatus(node->span(), lhs_type.get(),
                                    "Value to index is not an array.");
  }

  XLS_ASSIGN_OR_RETURN(std::unique_ptr<ConcreteType> index_type,
                       ctx->Deduce(ToAstNode(node->rhs())));
  XLS_RET_CHECK(index_type != nullptr);
  auto* index_bits = dynamic_cast<BitsType*>(index_type.get());
  if (index_bits == nullptr) {
    return TypeInferenceErrorStatus(node->span(), index_type.get(),
                                    "Index is not (scalar) bits typed.");
  }
  return array_type->element_type().CloneToUnique();
}

// Ensures that the name_def_tree bindings are aligned with the type "other"
// (which is the type for the matched value at this name_def_tree level).
static absl::Status Unify(NameDefTree* name_def_tree, const ConcreteType& other,
                          DeduceCtx* ctx) {
  XLS_ASSIGN_OR_RETURN(std::unique_ptr<ConcreteType> resolved_rhs_type,
                       Resolve(other, ctx));
  if (name_def_tree->is_leaf()) {
    NameDefTree::Leaf leaf = name_def_tree->leaf();
    if (absl::holds_alternative<NameDef*>(leaf)) {
      // Defining a name in the pattern match, we accept all types.
      ctx->type_info()->SetItem(ToAstNode(leaf), *resolved_rhs_type);
    } else if (absl::holds_alternative<WildcardPattern*>(leaf)) {
      // Nothing to do.
    } else if (absl::holds_alternative<Number*>(leaf) ||
               absl::holds_alternative<ColonRef*>(leaf)) {
      // For a reference (or literal) the types must be consistent.
      XLS_ASSIGN_OR_RETURN(std::unique_ptr<ConcreteType> resolved_leaf_type,
                           DeduceAndResolve(ToAstNode(leaf), ctx));
      if (*resolved_leaf_type != *resolved_rhs_type) {
        return XlsTypeErrorStatus(
            name_def_tree->span(), *resolved_rhs_type, *resolved_leaf_type,
            absl::StrFormat(
                "Conflicting types; pattern expects %s but got %s from value",
                resolved_rhs_type->ToString(), resolved_leaf_type->ToString()));
      }
    }
  } else {
    const NameDefTree::Nodes& nodes = name_def_tree->nodes();
    if (auto* type = dynamic_cast<const TupleType*>(&other);
        type != nullptr && type->size() == nodes.size()) {
      for (int64_t i = 0; i < nodes.size(); ++i) {
        const ConcreteType& subtype = type->GetMemberType(i);
        NameDefTree* subtree = nodes[i];
        XLS_RETURN_IF_ERROR(Unify(subtree, subtype, ctx));
      }
    }
  }
  return absl::OkStatus();
}

static std::string PatternsToString(MatchArm* arm) {
  return absl::StrJoin(arm->patterns(), " | ",
                       [](std::string* out, NameDefTree* ndt) {
                         absl::StrAppend(out, ndt->ToString());
                       });
}

absl::StatusOr<std::unique_ptr<ConcreteType>> DeduceMatch(Match* node,
                                                          DeduceCtx* ctx) {
  XLS_ASSIGN_OR_RETURN(std::unique_ptr<ConcreteType> matched,
                       ctx->Deduce(node->matched()));

  absl::flat_hash_set<std::string> seen_patterns;
  for (MatchArm* arm : node->arms()) {
    // We opportunistically identify syntactically identical match arms -- this
    // is a user error since the first should always match, the latter is
    // totally redundant.
    std::string patterns_string = PatternsToString(arm);
    if (auto [it, inserted] = seen_patterns.insert(patterns_string);
        !inserted) {
      return TypeInferenceErrorStatus(
          arm->GetPatternSpan(), nullptr,
          absl::StrFormat("Exact-duplicate pattern match detected `%s` -- only "
                          "the first could possibly match",
                          patterns_string));
    }

    for (NameDefTree* pattern : arm->patterns()) {
      XLS_RETURN_IF_ERROR(Unify(pattern, *matched, ctx));
    }
  }

  std::vector<std::unique_ptr<ConcreteType>> arm_types;
  for (MatchArm* arm : node->arms()) {
    XLS_ASSIGN_OR_RETURN(std::unique_ptr<ConcreteType> arm_type,
                         DeduceAndResolve(arm, ctx));
    arm_types.push_back(std::move(arm_type));
  }

  for (int64_t i = 1; i < arm_types.size(); ++i) {
    if (*arm_types[i] != *arm_types[0]) {
      return XlsTypeErrorStatus(
          node->arms()[i]->span(), *arm_types[i], *arm_types[0],
          "This match arm did not have the same type as the "
          "preceding match arms.");
    }
  }
  return std::move(arm_types[0]);
}

struct ValidatedStructMembers {
  // Names seen in the struct instance; e.g. for a SplatStructInstance can be a
  // subset of the struct member names.
  //
  // Note: we use a btree set so we can do set differencing via c_set_difference
  // (which works on ordered sets).
  absl::btree_set<std::string> seen_names;

  // Contains types that are deduced and must be owned, but are referred to by
  // reference in "args".
  std::vector<std::unique_ptr<ConcreteType>> owned_arg_types;

  std::vector<InstantiateArg> args;
  std::vector<std::unique_ptr<ConcreteType>> member_types;
};

// Validates a struct instantiation is a subset of 'members' with no dups.
//
// Args:
//  members: Sequence of members used in instantiation. Note this may be a
//    subset; e.g. in the case of splat instantiation.
//  struct_type: The deduced type for the struct (instantiation).
//  struct_text: Display name to use for the struct in case of an error.
//  ctx: Wrapper containing node to type mapping context.
//
// Returns:
//  A tuple containing:
//  * The set of struct member names that were instantiated
//  * The ConcreteTypes of the provided arguments
//  * The ConcreteTypes of the corresponding struct member definition.
static absl::StatusOr<ValidatedStructMembers> ValidateStructMembersSubset(
    absl::Span<const std::pair<std::string, Expr*>> members,
    const StructType& struct_type, absl::string_view struct_text,
    DeduceCtx* ctx) {
  ValidatedStructMembers result;
  for (auto& [name, expr] : members) {
    if (!result.seen_names.insert(name).second) {
      return TypeInferenceErrorStatus(
          expr->span(), nullptr,
          absl::StrFormat(
              "Duplicate value seen for '%s' in this '%s' struct instance.",
              name, struct_text));
    }
    XLS_ASSIGN_OR_RETURN(std::unique_ptr<ConcreteType> expr_type,
                         DeduceAndResolve(expr, ctx));
    result.owned_arg_types.push_back(std::move(expr_type));
    result.args.push_back(
        InstantiateArg{*result.owned_arg_types.back(), expr->span()});
    absl::optional<const ConcreteType*> maybe_type =
        struct_type.GetMemberTypeByName(name);
    if (maybe_type.has_value()) {
      result.member_types.push_back(maybe_type.value()->CloneToUnique());
    } else {
      return TypeInferenceErrorStatus(
          expr->span(), nullptr,
          absl::StrFormat("Struct '%s' has no member '%s', but it was provided "
                          "by this instance.",
                          struct_text, name));
    }
  }

  return result;
}

// Dereferences the "original" struct reference to a struct definition or
// returns an error.
//
// Args:
//  span: The span of the original construct trying to dereference the struct
//    (e.g. a StructInstance).
//  original: The original struct reference value (used in error reporting).
//  current: The current type definition being dereferenced towards a struct
//    definition (note there can be multiple levels of typedefs and such).
//  type_info: The type information that the "current" TypeDefinition resolves
//    against.
static absl::StatusOr<StructDef*> DerefToStruct(
    const Span& span, absl::string_view original_ref_text,
    TypeDefinition current, TypeInfo* type_info) {
  while (true) {
    if (absl::holds_alternative<StructDef*>(current)) {  // Done dereferencing.
      return absl::get<StructDef*>(current);
    }
    if (absl::holds_alternative<TypeDef*>(current)) {
      auto* type_def = absl::get<TypeDef*>(current);
      TypeAnnotation* annotation = type_def->type_annotation();
      TypeRefTypeAnnotation* type_ref =
          dynamic_cast<TypeRefTypeAnnotation*>(annotation);
      if (type_ref == nullptr) {
        return TypeInferenceErrorStatus(
            span, nullptr,
            absl::StrFormat("Could not resolve struct from %s; found: %s @ %s",
                            original_ref_text, annotation->ToString(),
                            annotation->span().ToString()));
      }
      current = type_ref->type_ref()->type_definition();
      continue;
    }
    if (absl::holds_alternative<ColonRef*>(current)) {
      auto* colon_ref = absl::get<ColonRef*>(current);
      // Colon ref has to be dereferenced, may be a module reference.
      ColonRef::Subject subject = colon_ref->subject();
      // TODO(leary): 2020-12-12 Original logic was this way, but we should be
      // able to violate this assertion.
      XLS_RET_CHECK(absl::holds_alternative<NameRef*>(subject));
      auto* name_ref = absl::get<NameRef*>(subject);
      AnyNameDef any_name_def = name_ref->name_def();
      XLS_RET_CHECK(absl::holds_alternative<NameDef*>(any_name_def));
      NameDef* name_def = absl::get<NameDef*>(any_name_def);
      AstNode* definer = name_def->definer();
      auto* import = dynamic_cast<Import*>(definer);
      if (import == nullptr) {
        return TypeInferenceErrorStatus(
            span, nullptr,
            absl::StrFormat("Could not resolve struct from %s; found: %s @ %s",
                            original_ref_text, name_ref->ToString(),
                            name_ref->span().ToString()));
      }
      absl::optional<const ImportedInfo*> imported =
          type_info->GetImported(import);
      XLS_RET_CHECK(imported.has_value());
      Module* module = imported.value()->module;
      XLS_ASSIGN_OR_RETURN(current,
                           module->GetTypeDefinition(colon_ref->attr()));
      return DerefToStruct(span, original_ref_text, current,
                           imported.value()->type_info);
    }
    XLS_RET_CHECK(absl::holds_alternative<EnumDef*>(current));
    auto* enum_def = absl::get<EnumDef*>(current);
    return TypeInferenceErrorStatus(
        span, nullptr,
        absl::StrFormat("Expected struct reference, but found enum: %s",
                        enum_def->identifier()));
  }
}

// Deduces the type for a ParametricBinding (via its type annotation).
static absl::StatusOr<std::unique_ptr<ConcreteType>> ParametricBindingToType(
    ParametricBinding* binding, DeduceCtx* ctx) {
  Module* binding_module = binding->owner();
  ImportData* import_data = ctx->import_data();
  XLS_ASSIGN_OR_RETURN(TypeInfo * binding_type_info,
                       import_data->GetRootTypeInfo(binding_module));
  auto binding_ctx = ctx->MakeCtx(binding_type_info, binding_module);
  return binding_ctx->Deduce(binding->type_annotation());
}

absl::StatusOr<std::unique_ptr<ConcreteType>> DeduceStructInstance(
    StructInstance* node, DeduceCtx* ctx) {
  XLS_VLOG(5) << "Deducing type for struct instance: " << node->ToString();

  XLS_ASSIGN_OR_RETURN(std::unique_ptr<ConcreteType> type,
                       ctx->Deduce(ToAstNode(node->struct_def())));
  auto* struct_type = dynamic_cast<StructType*>(type.get());
  XLS_RET_CHECK(struct_type != nullptr) << type->ToString();

  // TODO(leary): 2021-06-27 Grab parametric expressions off of the StructRef
  // and use those as "constraints" in InstantiateStruct() below.

  // Note what names we expect to be present.
  XLS_ASSIGN_OR_RETURN(std::vector<std::string> names,
                       struct_type->GetMemberNames());
  absl::btree_set<std::string> expected_names(names.begin(), names.end());

  XLS_ASSIGN_OR_RETURN(
      ValidatedStructMembers validated,
      ValidateStructMembersSubset(node->GetUnorderedMembers(), *struct_type,
                                  StructRefToText(node->struct_def()), ctx));
  if (validated.seen_names != expected_names) {
    absl::btree_set<std::string> missing_set;
    absl::c_set_difference(expected_names, validated.seen_names,
                           std::inserter(missing_set, missing_set.begin()));
    std::vector<std::string> missing(missing_set.begin(), missing_set.end());
    std::sort(missing.begin(), missing.end());
    return TypeInferenceErrorStatus(
        node->span(), nullptr,
        absl::StrFormat(
            "Struct instance is missing member(s): %s",
            absl::StrJoin(missing, ", ",
                          [](std::string* out, const std::string& piece) {
                            absl::StrAppendFormat(out, "'%s'", piece);
                          })));
  }

  StructRef struct_ref = node->struct_def();
  XLS_ASSIGN_OR_RETURN(
      StructDef * struct_def,
      DerefToStruct(node->span(), StructRefToText(struct_ref),
                    ToTypeDefinition(struct_ref), ctx->type_info()));

  XLS_ASSIGN_OR_RETURN(
      std::vector<ParametricConstraint> parametric_constraints,
      ParametricBindingsToConstraints(struct_def->parametric_bindings(), ctx));
  XLS_ASSIGN_OR_RETURN(
      TypeAndBindings tab,
      InstantiateStruct(node->span(), *struct_type, validated.args,
                        validated.member_types, ctx, parametric_constraints));

  return std::move(tab.type);
}

absl::StatusOr<std::unique_ptr<ConcreteType>> DeduceSplatStructInstance(
    SplatStructInstance* node, DeduceCtx* ctx) {
  XLS_ASSIGN_OR_RETURN(std::unique_ptr<ConcreteType> struct_type_ct,
                       ctx->Deduce(ToAstNode(node->struct_ref())));
  XLS_ASSIGN_OR_RETURN(std::unique_ptr<ConcreteType> splatted_type_ct,
                       ctx->Deduce(node->splatted()));

  // The splatted type should be (nominally) equivalent to the struct type,
  // because that's where we're filling in the default values from (those values
  // that were not directly provided by the user).
  auto* struct_type = dynamic_cast<StructType*>(struct_type_ct.get());
  auto* splatted_type = dynamic_cast<StructType*>(splatted_type_ct.get());

  // TODO(leary): 2020-12-13 Create a test case that hits this assertion, this
  // is a type error users can make; e.g. if they try to splat-instantiate a
  // tuple type alias.
  XLS_RET_CHECK(struct_type != nullptr);
  XLS_RET_CHECK(splatted_type != nullptr);
  if (&struct_type->nominal_type() != &splatted_type->nominal_type()) {
    return XlsTypeErrorStatus(
        node->span(), *struct_type, *splatted_type,
        absl::StrFormat("Attempting to fill values in '%s' instantiation from "
                        "a value of type '%s'",
                        struct_type->nominal_type().identifier(),
                        splatted_type->nominal_type().identifier()));
  }

  XLS_ASSIGN_OR_RETURN(
      ValidatedStructMembers validated,
      ValidateStructMembersSubset(node->members(), *struct_type,
                                  StructRefToText(node->struct_ref()), ctx));

  XLS_ASSIGN_OR_RETURN(std::vector<std::string> all_names,
                       struct_type->GetMemberNames());
  XLS_VLOG(5) << "SplatStructInstance @ " << node->span() << " seen names: ["
              << absl::StrJoin(validated.seen_names, ", ") << "] "
              << " all names: [" << absl::StrJoin(all_names, ", ") << "]";

  if (validated.seen_names.size() == all_names.size()) {
    XLS_LOG(WARNING) << "'Splatted' struct instance @ " << node->span()
                     << " has all members of struct defined.";
  }

  for (const std::string& name : all_names) {
    // If we didn't see the name, it comes from the "splatted" argument.
    if (!validated.seen_names.contains(name)) {
      const ConcreteType& splatted_member_type =
          *splatted_type->GetMemberTypeByName(name).value();
      const ConcreteType& struct_member_type =
          *struct_type->GetMemberTypeByName(name).value();

      validated.args.push_back(
          InstantiateArg{splatted_member_type, node->splatted()->span()});
      validated.member_types.push_back(struct_member_type.CloneToUnique());
    }
  }

  // At this point, we should have the same number of args compared to the
  // number of members defined in the struct.
  XLS_RET_CHECK_EQ(validated.args.size(), validated.member_types.size());

  StructRef struct_ref = node->struct_ref();
  XLS_ASSIGN_OR_RETURN(
      StructDef * struct_def,
      DerefToStruct(node->span(), StructRefToText(struct_ref),
                    ToTypeDefinition(struct_ref), ctx->type_info()));

  XLS_ASSIGN_OR_RETURN(
      std::vector<ParametricConstraint> parametric_constraints,
      ParametricBindingsToConstraints(struct_def->parametric_bindings(), ctx));
  XLS_ASSIGN_OR_RETURN(
      TypeAndBindings tab,
      InstantiateStruct(node->span(), *struct_type, validated.args,
                        validated.member_types, ctx, parametric_constraints));

  return std::move(tab.type);
}

absl::StatusOr<std::unique_ptr<ConcreteType>> DeduceBuiltinTypeAnnotation(
    BuiltinTypeAnnotation* node, DeduceCtx* ctx) {
  if (node->builtin_type() == BuiltinType::kToken) {
    return absl::make_unique<TokenType>();
  }
  return absl::make_unique<BitsType>(node->GetSignedness(),
                                     node->GetBitCount());
}

absl::StatusOr<std::unique_ptr<ConcreteType>> DeduceTupleTypeAnnotation(
    TupleTypeAnnotation* node, DeduceCtx* ctx) {
  std::vector<std::unique_ptr<ConcreteType>> members;
  for (TypeAnnotation* member : node->members()) {
    XLS_ASSIGN_OR_RETURN(std::unique_ptr<ConcreteType> type,
                         ctx->Deduce(member));
    members.push_back(std::move(type));
  }
  return absl::make_unique<TupleType>(std::move(members));
}

// Converts an AST expression in "dimension position" (e.g. in an array type
// annotation's size) and converts it into a ConcreteTypeDim value that can be
// used in a ConcreteStruct. The result is either a constexpr-evaluated value or
// a ParametricSymbol (for a parametric binding that has not yet been defined).
//
// Note: this is not capable of expressing more complex ASTs -- it assumes
// something is either fully constexpr-evaluatable, or symbolic.
static absl::StatusOr<ConcreteTypeDim> DimToConcrete(TypeAnnotation* node,
                                                     Expr* dim_expr,
                                                     DeduceCtx* ctx) {
  // We allow numbers in dimension position to go without type annotations -- we
  // implicitly make the type of the dimension u32, as we generally do with
  // dimension values.
  if (auto* number = dynamic_cast<Number*>(dim_expr)) {
    if (number->type_annotation() != nullptr) {
      return TypeInferenceErrorStatus(number->type_annotation()->span(),
                                      nullptr,
                                      "Please do not annotate a type on "
                                      "dimensions (they are implicitly u32).");
    }
    ctx->type_info()->SetItem(number, *BitsType::MakeU32());
    XLS_ASSIGN_OR_RETURN(int64_t value, number->GetAsUint64());
    return ConcreteTypeDim::CreateU32(value);
  }

  // If it's a name reference (and not a const reference), make it symbolic.
  //
  // TODO(leary): 2021-06-21 this is not a reasonable thing to assume. We need
  // to make something like a "ParametricRef" to reflect the fact there are
  // values that are constant when instantiated, but not fully constant (i.e.
  // before instantiation) and not runtime values. Right now if we refer to a
  // parameter in an array dimension it becomes a parametric symbol, which is no
  // good.
  //
  //    fn main(x: u32) -> () { u32[x]:[0, ...] }
  //                                ^-- becomes parametric symbol "x"
  if (auto* name_ref = dynamic_cast<NameRef*>(dim_expr);
      name_ref != nullptr && dynamic_cast<ConstRef*>(dim_expr) == nullptr) {
    return ConcreteTypeDim(absl::make_unique<ParametricSymbol>(
        name_ref->identifier(), dim_expr->span()));
  }

  XLS_ASSIGN_OR_RETURN(std::unique_ptr<ConcreteType> dim_type,
                       ctx->Deduce(dim_expr));
  auto u32_type = BitsType::MakeU32();
  if (*dim_type != *u32_type) {
    return XlsTypeErrorStatus(
        dim_expr->span(), *dim_type, *u32_type,
        absl::StrFormat(
            "Dimension %s must be a `u32` (soon to be `usize`, see "
            "https://github.com/google/xls/issues/450 for details).",
            dim_expr->ToString()));
  }

  absl::flat_hash_map<std::string, InterpValue> env = MakeConstexprEnv(
      dim_expr, ctx->fn_stack().back().symbolic_bindings(), ctx->type_info());
  XLS_RETURN_IF_ERROR(ctx->Deduce(dim_expr).status());
  absl::StatusOr<InterpValue> value_or = Interpreter::InterpretExpr(
      dim_expr->owner(),
      ctx->import_data()->GetRootTypeInfoForNode(dim_expr).value(),
      ctx->typecheck_module(), ctx->additional_search_paths(),
      ctx->import_data(), env, dim_expr);
  if (!value_or.ok()) {
    return TypeInferenceErrorStatus(
        dim_expr->span(), nullptr,
        absl::StrCat(
            "Could not evaluate dimension expression to a constant value: ",
            value_or.status().message()));
  }
  XLS_ASSIGN_OR_RETURN(uint64_t int_value, value_or->GetBitValueUint64());
  return ConcreteTypeDim::CreateU32(int_value);
}

absl::StatusOr<std::unique_ptr<ConcreteType>> DeduceArrayTypeAnnotation(
    ArrayTypeAnnotation* node, DeduceCtx* ctx) {
  XLS_VLOG(5) << "DeduceArrayTypeAnnotation; node: " << node->ToString();
  XLS_ASSIGN_OR_RETURN(ConcreteTypeDim dim,
                       DimToConcrete(node, node->dim(), ctx));
  if (auto* element_type =
          dynamic_cast<BuiltinTypeAnnotation*>(node->element_type());
      element_type != nullptr && element_type->GetBitCount() == 0) {
    return absl::make_unique<BitsType>(element_type->GetSignedness(),
                                       std::move(dim));
  }
  XLS_VLOG(5) << "DeduceArrayTypeAnnotation; element_type: "
              << node->element_type()->ToString();
  XLS_ASSIGN_OR_RETURN(std::unique_ptr<ConcreteType> element_type,
                       ctx->Deduce(node->element_type()));
  auto result =
      absl::make_unique<ArrayType>(std::move(element_type), std::move(dim));
  XLS_VLOG(4) << absl::StreamFormat("Array type annotation: %s => %s",
                                    node->ToString(), result->ToString());
  return result;
}

// Returns concretized struct type using the provided bindings.
//
// For example, if we have a struct defined as `struct Foo<N: u32, M: u32>`,
// the default TupleType will be (N, M). If a type annotation provides bindings,
// (e.g. `Foo<A, 16>`), we will replace N, M with those values. In the case
// above, we will return `(A, 16)` instead.
//
// Args:
//   type_annotation: The provided type annotation for this parametric struct.
//   struct_def: The struct definition AST node.
//   base_type: The TupleType of the struct, based only on the struct
//    definition (before parametrics are applied).
static absl::StatusOr<std::unique_ptr<ConcreteType>> ConcretizeStructAnnotation(
    TypeRefTypeAnnotation* type_annotation, StructDef* struct_def,
    const ConcreteType& base_type) {
  // Note: if there are too *few* annotated parametrics, some of them may be
  // derived.
  if (type_annotation->parametrics().size() >
      struct_def->parametric_bindings().size()) {
    return TypeInferenceErrorStatus(
        type_annotation->span(), &base_type,
        absl::StrFormat("Expected %d parametric arguments for '%s'; got %d in "
                        "type annotation",
                        struct_def->parametric_bindings().size(),
                        struct_def->identifier(),
                        type_annotation->parametrics().size()));
  }

  absl::flat_hash_map<
      std::string,
      absl::variant<int64_t, std::unique_ptr<ParametricExpression>>>
      defined_to_annotated;

  for (int64_t i = 0; i < type_annotation->parametrics().size(); ++i) {
    ParametricBinding* defined_parametric =
        struct_def->parametric_bindings()[i];
    Expr* annotated_parametric = type_annotation->parametrics()[i];
    // TODO(leary): 2020-12-13 This is kind of an ad hoc
    // constexpr-evaluate-to-int implementation, unify and consolidate it.
    if (auto* cast = dynamic_cast<Cast*>(annotated_parametric)) {
      Expr* expr = cast->expr();
      if (auto* number = dynamic_cast<Number*>(expr)) {
        XLS_ASSIGN_OR_RETURN(int64_t value, number->GetAsUint64());
        defined_to_annotated[defined_parametric->identifier()] = value;
      } else {
        auto* name_ref = dynamic_cast<NameRef*>(expr);
        XLS_RET_CHECK(name_ref != nullptr);
        defined_to_annotated[defined_parametric->identifier()] =
            absl::make_unique<ParametricSymbol>(name_ref->identifier(),
                                                name_ref->span());
      }
    } else if (auto* number = dynamic_cast<Number*>(annotated_parametric)) {
      XLS_ASSIGN_OR_RETURN(int value, number->GetAsUint64());
      defined_to_annotated[defined_parametric->identifier()] = value;
    } else {
      auto* name_ref = dynamic_cast<NameRef*>(annotated_parametric);
      XLS_RET_CHECK(name_ref != nullptr);
      defined_to_annotated[defined_parametric->identifier()] =
          absl::make_unique<ParametricSymbol>(name_ref->identifier(),
                                              name_ref->span());
    }
  }

  // For the remainder of the formal parameterics (i.e. after the explicitly
  // supplied ones given as arguments) we have to see if they're derived
  // parametrics. If they're *not* derived via an expression, we should have
  // been supplied some value in the annotation, so we have to flag an error!
  for (int64_t i = type_annotation->parametrics().size();
       i < struct_def->parametric_bindings().size(); ++i) {
    ParametricBinding* defined_parametric =
        struct_def->parametric_bindings()[i];
    if (defined_parametric->expr() == nullptr) {
      return TypeInferenceErrorStatus(
          type_annotation->span(), &base_type,
          absl::StrFormat("No parametric value provided for '%s' in '%s'",
                          defined_parametric->identifier(),
                          struct_def->identifier()));
    }
  }

  // Convert the defined_to_annotated map to use borrowed pointers for the
  // ParametricExpressions, as required by `ParametricExpression::Env` (so we
  // can `ParametricExpression::Evaluate()`).
  ParametricExpression::Env env;
  for (auto& item : defined_to_annotated) {
    if (absl::holds_alternative<int64_t>(item.second)) {
      env[item.first] = InterpValue::MakeU32(absl::get<int64_t>(item.second));
    } else {
      env[item.first] =
          absl::get<std::unique_ptr<ParametricExpression>>(item.second).get();
    }
  }

  // Now evaluate all the dimensions according to the values we've got.
  return base_type.MapSize(
      [&env](ConcreteTypeDim dim) -> absl::StatusOr<ConcreteTypeDim> {
        if (absl::holds_alternative<ConcreteTypeDim::OwnedParametric>(
                dim.value())) {
          auto& parametric =
              absl::get<ConcreteTypeDim::OwnedParametric>(dim.value());
          return ConcreteTypeDim(parametric->Evaluate(env));
        }
        return dim;
      });
}

absl::StatusOr<std::unique_ptr<ConcreteType>> DeduceTypeRefTypeAnnotation(
    TypeRefTypeAnnotation* node, DeduceCtx* ctx) {
  XLS_ASSIGN_OR_RETURN(std::unique_ptr<ConcreteType> base_type,
                       ctx->Deduce(node->type_ref()));
  TypeRef* type_ref = node->type_ref();
  TypeDefinition type_definition = type_ref->type_definition();
  absl::StatusOr<StructDef*> struct_def_or = DerefToStruct(
      node->span(), type_ref->ToString(), type_definition, ctx->type_info());
  if (struct_def_or.ok()) {
    auto* struct_def = struct_def_or.value();
    if (struct_def->IsParametric() && !node->parametrics().empty()) {
      XLS_ASSIGN_OR_RETURN(
          base_type, ConcretizeStructAnnotation(node, struct_def, *base_type));
    }
  }
  return base_type;
}

absl::StatusOr<std::unique_ptr<ConcreteType>> DeduceMatchArm(MatchArm* node,
                                                             DeduceCtx* ctx) {
  return ctx->Deduce(node->expr());
}

absl::StatusOr<std::unique_ptr<ConcreteType>> DeduceWhile(While* node,
                                                          DeduceCtx* ctx) {
  XLS_ASSIGN_OR_RETURN(std::unique_ptr<ConcreteType> init_type,
                       DeduceAndResolve(node->init(), ctx));
  XLS_ASSIGN_OR_RETURN(std::unique_ptr<ConcreteType> test_type,
                       DeduceAndResolve(node->test(), ctx));

  auto u1 = BitsType::MakeU1();
  if (*test_type != *u1) {
    return XlsTypeErrorStatus(node->test()->span(), *test_type, *u1,
                              "Expect while-loop test to be a bool value.");
  }

  XLS_ASSIGN_OR_RETURN(std::unique_ptr<ConcreteType> body_type,
                       DeduceAndResolve(node->body(), ctx));
  if (*init_type != *body_type) {
    return XlsTypeErrorStatus(
        node->span(), *init_type, *body_type,
        "While-loop init value did not match while-loop body's result type.");
  }
  return init_type;
}

absl::StatusOr<std::unique_ptr<ConcreteType>> DeduceCarry(Carry* node,
                                                          DeduceCtx* ctx) {
  return ctx->Deduce(node->loop()->init());
}

// Validates the parametric function body using the invocation's symbolic
// bindings.
//
// Args:
//  parametric_fn: The function being instantiated.
//  invocation: The invocation causing the instantiation.
//  symbolic_bindings: The caller's symbolic bindings (the caller being the
//    entity containing the invocation).
//  ctx: Deduction context.
static absl::Status ValidateParametricInvocation(
    Function* parametric_fn, Invocation* invocation,
    const SymbolicBindings& symbolic_bindings, DeduceCtx* ctx) {
  XLS_RET_CHECK(parametric_fn->IsParametric());
  XLS_VLOG(5) << "ValidateParametricInvocation; parametric_fn: "
              << parametric_fn->ToString()
              << "; invocation: " << invocation->ToString();
  if (auto* colon_ref = dynamic_cast<ColonRef*>(invocation->callee())) {
    // We need to typecheck this function with respect to its own module.

    if (ctx->type_info()->HasInstantiation(invocation, symbolic_bindings)) {
      return absl::OkStatus();
    }

    ColonRef::Subject subject = colon_ref->subject();
    // TODO(leary): 2020-12-14 Seems possible to violate this assertion? Attempt
    // to make a test that hits it.
    XLS_RET_CHECK(absl::holds_alternative<NameRef*>(subject));
    auto* subject_name_ref = absl::get<NameRef*>(subject);
    AstNode* definer =
        absl::get<NameDef*>(subject_name_ref->name_def())->definer();
    auto* import_node = dynamic_cast<Import*>(definer);
    XLS_RET_CHECK(import_node != nullptr);
    absl::optional<const ImportedInfo*> imported =
        ctx->type_info()->GetImported(import_node);
    XLS_RET_CHECK(imported.has_value());

    XLS_ASSIGN_OR_RETURN(
        TypeInfo * invocation_imported_type_info,
        ctx->type_info_owner().New((*imported)->module,
                                   /*parent=*/(*imported)->type_info));
    std::unique_ptr<DeduceCtx> imported_ctx =
        ctx->MakeCtx(invocation_imported_type_info, (*imported)->module);
    imported_ctx->fn_stack().push_back(
        FnStackEntry::Make(parametric_fn, symbolic_bindings));

    XLS_VLOG(5) << "Typechecking parametric function: "
                << parametric_fn->identifier() << " via " << symbolic_bindings;

    // Use typecheck_function callback to do this, in case we run into more
    // dependencies in that module.
    XLS_RETURN_IF_ERROR(
        ctx->typecheck_function()(parametric_fn, imported_ctx.get()));

    XLS_VLOG(5) << "TypeInfo::SetInstantiationTypeInfo; invocation: "
                << invocation->ToString()
                << " symbolic_bindings: " << symbolic_bindings;
    ctx->type_info()->SetInstantiationTypeInfo(
        invocation, /*caller=*/symbolic_bindings,
        /*type_info=*/invocation_imported_type_info);
    return absl::OkStatus();
  }

  auto* name_ref = dynamic_cast<NameRef*>(invocation->callee());
  if (ctx->type_info()->HasInstantiation(invocation, symbolic_bindings)) {
    // If we've already typechecked the parametric function with the current
    // symbolic bindings, no need to do it again.
    return absl::OkStatus();
  }

  if (!ctx->type_info()->Contains(parametric_fn->body())) {
    // Typecheck this parametric function using the symbolic bindings we just
    // derived to make sure they check out ok.
    AstNode* type_missing_error_node = ToAstNode(name_ref->name_def());
    ctx->fn_stack().push_back(
        FnStackEntry::Make(parametric_fn, symbolic_bindings));
    ctx->AddDerivedTypeInfo();
    XLS_VLOG(5) << "Throwing to typecheck parametric function: "
                << parametric_fn->identifier() << " via " << symbolic_bindings
                << " in child type info: " << ctx->type_info()
                << "; parent: " << ctx->type_info()->parent();
    return TypeMissingErrorStatus(type_missing_error_node, nullptr);
  }

  // If we haven't yet stored a type_info for these symbolic bindings and
  // we're at this point, it means we've just finished typechecking the
  // parametric function. Let's store the results.
  XLS_VLOG(5) << "TypeInfo::SetInstantiationTypeInfo; "
              << ctx->type_info()->parent()
              << "; invocation: " << invocation->ToString()
              << "; symbolic_bindings: " << symbolic_bindings
              << "; instantiated: " << ctx->type_info();
  ctx->type_info()->parent()->SetInstantiationTypeInfo(
      invocation, symbolic_bindings, ctx->type_info());
  XLS_RETURN_IF_ERROR(ctx->PopDerivedTypeInfo());
  return absl::OkStatus();
}

// Creates a function invocation on the first element of the given array.
//
// We need to create a fake invocation to deduce the type of a function
// in the case where map is called with a builtin as the map function. Normally,
// map functions (including parametric ones) have their types deduced when their
// ast.Function nodes are encountered (where a similar fake ast.Invocation node
// is created).
//
// Builtins don't have ast.Function nodes, so that inference can't occur, so we
// essentually perform that synthesis and deduction here.
//
// Args:
//   module: AST node owner.
//   span_: The location in the code where analysis is occurring.
//   callee: The function to be invoked.
//   arg_array: The array of arguments (at least one) to the function.
//
// Returns:
//   An invocation node for the given function when called with an element in
//   the argument array.
static Invocation* CreateElementInvocation(Module* module, const Span& span,
                                           NameRef* callee, Expr* arg_array) {
  auto* annotation =
      module->Make<BuiltinTypeAnnotation>(span, BuiltinType::kU32);
  auto* index_number =
      module->Make<Number>(span, "32", NumberKind::kOther, annotation);
  auto* index = module->Make<Index>(span, arg_array, index_number);
  return module->Make<Invocation>(span, callee, std::vector<Expr*>{index});
}

// Returns the names of all the builtin functions that are parametric (as a
// singleton set).
static const absl::flat_hash_set<std::string>& GetParametricBuiltinNames() {
  static const auto* set = new absl::flat_hash_set<std::string>{
      "add_with_carry",
      "assert_eq",
      "assert_lt",
      "bit_slice",
      "bit_slice_update",
      "clz",
      "cover!",
      "ctz",
      "concat",
      "fail!",
      "map",
      "one_hot",
      "one_hot_sel",
      "rev",
      "select",
      // Bitwise reduction ops.
      "and_reduce",
      "or_reduce",
      "xor_reduce",
      // Use a dummy value to determine size.
      "signex",
      "slice",
      "trace!",
      "update",
      "enumerate",
      // Require-const-argument.
      "range",
  };
  return *set;
}

// Updates the "user" field of the given TypeMissingError status, and returns
// the new (updated) status.
static absl::Status TypeMissingErrorStatusUpdateUser(const absl::Status& status,
                                                     const Span& span,
                                                     AstNode* new_user) {
  auto [node, user] = ParseTypeMissingErrorMessage(status.message());
  auto result = TypeMissingErrorStatus(node, new_user);
  XLS_VLOG(5) << "Updated from " << status << " to " << result;
  return result;
}

absl::StatusOr<std::unique_ptr<ConcreteType>> DeduceInvocation(Invocation* node,
                                                               DeduceCtx* ctx) {
  const SymbolicBindings& caller_symbolic_bindings =
      ctx->fn_stack().back().symbolic_bindings();
  XLS_VLOG(5) << "Deducing type for invocation: " << node->ToString()
              << " caller symbolic bindings: " << caller_symbolic_bindings;

  // Gather up the type of all the (actual) arguments.
  std::vector<std::unique_ptr<ConcreteType>> owned_arg_types;
  std::vector<InstantiateArg> args;
  for (Expr* arg : node->args()) {
    absl::StatusOr<std::unique_ptr<ConcreteType>> type =
        DeduceAndResolve(arg, ctx);
    if (IsTypeMissingErrorStatus(type.status())) {
      auto* callee_name_ref = dynamic_cast<NameRef*>(node->callee());
      auto* arg_name_ref = dynamic_cast<NameRef*>(arg);
      // TODO(leary): 2020-12-13 This is not very general, but right now there's
      // not a way to write a custom higher order function (e.g. that wraps
      // 'map'), so it may be reasonably workable.
      bool callee_is_map =
          callee_name_ref != nullptr && callee_name_ref->identifier() == "map";
      bool arg_is_builtin_parametric =
          arg_name_ref != nullptr &&
          absl::holds_alternative<BuiltinNameDef*>(arg_name_ref->name_def()) &&
          GetParametricBuiltinNames().contains(arg_name_ref->identifier());
      if (callee_is_map && arg_is_builtin_parametric) {
        Invocation* invocation = CreateElementInvocation(
            ctx->module(), node->span(), arg_name_ref, node->args()[0]);
        XLS_ASSIGN_OR_RETURN(std::unique_ptr<ConcreteType> invocation_type,
                             DeduceAndResolve(invocation, ctx));
        owned_arg_types.push_back(std::move(invocation_type));
      } else {
        return type.status();
      }
    } else {
      XLS_RETURN_IF_ERROR(type.status());
      owned_arg_types.push_back(std::move(type).value());
    }
    args.push_back(InstantiateArg{*owned_arg_types.back(), arg->span()});
  }

  // This will get us the type signature of the function. If the function is
  // parametric, we won't check its body until after we have symbolic bindings
  // for it.
  absl::StatusOr<std::unique_ptr<ConcreteType>> callee_type_or =
      ctx->Deduce(node->callee());
  if (IsTypeMissingErrorStatus(callee_type_or.status())) {
    XLS_VLOG(5) << "TypeMissingError status: " << callee_type_or.status();
    return TypeMissingErrorStatusUpdateUser(callee_type_or.status(),
                                            node->span(), node);
  }
  XLS_RETURN_IF_ERROR(callee_type_or.status());

  FunctionType* callee_type =
      dynamic_cast<FunctionType*>(callee_type_or.value().get());
  if (callee_type == nullptr) {
    return TypeInferenceErrorStatus(node->callee()->span(), callee_type,
                                    "Callee does not have a function type.");
  }

  // Find the callee as a DSLX function from the expression.
  Expr* callee = node->callee();
  Function* callee_fn;
  if (auto* colon_ref = dynamic_cast<ColonRef*>(callee)) {
    XLS_ASSIGN_OR_RETURN(callee_fn, ResolveColonRefToFn(colon_ref, ctx));
  } else {
    auto* name_ref = dynamic_cast<NameRef*>(callee);
    XLS_RET_CHECK(name_ref != nullptr);
    const std::string& callee_name = name_ref->identifier();
    XLS_ASSIGN_OR_RETURN(callee_fn,
                         ctx->module()->GetFunctionOrError(callee_name));
  }

  // We need to deduce the type of all Invocation parametrics (the expressions
  // in the instantiation brackets <>) so they're in the type cache.
  for (Expr* parametric : node->parametrics()) {
    XLS_RETURN_IF_ERROR(ctx->Deduce(parametric).status());
  }

  // The number of given parametric expressions always has to be <= the
  // parametric bindings count.
  if (node->parametrics().size() > callee_fn->parametric_bindings().size()) {
    return ArgCountMismatchErrorStatus(
        node->span(),
        absl::StrFormat(
            "Too many parametric values supplied; limit: %d given: %d",
            callee_fn->parametric_bindings().size(),
            node->parametrics().size()));
  }

  // Create "ParametricConstraint" records that reflect the parametrics
  // specified by the caller invocation.
  std::vector<ParametricConstraint> parametric_constraints;
  parametric_constraints.reserve(callee_fn->parametric_bindings().size());
  for (int64_t i = 0; i < node->parametrics().size(); ++i) {
    ParametricBinding* binding = callee_fn->parametric_bindings()[i];
    Expr* value = node->parametrics()[i];

    XLS_VLOG(5) << "Populating callee parametric `" << binding->ToString()
                << "` via invocation expression: " << value->ToString();

    XLS_ASSIGN_OR_RETURN(std::unique_ptr<ConcreteType> binding_type,
                         ParametricBindingToType(binding, ctx));
    XLS_ASSIGN_OR_RETURN(std::unique_ptr<ConcreteType> value_type,
                         ctx->Deduce(value));

    if (*binding_type != *value_type) {
      return XlsTypeErrorStatus(node->callee()->span(), *binding_type,
                                *value_type,
                                "Explicit parametric type mismatch.");
    }
    parametric_constraints.push_back(
        ParametricConstraint(*binding, std::move(binding_type), value));
  }

  // The bindings that were not explicitly filled by the caller are taken from
  // the callee directly; e.g. if caller invokes as `parametric()` it supplies
  // 0 parmetrics directly, but callee may have:
  //
  //    `fn parametric<N: u32 = 5>() { ... }`
  //
  // and thus needs the `N: u32 = 5` to be filled here.
  for (ParametricBinding* remaining_binding :
       absl::MakeSpan(callee_fn->parametric_bindings())
           .subspan(node->parametrics().size())) {
    XLS_ASSIGN_OR_RETURN(std::unique_ptr<ConcreteType> binding_type,
                         ParametricBindingToType(remaining_binding, ctx));
    parametric_constraints.push_back(
        ParametricConstraint(*remaining_binding, std::move(binding_type)));
  }

  absl::flat_hash_map<std::string, InterpValue> caller_symbolic_bindings_map =
      caller_symbolic_bindings.ToMap();

  // Map resolved parametrics from the caller's context onto the corresponding
  // symbols in the callee's.
  absl::flat_hash_map<std::string, InterpValue> explicit_bindings;
  for (const ParametricConstraint& constraint : parametric_constraints) {
    if (auto* name_ref = dynamic_cast<NameRef*>(constraint.expr());
        name_ref != nullptr &&
        caller_symbolic_bindings_map.contains(name_ref->identifier())) {
      explicit_bindings.insert(
          {constraint.identifier(),
           caller_symbolic_bindings_map.at(name_ref->identifier())});
    }
  }

  XLS_ASSIGN_OR_RETURN(
      TypeAndBindings tab,
      InstantiateFunction(node->span(), *callee_type, args, ctx,
                          /*parametric_constraints=*/parametric_constraints,
                          /*explicit_constraints=*/&explicit_bindings));
  const SymbolicBindings& callee_symbolic_bindings = tab.symbolic_bindings;

  if (callee_fn->IsParametric()) {
    // Make a note that this invocation node transitions us from these caller to
    // these callee invocation bindings.
    XLS_VLOG(5) << "TypeInfo::AddInstantiationCallBindings; type_info: "
                << ctx->type_info() << "; node: `" << node->ToString()
                << "`; caller: " << caller_symbolic_bindings
                << "; callee: " << callee_symbolic_bindings;
    ctx->type_info()->AddInstantiationCallBindings(
        node, /*caller=*/caller_symbolic_bindings,
        /*callee=*/callee_symbolic_bindings);

    // Now that we have callee_symbolic_bindings, let's use them to typecheck
    // the body of callee_fn to make sure these values actually work.
    XLS_RETURN_IF_ERROR(ValidateParametricInvocation(
        callee_fn, node, callee_symbolic_bindings, ctx));
  }

  // If the callee function needs an implicit token type (e.g. because it has a
  // fail!() or cover!() operation transitively) then so do we.
  if (absl::optional<bool> callee_opt =
          ctx->import_data()
              ->GetRootTypeInfoForNode(callee_fn)
              .value()
              ->GetRequiresImplicitToken(callee_fn);
      callee_opt.value()) {
    Function* caller_fn = ctx->fn_stack().back().function();
    // Note: caller_fn could be nullptr; e.g. when we're calling a function that
    // can fail!() from the top level of a module; e.g. in a module-level const
    // expression.
    if (caller_fn != nullptr) {
      ctx->type_info()->NoteRequiresImplicitToken(caller_fn, true);
    }
  }

  return std::move(tab.type);
}

absl::StatusOr<std::unique_ptr<ConcreteType>> DeduceNameRef(NameRef* node,
                                                            DeduceCtx* ctx) {
  AstNode* name_def = ToAstNode(node->name_def());
  absl::optional<ConcreteType*> item = ctx->type_info()->GetItem(name_def);
  if (item.has_value()) {
    return (*item)->CloneToUnique();
  }
  return TypeMissingErrorStatus(name_def, node);
}

absl::StatusOr<std::unique_ptr<ConcreteType>> DeduceConstRef(ConstRef* node,
                                                             DeduceCtx* ctx) {
  // ConstRef is a subtype of NameRef, same deduction rule works.
  return DeduceNameRef(node, ctx);
}

class DeduceVisitor : public AstNodeVisitor {
 public:
  explicit DeduceVisitor(DeduceCtx* ctx) : ctx_(ctx) {}

#define DEDUCE_DISPATCH(__type)                     \
  absl::Status Handle##__type(__type* n) override { \
    result_ = Deduce##__type(n, ctx_);              \
    return result_.status();                        \
  }

  DEDUCE_DISPATCH(Unop)
  DEDUCE_DISPATCH(Param)
  DEDUCE_DISPATCH(ConstantDef)
  DEDUCE_DISPATCH(Number)
  DEDUCE_DISPATCH(String)
  DEDUCE_DISPATCH(TypeRef)
  DEDUCE_DISPATCH(TypeDef)
  DEDUCE_DISPATCH(XlsTuple)
  DEDUCE_DISPATCH(Ternary)
  DEDUCE_DISPATCH(Binop)
  DEDUCE_DISPATCH(EnumDef)
  DEDUCE_DISPATCH(Let)
  DEDUCE_DISPATCH(For)
  DEDUCE_DISPATCH(Cast)
  DEDUCE_DISPATCH(StructDef)
  DEDUCE_DISPATCH(Array)
  DEDUCE_DISPATCH(Attr)
  DEDUCE_DISPATCH(ConstantArray)
  DEDUCE_DISPATCH(ColonRef)
  DEDUCE_DISPATCH(Index)
  DEDUCE_DISPATCH(Match)
  DEDUCE_DISPATCH(StructInstance)
  DEDUCE_DISPATCH(SplatStructInstance)
  DEDUCE_DISPATCH(BuiltinTypeAnnotation)
  DEDUCE_DISPATCH(ArrayTypeAnnotation)
  DEDUCE_DISPATCH(TupleTypeAnnotation)
  DEDUCE_DISPATCH(TypeRefTypeAnnotation)
  DEDUCE_DISPATCH(MatchArm)
  DEDUCE_DISPATCH(While)
  DEDUCE_DISPATCH(Carry)
  DEDUCE_DISPATCH(Invocation)
  DEDUCE_DISPATCH(ConstRef)
  DEDUCE_DISPATCH(NameRef)

  // Unhandled nodes for deduction, either they are custom visited or not
  // visited "automatically" in the traversal process (e.g. top level module
  // members).
  absl::Status HandleNext(Next* n) override { return Fatal(n); }
  absl::Status HandleProc(Proc* n) override { return Fatal(n); }
  absl::Status HandleSlice(Slice* n) override { return Fatal(n); }
  absl::Status HandleImport(Import* n) override { return Fatal(n); }
  absl::Status HandleFunction(Function* n) override { return Fatal(n); }
  absl::Status HandleQuickCheck(QuickCheck* n) override { return Fatal(n); }
  absl::Status HandleTestFunction(TestFunction* n) override { return Fatal(n); }
  absl::Status HandleModule(Module* n) override { return Fatal(n); }
  absl::Status HandleWidthSlice(WidthSlice* n) override { return Fatal(n); }
  absl::Status HandleNameDefTree(NameDefTree* n) override { return Fatal(n); }
  absl::Status HandleNameDef(NameDef* n) override { return Fatal(n); }
  absl::Status HandleBuiltinNameDef(BuiltinNameDef* n) override {
    return Fatal(n);
  }
  absl::Status HandleParametricBinding(ParametricBinding* n) override {
    return Fatal(n);
  }
  absl::Status HandleWildcardPattern(WildcardPattern* n) override {
    return Fatal(n);
  }

  absl::StatusOr<std::unique_ptr<ConcreteType>>& result() { return result_; }

 private:
  absl::Status Fatal(AstNode* n) {
    XLS_LOG(FATAL) << "Got unhandled AST node for deduction: " << n->ToString();
  }

  DeduceCtx* ctx_;
  absl::StatusOr<std::unique_ptr<ConcreteType>> result_;
};

absl::StatusOr<std::unique_ptr<ConcreteType>> DeduceInternal(AstNode* node,
                                                             DeduceCtx* ctx) {
  DeduceVisitor visitor(ctx);
  XLS_RETURN_IF_ERROR(node->Accept(&visitor));
  return std::move(visitor.result());
}

}  // namespace

absl::StatusOr<std::unique_ptr<ConcreteType>> Resolve(const ConcreteType& type,
                                                      DeduceCtx* ctx) {
  XLS_RET_CHECK(!ctx->fn_stack().empty());
  const FnStackEntry& entry = ctx->fn_stack().back();
  const SymbolicBindings& fn_symbolic_bindings = entry.symbolic_bindings();

  return type.MapSize([&fn_symbolic_bindings](ConcreteTypeDim dim)
                          -> absl::StatusOr<ConcreteTypeDim> {
    if (absl::holds_alternative<ConcreteTypeDim::OwnedParametric>(
            dim.value())) {
      const auto& parametric =
          absl::get<ConcreteTypeDim::OwnedParametric>(dim.value());
      ParametricExpression::Env env = ToParametricEnv(fn_symbolic_bindings);
      return ConcreteTypeDim(parametric->Evaluate(env));
    }
    return dim;
  });
}

absl::StatusOr<std::unique_ptr<ConcreteType>> Deduce(AstNode* node,
                                                     DeduceCtx* ctx) {
  if (absl::optional<ConcreteType*> type = ctx->type_info()->GetItem(node)) {
    return (*type)->CloneToUnique();
  }
  XLS_ASSIGN_OR_RETURN(std::unique_ptr<ConcreteType> type,
                       DeduceInternal(node, ctx));
  ctx->type_info()->SetItem(node, *type);
  XLS_VLOG(5) << "Deduced type of " << node->ToString() << " => "
              << type->ToString();
  return type;
}

absl::StatusOr<std::unique_ptr<ConcreteType>> DeduceAndResolve(AstNode* node,
                                                               DeduceCtx* ctx) {
  XLS_ASSIGN_OR_RETURN(std::unique_ptr<ConcreteType> deduced,
                       ctx->Deduce(node));
  return Resolve(*deduced, ctx);
}

// Converts a sequence of ParametricBinding AST nodes into a sequence of
// ParametricConstraints (which decorate the ParametricBinding nodes with their
// deduced ConcreteTypes).
absl::StatusOr<std::vector<ParametricConstraint>>
ParametricBindingsToConstraints(absl::Span<ParametricBinding* const> bindings,
                                DeduceCtx* ctx) {
  std::vector<ParametricConstraint> parametric_constraints;
  for (ParametricBinding* binding : bindings) {
    XLS_ASSIGN_OR_RETURN(std::unique_ptr<ConcreteType> binding_type,
                         ParametricBindingToType(binding, ctx));
    parametric_constraints.push_back(
        ParametricConstraint(*binding, std::move(binding_type)));
  }
  return parametric_constraints;
}

}  // namespace xls::dslx
