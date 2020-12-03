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

namespace xls::dslx {

absl::Status CheckBitwidth(const Number& number, const ConcreteType& type) {
  XLS_ASSIGN_OR_RETURN(ConcreteTypeDim bits_dim, type.GetTotalBitCount());
  XLS_RET_CHECK(absl::holds_alternative<int64>(bits_dim.value()));
  int64 bit_count = absl::get<int64>(bits_dim.value());
  absl::StatusOr<Bits> bits = number.GetBits(bit_count);
  if (!bits.ok()) {
    return absl::InternalError(
        absl::StrFormat("TypeInferenceError: %s %s Value '%s' does not fit in "
                        "the bitwidth of a %s (%d)",
                        number.span().ToString(), type.ToString(),
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
  return ctx->Deduce(node->type());
}

absl::StatusOr<std::unique_ptr<ConcreteType>> DeduceConstantDef(
    ConstantDef* node, DeduceCtx* ctx) {
  XLS_ASSIGN_OR_RETURN(std::unique_ptr<ConcreteType> result,
                       ctx->Deduce(node->value()));
  ctx->type_info()->SetItem(node->name_def(), *result);
  ctx->type_info()->NoteConstant(node->name_def(), node);
  return result;
}

absl::StatusOr<std::unique_ptr<ConcreteType>> DeduceTypeRef(TypeRef* node,
                                                            DeduceCtx* ctx) {
  return ctx->Deduce(ToAstNode(node->type_definition()));
}

absl::StatusOr<std::unique_ptr<ConcreteType>> DeduceTypeDef(TypeDef* node,
                                                            DeduceCtx* ctx) {
  XLS_ASSIGN_OR_RETURN(std::unique_ptr<ConcreteType> type,
                       ctx->Deduce(node->type()));
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

absl::StatusOr<std::unique_ptr<ConcreteType>> Resolve(const ConcreteType& type,
                                                      DeduceCtx* ctx) {
  XLS_RET_CHECK(!ctx->fn_stack().empty());
  const FnStackEntry& entry = ctx->fn_stack().back();
  const SymbolicBindings& fn_symbolic_bindings = entry.symbolic_bindings;
  return type.MapSize([&fn_symbolic_bindings](ConcreteTypeDim dim)
                          -> absl::StatusOr<ConcreteTypeDim> {
    if (absl::holds_alternative<ConcreteTypeDim::OwnedParametric>(
            dim.value())) {
      const auto& parametric =
          absl::get<ConcreteTypeDim::OwnedParametric>(dim.value());
      ParametricExpression::Env env;
      for (const SymbolicBinding& binding : fn_symbolic_bindings.bindings()) {
        env[binding.identifier] = binding.value;
      }
      return ConcreteTypeDim(parametric->Evaluate(env));
    }
    return dim;
  });
}

static absl::StatusOr<std::unique_ptr<ConcreteType>> DeduceAndResolve(
    AstNode* node, DeduceCtx* ctx) {
  XLS_ASSIGN_OR_RETURN(std::unique_ptr<ConcreteType> deduced,
                       ctx->Deduce(node));
  return Resolve(*deduced, ctx);
}

absl::StatusOr<std::unique_ptr<ConcreteType>> DeduceNumber(Number* node,
                                                           DeduceCtx* ctx) {
  if (node->type() == nullptr) {
    switch (node->kind()) {
      case NumberKind::kBool:
        return BitsType::MakeU1();
      case NumberKind::kCharacter:
        return BitsType::MakeU8();
      default:
        break;
    }
    return absl::InternalError(
        absl::StrFormat("TypeInferenceError: %s <> Could not infer a type for "
                        "this number, please annotate a type.",
                        node->span().ToString()));
  }

  XLS_ASSIGN_OR_RETURN(std::unique_ptr<ConcreteType> concrete_type,
                       ctx->Deduce(node->type()));
  XLS_ASSIGN_OR_RETURN(concrete_type, Resolve(*concrete_type, ctx));
  XLS_RETURN_IF_ERROR(CheckBitwidth(*node, *concrete_type));
  return concrete_type;
}

absl::StatusOr<std::unique_ptr<ConcreteType>> DeduceTernary(Ternary* node,
                                                            DeduceCtx* ctx) {
  XLS_ASSIGN_OR_RETURN(std::unique_ptr<ConcreteType> test_type,
                       ctx->Deduce(node->test()));
  XLS_ASSIGN_OR_RETURN(test_type, Resolve(*test_type, ctx));
  auto test_want = BitsType::MakeU1();
  if (*test_type != *test_want) {
    return absl::InternalError(absl::StrFormat(
        "XlsTypeError: %s %s %s Test type for conditional expression is not "
        "\"bool\"",
        node->span().ToString(), test_type->ToString(), test_want->ToString()));
  }

  XLS_ASSIGN_OR_RETURN(std::unique_ptr<ConcreteType> consequent_type,
                       ctx->Deduce(node->consequent()));
  XLS_ASSIGN_OR_RETURN(consequent_type, Resolve(*consequent_type, ctx));
  XLS_ASSIGN_OR_RETURN(std::unique_ptr<ConcreteType> alternate_type,
                       ctx->Deduce(node->alternate()));
  XLS_ASSIGN_OR_RETURN(alternate_type, Resolve(*alternate_type, ctx));

  if (*consequent_type != *alternate_type) {
    return absl::InternalError(absl::StrFormat(
        "XlsTypeError: %s %s %s Ternary consequent type (in the 'then' clause) "
        "did not match alternative type (in the 'else' clause)",
        node->span().ToString(), consequent_type->ToString(),
        alternate_type->ToString()));
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
    return absl::InternalError(absl::StrFormat(
        "XlsTypeError: %s %s %s Attempting to concatenate array/non-array "
        "values together.",
        node->span().ToString(), lhs->ToString(), rhs->ToString()));
  }

  if (lhs_is_array && lhs_array->element_type() != rhs_array->element_type()) {
    return absl::InternalError(absl::StrFormat(
        "XlsTypeError: %s %s %s Array concatenation requires element types to "
        "be the same.",
        node->span().ToString(), lhs->ToString(), rhs->ToString()));
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

  XLS_ASSIGN_OR_RETURN(std::unique_ptr<ConcreteType> lhs,
                       DeduceAndResolve(node->lhs(), ctx));
  XLS_ASSIGN_OR_RETURN(std::unique_ptr<ConcreteType> rhs,
                       DeduceAndResolve(node->rhs(), ctx));

  if (*lhs != *rhs) {
    return absl::InternalError(
        absl::StrFormat("XlsTypeError: %s %s %s Could not deduce type for "
                        "binary operation '%s'",
                        node->span().ToString(), lhs->ToString(),
                        rhs->ToString(), BinopKindFormat(node->kind())));
  }

  if (auto* enum_type = dynamic_cast<EnumType*>(lhs.get());
      enum_type != nullptr && !GetEnumOkKinds().contains(node->kind())) {
    return absl::InternalError(absl::StrFormat(
        "TypeInferenceError: %s <> Cannot use '%s' on values with enum type "
        "%s.",
        node->span().ToString(), BinopKindFormat(node->kind()),
        enum_type->nominal_type()->identifier()));
  }

  if (GetBinopComparisonKinds().contains(node->kind())) {
    return BitsType::MakeU1();
  }

  return lhs;
}

absl::StatusOr<std::unique_ptr<ConcreteType>> DeduceEnumDef(EnumDef* node,
                                                            DeduceCtx* ctx) {
  XLS_ASSIGN_OR_RETURN(std::unique_ptr<ConcreteType> type,
                       DeduceAndResolve(node->type(), ctx));
  auto* bits_type = dynamic_cast<BitsType*>(type.get());
  if (bits_type == nullptr) {
    return absl::InternalError(
        absl::StrFormat("TypeInferenceError: %s %s Underlying type for an enum "
                        "must be a bits type.",
                        node->span().ToString(), bits_type->ToString()));
  }

  // Grab the bit count of the Enum's underlying type.
  const ConcreteTypeDim& bit_count = bits_type->size();
  node->set_signedness(bits_type->is_signed());

  auto result = absl::make_unique<EnumType>(node, bit_count);
  for (const EnumMember& member : node->values()) {
    // Note: the parser places the type from the enum on the value when it is a
    // number, so this deduction flags inappropriate numbers.
    XLS_RETURN_IF_ERROR(ctx->Deduce(ToAstNode(member.value)).status());
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
    return absl::InternalError(absl::StrFormat(
        "TypeInferenceError: %s %s Expected a tuple type for these names, but "
        "got %s.",
        name_def_tree->span().ToString(), type.ToString(), type.ToString()));
  }

  if (name_def_tree->nodes().size() != tuple_type->size()) {
    return absl::InternalError(absl::StrFormat(
        "TypeInferenceError: %s %s Could not bind names, names are mismatched "
        "in number vs type; at this level of the tuple: %d names, %d types.",
        name_def_tree->span().ToString(), type.ToString(),
        name_def_tree->nodes().size(), tuple_type->size()));
  }

  for (int64 i = 0; i < name_def_tree->nodes().size(); ++i) {
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

  if (node->type() != nullptr) {
    XLS_ASSIGN_OR_RETURN(std::unique_ptr<ConcreteType> annotated,
                         DeduceAndResolve(node->type(), ctx));
    if (*rhs != *annotated) {
      return absl::InternalError(absl::StrFormat(
          "XlsTypeError: %s %s %s Annotated type did not match inferred type "
          "of right hand side expression.",
          node->span().ToString(), annotated->ToString(), rhs->ToString()));
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
  XLS_ASSIGN_OR_RETURN(std::unique_ptr<ConcreteType> init_type,
                       DeduceAndResolve(node->init(), ctx));
  XLS_ASSIGN_OR_RETURN(std::unique_ptr<ConcreteType> annotated_type,
                       ctx->Deduce(node->type()));

  XLS_RETURN_IF_ERROR(BindNames(node->names(), *annotated_type, ctx));
  XLS_ASSIGN_OR_RETURN(std::unique_ptr<ConcreteType> body_type,
                       DeduceAndResolve(node->body(), ctx));

  XLS_RETURN_IF_ERROR(ctx->Deduce(node->iterable()).status());

  if (*init_type != *body_type) {
    return absl::InternalError(absl::StrFormat(
        "XlsTypeError: %s %s %s For-loop init value type did not match "
        "for-loop body's result type.",
        node->span().ToString(), init_type->ToString(), body_type->ToString()));
  }

  // TODO(leary): 2019-02-19 Type check annotated_type (the bound names each
  // iteration) against init_type/body_type -- this requires us to understand
  // how iterables turn into induction values.
  return init_type;
}

// TODO(leary): 2020-12-02 Seems like acceptable casts should be much more
// restrictive than this...
static bool IsAcceptableCast(const ConcreteType& from, const ConcreteType& to) {
  auto is_array = [](const ConcreteType& ct) -> bool {
    return dynamic_cast<const ArrayType*>(&ct) != nullptr;
  };
  auto is_bits = [](const ConcreteType& ct) -> bool {
    return dynamic_cast<const BitsType*>(&ct) != nullptr;
  };
  if ((is_array(from) && is_bits(to)) || (is_bits(from) && is_array(to))) {
    return from.GetTotalBitCount() == to.GetTotalBitCount();
  }
  return true;
}

absl::StatusOr<std::unique_ptr<ConcreteType>> DeduceCast(Cast* node,
                                                         DeduceCtx* ctx) {
  XLS_ASSIGN_OR_RETURN(std::unique_ptr<ConcreteType> type,
                       DeduceAndResolve(node->type(), ctx));
  XLS_ASSIGN_OR_RETURN(std::unique_ptr<ConcreteType> expr,
                       DeduceAndResolve(node->expr(), ctx));

  if (!IsAcceptableCast(/*from=*/*expr, /*to=*/*type)) {
    return absl::InternalError(absl::StrFormat(
        "XlsTypeError: %s %s %s Cannot cast from expression type %s to %s.",
        node->span().ToString(), expr->ToString(), type->ToString(),
        expr->ToString(), type->ToString()));
  }
  return type;
}

absl::StatusOr<std::unique_ptr<ConcreteType>> DeduceStructDef(StructDef* node,
                                                              DeduceCtx* ctx) {
  for (const ParametricBinding* parametric : node->parametric_bindings()) {
    XLS_ASSIGN_OR_RETURN(std::unique_ptr<ConcreteType> parametric_binding_type,
                         ctx->Deduce(parametric->type()));
    if (parametric->expr() != nullptr) {
      XLS_ASSIGN_OR_RETURN(std::unique_ptr<ConcreteType> expr_type,
                           ctx->Deduce(parametric->expr()));
      if (*expr_type != *parametric_binding_type) {
        return absl::InternalError(
            absl::StrFormat("XlsTypeError: %s %s %s Annotated type of "
                            "parametric value did not match inferred type.",
                            node->span().ToString(), expr_type->ToString(),
                            parametric_binding_type->ToString()));
      }
    }
    ctx->type_info()->SetItem(parametric->name_def(), *parametric_binding_type);
  }

  TupleType::NamedMembers members;
  for (auto [name_def, type] : node->members()) {
    XLS_ASSIGN_OR_RETURN(std::unique_ptr<ConcreteType> concrete,
                         DeduceAndResolve(type, ctx));
    members.push_back({name_def->identifier(), std::move(concrete)});
  }
  auto result = absl::make_unique<TupleType>(std::move(members), node);
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

  for (int64 i = 1; i < member_types.size(); ++i) {
    if (*member_types[0] != *member_types[i]) {
      return absl::InternalError(
          absl::StrFormat("XlsTypeError: %s %s %s Array member did not have "
                          "same type as other members.",
                          node->span().ToString(), member_types[0]->ToString(),
                          member_types[i]->ToString()));
    }
  }

  auto inferred = absl::make_unique<ArrayType>(
      member_types[0]->CloneToUnique(),
      ConcreteTypeDim(static_cast<int64>(member_types.size())));

  if (node->type() == nullptr) {
    return inferred;
  }

  XLS_ASSIGN_OR_RETURN(std::unique_ptr<ConcreteType> annotated,
                       ctx->Deduce(node->type()));
  auto* array_type = dynamic_cast<ArrayType*>(annotated.get());
  if (array_type == nullptr) {
    return absl::InternalError(absl::StrFormat(
        "TypeInferenceError: %s %s Array was not annotated with an array type.",
        node->span().ToString(), annotated->ToString()));
  }

  XLS_ASSIGN_OR_RETURN(std::unique_ptr<ConcreteType> resolved_element_type,
                       Resolve(array_type->element_type(), ctx));
  if (*resolved_element_type != *member_types[0]) {
    return absl::InternalError(absl::StrFormat(
        "XlsTypeError: %s %s %s Annotated element type did not match inferred "
        "element type.",
        node->span().ToString(), resolved_element_type->ToString(),
        member_types[0]->ToString()));
  }

  if (node->has_ellipsis()) {
    return annotated;
  }

  if (array_type->size() !=
      ConcreteTypeDim(static_cast<int64>(member_types.size()))) {
    return absl::InternalError(absl::StrFormat(
        "XlsTypeError: %s %s %s Annotated array size %s does not match "
        "inferred array size %d.",
        node->span().ToString(), array_type->ToString(), inferred->ToString(),
        array_type->size().ToString(), member_types.size()));
  }

  return inferred;
}

}  // namespace xls::dslx
