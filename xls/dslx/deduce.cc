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
      ParametricExpression::Env env = ToParametricEnv(fn_symbolic_bindings);
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
    return TypeInferenceErrorStatus(node->span(), nullptr,
                                    "Could not infer a type for "
                                    "this number, please annotate a type.");
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

  return lhs;
}

absl::StatusOr<std::unique_ptr<ConcreteType>> DeduceEnumDef(EnumDef* node,
                                                            DeduceCtx* ctx) {
  XLS_ASSIGN_OR_RETURN(std::unique_ptr<ConcreteType> type,
                       DeduceAndResolve(node->type(), ctx));
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
      return XlsTypeErrorStatus(node->span(), *annotated, *rhs,
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
  XLS_ASSIGN_OR_RETURN(std::unique_ptr<ConcreteType> init_type,
                       DeduceAndResolve(node->init(), ctx));
  XLS_ASSIGN_OR_RETURN(std::unique_ptr<ConcreteType> annotated_type,
                       ctx->Deduce(node->type()));

  XLS_RETURN_IF_ERROR(BindNames(node->names(), *annotated_type, ctx));
  XLS_ASSIGN_OR_RETURN(std::unique_ptr<ConcreteType> body_type,
                       DeduceAndResolve(node->body(), ctx));

  XLS_RETURN_IF_ERROR(ctx->Deduce(node->iterable()).status());

  if (*init_type != *body_type) {
    return XlsTypeErrorStatus(node->span(), *init_type, *body_type,
                              "For-loop init value type did not match "
                              "for-loop body's result type.");
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
    return XlsTypeErrorStatus(
        node->span(), *expr, *type,
        absl::StrFormat("Cannot cast from expression type %s to %s.",
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
        return XlsTypeErrorStatus(
            node->span(), *expr_type, *parametric_binding_type,
            "Annotated type of "
            "parametric value did not match inferred type.");
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
      return XlsTypeErrorStatus(
          node->span(), *member_types[0], *member_types[i],
          "Array member did not have same type as other members.");
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
    return TypeInferenceErrorStatus(
        node->span(), annotated.get(),
        "Array was not annotated with an array type.");
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

  if (array_type->size() !=
      ConcreteTypeDim(static_cast<int64>(member_types.size()))) {
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
  XLS_ASSIGN_OR_RETURN(std::unique_ptr<ConcreteType> struct_type,
                       ctx->Deduce(node->lhs()));
  auto* tuple_type = dynamic_cast<TupleType*>(struct_type.get());
  if (tuple_type == nullptr ||
      // If the (concrete) tuple type is unnamed, then it's not a struct, it's a
      // tuple.
      !tuple_type->is_named()) {
    return TypeInferenceErrorStatus(node->span(), struct_type.get(),
                                    absl::StrFormat("Expected a struct for "
                                                    "attribute access; got %s",
                                                    struct_type->ToString()));
  }

  const std::string& attr_name = node->attr()->identifier();
  if (!tuple_type->HasNamedMember(attr_name)) {
    return TypeInferenceErrorStatus(
        node->span(), nullptr,
        absl::StrFormat("Struct '%s' does not have a "
                        "member with name "
                        "'%s'",
                        tuple_type->nominal_type()->identifier(), attr_name));
  }

  absl::optional<const ConcreteType*> result =
      tuple_type->GetMemberTypeByName(attr_name);
  XLS_RET_CHECK(result.has_value());  // We checked above we had named member.
  return result.value()->CloneToUnique();
}

absl::StatusOr<std::unique_ptr<ConcreteType>> DeduceConstantArray(
    ConstantArray* node, DeduceCtx* ctx) {
  if (node->type() == nullptr) {
    return DeduceArray(node, ctx);
  }

  XLS_ASSIGN_OR_RETURN(std::unique_ptr<ConcreteType> type,
                       ctx->Deduce(node->type()));
  auto* array_type = dynamic_cast<ArrayType*>(type.get());
  if (array_type == nullptr) {
    return TypeInferenceErrorStatus(
        node->type()->span(), type.get(),
        absl::StrFormat("Annotated type for array "
                        "literal must be an array type; got %s %s",
                        type->GetDebugTypeName(), node->type()->ToString()));
  }

  const ConcreteType& element_type = array_type->element_type();
  for (Expr* member : node->members()) {
    XLS_RET_CHECK(IsConstant(member));
    if (Number* number = dynamic_cast<Number*>(member);
        number != nullptr && number->type() == nullptr) {
      ctx->type_info()->SetItem(member, element_type);
      XLS_RETURN_IF_ERROR(CheckBitwidth(*number, element_type));
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
  if (absl::holds_alternative<Test*>(member)) {
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
  const std::shared_ptr<Module>& imported_module = (*imported)->module;
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

  std::shared_ptr<TypeInfo> imported_type_info = (*imported)->type_info;
  if (absl::holds_alternative<Function*>(*elem.value())) {
    auto* function = absl::get<Function*>(*elem.value());
    if (!imported_type_info->Contains(function->name_def())) {
      XLS_VLOG(2) << "Function name not in imported_type_info; indicates it is "
                     "parametric.";
      XLS_RET_CHECK(function->is_parametric());
      // We don't type check parametric functions until invocations.
      // Let's typecheck this imported parametric function with respect to its
      // module (this will only get the type signature, the body gets
      // typechecked after parametric instantiation).
      DeduceCtx imported_ctx =
          ctx->MakeCtx(imported_type_info, imported_module);
      const FnStackEntry& peek_entry = ctx->fn_stack().back();
      imported_ctx.fn_stack().push_back(
          FnStackEntry{peek_entry.name, peek_entry.symbolic_bindings});
      XLS_RETURN_IF_ERROR(ctx->typecheck_function()(function, &imported_ctx));
      ctx->type_info()->Update(*imported_ctx.type_info());
      imported_type_info = imported_ctx.type_info();
    }
  }

  AstNode* member_node = ToAstNode(*elem.value());
  absl::optional<ConcreteType*> type = imported_type_info->GetItem(member_node);
  XLS_RET_CHECK(type.has_value()) << member_node->ToString();
  return type.value()->CloneToUnique();
}

absl::StatusOr<std::unique_ptr<ConcreteType>> DeduceColonRef(ColonRef* node,
                                                             DeduceCtx* ctx) {
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
  XLS_RET_CHECK(enum_type != nullptr);
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

  if (number->type() == nullptr) {
    ctx->type_info()->SetItem(number, *BitsType::MakeU32());
  } else {
    // If the number has an annotated type, flag it as unnecessary.
    XLS_RETURN_IF_ERROR(ctx->Deduce(number).status());
    XLS_LOG(WARNING) << absl::StreamFormat(
        "Warning: type annotation for tuple index is unnecessary @ %s: %s",
        node->span().ToString(), node->ToString());
  }

  XLS_ASSIGN_OR_RETURN(int64 value, number->GetAsUint64());
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
    int64 bit_count, absl::optional<int64> start_opt,
    absl::optional<int64> limit_opt) {
  XLS_RET_CHECK_GE(bit_count, 0);
  int64 start = 0;
  int64 limit = bit_count;

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

  limit = std::min(std::max(limit, int64{0}), bit_count);
  start = std::min(std::max(start, int64{0}), limit);
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
      start_number != nullptr && start_number->type() == nullptr) {
    // A literal number with no annotated type as the slice start.
    //
    // By default, we use the "subject" type (converted to unsigned) as the type
    // for the slice start.
    start_type_owned = subject_type.ToUBits();
    start_type = dynamic_cast<BitsType*>(start_type_owned.get());

    // Get the start number as an integral value, after we make sure it fits.
    XLS_ASSIGN_OR_RETURN(Bits start_bits, start_number->GetBits(64));
    XLS_ASSIGN_OR_RETURN(int64 start_int, start_bits.ToInt64());

    XLS_ASSIGN_OR_RETURN(std::unique_ptr<ConcreteType> resolved_start_type,
                         Resolve(*start_type, ctx));
    XLS_ASSIGN_OR_RETURN(ConcreteTypeDim bit_count_ctd,
                         resolved_start_type->GetTotalBitCount());
    int64 bit_count = absl::get<int64>(bit_count_ctd.value());

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

  // Check the start is unsigned.
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
  if (absl::holds_alternative<int64>(width_ctd.value()) &&
      absl::holds_alternative<int64>(subject_ctd.value())) {
    int64 width_bits = absl::get<int64>(width_ctd.value());
    int64 subject_bits = absl::get<int64>(subject_ctd.value());
    if (width_bits > subject_bits) {
      return XlsTypeErrorStatus(
          start->span(), subject_type, *width_type,
          absl::StrFormat("Slice type must have <= original number of bits; "
                          "attempted slice from %d to %d bits.",
                          subject_bits, width_bits));
    }
  }

  // Check the width type is bits-based (e.g. no enums, since sliced value
  // could be out of range of the valid enum values).
  if (dynamic_cast<BitsType*>(width_type.get()) == nullptr) {
    return TypeInferenceErrorStatus(
        node->span(), width_type.get(),
        "A bits type is required for a width-based slice.");
  }

  // The width type is the thing returned from the width-slice.
  return width_type;
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

  auto* slice = absl::get<Slice*>(node->rhs());
  absl::optional<int64> limit;
  if (slice->limit() != nullptr) {
    XLS_ASSIGN_OR_RETURN(int64 limit_value, slice->limit()->GetAsUint64());
    limit = limit_value;
  }
  absl::optional<int64> start;
  if (slice->start() != nullptr) {
    XLS_ASSIGN_OR_RETURN(int64 start_value, slice->start()->GetAsUint64());
    start = start_value;
  }

  const SymbolicBindings& fn_symbolic_bindings =
      ctx->fn_stack().back().symbolic_bindings;
  XLS_ASSIGN_OR_RETURN(ConcreteTypeDim lhs_bit_count_ctd,
                       lhs_type->GetTotalBitCount());
  int64 bit_count;
  if (absl::holds_alternative<ConcreteTypeDim::OwnedParametric>(
          lhs_bit_count_ctd.value())) {
    auto& owned_parametric =
        absl::get<ConcreteTypeDim::OwnedParametric>(lhs_bit_count_ctd.value());
    ParametricExpression::Evaluated evaluated =
        owned_parametric->Evaluate(ToParametricEnv(fn_symbolic_bindings));
    XLS_RET_CHECK(absl::holds_alternative<int64>(evaluated));
    bit_count = absl::get<int64>(evaluated);
  } else {
    bit_count = absl::get<int64>(lhs_bit_count_ctd.value());
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
      for (int64 i = 0; i < nodes.size(); ++i) {
        const ConcreteType& subtype = type->GetMemberType(i);
        NameDefTree* subtree = nodes[i];
        XLS_RETURN_IF_ERROR(Unify(subtree, subtype, ctx));
      }
    }
  }
  return absl::OkStatus();
}

absl::StatusOr<std::unique_ptr<ConcreteType>> DeduceMatch(Match* node,
                                                          DeduceCtx* ctx) {
  XLS_ASSIGN_OR_RETURN(std::unique_ptr<ConcreteType> matched,
                       ctx->Deduce(node->matched()));

  for (MatchArm* arm : node->arms()) {
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

  for (int64 i = 1; i < arm_types.size(); ++i) {
    if (*arm_types[i] != *arm_types[0]) {
      return XlsTypeErrorStatus(
          node->arms()[i]->span(), *arm_types[i], *arm_types[0],
          "This match arm did not have the same type as the "
          "preceding match arms.");
    }
  }
  return std::move(arm_types[0]);
}

}  // namespace xls::dslx
