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

#include "xls/dslx/cpp_evaluate.h"

#include "xls/common/status/ret_check.h"
#include "xls/dslx/type_info.h"

namespace xls::dslx {
namespace {

using Value = InterpValue;
using Tag = InterpValueTag;

}  // namespace

absl::StatusOr<InterpValue> EvaluateIndexBitslice(TypeInfo* type_info,
                                                  Index* expr,
                                                  InterpBindings* bindings,
                                                  const Bits& bits) {
  IndexRhs index = expr->rhs();
  XLS_RET_CHECK(absl::holds_alternative<Slice*>(index));
  auto index_slice = absl::get<Slice*>(index);

  const SymbolicBindings& sym_bindings = bindings->fn_ctx()->sym_bindings;

  absl::optional<SliceData::StartWidth> maybe_saw =
      type_info->GetSliceStartWidth(index_slice, sym_bindings);
  XLS_RET_CHECK(maybe_saw.has_value());
  const auto& saw = maybe_saw.value();
  return Value::MakeBits(Tag::kUBits, bits.Slice(saw.start, saw.width));
}

absl::StatusOr<InterpValue> EvaluateNameRef(NameRef* expr,
                                            InterpBindings* bindings,
                                            ConcreteType* type_context) {
  return bindings->ResolveValue(expr);
}

absl::StatusOr<InterpValue> EvaluateConstRef(ConstRef* expr,
                                             InterpBindings* bindings,
                                             ConcreteType* type_context) {
  return bindings->ResolveValue(expr);
}

absl::StatusOr<InterpValue> EvaluateEnumRef(EnumRef* expr,
                                            InterpBindings* bindings,
                                            ConcreteType* type_context,
                                            InterpCallbackData* callbacks) {
  XLS_ASSIGN_OR_RETURN(
      EnumDef * enum_def,
      EvaluateToEnum(ToTypeDefinition(ToAstNode(expr->enum_def())).value(),
                     bindings, callbacks));
  XLS_ASSIGN_OR_RETURN(auto value_node, enum_def->GetValue(expr->attr()));
  XLS_ASSIGN_OR_RETURN(
      InterpBindings fresh_bindings,
      MakeTopLevelBindings(expr->owner()->shared_from_this(), callbacks));
  XLS_ASSIGN_OR_RETURN(
      std::unique_ptr<ConcreteType> concrete_type,
      ConcretizeTypeAnnotation(enum_def->type(), &fresh_bindings, callbacks));
  Expr* value_expr = ToExprNode(value_node);
  XLS_ASSIGN_OR_RETURN(InterpValue raw_value,
                       callbacks->eval(value_expr, &fresh_bindings));
  return InterpValue::MakeEnum(raw_value.GetBitsOrDie(), enum_def);
}

absl::StatusOr<InterpValue> EvaluateUnop(Unop* expr, InterpBindings* bindings,
                                         ConcreteType* type_context,
                                         InterpCallbackData* callbacks) {
  XLS_ASSIGN_OR_RETURN(InterpValue arg,
                       callbacks->eval(expr->operand(), bindings));
  switch (expr->kind()) {
    case UnopKind::kInvert:
      return arg.BitwiseNegate();
    case UnopKind::kNegate:
      return arg.ArithmeticNegate();
  }
  return absl::InternalError(absl::StrCat("Invalid unary operation kind: ",
                                          static_cast<int64>(expr->kind())));
}

absl::StatusOr<InterpValue> EvaluateBinop(Binop* expr, InterpBindings* bindings,
                                          ConcreteType* type_context,
                                          InterpCallbackData* callbacks) {
  XLS_ASSIGN_OR_RETURN(InterpValue lhs, callbacks->eval(expr->lhs(), bindings));
  XLS_ASSIGN_OR_RETURN(InterpValue rhs, callbacks->eval(expr->rhs(), bindings));

  // Check some preconditions; e.g. all logical operands are guaranteed to have
  // single-bit inputs by type checking so we can share the implementation with
  // bitwise or/and.
  switch (expr->kind()) {
    case BinopKind::kLogicalOr:
    case BinopKind::kLogicalAnd:
      XLS_RET_CHECK_EQ(lhs.GetBitCount().value(), 1);
      XLS_RET_CHECK_EQ(rhs.GetBitCount().value(), 1);
      break;
    default:
      break;
  }

  switch (expr->kind()) {
    case BinopKind::kAdd:
      return lhs.Add(rhs);
    case BinopKind::kSub:
      return lhs.Sub(rhs);
    case BinopKind::kConcat:
      return lhs.Concat(rhs);
    case BinopKind::kMul:
      return lhs.Mul(rhs);
    case BinopKind::kDiv:
      return lhs.FloorDiv(rhs);
    case BinopKind::kOr:
    case BinopKind::kLogicalOr:
      return lhs.BitwiseOr(rhs);
    case BinopKind::kAnd:
    case BinopKind::kLogicalAnd:
      return lhs.BitwiseAnd(rhs);
    case BinopKind::kXor:
      return lhs.BitwiseXor(rhs);
    case BinopKind::kShll:
      return lhs.Shll(rhs);
    case BinopKind::kShrl:
      return lhs.Shrl(rhs);
    case BinopKind::kShra:
      return lhs.Shra(rhs);
    case BinopKind::kEq:
      return Value::MakeBool(lhs.Eq(rhs));
    case BinopKind::kNe:
      return Value::MakeBool(lhs.Ne(rhs));
    case BinopKind::kGt:
      return lhs.Gt(rhs);
    case BinopKind::kLt:
      return lhs.Lt(rhs);
    case BinopKind::kLe:
      return lhs.Le(rhs);
    case BinopKind::kGe:
      return lhs.Ge(rhs);
  }
  return absl::InternalError(absl::StrCat("Invalid binary operation kind: ",
                                          static_cast<int64>(expr->kind())));
}

absl::StatusOr<InterpValue> EvaluateTernary(Ternary* expr,
                                            InterpBindings* bindings,
                                            ConcreteType* type_context,
                                            InterpCallbackData* callbacks) {
  XLS_ASSIGN_OR_RETURN(InterpValue test,
                       callbacks->eval(expr->test(), bindings));
  if (test.IsTrue()) {
    return callbacks->eval(expr->consequent(), bindings);
  }
  return callbacks->eval(expr->alternate(), bindings);
}

absl::StatusOr<InterpValue> EvaluateAttr(Attr* expr, InterpBindings* bindings,
                                         ConcreteType* type_context,
                                         InterpCallbackData* callbacks) {
  XLS_ASSIGN_OR_RETURN(InterpValue lhs, callbacks->eval(expr->lhs(), bindings));
  std::shared_ptr<TypeInfo> type_info = callbacks->get_type_info();
  absl::optional<ConcreteType*> maybe_type = type_info->GetItem(expr->lhs());
  XLS_RET_CHECK(maybe_type.has_value()) << "LHS of attr should have type info";
  TupleType* tuple_type = dynamic_cast<TupleType*>(maybe_type.value());
  XLS_RET_CHECK(tuple_type != nullptr) << (*maybe_type)->ToString();
  absl::optional<int64> index;
  for (int64 i = 0; i < tuple_type->size(); ++i) {
    absl::string_view name = tuple_type->GetMemberName(i);
    if (name == expr->attr()->identifier()) {
      index = i;
      break;
    }
  }
  XLS_RET_CHECK(index.has_value())
      << "Unable to find attribute " << expr->attr()
      << ": should be caught by type inference";
  return lhs.GetValuesOrDie().at(*index);
}

// Returns whether this matcher pattern is accepted.
//
// Note that some patterns don't always match -- we call those "refutable"; e.g.
// in:
//
//       match (u32:3, u32:4) {
//         (u32:2, y) => y;
//         (x, _) => x;
//       }
//
// The first pattern will not match, and so this method would return false for
// that match arm. If you had a pattern that was just `_` it would match
// everything, and thus it is "irrefutable".
//
// Args:
//  pattern: Decribes the pattern attempting to match against the value.
//  to_match: The value being matched against.
//  bindings: The bindings to populate if the pattern has bindings associated
//    with it.
static absl::StatusOr<bool> EvaluateMatcher(NameDefTree* pattern,
                                            const InterpValue& to_match,
                                            InterpBindings* bindings,
                                            InterpCallbackData* callbacks) {
  if (pattern->is_leaf()) {
    NameDefTree::Leaf leaf = pattern->leaf();
    if (absl::holds_alternative<WildcardPattern*>(leaf)) {
      return true;
    }
    if (absl::holds_alternative<NameDef*>(leaf)) {
      bindings->AddValue(absl::get<NameDef*>(leaf)->identifier(), to_match);
      return true;
    }
    if (absl::holds_alternative<Number*>(leaf) ||
        absl::holds_alternative<EnumRef*>(leaf)) {
      XLS_ASSIGN_OR_RETURN(InterpValue target,
                           callbacks->eval(ToExprNode(leaf), bindings));
      return target.Eq(to_match);
    }
    XLS_RET_CHECK(absl::holds_alternative<NameRef*>(leaf));
    XLS_ASSIGN_OR_RETURN(InterpValue target,
                         callbacks->eval(absl::get<NameRef*>(leaf), bindings));
    return target.Eq(to_match);
  }

  XLS_RET_CHECK_EQ(to_match.GetLength().value(), pattern->nodes().size());
  for (int64 i = 0; i < pattern->nodes().size(); ++i) {
    NameDefTree* subtree = pattern->nodes()[i];
    const InterpValue& member = to_match.GetValuesOrDie().at(i);
    XLS_ASSIGN_OR_RETURN(bool matched,
                         EvaluateMatcher(subtree, member, bindings, callbacks));
    if (!matched) {
      return false;
    }
  }
  return true;
}

absl::StatusOr<InterpValue> EvaluateMatch(Match* expr, InterpBindings* bindings,
                                          ConcreteType* type_context,
                                          InterpCallbackData* callbacks) {
  XLS_ASSIGN_OR_RETURN(InterpValue to_match,
                       callbacks->eval(expr->matched(), bindings));
  for (MatchArm* arm : expr->arms()) {
    for (NameDefTree* pattern : arm->patterns()) {
      auto arm_bindings = std::make_shared<InterpBindings>(
          /*parent=*/bindings->shared_from_this());
      XLS_ASSIGN_OR_RETURN(
          bool did_match,
          EvaluateMatcher(pattern, to_match, arm_bindings.get(), callbacks));
      if (did_match) {
        return callbacks->eval(arm->expr(), arm_bindings.get());
      }
    }
  }
  return absl::InternalError(
      absl::StrFormat("FailureError: %s The program being interpreted failed "
                      "with an incomplete match; value: %s",
                      expr->span().ToString(), to_match.ToString()));
}

absl::StatusOr<InterpBindings> MakeTopLevelBindings(
    const std::shared_ptr<Module>& module, InterpCallbackData* callbacks) {
  XLS_VLOG(3) << "Making top level bindings for module: " << module->name();
  InterpBindings b(/*parent=*/nullptr);

  // Add all the builtin functions.
  for (Builtin builtin : kAllBuiltins) {
    b.AddFn(BuiltinToString(builtin), InterpValue::MakeFunction(builtin));
  }

  // Add all the functions in the top level scope for the module.
  for (Function* f : module->GetFunctions()) {
    b.AddFn(f->identifier(),
            InterpValue::MakeFunction(InterpValue::UserFnData{module, f}));
  }

  // Add all the type definitions in the top level scope for the module to the
  // bindings.
  for (TypeDefinition td : module->GetTypeDefinitions()) {
    if (absl::holds_alternative<TypeDef*>(td)) {
      auto* type_def = absl::get<TypeDef*>(td);
      b.AddTypeDef(type_def->identifier(), type_def);
    } else if (absl::holds_alternative<StructDef*>(td)) {
      auto* struct_def = absl::get<StructDef*>(td);
      b.AddStructDef(struct_def->identifier(), struct_def);
    } else {
      auto* enum_def = absl::get<EnumDef*>(td);
      b.AddEnumDef(enum_def->identifier(), enum_def);
    }
  }

  // Add constants/imports present at the top level to the bindings.
  for (ModuleMember member : module->top()) {
    XLS_VLOG(3) << "Evaluating module member: "
                << ToAstNode(member)->ToString();
    if (absl::holds_alternative<ConstantDef*>(member)) {
      auto* constant_def = absl::get<ConstantDef*>(member);
      if (callbacks->is_wip(constant_def)) {
        XLS_VLOG(3) << "Saw WIP constant definition; breaking early! "
                    << constant_def->ToString();
        break;
      }
      XLS_VLOG(3) << "MakeTopLevelBindings evaluating: "
                  << constant_def->ToString();
      absl::optional<InterpValue> precomputed =
          callbacks->note_wip(constant_def, absl::nullopt);
      absl::optional<InterpValue> result;
      if (precomputed.has_value()) {  // If we already computed it, use that.
        result = precomputed.value();
      } else {  // Otherwise, evaluate it and make a note.
        XLS_ASSIGN_OR_RETURN(result,
                             callbacks->eval(constant_def->value(), &b));
        callbacks->note_wip(constant_def, *result);
      }
      XLS_CHECK(result.has_value());
      b.AddValue(constant_def->identifier(), *result);
      XLS_VLOG(3) << "MakeTopLevelBindings evaluated: "
                  << constant_def->ToString() << " to " << result->ToString();
      continue;
    }
    if (absl::holds_alternative<Import*>(member)) {
      auto* import = absl::get<Import*>(member);
      XLS_VLOG(3) << "MakeTopLevelBindings importing: " << import->ToString();
      XLS_ASSIGN_OR_RETURN(
          const ModuleInfo* imported,
          DoImport(callbacks->typecheck, ImportTokens(import->subject()),
                   callbacks->cache));
      XLS_VLOG(3) << "MakeTopLevelBindings adding import " << import->ToString()
                  << " as \"" << import->identifier() << "\"";
      b.AddModule(import->identifier(), imported->module.get());
      continue;
    }
  }

  // Add a helpful value to the binding keys just to indicate what module these
  // top level bindings were created for, helpful for debugging.
  b.AddValue(absl::StrCat("__top_level_bindings_", module->name()),
             InterpValue::MakeNil());

  return b;
}

absl::StatusOr<int64> ResolveDim(
    absl::variant<Expr*, int64, ConcreteTypeDim> dim,
    InterpBindings* bindings) {
  if (absl::holds_alternative<int64>(dim)) {
    return absl::get<int64>(dim);
  }
  if (absl::holds_alternative<Expr*>(dim)) {
    Expr* expr = absl::get<Expr*>(dim);
    if (Number* number = dynamic_cast<Number*>(expr)) {
      return number->GetAsInt64();
    }
    if (NameRef* name_ref = dynamic_cast<NameRef*>(expr)) {
      const std::string& identifier = name_ref->identifier();
      XLS_ASSIGN_OR_RETURN(InterpValue value,
                           bindings->ResolveValueFromIdentifier(identifier));
      return value.GetBitValueInt64();
    }
    return absl::UnimplementedError(
        "Resolve dim expression: " + expr->ToString() + " @ " +
        expr->span().ToString());
  }

  XLS_RET_CHECK(absl::holds_alternative<ConcreteTypeDim>(dim));
  ConcreteTypeDim ctdim = absl::get<ConcreteTypeDim>(dim);
  if (const int64* value = absl::get_if<int64>(&ctdim.value())) {
    return *value;
  }

  const auto& parametric_expr =
      absl::get<ConcreteTypeDim::OwnedParametric>(ctdim.value());
  if (auto* parametric_symbol =
          dynamic_cast<ParametricSymbol*>(parametric_expr.get())) {
    XLS_ASSIGN_OR_RETURN(
        InterpValue value,
        bindings->ResolveValueFromIdentifier(parametric_symbol->identifier()));
    return value.GetBitValueInt64();
  }

  return absl::UnimplementedError("Resolve dim");
}

absl::StatusOr<DerefVariant> EvaluateToStructOrEnumOrAnnotation(
    TypeDefinition type_definition, InterpBindings* bindings,
    InterpCallbackData* callbacks) {
  while (absl::holds_alternative<TypeDef*>(type_definition)) {
    TypeDef* type_def = absl::get<TypeDef*>(type_definition);
    TypeAnnotation* annotation = type_def->type();
    if (auto* type_ref = dynamic_cast<TypeRefTypeAnnotation*>(annotation)) {
      type_definition = type_ref->type_ref()->type_definition();
    } else {
      return annotation;
    }
  }

  if (absl::holds_alternative<StructDef*>(type_definition)) {
    return absl::get<StructDef*>(type_definition);
  }
  if (absl::holds_alternative<EnumDef*>(type_definition)) {
    return absl::get<EnumDef*>(type_definition);
  }

  ModRef* modref = absl::get<ModRef*>(type_definition);
  XLS_ASSIGN_OR_RETURN(Module * imported_module,
                       bindings->ResolveModule(modref->import()->identifier()));
  XLS_ASSIGN_OR_RETURN(TypeDefinition td,
                       imported_module->GetTypeDefinition(modref->attr()));
  XLS_ASSIGN_OR_RETURN(
      InterpBindings imported_bindings,
      MakeTopLevelBindings(imported_module->shared_from_this(), callbacks));
  return EvaluateToStructOrEnumOrAnnotation(td, &imported_bindings, callbacks);
}

static absl::StatusOr<DerefVariant> DerefTypeRef(
    TypeRef* type_ref, InterpBindings* bindings,
    InterpCallbackData* callbacks) {
  if (absl::holds_alternative<ModRef*>(type_ref->type_definition())) {
    auto* mod_ref = absl::get<ModRef*>(type_ref->type_definition());
    return EvaluateToStructOrEnumOrAnnotation(mod_ref, bindings, callbacks);
  }

  XLS_ASSIGN_OR_RETURN(auto result,
                       bindings->ResolveTypeDefinition(type_ref->text()));
  return result;
}

// Returns new (derived) Bindings populated with `parametrics`.
//
// For example, if we have a struct defined as `struct [N: u32, M: u32] Foo`,
// and provided parametrics with values [A, 16], we'll create a new set of
// Bindings out of `bindings` and add (N, A) and (M, 16) to that.
//
// Args:
//   struct: The struct that may have parametric bindings.
//   parametrics: The parametric bindings that correspond to those on the
//     struct.
//   bindings: Bindings to use as the parent.
static absl::StatusOr<InterpBindings> BindingsWithStructParametrics(
    StructDef* struct_def, const std::vector<Expr*>& parametrics,
    InterpBindings* bindings) {
  InterpBindings nested_bindings(bindings->shared_from_this());
  XLS_CHECK_EQ(struct_def->parametric_bindings().size(), parametrics.size());
  for (int64 i = 0; i < parametrics.size(); ++i) {
    ParametricBinding* p = struct_def->parametric_bindings()[i];
    Expr* d = parametrics[i];
    if (Number* n = dynamic_cast<Number*>(d)) {
      int64 value = n->GetAsInt64().value();
      TypeAnnotation* type = n->type();
      if (type == nullptr) {
        // If the number didn't have a type annotation, use the one from the
        // parametric we're binding to.
        type = p->type();
      }
      XLS_RET_CHECK(type != nullptr)
          << "`" << n->ToString() << "` @ " << n->span();
      auto* builtin_type = dynamic_cast<BuiltinTypeAnnotation*>(type);
      XLS_CHECK(builtin_type != nullptr);
      int64 bit_count = builtin_type->GetBitCount();
      nested_bindings.AddValue(p->name_def()->identifier(),
                               InterpValue::MakeUBits(bit_count, value));
    } else {
      auto* name_ref = dynamic_cast<NameRef*>(d);
      XLS_CHECK(name_ref != nullptr)
          << d->GetNodeTypeName() << " " << d->ToString();
      InterpValue value =
          nested_bindings.ResolveValueFromIdentifier(name_ref->identifier())
              .value();
      nested_bindings.AddValue(p->name_def()->identifier(), value);
    }
  }
  return nested_bindings;
}

// Turns the various possible subtypes for a TypeAnnotation AST node into a
// concrete type.
absl::StatusOr<std::unique_ptr<ConcreteType>> ConcretizeTypeAnnotation(
    TypeAnnotation* type, InterpBindings* bindings,
    InterpCallbackData* callbacks) {
  XLS_VLOG(3) << "Concretizing type annotation: " << type->ToString();

  // class TypeRefTypeAnnotation
  if (auto* type_ref = dynamic_cast<TypeRefTypeAnnotation*>(type)) {
    XLS_ASSIGN_OR_RETURN(DerefVariant deref, DerefTypeRef(type_ref->type_ref(),
                                                          bindings, callbacks));
    absl::optional<InterpBindings> struct_parametric_bindings;
    if (type_ref->HasParametrics()) {
      XLS_RET_CHECK(absl::holds_alternative<StructDef*>(deref));
      auto* struct_def = absl::get<StructDef*>(deref);
      XLS_ASSIGN_OR_RETURN(struct_parametric_bindings,
                           BindingsWithStructParametrics(
                               struct_def, type_ref->parametrics(), bindings));
      bindings = &struct_parametric_bindings.value();
    }
    TypeDefinition type_defn = type_ref->type_ref()->type_definition();
    if (absl::holds_alternative<EnumDef*>(type_defn)) {
      auto* enum_def = absl::get<EnumDef*>(type_defn);
      XLS_ASSIGN_OR_RETURN(
          std::unique_ptr<ConcreteType> underlying_type,
          ConcretizeType(enum_def->type(), bindings, callbacks));
      XLS_ASSIGN_OR_RETURN(ConcreteTypeDim bit_count,
                           underlying_type->GetTotalBitCount());
      return absl::make_unique<EnumType>(enum_def, bit_count);
    }
    return ConcretizeType(deref, bindings, callbacks);
  }

  // class TupleTypeAnnotation
  if (auto* tuple = dynamic_cast<TupleTypeAnnotation*>(type)) {
    std::vector<std::unique_ptr<ConcreteType>> members;
    for (TypeAnnotation* member : tuple->members()) {
      XLS_ASSIGN_OR_RETURN(std::unique_ptr<ConcreteType> concrete_member,
                           ConcretizeType(member, bindings, callbacks));
      members.push_back(std::move(concrete_member));
    }
    return absl::make_unique<TupleType>(std::move(members));
  }

  // class ArrayTypeAnnotation
  if (auto* array = dynamic_cast<ArrayTypeAnnotation*>(type)) {
    XLS_ASSIGN_OR_RETURN(int64 dim, ResolveDim(array->dim(), bindings));
    TypeAnnotation* elem_type = array->element_type();
    XLS_VLOG(3) << "Resolved array dim to: " << dim
                << " elem_type: " << elem_type->ToString();
    if (auto* builtin_elem = dynamic_cast<BuiltinTypeAnnotation*>(elem_type);
        builtin_elem != nullptr && builtin_elem->GetBitCount() == 0) {
      return std::make_unique<BitsType>(builtin_elem->GetSignedness(), dim);
    }
    XLS_ASSIGN_OR_RETURN(std::unique_ptr<ConcreteType> concrete_elem_type,
                         ConcretizeType(elem_type, bindings, callbacks));
    return std::make_unique<ArrayType>(std::move(concrete_elem_type),
                                       ConcreteTypeDim(dim));
  }

  // class BuiltinTypeAnnotation
  if (auto* builtin = dynamic_cast<BuiltinTypeAnnotation*>(type)) {
    bool signedness = builtin->GetSignedness();
    int64 bit_count = builtin->GetBitCount();
    return absl::make_unique<BitsType>(signedness, bit_count);
  }

  return absl::UnimplementedError("Cannot concretize type annotation: " +
                                  type->ToString());
}

absl::StatusOr<std::unique_ptr<ConcreteType>> ConcretizeType(
    ConcretizeVariant type, InterpBindings* bindings,
    InterpCallbackData* callbacks) {
  // class EnumDef
  if (EnumDef** penum_def = absl::get_if<EnumDef*>(&type)) {
    return ConcretizeType((*penum_def)->type(), bindings, callbacks);
  }
  // class StructDef
  if (StructDef** pstruct_def = absl::get_if<StructDef*>(&type)) {
    std::vector<std::unique_ptr<ConcreteType>> members;
    for (auto& [name_def, type_annotation] : (*pstruct_def)->members()) {
      XLS_ASSIGN_OR_RETURN(
          std::unique_ptr<ConcreteType> concretized,
          ConcretizeTypeAnnotation(type_annotation, bindings, callbacks));
      members.push_back(std::move(concretized));
    }
    return absl::make_unique<TupleType>(std::move(members));
  }
  // class TypeAnnotation
  return ConcretizeTypeAnnotation(absl::get<TypeAnnotation*>(type), bindings,
                                  callbacks);
}

absl::StatusOr<EnumDef*> EvaluateToEnum(TypeDefinition type_definition,
                                        InterpBindings* bindings,
                                        InterpCallbackData* callbacks) {
  XLS_ASSIGN_OR_RETURN(
      DerefVariant deref,
      EvaluateToStructOrEnumOrAnnotation(type_definition, bindings, callbacks));
  if (absl::holds_alternative<EnumDef*>(deref)) {
    return absl::get<EnumDef*>(deref);
  }
  return absl::InvalidArgumentError(
      absl::StrCat("Type definition did not dereference to an enum, found: ",
                   ToAstNode(deref)->GetNodeTypeName()));
}

}  // namespace xls::dslx
