// Copyright 2021 The XLS Authors
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
#include "xls/dslx/constexpr_evaluator.h"

#include <variant>

#include "absl/strings/match.h"
#include "absl/types/variant.h"
#include "xls/dslx/ast_utils.h"
#include "xls/dslx/builtins_metadata.h"
#include "xls/dslx/bytecode_emitter.h"
#include "xls/dslx/bytecode_interpreter.h"
#include "xls/dslx/errors.h"
#include "xls/dslx/interp_value.h"

namespace xls::dslx {
namespace {

// Visitor to collect all NameRefs defined externally to a given expression,
// notably a "for" expression. Only those nodes capable of containing an outside
// NameRef are populated, e.g., `Number` isn't populated.
class NameRefCollector : public ExprVisitor {
 public:
  absl::Status HandleArray(const Array* expr) override {
    for (const auto* member : expr->members()) {
      XLS_RETURN_IF_ERROR(member->AcceptExpr(this));
    }
    return absl::OkStatus();
  }
  absl::Status HandleAttr(const Attr* expr) override {
    XLS_RETURN_IF_ERROR(expr->lhs()->AcceptExpr(this));
    return absl::OkStatus();
  }
  absl::Status HandleBinop(const Binop* expr) override {
    XLS_RETURN_IF_ERROR(expr->lhs()->AcceptExpr(this));
    XLS_RETURN_IF_ERROR(expr->rhs()->AcceptExpr(this));
    return absl::OkStatus();
  }
  absl::Status HandleCast(const Cast* expr) override {
    XLS_RETURN_IF_ERROR(expr->expr()->AcceptExpr(this));
    return absl::OkStatus();
  }
  absl::Status HandleChannelDecl(const ChannelDecl* expr) override {
    return absl::OkStatus();
  }
  absl::Status HandleColonRef(const ColonRef* expr) override {
    return absl::OkStatus();
  }
  absl::Status HandleConstRef(const ConstRef* expr) override {
    XLS_RETURN_IF_ERROR(expr->GetValue()->AcceptExpr(this));
    return absl::OkStatus();
  }
  absl::Status HandleFor(const For* expr) override {
    XLS_RETURN_IF_ERROR(expr->init()->AcceptExpr(this));
    XLS_RETURN_IF_ERROR(expr->body()->AcceptExpr(this));
    return absl::OkStatus();
  }
  absl::Status HandleFormatMacro(const FormatMacro* expr) override {
    return absl::OkStatus();
  }
  absl::Status HandleIndex(const Index* expr) override {
    XLS_RETURN_IF_ERROR(expr->lhs()->AcceptExpr(this));
    if (std::holds_alternative<Expr*>(expr->rhs())) {
      XLS_RETURN_IF_ERROR(std::get<Expr*>(expr->rhs())->AcceptExpr(this));
    }
    // No NameRefs in slice RHSes.
    return absl::OkStatus();
  }
  absl::Status HandleInvocation(const Invocation* expr) override {
    for (const auto* arg : expr->args()) {
      XLS_RETURN_IF_ERROR(arg->AcceptExpr(this));
    }
    return absl::OkStatus();
  }
  absl::Status HandleJoin(const Join* expr) override {
    return absl::OkStatus();
  }
  absl::Status HandleLet(const Let* expr) override {
    XLS_RETURN_IF_ERROR(expr->rhs()->AcceptExpr(this));
    XLS_RETURN_IF_ERROR(expr->body()->AcceptExpr(this));

    std::vector<NameDefTree::Leaf> leaves = expr->name_def_tree()->Flatten();
    for (const auto& leaf : leaves) {
      if (std::holds_alternative<NameDef*>(leaf)) {
        name_defs_.insert(std::get<NameDef*>(leaf));
      }
    }
    return absl::OkStatus();
  }
  absl::Status HandleMatch(const Match* expr) override {
    XLS_RETURN_IF_ERROR(expr->matched()->AcceptExpr(this));
    for (const MatchArm* arm : expr->arms()) {
      XLS_RETURN_IF_ERROR(arm->expr()->AcceptExpr(this));
    }
    return absl::OkStatus();
  }
  absl::Status HandleNameRef(const NameRef* expr) override {
    name_refs_.push_back(expr);
    return absl::OkStatus();
  }
  absl::Status HandleNumber(const Number* expr) override {
    return absl::OkStatus();
  }
  absl::Status HandleRecv(const Recv* expr) override {
    return absl::OkStatus();
  }
  absl::Status HandleRecvIf(const RecvIf* expr) override {
    XLS_RETURN_IF_ERROR(expr->condition()->AcceptExpr(this));
    return absl::OkStatus();
  }
  absl::Status HandleSend(const Send* expr) override {
    XLS_RETURN_IF_ERROR(expr->payload()->AcceptExpr(this));
    return absl::OkStatus();
  }
  absl::Status HandleSendIf(const SendIf* expr) override {
    XLS_RETURN_IF_ERROR(expr->condition()->AcceptExpr(this));
    XLS_RETURN_IF_ERROR(expr->payload()->AcceptExpr(this));
    return absl::OkStatus();
  }
  absl::Status HandleSpawn(const Spawn* expr) override {
    return absl::OkStatus();
  }
  absl::Status HandleString(const String* expr) override {
    return absl::OkStatus();
  }
  absl::Status HandleSplatStructInstance(
      const SplatStructInstance* expr) override {
    for (const auto& [name, member_expr] : expr->members()) {
      XLS_RETURN_IF_ERROR(member_expr->AcceptExpr(this));
    }
    XLS_RETURN_IF_ERROR(expr->splatted()->AcceptExpr(this));
    return absl::OkStatus();
  }
  absl::Status HandleStructInstance(const StructInstance* expr) override {
    for (const auto& [name, member_expr] : expr->GetUnorderedMembers()) {
      XLS_RETURN_IF_ERROR(member_expr->AcceptExpr(this));
    }
    return absl::OkStatus();
  }
  absl::Status HandleTernary(const Ternary* expr) override {
    XLS_RETURN_IF_ERROR(expr->test()->AcceptExpr(this));
    XLS_RETURN_IF_ERROR(expr->consequent()->AcceptExpr(this));
    XLS_RETURN_IF_ERROR(expr->alternate()->AcceptExpr(this));
    return absl::OkStatus();
  }
  absl::Status HandleUnop(const Unop* expr) override {
    XLS_RETURN_IF_ERROR(expr->operand()->AcceptExpr(this));
    return absl::OkStatus();
  }
  absl::Status HandleXlsTuple(const XlsTuple* expr) override {
    for (const Expr* member : expr->members()) {
      XLS_RETURN_IF_ERROR(member->AcceptExpr(this));
    }
    return absl::OkStatus();
  }

  std::vector<const NameRef*> outside_name_refs() {
    std::vector<const NameRef*> result;
    for (const NameRef* name_ref : name_refs_) {
      if (std::holds_alternative<BuiltinNameDef*>(name_ref->name_def())) {
        continue;
      }

      if (!name_defs_.contains(std::get<NameDef*>(name_ref->name_def())) &&
          !IsNameParametricBuiltin(name_ref->identifier())) {
        result.push_back(name_ref);
      }
    }

    return result;
  }

 private:
  std::vector<const NameRef*> name_refs_;
  absl::flat_hash_set<const NameDef*> name_defs_;
};

// Fully instantiate the given parametric BitsType using the symbol mappings in
// `env`.
absl::StatusOr<std::unique_ptr<BitsType>> InstantiateParametricNumberType(
    const absl::flat_hash_map<std::string, InterpValue>& env,
    const BitsType* bits_type) {
  ParametricExpression::Env parametric_env;
  for (const auto& [k, v] : env) {
    parametric_env[k] = v;
  }
  ParametricExpression::Evaluated e =
      bits_type->size().parametric().Evaluate(parametric_env);
  if (!absl::holds_alternative<InterpValue>(e)) {
    return absl::InternalError(
        absl::StrCat("Parametric number size did not evaluate to a constant: ",
                     bits_type->size().ToString()));
  }
  return std::make_unique<BitsType>(
      bits_type->is_signed(),
      absl::get<InterpValue>(e).GetBitValueInt64().value());
}

}  // namespace

bool ConstexprEvaluator::IsConstExpr(const Expr* expr) {
  return ctx_->type_info()->GetConstExpr(expr).has_value();
}

absl::Status ConstexprEvaluator::HandleAttr(const Attr* expr) {
  if (IsConstExpr(expr->lhs())) {
    XLS_RETURN_IF_ERROR(InterpretExpr(expr));
  }

  return absl::OkStatus();
}

absl::Status ConstexprEvaluator::HandleArray(const Array* expr) {
  std::vector<InterpValue> values;
  for (const Expr* member : expr->members()) {
    absl::optional<InterpValue> maybe_value =
        ctx_->type_info()->GetConstExpr(member);
    if (!maybe_value.has_value()) {
      return absl::OkStatus();
    }

    values.push_back(maybe_value.value());
  }

  if (concrete_type_ != nullptr) {
    auto* array_type = dynamic_cast<const ArrayType*>(concrete_type_);
    if (array_type == nullptr) {
      return absl::InternalError(
          absl::StrCat(expr->span().ToString(), " : ",
                       "Array ConcreteType was not an ArrayType!"));
    }

    ConcreteTypeDim size = array_type->size();
    absl::StatusOr<int64_t> int_size_or = size.GetAsInt64();
    if (!int_size_or.ok()) {
      return absl::InternalError(absl::StrCat(expr->span().ToString(), " : ",
                                              int_size_or.status().message()));
    }

    int64_t int_size = int_size_or.value();
    int64_t remaining = int_size - values.size();
    while (remaining-- > 0) {
      values.push_back(values.back());
    }
  }

  // No need to fire up the interpreter. We can handle this one.
  XLS_ASSIGN_OR_RETURN(InterpValue array, InterpValue::MakeArray(values));
  ctx_->type_info()->NoteConstExpr(expr, array);
  return absl::OkStatus();
}

absl::Status ConstexprEvaluator::HandleBinop(const Binop* expr) {
  XLS_VLOG(3) << "ConstexprEvaluator::HandleBinop : " << expr->ToString();
  if (IsConstExpr(expr->lhs()) && IsConstExpr(expr->rhs())) {
    XLS_RETURN_IF_ERROR(InterpretExpr(expr));
  }
  return absl::OkStatus();
}

absl::Status ConstexprEvaluator::HandleCast(const Cast* expr) {
  XLS_VLOG(3) << "ConstexprEvaluator::HandleCast : " << expr->ToString();
  if (IsConstExpr(expr->expr())) {
    XLS_RETURN_IF_ERROR(InterpretExpr(expr));
  }
  return absl::OkStatus();
}

// Creates an InterpValue for the described channel or array of channels.
absl::StatusOr<InterpValue> CreateChannelValue(
    const ConcreteType* concrete_type) {
  if (auto* array_type = dynamic_cast<const ArrayType*>(concrete_type)) {
    XLS_ASSIGN_OR_RETURN(int dim_int, array_type->size().GetAsInt64());
    std::vector<InterpValue> elements;
    elements.reserve(dim_int);
    for (int i = 0; i < dim_int; i++) {
      XLS_ASSIGN_OR_RETURN(InterpValue element,
                           CreateChannelValue(&array_type->element_type()));
      elements.push_back(element);
    }
    return InterpValue::MakeArray(elements);
  }

  // There can't be tuples or structs of channels, only arrays.
  const ChannelType* ct = dynamic_cast<const ChannelType*>(concrete_type);
  XLS_RET_CHECK_NE(ct, nullptr);
  return InterpValue::MakeChannel();
}

// While a channel's *contents* aren't constexpr, the existence of the channel
// itself is.
absl::Status ConstexprEvaluator::HandleChannelDecl(const ChannelDecl* expr) {
  XLS_VLOG(3) << "ConstexprEvaluator::HandleChannelDecl : " << expr->ToString();
  // Keep in mind that channels come in tuples, so peel out the first element.
  absl::optional<ConcreteType*> maybe_decl_type =
      ctx_->type_info()->GetItem(expr);
  if (!maybe_decl_type.has_value()) {
    return absl::InternalError(
        absl::StrFormat("Could not find type for expr \"%s\" @ %s",
                        expr->ToString(), expr->span().ToString()));
  }

  auto* tuple_type = dynamic_cast<TupleType*>(maybe_decl_type.value());
  if (tuple_type == nullptr) {
    return TypeInferenceErrorStatus(expr->span(), maybe_decl_type.value(),
                                    "Channel decl did not have tuple type:");
  }

  // Verify that the channel tuple has exactly two elements; just yank one out
  // for channel [array] creation (they both point to the same object).
  if (tuple_type->size() != 2) {
    return TypeInferenceErrorStatus(
        expr->span(), tuple_type, "ChannelDecl type was a two-element tuple.");
  }

  absl::StatusOr<InterpValue> channels =
      CreateChannelValue(&tuple_type->GetMemberType(0));
  ctx_->type_info()->NoteConstExpr(
      expr, InterpValue::MakeTuple({channels.value(), channels.value()}));
  return absl::OkStatus();
}

absl::Status ConstexprEvaluator::HandleColonRef(const ColonRef* expr) {
  TypeInfo* type_info = ctx_->type_info();
  XLS_ASSIGN_OR_RETURN(
      auto subject,
      ResolveColonRefSubject(ctx_->import_data(), type_info, expr));

  if (absl::holds_alternative<EnumDef*>(subject)) {
    // LHS is an EnumDef! Extract the value of the attr being referenced.
    EnumDef* enum_def = absl::get<EnumDef*>(subject);
    XLS_ASSIGN_OR_RETURN(Expr * member_value_expr,
                         enum_def->GetValue(expr->attr()));

    // Since enum defs can't [currently] be parameterized, this is safe.
    XLS_ASSIGN_OR_RETURN(TypeInfo * type_info,
                         ctx_->import_data()->GetRootTypeInfoForNode(enum_def));

    absl::optional<ConcreteType*> maybe_concrete_type =
        type_info->GetItem(enum_def);
    if (!maybe_concrete_type.has_value()) {
      return absl::InternalError(absl::StrCat(
          "Could not find concrete type for EnumDef: ", enum_def->ToString()));
    }

    absl::optional<InterpValue> maybe_final_value =
        type_info->GetConstExpr(member_value_expr);
    if (!maybe_final_value.has_value()) {
      return absl::InternalError(absl::StrCat("Failed to constexpr evaluate: ",
                                              member_value_expr->ToString()));
    }
    ctx_->type_info()->NoteConstExpr(expr, maybe_final_value.value());
    return absl::OkStatus();
  }

  // Ok! The subject is a module. The only case we care about here is if the
  // attr is a constant.
  Module* module = absl::get<Module*>(subject);
  absl::optional<ModuleMember*> maybe_member =
      module->FindMemberWithName(expr->attr());
  if (!maybe_member.has_value()) {
    return absl::InternalError(
        absl::StrFormat("\"%s\" is not a member of module \"%s\".",
                        expr->attr(), module->name()));
  }

  if (!absl::holds_alternative<ConstantDef*>(*maybe_member.value())) {
    XLS_VLOG(3) << "ConstRef \"" << expr->ToString()
                << "\" is not constexpr evaluatable.";
    return absl::OkStatus();
  }

  XLS_ASSIGN_OR_RETURN(type_info, ctx_->import_data()->GetRootTypeInfo(module));

  ConstantDef* constant_def = absl::get<ConstantDef*>(*maybe_member.value());
  absl::optional<InterpValue> maybe_value =
      type_info->GetConstExpr(constant_def->value());
  if (!maybe_value.has_value()) {
    return absl::InternalError(
        absl::StrCat("Could not find constexpr value for ConstantDef \"",
                     constant_def->ToString(), "\"."));
  }

  ctx_->type_info()->NoteConstExpr(expr, maybe_value.value());
  return absl::OkStatus();
}

absl::Status ConstexprEvaluator::HandleConstRef(const ConstRef* expr) {
  return HandleNameRef(expr);
}

absl::Status ConstexprEvaluator::HandleFor(const For* expr) {
  // A `for` loop evaluates constexpr if its init and enumeration values as
  // well as any external NameRefs are constexpr.
  XLS_VLOG(3) << "ConstexprEvaluator::HandleFor: " << expr->ToString();
  if (!IsConstExpr(expr->init()) || !IsConstExpr(expr->iterable())) {
    return absl::OkStatus();
  }

  // Since a `for` loop can refer to vars outside the loop body itself, we need
  // to make sure that every NameRef is also constexpr.
  std::vector<NameDefTree::Leaf> bound_refs = expr->names()->Flatten();
  absl::flat_hash_set<const NameDef*> bound_defs;
  for (const auto& leaf : bound_refs) {
    if (std::holds_alternative<NameDef*>(leaf)) {
      bound_defs.insert(std::get<NameDef*>(leaf));
    }
  }

  NameRefCollector collector;
  XLS_RETURN_IF_ERROR(expr->body()->AcceptExpr(&collector));
  for (const NameRef* name_ref : collector.outside_name_refs()) {
    // We can't bind to a BuiltinNameDef, so this std::get is safe.
    if (!IsConstExpr(name_ref) &&
        !bound_defs.contains(std::get<NameDef*>(name_ref->name_def()))) {
      return absl::OkStatus();
    }
  }

  // We don't [yet] have a static assert fn, meaning that we don't want to catch
  // runtime errors here. If we detect that a program has failed (due to
  // execution of a `fail!` or unmatched `match`, then just assume we're ok.
  absl::Status status = InterpretExpr(expr, bound_defs);
  if (!status.ok() && !absl::StartsWith(status.message(), "FailureError")) {
    return status;
  }
  return absl::OkStatus();
}

absl::Status ConstexprEvaluator::HandleIndex(const Index* expr) {
  XLS_VLOG(3) << "ConstexprEvaluator::HandleIndex : " << expr->ToString();
  bool rhs_is_constexpr;
  if (absl::holds_alternative<Expr*>(expr->rhs())) {
    rhs_is_constexpr = IsConstExpr(absl::get<Expr*>(expr->rhs()));
  } else if (absl::holds_alternative<Slice*>(expr->rhs())) {
    Slice* slice = absl::get<Slice*>(expr->rhs());
    rhs_is_constexpr =
        IsConstExpr(slice->start()) && IsConstExpr(slice->limit());
  } else {
    WidthSlice* width_slice = absl::get<WidthSlice*>(expr->rhs());
    rhs_is_constexpr = IsConstExpr(width_slice->start());
  }

  if (IsConstExpr(expr->lhs()) && rhs_is_constexpr) {
    return InterpretExpr(expr);
  }

  return absl::OkStatus();
}

absl::Status ConstexprEvaluator::HandleInvocation(const Invocation* expr) {
  // Map "invocations" are special - only the first (of two) args must be
  // constexpr (the second must be a fn to apply).
  auto* callee_name_ref = dynamic_cast<NameRef*>(expr->callee());
  bool callee_is_map =
      callee_name_ref != nullptr && callee_name_ref->identifier() == "map";
  if (callee_is_map) {
    if (!IsConstExpr(expr->args()[0])) {
      return absl::OkStatus();
    }
  } else {
    // A regular invocation is constexpr iff its args are constexpr.
    for (const auto* arg : expr->args()) {
      if (!IsConstExpr(arg)) {
        return absl::OkStatus();
      }
    }
  }

  // We don't [yet] have a static assert fn, meaning that we don't want to catch
  // runtime errors here. If we detect that a program has failed (due to
  // execution of a `fail!` or unmatched `match`, then just assume we're ok.
  absl::Status status = InterpretExpr(expr);
  if (!status.ok() && !absl::StartsWith(status.message(), "FailureError")) {
    return status;
  }

  return absl::OkStatus();
}

absl::Status ConstexprEvaluator::HandleMatch(const Match* expr) {
  if (!IsConstExpr(expr->matched())) {
    return absl::OkStatus();
  }

  for (const auto* arm : expr->arms()) {
    if (!IsConstExpr(arm->expr())) {
      return absl::OkStatus();
    }
  }

  return InterpretExpr(expr);
}

absl::Status ConstexprEvaluator::HandleNameRef(const NameRef* expr) {
  absl::optional<InterpValue> constexpr_value =
      ctx_->type_info()->GetConstExpr(ToAstNode(expr->name_def()));

  if (constexpr_value.has_value()) {
    ctx_->type_info()->NoteConstExpr(expr, constexpr_value.value());
  }
  return absl::OkStatus();
}

// Evaluates a Number AST node to an InterpValue.
absl::StatusOr<InterpValue> EvaluateNumber(const Number* expr,
                                           const ConcreteType* type) {
  XLS_VLOG(4) << "Evaluating number: " << expr->ToString() << " @ "
              << expr->span();
  const BitsType* bits_type = dynamic_cast<const BitsType*>(type);
  XLS_RET_CHECK(bits_type != nullptr)
      << "Type for number should be 'bits' kind.";
  InterpValueTag tag =
      bits_type->is_signed() ? InterpValueTag::kSBits : InterpValueTag::kUBits;
  XLS_ASSIGN_OR_RETURN(
      int64_t bit_count,
      absl::get<InterpValue>(bits_type->size().value()).GetBitValueInt64());
  XLS_ASSIGN_OR_RETURN(Bits bits, expr->GetBits(bit_count));
  return InterpValue::MakeBits(tag, std::move(bits));
}

absl::Status ConstexprEvaluator::HandleNumber(const Number* expr) {
  // Numbers should always be [constexpr] evaluatable.
  absl::flat_hash_map<std::string, InterpValue> env;
  if (!ctx_->fn_stack().empty()) {
    env = MakeConstexprEnv(expr, ctx_->fn_stack().back().symbolic_bindings(),
                           ctx_->type_info());
  }

  std::unique_ptr<BitsType> temp_type;
  const ConcreteType* type_ptr;
  if (expr->type_annotation() != nullptr) {
    // If the number is annotated with a type, then extract it to pass to
    // EvaluateNumber (for consistency checking). It might be that the type is
    // parametric, in which case we'll need to fully instantiate it.
    type_ptr = ctx_->type_info()->GetItem(expr->type_annotation()).value();
    const BitsType* bt = down_cast<const BitsType*>(type_ptr);
    if (bt->size().IsParametric()) {
      XLS_ASSIGN_OR_RETURN(temp_type, InstantiateParametricNumberType(env, bt));
      type_ptr = temp_type.get();
    }
  } else if (concrete_type_ != nullptr) {
    type_ptr = concrete_type_;
  } else if (expr->number_kind() == NumberKind::kBool) {
    temp_type = std::make_unique<BitsType>(false, 1);
    type_ptr = temp_type.get();
  } else if (expr->number_kind() == NumberKind::kCharacter) {
    temp_type = std::make_unique<BitsType>(false, 8);
    type_ptr = temp_type.get();
  } else {
    // "Undecorated" numbers that make it through typechecking are `usize`,
    // which currently is u32.
    temp_type = std::make_unique<BitsType>(false, 32);
    type_ptr = temp_type.get();
  }

  XLS_ASSIGN_OR_RETURN(InterpValue value, EvaluateNumber(expr, type_ptr));
  ctx_->type_info()->NoteConstExpr(expr, value);

  return absl::OkStatus();
}

absl::Status ConstexprEvaluator::HandleSplatStructInstance(
    const SplatStructInstance* expr) {
  // A struct instance is constexpr iff all its members and the basis struct are
  // constexpr.
  if (!IsConstExpr(expr->splatted())) {
    return absl::OkStatus();
  }

  for (const auto& [k, v] : expr->members()) {
    if (!IsConstExpr(v)) {
      return absl::OkStatus();
    }
  }
  return InterpretExpr(expr);
}

absl::Status ConstexprEvaluator::HandleStructInstance(
    const StructInstance* expr) {
  // A struct instance is constexpr iff all its members are constexpr.
  for (const auto& [k, v] : expr->GetUnorderedMembers()) {
    if (!IsConstExpr(v)) {
      return absl::OkStatus();
    }
  }
  return InterpretExpr(expr);
}

absl::Status ConstexprEvaluator::HandleTernary(const Ternary* expr) {
  // Simple enough that we don't need to invoke the interpreter.
  if (!IsConstExpr(expr->test()) || !IsConstExpr(expr->consequent()) ||
      !IsConstExpr(expr->alternate())) {
    return absl::OkStatus();
  }

  TypeInfo* type_info = ctx_->type_info();
  InterpValue test = type_info->GetConstExpr(expr->test()).value();
  if (test.IsTrue()) {
    type_info->NoteConstExpr(
        expr, type_info->GetConstExpr(expr->consequent()).value());
  } else {
    type_info->NoteConstExpr(
        expr, type_info->GetConstExpr(expr->alternate()).value());
  }

  return absl::OkStatus();
}

absl::Status ConstexprEvaluator::HandleUnop(const Unop* expr) {
  if (!IsConstExpr(expr->operand())) {
    return absl::OkStatus();
  }

  return InterpretExpr(expr);
}

absl::Status ConstexprEvaluator::HandleXlsTuple(const XlsTuple* expr) {
  std::vector<InterpValue> values;
  for (const Expr* member : expr->members()) {
    absl::optional<InterpValue> maybe_value =
        ctx_->type_info()->GetConstExpr(member);
    if (!maybe_value.has_value()) {
      return absl::OkStatus();
    }

    values.push_back(maybe_value.value());
  }

  // No need to fire up the interpreter. We can handle this one.
  ctx_->type_info()->NoteConstExpr(expr, InterpValue::MakeTuple(values));
  return absl::OkStatus();
}

absl::Status ConstexprEvaluator::InterpretExpr(
    const Expr* expr, absl::flat_hash_set<const NameDef*> bypass_env) {
  absl::optional<FnCtx> fn_ctx;
  absl::flat_hash_map<std::string, InterpValue> env;
  if (!ctx_->fn_stack().empty()) {
    env = MakeConstexprEnv(expr, ctx_->fn_stack().back().symbolic_bindings(),
                           ctx_->type_info(), bypass_env);

    const FnStackEntry& peek_entry = ctx_->fn_stack().back();
    if (peek_entry.f() != nullptr) {
      fn_ctx.emplace(FnCtx{peek_entry.module()->name(), peek_entry.name(),
                           peek_entry.symbolic_bindings()});
    }
  }

  SymbolicBindings symbolic_bindings;
  if (!ctx_->fn_stack().empty()) {
    const FnStackEntry& entry = ctx_->fn_stack().back();
    symbolic_bindings = entry.symbolic_bindings();
  }

  XLS_ASSIGN_OR_RETURN(
      std::unique_ptr<BytecodeFunction> bf,
      BytecodeEmitter::EmitExpression(ctx_->import_data(), ctx_->type_info(),
                                      expr, env, symbolic_bindings));

  XLS_ASSIGN_OR_RETURN(
      InterpValue constexpr_value,
      BytecodeInterpreter::Interpret(ctx_->import_data(), bf.get(),
                                     /*args=*/{}));
  ctx_->type_info()->NoteConstExpr(expr, constexpr_value);

  return absl::OkStatus();
}

}  // namespace xls::dslx
