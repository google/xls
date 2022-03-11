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

#include "absl/strings/match.h"
#include "absl/types/variant.h"
#include "xls/dslx/ast_utils.h"
#include "xls/dslx/bytecode_emitter.h"
#include "xls/dslx/bytecode_interpreter.h"
#include "xls/dslx/evaluate.h"
#include "xls/dslx/interp_value.h"

namespace xls::dslx {
namespace {

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

void ConstexprEvaluator::HandleAttr(Attr* expr) {
  if (IsConstExpr(expr->lhs())) {
    status_.Update(SimpleEvaluate(expr));
  }
}

void ConstexprEvaluator::HandleArray(Array* expr) {
  std::vector<InterpValue> values;
  for (const Expr* member : expr->members()) {
    absl::optional<InterpValue> maybe_value =
        ctx_->type_info()->GetConstExpr(member);
    if (!maybe_value.has_value()) {
      return;
    }

    values.push_back(maybe_value.value());
  }

  if (concrete_type_ != nullptr) {
    auto* array_type = dynamic_cast<const ArrayType*>(concrete_type_);
    if (array_type == nullptr) {
      status_ = absl::InternalError(
          absl::StrCat(expr->span().ToString(), " : ",
                       "Array ConcreteType was not an ArrayType!"));
      return;
    }

    ConcreteTypeDim size = array_type->size();
    absl::StatusOr<int64_t> int_size_or = size.GetAsInt64();
    if (!int_size_or.ok()) {
      status_ = absl::InternalError(absl::StrCat(
          expr->span().ToString(), " : ", int_size_or.status().message()));
      return;
    }

    int64_t int_size = int_size_or.value();
    int64_t remaining = int_size - values.size();
    while (remaining-- > 0) {
      values.push_back(values.back());
    }
  }

  // No need to fire up the interpreter. We can handle this one.
  absl::StatusOr<InterpValue> array_or(InterpValue::MakeArray(values));
  if (!array_or.ok()) {
    status_ = array_or.status();
    return;
  }

  ctx_->type_info()->NoteConstExpr(expr, array_or.value());
}

void ConstexprEvaluator::HandleBinop(Binop* expr) {
  XLS_VLOG(3) << "ConstexprEvaluator::HandleBinop : " << expr->ToString();
  if (IsConstExpr(expr->lhs()) && IsConstExpr(expr->rhs())) {
    status_.Update(SimpleEvaluate(expr));
  }
}

void ConstexprEvaluator::HandleCast(Cast* expr) {
  XLS_VLOG(3) << "ConstexprEvaluator::HandleCast : " << expr->ToString();
  if (IsConstExpr(expr->expr())) {
    status_ = SimpleEvaluate(expr);
  }
}

void ConstexprEvaluator::HandleColonRef(ColonRef* expr) {
  TypeInfo* type_info = ctx_->type_info();
  absl::StatusOr<absl::variant<Module*, EnumDef*>> subject_or =
      ResolveColonRefSubject(ctx_->import_data(), type_info, expr);
  if (!subject_or.ok()) {
    status_ = subject_or.status();
    return;
  }

  auto subject = subject_or.value();
  if (absl::holds_alternative<EnumDef*>(subject)) {
    // LHS is an EnumDef! Extract the value of the attr being referenced.
    EnumDef* enum_def = absl::get<EnumDef*>(subject);
    absl::StatusOr<Expr*> member_value_expr_or =
        enum_def->GetValue(expr->attr());
    if (!member_value_expr_or.ok()) {
      status_ = member_value_expr_or.status();
      return;
    }

    // Since enum defs can't [currently] be parameterized, this is safe.
    absl::StatusOr<TypeInfo*> type_info_or =
        ctx_->import_data()->GetRootTypeInfoForNode(enum_def);
    if (!type_info_or.ok()) {
      status_ = type_info_or.status();
      return;
    }
    type_info = type_info_or.value();

    Expr* member_value_expr = member_value_expr_or.value();
    absl::optional<ConcreteType*> maybe_concrete_type =
        type_info->GetItem(enum_def);
    if (!maybe_concrete_type.has_value()) {
      status_ = absl::InternalError(absl::StrCat(
          "Could not find concrete type for EnumDef: ", enum_def->ToString()));
      return;
    }

    absl::optional<InterpValue> maybe_final_value =
        type_info->GetConstExpr(member_value_expr);
    if (!maybe_final_value.has_value()) {
      status_ = absl::InternalError(absl::StrCat(
          "Failed to constexpr evaluate: ", member_value_expr->ToString()));
    }
    ctx_->type_info()->NoteConstExpr(expr, maybe_final_value.value());
    return;
  }

  // Ok! The subject is a module. The only case we care about here is if the
  // attr is a constant.
  Module* module = absl::get<Module*>(subject);
  absl::optional<ModuleMember*> maybe_member =
      module->FindMemberWithName(expr->attr());
  if (!maybe_member.has_value()) {
    status_ = absl::InternalError(
        absl::StrFormat("\"%s\" is not a member of module \"%s\".",
                        expr->attr(), module->name()));
    return;
  }
  if (!absl::holds_alternative<ConstantDef*>(*maybe_member.value())) {
    XLS_VLOG(3) << "ConstRef \"" << expr->ToString()
                << "\" is not constexpr evaluatable.";
    return;
  }

  absl::StatusOr<TypeInfo*> type_info_or =
      ctx_->import_data()->GetRootTypeInfo(module);
  if (!type_info_or.ok()) {
    status_ = absl::InternalError(absl::StrCat(
        "Could not find type info for module \"", module->name(), "\"."));
    return;
  }
  type_info = type_info_or.value();

  ConstantDef* constant_def = absl::get<ConstantDef*>(*maybe_member.value());
  absl::optional<InterpValue> maybe_value =
      type_info->GetConstExpr(constant_def->value());
  if (!maybe_value.has_value()) {
    status_ = absl::InternalError(
        absl::StrCat("Could not find constexpr value for ConstantDef \"",
                     constant_def->ToString(), "\"."));
    return;
  }

  ctx_->type_info()->NoteConstExpr(expr, maybe_value.value());
}

void ConstexprEvaluator::HandleConstRef(ConstRef* expr) {
  return HandleNameRef(expr);
}

void ConstexprEvaluator::HandleIndex(Index* expr) {
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
    status_ = SimpleEvaluate(expr);
  }
}

void ConstexprEvaluator::HandleInvocation(Invocation* expr) {
  // Map "invocations" are special - only the first (of two) args must be
  // constexpr (the second must be a fn to apply).
  auto* callee_name_ref = dynamic_cast<NameRef*>(expr->callee());
  bool callee_is_map =
      callee_name_ref != nullptr && callee_name_ref->identifier() == "map";
  if (callee_is_map) {
    if (!IsConstExpr(expr->args()[0])) {
      return;
    }
  } else {
    // A regular invocation is constexpr iff its args are constexpr.
    for (const auto* arg : expr->args()) {
      if (!IsConstExpr(arg)) {
        return;
      }
    }
  }

  // We don't [yet] have a static assert fn, meaning that we don't want to catch
  // runtime errors here. If we detect that a program has failed (due to
  // execution of a `fail!` or unmatched `match`, then just assume we're ok.
  status_ = SimpleEvaluate(expr);
  if (!status_.ok()) {
    if (absl::StartsWith(status_.message(), "FailureError")) {
      status_ = absl::OkStatus();
    }
  }
}

void ConstexprEvaluator::HandleNameRef(NameRef* expr) {
  absl::optional<InterpValue> constexpr_value =
      ctx_->type_info()->GetConstExpr(ToAstNode(expr->name_def()));

  if (constexpr_value.has_value()) {
    ctx_->type_info()->NoteConstExpr(expr, constexpr_value.value());
  }
}

void ConstexprEvaluator::HandleNumber(Number* expr) {
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
      absl::StatusOr<std::unique_ptr<BitsType>> temp_type_or =
          InstantiateParametricNumberType(env, bt);
      if (!temp_type_or.ok()) {
        status_ = temp_type_or.status();
        return;
      }
      temp_type = std::move(temp_type_or.value());
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

  // Evaluating a number with a type context doesn't require bindings or an
  // interpreter.
  absl::StatusOr<InterpValue> value =
      EvaluateNumber(expr, /*bindings=*/nullptr, type_ptr, /*interp=*/nullptr);
  status_ = value.status();
  if (value.ok()) {
    ctx_->type_info()->NoteConstExpr(expr, value.value());
  }
}

void ConstexprEvaluator::HandleStructInstance(StructInstance* expr) {
  // A struct instance is constexpr iff all its members are constexpr.
  for (const auto& [k, v] : expr->GetUnorderedMembers()) {
    if (!IsConstExpr(v)) {
      return;
    }
  }
  status_ = SimpleEvaluate(expr);
}

void ConstexprEvaluator::HandleTernary(Ternary* expr) {
  // Simple enough that we don't need to invoke the interpreter.
  if (!IsConstExpr(expr->test()) || !IsConstExpr(expr->consequent()) ||
      !IsConstExpr(expr->alternate())) {
    return;
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
}

void ConstexprEvaluator::HandleXlsTuple(XlsTuple* expr) {
  std::vector<InterpValue> values;
  for (const Expr* member : expr->members()) {
    absl::optional<InterpValue> maybe_value =
        ctx_->type_info()->GetConstExpr(member);
    if (!maybe_value.has_value()) {
      return;
    }

    values.push_back(maybe_value.value());
  }

  // No need to fire up the interpreter. We can handle this one.
  ctx_->type_info()->NoteConstExpr(expr, InterpValue::MakeTuple(values));
}

absl::Status ConstexprEvaluator::SimpleEvaluate(Expr* expr) {
  absl::optional<FnCtx> fn_ctx;
  absl::flat_hash_map<std::string, InterpValue> env;
  if (!ctx_->fn_stack().empty()) {
    env = MakeConstexprEnv(expr, ctx_->fn_stack().back().symbolic_bindings(),
                           ctx_->type_info());

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

  XLS_ASSIGN_OR_RETURN(InterpValue constexpr_value,
                       BytecodeInterpreter::Interpret(ctx_->import_data(),
                                                      bf.get(), /*args=*/{}));
  ctx_->type_info()->NoteConstExpr(expr, constexpr_value);

  return absl::OkStatus();
}

}  // namespace xls::dslx
