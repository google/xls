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

#include "xls/dslx/evaluate.h"
#include "xls/dslx/interp_value.h"
#include "xls/dslx/interpreter.h"

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

bool ConstexprEvaluator::IsConstExpr(Expr* expr) {
  return ctx_->type_info()->GetConstExpr(expr).has_value();
}

void ConstexprEvaluator::HandleAttr(Attr* expr) {
  status_.Update(SimpleEvaluate(expr));
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

void ConstexprEvaluator::HandleConstRef(ConstRef* expr) {
  return HandleNameRef(expr);
}

void ConstexprEvaluator::HandleInvocation(Invocation* expr) {
  // An invocation is constexpr iff its args are constexpr.
  for (const auto& arg : expr->args()) {
    if (!IsConstExpr(arg)) {
      return;
    }
  }
  status_ = SimpleEvaluate(expr);
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
  ConcreteType* type_ptr;
  if (expr->type_annotation() != nullptr) {
    // If the number is annotated with a type, then extract it to pass to
    // EvaluateNumber (for consistency checking). It might be that the type is
    // parametric, in which case we'll need to fully instantiate it.
    type_ptr = ctx_->type_info()->GetItem(expr->type_annotation()).value();
    BitsType* bt = down_cast<BitsType*>(type_ptr);
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
  } else if (expr->number_kind() == NumberKind::kBool) {
    temp_type = std::make_unique<BitsType>(false, 1);
    type_ptr = temp_type.get();
  } else if (expr->number_kind() == NumberKind::kCharacter) {
    temp_type = std::make_unique<BitsType>(false, 8);
    type_ptr = temp_type.get();
  } else {
    status_ = absl::InternalError(absl::StrCat(
        "Found undecorated \"other\" type number: ", expr->ToString(),
        ". Should have been caught earlier in typechecking."));
    return;
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

  absl::StatusOr<InterpValue> constexpr_value = Interpreter::InterpretExpr(
      expr->owner(), ctx_->type_info(), ctx_->typecheck_module(),
      ctx_->import_data(), env, expr,
      fn_ctx.has_value() ? &fn_ctx.value() : nullptr, concrete_type_);
  if (constexpr_value.ok()) {
    ctx_->type_info()->NoteConstExpr(expr, constexpr_value.value());
  }

  return absl::OkStatus();
}

}  // namespace xls::dslx
