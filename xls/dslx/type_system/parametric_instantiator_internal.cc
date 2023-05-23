// Copyright 2023 The XLS Authors
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

#include "xls/dslx/type_system/parametric_instantiator_internal.h"

#include "absl/strings/match.h"
#include "xls/common/status/ret_check.h"
#include "xls/dslx/bytecode/bytecode_emitter.h"
#include "xls/dslx/bytecode/bytecode_interpreter.h"
#include "xls/dslx/constexpr_evaluator.h"
#include "xls/dslx/errors.h"
#include "xls/dslx/type_system/parametric_bind.h"

namespace xls::dslx {
namespace internal {
namespace {

absl::StatusOr<InterpValue> InterpretExpr(DeduceCtx* ctx, Expr* expr,
                                          const ParametricEnv& parametric_env) {
  // If we're interpreting something in another module, switch to its root type
  // info, otherwise truck on with the current DeduceCtx.
  std::unique_ptr<DeduceCtx> new_ctx_holder;
  if (expr->owner() != ctx->module()) {
    XLS_ASSIGN_OR_RETURN(TypeInfo * expr_type_info,
                         ctx->import_data()->GetRootTypeInfo(expr->owner()));
    new_ctx_holder = ctx->MakeCtx(expr_type_info, expr->owner());
    ctx = new_ctx_holder.get();
  }

  absl::flat_hash_map<std::string, InterpValue> env;
  XLS_ASSIGN_OR_RETURN(
      env, MakeConstexprEnv(ctx->import_data(), ctx->type_info(), expr,
                            parametric_env));

  XLS_ASSIGN_OR_RETURN(
      std::unique_ptr<BytecodeFunction> bf,
      BytecodeEmitter::EmitExpression(ctx->import_data(), ctx->type_info(),
                                      expr, env, std::nullopt));
  return BytecodeInterpreter::Interpret(ctx->import_data(), bf.get(),
                                        /*args=*/{});
}

}  // namespace

ParametricInstantiator::ParametricInstantiator(
    Span span, absl::Span<const InstantiateArg> args, DeduceCtx* ctx,
    absl::Span<const ParametricConstraint> parametric_constraints,
    const absl::flat_hash_map<std::string, InterpValue>& explicit_parametrics)
    : span_(std::move(span)),
      args_(args),
      ctx_(XLS_DIE_IF_NULL(ctx)),
      parametric_env_(explicit_parametrics) {
  // We add derived type information so we can resolve types based on
  // parametrics; e.g. in
  //
  //    f<BITS: u32, MINVAL: sN[BITS] = sN[BITS]:0>(...)
  //                                    ^~~~~~~~~^
  //  The underlined portion wants a concrete type definition so it can
  //  interpret the expression to an InterpValue.
  ctx_->AddDerivedTypeInfo();

  // Explicit constraints are conceptually evaluated before other parametric
  // expressions.
  absl::flat_hash_set<std::string> ordered;
  for (const auto& [identifier, value] : explicit_parametrics) {
    constraint_order_.push_back(identifier);
    ordered.insert(identifier);
  }

  for (const ParametricConstraint& constraint : parametric_constraints) {
    const std::string& identifier = constraint.identifier();
    if (!ordered.contains(identifier)) {
      constraint_order_.push_back(identifier);
      ordered.insert(identifier);
    }

    std::unique_ptr<ConcreteType> parametric_expr_type =
        constraint.type().CloneToUnique();

    if (constraint.expr() != nullptr) {
      ctx_->type_info()->SetItem(constraint.expr(), *parametric_expr_type);
    }
    parametric_binding_types_.emplace(identifier,
                                      std::move(parametric_expr_type));
    parametric_default_exprs_[identifier] = constraint.expr();
  }
}

ParametricInstantiator::~ParametricInstantiator() {
  XLS_CHECK_OK(ctx_->PopDerivedTypeInfo());
}

absl::Status ParametricInstantiator::InstantiateOneArg(
    int64_t i, const ConcreteType& param_type, const ConcreteType& arg_type) {
  if (typeid(param_type) != typeid(arg_type)) {
    std::string message = absl::StrFormat(
        "Parameter %d and argument types are different kinds (%s vs %s).", i,
        param_type.GetDebugTypeName(), arg_type.GetDebugTypeName());
    return ctx_->TypeMismatchError(span_, nullptr, param_type, nullptr,
                                   arg_type, message);
  }

  XLS_VLOG(5) << absl::StreamFormat(
      "Symbolically binding param %d formal %s against arg %s", i,
      param_type.ToString(), arg_type.ToString());
  ParametricBindContext ctx{span_, parametric_binding_types_,
                            parametric_default_exprs_, parametric_env_,
                            this->ctx()};
  XLS_RETURN_IF_ERROR(ParametricBind(param_type, arg_type, ctx));
  return absl::OkStatus();
}

absl::StatusOr<std::unique_ptr<ConcreteType>> ParametricInstantiator::Resolve(
    const ConcreteType& annotated) {
  XLS_RETURN_IF_ERROR(VerifyConstraints());

  XLS_ASSIGN_OR_RETURN(
      std::unique_ptr<ConcreteType> resolved,
      annotated.MapSize(
          [this](ConcreteTypeDim dim) -> absl::StatusOr<ConcreteTypeDim> {
            if (!std::holds_alternative<ConcreteTypeDim::OwnedParametric>(
                    dim.value())) {
              return dim;
            }
            const auto& parametric_expr =
                std::get<ConcreteTypeDim::OwnedParametric>(dim.value());
            ParametricExpression::Evaluated evaluated =
                parametric_expr->Evaluate(
                    ToParametricEnv(ParametricEnv(parametric_env_)));
            return ConcreteTypeDim(std::move(evaluated));
          }));
  XLS_VLOG(5) << "Resolved " << annotated.ToString() << " to "
              << resolved->ToString();
  return resolved;
}

absl::Status ParametricInstantiator::VerifyConstraints() {
  XLS_VLOG(5) << "Verifying " << parametric_default_exprs_.size()
              << " constraints";
  for (const auto& name : constraint_order_) {
    Expr* expr = parametric_default_exprs_[name];
    XLS_VLOG(5) << "name: " << name
                << " expr: " << (expr == nullptr ? "<none>" : expr->ToString());
    if (expr == nullptr) {  // e.g. <X: u32> has no expr
      continue;
    }

    const FnStackEntry& entry = ctx_->fn_stack().back();
    FnCtx fn_ctx{ctx_->module()->name(), entry.name(), entry.parametric_env()};
    const ParametricEnv env(parametric_env_);
    XLS_VLOG(5) << absl::StreamFormat("Evaluating expr: `%s` in env: %s",
                                      expr->ToString(), env.ToString());
    absl::StatusOr<InterpValue> result = InterpretExpr(ctx_, expr, env);
    XLS_VLOG(5) << "Interpreted expr: " << expr->ToString() << " @ "
                << expr->span() << " to status: " << result.status();
    if (!result.ok() && result.status().code() == absl::StatusCode::kNotFound &&
        (absl::StartsWith(
            result.status().message(),
            "InterpBindings could not find bindings entry for identifier"))) {
      // We haven't seen enough bindings to evaluate this constraint yet.
      continue;
    }
    if (!result.ok() && result.status().code() == absl::StatusCode::kInternal &&
        absl::StartsWith(
            result.status().message(),
            "BytecodeEmitter could not find slot or binding for name")) {
      // We haven't seen enough bindings to evaluate this constraint yet.
      continue;
    }

    if (auto it = parametric_env_.find(name); it != parametric_env_.end()) {
      InterpValue seen = it->second;
      if (result.value() != seen) {
        XLS_ASSIGN_OR_RETURN(std::unique_ptr<ConcreteType> lhs_type,
                             ConcreteType::FromInterpValue(seen));
        XLS_ASSIGN_OR_RETURN(std::unique_ptr<ConcreteType> rhs_type,
                             ConcreteType::FromInterpValue(result.value()));
        std::string message = absl::StrFormat(
            "Inconsistent parametric instantiation of %s, first saw %s = %s; "
            "then saw %s = "
            "%s = %s",
            GetKindName(), name, seen.ToString(), name, expr->ToString(),
            result.value().ToString());
        return TypeInferenceErrorStatus(span_, nullptr, message);
      }
    } else {
      parametric_env_.insert({name, result.value()});
    }
  }
  return absl::OkStatus();
}

/* static */ absl::StatusOr<std::unique_ptr<FunctionInstantiator>>
FunctionInstantiator::Make(
    Span span, const FunctionType& function_type,
    absl::Span<const InstantiateArg> args, DeduceCtx* ctx,
    absl::Span<const ParametricConstraint> parametric_constraints,
    const absl::flat_hash_map<std::string, InterpValue>& explicit_parametrics) {
  XLS_VLOG(5) << "Making FunctionInstantiator for " << function_type.ToString()
              << " with " << parametric_constraints.size()
              << " parametric constraints and " << explicit_parametrics.size()
              << " explicit constraints";
  if (args.size() != function_type.params().size()) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "ArgCountMismatchError: %s Expected %d parameter(s) but got %d "
        "argument(s)",
        span.ToString(), function_type.params().size(), args.size()));
  }
  return absl::WrapUnique(
      new FunctionInstantiator(std::move(span), function_type, args, ctx,
                               parametric_constraints, explicit_parametrics));
}

absl::StatusOr<TypeAndBindings> FunctionInstantiator::Instantiate() {
  // Phase 1: instantiate actuals against parametrics in left-to-right order.
  XLS_VLOG(10) << "Phase 1: isntantiate actuals";
  for (int64_t i = 0; i < args().size(); ++i) {
    const ConcreteType& param_type = *param_types_[i];
    const ConcreteType& arg_type = *args()[i].type();
    XLS_RETURN_IF_ERROR(InstantiateOneArg(i, param_type, arg_type));
  }

  // Phase 2: resolve and check.
  XLS_VLOG(10) << "Phase 2: resolve-and-check";
  for (int64_t i = 0; i < args().size(); ++i) {
    const ConcreteType& param_type = *param_types_[i];
    const ConcreteType& arg_type = *args()[i].type();
    XLS_ASSIGN_OR_RETURN(std::unique_ptr<ConcreteType> instantiated_param_type,
                         Resolve(param_type));
    if (*instantiated_param_type != arg_type) {
      // Although it's not the *original* parameter (which could be a little
      // confusing to the user) we want to show what the mismatch was directly,
      // so we use the instantiated_param_type here.
      return ctx().TypeMismatchError(
          args()[i].span(), nullptr, *instantiated_param_type, nullptr,
          arg_type,
          "Mismatch between parameter and argument types "
          "(after instantiation).");
    }
  }

  // Resolve the return type according to the bindings we collected.
  const ConcreteType& orig = function_type_->return_type();
  XLS_ASSIGN_OR_RETURN(std::unique_ptr<ConcreteType> resolved, Resolve(orig));
  XLS_VLOG(5) << "Resolved return type from " << orig.ToString() << " to "
              << resolved->ToString();
  return TypeAndBindings{std::move(resolved), ParametricEnv(parametric_env())};
}

/* static */ absl::StatusOr<std::unique_ptr<StructInstantiator>>
StructInstantiator::Make(
    Span span, const StructType& struct_type,
    absl::Span<const InstantiateArg> args,
    absl::Span<std::unique_ptr<ConcreteType> const> member_types,
    DeduceCtx* ctx,
    absl::Span<const ParametricConstraint> parametric_bindings) {
  XLS_RET_CHECK_EQ(args.size(), member_types.size());
  return absl::WrapUnique(new StructInstantiator(std::move(span), struct_type,
                                                 args, member_types, ctx,
                                                 parametric_bindings));
}

absl::StatusOr<TypeAndBindings> StructInstantiator::Instantiate() {
  // Phase 1: instantiate actuals against parametrics in left-to-right order.
  for (int64_t i = 0; i < member_types_.size(); ++i) {
    const ConcreteType& member_type = *member_types_[i];
    const ConcreteType& arg_type = *args()[i].type();
    XLS_RETURN_IF_ERROR(InstantiateOneArg(i, member_type, arg_type));
  }
  // Phase 2: resolve and check.
  for (int64_t i = 0; i < member_types_.size(); ++i) {
    const ConcreteType& member_type = *member_types_[i];
    const ConcreteType& arg_type = *args()[i].type();
    XLS_ASSIGN_OR_RETURN(std::unique_ptr<ConcreteType> instantiated_member_type,
                         Resolve(member_type));
    if (*instantiated_member_type != arg_type) {
      return ctx().TypeMismatchError(
          args()[i].span(), nullptr, *instantiated_member_type, nullptr,
          arg_type, "Mismatch between member and argument types.");
    }
  }

  XLS_ASSIGN_OR_RETURN(std::unique_ptr<ConcreteType> resolved,
                       Resolve(*struct_type_));
  return TypeAndBindings{std::move(resolved), ParametricEnv(parametric_env())};
}

}  // namespace internal
}  // namespace xls::dslx
