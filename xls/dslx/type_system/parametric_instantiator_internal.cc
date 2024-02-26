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

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/die_if_null.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "xls/common/logging/logging.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/dslx/bytecode/bytecode.h"
#include "xls/dslx/bytecode/bytecode_emitter.h"
#include "xls/dslx/bytecode/bytecode_interpreter.h"
#include "xls/dslx/bytecode/bytecode_interpreter_options.h"
#include "xls/dslx/constexpr_evaluator.h"
#include "xls/dslx/errors.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/frontend/pos.h"
#include "xls/dslx/interp_value.h"
#include "xls/dslx/type_system/concrete_type.h"
#include "xls/dslx/type_system/deduce_ctx.h"
#include "xls/dslx/type_system/parametric_bind.h"
#include "xls/dslx/type_system/parametric_constraint.h"
#include "xls/dslx/type_system/parametric_env.h"
#include "xls/dslx/type_system/parametric_expression.h"
#include "xls/dslx/type_system/scoped_fn_stack_entry.h"
#include "xls/dslx/type_system/type_and_parametric_env.h"
#include "xls/dslx/type_system/type_info.h"
#include "xls/dslx/warning_kind.h"

namespace xls::dslx {
namespace internal {
namespace {

// Resolves possibly-parametric type 'annotated' via 'parametric_env_map'.
absl::StatusOr<std::unique_ptr<ConcreteType>> Resolve(
    const ConcreteType& annotated,
    const absl::flat_hash_map<std::string, InterpValue>& parametric_env_map) {
  XLS_ASSIGN_OR_RETURN(
      std::unique_ptr<ConcreteType> resolved,
      annotated.MapSize(
          [&](ConcreteTypeDim dim) -> absl::StatusOr<ConcreteTypeDim> {
            if (!std::holds_alternative<ConcreteTypeDim::OwnedParametric>(
                    dim.value())) {
              return dim;
            }
            const auto& parametric_expr =
                std::get<ConcreteTypeDim::OwnedParametric>(dim.value());
            ParametricExpression::Evaluated evaluated =
                parametric_expr->Evaluate(
                    ToParametricEnv(ParametricEnv(parametric_env_map)));
            return ConcreteTypeDim(std::move(evaluated));
          }));
  XLS_VLOG(5) << "Resolved " << annotated.ToString() << " to "
              << resolved->ToString();
  return resolved;
}

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
  XLS_ASSIGN_OR_RETURN(env,
                       MakeConstexprEnv(ctx->import_data(), ctx->type_info(),
                                        ctx->warnings(), expr, parametric_env));

  XLS_ASSIGN_OR_RETURN(
      std::unique_ptr<BytecodeFunction> bf,
      BytecodeEmitter::EmitExpression(ctx->import_data(), ctx->type_info(),
                                      expr, env, std::nullopt));

  std::vector<Span> rollovers;
  BytecodeInterpreterOptions options;
  options.rollover_hook([&](const Span& s) { rollovers.push_back(s); });

  XLS_ASSIGN_OR_RETURN(InterpValue value, BytecodeInterpreter::Interpret(
                                              ctx->import_data(), bf.get(),
                                              /*args=*/{}, options));

  for (const Span& s : rollovers) {
    ctx->warnings()->Add(s, WarningKind::kConstexprEvalRollover,
                         "constexpr evaluation detected rollover in operation");
  }

  return value;
}

// Verifies that all parametrics adhere to signature-annotated types, and
// attempts to eagerly evaluate as many parametric values as possible.
//
// Take the following function signature for example:
//
//  fn f<X: u32 = {u32:5}>(x: bits[X])
//
//  fn main() {
//    f(u8:255)
//  }
//
// The parametric X has two ways of being determined in its instantiated in
// main:
//
// * the default-expression for the parametric (i.e. `5`)
// * the deduced value from the given argument type (i.e. `8`)
//
// This function is responsible for computing any parametric expressions and
// asserting that their values are consistent with other constraints (argument
// types).
absl::Status EagerlyPopulateParametricEnvMap(
    absl::Span<const std::string> parametric_order,
    const absl::flat_hash_map<std::string, Expr*>& parametric_default_exprs,
    absl::flat_hash_map<std::string, InterpValue>& parametric_env_map,
    const Span& span, std::string_view kind_name, DeduceCtx* ctx) {
  // Attempt to interpret the parametric "default expressions" in order.
  for (const auto& name : parametric_order) {
    Expr* expr = nullptr;
    if (auto it = parametric_default_exprs.find(name);
        it != parametric_default_exprs.end() && it->second != nullptr) {
      expr = it->second;
    } else {
      continue;  // e.g. <X: u32> has no default expr
    }
    XLS_VLOG(5) << "name: " << name << " expr: " << expr->ToString();

    // Note: we may have created values in early parametrics that are used in
    // the types of subsequent expressions; e.g.:
    //
    //    fn f<X: u32, Y: sN[X] = {sN[X]:0}>()
    //                            ^-------^-- expr that uses previous binding
    //
    // So before we evaluate an expression we make sure it has the most up to
    // date type info available: we deduce the type of the expression, resolve
    // it, and ensure it's set as the resolved type in the type info before
    // interpretation.
    XLS_ASSIGN_OR_RETURN(std::unique_ptr<ConcreteType> expr_type,
                         ctx->Deduce(expr));
    XLS_ASSIGN_OR_RETURN(expr_type, Resolve(*expr_type, parametric_env_map));
    XLS_RET_CHECK(!expr_type->HasParametricDims()) << expr_type->ToString();
    ctx->type_info()->SetItem(expr, *expr_type);

    // Create the environment in which we evaluate the parametric default
    // expression.
    const ParametricEnv env(parametric_env_map);

    XLS_VLOG(5) << absl::StreamFormat("Evaluating expr: `%s` in env: %s",
                                      expr->ToString(), env.ToString());

    absl::StatusOr<InterpValue> result = InterpretExpr(ctx, expr, env);

    XLS_VLOG(5) << "Interpreted expr: " << expr->ToString() << " @ "
                << expr->span() << " to status: " << result.status();

    XLS_RETURN_IF_ERROR(result.status());

    if (auto it = parametric_env_map.find(name);
        it != parametric_env_map.end()) {
      // Here we check that there is no contradiction between what we previously
      // determined and have currently determined.
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
            kind_name, name, seen.ToString(), name, expr->ToString(),
            result.value().ToString());
        return TypeInferenceErrorStatus(span, nullptr, message);
      }
    } else {
      parametric_env_map.insert({name, result.value()});
    }
  }
  return absl::OkStatus();
}

}  // namespace

ParametricInstantiator::ParametricInstantiator(
    Span span, absl::Span<const InstantiateArg> args, DeduceCtx* ctx,
    absl::Span<const ParametricConstraint> parametric_constraints,
    const absl::flat_hash_map<std::string, InterpValue>& explicit_parametrics)
    : span_(std::move(span)),
      args_(args),
      ctx_(ABSL_DIE_IF_NULL(ctx)),
      parametric_env_map_(explicit_parametrics) {
  // We add derived type information so we can resolve types based on
  // parametrics; e.g. in
  //
  //    f<BITS: u32, MINVAL: sN[BITS] = sN[BITS]:0>(...)
  //                                    ^~~~~~~~~^
  //  The underlined portion wants a concrete type definition so it can
  //  interpret the expression to an InterpValue.
  derived_type_info_ = ctx_->AddDerivedTypeInfo();

  // Explicit constraints are conceptually evaluated before other parametric
  // expressions.
  absl::flat_hash_set<std::string> ordered;
  for (const auto& [identifier, value] : explicit_parametrics) {
    constraint_order_.push_back(identifier);
    ordered.insert(identifier);
  }

  XLS_VLOG(5) << "ParametricInstantiator; span: " << span_ << " ordered: ["
              << absl::StrJoin(ordered, ", ") << "]";

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
  XLS_CHECK_OK(ctx_->PopDerivedTypeInfo(derived_type_info_));
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
                            parametric_default_exprs_, parametric_env_map_,
                            this->ctx()};
  XLS_RETURN_IF_ERROR(ParametricBind(param_type, arg_type, ctx));
  return absl::OkStatus();
}

/* static */ absl::StatusOr<std::unique_ptr<FunctionInstantiator>>
FunctionInstantiator::Make(
    Span span, Function& callee_fn, const FunctionType& function_type,
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
  return absl::WrapUnique(new FunctionInstantiator(
      std::move(span), callee_fn, function_type, args, ctx,
      parametric_constraints, explicit_parametrics));
}

absl::StatusOr<TypeAndParametricEnv> FunctionInstantiator::Instantiate() {
  ScopedFnStackEntry parametric_env_expr_scope(callee_fn_, &ctx(),
                                               WithinProc::kNo);
  XLS_VLOG(5) << absl::StreamFormat(
      "Entering parametric env scope; callee fn: `%s`",
      callee_fn_.identifier());

  // Phase 1: instantiate actuals against parametrics in left-to-right order.
  XLS_VLOG(10) << "Phase 1: instantiate actuals";
  for (int64_t i = 0; i < args().size(); ++i) {
    const ConcreteType& param_type = *param_types_[i];
    const ConcreteType& arg_type = *args()[i].type();
    XLS_RETURN_IF_ERROR(InstantiateOneArg(i, param_type, arg_type));
  }

  XLS_RETURN_IF_ERROR(EagerlyPopulateParametricEnvMap(
      constraint_order(), parametric_default_exprs(), parametric_env_map(),
      span(), GetKindName(), &ctx()));

  // Phase 2: resolve and check.
  XLS_VLOG(10) << "Phase 2: resolve-and-check";
  for (int64_t i = 0; i < args().size(); ++i) {
    const ConcreteType& param_type = *param_types_[i];
    const ConcreteType& arg_type = *args()[i].type();
    XLS_ASSIGN_OR_RETURN(std::unique_ptr<ConcreteType> instantiated_param_type,
                         Resolve(param_type, parametric_env_map()));
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
  XLS_ASSIGN_OR_RETURN(std::unique_ptr<ConcreteType> resolved,
                       Resolve(orig, parametric_env_map()));
  XLS_VLOG(5) << "Resolved return type from " << orig.ToString() << " to "
              << resolved->ToString();

  if (resolved->HasParametricDims()) {
    return TypeInferenceErrorStatus(
        span(), resolved.get(),
        "Instantiated return type did not have all parametrics resolved.");
  }

  parametric_env_expr_scope.Finish();

  return TypeAndParametricEnv{std::move(resolved),
                              ParametricEnv(parametric_env_map())};
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

absl::StatusOr<TypeAndParametricEnv> StructInstantiator::Instantiate() {
  // Phase 1: instantiate actuals against parametrics in left-to-right order.
  for (int64_t i = 0; i < member_types_.size(); ++i) {
    const ConcreteType& member_type = *member_types_[i];
    const ConcreteType& arg_type = *args()[i].type();
    XLS_RETURN_IF_ERROR(InstantiateOneArg(i, member_type, arg_type));
  }

  XLS_RETURN_IF_ERROR(EagerlyPopulateParametricEnvMap(
      constraint_order(), parametric_default_exprs(), parametric_env_map(),
      span(), GetKindName(), &ctx()));

  // Phase 2: resolve and check.
  for (int64_t i = 0; i < member_types_.size(); ++i) {
    const ConcreteType& member_type = *member_types_[i];
    const ConcreteType& arg_type = *args()[i].type();
    XLS_ASSIGN_OR_RETURN(std::unique_ptr<ConcreteType> instantiated_member_type,
                         Resolve(member_type, parametric_env_map()));
    if (*instantiated_member_type != arg_type) {
      return ctx().TypeMismatchError(
          args()[i].span(), nullptr, *instantiated_member_type, nullptr,
          arg_type, "Mismatch between member and argument types.");
    }
  }

  XLS_ASSIGN_OR_RETURN(std::unique_ptr<ConcreteType> resolved,
                       Resolve(*struct_type_, parametric_env_map()));
  return TypeAndParametricEnv{std::move(resolved),
                              ParametricEnv(parametric_env_map())};
}

}  // namespace internal
}  // namespace xls::dslx
