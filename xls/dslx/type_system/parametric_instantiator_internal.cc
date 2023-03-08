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
                                      expr, env, absl::nullopt));
  return BytecodeInterpreter::Interpret(ctx->import_data(), bf.get(),
                                        /*args=*/{});
}

}  // namespace

ParametricInstantiator::ParametricInstantiator(
    Span span, absl::Span<const InstantiateArg> args, DeduceCtx* ctx,
    std::optional<absl::Span<const ParametricConstraint>>
        parametric_constraints,
    const absl::flat_hash_map<std::string, InterpValue>* explicit_constraints)
    : span_(std::move(span)), args_(args), ctx_(XLS_DIE_IF_NULL(ctx)) {
  // We add derived type information so we can resolve types based on
  // parametrics; e.g. in
  //
  //    f<BITS: u32, MINVAL: sN[BITS] = sN[BITS]:0>(...)
  //                                    ^~~~~~~~~^
  //  The underlined portion wants a concrete type definition so it can
  //  interpret the expression to an InterpValue.
  ctx_->AddDerivedTypeInfo();

  if (explicit_constraints != nullptr) {
    parametric_env_ = *explicit_constraints;

    // Explicit constraints are conceptually evaluated before other parametric
    // expressions.
    for (const auto& [identifier, value] : *explicit_constraints) {
      constraint_order_.push_back(identifier);
    }
  }

  if (parametric_constraints.has_value()) {
    for (const ParametricConstraint& constraint :
         parametric_constraints.value()) {
      const std::string& identifier = constraint.identifier();
      constraint_order_.push_back(identifier);
      std::unique_ptr<ConcreteType> resolved_type =
          Resolve(constraint.type()).value();
      if (constraint.expr() != nullptr) {
        ctx_->type_info()->SetItem(constraint.expr(), *resolved_type);
      }
      parametric_binding_types_.emplace(identifier, std::move(resolved_type));
      parametric_default_exprs_[identifier] = constraint.expr();
    }
  }
}

ParametricInstantiator::~ParametricInstantiator() {
  XLS_CHECK_OK(ctx_->PopDerivedTypeInfo());
}

absl::StatusOr<std::unique_ptr<ConcreteType>>
ParametricInstantiator::InstantiateOneArg(int64_t i,
                                          const ConcreteType& param_type,
                                          const ConcreteType& arg_type) {
  if (typeid(param_type) != typeid(arg_type)) {
    std::string message = absl::StrFormat(
        "Parameter %d and argument types are different kinds (%s vs %s).", i,
        param_type.GetDebugTypeName(), arg_type.GetDebugTypeName());
    return XlsTypeErrorStatus(span_, param_type, arg_type, message);
  }

  XLS_VLOG(5) << absl::StreamFormat(
      "Symbolically binding param %d formal %s against arg %s", i,
      param_type.ToString(), arg_type.ToString());
  XLS_RETURN_IF_ERROR(SymbolicBind(param_type, arg_type));
  XLS_ASSIGN_OR_RETURN(std::unique_ptr<ConcreteType> resolved,
                       Resolve(param_type));
  XLS_VLOG(5) << "Resolved parameter type from " << param_type.ToString()
              << " to " << resolved->ToString();
  return resolved;
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
    absl::StatusOr<InterpValue> result =
        InterpretExpr(ctx_, expr, ParametricEnv(parametric_env_));
    XLS_VLOG(5) << "Interpreted expr: " << expr->ToString() << " @ "
                << expr->span() << " to status: " << result.status();
    if (!result.ok() && result.status().code() == absl::StatusCode::kNotFound &&
        (absl::StartsWith(result.status().message(),
                          "Could not find bindings entry for identifier") ||
         absl::StartsWith(result.status().message(),
                          "Could not find callee bindings in type info"))) {
      // We haven't seen enough bindings to evaluate this constraint yet.
      continue;
    }
    if (!result.ok() && result.status().code() == absl::StatusCode::kInternal &&
        absl::StartsWith(result.status().message(),
                         "Could not find slot or binding for name")) {
      // We haven't seen enough bindings to evaluate this constraint yet.
      continue;
    }

    if (auto it = parametric_env_.find(name); it != parametric_env_.end()) {
      InterpValue seen = it->second;
      if (result.value() != seen) {
        XLS_ASSIGN_OR_RETURN(auto lhs_type,
                             ConcreteType::FromInterpValue(seen));
        XLS_ASSIGN_OR_RETURN(auto rhs_type,
                             ConcreteType::FromInterpValue(result.value()));
        std::string message = absl::StrFormat(
            "Parametric constraint violated, first saw %s = %s; then saw %s = "
            "%s = %s",
            name, seen.ToString(), name, expr->ToString(),
            result.value().ToString());
        return XlsTypeErrorStatus(span_, *rhs_type, *lhs_type,
                                  std::move(message));
      }
    } else {
      parametric_env_.insert({name, result.value()});
    }
  }
  return absl::OkStatus();
}

template <typename T>
absl::Status ParametricInstantiator::SymbolicBindDims(const T& param_type,
                                                      const T& arg_type) {
  // Create bindings for symbolic parameter dimensions based on argument values
  // passed.
  const ConcreteTypeDim& param_dim = param_type.size();
  const ConcreteTypeDim& arg_dim = arg_type.size();
  return ParametricBindConcreteTypeDim(
      param_type, param_dim, arg_type, arg_dim, span_,
      parametric_binding_types_, parametric_default_exprs_, parametric_env_);
}

absl::Status ParametricInstantiator::SymbolicBindTuple(
    const TupleType& param_type, const TupleType& arg_type) {
  XLS_RET_CHECK_EQ(param_type.size(), arg_type.size());
  for (int64_t i = 0; i < param_type.size(); ++i) {
    const ConcreteType& param_member = param_type.GetMemberType(i);
    const ConcreteType& arg_member = arg_type.GetMemberType(i);
    XLS_RETURN_IF_ERROR(SymbolicBind(param_member, arg_member));
  }
  return absl::OkStatus();
}

absl::Status ParametricInstantiator::SymbolicBindStruct(
    const StructType& param_type, const StructType& arg_type) {
  XLS_RET_CHECK_EQ(param_type.size(), arg_type.size());
  for (int64_t i = 0; i < param_type.size(); ++i) {
    const ConcreteType& param_member = param_type.GetMemberType(i);
    const ConcreteType& arg_member = arg_type.GetMemberType(i);
    XLS_RETURN_IF_ERROR(SymbolicBind(param_member, arg_member));
  }
  return absl::OkStatus();
}

absl::Status ParametricInstantiator::SymbolicBindBits(
    const ConcreteType& param_type, const ConcreteType& arg_type) {
  if (dynamic_cast<const EnumType*>(&param_type) != nullptr) {
    return absl::OkStatus();  // Enums have no size, so nothing to bind.
  }

  auto* param_bits = dynamic_cast<const BitsType*>(&param_type);
  XLS_RET_CHECK(param_bits != nullptr);
  auto* arg_bits = dynamic_cast<const BitsType*>(&arg_type);
  XLS_RET_CHECK(arg_bits != nullptr);
  return SymbolicBindDims(*param_bits, *arg_bits);
}

absl::Status ParametricInstantiator::SymbolicBindArray(
    const ArrayType& param_type, const ArrayType& arg_type) {
  XLS_RETURN_IF_ERROR(
      SymbolicBind(param_type.element_type(), arg_type.element_type()));
  return SymbolicBindDims(param_type, arg_type);
}

absl::Status ParametricInstantiator::SymbolicBindFunction(
    const FunctionType& param_type, const FunctionType& arg_type) {
  return absl::UnimplementedError("SymbolicBindFunction()");
}

absl::Status ParametricInstantiator::SymbolicBind(
    const ConcreteType& param_type, const ConcreteType& arg_type) {
  if (auto* param_bits = dynamic_cast<const BitsType*>(&param_type)) {
    auto* arg_bits = dynamic_cast<const BitsType*>(&arg_type);
    XLS_RET_CHECK(arg_bits != nullptr);
    return SymbolicBindBits(*param_bits, *arg_bits);
  }
  if (auto* param_enum = dynamic_cast<const EnumType*>(&param_type)) {
    auto* arg_enum = dynamic_cast<const EnumType*>(&arg_type);
    XLS_RET_CHECK(arg_enum != nullptr);
    XLS_RET_CHECK_EQ(&param_enum->nominal_type(), &arg_enum->nominal_type());
    // If the enums are the same, we do the same thing as we do with bits
    // (ignore the primitive and symbolic bind the dims).
    return SymbolicBindBits(*param_enum, *arg_enum);
  }
  if (auto* param_tuple = dynamic_cast<const TupleType*>(&param_type)) {
    auto* arg_tuple = dynamic_cast<const TupleType*>(&arg_type);
    return SymbolicBindTuple(*param_tuple, *arg_tuple);
  }
  if (auto* param_struct = dynamic_cast<const StructType*>(&param_type)) {
    auto* arg_struct = dynamic_cast<const StructType*>(&arg_type);
    const StructDef& param_nominal = param_struct->nominal_type();
    const StructDef& arg_nominal = arg_struct->nominal_type();
    if (&param_nominal != &arg_nominal) {
      std::string message =
          absl::StrFormat("parameter type name: '%s'; argument type name: '%s'",
                          param_nominal.identifier(), arg_nominal.identifier());
      return XlsTypeErrorStatus(span_, param_type, arg_type, message);
    }
    return SymbolicBindStruct(*param_struct, *arg_struct);
  }
  if (auto* param_array = dynamic_cast<const ArrayType*>(&param_type)) {
    auto* arg_array = dynamic_cast<const ArrayType*>(&arg_type);
    XLS_RET_CHECK(arg_array != nullptr);
    return SymbolicBindArray(*param_array, *arg_array);
  }
  if (auto* param_fn = dynamic_cast<const FunctionType*>(&param_type)) {
    auto* arg_fn = dynamic_cast<const FunctionType*>(&arg_type);
    XLS_RET_CHECK(arg_fn != nullptr);
    return SymbolicBindFunction(*param_fn, *arg_fn);
  }
  if (dynamic_cast<const TokenType*>(&param_type) != nullptr) {
    // Tokens aren't parameterizable.
    return absl::OkStatus();
  }
  if (dynamic_cast<const ChannelType*>(&param_type) != nullptr) {
    // Neither are channels.
    return absl::OkStatus();
  }

  return absl::InternalError(
      absl::StrFormat("Unhandled parameter type for symbolic binding: %s @ %s",
                      param_type.ToString(), span_.ToString()));
}

/* static */ absl::StatusOr<std::unique_ptr<FunctionInstantiator>>
FunctionInstantiator::Make(
    Span span, const FunctionType& function_type,
    absl::Span<const InstantiateArg> args, DeduceCtx* ctx,
    std::optional<absl::Span<const ParametricConstraint>>
        parametric_constraints,
    const absl::flat_hash_map<std::string, InterpValue>* explicit_constraints) {
  XLS_VLOG(5)
      << "Making FunctionInstantiator for " << function_type.ToString()
      << " with "
      << (parametric_constraints.has_value() ? parametric_constraints->size()
                                             : 0)
      << " parametric constraints and "
      << (explicit_constraints == nullptr ? 0 : explicit_constraints->size())
      << " explicit constraints";
  if (args.size() != function_type.params().size()) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "ArgCountMismatchError: %s Expected %d parameter(s) but got %d "
        "argument(s)",
        span.ToString(), function_type.params().size(), args.size()));
  }
  return absl::WrapUnique(
      new FunctionInstantiator(std::move(span), function_type, args, ctx,
                               parametric_constraints, explicit_constraints));
}

absl::StatusOr<TypeAndBindings> FunctionInstantiator::Instantiate() {
  // Walk through all the params/args to collect symbolic bindings.
  for (int64_t i = 0; i < args().size(); ++i) {
    const ConcreteType& param_type = *param_types_[i];
    const ConcreteType& arg_type = *args()[i].type;
    XLS_ASSIGN_OR_RETURN(std::unique_ptr<ConcreteType> instantiated_param_type,
                         InstantiateOneArg(i, param_type, arg_type));
    if (*instantiated_param_type != arg_type) {
      // Although it's not the *original* parameter (which could be a little
      // confusing to the user) we want to show what the mismatch was directly,
      // so we use the instantiated_param_type here.
      return XlsTypeErrorStatus(args()[i].span, *instantiated_param_type,
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
    std::optional<absl::Span<const ParametricConstraint>> parametric_bindings) {
  XLS_RET_CHECK_EQ(args.size(), member_types.size());
  return absl::WrapUnique(new StructInstantiator(std::move(span), struct_type,
                                                 args, member_types, ctx,
                                                 parametric_bindings));
}

absl::StatusOr<TypeAndBindings> StructInstantiator::Instantiate() {
  for (int64_t i = 0; i < member_types_.size(); ++i) {
    const ConcreteType& member_type = *member_types_[i];
    const ConcreteType& arg_type = *args()[i].type;
    XLS_ASSIGN_OR_RETURN(std::unique_ptr<ConcreteType> instantiated_member_type,
                         InstantiateOneArg(i, member_type, arg_type));
    if (*instantiated_member_type != arg_type) {
      return XlsTypeErrorStatus(args()[i].span, *instantiated_member_type,
                                arg_type,
                                "Mismatch between member and argument types.");
    }
  }

  XLS_ASSIGN_OR_RETURN(std::unique_ptr<ConcreteType> resolved,
                       Resolve(*struct_type_));
  return TypeAndBindings{std::move(resolved), ParametricEnv(parametric_env())};
}

}  // namespace internal
}  // namespace xls::dslx
