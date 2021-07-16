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

#include "xls/dslx/parametric_instantiator.h"

#include "absl/strings/match.h"
#include "absl/types/variant.h"
#include "xls/dslx/interpreter.h"

namespace xls::dslx {
namespace internal {

ParametricInstantiator::ParametricInstantiator(
    Span span, absl::Span<const InstantiateArg> args, DeduceCtx* ctx,
    absl::optional<absl::Span<const ParametricConstraint>>
        parametric_constraints,
    const absl::flat_hash_map<std::string, InterpValue>* explicit_constraints)
    : span_(std::move(span)), args_(args), ctx_(ctx) {
  if (explicit_constraints != nullptr) {
    symbolic_bindings_ = *explicit_constraints;
  }

  if (parametric_constraints.has_value()) {
    for (const ParametricConstraint& constraint :
         parametric_constraints.value()) {
      const std::string& identifier = constraint.identifier();
      constraint_order_.push_back(identifier);
      ConcreteTypeDim bit_count = constraint.type().GetTotalBitCount().value();
      bit_widths_.emplace(identifier,
                          absl::get<InterpValue>(bit_count.value()));
      constraints_[identifier] = constraint.expr();
    }
  }
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
      annotated.MapSize([this](ConcreteTypeDim dim)
                            -> absl::StatusOr<ConcreteTypeDim> {
        if (!absl::holds_alternative<ConcreteTypeDim::OwnedParametric>(
                dim.value())) {
          return dim;
        }
        const auto& parametric_expr =
            absl::get<ConcreteTypeDim::OwnedParametric>(dim.value());
        ParametricExpression::Evaluated evaluated = parametric_expr->Evaluate(
            ToParametricEnv(SymbolicBindings(symbolic_bindings_)));
        return ConcreteTypeDim(std::move(evaluated));
      }));
  XLS_VLOG(5) << "Resolved " << annotated.ToString() << " to "
              << resolved->ToString();
  return resolved;
}

absl::Status ParametricInstantiator::VerifyConstraints() {
  XLS_VLOG(5) << "Verifying " << constraints_.size() << " constraints";
  for (const auto& name : constraint_order_) {
    Expr* expr = constraints_[name];
    XLS_VLOG(5) << "name: " << name
                << " expr: " << (expr == nullptr ? "<none>" : expr->ToString());
    if (expr == nullptr) {  // e.g. <X: u32> has no expr
      continue;
    }
    const FnStackEntry& entry = ctx_->fn_stack().back();
    FnCtx fn_ctx{ctx_->module()->name(), entry.name(),
                 entry.symbolic_bindings()};
    Module* expr_module = expr->owner();
    XLS_ASSIGN_OR_RETURN(TypeInfo * expr_type_info,
                         ctx_->import_data()->GetRootTypeInfo(expr_module));
    absl::flat_hash_map<std::string, InterpValue> constexpr_env =
        MakeConstexprEnv(expr, SymbolicBindings(symbolic_bindings_),
                         expr_type_info);
    absl::StatusOr<InterpValue> result = Interpreter::InterpretExpr(
        expr_module, expr_type_info, ctx_->typecheck_module(),
        ctx_->additional_search_paths(), ctx_->import_data(), constexpr_env,
        expr, &fn_ctx);
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

    if (auto it = symbolic_bindings_.find(name);
        it != symbolic_bindings_.end()) {
      InterpValue seen = it->second;
      int64_t result_value = result.value().GetBitValueInt64().value();
      int64_t seen_value = seen.GetBitValueInt64().value();
      if (result.value() != seen) {
        XLS_ASSIGN_OR_RETURN(auto lhs_type,
                             ConcreteType::FromInterpValue(result.value()));
        XLS_ASSIGN_OR_RETURN(auto rhs_type,
                             ConcreteType::FromInterpValue(result.value()));
        std::string message = absl::StrFormat(
            "Parametric constraint violated, first saw %s = %d; then saw %s = "
            "%s = %d",
            name, seen_value, name, expr->ToString(), result_value);
        return XlsTypeErrorStatus(span_, *rhs_type, *lhs_type,
                                  std::move(message));
      }
    } else {
      symbolic_bindings_.insert({name, result.value()});
    }
  }
  return absl::OkStatus();
}

static const ParametricSymbol* TryGetParametricSymbol(
    const ConcreteTypeDim& dim) {
  if (!absl::holds_alternative<ConcreteTypeDim::OwnedParametric>(dim.value())) {
    return nullptr;
  }
  const ParametricExpression* parametric =
      absl::get<ConcreteTypeDim::OwnedParametric>(dim.value()).get();
  return dynamic_cast<const ParametricSymbol*>(parametric);
}

template <typename T>
absl::Status ParametricInstantiator::SymbolicBindDims(const T& param_type,
                                                      const T& arg_type) {
  // Create bindings for symbolic parameter dimensions based on argument values
  // passed.
  ConcreteTypeDim param_dim = param_type.size();

  const ParametricSymbol* symbol = TryGetParametricSymbol(param_dim);
  if (symbol == nullptr) {
    return absl::OkStatus();  // Nothing to bind in the formal argument type.
  }

  XLS_ASSIGN_OR_RETURN(int64_t arg_dim,
                       ConcreteTypeDim::GetAs64Bits(arg_type.size().value()));

  const std::string& pdim_name = symbol->identifier();
  if (!symbolic_bindings_.contains(pdim_name)) {
    XLS_RET_CHECK(bit_widths_.contains(pdim_name))
        << "Cannot bind " << pdim_name << " : it has no associated bit width.";
    XLS_VLOG(5) << "Binding " << pdim_name << " to " << arg_dim;
    XLS_ASSIGN_OR_RETURN(int64_t width,
                         bit_widths_.at(pdim_name).GetBitValueInt64());
    symbolic_bindings_.emplace(
        pdim_name,
        InterpValue::MakeUBits(/*bit_count=*/width, /*value=*/arg_dim));
    return absl::OkStatus();
  }

  const InterpValue& seen = symbolic_bindings_.at(pdim_name);
  int64_t seen_value = seen.GetBitValueInt64().value();
  if (seen_value == arg_dim) {
    return absl::OkStatus();  // No contradiction.
  }

  // We see a conflict between something we previously observed and something
  // we are now observing.
  if (Expr* expr = constraints_[pdim_name]) {
    // Error is violated constraint.
    std::string message = absl::StrFormat(
        "Parametric constraint violated, saw %s = %d; then %s = %s = %d",
        pdim_name, seen_value, pdim_name, expr->ToString(), arg_dim);
    auto saw_type =
        absl::make_unique<BitsType>(/*signed=*/false, /*size=*/seen_value);
    return XlsTypeErrorStatus(span_, *saw_type, arg_type, message);
  } else {
    // Error is conflicting argument types.
    std::string message = absl::StrFormat(
        "Parametric value %s was bound to different values at different "
        "places in invocation; saw: %d; then: %d",
        pdim_name, seen_value, arg_dim);
    return XlsTypeErrorStatus(span_, param_type, arg_type, message);
  }
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
    XLS_RET_CHECK_EQ(param_enum->nominal_type(), arg_enum->nominal_type());
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
    StructDef& param_nominal = param_struct->nominal_type();
    StructDef& arg_nominal = arg_struct->nominal_type();
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

  return absl::InternalError(
      absl::StrFormat("Unhandled parameter type for symbolic binding: %s @ %s",
                      param_type.ToString(), span_.ToString()));
}

/* static */ absl::StatusOr<FunctionInstantiator> FunctionInstantiator::Make(
    Span span, const FunctionType& function_type,
    absl::Span<const InstantiateArg> args, DeduceCtx* ctx,
    absl::optional<absl::Span<const ParametricConstraint>>
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
  return FunctionInstantiator(std::move(span), function_type, args, ctx,
                              parametric_constraints, explicit_constraints);
}

absl::StatusOr<TypeAndBindings> FunctionInstantiator::Instantiate() {
  // Walk through all the params/args to collect symbolic bindings.
  for (int64_t i = 0; i < args().size(); ++i) {
    const ConcreteType& param_type = *param_types_[i];
    const ConcreteType& arg_type = args()[i].type;
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
  return TypeAndBindings{std::move(resolved),
                         SymbolicBindings(symbolic_bindings())};
}

/* static */ absl::StatusOr<StructInstantiator> StructInstantiator::Make(
    Span span, const StructType& struct_type,
    absl::Span<const InstantiateArg> args,
    absl::Span<std::unique_ptr<ConcreteType> const> member_types,
    DeduceCtx* ctx,
    absl::optional<absl::Span<const ParametricConstraint>>
        parametric_bindings) {
  XLS_RET_CHECK_EQ(args.size(), member_types.size());
  return StructInstantiator(std::move(span), struct_type, args, member_types,
                            ctx, parametric_bindings);
}

absl::StatusOr<TypeAndBindings> StructInstantiator::Instantiate() {
  for (int64_t i = 0; i < member_types_.size(); ++i) {
    const ConcreteType& member_type = *member_types_[i];
    const ConcreteType& arg_type = args()[i].type;
    XLS_ASSIGN_OR_RETURN(std::unique_ptr<ConcreteType> instantiated_member_type,
                         InstantiateOneArg(i, member_type, arg_type));
    if (*instantiated_member_type != arg_type) {
      return XlsTypeErrorStatus(span(), *instantiated_member_type, arg_type,
                                "Mismatch between member and argument types.");
    }
  }

  XLS_ASSIGN_OR_RETURN(std::unique_ptr<ConcreteType> resolved,
                       Resolve(*struct_type_));
  return TypeAndBindings{std::move(resolved),
                         SymbolicBindings(symbolic_bindings())};
}

}  // namespace internal

ParametricConstraint::ParametricConstraint(const ParametricBinding& binding,
                                           std::unique_ptr<ConcreteType> type)
    : ParametricConstraint(binding, std::move(type), binding.expr()) {}

ParametricConstraint::ParametricConstraint(const ParametricBinding& binding,
                                           std::unique_ptr<ConcreteType> type,
                                           Expr* expr)
    : binding_(&binding), type_(std::move(type)), expr_(expr) {}

std::string ParametricConstraint::ToString() const {
  if (expr_ == nullptr) {
    return absl::StrFormat("%s: %s", identifier(), type().ToString());
  }
  return absl::StrFormat("%s: %s = %s", identifier(), type().ToString(),
                         expr()->ToString());
}

// Helper ToString()s for debug logging.
static std::string ToTypesString(absl::Span<const InstantiateArg> ts) {
  if (ts.empty()) {
    return "none";
  }
  return absl::StrJoin(ts, ", ", [](std::string* out, const auto& t) {
    absl::StrAppend(out, t.type.ToString());
  });
}
static std::string ToString(
    absl::Span<std::unique_ptr<ConcreteType> const> ts) {
  if (ts.empty()) {
    return "none";
  }
  return absl::StrJoin(ts, ", ", [](std::string* out, const auto& t) {
    absl::StrAppend(out, t->ToString());
  });
}
static std::string ToString(
    const absl::flat_hash_map<std::string, InterpValue>* map) {
  if (map == nullptr) {
    return "none";
  }
  if (map->empty()) {
    return "{}";
  }
  return absl::StrJoin(
      *map, ", ",
      [](std::string* out, const std::pair<std::string, InterpValue>& p) {
        out->append(absl::StrCat(p.first, ":", p.second.ToString()));
      });
}

static std::string ToString(
    const absl::optional<absl::Span<const ParametricConstraint>>&
        parametric_constraints) {
  if (parametric_constraints.has_value()) {
    return "none";
  }
  if (parametric_constraints.value().empty()) {
    return "[]";
  }
  return absl::StrJoin(parametric_constraints.value(), ", ",
                       [](std::string* out, const ParametricConstraint& c) {
                         absl::StrAppend(out, c.ToString());
                       });
}

absl::StatusOr<TypeAndBindings> InstantiateFunction(
    Span span, const FunctionType& function_type,
    absl::Span<const InstantiateArg> args, DeduceCtx* ctx,
    absl::optional<absl::Span<const ParametricConstraint>>
        parametric_constraints,
    const absl::flat_hash_map<std::string, InterpValue>* explicit_constraints) {
  XLS_VLOG(5) << "Function instantiation @ " << span
              << " type: " << function_type.ToString();
  XLS_VLOG(5) << " parametric constraints: "
              << ToString(parametric_constraints);
  XLS_VLOG(5) << " arg types:              " << ToTypesString(args);
  XLS_VLOG(5) << " explicit constraints:   " << ToString(explicit_constraints);
  XLS_ASSIGN_OR_RETURN(auto instantiator,
                       internal::FunctionInstantiator::Make(
                           std::move(span), function_type, args, ctx,
                           parametric_constraints, explicit_constraints));
  return instantiator.Instantiate();
}

absl::StatusOr<TypeAndBindings> InstantiateStruct(
    Span span, const StructType& struct_type,
    absl::Span<const InstantiateArg> args,
    absl::Span<std::unique_ptr<ConcreteType> const> member_types,
    DeduceCtx* ctx,
    absl::optional<absl::Span<const ParametricConstraint>>
        parametric_bindings) {
  XLS_VLOG(5) << "Struct instantiation @ " << span
              << " type: " << struct_type.ToString();
  XLS_VLOG(5) << " arg types:           " << ToTypesString(args);
  XLS_VLOG(5) << " member types:        " << ToString(member_types);
  XLS_ASSIGN_OR_RETURN(auto instantiator,
                       internal::StructInstantiator::Make(
                           std::move(span), struct_type, args, member_types,
                           ctx, parametric_bindings));
  return instantiator.Instantiate();
}

}  // namespace xls::dslx
