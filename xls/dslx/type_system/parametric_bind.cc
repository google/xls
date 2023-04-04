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

#include "xls/dslx/type_system/parametric_bind.h"

#include "xls/common/status/ret_check.h"

namespace xls::dslx {
namespace {

template <typename T>
absl::Status ParametricBindDims(const T& param_type, const T& arg_type,
                                ParametricBindContext& ctx) {
  // Create bindings for symbolic parameter dimensions based on argument values
  // passed.
  const ConcreteTypeDim& param_dim = param_type.size();
  const ConcreteTypeDim& arg_dim = arg_type.size();
  return ParametricBindConcreteTypeDim(param_type, param_dim, arg_type, arg_dim,
                                       ctx);
}

absl::Status ParametricBindBits(const BitsType& param_bits,
                                const BitsType& arg_bits,
                                ParametricBindContext& ctx) {
  return ParametricBindConcreteTypeDim(param_bits, param_bits.size(), arg_bits,
                                       arg_bits.size(), ctx);
}

absl::Status ParametricBindTuple(const TupleType& param_type,
                                 const TupleType& arg_type,
                                 ParametricBindContext& ctx) {
  XLS_RET_CHECK_EQ(param_type.size(), arg_type.size());
  for (int64_t i = 0; i < param_type.size(); ++i) {
    const ConcreteType& param_member = param_type.GetMemberType(i);
    const ConcreteType& arg_member = arg_type.GetMemberType(i);
    XLS_RETURN_IF_ERROR(ParametricBind(param_member, arg_member, ctx));
  }
  return absl::OkStatus();
}

absl::Status ParametricBindStruct(const StructType& param_type,
                                  const StructType& arg_type,
                                  ParametricBindContext& ctx) {
  XLS_RET_CHECK_EQ(param_type.size(), arg_type.size());
  for (int64_t i = 0; i < param_type.size(); ++i) {
    const ConcreteType& param_member = param_type.GetMemberType(i);
    const ConcreteType& arg_member = arg_type.GetMemberType(i);
    XLS_RETURN_IF_ERROR(ParametricBind(param_member, arg_member, ctx));
  }
  return absl::OkStatus();
}

absl::Status ParametricBindArray(const ArrayType& param_type,
                                 const ArrayType& arg_type,
                                 ParametricBindContext& ctx) {
  XLS_RETURN_IF_ERROR(
      ParametricBind(param_type.element_type(), arg_type.element_type(), ctx));
  return ParametricBindDims(param_type, arg_type, ctx);
}

}  // namespace

absl::Status ParametricBindConcreteTypeDim(const ConcreteType& param_type,
                                           const ConcreteTypeDim& param_dim,
                                           const ConcreteType& arg_type,
                                           const ConcreteTypeDim& arg_dim,
                                           ParametricBindContext& ctx) {
  XLS_RET_CHECK(!arg_dim.IsParametric());

  // See if there's a parametric symbol in the formal argument we need to bind
  // vs the actual argument.
  const ParametricSymbol* symbol = TryGetParametricSymbol(param_dim);
  if (symbol == nullptr) {
    return absl::OkStatus();  // Nothing to bind in the formal argument type.
  }

  XLS_ASSIGN_OR_RETURN(int64_t arg_dim_i64, arg_dim.GetAsInt64());

  // See if this is the first time we're binding this parametric symbol.
  const std::string& pdim_name = symbol->identifier();
  if (!ctx.parametric_env.contains(pdim_name)) {
    XLS_RET_CHECK(ctx.parametric_binding_types.contains(pdim_name))
        << "Cannot bind " << pdim_name << " : it has no associated type.";
    XLS_VLOG(5) << "Binding " << pdim_name << " to " << arg_dim_i64;
    const ConcreteType& type = *ctx.parametric_binding_types.at(pdim_name);
    XLS_ASSIGN_OR_RETURN(ConcreteTypeDim bit_count, type.GetTotalBitCount());
    XLS_ASSIGN_OR_RETURN(int64_t width, bit_count.GetAsInt64());
    ctx.parametric_env.emplace(
        pdim_name,
        InterpValue::MakeUBits(/*bit_count=*/width, /*value=*/arg_dim_i64));
    return absl::OkStatus();
  }

  const InterpValue& seen = ctx.parametric_env.at(pdim_name);
  XLS_ASSIGN_OR_RETURN(int64_t seen_value, seen.GetBitValueInt64());
  if (seen_value == arg_dim_i64) {
    return absl::OkStatus();  // No contradiction.
  }

  // We see a conflict between something we previously observed and something
  // we are now observing -- make an appropriate error.
  if (auto it = ctx.parametric_default_exprs.find(pdim_name);
      it != ctx.parametric_default_exprs.end() && it->second != nullptr) {
    const Expr* expr = it->second;
    // Error is violated constraint.
    std::string message = absl::StrFormat(
        "Parametric constraint violated, saw %s = %d; then %s = %s = %d",
        pdim_name, seen_value, pdim_name, expr->ToString(), arg_dim_i64);
    auto saw_type =
        std::make_unique<BitsType>(/*signed=*/false, /*size=*/seen_value);
    return ctx.deduce_ctx.TypeMismatchError(ctx.span, expr, *saw_type, nullptr,
                                            arg_type, message);
  }

  // Error is conflicting argument types.
  std::string message = absl::StrFormat(
      "Parametric value %s was bound to different values at different "
      "places in invocation; saw: %d; then: %d",
      pdim_name, seen_value, arg_dim_i64);
  return ctx.deduce_ctx.TypeMismatchError(ctx.span, nullptr, param_type,
                                          nullptr, arg_type, message);
}

absl::Status ParametricBind(const ConcreteType& param_type,
                            const ConcreteType& arg_type,
                            ParametricBindContext& ctx) {
  if (auto* param_bits = dynamic_cast<const BitsType*>(&param_type)) {
    auto* arg_bits = dynamic_cast<const BitsType*>(&arg_type);
    XLS_RET_CHECK(arg_bits != nullptr);
    return ParametricBindBits(*param_bits, *arg_bits, ctx);
  }
  if (auto* param_tuple = dynamic_cast<const TupleType*>(&param_type)) {
    auto* arg_tuple = dynamic_cast<const TupleType*>(&arg_type);
    return ParametricBindTuple(*param_tuple, *arg_tuple, ctx);
  }
  if (auto* param_struct = dynamic_cast<const StructType*>(&param_type)) {
    auto* arg_struct = dynamic_cast<const StructType*>(&arg_type);
    const StructDef& param_nominal = param_struct->nominal_type();
    const StructDef& arg_nominal = arg_struct->nominal_type();
    if (&param_nominal != &arg_nominal) {
      std::string message =
          absl::StrFormat("parameter type name: '%s'; argument type name: '%s'",
                          param_nominal.identifier(), arg_nominal.identifier());
      return ctx.deduce_ctx.TypeMismatchError(ctx.span, nullptr, param_type,
                                              nullptr, arg_type, message);
    }
    return ParametricBindStruct(*param_struct, *arg_struct, ctx);
  }
  if (auto* param_array = dynamic_cast<const ArrayType*>(&param_type)) {
    auto* arg_array = dynamic_cast<const ArrayType*>(&arg_type);
    XLS_RET_CHECK(arg_array != nullptr);
    return ParametricBindArray(*param_array, *arg_array, ctx);
  }
  if (dynamic_cast<const EnumType*>(&param_type) != nullptr) {
    // Enum types aren't parameterizable.
    return absl::OkStatus();
  }
  if (dynamic_cast<const FunctionType*>(&param_type) != nullptr) {
    // Function types aren't parameterizable.
    return absl::OkStatus();
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
                      param_type.ToString(), ctx.span.ToString()));
}

}  // namespace xls::dslx
