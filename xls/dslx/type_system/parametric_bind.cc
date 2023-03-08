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
#include "xls/dslx/errors.h"

namespace xls::dslx {

absl::Status ParametricBindConcreteTypeDim(
    const ConcreteType& param_type, const ConcreteTypeDim& param_dim,
    const ConcreteType& arg_type, const ConcreteTypeDim& arg_dim,
    const Span& span,
    const absl::flat_hash_map<std::string, std::unique_ptr<ConcreteType>>&
        parametric_binding_types,
    const absl::flat_hash_map<std::string, Expr*>& parametric_default_exprs,
    absl::flat_hash_map<std::string, InterpValue>& parametric_env) {
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
  if (!parametric_env.contains(pdim_name)) {
    XLS_RET_CHECK(parametric_binding_types.contains(pdim_name))
        << "Cannot bind " << pdim_name << " : it has no associated type.";
    XLS_VLOG(5) << "Binding " << pdim_name << " to " << arg_dim_i64;
    const ConcreteType& type = *parametric_binding_types.at(pdim_name);
    XLS_ASSIGN_OR_RETURN(ConcreteTypeDim bit_count, type.GetTotalBitCount());
    XLS_ASSIGN_OR_RETURN(int64_t width, bit_count.GetAsInt64());
    parametric_env.emplace(
        pdim_name,
        InterpValue::MakeUBits(/*bit_count=*/width, /*value=*/arg_dim_i64));
    return absl::OkStatus();
  }

  const InterpValue& seen = parametric_env.at(pdim_name);
  XLS_ASSIGN_OR_RETURN(int64_t seen_value, seen.GetBitValueInt64());
  if (seen_value == arg_dim_i64) {
    return absl::OkStatus();  // No contradiction.
  }

  // We see a conflict between something we previously observed and something
  // we are now observing -- make an appropriate error.
  if (auto it = parametric_default_exprs.find(pdim_name);
      it != parametric_default_exprs.end() && it->second != nullptr) {
    const Expr* expr = it->second;
    // Error is violated constraint.
    std::string message = absl::StrFormat(
        "Parametric constraint violated, saw %s = %d; then %s = %s = %d",
        pdim_name, seen_value, pdim_name, expr->ToString(), arg_dim_i64);
    auto saw_type =
        std::make_unique<BitsType>(/*signed=*/false, /*size=*/seen_value);
    return XlsTypeErrorStatus(span, *saw_type, arg_type, message);
  }

  // Error is conflicting argument types.
  std::string message = absl::StrFormat(
      "Parametric value %s was bound to different values at different "
      "places in invocation; saw: %d; then: %d",
      pdim_name, seen_value, arg_dim_i64);
  return XlsTypeErrorStatus(span, param_type, arg_type, message);
}

}  // namespace xls::dslx
