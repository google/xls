// Copyright 2026 The XLS Authors
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

#include "xls/solvers/symex/symbolic_path.h"

#include <utility>
#include <vector>

#include "absl/cleanup/cleanup.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/nodes.h"
#include "xls/ir/type.h"
#include "xls/ir/value.h"
#include "xls/solvers/z3_utils.h"
#include "z3/src/api/z3.h"

namespace xls::solvers::symex {

absl::StatusOr<std::vector<std::pair<const xls::Param*, xls::Value>>>
SymbolicPath::Solve(Z3_context ctx,
                    absl::Span<const xls::Param* const> params) const {
  if (!is_feasible || path_condition == nullptr) {
    return absl::FailedPreconditionError(
        "Cannot solve an infeasible or unconstrained path.");
  }

  Z3_solver solver = xls::solvers::z3::CreateSolver(ctx, /*num_threads=*/1);
  Z3_solver_inc_ref(ctx, solver);
  auto solver_cleanup = absl::Cleanup([&] { Z3_solver_dec_ref(ctx, solver); });
  Z3_solver_assert(ctx, solver, path_condition);

  Z3_lbool check_result = Z3_solver_check(ctx, solver);
  if (check_result != Z3_L_TRUE) {
    return absl::NotFoundError(
        "Path condition is unsatisfiable under SMT solver.");
  }

  Z3_model model = Z3_solver_get_model(ctx, solver);
  Z3_model_inc_ref(ctx, model);
  auto model_cleanup = absl::Cleanup([&] { Z3_model_dec_ref(ctx, model); });

  std::vector<std::pair<const xls::Param*, xls::Value>> param_values;
  param_values.reserve(params.size());
  for (const xls::Param* param : params) {
    auto it = node_translations.find(param);
    Z3_ast param_ast = nullptr;
    if (it != node_translations.end()) {
      param_ast = it->second;
    } else {
      std::string param_name(param->name());
      Z3_symbol sym = Z3_mk_string_symbol(ctx, param_name.c_str());
      Z3_sort sort = xls::solvers::z3::TypeToSort(ctx, *param->GetType());
      param_ast = Z3_mk_const(ctx, sym, sort);
    }
    XLS_ASSIGN_OR_RETURN(
        xls::Value val,
        xls::solvers::z3::NodeValue(ctx, model, param_ast,
                                    const_cast<xls::Type*>(param->GetType())));
    param_values.push_back({param, val});
  }

  return param_values;
}

}  // namespace xls::solvers::symex
