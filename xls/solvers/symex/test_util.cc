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

#include "xls/solvers/symex/test_util.h"

#include <vector>

#include "absl/types/span.h"
#include "xls/solvers/symex/symbolic_path.h"
#include "z3/src/api/z3.h"  // IWYU pragma: keep
#include "z3/src/api/z3_api.h"

namespace xls::solvers::symex {

bool AreMutuallyExclusive(Z3_context ctx, Z3_ast cond1, Z3_ast cond2) {
  Z3_solver solver = Z3_mk_solver(ctx);
  Z3_solver_inc_ref(ctx, solver);
  Z3_solver_assert(ctx, solver, cond1);
  Z3_solver_assert(ctx, solver, cond2);
  Z3_lbool result = Z3_solver_check(ctx, solver);
  Z3_solver_dec_ref(ctx, solver);
  return result == Z3_L_FALSE;
}

bool IsExhaustiveCoverage(Z3_context ctx,
                          absl::Span<const SymbolicPath> paths) {
  if (paths.empty()) {
    return false;
  }
  std::vector<Z3_ast> conds;
  conds.reserve(paths.size());
  for (const SymbolicPath& path : paths) {
    conds.push_back(path.path_condition);
  }
  Z3_ast combined = Z3_mk_or(ctx, conds.size(), conds.data());
  Z3_ast not_combined = Z3_mk_not(ctx, combined);

  Z3_solver solver = Z3_mk_solver(ctx);
  Z3_solver_inc_ref(ctx, solver);
  Z3_solver_assert(ctx, solver, not_combined);
  Z3_lbool result = Z3_solver_check(ctx, solver);
  Z3_solver_dec_ref(ctx, solver);
  return result == Z3_L_FALSE;
}

}  // namespace xls::solvers::symex
