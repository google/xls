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

#include "xls/solvers/symex/cfg_symex_engine.h"

#include <cstdint>
#include <iterator>
#include <utility>
#include <vector>

#include "absl/cleanup/cleanup.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/function.h"
#include "xls/ir/node.h"
#include "xls/ir/node_util.h"
#include "xls/ir/nodes.h"
#include "xls/ir/topo_sort.h"
#include "xls/ir/value.h"
#include "xls/solvers/symex/symbolic_path.h"
#include "xls/solvers/symex/z3_semantics_encoder.h"
#include "xls/solvers/z3_utils.h"
#include "z3/src/api/z3.h"  // IWYU pragma: keep
#include "z3/src/api/z3_api.h"

namespace xls::solvers::symex {

CfgSymExEngine::CfgSymExEngine(Z3_context ctx) : ctx_(ctx), encoder_(ctx) {
  CHECK_NE(ctx_, nullptr);
}

absl::StatusOr<std::vector<SymbolicPath>> CfgSymExEngine::ExplorePaths(
    Function* fn) {
  // Topological sort ensures operands are evaluated before their users.
  XLS_ASSIGN_OR_RETURN(std::vector<Node*> topo_nodes, TopoSort(fn));

  std::vector<SymbolicPath> completed_paths;

  // Solver stack using push/pop during DFS.
  Z3_solver solver = solvers::z3::CreateSolver(ctx_, /*num_threads=*/1);
  Z3_solver_inc_ref(ctx_, solver);
  auto solver_cleanup = absl::Cleanup([&] { Z3_solver_dec_ref(ctx_, solver); });

  absl::flat_hash_map<const Node*, Z3_ast> env;
  std::vector<BranchDecision> current_decisions;
  Z3_ast initial_path_condition = Z3_mk_true(ctx_);

  std::vector<const Node*> const_topo_nodes(topo_nodes.begin(),
                                            topo_nodes.end());

  XLS_RETURN_IF_ERROR(ExploreDfs(fn, const_topo_nodes, /*node_idx=*/0, env,
                                 current_decisions, initial_path_condition,
                                 solver, completed_paths));

  return completed_paths;
}

absl::Status CfgSymExEngine::ExploreDfs(
    Function* fn, absl::Span<const Node* const> topo_nodes, int64_t node_idx,
    absl::flat_hash_map<const Node*, Z3_ast>& env,
    std::vector<BranchDecision>& current_decisions,
    Z3_ast current_path_condition, Z3_solver solver,
    std::vector<SymbolicPath>& completed_paths) {
  // Leaf node: all IR nodes evaluated along this path.
  if (node_idx >= static_cast<int64_t>(topo_nodes.size())) {
    SymbolicPath& path = completed_paths.emplace_back();
    path.branch_decisions = current_decisions;
    path.return_value = env.at(fn->return_value());
    path.path_condition = current_path_condition;

    // Extract satisfying parameter assignments from the active solver model.
    if (solver == nullptr || Z3_solver_check(ctx_, solver) != Z3_L_TRUE) {
      return absl::OkStatus();
    }
    Z3_model model = Z3_solver_get_model(ctx_, solver);
    if (model == nullptr) {
      return absl::OkStatus();
    }
    Z3_model_inc_ref(ctx_, model);
    auto model_cleanup = absl::Cleanup([&] { Z3_model_dec_ref(ctx_, model); });
    for (Param* param : fn->params()) {
      Z3_ast param_ast = env.at(param);
      XLS_ASSIGN_OR_RETURN(
          Value val,
          solvers::z3::NodeValue(ctx_, model, param_ast, param->GetType()));
      path.generated_test.push_back(
          ParamAssignment{.param = param, .value = val});
    }
    return absl::OkStatus();
  }

  const Node* node = topo_nodes[node_idx];

  // Fork path exploration at multiplexer (Select / PrioritySelect) nodes.
  if (GenericSelect::IsSelect(node)) {
    GenericSelect sel = *GenericSelect::TryFrom(const_cast<Node*>(node));
    Z3_ast selector_ast = env.at(sel.selector());
    int64_t num_cases = sel.cases().size();
    int64_t total_arms =
        sel.default_value().has_value() ? num_cases + 1 : num_cases;

    for (int64_t arm = 0; arm < total_arms; ++arm) {
      bool is_default = (arm >= num_cases);
      const Node* chosen_arm =
          is_default ? *sel.default_value() : sel.cases()[arm];

      XLS_ASSIGN_OR_RETURN(
          Z3_ast branch_cond,
          encoder_.EncodeMuxBranchCondition(node, arm, selector_ast));

      // Push branch condition onto the incremental solver stack.
      Z3_solver_push(ctx_, solver);
      auto pop_cleanup = absl::Cleanup([&] { Z3_solver_pop(ctx_, solver, 1); });
      Z3_solver_assert(ctx_, solver, branch_cond);

      if (Z3_solver_check(ctx_, solver) == Z3_L_FALSE) {
        // Branch is infeasible; backtrack immediately.
        continue;
      }

      // If feasible, bind multiplexer output to the chosen arm and recurse.
      Z3_ast chosen_arm_ast = env.at(chosen_arm);
      XLS_RET_CHECK(!env.contains(node));
      env[node] = chosen_arm_ast;
      auto env_cleanup = absl::Cleanup([&] { env.erase(node); });

      current_decisions.push_back(BranchDecision{
          .mux_node = node,
          .arm_index = arm,
      });
      auto decision_cleanup =
          absl::Cleanup([&] { current_decisions.pop_back(); });

      // Accumulate path condition.
      Z3_ast args[] = {current_path_condition, branch_cond};
      Z3_ast next_path_condition =
          Z3_mk_and(ctx_, /*num_args=*/std::size(args), args);

      XLS_RETURN_IF_ERROR(ExploreDfs(fn, topo_nodes, node_idx + 1, env,
                                     current_decisions, next_path_condition,
                                     solver, completed_paths));
    }
    return absl::OkStatus();
  }

  // Evaluate non-branching node from translated operand ASTs.
  std::vector<Z3_ast> operands;
  operands.reserve(node->operand_count());
  for (const Node* op : node->operands()) {
    operands.push_back(env.at(op));
  }

  XLS_ASSIGN_OR_RETURN(Z3_ast ast, encoder_.TranslateNode(node, operands));
  XLS_RET_CHECK(!env.contains(node));
  env[node] = ast;
  auto env_cleanup = absl::Cleanup([&] { env.erase(node); });

  return ExploreDfs(fn, topo_nodes, node_idx + 1, env, current_decisions,
                    current_path_condition, solver, completed_paths);
}

}  // namespace xls::solvers::symex
