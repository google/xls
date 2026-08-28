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

#include "xls/solvers/symex/symex_engine.h"

#include <cstdint>
#include <utility>
#include <vector>

#include "absl/cleanup/cleanup.h"
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
#include "xls/ir/op.h"
#include "xls/ir/topo_sort.h"
#include "xls/ir/value.h"
#include "xls/solvers/symex/symbolic_path.h"
#include "xls/solvers/symex/z3_encoding_visitor.h"
#include "xls/solvers/z3_utils.h"
#include "z3/src/api/z3.h"  // IWYU pragma: keep
#include "z3/src/api/z3_api.h"

namespace xls::solvers::symex {
namespace {

int64_t GetTotalArms(GenericSelect sel) {
  return sel.default_value().has_value() ? sel.cases().size() + 1
                                         : sel.cases().size();
}

}  // namespace

absl::StatusOr<SymExEngine> SymExEngine::Create(Z3_context ctx,
                                                SymExOptions options) {
  XLS_RET_CHECK_NE(ctx, nullptr);
  return SymExEngine(ctx, std::move(options));
}

SymExEngine::SymExEngine(Z3_context ctx, SymExOptions options)
    : ctx_(ctx), options_(std::move(options)) {
  CHECK_NE(ctx_, nullptr);
}

absl::StatusOr<SymbolicPath> SymExEngine::ExtractSymbolicPath(
    Function* fn, Z3_solver solver, const Z3EncodingVisitor& encoder,
    absl::Span<const BranchDecision> decisions,
    absl::Span<const Z3_ast> conds) {
  Z3_model model = Z3_solver_get_model(ctx_, solver);
  XLS_RET_CHECK_NE(model, nullptr)
      << "Failed to extract model from satisfiable solver state";
  Z3_model_inc_ref(ctx_, model);
  absl::Cleanup model_cleanup = [&] { Z3_model_dec_ref(ctx_, model); };

  SymbolicPath path;
  path.branch_decisions.assign(decisions.begin(), decisions.end());
  // Compute symbolic return value along this path by substituting each
  // multiplexer's unconstrained SSA variable with its chosen arm expression.
  std::vector<Z3_ast> from_asts;
  std::vector<Z3_ast> to_asts;
  from_asts.reserve(decisions.size());
  to_asts.reserve(decisions.size());

  for (const BranchDecision& decision : decisions) {
    if (decision.mux_node == nullptr) {
      continue;
    }
    GenericSelect sel =
        *GenericSelect::TryFrom(const_cast<Node*>(decision.mux_node));
    Node* chosen_arm = (decision.arm_index >= sel.cases().size())
                           ? *sel.default_value()
                           : sel.cases()[decision.arm_index];
    Z3_ast arm_ast = encoder.GetNodeAst(chosen_arm);
    if (!from_asts.empty()) {
      arm_ast = Z3_substitute(ctx_, arm_ast, from_asts.size(), from_asts.data(),
                              to_asts.data());
    }
    from_asts.push_back(encoder.GetNodeAst(decision.mux_node));
    to_asts.push_back(arm_ast);
  }

  Z3_ast return_ast = encoder.GetNodeAst(fn->return_value());
  if (!from_asts.empty()) {
    return_ast = Z3_substitute(ctx_, return_ast, from_asts.size(),
                               from_asts.data(), to_asts.data());
  }
  path.return_value = return_ast;

  path.path_condition =
      (conds.empty())
          ? Z3_mk_true(ctx_)
          : (conds.size() == 1 ? conds[0]
                               : Z3_mk_and(ctx_, conds.size(), conds.data()));

  for (Param* param : fn->params()) {
    Z3_ast param_ast = encoder.GetNodeAst(param);
    XLS_ASSIGN_OR_RETURN(
        Value val,
        solvers::z3::NodeValue(ctx_, model, param_ast, param->GetType()));
    path.generated_test.push_back(
        ParamAssignment{.param = param, .value = val});
  }

  return path;
}

void SymExEngine::ExplorePathsInternal(
    int64_t select_idx, absl::Span<const GenericSelect> selects, Function* fn,
    Z3_solver solver, Z3EncodingVisitor& encoder,
    std::vector<BranchDecision>& current_decisions,
    std::vector<Z3_ast>& current_conds,
    std::vector<SymbolicPath>& completed_paths) {
  if (ReachedMaxPaths(completed_paths.size())) {
    return;
  }

  // Base case: leaf reached where all multiplexer branches have been chosen
  // and verified satisfiable along this path.
  if (select_idx == selects.size()) {
    if (Z3_solver_check(ctx_, solver) == Z3_L_TRUE) {
      absl::StatusOr<SymbolicPath> path_or = ExtractSymbolicPath(
          fn, solver, encoder, current_decisions, current_conds);
      if (path_or.ok()) {
        completed_paths.push_back(std::move(*path_or));
      }
    }
    return;
  }

  GenericSelect sel = selects[select_idx];
  int64_t total_arms = GetTotalArms(sel);

  for (int64_t branch = 0;
       branch < total_arms && !ReachedMaxPaths(completed_paths.size());
       ++branch) {
    absl::StatusOr<Z3_ast> branch_cond_or =
        encoder.EncodeMuxBranchCondition(sel.AsNode(), branch);
    if (!branch_cond_or.ok()) {
      continue;
    }
    absl::StatusOr<Z3_ast> arm_equality_or =
        encoder.EncodeMuxArmEquality(sel.AsNode(), branch);
    if (!arm_equality_or.ok()) {
      continue;
    }

    Z3_ast branch_cond = *branch_cond_or;
    Z3_ast arm_equality = *arm_equality_or;

    // Push solver frame for this branch decision.
    Z3_solver_push(ctx_, solver);

    Z3_solver_assert(ctx_, solver, branch_cond);
    Z3_solver_assert(ctx_, solver, arm_equality);

    // Only recurse if the path condition is satisfiable.
    if (Z3_solver_check(ctx_, solver) == Z3_L_TRUE) {
      current_decisions.push_back(BranchDecision{
          .mux_node = sel.AsNode(),
          .arm_index = branch,
      });
      current_conds.push_back(branch_cond);

      ExplorePathsInternal(select_idx + 1, selects, fn, solver, encoder,
                           current_decisions, current_conds, completed_paths);

      current_conds.pop_back();
      current_decisions.pop_back();
    }

    Z3_solver_pop(ctx_, solver, 1);
  }
}

absl::StatusOr<std::vector<SymbolicPath>> SymExEngine::ExplorePaths(
    Function* fn) {
  // Translate all function nodes into Z3 expressions in a single post-order
  // pass, representing multiplexers as unconstrained SSA variables.
  Z3EncodingVisitor encoder(ctx_, fn);
  XLS_RETURN_IF_ERROR(fn->Accept(&encoder));

  // Collect all multiplexers in topological order.
  XLS_ASSIGN_OR_RETURN(std::vector<Node*> topo_nodes, TopoSort(fn));
  std::vector<GenericSelect> selects;
  for (Node* node : topo_nodes) {
    if (node->OpIn({Op::kSel, Op::kPrioritySel})) {
      selects.push_back(*GenericSelect::TryFrom(node));
    }
  }

  // Explore paths over selector choices using incremental push/pop DFS.
  Z3_solver solver = solvers::z3::CreateSolver(ctx_, /*num_threads=*/1);
  Z3_solver_inc_ref(ctx_, solver);
  absl::Cleanup solver_cleanup = [&] { Z3_solver_dec_ref(ctx_, solver); };

  std::vector<SymbolicPath> completed_paths;
  std::vector<BranchDecision> current_decisions;
  std::vector<Z3_ast> current_conds;

  ExplorePathsInternal(0, selects, fn, solver, encoder, current_decisions,
                       current_conds, completed_paths);
  return completed_paths;
}

}  // namespace xls::solvers::symex
