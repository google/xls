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
#include <functional>
#include <memory>
#include <utility>
#include <vector>

#include "absl/cleanup/cleanup.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/function.h"
#include "xls/ir/node.h"
#include "xls/ir/nodes.h"
#include "xls/ir/topo_sort.h"
#include "xls/solvers/symex/symbolic_path.h"
#include "xls/solvers/symex/symex_engine.h"
#include "xls/solvers/symex/z3_semantics_encoder.h"
#include "xls/solvers/z3_utils.h"
#include "z3/src/api/z3.h"

namespace xls::solvers::symex {

CfgSymExEngine::CfgSymExEngine(Z3_context ctx, SymExOptions options)
    : ctx_(ctx),
      options_(std::move(options)),
      encoder_(std::make_unique<Z3SemanticsEncoder>(ctx)) {}

absl::StatusOr<std::vector<SymbolicPath>> CfgSymExEngine::ExplorePaths(
    xls::Function* fn) {
  XLS_ASSIGN_OR_RETURN(std::vector<xls::Node*> topo_nodes, xls::TopoSort(fn));

  std::vector<SymbolicPath> completed_paths;
  total_explored_paths_ = 0;
  feasible_paths_ = 0;

  absl::flat_hash_map<const xls::Select*, int64_t> decisions;
  absl::Status exploration_status = absl::OkStatus();

  auto find_next_undecided_mux = [&]() -> const xls::Select* {
    std::vector<const xls::Node*> worklist = {fn->return_value()};
    absl::flat_hash_set<const xls::Node*> visited;

    while (!worklist.empty()) {
      const xls::Node* curr = worklist.back();
      worklist.pop_back();

      if (!visited.insert(curr).second) {
        continue;
      }

      if (curr->Is<xls::Select>()) {
        const auto* sel = curr->As<xls::Select>();
        auto it = decisions.find(sel);
        if (it != decisions.end()) {
          int64_t arm = it->second;
          if (arm < sel->cases().size()) {
            worklist.push_back(sel->cases()[arm]);
          } else if (sel->default_value().has_value()) {
            worklist.push_back(*sel->default_value());
          }
          worklist.push_back(sel->selector());
        } else {
          return sel;
        }
      } else {
        for (const xls::Node* op : curr->operands()) {
          worklist.push_back(op);
        }
      }
    }
    return nullptr;
  };

  std::function<void()> explore = [&]() {
    if (!exploration_status.ok() ||
        completed_paths.size() >= static_cast<size_t>(options_.max_paths)) {
      return;
    }

    const xls::Select* next_mux = find_next_undecided_mux();
    if (next_mux != nullptr) {
      int64_t num_cases = next_mux->cases().size();
      int64_t total_arms =
          next_mux->default_value().has_value() ? num_cases + 1 : num_cases;

      for (int64_t arm = 0; arm < total_arms; ++arm) {
        auto cond_or = encoder_->EncodeMuxBranchCondition(next_mux, arm);
        if (!cond_or.ok()) {
          exploration_status = cond_or.status();
          return;
        }

        decisions[next_mux] = arm;

        bool feasible = true;
        if (options_.check_feasibility && ctx_ != nullptr) {
          Z3_solver solver = xls::solvers::z3::CreateSolver(ctx_, 1);
          Z3_solver_inc_ref(ctx_, solver);
          auto solver_cleanup =
              absl::Cleanup([&] { Z3_solver_dec_ref(ctx_, solver); });

          for (const auto& [mux, chosen_arm] : decisions) {
            auto arm_cond = encoder_->EncodeMuxBranchCondition(mux, chosen_arm);
            if (!arm_cond.ok()) {
              exploration_status = arm_cond.status();
              return;
            }
            Z3_solver_assert(ctx_, solver, *arm_cond);
          }

          Z3_lbool res = Z3_solver_check(ctx_, solver);
          if (res == Z3_L_FALSE) {
            feasible = false;
          }
        }

        if (feasible) {
          explore();
        }

        decisions.erase(next_mux);

        if (!exploration_status.ok() ||
            completed_paths.size() >= static_cast<size_t>(options_.max_paths)) {
          return;
        }
      }
      return;
    }

    total_explored_paths_++;
    absl::flat_hash_map<const xls::Node*, Z3_ast> env;
    std::vector<BranchDecision> branch_decisions;
    std::vector<Z3_ast> path_conditions;

    for (xls::Node* node : topo_nodes) {
      if (node->Is<xls::Select>()) {
        const auto* sel = node->As<xls::Select>();
        auto it = decisions.find(sel);
        if (it != decisions.end()) {
          int64_t arm = it->second;
          if (arm < sel->cases().size()) {
            env[node] = env.at(sel->cases()[arm]);
          } else {
            env[node] = env.at(*sel->default_value());
          }
          auto cond_or = encoder_->EncodeMuxBranchCondition(sel, arm);
          if (!cond_or.ok()) {
            exploration_status = cond_or.status();
            return;
          }
          path_conditions.push_back(*cond_or);
          branch_decisions.push_back(BranchDecision{
              .mux_node = node,
              .arm_index = arm,
              .is_feasible = true,
          });
          continue;
        }
      }

      std::vector<Z3_ast> operands;
      operands.reserve(node->operand_count());
      for (const xls::Node* op : node->operands()) {
        operands.push_back(env.at(op));
      }

      auto translated_or = encoder_->TranslateNode(node, operands);
      if (!translated_or.ok()) {
        exploration_status = translated_or.status();
        return;
      }
      env[node] = *translated_or;
    }

    SymbolicPath path;
    path.is_feasible = true;
    path.branch_decisions = std::move(branch_decisions);
    path.return_value = env.at(fn->return_value());
    path.node_translations = std::move(env);
    if (path_conditions.empty()) {
      path.path_condition = Z3_mk_true(ctx_);
    } else if (path_conditions.size() == 1) {
      path.path_condition = path_conditions[0];
    } else {
      path.path_condition =
          Z3_mk_and(ctx_, path_conditions.size(), path_conditions.data());
    }

    completed_paths.push_back(std::move(path));
  };

  explore();
  XLS_RETURN_IF_ERROR(exploration_status);

  feasible_paths_ = static_cast<int64_t>(completed_paths.size());
  return completed_paths;
}

}  // namespace xls::solvers::symex
