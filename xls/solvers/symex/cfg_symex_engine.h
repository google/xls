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

#ifndef XLS_SOLVERS_SYMEX_CFG_SYMEX_ENGINE_H_
#define XLS_SOLVERS_SYMEX_CFG_SYMEX_ENGINE_H_

#include <cstdint>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xls/ir/function.h"
#include "xls/ir/node.h"
#include "xls/ir/nodes.h"
#include "xls/solvers/symex/symbolic_path.h"
#include "xls/solvers/symex/z3_semantics_encoder.h"
#include "z3/src/api/z3.h"  // IWYU pragma: keep
#include "z3/src/api/z3_api.h"

namespace xls::solvers::symex {

// Control Flow Graph (CFG) Symbolic Execution Engine for XLS functions.
//
// Performs forward Depth-First Search (DFS) over a topologically linearized
// sequence of XLS IR nodes. Multiplexers (`xls::Select`, `xls::PrioritySelect`)
// act as branch points: the engine forks execution across each candidate arm,
// asserting the arm's predicate into an incremental Z3 solver stack
// (`Z3_solver_push`/`pop`) and pruning infeasible paths.
//
// Non-branching arithmetic, logic, and tuple operations are translated
// symbolically using `Z3SemanticsEncoder` and propagated through an on-the-fly
// symbolic environment. At each completed path leaf, the engine extracts a
// concrete witness input assignment from the active solver model and returns
// the resulting `SymbolicPath`.
class CfgSymExEngine {
 public:
  explicit CfgSymExEngine(Z3_context ctx);

  CfgSymExEngine(const CfgSymExEngine&) = delete;
  CfgSymExEngine& operator=(const CfgSymExEngine&) = delete;

  // Explores all feasible symbolic execution paths through `fn`.
  absl::StatusOr<std::vector<SymbolicPath>> ExplorePaths(Function* fn);

 private:
  // Recursive DFS step traversing linearized nodes starting at `node_idx`.
  absl::Status ExploreDfs(Function* fn,
                          absl::Span<const Node* const> topo_nodes,
                          int64_t node_idx,
                          absl::flat_hash_map<const Node*, Z3_ast>& env,
                          std::vector<BranchDecision>& current_decisions,
                          Z3_ast current_path_condition, Z3_solver solver,
                          std::vector<SymbolicPath>& completed_paths);

  Z3_context ctx_;
  Z3SemanticsEncoder encoder_;
};

}  // namespace xls::solvers::symex

#endif  // XLS_SOLVERS_SYMEX_CFG_SYMEX_ENGINE_H_
