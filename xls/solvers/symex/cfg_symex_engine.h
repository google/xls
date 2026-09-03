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
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xls/ir/function.h"
#include "xls/ir/node.h"
#include "xls/ir/nodes.h"
#include "xls/ir/value.h"
#include "xls/solvers/symex/concolic_input_spec.h"
#include "xls/solvers/symex/symbolic_path.h"
#include "xls/solvers/symex/z3_semantics_encoder.h"
#include "z3/src/api/z3.h"  // IWYU pragma: keep
#include "z3/src/api/z3_api.h"

namespace xls::solvers::symex {

// Options configuring symbolic path exploration.
struct SymExOptions {
  // Concrete input parameter bindings (concolic execution).
  // Parameters bound in this specification are treated as known concrete
  // constants during path exploration, enabling upfront branch pruning.
  ConcolicInputSpec concrete_inputs;

  // Maximum number of feasible paths to explore before stopping early.
  // If not set (nullopt), all feasible paths will be explored.
  std::optional<int64_t> max_paths = std::nullopt;
};

// Control Flow Graph (CFG) Symbolic Execution Engine for XLS functions.
//
// Performs forward Depth-First Search (DFS) over a topologically linearized
// sequence of XLS IR nodes. Multiplexers (`xls::Select`) act as branch points:
// the engine forks execution across each candidate arm, asserting the arm's
// predicate into an incremental Z3 solver stack (`Z3_solver_push`/`pop`) and
// pruning infeasible paths.
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

  // Explores feasible symbolic execution paths through `fn` according to
  // `options`.
  absl::StatusOr<std::vector<SymbolicPath>> ExplorePaths(
      Function* fn, const SymExOptions& options = {});

 private:
  // Recursive DFS step traversing linearized nodes starting at `node_idx`.
  absl::Status ExploreDfs(Function* fn,
                          absl::Span<const Node* const> topo_nodes,
                          int64_t node_idx,
                          absl::flat_hash_map<const Node*, Z3_ast>& env,
                          std::vector<BranchDecision>& current_decisions,
                          Z3_ast current_path_condition, Z3_solver solver,
                          const SymExOptions& options,
                          std::vector<SymbolicPath>& completed_paths);

  Z3_context ctx_;
  std::unique_ptr<Z3SemanticsEncoder> encoder_;
};

}  // namespace xls::solvers::symex

#endif  // XLS_SOLVERS_SYMEX_CFG_SYMEX_ENGINE_H_
