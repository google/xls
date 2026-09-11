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

#ifndef XLS_SOLVERS_SYMEX_SYMEX_ENGINE_H_
#define XLS_SOLVERS_SYMEX_SYMEX_ENGINE_H_

#include <cstddef>
#include <cstdint>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xls/ir/function.h"
#include "xls/ir/node_util.h"
#include "xls/solvers/symex/symbolic_path.h"
#include "xls/solvers/symex/z3_encoding_visitor.h"
#include "z3/src/api/z3.h"  // IWYU pragma: keep
#include "z3/src/api/z3_api.h"

namespace xls::solvers::symex {

// Configuration options for symbolic execution path exploration.
struct SymExOptions {
  // Maximum number of feasible paths to explore before terminating.
  // When <= 0, path exploration is unbounded.
  int64_t max_paths = 1000;
};

// Symbolic execution engine for XLS IR functions.
//
// Explores feasible symbolic execution paths through an XLS function using
// incremental SMT tree traversal (DFS with solver push/pop):
//
// 1. Function Encoding: Translates the function DAG into Z3 formulas up-front
//    via `Z3EncodingVisitor`, representing multiplexers as unconstrained SSA
//    variables.
//
// 2. Incremental SMT DFS: Traverses multiplexers in topological order. At each
//    branch point, pushes a solver frame, asserts branch condition and arm
//    equality constraints, and checks feasibility. If the partial path is
//    UNSAT, backtracks immediately; otherwise, recurses down the branch.
//
// 3. Leaf Test Generation: When all multiplexers are resolved along a feasible
//    path, extracts concrete satisfying test inputs from the solver model.
class SymExEngine {
 public:
  // Creates a SymExEngine associated with `ctx` and `options`. Returns an
  // error if `ctx == nullptr`.
  static absl::StatusOr<SymExEngine> Create(
      Z3_context ctx, SymExOptions options = SymExOptions());

  SymExEngine(const SymExEngine&) = delete;
  SymExEngine& operator=(const SymExEngine&) = delete;
  SymExEngine(SymExEngine&&) = default;
  SymExEngine& operator=(SymExEngine&&) = default;

  // Explores feasible symbolic execution paths through `fn`.
  //
  // Returns a list of all feasible `SymbolicPath` instances discovered during
  // traversal, up to `options.max_paths`.
  absl::StatusOr<std::vector<SymbolicPath>> ExplorePaths(Function* fn);

  // Returns the configuration options configured on this engine.
  const SymExOptions& options() const { return options_; }

 private:
  explicit SymExEngine(Z3_context ctx, SymExOptions options);

  // Recursive DFS traversal with incremental push/pop solver frames.
  void ExplorePathsInternal(int64_t select_idx,
                            absl::Span<const GenericSelect> selects,
                            Function* fn, Z3_solver solver,
                            Z3EncodingVisitor& encoder,
                            std::vector<BranchDecision>& current_decisions,
                            std::vector<Z3_ast>& current_conds,
                            std::vector<SymbolicPath>& completed_paths);

  // Extracts a SymbolicPath from the current satisfiable solver state.
  absl::StatusOr<SymbolicPath> ExtractSymbolicPath(
      Function* fn, Z3_solver solver, const Z3EncodingVisitor& encoder,
      absl::Span<const BranchDecision> decisions,
      absl::Span<const Z3_ast> conds);

  // Returns true if path exploration has reached the configured limit.
  bool ReachedMaxPaths(size_t path_count) const {
    return options_.max_paths > 0 &&
           static_cast<int64_t>(path_count) >= options_.max_paths;
  }

  Z3_context ctx_ = nullptr;
  SymExOptions options_;
};

}  // namespace xls::solvers::symex

#endif  // XLS_SOLVERS_SYMEX_SYMEX_ENGINE_H_
