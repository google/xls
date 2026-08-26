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

#ifndef XLS_SOLVERS_SYMEX_SYMBOLIC_PATH_H_
#define XLS_SOLVERS_SYMEX_SYMBOLIC_PATH_H_

#include <cstdint>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xls/ir/node.h"
#include "xls/ir/nodes.h"
#include "xls/ir/value.h"
#include "z3/src/api/z3.h"

namespace xls::solvers::symex {

// Represents a branch decision made at a multiplexer or select node.
struct BranchDecision {
  const xls::Node* mux_node = nullptr;  // The select / priority select node
  int64_t arm_index = 0;                // Selected arm index
  bool is_feasible = false;             // Feasibility status via SMT solver
};

// Represents a complete symbolic execution path through an XLS IR function.
struct SymbolicPath {
  Z3_ast path_condition = nullptr;  // Conjunction of all path branch conditions
  Z3_ast return_value = nullptr;    // Symbolic expression for function output
  bool is_feasible = false;         // True if path condition is satisfiable
  std::vector<BranchDecision> branch_decisions;  // Sequence of branch choices

  // Mapping from IR nodes/parameters to their translated solver ASTs.
  absl::flat_hash_map<const xls::Node*, Z3_ast> node_translations;

  // Solves the path condition using Z3 and extracts a concrete input vector.
  absl::StatusOr<std::vector<std::pair<const xls::Param*, xls::Value>>> Solve(
      Z3_context ctx, absl::Span<const xls::Param* const> params) const;
};

}  // namespace xls::solvers::symex

#endif  // XLS_SOLVERS_SYMEX_SYMBOLIC_PATH_H_
