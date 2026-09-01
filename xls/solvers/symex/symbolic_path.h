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
#include <vector>

#include "xls/ir/node.h"
#include "z3/src/api/z3.h"  // IWYU pragma: keep
#include "z3/src/api/z3_api.h"

namespace xls::solvers::symex {

// Represents a branch decision made at a multiplexer or select node during
// exploration.
struct BranchDecision {
  const Node* mux_node = nullptr;  // The select / priority select node
  int64_t arm_index = 0;           // Selected arm index
};

// Represents a complete symbolic execution path through an XLS IR function.
struct SymbolicPath {
  Z3_ast path_condition =
      nullptr;  // Conjunction of all branch conditions along this path
  Z3_ast return_value = nullptr;  // Symbolic expression for the function output
  std::vector<BranchDecision> branch_decisions;  // Sequence of branch choices
};

}  // namespace xls::solvers::symex

#endif  // XLS_SOLVERS_SYMEX_SYMBOLIC_PATH_H_
