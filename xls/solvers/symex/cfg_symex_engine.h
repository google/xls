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

#include "absl/status/statusor.h"
#include "xls/ir/function.h"
#include "xls/solvers/symex/symbolic_path.h"
#include "z3/src/api/z3.h"  // IWYU pragma: keep
#include "z3/src/api/z3_api.h"

namespace xls::solvers::symex {

// Configuration options for symbolic execution path exploration.
struct SymExOptions {
  // Maximum number of feasible paths to explore before terminating.
  int64_t max_paths = 1000;

  // Maximum search depth (multiplexer branch depth) along any path.
  int64_t max_depth = 256;
};

// Control Flow Graph (CFG) Symbolic Execution Engine for XLS IR functions.
//
// Explores symbolic execution paths by performing a forward Depth-First Search
// (DFS) over the linearized (topologically sorted) sequence of XLS IR nodes.
class CfgSymExEngine {
 public:
  explicit CfgSymExEngine(Z3_context ctx,
                          SymExOptions options = SymExOptions());

  CfgSymExEngine(const CfgSymExEngine&) = delete;
  CfgSymExEngine& operator=(const CfgSymExEngine&) = delete;

  // Explores feasible symbolic execution paths through `fn`.
  absl::StatusOr<std::vector<SymbolicPath>> ExplorePaths(Function* fn);

  int64_t total_explored_paths() const { return total_explored_paths_; }
  int64_t feasible_paths() const { return feasible_paths_; }

 private:
  Z3_context ctx_;
  SymExOptions options_;
  int64_t total_explored_paths_ = 0;
  int64_t feasible_paths_ = 0;
};

}  // namespace xls::solvers::symex

#endif  // XLS_SOLVERS_SYMEX_CFG_SYMEX_ENGINE_H_
