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

#include <cstdint>
#include <vector>

#include "absl/status/statusor.h"
#include "xls/ir/function.h"
#include "xls/solvers/symex/symbolic_path.h"
#include "z3/src/api/z3.h"

namespace xls::solvers::symex {

// Configuration options for symbolic execution exploration.
struct SymExOptions {
  // Maximum number of distinct execution paths to explore before terminating.
  int64_t max_paths = 1000;
  // Maximum branch decision depth.
  int64_t max_depth = 500;
  // Whether to check feasibility via SMT solver at each branch point.
  bool check_feasibility = true;
};

// Common abstract base interface for XLS Symbolic Execution engines.
class SymExEngine {
 public:
  virtual ~SymExEngine() = default;

  // Explores feasible symbolic execution paths through the given XLS function.
  virtual absl::StatusOr<std::vector<SymbolicPath>> ExplorePaths(
      xls::Function* fn) = 0;

  // Returns total number of paths explored.
  virtual int64_t total_explored_paths() const = 0;

  // Returns number of feasible paths discovered.
  virtual int64_t feasible_paths() const = 0;
};

}  // namespace xls::solvers::symex

#endif  // XLS_SOLVERS_SYMEX_SYMEX_ENGINE_H_
