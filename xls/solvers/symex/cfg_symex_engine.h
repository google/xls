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
#include <vector>

#include "absl/status/statusor.h"
#include "xls/ir/function.h"
#include "xls/solvers/symex/symbolic_path.h"
#include "xls/solvers/symex/symex_engine.h"
#include "xls/solvers/symex/z3_semantics_encoder.h"
#include "z3/src/api/z3.h"

namespace xls::solvers::symex {

// Control Flow Graph (CFG) Symbolic Execution Engine for XLS IR functions.
//
// Performs topological depth-first search (DFS) path exploration over XLS
// hardware Intermediate Representations, linearizing multiplexer/select nodes
// into path branches and verifying branch feasibility using incremental SMT
// solving.
class CfgSymExEngine : public SymExEngine {
 public:
  explicit CfgSymExEngine(Z3_context ctx,
                          SymExOptions options = SymExOptions());

  // Explores feasible symbolic execution paths through `fn`.
  absl::StatusOr<std::vector<SymbolicPath>> ExplorePaths(
      xls::Function* fn) override;

  int64_t total_explored_paths() const override {
    return total_explored_paths_;
  }
  int64_t feasible_paths() const override { return feasible_paths_; }

 private:
  Z3_context ctx_;
  SymExOptions options_;
  std::unique_ptr<Z3SemanticsEncoder> encoder_;
  int64_t total_explored_paths_ = 0;
  int64_t feasible_paths_ = 0;
};

}  // namespace xls::solvers::symex

#endif  // XLS_SOLVERS_SYMEX_CFG_SYMEX_ENGINE_H_
