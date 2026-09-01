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

#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xls/ir/function.h"
#include "xls/solvers/symex/symbolic_path.h"
#include "z3/src/api/z3.h"  // IWYU pragma: keep
#include "z3/src/api/z3_api.h"

namespace xls::solvers::symex {

CfgSymExEngine::CfgSymExEngine(Z3_context ctx, SymExOptions options)
    : ctx_(ctx), options_(std::move(options)) {}

absl::StatusOr<std::vector<SymbolicPath>> CfgSymExEngine::ExplorePaths(
    Function* fn) {
  return absl::UnimplementedError(
      "CFG Symbolic Execution engine exploration is not yet implemented.");
}

}  // namespace xls::solvers::symex
