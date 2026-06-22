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

#ifndef XLS_PASSES_PROC_STATE_ANALYSIS_H_
#define XLS_PASSES_PROC_STATE_ANALYSIS_H_

#include <vector>

#include "absl/status/statusor.h"
#include "xls/ir/proc.h"
#include "xls/solvers/solver.h"

namespace xls {

// Analyzes the state elements of a Proc and returns a set of PredicateOfNode
// assumptions characterizing the ranges and known bits of those state elements.
absl::StatusOr<std::vector<solvers::PredicateOfNode>> GetProcStateAssumptions(
    Proc* proc);

}  // namespace xls

#endif  // XLS_PASSES_PROC_STATE_ANALYSIS_H_
