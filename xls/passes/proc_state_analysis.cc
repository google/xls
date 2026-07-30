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

#include "xls/passes/proc_state_analysis.h"

#include <optional>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/status/statusor.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/interval_set.h"
#include "xls/ir/node.h"
#include "xls/ir/nodes.h"
#include "xls/ir/proc.h"
#include "xls/passes/proc_state_range_query_engine.h"
#include "xls/passes/query_engine.h"
#include "xls/solvers/solver.h"

namespace xls {

absl::StatusOr<std::vector<solvers::PredicateOfNode>> GetProcStateAssumptions(
    Proc* proc) {
  std::vector<solvers::PredicateOfNode> assumptions;
  ProcStateRangeQueryEngine query_engine;
  XLS_RETURN_IF_ERROR(query_engine.Populate(proc).status());
  for (Node* node : proc->nodes()) {
    if (!node->Is<StateRead>()) {
      continue;
    }
    IntervalSetTree ranges = query_engine.GetIntervals(node);
    if (absl::c_any_of(ranges.elements(), [](const IntervalSet& range) {
          return !range.IsMaximal();
        })) {
      assumptions.emplace_back(
          node, solvers::Predicate::IsCompatibleWith(std::move(ranges)));
    }
    std::optional<SharedTernaryTree> ternaries = query_engine.GetTernary(node);
    if (ternaries.has_value()) {
      assumptions.emplace_back(
          node, solvers::Predicate::IsCompatibleWith(std::move(*ternaries)));
    }
  }
  return assumptions;
}

}  // namespace xls
