// Copyright 2022 The XLS Authors
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

#ifndef XLS_DATA_STRUCTURES_MAXIMUM_CLIQUE_H_
#define XLS_DATA_STRUCTURES_MAXIMUM_CLIQUE_H_

#include <functional>
#include <memory>
#include <optional>
#include <string>

#include "absl/container/btree_map.h"
#include "absl/container/btree_set.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "ortools/linear_solver/linear_solver.h"

namespace xls {

// Compute the maximum clique in the given graph. This supports graphs of up to
// around 100 nodes. More cleverness will be needed if we want to support bigger
// graphs than that.
template <typename V, typename Compare = std::less<V>>
absl::StatusOr<absl::btree_set<V, Compare>> MaximumClique(
    const absl::btree_set<V, Compare>& vertices,
    std::function<absl::btree_set<V, Compare>(const V&)> neighborhood) {
  namespace or_tools = ::operations_research;

  std::unique_ptr<or_tools::MPSolver> solver(
      or_tools::MPSolver::CreateSolver("SAT"));
  if (!solver) {
    return absl::InternalError("SAT solver unavailable");
  }

  const double infinity = solver->infinity();

  absl::btree_map<V, or_tools::MPVariable*, Compare> variables;
  {
    int64_t i = 0;
    for (const V& vertex : vertices) {
      std::string name = absl::StrFormat("var%d", i);
      variables[vertex] = solver->MakeIntVar(0, 1, name);
      ++i;
    }
  }

  for (const V& x : vertices) {
    for (const V& y : vertices) {
      if (neighborhood(x).contains(y)) {
        continue;
      }
      or_tools::MPConstraint* constraint =
          solver->MakeRowConstraint(-infinity, 1);
      constraint->SetCoefficient(variables.at(x), 1);
      constraint->SetCoefficient(variables.at(y), 1);
    }
  }

  or_tools::MPObjective* objective = solver->MutableObjective();
  for (const V& vertex : vertices) {
    objective->SetCoefficient(variables.at(vertex), 1);
  }
  objective->SetMaximization();

  or_tools::MPSolver::ResultStatus status = solver->Solve();
  if (status != or_tools::MPSolver::OPTIMAL) {
    return absl::InternalError("Could not find the maximum clique");
  }

  absl::btree_set<V, Compare> result;
  for (const V& vertex : vertices) {
    // This should really be == 1.0 but comparing floats for equality is dodgy.
    double value = variables.at(vertex)->solution_value();
    if (value > 0.9) {
      result.insert(vertex);
    }
    if ((value != 1.0) && (value != 0.0)) {
      LOG(WARNING) << "maximum clique ILP gave non-zero/one answer: " << value;
    }
  }

  return result;
}

}  // namespace xls

#endif  // XLS_DATA_STRUCTURES_MAXIMUM_CLIQUE_H_
