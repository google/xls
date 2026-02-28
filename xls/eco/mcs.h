// Copyright 2025 The XLS Authors
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

#ifndef XLS_ECO_MCS_H_
#define XLS_ECO_MCS_H_

#include "absl/container/flat_hash_set.h"
#include "absl/container/flat_hash_map.h"
#include <vector>

#include "xls/eco/graph.h"

namespace mcs {
// Maximum Common Subgraph (MCS) solver using recursive backtracking

// Types for MCS search
using State = std::vector<std::pair<int, int>>;
using CandidateSet = absl::flat_hash_map<int, std::vector<int>>;
using ForbiddenSet = absl::flat_hash_set<int>;

// Result structure for MCS
struct MCSResult {
  State mapping;                  // Best node mapping found (u -> v pairs)
  std::vector<int> unmatched_g1;  // Unmatched nodes from graph1
  std::vector<int> unmatched_g2;  // Unmatched nodes from graph2
  std::vector<std::pair<int, int>>
      edge_mapping;   // Best edge mapping found (e1_idx -> e2_idx pairs)
  int size = 0;       // Size of the MCS (number of matched nodes)
  int edge_size = 0;  // Number of matched edges
};

// Solve Maximum Common Subgraph problem
// Returns the largest common subgraph mapping between graph1 and graph2
// mcs_cutoff: Stop early when remaining unmatched nodes <= this value
//             (negative = disabled, runs to completion)
MCSResult SolveMCS(const XLSGraph& graph1, const XLSGraph& graph2,
                   int mcs_cutoff = -1);
absl::flat_hash_map<int, int> GetBoundaryNodes(const MCSResult& mcs,
                                              const XLSGraph& graph1,
                                              const XLSGraph& graph2);
}  // namespace mcs
#endif  // MCS_H
