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

#include <cstdint>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "xls/eco/graph.h"

namespace mcs {

// Types for MCS search
using State = std::vector<std::pair<int, int>>;
using CandidateSet = absl::flat_hash_map<int, std::vector<int>>;
using ForbiddenSet = absl::flat_hash_set<std::uint64_t>;

// Result structure for MCS
struct MCSResult {
  State mapping;
  std::vector<int> unmatched_g1;
  std::vector<int> unmatched_g2;
  std::vector<std::pair<int, int>> edge_mapping;
  int size = 0;
  int edge_size = 0;
};

// Solve Maximum Common Subgraph problem
MCSResult SolveMCS(const XLSGraph& graph1, const XLSGraph& graph2,
                   int mcs_cutoff = -1);

absl::flat_hash_map<int, int> GetBoundaryNodes(const MCSResult& mcs,
                                               const XLSGraph& graph1,
                                               const XLSGraph& graph2);

}  // namespace mcs

#endif  // XLS_ECO_MCS_H_