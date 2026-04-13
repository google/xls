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

#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "xls/eco/graph.h"

namespace mcs {

// Partial solution S in the paper: each pair maps one query node to one
// target node.
using State = std::vector<std::pair<int, int>>;

struct MCSResult {
  State mapping;                  // Best node mapping S* found by the search.
  std::vector<int> unmatched_g1;  // graph1 nodes left outside the MCS.
  std::vector<int> unmatched_g2;  // graph2 nodes left outside the MCS.
  // Edge matches induced by `mapping`; this is not part of the paper but an
  // additional extension for our use case.
  std::vector<std::pair<int, int>> edge_mapping;
  int size = 0;       // |mapping|
  int edge_size = 0;  // |edge_mapping|
};

// Arguments:
//   mcs_cutoff: Stop early when remaining unmatched nodes <= this value
//       (negative disables the cutoff).
//   mcs_optimal: If false, allow the XLS ECO plateau heuristic to terminate
//       early.
//   mcs_timeout_sec: Stop MCS after this many seconds if non-negative.
MCSResult SolveMCS(const XLSGraph& graph1, const XLSGraph& graph2,
                   int mcs_cutoff = -1, bool mcs_optimal = true,
                   int mcs_timeout_sec = -1);

// Returns the subset of MCS pairs that touch at least one non-MCS neighbor.
absl::flat_hash_map<int, int> GetBoundaryNodes(const MCSResult& mcs,
                                               const XLSGraph& graph1,
                                               const XLSGraph& graph2);
// Utility functions exposed for testing
namespace internal {

// Candidate partition P(C) from the paper, represented as disjoint Xi x Yi
// classes.
struct CandidateClass {
  std::vector<int> query_nodes;
  std::vector<int> target_nodes;
};

using CandidatePartition = std::vector<CandidateClass>;
// Exclusion set D from Section 4.1, organized as u -> {v1, v2, ...}.
using ExclusionSet = absl::flat_hash_map<int, absl::flat_hash_set<int>>;

// Query-side structural equivalence classes Psi(u) from Definition 5 /
// Equation (7), adapted to XLS relation profiles.
struct EquivalenceInfo {
  std::vector<int> class_id;
  std::vector<std::vector<int>> members;
};

// Testing-facing hooks for the paper-specific pieces of RRSplit:
EquivalenceInfo ComputeStructuralEquivalenceForTesting(const XLSGraph& query);
CandidatePartition RefineCandidatePartitionForTesting(
    const CandidatePartition& candidates, int u, int v, const XLSGraph& query,
    const XLSGraph& target);
CandidatePartition RemoveEquivalentQueryNodesForTesting(
    const CandidatePartition& candidates, int u,
    const EquivalenceInfo& equivalence);
bool ShouldPruneByExclusionForTesting(int u, int v,
                                      const ExclusionSet& excluded,
                                      const EquivalenceInfo& equivalence);
bool SatisfiesMaximalityReductionForTesting(
    const CandidatePartition& candidates, int u, int v, const XLSGraph& query,
    const XLSGraph& target);
int ComputeUpperBoundForTesting(const State& partial_mapping,
                                const CandidatePartition& candidates,
                                const ExclusionSet& excluded,
                                const EquivalenceInfo& equivalence);

}  // namespace internal

}  // namespace mcs

#endif  // XLS_ECO_MCS_H_
