#ifndef MCS_H
#define MCS_H

#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "graph.h"

namespace mcs {
// Maximum Common Subgraph (MCS) solver using recursive backtracking

// Types for MCS search
using State = std::vector<std::pair<int, int>>;
using CandidateSet = std::unordered_map<int, std::vector<int>>;
using ForbiddenSet = std::unordered_set<int>;

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
std::unordered_map<int, int> GetBoundaryNodes(const MCSResult& mcs,
                                              const XLSGraph& graph1,
                                              const XLSGraph& graph2);
}  // namespace mcs
#endif  // MCS_H
