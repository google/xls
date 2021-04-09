// Copyright 2021 The XLS Authors
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

#include "xls/data_structures/path_cut.h"

#include <stdint.h>

#include <algorithm>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/types/optional.h"
#include "absl/types/span.h"
#include "xls/common/logging/log_lines.h"
#include "xls/common/logging/log_message.h"
#include "xls/common/logging/logging.h"
#include "xls/common/logging/vlog_is_on.h"
#include "xls/common/strong_int.h"

namespace xls {

std::string PathCutToString(const PathCut& cut) {
  std::vector<std::string> pieces;
  for (const auto& piece : cut) {
    pieces.push_back(absl::StrFormat(
        "[%s]",
        absl::StrJoin(piece, ", ", [](std::string* out, PathNodeId node) {
          absl::StrAppend(out, node.value());
        })));
  }
  return absl::StrJoin(pieces, ", ");
}

PathCut CutEdgesToPathCut(const PathGraph& path,
                          absl::Span<const PathEdgeId> cut_edges) {
  PathCut result;

  int32_t i = 0;
  std::vector<PathNodeId> piece;
  for (PathNodeId n(0); static_cast<int32_t>(n) < path.NumNodes(); n++) {
    piece.push_back(n);
    if ((i < cut_edges.size()) && (n == path.EdgeSource(cut_edges[i]))) {
      result.push_back(piece);
      piece = std::vector<PathNodeId>();
      i++;
    }
  }
  result.push_back(piece);

  return result;
}

// Compute all the prefix sums of the list of node weights.
static absl::flat_hash_map<PathNodeId, PathNodeWeight> ComputePrefixSums(
    const PathGraph& path) {
  absl::flat_hash_map<PathNodeId, PathNodeWeight> result;

  result[PathNodeId(0)] = path.WeightOfNode(PathNodeId(0));
  for (PathNodeId n(1); static_cast<int32_t>(n) < path.NumNodes(); n++) {
    result[n] = result[n - PathNodeId(1)] + path.WeightOfNode(n);
  }

  return result;
}

using CacheKey = absl::optional<PathNodeId>;

struct CacheItem {
  PathEdgeWeight cost;
  std::vector<PathEdgeId> cut_edges;
};

// Display the value of the dynamic programming cache as a string.
static std::string CacheToString(
    const PathGraph& path, absl::flat_hash_map<CacheKey, CacheItem> cache) {
  std::vector<std::string> items;
  for (PathNodeId n(0); static_cast<int32_t>(n) < path.NumNodes(); n++) {
    items.push_back(
        absl::StrFormat("(%d, [%s])", static_cast<int64_t>(cache[n].cost),
                        absl::StrJoin(cache[n].cut_edges, ", ",
                                      [](std::string* out, PathEdgeId edge) {
                                        absl::StrAppend(out, edge.value());
                                      })));
  }
  return absl::StrJoin(items, ", ");
}

absl::optional<PathCut> ComputePathCut(const PathGraph& path,
                                       PathNodeWeight maximum_weight) {
  // Set up a cache for prefix sums of the node weight list.
  absl::flat_hash_map<PathNodeId, PathNodeWeight> prefix_sums =
      ComputePrefixSums(path);

  // Elements of this cache represent the optimal solution to the problem,
  // assuming that the solution contains a cut edge immediately after a
  // particular node.
  //
  // The recurrence we're solving is:
  //
  //     A[ø] = 0
  //     A[k] = min { A[t]
  //                | t ∈ {0 .. k - 1} ∪ {ø}
  //                , (t = ø) ∨ (Σ{wₙ(i) | t + 1 ≤ i ≤ k} ≤ m)
  //                }
  //            + wₑ(k, k + 1)
  //
  // In other words, A[k] gives the solution to the minimum cost cut of a
  // prefix of the path graph, subject to the constraint that no piece has total
  // node weight greater than `m`.
  //
  // The reason we need the `absl::nullopt` is to account for the possibility
  // that there is no cut edge prior to `k`. It forms the base case of the
  // recurrence.
  //
  //    0     1     2     3     t    t+1   k-1    k    k+1    n
  //    o─────o─────o─────o─ … ─o─────o─ … ─o─────o─────o─ … ─o
  //                               ∧                 ∧
  //    The previously cut edge ───┘                 └─── A[k] assumes there is
  //    assumed by A[k].                                  a cut here.
  //
  // Note that when k = n, there is no "cut edge"; this is by design, so that we
  // don't need to duplicate the inner loop logic to find the optimal "leftover"
  // non-prefix piece at the end.
  absl::flat_hash_map<CacheKey, CacheItem> cache;

  // Initialize the base case of the recurrence.
  cache[absl::nullopt] = {PathEdgeWeight(0), {}};

  for (PathNodeId k(0); static_cast<int32_t>(k) < path.NumNodes(); k++) {
    absl::optional<CacheItem> best;

    {
      // The body of the dynamic programming inner loop; corresponds to the
      // min{A[t] | …} in the recurrence above.
      auto loop_body = [&](CacheKey t) {
        PathNodeWeight prefix_sum_diff =
            prefix_sums[k] - (t ? prefix_sums[*t] : PathNodeWeight(0));
        if ((prefix_sum_diff <= maximum_weight) &&
            (!best || (cache[t].cost < best->cost))) {
          best = cache[t];
        }
      };

      // Actually run the dynamic programming inner loop.
      loop_body(absl::nullopt);
      for (PathNodeId t(0); t < k; t++) {
        loop_body(t);
      }
    }

    // This means there was a node that was too big for the given value of
    // `maximum_weight`.
    if (!best.has_value()) {
      return absl::nullopt;
    }

    // Corresponds to the … + wₑ(k, k + 1) part of the recurrence.
    if (absl::optional<PathEdgeId> e = path.NodeSuccessorEdge(k)) {
      best->cost += path.WeightOfEdge(*e);
      best->cut_edges.push_back(*e);
    }

    cache[k] = *best;
  }

  XLS_VLOG(3) << "cache = " << CacheToString(path, cache) << "\n";

  CacheItem optimal = cache[PathNodeId(path.NumNodes() - 1)];
  std::sort(optimal.cut_edges.begin(), optimal.cut_edges.end());

  return CutEdgesToPathCut(path, optimal.cut_edges);
}

}  // namespace xls
