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

#ifndef XLS_DATA_STRUCTURES_PATH_CUT_H_
#define XLS_DATA_STRUCTURES_PATH_CUT_H_

#include <cstdint>
#include <functional>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "xls/common/strong_int.h"

namespace xls {

XLS_DEFINE_STRONG_INT_TYPE(PathNodeId, int32_t);
XLS_DEFINE_STRONG_INT_TYPE(PathEdgeId, int32_t);

// This is a function table representing types that have the ability to take
// sums and differences, with a neutral zero element. The classic example of a
// set that has this structure is the natural numbers; taking the difference of
// two natural numbers is not always a natural number. The reason we have
// `difference` rather than `inverse` (making this a partial group) is because
// for sets like the naturals the inverse is _never_ defined, whereas the
// difference sometimes is.
//
// Concretely, this is needed in the scheduler for a node weight data structure
// that stores the critical path lengths as the path cut algorithm builds up
// parts of the partition.
template <typename T>
struct PartialDifferenceMonoid {
  std::function<T()> zero;
  std::function<std::optional<T>(const T&, const T&)> sum;
  std::function<std::optional<T>(const T&, const T&)> difference;
};

// This is a function table representing types equipped with a total order, to
// avoid depending on overloads of comparison operators that may not correspond
// to the desired behavior.
template <typename T>
struct TotalOrder {
  std::function<bool(const T&, const T&)> equals;
  std::function<bool(const T&, const T&)> less_than;

  bool less_than_or_eq(const T& x, const T& y) {
    return less_than(x, y) || equals(x, y);
  }
};

// A partition of the set of nodes into equivalence classes that are contiguous.
//
// The nodes within each equivalence class should be in increasing order, and
// the equivalence classes themselves should be sorted in lexicographic order.
using PathCut = std::vector<std::vector<PathNodeId>>;

// Display a `PathCut` as a string.
std::string PathCutToString(const PathCut& cut);

// A type of nonempty path/linear graphs, i.e.: graphs that look like:
//
//   0   1   2   3   4   5   6
//   o───o───o───o───o───o───o
//     0   1   2   3   4   5
//
// with weights on both the nodes and edges.
//
// The node weight type is the `NW` parameter; the edge weight type is the `EW`
// parameter.
template <typename NW, typename EW>
class PathGraph {
 public:
  // Create a new nonempty path graph. There must be exactly one more node
  // weight than edge weight in the two input vectors, since a path graph always
  // has one more node than edge.
  static absl::StatusOr<PathGraph> Create(
      absl::Span<const NW> node_weights, absl::Span<const EW> edge_weights,
      PartialDifferenceMonoid<NW> node_weight_pdm,
      PartialDifferenceMonoid<EW> edge_weight_pdm,
      TotalOrder<NW> node_weight_total_order,
      TotalOrder<EW> edge_weight_total_order) {
    if (edge_weights.size() + 1 != node_weights.size()) {
      return absl::InternalError(
          absl::StrFormat("Size of node weights (%d) must be 1 greater than "
                          "size of edge weights (%d).",
                          node_weights.size(), edge_weights.size()));
    }
    PathGraph result;
    result.node_weights_ =
        std::vector<NW>(node_weights.begin(), node_weights.end());
    result.edge_weights_ =
        std::vector<EW>(edge_weights.begin(), edge_weights.end());
    result.node_weight_pdm_ = node_weight_pdm;
    result.edge_weight_pdm_ = edge_weight_pdm;
    result.node_weight_total_order_ = node_weight_total_order;
    result.edge_weight_total_order_ = edge_weight_total_order;
    return result;
  }

  // Returns the number of nodes, which is always one greater than the number of
  // edges in the path graph.
  int32_t NumNodes() const { return node_weights_.size(); }

  // Returns the number of edges, which is always one fewer than the number of
  // nodes in the path graph.
  int32_t NumEdges() const { return edge_weights_.size(); }

  // Returns the node weight associated with the given node.
  NW WeightOfNode(PathNodeId node) const {
    return node_weights_.at(static_cast<int32_t>(node));
  }

  // Returns the edge weight associated with the given edge.
  EW WeightOfEdge(PathEdgeId edge) const {
    return edge_weights_.at(static_cast<int32_t>(edge));
  }

  // Returns the node that comes before the given edge.
  //
  // Precondition: the given edge is a valid edge.
  PathNodeId EdgeSource(PathEdgeId edge) const {
    CHECK_GE(static_cast<int32_t>(edge), 0);
    CHECK_LT(static_cast<int32_t>(edge), edge_weights_.size());
    return PathNodeId(static_cast<int32_t>(edge));
  }

  // Returns the node that comes after the given edge.
  //
  // Precondition: the given edge is a valid edge.
  PathNodeId EdgeTarget(PathEdgeId edge) const {
    CHECK_GE(static_cast<int32_t>(edge), 0);
    CHECK_LT(static_cast<int32_t>(edge), edge_weights_.size());
    return PathNodeId(static_cast<int32_t>(edge) + 1);
  }

  // Returns the edge after the given node, if there is one.
  std::optional<PathEdgeId> NodeSuccessorEdge(PathNodeId node) const {
    CHECK_GE(static_cast<int32_t>(node), 0);
    CHECK_LT(static_cast<int32_t>(node), node_weights_.size());
    if (static_cast<int32_t>(node) == node_weights_.size() - 1) {
      return std::nullopt;
    }
    return PathEdgeId(static_cast<int32_t>(node));
  }

  // Returns the edge previous to the given node, if there is one.
  std::optional<PathEdgeId> NodePredecessorEdge(PathNodeId node) const {
    CHECK_GE(static_cast<int32_t>(node), 0);
    CHECK_LT(static_cast<int32_t>(node), node_weights_.size());
    if (static_cast<int32_t>(node) <= 0) {
      return std::nullopt;
    }
    return PathEdgeId(static_cast<int32_t>(node) - 1);
  }

  // Compute the optimal (in terms of minimizing the total weight of cut edges)
  // `PathCut` for a given `Path`, assuming that the sum of the node weights in
  // each piece of the `PathCut` must be less than or equal to the
  // given `PathNodeWeight`.
  //
  // Returns `std::nullopt` when there is no cut that satisfies the constraint
  // given by `maximum_weight`.
  //
  // The algorithm is based on https://cs.stackexchange.com/a/138417
  std::optional<PathCut> ComputePathCut(NW maximum_weight) {
    // Set up a cache for prefix sums of the node weight list.
    absl::flat_hash_map<PathNodeId, NW> prefix_sums = ComputePrefixSums();

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
    // prefix of the path graph, subject to the constraint that no piece has
    // total node weight greater than `m`.
    //
    // The reason we need the `std::nullopt` is to account for the possibility
    // that there is no cut edge prior to `k`. It forms the base case of the
    // recurrence.
    //
    //    0     1     2     3     t    t+1   k-1    k    k+1    n
    //    o─────o─────o─────o─ … ─o─────o─ … ─o─────o─────o─ … ─o
    //                               ∧                 ∧
    //    The previously cut edge ───┘                 └─── A[k] assumes there
    //    assumed by A[k].                                  is a cut here.
    //
    // Note that when k = n, there is no "cut edge"; this is by design, so that
    // we don't need to duplicate the inner loop logic to find the optimal
    // "leftover" non-prefix piece at the end.
    absl::flat_hash_map<CacheKey, CacheItem> cache;

    // Initialize the base case of the recurrence.
    cache[std::nullopt] = {edge_weight_pdm_.zero(), {}};

    for (PathNodeId k(0); static_cast<int32_t>(k) < NumNodes(); k++) {
      std::optional<CacheItem> best;

      {
        // The body of the dynamic programming inner loop; corresponds to the
        // min{A[t] | …} in the recurrence above.
        auto loop_body = [&](CacheKey t) {
          std::optional<NW> prefix_sum_diff_maybe =
              t.has_value()
                  ? node_weight_pdm_.difference(prefix_sums[k], prefix_sums[*t])
                  : prefix_sums[k];
          CHECK(prefix_sum_diff_maybe.has_value())
              << "The given PartialDifferenceMonoid for node weights failed";
          NW prefix_sum_diff = *prefix_sum_diff_maybe;
          if (node_weight_total_order_.less_than_or_eq(prefix_sum_diff,
                                                       maximum_weight) &&
              (!best.has_value() ||
               edge_weight_total_order_.less_than(cache[t].cost, best->cost))) {
            best = cache[t];
          }
        };

        // Actually run the dynamic programming inner loop.
        loop_body(std::nullopt);
        for (PathNodeId t(0); t < k; t++) {
          loop_body(t);
        }
      }

      // This means there was a node that was too big for the given value of
      // `maximum_weight`.
      if (!best.has_value()) {
        return std::nullopt;
      }

      // Corresponds to the … + wₑ(k, k + 1) part of the recurrence.
      if (std::optional<PathEdgeId> e = NodeSuccessorEdge(k)) {
        best->cost += WeightOfEdge(*e);
        best->cut_edges.push_back(*e);
      }

      cache[k] = *best;
    }

    VLOG(3) << "cache = " << CacheToString(cache) << "\n";

    CacheItem optimal = cache[PathNodeId(NumNodes() - 1)];
    std::sort(optimal.cut_edges.begin(), optimal.cut_edges.end());

    return CutEdgesToPathCut(optimal.cut_edges);
  }

  // Convert a list of "cut edges" into a `PathCut`, i.e.: if the given `Path`
  // has six edges, and the given list of cut edges is `[2, 4]`, then the
  // returned partition will be `[[0, 1, 2], [3, 4], [5]]`.
  PathCut CutEdgesToPathCut(absl::Span<const PathEdgeId> cut_edges) const {
    PathCut result;

    int32_t i = 0;
    std::vector<PathNodeId> piece;
    for (PathNodeId n(0); static_cast<int32_t>(n) < NumNodes(); n++) {
      piece.push_back(n);
      if ((i < cut_edges.size()) && (n == EdgeSource(cut_edges[i]))) {
        result.push_back(piece);
        piece = std::vector<PathNodeId>();
        i++;
      }
    }
    result.push_back(piece);

    return result;
  }

 private:
  PathGraph() = default;
  std::vector<NW> node_weights_;
  std::vector<EW> edge_weights_;
  PartialDifferenceMonoid<NW> node_weight_pdm_;
  PartialDifferenceMonoid<EW> edge_weight_pdm_;
  TotalOrder<NW> node_weight_total_order_;
  TotalOrder<EW> edge_weight_total_order_;

  using CacheKey = std::optional<PathNodeId>;

  struct CacheItem {
    EW cost;
    std::vector<PathEdgeId> cut_edges;
  };

  // Display the value of the dynamic programming cache as a string.
  std::string CacheToString(
      absl::flat_hash_map<CacheKey, CacheItem>& cache) const {
    std::vector<std::string> items;
    for (PathNodeId n(0); static_cast<int32_t>(n) < NumNodes(); n++) {
      std::stringstream ss;
      ss << cache[n].cost;
      items.push_back(
          absl::StrFormat("(%s, [%s])", ss.str(),
                          absl::StrJoin(cache[n].cut_edges, ", ",
                                        [](std::string* out, PathEdgeId edge) {
                                          absl::StrAppend(out, edge.value());
                                        })));
    }
    return absl::StrJoin(items, ", ");
  }

  absl::flat_hash_map<PathNodeId, NW> ComputePrefixSums() {
    absl::flat_hash_map<PathNodeId, NW> result;

    result[PathNodeId(0)] = WeightOfNode(PathNodeId(0));
    for (PathNodeId n(1); static_cast<int32_t>(n) < NumNodes(); n++) {
      auto sum_maybe =
          node_weight_pdm_.sum(result[n - PathNodeId(1)], WeightOfNode(n));
      CHECK(sum_maybe)
          << "The given PartialDifferenceMonoid for node weights failed";
      result[n] = *sum_maybe;
    }

    return result;
  }
};

}  // namespace xls

#endif  // XLS_DATA_STRUCTURES_PATH_CUT_H_
