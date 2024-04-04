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

#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "gtest/gtest.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "xls/common/strong_int.h"

namespace xls {
namespace {

// Increment a bitvector (e.g.: considered as an unsigned integer), returning
// false if there was overflow and true otherwise.
bool IncrementBitVector(std::vector<bool>* vec) {
  bool carry = true;
  for (int32_t i = 0; i < vec->size(); i++) {
    if (carry) {
      carry = (*vec)[i];
      (*vec)[i] = !(*vec)[i];
    }
  }
  return !carry;
}

// Convert a bitvector whose length is equal to the number of edges in the path
// into a cut on the path, where bits equal to 1 represent cut edges and bits
// equal to 0 represent uncut edges.
PathCut CutFromBitVector(const PathGraph<int32_t, int32_t>& path,
                         const std::vector<bool>& vec) {
  CHECK_EQ(vec.size(), path.NumEdges())
      << "Size of bitvector must be equal to number of edges";
  std::vector<PathEdgeId> cut_edges;
  for (int32_t i = 0; i < path.NumEdges(); i++) {
    if (vec[i]) {
      cut_edges.push_back(PathEdgeId(i));
    }
  }
  return path.CutEdgesToPathCut(cut_edges);
}

// Enumerate all cuts for a path, calling the given function on each cut.
void EnumerateAllCuts(const PathGraph<int32_t, int32_t>& path,
                      std::function<void(const PathCut&)> callback) {
  std::vector<bool> bitvector;
  bitvector.resize(path.NumEdges(), false);
  do {
    PathCut cut = CutFromBitVector(path, bitvector);
    callback(cut);
  } while (IncrementBitVector(&bitvector));
}

// Determine whether the given cut violates the max node weight constraint.
bool PathCutIsValid(const PathGraph<int32_t, int32_t>& path, const PathCut& cut,
                    int32_t maximum_weight) {
  for (const std::vector<PathNodeId>& piece : cut) {
    int32_t piece_weight(0);
    for (PathNodeId node : piece) {
      piece_weight += path.WeightOfNode(node);
    }
    if (piece_weight > maximum_weight) {
      return false;
    }
  }
  return true;
}

// Compute the cost of the given cut.
int32_t PathCutCost(const PathGraph<int32_t, int32_t>& path,
                    const PathCut& cut) {
  int32_t result(0);
  for (int i = 0; i < cut.size(); i++) {
    if (std::optional<PathEdgeId> edge =
            path.NodeSuccessorEdge(cut[i].back())) {
      result += path.WeightOfEdge(*edge);
    }
  }
  return result;
}

// A brute force solution for the path cut problem, to compare against the
// smarter dynamic programming solution in `path_cut.cc`.
std::optional<PathCut> BruteForcePathCut(
    const PathGraph<int32_t, int32_t>& path, int32_t maximum_weight) {
  std::optional<PathCut> best;
  std::optional<int32_t> best_cost;
  EnumerateAllCuts(path, [&](const PathCut& cut) {
    if (!PathCutIsValid(path, cut, maximum_weight)) {
      return;
    }
    int32_t cost = PathCutCost(path, cut);
    if (!best_cost || (cost < *best_cost)) {
      best = cut;
      best_cost = cost;
    }
  });
  return best;
}

using PG = PathGraph<int32_t, int32_t>;

template <typename T>
PartialDifferenceMonoid<T> AddSubPDM() {
  return {[]() { return 0; },
          [](T x, T y) -> std::optional<T> { return x + y; },
          [](T x, T y) -> std::optional<T> { return x - y; }};
}

template <typename T>
TotalOrder<T> LessThanTotalOrder() {
  return {[](T x, T y) { return x == y; }, [](T x, T y) { return x < y; }};
}

PG CreatePathGraph(absl::Span<const int32_t> node_weights,
                   absl::Span<const int32_t> edge_weights) {
  return *PG::Create(node_weights, edge_weights, AddSubPDM<int32_t>(),
                     AddSubPDM<int32_t>(), LessThanTotalOrder<int32_t>(),
                     LessThanTotalOrder<int32_t>());
}

TEST(PathCutTest, SingleNodeTest) {
  // (50)
  PG path = CreatePathGraph({50}, {});
  EXPECT_EQ(path.ComputePathCut(30), std::nullopt);
  EXPECT_EQ(path.ComputePathCut(70).value(), PathCut({{PathNodeId(0)}}));
}

TEST(PathCutTest, SimpleTest) {
  // (50) >-- 10 --> (10) >-- 10 --> (20) >-- 10 --> (50)
  PG path = CreatePathGraph({50, 10, 20, 50}, {10, 10, 10});
  EXPECT_EQ(path.ComputePathCut(70),
            std::make_optional<PathCut>({{PathNodeId(0), PathNodeId(1)},
                                         {PathNodeId(2), PathNodeId(3)}}));
  for (int32_t i = 0; i < 100; i += 5) {
    EXPECT_EQ(path.ComputePathCut(i), BruteForcePathCut(path, i));
  }
}

TEST(PathCutTest, ComplexTest) {
  // Generated by fair dice roll
  PG path = CreatePathGraph({17, 16, 18, 93, 55, 75, 51, 63},
                            {23, 34, 61, 22, 76, 54, 77});
  for (int32_t i = 0; i < 300; i += 1) {
    VLOG(3) << "i = " << i << "\n";
    int32_t max_weight(i);
    std::optional<PathCut> smart = path.ComputePathCut(max_weight);
    std::optional<PathCut> brute = BruteForcePathCut(path, max_weight);
    EXPECT_EQ(smart.has_value(), brute.has_value());
    if (smart.has_value()) {
      VLOG(3) << "brute = " << PathCutToString(*brute) << "\n";
      VLOG(3) << "smart = " << PathCutToString(*smart) << "\n";
      EXPECT_TRUE(PathCutIsValid(path, *brute, max_weight));
      EXPECT_TRUE(PathCutIsValid(path, *smart, max_weight));
      EXPECT_EQ(PathCutCost(path, *smart), PathCutCost(path, *brute));
    }
  }
}

struct ColoredNodeWeight {
  std::optional<std::pair<PathNodeId, PathNodeId>> interval;
  int32_t weight;
};

PartialDifferenceMonoid<ColoredNodeWeight> ColoredNodeWeightPDM() {
  return {[]() -> ColoredNodeWeight {
            return {std::nullopt, 0};
          },
          [](const ColoredNodeWeight& x,
             const ColoredNodeWeight& y) -> std::optional<ColoredNodeWeight> {
            if (!x.interval) {
              CHECK_EQ(x.weight, 0);
              return {{y.interval, y.weight}};
            }
            if (!y.interval) {
              CHECK_EQ(y.weight, 0);
              return {{x.interval, x.weight}};
            }
            if (static_cast<int32_t>(x.interval->second) + 1 !=
                static_cast<int32_t>(y.interval->first)) {
              return std::nullopt;
            }
            return {{{{x.interval->first, y.interval->second}},
                     x.weight + y.weight}};
          },
          [](const ColoredNodeWeight& x,
             const ColoredNodeWeight& y) -> std::optional<ColoredNodeWeight> {
            if (!x.interval) {
              return std::nullopt;
            }
            if (!y.interval) {
              CHECK_EQ(y.weight, 0);
              return x;
            }
            if (x.interval->first != y.interval->first) {
              return std::nullopt;
            }
            PathNodeId start(static_cast<int32_t>(y.interval->second) + 1);
            PathNodeId end(x.interval->second);
            return {{{{start, end}}, x.weight - y.weight}};
          }};
}

TotalOrder<ColoredNodeWeight> ColoredNodeWeightTotalOrder() {
  return {[](const ColoredNodeWeight& x, const ColoredNodeWeight& y) -> bool {
            return x.weight == y.weight;
          },
          [](const ColoredNodeWeight& x, const ColoredNodeWeight& y) -> bool {
            return x.weight < y.weight;
          }};
}

TEST(PathCutTest, NonIntNodeWeightsTest) {
  EXPECT_TRUE(true);
  auto cnw = [](int32_t weight, int32_t node) -> ColoredNodeWeight {
    PathNodeId path_node(node);
    return {{{path_node, path_node}}, weight};
  };
  PathGraph<ColoredNodeWeight, int32_t> path =
      *PathGraph<ColoredNodeWeight, int32_t>::Create(
          {cnw(50, 0), cnw(10, 1), cnw(20, 2), cnw(50, 3)}, {10, 10, 10},
          ColoredNodeWeightPDM(), AddSubPDM<int32_t>(),
          ColoredNodeWeightTotalOrder(), LessThanTotalOrder<int32_t>());

  EXPECT_EQ(path.ComputePathCut({std::nullopt, 70}),
            std::make_optional<PathCut>({{PathNodeId(0), PathNodeId(1)},
                                         {PathNodeId(2), PathNodeId(3)}}));
}

}  // namespace
}  // namespace xls
