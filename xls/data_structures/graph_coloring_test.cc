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

#include "xls/data_structures/graph_coloring.h"

#include <string>
#include <vector>

#include "gtest/gtest.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"

namespace xls {
namespace {

using V = std::string;

std::vector<absl::flat_hash_set<V>> RLFFromMap(
    const absl::flat_hash_map<V, absl::flat_hash_set<V>>& neighborhood) {
  absl::flat_hash_map<V, absl::flat_hash_set<V>> symmetric_neighborhood;
  absl::flat_hash_set<V> nodes;
  for (const auto& [node, neighbors] : neighborhood) {
    for (const auto& neighbor : neighbors) {
      nodes.insert(node);
      nodes.insert(neighbor);
      symmetric_neighborhood[node].insert(neighbor);
      symmetric_neighborhood[neighbor].insert(node);
    }
  }

  return RecursiveLargestFirstColoring<V>(
      nodes, [&](const V& node) -> absl::flat_hash_set<V> {
        return symmetric_neighborhood.at(node);
      });
}

std::vector<absl::flat_hash_set<V>> Z3FromMap(
    const absl::flat_hash_map<V, absl::flat_hash_set<V>>& neighborhood) {
  absl::flat_hash_map<V, absl::flat_hash_set<V>> symmetric_neighborhood;
  absl::flat_hash_set<V> nodes;
  for (const auto& [node, neighbors] : neighborhood) {
    for (const auto& neighbor : neighbors) {
      nodes.insert(node);
      nodes.insert(neighbor);
      symmetric_neighborhood[node].insert(neighbor);
      symmetric_neighborhood[neighbor].insert(node);
    }
  }

  return Z3Coloring<V>(nodes, [&](const V& node) -> absl::flat_hash_set<V> {
    return symmetric_neighborhood.at(node);
  });
}

bool IsValidColoring(
    const absl::flat_hash_map<V, absl::flat_hash_set<V>>& neighborhood,
    const std::vector<absl::flat_hash_set<V>>& coloring) {
  absl::flat_hash_set<V> nodes;
  for (const auto& [node, neighbors] : neighborhood) {
    for (const auto& neighbor : neighbors) {
      nodes.insert(node);
      nodes.insert(neighbor);
    }
  }

  // Every node must be used exactly once.
  for (const absl::flat_hash_set<V>& color_class : coloring) {
    for (const V& element : color_class) {
      if (!nodes.contains(element)) {
        return false;
      }
      nodes.erase(element);
    }
  }
  return nodes.empty();
}

TEST(GraphColoringTest, Bipartite) {
  absl::flat_hash_map<V, absl::flat_hash_set<V>> graph;
  graph["a"].insert("x");
  graph["a"].insert("y");
  graph["a"].insert("z");
  graph["b"].insert("x");
  graph["c"].insert("w");
  EXPECT_LE(RLFFromMap(graph).size(), 2);
  EXPECT_TRUE(IsValidColoring(graph, RLFFromMap(graph)));
  EXPECT_LE(Z3FromMap(graph).size(), 2);
  EXPECT_TRUE(IsValidColoring(graph, Z3FromMap(graph)));
}

TEST(GraphColoringTest, Cycle) {
  absl::flat_hash_map<V, absl::flat_hash_set<V>> graph;
  graph["a"].insert("b");
  graph["b"].insert("c");
  graph["c"].insert("d");
  graph["d"].insert("e");
  graph["e"].insert("a");
  EXPECT_EQ(RLFFromMap(graph).size(), 3);
  EXPECT_TRUE(IsValidColoring(graph, RLFFromMap(graph)));
  EXPECT_EQ(Z3FromMap(graph).size(), 3);
  EXPECT_TRUE(IsValidColoring(graph, Z3FromMap(graph)));
  graph.erase("e");
  graph["d"].erase("e");
  graph["d"].insert("a");
  EXPECT_EQ(RLFFromMap(graph).size(), 2);
  EXPECT_TRUE(IsValidColoring(graph, RLFFromMap(graph)));
  EXPECT_EQ(Z3FromMap(graph).size(), 2);
  EXPECT_TRUE(IsValidColoring(graph, Z3FromMap(graph)));
}

TEST(GraphColoringTest, Wheel) {
  absl::flat_hash_map<V, absl::flat_hash_set<V>> graph;
  graph["a"].insert("b");
  graph["b"].insert("c");
  graph["c"].insert("d");
  graph["d"].insert("e");
  graph["e"].insert("a");
  graph["center"].insert("a");
  graph["center"].insert("b");
  graph["center"].insert("c");
  graph["center"].insert("d");
  graph["center"].insert("e");
  EXPECT_EQ(RLFFromMap(graph).size(), 4);
  EXPECT_TRUE(IsValidColoring(graph, RLFFromMap(graph)));
  graph.erase("e");
  graph["d"].erase("e");
  graph["d"].insert("a");
  graph["center"].erase("e");
  EXPECT_EQ(RLFFromMap(graph).size(), 3);
  EXPECT_TRUE(IsValidColoring(graph, RLFFromMap(graph)));
  EXPECT_EQ(Z3FromMap(graph).size(), 3);
  EXPECT_TRUE(IsValidColoring(graph, Z3FromMap(graph)));
}

}  // namespace
}  // namespace xls
