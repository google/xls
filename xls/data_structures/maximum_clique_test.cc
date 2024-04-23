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

#include "xls/data_structures/maximum_clique.h"

#include <cstdint>
#include <functional>
#include <random>
#include <string>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/container/btree_map.h"
#include "absl/container/btree_set.h"
#include "absl/random/distributions.h"

namespace xls {
namespace {

using V = std::string;

using Graph = absl::btree_map<V, absl::btree_set<V>>;

Graph Symmetrize(const Graph& neighborhood) {
  Graph symmetric_neighborhood;
  for (const auto& [node, neighbors] : neighborhood) {
    for (const auto& neighbor : neighbors) {
      symmetric_neighborhood[node].insert(neighbor);
      symmetric_neighborhood[neighbor].insert(node);
    }
  }
  return symmetric_neighborhood;
}

absl::btree_set<V> VerticesOf(const Graph& neighborhood) {
  absl::btree_set<V> nodes;
  for (const auto& [node, neighbors] : neighborhood) {
    for (const auto& neighbor : neighbors) {
      nodes.insert(node);
      nodes.insert(neighbor);
    }
  }
  return nodes;
}

absl::btree_set<V> CliqueFromMap(const Graph& neighborhood) {
  Graph symmetric_neighborhood = Symmetrize(neighborhood);
  absl::btree_set<V> nodes = VerticesOf(neighborhood);

  return MaximumClique<V, std::less<V>>(
             nodes,
             [&](const V& node) -> absl::btree_set<V> {
               return symmetric_neighborhood.at(node);
             })
      .value();
}

bool IsValidClique(const absl::btree_map<V, absl::btree_set<V>>& neighborhood,
                   const absl::btree_set<V>& clique) {
  Graph symmetric_neighborhood = Symmetrize(neighborhood);
  for (const V& x : clique) {
    for (const V& y : clique) {
      if ((x != y) && !symmetric_neighborhood.at(x).contains(y)) {
        return false;
      }
    }
  }
  return true;
}

TEST(MaximumCliqueTest, DisjointUnionOfEdges) {
  absl::btree_map<V, absl::btree_set<V>> graph;
  graph["x1"].insert("y1");
  graph["x2"].insert("y2");
  graph["x3"].insert("y3");
  graph["x4"].insert("y4");
  EXPECT_EQ(CliqueFromMap(graph).size(), 2);
  EXPECT_TRUE(IsValidClique(graph, CliqueFromMap(graph)));
}

TEST(MaximumCliqueTest, CompleteGraph4) {
  absl::btree_map<V, absl::btree_set<V>> graph;
  graph["a"].insert("b");
  graph["a"].insert("c");
  graph["a"].insert("d");
  graph["b"].insert("c");
  graph["b"].insert("d");
  graph["c"].insert("d");
  EXPECT_EQ(CliqueFromMap(graph).size(), 4);
  EXPECT_TRUE(IsValidClique(graph, CliqueFromMap(graph)));
}

TEST(MaximumCliqueTest, UnionOfCG4AndCG3) {
  absl::btree_map<V, absl::btree_set<V>> graph;
  graph["a"].insert("b");
  graph["a"].insert("c");
  graph["a"].insert("d");
  graph["b"].insert("c");
  graph["b"].insert("d");
  graph["c"].insert("d");
  graph["x"].insert("y");
  graph["x"].insert("z");
  graph["y"].insert("z");
  EXPECT_EQ(CliqueFromMap(graph).size(), 4);
  EXPECT_TRUE(IsValidClique(graph, CliqueFromMap(graph)));
}

TEST(MaximumCliqueTest, ConnectedUnionOfCG4AndCG3) {
  absl::btree_map<V, absl::btree_set<V>> graph;
  graph["a"].insert("b");
  graph["a"].insert("c");
  graph["a"].insert("d");
  graph["b"].insert("c");
  graph["b"].insert("d");
  graph["c"].insert("d");
  graph["x"].insert("y");
  graph["x"].insert("z");
  graph["y"].insert("z");
  graph["a"].insert("x");
  graph["b"].insert("y");
  graph["c"].insert("z");
  EXPECT_EQ(CliqueFromMap(graph).size(), 4);
  EXPECT_TRUE(IsValidClique(graph, CliqueFromMap(graph)));
}

TEST(MaximumCliqueTest, Big) {
  std::vector<V> nodes;
  for (int64_t i = 0; i < 100; ++i) {
    nodes.push_back(absl::StrFormat("%d", i));
  }

  absl::btree_map<V, absl::btree_set<V>> graph;

  std::mt19937_64 bit_gen;
  for (const V& x : nodes) {
    for (const V& y : nodes) {
      if (absl::Bernoulli(bit_gen, 0.5)) {
        graph[x].insert(y);
      }
    }
  }

  absl::btree_set<V> clique = CliqueFromMap(graph);
  EXPECT_EQ(clique.size(), 16);
  EXPECT_TRUE(IsValidClique(graph, clique));
}

}  // namespace
}  // namespace xls
