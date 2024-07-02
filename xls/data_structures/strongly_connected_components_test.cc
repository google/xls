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

#include "xls/data_structures/strongly_connected_components.h"

#include <cstdint>
#include <string>
#include <vector>

#include "absl/container/btree_set.h"
#include "absl/container/btree_set.h"
#include "absl/container/btree_set.h"
#include "absl/container/btree_set.h"
#include "absl/container/btree_map.h"
#include "absl/container/btree_set.h"
#include "absl/container/btree_set.h"
#include "absl/container/btree_map.h"
#include "absl/container/btree_set.h"
#include "absl/container/btree_set.h"
#include "absl/container/btree_map.h"
#include "absl/container/btree_set.h"
#include "absl/container/btree_set.h"
#include "absl/container/btree_map.h"
#include "absl/container/btree_set.h"
#include "absl/container/btree_set.h"
#include "absl/container/btree_map.h"
#include "absl/container/btree_set.h"
#include "absl/container/btree_set.h"
#include "absl/container/btree_map.h"
#include "absl/container/btree_set.h"
#include "absl/container/btree_set.h"
#include "absl/container/btree_map.h"
#include "absl/container/btree_set.h"
#include "absl/container/btree_set.h"
#include "absl/container/btree_map.h"
#include "absl/container/btree_set.h"
#include "absl/container/btree_set.h"
#include "gtest/gtest.h"

namespace xls {
namespace {

using V = std::string;

absl::btree_set<V> FlattenSCCs(const std::vector<absl::btree_set<V>>& sccs) {
  absl::btree_set<V> result;
  for (const absl::btree_set<V>& scc : sccs) {
    for (const V& vertex : scc) {
      result.insert(vertex);
    }
  }
  return result;
}

int64_t GraphSize(const absl::btree_map<V, absl::btree_set<V>>& graph) {
  absl::btree_set<V> vertices;
  for (const auto& [source, targets] : graph) {
    for (const V& target : targets) {
      vertices.insert(source);
      vertices.insert(target);
    }
  }
  return vertices.size();
}

TEST(StronglyConnectedComponentsTest, Cycle) {
  absl::btree_map<V, absl::btree_set<V>> graph;
  graph["a"].insert("b");
  graph["b"].insert("c");
  graph["c"].insert("d");
  graph["d"].insert("a");
  std::vector<absl::btree_set<V>> sccs = StronglyConnectedComponents<V>(graph);
  EXPECT_EQ(sccs.size(), 1);
  EXPECT_EQ(FlattenSCCs(sccs).size(), GraphSize(graph));
}

TEST(StronglyConnectedComponentsTest, CycleWithSpike) {
  absl::btree_map<V, absl::btree_set<V>> graph;
  graph["a"].insert("b");
  graph["b"].insert("c");
  graph["c"].insert("d");
  graph["d"].insert("a");
  graph["a"].insert("spike0");
  graph["a"].insert("spike1");
  graph["a"].insert("spike2");
  graph["a"].insert("spike3");
  std::vector<absl::btree_set<V>> sccs = StronglyConnectedComponents<V>(graph);
  EXPECT_EQ(sccs.size(), 5);
  EXPECT_EQ(FlattenSCCs(sccs).size(), GraphSize(graph));
}

TEST(StronglyConnectedComponentsTest, Figure8) {
  absl::btree_map<V, absl::btree_set<V>> graph;
  graph["a"].insert("b1");
  graph["b1"].insert("c1");
  graph["c1"].insert("d1");
  graph["d1"].insert("a");
  graph["a"].insert("b2");
  graph["b2"].insert("c2");
  graph["c2"].insert("d2");
  graph["d2"].insert("a");
  std::vector<absl::btree_set<V>> sccs = StronglyConnectedComponents<V>(graph);
  EXPECT_EQ(sccs.size(), 1);
  EXPECT_EQ(FlattenSCCs(sccs).size(), GraphSize(graph));
}

TEST(StronglyConnectedComponentsTest, DisjointUnionOfCycles) {
  absl::btree_map<V, absl::btree_set<V>> graph;
  graph["a1"].insert("b1");
  graph["b1"].insert("c1");
  graph["c1"].insert("d1");
  graph["d1"].insert("a1");
  graph["a2"].insert("b2");
  graph["b2"].insert("c2");
  graph["c2"].insert("d2");
  graph["d2"].insert("a2");
  std::vector<absl::btree_set<V>> sccs = StronglyConnectedComponents<V>(graph);
  EXPECT_EQ(sccs.size(), 2);
  EXPECT_EQ(FlattenSCCs(sccs).size(), GraphSize(graph));
}

TEST(StronglyConnectedComponentsTest, BarBell) {
  absl::btree_map<V, absl::btree_set<V>> graph;
  graph["a1"].insert("b1");
  graph["b1"].insert("c1");
  graph["c1"].insert("d1");
  graph["d1"].insert("a1");
  graph["a2"].insert("b2");
  graph["b2"].insert("c2");
  graph["c2"].insert("d2");
  graph["d2"].insert("a2");
  graph["a1"].insert("a2");  // connect the two cycles with a directed edge
  std::vector<absl::btree_set<V>> sccs = StronglyConnectedComponents<V>(graph);
  EXPECT_EQ(sccs.size(), 2);
  EXPECT_EQ(FlattenSCCs(sccs).size(), GraphSize(graph));
}

TEST(StronglyConnectedComponentsTest, Chain) {
  absl::btree_map<V, absl::btree_set<V>> graph;
  graph["a"].insert("b");
  graph["b"].insert("c");
  graph["c"].insert("d");
  std::vector<absl::btree_set<V>> sccs = StronglyConnectedComponents<V>(graph);
  EXPECT_EQ(sccs.size(), GraphSize(graph));
  EXPECT_EQ(FlattenSCCs(sccs).size(), GraphSize(graph));
}

TEST(StronglyConnectedComponentsTest, Tree) {
  absl::btree_map<V, absl::btree_set<V>> graph;
  graph["root"].insert("l");
  graph["root"].insert("r");
  graph["l"].insert("ll");
  graph["l"].insert("lr");
  graph["lr"].insert("lrl");
  graph["lr"].insert("lrr");
  graph["r"].insert("rl");
  graph["r"].insert("rr");
  std::vector<absl::btree_set<V>> sccs = StronglyConnectedComponents<V>(graph);
  EXPECT_EQ(sccs.size(), GraphSize(graph));
  EXPECT_EQ(FlattenSCCs(sccs).size(), GraphSize(graph));
}

}  // namespace
}  // namespace xls
