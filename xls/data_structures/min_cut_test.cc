// Copyright 2020 The XLS Authors
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

#include "xls/data_structures/min_cut.h"

#include <cstdint>
#include <limits>
#include <random>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/random/distributions.h"
#include "absl/random/mocking_bit_gen.h"
#include "absl/random/random.h"
#include "absl/random/uniform_int_distribution.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "xls/common/random_util.h"

namespace xls {
namespace min_cut {
namespace {

using ::testing::UnorderedElementsAre;

TEST(MinCutTest, TrivialGraph) {
  //    s
  //    |
  // 42 |
  //    |
  //    t
  Graph graph;
  NodeId s = graph.AddNode("s");
  NodeId t = graph.AddNode("t");
  graph.AddEdge(s, t, 42);
  GraphCut min_cut = MinCutBetweenNodes(graph, s, t);
  EXPECT_EQ(min_cut.weight, 42);
  EXPECT_THAT(min_cut.source_partition, UnorderedElementsAre(s));
  EXPECT_THAT(min_cut.sink_partition, UnorderedElementsAre(t));
}

TEST(MinCutTest, TrivialUnconnectedGraph) {
  Graph graph;
  NodeId s = graph.AddNode("s");
  NodeId t = graph.AddNode("t");
  GraphCut min_cut = MinCutBetweenNodes(graph, s, t);
  EXPECT_EQ(min_cut.weight, 0);
  EXPECT_THAT(min_cut.source_partition, UnorderedElementsAre(s));
  EXPECT_THAT(min_cut.sink_partition, UnorderedElementsAre(t));
}

TEST(MinCutTest, DiamondGraph) {
  //       a
  //      / \
  // 100 /   \ 1
  //    /     \
  //   b       c
  //    \     /
  //  42 \   / 1234
  //      \ /
  //       d
  Graph graph;
  auto a = graph.AddNode("a");
  auto b = graph.AddNode("b");
  auto c = graph.AddNode("c");
  auto d = graph.AddNode("d");
  graph.AddEdge(a, b, 100);
  graph.AddEdge(a, c, 1);
  graph.AddEdge(b, d, 42);
  graph.AddEdge(c, d, 1234);
  GraphCut min_cut = MinCutBetweenNodes(graph, a, d);
  EXPECT_EQ(min_cut.weight, 43);
  EXPECT_THAT(min_cut.source_partition, UnorderedElementsAre(a, b));
  EXPECT_THAT(min_cut.sink_partition, UnorderedElementsAre(c, d));
}

TEST(MinCutTest, DiamondGraphWithReplicatedEdges) {
  //            a
  //           / \
  //      100 /   \ 1 x (2)
  //         /     \
  //        b       c
  //         \     /
  // 42 x (3) \   / 1234 x (3)
  //           \ /
  //            d
  Graph graph;
  auto a = graph.AddNode("a");
  auto b = graph.AddNode("b");
  auto c = graph.AddNode("c");
  auto d = graph.AddNode("d");
  graph.AddEdge(a, b, 100);

  graph.AddEdge(a, c, 1);
  graph.AddEdge(a, c, 1);

  graph.AddEdge(b, d, 42);
  graph.AddEdge(b, d, 42);
  graph.AddEdge(b, d, 42);

  graph.AddEdge(c, d, 1234);
  graph.AddEdge(c, d, 1234);
  graph.AddEdge(c, d, 1234);
  graph.AddEdge(c, d, 1234);

  GraphCut min_cut = MinCutBetweenNodes(graph, a, d);
  EXPECT_EQ(min_cut.weight, 102);
  EXPECT_THAT(min_cut.source_partition, UnorderedElementsAre(a));
  EXPECT_THAT(min_cut.sink_partition, UnorderedElementsAre(b, c, d));
}

TEST(MinCutTest, ComplexGraph) {
  //       a
  //      / \
  // 100 /   \ 16
  //    /     \
  //   b       c
  //    \   8 /|
  //  42 \   / | 123
  //      \ /  |
  //       d   e
  //      / \  |
  //     /   \ | 4
  //    |   6 \|
  // 33 |      f
  //    |     /
  //     \   / 12
  //      \ /
  //       g
  Graph graph;
  auto a = graph.AddNode("a");
  auto b = graph.AddNode("b");
  auto c = graph.AddNode("c");
  auto d = graph.AddNode("d");
  auto e = graph.AddNode("e");
  auto f = graph.AddNode("f");
  auto g = graph.AddNode("g");
  graph.AddEdge(a, b, 100);
  graph.AddEdge(a, c, 16);
  graph.AddEdge(b, d, 42);
  graph.AddEdge(c, d, 8);
  graph.AddEdge(c, e, 123);
  graph.AddEdge(d, g, 33);
  graph.AddEdge(d, f, 6);
  graph.AddEdge(e, f, 4);
  graph.AddEdge(f, g, 12);
  GraphCut min_cut = MinCutBetweenNodes(graph, a, g);
  EXPECT_EQ(min_cut.weight, 43);
  EXPECT_THAT(min_cut.source_partition, UnorderedElementsAre(a, b, c, d, e));
  EXPECT_THAT(min_cut.sink_partition, UnorderedElementsAre(f, g));
}

// Returns the cost of the cut defined by the partitions source_set and
// sink_set of the given graph.
int64_t CutCost(const Graph& graph, absl::flat_hash_set<NodeId> source_set,
                absl::flat_hash_set<NodeId> sink_set) {
  int64_t cost = 0;
  for (NodeId node : source_set) {
    for (EdgeId edge_id : graph.successors(node)) {
      if (sink_set.contains(graph.edge(edge_id).to)) {
        // To avoid overflow handle maximum weight edges specially.
        if (graph.edge(edge_id).weight == std::numeric_limits<int64_t>::max()) {
          return std::numeric_limits<int64_t>::max();
        }
        cost += graph.edge(edge_id).weight;
      }
    }
  }
  return cost;
}

// Returns a large graph consisting of layers of nodes with randomly generated
// edges with a single source and sink node on top and bottom. If acyclic, all
// edges are forward-directed (from lower layers to higher layers). Sets
// 'source' and 'sink' to the NodeId of the source and sink nodes in the graph.
Graph MakeLargeGraph(bool acyclic, NodeId* source, NodeId* sink,
                     int64_t layer_count, int64_t nodes_in_layer) {
  const int64_t kMaxFanOut = 10;
  Graph graph;
  *source = graph.AddNode("source");
  *sink = graph.AddNode("sink");
  std::vector<std::vector<NodeId>> layers(layer_count);
  for (int64_t i = 0; i < layer_count; ++i) {
    for (int64_t j = 0; j < nodes_in_layer; ++j) {
      layers[i].push_back(graph.AddNode(absl::StrFormat("node_%d_%d", i, j)));
    }
  }
  for (NodeId node : layers[0]) {
    graph.AddEdge(*source, node, std::numeric_limits<int64_t>::max());
  }
  std::mt19937_64 bit_gen;
  for (int64_t i = 0; i < layer_count - 1; ++i) {
    // If graph is acyclic then edges can only extend to nodes in later
    // layers. If cyclic, then edges can extend to nodes in any layer.
    absl::Span<NodeId> from_layer = absl::MakeSpan(layers[i]);
    absl::Span<std::vector<NodeId>> to_layers =
        absl::MakeSpan(layers).subspan(acyclic ? i + 1 : 0);

    for (NodeId from : from_layer) {
      int64_t fanout =
          absl::Uniform(absl::IntervalClosed, bit_gen, 0, kMaxFanOut);
      for (int64_t j = 0; j < fanout; ++j) {
        std::vector<NodeId> to_layer = RandomChoice(to_layers, bit_gen);
        NodeId to = RandomChoice(to_layer, bit_gen);
        int64_t weight = absl::Uniform(absl::IntervalClosed, bit_gen, 0, 10);
        graph.AddEdge(from, to, weight);
      }
    }
  }
  for (NodeId node : layers[layer_count - 1]) {
    graph.AddEdge(node, *sink, std::numeric_limits<int64_t>::max());
  }

  return graph;
}

TEST(MinCutTest, LargeDirectedGraphs) {
  // Construct some random directed graphs and verify that the min cut is at
  // least at a local minimum. Verifying it actually is the min cut is hard
  // without reimplementing the algorithm.
  for (bool acyclic : {false, true}) {
    for (int64_t layer_count = 5; layer_count < 20; layer_count += 2) {
      for (int64_t nodes_in_layer = 5; nodes_in_layer < 20;
           nodes_in_layer += 2) {
        NodeId source;
        NodeId sink;
        Graph graph = MakeLargeGraph(acyclic, &source, &sink, layer_count,
                                     nodes_in_layer);
        // It's hard to verify that this is *actually* the min cut, but do basic
        // validation of the partition.
        GraphCut min_cut = MinCutBetweenNodes(graph, source, sink);
        EXPECT_EQ(
            min_cut.source_partition.size() + min_cut.sink_partition.size(),
            graph.node_count());

        VLOG(3) << graph.ToString();
        VLOG(3) << min_cut.ToString(graph);

        // Verify that moving a node from source partition to sink partition or
        // vice versa does not decrease the cut cost.
        absl::flat_hash_set<NodeId> source_set(min_cut.source_partition.begin(),
                                               min_cut.source_partition.end());
        absl::flat_hash_set<NodeId> sink_set(min_cut.sink_partition.begin(),
                                             min_cut.sink_partition.end());
        EXPECT_EQ(min_cut.weight, CutCost(graph, source_set, sink_set));
        for (NodeId node : min_cut.source_partition) {
          source_set.erase(node);
          sink_set.insert(node);

          EXPECT_LE(min_cut.weight, CutCost(graph, source_set, sink_set))
              << "Moving node from source to sink reduced cost of the cut: "
              << graph.name(node);

          source_set.insert(node);
          sink_set.erase(node);
        }
        for (NodeId node : min_cut.sink_partition) {
          sink_set.erase(node);
          source_set.insert(node);

          EXPECT_LE(min_cut.weight, CutCost(graph, source_set, sink_set))
              << "Moving node from sink to source reduced cost of the cut: "
              << graph.name(node);

          sink_set.insert(node);
          source_set.erase(node);
        }
      }
    }
  }
}

TEST(MinCutTest, MaxFlowToMinCutTraversalTest) {
  // Test a fix for b/155115565 where the residual graph was not properly
  // traversed to identify the partitions after max flow was computed.
  Graph graph;
  auto source = graph.AddNode("source");
  auto a = graph.AddNode("a");
  auto b = graph.AddNode("b");
  auto c = graph.AddNode("c");
  auto d = graph.AddNode("d");
  auto e = graph.AddNode("e");
  auto f = graph.AddNode("f");
  auto g = graph.AddNode("g");
  auto sink = graph.AddNode("sink");
  graph.AddEdge(source, a, std::numeric_limits<int64_t>::max());
  graph.AddEdge(source, b, std::numeric_limits<int64_t>::max());

  graph.AddEdge(a, c, 7);
  graph.AddEdge(a, f, 1);
  graph.AddEdge(b, d, 13);
  graph.AddEdge(c, e, 9);
  graph.AddEdge(d, e, 7);
  graph.AddEdge(e, g, 10);

  graph.AddEdge(f, sink, std::numeric_limits<int64_t>::max());
  graph.AddEdge(g, sink, std::numeric_limits<int64_t>::max());

  GraphCut min_cut = MinCutBetweenNodes(graph, source, sink);
  EXPECT_EQ(min_cut.weight, 11);
}

TEST(MinCutTest, ResidualGraphTraversalTest) {
  // Test a graph which requires traversing a backeards edge in the residual
  // graph in an augmented path. The first augmenting path is
  // source->a->d->sink. The next augmenting path is
  // source->b->c->d->a->e->f->sink. Note the traversal along the backwards
  // residual edge d->a.
  //
  //     source
  //      / \
  //   1 /   \ 1
  //    /     \
  //   a       b
  //   |\      |
  // 1 | \     | 1
  //   |  \ 1  |
  //   e   \   c
  //   |    \  |
  // 1 |     \ | 1
  //   |      \|
  //   f       d
  //    \     /
  //   1 \   / 1
  //      \ /
  //      sink
  //
  Graph graph;
  auto source = graph.AddNode("source");
  auto a = graph.AddNode("a");
  auto b = graph.AddNode("b");
  auto c = graph.AddNode("c");
  auto d = graph.AddNode("d");
  auto e = graph.AddNode("e");
  auto f = graph.AddNode("f");
  auto sink = graph.AddNode("sink");

  graph.AddEdge(source, a, 1);
  graph.AddEdge(source, b, 1);
  graph.AddEdge(a, d, 1);
  graph.AddEdge(a, e, 1);
  graph.AddEdge(b, c, 1);
  graph.AddEdge(c, d, 1);
  graph.AddEdge(d, sink, 1);
  graph.AddEdge(e, f, 1);
  graph.AddEdge(f, sink, 1);

  GraphCut min_cut = MinCutBetweenNodes(graph, source, sink);
  EXPECT_EQ(min_cut.weight, 2);
}

}  // namespace
}  // namespace min_cut
}  // namespace xls
