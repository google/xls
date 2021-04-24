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

#include "xls/data_structures/graph_contraction.h"

#include <cstdint>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/types/optional.h"
#include "absl/types/variant.h"

namespace xls {
namespace {

using ::testing::UnorderedElementsAre;

static absl::monostate MergeMonostate(absl::monostate x, absl::monostate y) {
  return absl::monostate();
}

static int32_t Sum(int32_t x, int32_t y) { return x + y; }

TEST(GraphContractionTest, NodeWeightsAreAdded) {
  // Test that node weights are added
  GraphContraction<char, int32_t, absl::monostate> gc;
  gc.AddVertex('a', 5);
  gc.AddVertex('b', 10);
  EXPECT_EQ(gc.Vertices(), absl::flat_hash_set<char>({'a', 'b'}));
  EXPECT_TRUE(gc.IdentifyVertices('a', 'b', Sum, MergeMonostate));
  EXPECT_EQ(gc.WeightOf('a'), 15);
  EXPECT_EQ(gc.WeightOf('b'), 15);
  EXPECT_EQ(gc.Vertices(), absl::flat_hash_set<char>({'a'}));
}

TEST(GraphContractionTest, AToBYieldsSelfLoop) {
  // Test that {a -> b} yields a self-loop when a and b are merged.
  GraphContraction<char, absl::monostate, int32_t> gc;
  gc.AddVertex('a', absl::monostate());
  gc.AddVertex('b', absl::monostate());
  gc.AddEdge('a', 'b', 20);
  EXPECT_TRUE(gc.IdentifyVertices('a', 'b', MergeMonostate, Sum));
  EXPECT_THAT(gc.Vertices(), UnorderedElementsAre('a'));
  EXPECT_THAT(gc.EdgesOutOf('a'), UnorderedElementsAre(std::pair('a', 20)));
  EXPECT_THAT(gc.EdgesInto('a'), UnorderedElementsAre(std::pair('a', 20)));
}

TEST(GraphContractionTest, AToBAndBToAYieldsSelfLoop) {
  // Test that {a -> b, b -> a} yields a self-loop with added weights
  // when a and b are merged.
  GraphContraction<char, absl::monostate, int32_t> gc;
  gc.AddVertex('a', absl::monostate());
  gc.AddVertex('b', absl::monostate());
  gc.AddEdge('a', 'b', 20);
  gc.AddEdge('b', 'a', 30);
  EXPECT_TRUE(gc.IdentifyVertices('a', 'b', MergeMonostate, Sum));
  EXPECT_THAT(gc.Vertices(), UnorderedElementsAre('a'));
  EXPECT_THAT(gc.EdgesOutOf('a'), UnorderedElementsAre(std::pair('a', 50)));
  EXPECT_THAT(gc.EdgesInto('a'), UnorderedElementsAre(std::pair('a', 50)));
}

TEST(GraphContractionTest, AllABCombinationsYieldSelfLoop) {
  // Test that {a -> a, b -> b, a -> b, b -> a} yields a self-loop with
  // added weights when a and b are merged.
  GraphContraction<char, absl::monostate, int32_t> gc;
  gc.AddVertex('a', absl::monostate());
  gc.AddVertex('b', absl::monostate());
  gc.AddEdge('a', 'a', 10);
  gc.AddEdge('b', 'b', 20);
  gc.AddEdge('a', 'b', 30);
  gc.AddEdge('b', 'a', 40);
  EXPECT_TRUE(gc.IdentifyVertices('a', 'b', MergeMonostate, Sum));
  EXPECT_THAT(gc.Vertices(), UnorderedElementsAre('a'));
  EXPECT_THAT(gc.EdgesOutOf('a'), UnorderedElementsAre(std::pair('a', 100)));
  EXPECT_THAT(gc.EdgesInto('a'), UnorderedElementsAre(std::pair('a', 100)));
}

TEST(GraphContractionTest, AToCAndBToCYieldsAToC) {
  // Test that {a -> c, b -> c} yields {a -> c} with added weights
  // when a and b are merged.
  GraphContraction<char, absl::monostate, int32_t> gc;
  gc.AddVertex('a', absl::monostate());
  gc.AddVertex('b', absl::monostate());
  gc.AddVertex('c', absl::monostate());
  gc.AddEdge('a', 'c', 30);
  gc.AddEdge('b', 'c', 40);
  EXPECT_TRUE(gc.IdentifyVertices('a', 'b', MergeMonostate, Sum));
  EXPECT_THAT(gc.Vertices(), UnorderedElementsAre('a', 'c'));
  EXPECT_THAT(gc.EdgesOutOf('a'), UnorderedElementsAre(std::pair('c', 70)));
  EXPECT_THAT(gc.EdgesInto('c'), UnorderedElementsAre(std::pair('a', 70)));
}

TEST(GraphContractionTest, CToAAndCToBYieldsCToA) {
  // Test that {c -> a, c -> b} yields {c -> a} with added weights
  // when a and b are merged.
  GraphContraction<char, absl::monostate, int32_t> gc;
  gc.AddVertex('a', absl::monostate());
  gc.AddVertex('b', absl::monostate());
  gc.AddVertex('c', absl::monostate());
  EXPECT_TRUE(gc.AddEdge('c', 'a', 30));
  EXPECT_TRUE(gc.AddEdge('c', 'b', 40));
  EXPECT_TRUE(gc.IdentifyVertices('a', 'b', MergeMonostate, Sum));
  EXPECT_THAT(gc.Vertices(), UnorderedElementsAre('a', 'c'));
  EXPECT_THAT(gc.EdgesOutOf('c'), UnorderedElementsAre(std::pair('a', 70)));
  EXPECT_THAT(gc.EdgesInto('a'), UnorderedElementsAre(std::pair('c', 70)));
}

TEST(GraphContractionTest, CToBYieldsCToA) {
  // Test that {c -> b} yields {c -> a} when a and b are merged.
  GraphContraction<char, absl::monostate, int32_t> gc;
  gc.AddVertex('a', absl::monostate());
  gc.AddVertex('b', absl::monostate());
  gc.AddVertex('c', absl::monostate());
  EXPECT_TRUE(gc.AddEdge('c', 'b', 40));
  EXPECT_TRUE(gc.IdentifyVertices('a', 'b', MergeMonostate, Sum));
  EXPECT_THAT(gc.Vertices(), UnorderedElementsAre('a', 'c'));
  EXPECT_THAT(gc.EdgesOutOf('c'), UnorderedElementsAre(std::pair('a', 40)));
  EXPECT_THAT(gc.EdgesInto('a'), UnorderedElementsAre(std::pair('c', 40)));
}

TEST(GraphContractionTest, Diamond) {
  // Test that {c -> a, c -> b, a -> d, b -> d} yields {c -> a -> d} when
  // a and b are merged.
  GraphContraction<char, absl::monostate, int32_t> gc;
  gc.AddVertex('a', absl::monostate());
  gc.AddVertex('b', absl::monostate());
  gc.AddVertex('c', absl::monostate());
  gc.AddVertex('d', absl::monostate());
  EXPECT_TRUE(gc.AddEdge('c', 'a', 10));
  EXPECT_TRUE(gc.AddEdge('c', 'b', 20));
  EXPECT_TRUE(gc.AddEdge('a', 'd', 30));
  EXPECT_TRUE(gc.AddEdge('b', 'd', 40));
  EXPECT_TRUE(gc.IdentifyVertices('a', 'b', MergeMonostate, Sum));
  EXPECT_THAT(gc.Vertices(), UnorderedElementsAre('c', 'a', 'd'));
  EXPECT_THAT(gc.EdgesOutOf('c'), UnorderedElementsAre(std::pair('a', 30)));
  EXPECT_THAT(gc.EdgesOutOf('a'), UnorderedElementsAre(std::pair('d', 70)));
  EXPECT_THAT(gc.EdgesInto('a'), UnorderedElementsAre(std::pair('c', 30)));
  EXPECT_THAT(gc.EdgesInto('d'), UnorderedElementsAre(std::pair('a', 70)));
}

TEST(GraphContractionTest, WeightOfWorks) {
  // Test that `WeightOf` works properly.
  GraphContraction<char, int32_t, int32_t> gc;
  gc.AddVertex('a', 5);
  gc.AddVertex('b', 7);
  gc.AddEdge('a', 'b', 10);
  EXPECT_EQ(gc.WeightOf('a'), 5);
  EXPECT_EQ(gc.WeightOf('b'), 7);
  EXPECT_EQ(gc.WeightOf('a', 'b'), 10);
}

TEST(GraphContractionTest, ContainsWorks) {
  // Test that `Contains` works properly.
  GraphContraction<char, absl::monostate, absl::monostate> gc;
  gc.AddVertex('a', absl::monostate());
  gc.AddVertex('b', absl::monostate());
  EXPECT_TRUE(gc.Contains('a'));
  EXPECT_TRUE(gc.Contains('b'));
  EXPECT_FALSE(gc.Contains('c'));
}

TEST(GraphContractionTest, RepresentativeOfWorks) {
  // Test that `RepresentativeOf` works properly.
  GraphContraction<char, absl::monostate, absl::monostate> gc;
  gc.AddVertex('a', absl::monostate());
  gc.AddVertex('b', absl::monostate());
  EXPECT_TRUE(gc.IdentifyVertices('a', 'b', MergeMonostate, MergeMonostate));
  EXPECT_EQ(gc.RepresentativeOf('a'), 'a');
  EXPECT_EQ(gc.RepresentativeOf('b'), 'a');
  EXPECT_EQ(gc.RepresentativeOf('c'), absl::nullopt);
}

TEST(GraphContractionTest, AddEdgeWithNonexistentVertices) {
  // Test that adding an edge with source/target that was not previously
  // inserted returns `false`.
  GraphContraction<char, absl::monostate, absl::monostate> gc;
  gc.AddVertex('a', absl::monostate());
  EXPECT_FALSE(gc.AddEdge('b', 'c', absl::monostate()));
  EXPECT_FALSE(gc.AddEdge('a', 'b', absl::monostate()));
  EXPECT_FALSE(gc.AddEdge('b', 'a', absl::monostate()));
}

TEST(GraphContractionTest, IdentifyNonExistentVertices) {
  // Test that identifying two vertices where one or more of them was not
  // previously inserted returns `false`.
  GraphContraction<char, absl::monostate, absl::monostate> gc;
  gc.AddVertex('a', absl::monostate());
  EXPECT_FALSE(gc.IdentifyVertices('b', 'c', MergeMonostate, MergeMonostate));
  EXPECT_FALSE(gc.IdentifyVertices('a', 'b', MergeMonostate, MergeMonostate));
  EXPECT_FALSE(gc.IdentifyVertices('b', 'a', MergeMonostate, MergeMonostate));
}

TEST(GraphContractionTest, SelfIdentification) {
  // Test that identify a node with itself returns `true`.
  GraphContraction<char, absl::monostate, absl::monostate> gc;
  gc.AddVertex('a', absl::monostate());
  EXPECT_TRUE(gc.IdentifyVertices('a', 'a', MergeMonostate, MergeMonostate));
}

TEST(GraphContractionTest, QueryNonexistentVerticesAndEdges) {
  // Test that `EdgesOutOf`, `EdgesInto`, and `WeightOf` returns the right
  // thing for nonexistent vertices and edges.
  GraphContraction<char, absl::monostate, absl::monostate> gc;
  gc.AddVertex('a', absl::monostate());
  gc.AddVertex('b', absl::monostate());
  gc.AddEdge('a', 'b', absl::monostate());
  EXPECT_TRUE(gc.EdgesOutOf('c').empty());
  EXPECT_TRUE(gc.EdgesInto('c').empty());
  EXPECT_FALSE(gc.WeightOf('c').has_value());
  EXPECT_FALSE(gc.WeightOf('a', 'c').has_value());
  EXPECT_FALSE(gc.WeightOf('c', 'a').has_value());
  EXPECT_FALSE(gc.WeightOf('c', 'd').has_value());
}

// TODO(taktoa): 2021-04-23 add a test that constructs a random graph and
// identifies all the vertices together in some random order, which should
// always result in a graph with a single vertex and possibly a self-edge
// (and the vertex/edge weights should be the sum of all vertex/edge weights).

}  // namespace
}  // namespace xls
