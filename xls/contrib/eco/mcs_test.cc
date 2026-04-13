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

#include "xls/contrib/eco/mcs.h"

#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "gtest/gtest.h"

namespace mcs {
namespace {

using internal::CandidateClass;
using internal::CandidatePartition;
using internal::EquivalenceInfo;
using internal::ExclusionSet;

XLSGraph MakeGraph(const std::vector<std::string>& labels,
                   const std::vector<std::tuple<int, int, int>>& edges) {
  XLSGraph graph;
  for (int i = 0; i < static_cast<int>(labels.size()); ++i) {
    graph.add_node(
        XLSNode("n" + std::to_string(i), "label=" + labels[i]));
  }
  for (const auto& [from, to, index] : edges) {
    graph.add_edge(XLSEdge(from, to, "", index));
  }
  graph.populate_node_signatures();
  return graph;
}

absl::flat_hash_map<int, int> ToMap(const State& mapping) {
  absl::flat_hash_map<int, int> result;
  for (const auto& [u, v] : mapping) {
    result[u] = v;
  }
  return result;
}

TEST(McsTest, SolveMCSUsesExcludeBranchWhenBestMatchOmitsBranchNode) {
  XLSGraph query = MakeGraph(
      {"A", "A", "B", "B", "C", "C"},
      {
          {0, 2, 0},
          {1, 3, 0},
          {2, 4, 1},
          {3, 5, 2},
      });
  XLSGraph target = MakeGraph(
      {"A", "B", "C"},
      {
          {0, 1, 0},
          {1, 2, 2},
      });

  MCSResult result = SolveMCS(query, target);
  auto mapping = ToMap(result.mapping);

  EXPECT_EQ(result.size, 3);
  EXPECT_EQ(mapping[1], 0);
  EXPECT_EQ(mapping[3], 1);
  EXPECT_EQ(mapping[5], 2);
  EXPECT_FALSE(mapping.contains(0));
  EXPECT_FALSE(mapping.contains(2));
  EXPECT_FALSE(mapping.contains(4));
}

TEST(McsTest, ExclusionReductionPrunesEquivalentQueryNodes) {
  XLSGraph query = MakeGraph({"A", "A", "B"}, {});
  EquivalenceInfo equivalence =
      internal::ComputeStructuralEquivalenceForTesting(query);

  ExclusionSet excluded;
  excluded[0].insert(7);

  EXPECT_EQ(equivalence.class_id[0], equivalence.class_id[1]);
  EXPECT_TRUE(
      internal::ShouldPruneByExclusionForTesting(1, 7, excluded, equivalence));
  EXPECT_FALSE(
      internal::ShouldPruneByExclusionForTesting(1, 8, excluded, equivalence));
}

TEST(McsTest, SecondGroupRemovesEquivalentQueryNodes) {
  XLSGraph query = MakeGraph({"A", "A", "B"}, {});
  EquivalenceInfo equivalence =
      internal::ComputeStructuralEquivalenceForTesting(query);

  CandidatePartition candidates = {
      CandidateClass{{0, 1}, {0, 1}},
      CandidateClass{{2}, {2}},
  };

  CandidatePartition reduced = internal::RemoveEquivalentQueryNodesForTesting(
      candidates, 0, equivalence);

  ASSERT_EQ(reduced.size(), 1);
  EXPECT_EQ(reduced[0].query_nodes, std::vector<int>({2}));
  EXPECT_EQ(reduced[0].target_nodes, std::vector<int>({2}));
}

TEST(McsTest, RefineCandidatePartitionRespectsDirectedRelationProfiles) {
  XLSGraph query = MakeGraph(
      {"A", "B", "B"},
      {
          {0, 1, 0},
          {2, 0, 0},
      });
  XLSGraph target = MakeGraph(
      {"A", "B", "B"},
      {
          {0, 1, 0},
          {2, 0, 0},
      });

  CandidatePartition candidates = {
      CandidateClass{{1, 2}, {1, 2}},
  };

  CandidatePartition refined = internal::RefineCandidatePartitionForTesting(
      candidates, 0, 0, query, target);

  ASSERT_EQ(refined.size(), 2);
  EXPECT_EQ(refined[0].query_nodes, std::vector<int>({1}));
  EXPECT_EQ(refined[0].target_nodes, std::vector<int>({1}));
  EXPECT_EQ(refined[1].query_nodes, std::vector<int>({2}));
  EXPECT_EQ(refined[1].target_nodes, std::vector<int>({2}));
}

TEST(McsTest, RefineCandidatePartitionMatchesEdgeIndices) {
  XLSGraph query = MakeGraph(
      {"A", "B", "B"},
      {
          {0, 1, 1},
          {0, 2, 2},
      });
  XLSGraph target = MakeGraph(
      {"A", "B"},
      {
          {0, 1, 1},
      });

  CandidatePartition candidates = {
      CandidateClass{{1, 2}, {1}},
  };

  CandidatePartition refined = internal::RefineCandidatePartitionForTesting(
      candidates, 0, 0, query, target);

  ASSERT_EQ(refined.size(), 1);
  EXPECT_EQ(refined[0].query_nodes, std::vector<int>({1}));
  EXPECT_EQ(refined[0].target_nodes, std::vector<int>({1}));
}

TEST(McsTest, MaximalityReductionRecognizesUniformClasses) {
  XLSGraph query = MakeGraph({"A", "A"}, {});
  XLSGraph target = MakeGraph({"A", "A"}, {});

  CandidatePartition candidates = {
      CandidateClass{{0, 1}, {0, 1}},
  };

  EXPECT_TRUE(
      internal::SatisfiesMaximalityReductionForTesting(candidates, 0, 0, query,
                                                       target));
}

TEST(McsTest, MaximalityReductionIgnoresBranchVertexSelfRelation) {
  XLSGraph query = MakeGraph(
      {"A", "A"},
      {
          {0, 1, 0},
          {1, 0, 0},
      });
  XLSGraph target = MakeGraph(
      {"A", "A"},
      {
          {0, 1, 0},
          {1, 0, 0},
      });

  CandidatePartition candidates = {
      CandidateClass{{0, 1}, {0, 1}},
  };

  EXPECT_TRUE(
      internal::SatisfiesMaximalityReductionForTesting(candidates, 0, 0, query,
                                                       target));
}

TEST(McsTest, UpperBoundUsesExclusionSetAndEquivalenceClasses) {
  XLSGraph query = MakeGraph({"A", "A", "B"}, {});
  EquivalenceInfo equivalence =
      internal::ComputeStructuralEquivalenceForTesting(query);

  CandidatePartition candidates = {
      CandidateClass{{0, 1}, {0, 1}},
      CandidateClass{{2}, {2}},
  };
  ExclusionSet excluded;
  excluded[0].insert(0);

  EXPECT_EQ(internal::ComputeUpperBoundForTesting(
                State{}, candidates, excluded, equivalence),
            2);
}

}  // namespace
}  // namespace mcs
