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

#include "xls/data_structures/union_find.h"

#include <memory>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/memory/memory.h"

namespace xls {
namespace {

TEST(UnionFindTest, BasicUsage) {
  constexpr int64_t kNodeCount = 5;
  struct Cluster {};
  std::vector<Cluster> clusters;
  std::vector<UnionFind<Cluster*>> cluster_for_node;

  clusters.resize(kNodeCount);
  cluster_for_node.resize(kNodeCount);
  for (int64_t i = 0; i < kNodeCount; ++i) {
    cluster_for_node[i].Get() = &clusters[i];
  }

  // A B C D E
  // \ | |/|
  // |\. . |
  // |     |
  // +-----+
  cluster_for_node[0].Merge(&cluster_for_node[1]);
  cluster_for_node[2].Merge(&cluster_for_node[3]);
  cluster_for_node[0].Merge(&cluster_for_node[3]);

  EXPECT_EQ(cluster_for_node[0].Get(), cluster_for_node[1].Get());
  EXPECT_EQ(cluster_for_node[2].Get(), cluster_for_node[3].Get());
  EXPECT_NE(cluster_for_node[3].Get(), cluster_for_node[4].Get());
}

}  // namespace
}  // namespace xls
