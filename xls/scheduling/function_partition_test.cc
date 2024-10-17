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

#include "xls/scheduling/function_partition.h"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "xls/common/status/matchers.h"
#include "xls/examples/sample_packages.h"
#include "xls/ir/bits.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/nodes.h"
#include "xls/ir/package.h"
#include "xls/ir/topo_sort.h"

namespace xls {
namespace sched {
namespace {

using ::testing::UnorderedElementsAre;

class FunctionPartitionTest : public IrTestBase {
 protected:
  // Returns a vector of all the nodes in the function.
  std::vector<Node*> AllNodes(Function* f) {
    return std::vector<Node*>(f->nodes().begin(), f->nodes().end());
  }

  // Returns the cost of the given partitioning (sum of bit widths of nodes live
  // across the partition boundary).
  int64_t PartitionCost(absl::Span<Node* const> first_partition,
                        absl::Span<Node* const> second_partition) {
    absl::flat_hash_set<Node*> first_set(first_partition.begin(),
                                         first_partition.end());
    absl::flat_hash_set<Node*> second_set(second_partition.begin(),
                                          second_partition.end());
    absl::flat_hash_set<Node*> cut_nodes;

    for (Node* node : first_partition) {
      for (Node* user : node->users()) {
        if (!first_set.contains(user)) {
          cut_nodes.insert(user);
        }
      }
    }
    for (Node* node : second_partition) {
      for (Node* operand : node->operands()) {
        if (!second_set.contains(operand)) {
          cut_nodes.insert(operand);
        }
      }
    }

    int64_t cost = 0;
    for (Node* node : cut_nodes) {
      cost += node->GetType()->GetFlatBitCount();
    }
    return cost;
  }
};

TEST_F(FunctionPartitionTest, ParameterOnlyCut) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  auto x = fb.Param("x", p->GetBitsType(32));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());

  {
    auto partition = MinCostFunctionPartition(f, {x.node()});
    EXPECT_THAT(partition.first, UnorderedElementsAre(x.node()));
    EXPECT_THAT(partition.second, UnorderedElementsAre());
  }

  {
    // If the partitionable set is empty, both partitions should be empty.
    auto partition = MinCostFunctionPartition(f, {});
    EXPECT_THAT(partition.first, UnorderedElementsAre());
    EXPECT_THAT(partition.second, UnorderedElementsAre());
  }
}

TEST_F(FunctionPartitionTest, LinearGraphCut) {
  // Partition a simple linear graph.
  //
  //      x
  //      |
  //  bit-slice
  //      |
  //   zero-extend
  //      |
  //    negate
  //
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  auto x = fb.Param("x", p->GetBitsType(32));
  auto bit_slice = fb.BitSlice(x, /*start=*/0, /*width=*/16);
  auto zext = fb.ZeroExtend(bit_slice, /*new_bit_count=*/128);
  fb.Negate(zext);
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());

  {
    auto partition =
        MinCostFunctionPartition(f, {x.node(), bit_slice.node(), zext.node()});
    EXPECT_THAT(partition.first,
                UnorderedElementsAre(x.node(), bit_slice.node()));
    EXPECT_THAT(partition.second, UnorderedElementsAre(zext.node()));
  }

  {
    // Partitioning just the bit-slice should result in the cut after the
    // bit-slice because the output of the bit-slice is narrower than its input.
    auto partition = MinCostFunctionPartition(f, {bit_slice.node()});
    EXPECT_THAT(partition.first, UnorderedElementsAre(bit_slice.node()));
    EXPECT_THAT(partition.second, UnorderedElementsAre());
  }
}

TEST_F(FunctionPartitionTest, DisconnectedGraph) {
  // Partition a disconnected graph:
  //
  //      x
  //      |
  //     not
  //
  //    literal
  //      |
  //     not
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  auto x = fb.Param("x", p->GetBitsType(32));
  auto not_x = fb.Not(x);
  auto literal = fb.Literal(UBits(123, 8));
  auto not_literal = fb.Not(literal);
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());

  auto partition = MinCostFunctionPartition(f, AllNodes(f));
  EXPECT_THAT(partition.first, UnorderedElementsAre(x.node(), not_x.node()));
  EXPECT_THAT(partition.second,
              UnorderedElementsAre(literal.node(), not_literal.node()));
}

TEST_F(FunctionPartitionTest, BenchmarkTest) {
  // Compute the minimum cost partition of each benchmark and validate the
  // results.
  XLS_ASSERT_OK_AND_ASSIGN(std::vector<std::string> benchmark_names,
                           sample_packages::GetBenchmarkNames());
  for (const std::string& benchmark_name : benchmark_names) {
    XLS_ASSERT_OK_AND_ASSIGN(
        std::unique_ptr<Package> p,
        sample_packages::GetBenchmark(benchmark_name, /*optimized=*/true));

    absl::StatusOr<Function*> f_status = p->GetTopAsFunction();
    if (!f_status.ok()) {
      // Skip packages which need the entry to be specified explicitly.
      continue;
    }
    Function* f = f_status.value();
    auto topo_sort_it = TopoSort(f);
    std::vector<Node*> topo_sort(topo_sort_it.begin(), topo_sort_it.end());

    // Create a subspan of the given span with the given start and end indices
    // (inclusive).
    auto make_span = [](absl::Span<Node* const> v, int64_t start, int64_t end) {
      return absl::MakeConstSpan(v.data() + start, v.data() + end);
    };

    // For each benchmark try partitioning two different sets of nodes:
    //
    // (1) all the nodes in the function
    //
    // (2) the middle half of the nodes in the topological sort of the function
    //     (nodes at indices between 25% and 75% of the total number of nodes).
    //
    // Case (2) can result in more interesting partition problem because
    // typically when partitioning all the nodes the minimum cost partition is
    // right after the parameters or before the return value.
    for (auto start_end :
         {std::make_pair(int64_t{0}, f->node_count() - 1),
          std::make_pair(f->node_count() / 4, f->node_count() * 3 / 4)}) {
      // The start/end indices of the nodes to partition in the topological
      // sort.
      int64_t start = start_end.first;
      int64_t end = start_end.second;
      auto nodes_to_partition = make_span(topo_sort, start, end);

      auto partition = MinCostFunctionPartition(f, nodes_to_partition);

      EXPECT_EQ(partition.first.size() + partition.second.size(),
                nodes_to_partition.size());

      // No params should be in the second partition.
      EXPECT_TRUE(std::all_of(partition.second.begin(), partition.second.end(),
                              [](Node* n) { return !n->Is<Param>(); }));

      absl::flat_hash_set<Node*> first_partition(partition.first.begin(),
                                                 partition.first.end());
      absl::flat_hash_set<Node*> second_partition(partition.second.begin(),
                                                  partition.second.end());

      // There should be no edge from a node in the second partition to a node
      // in the first partition.
      for (Node* node : partition.second) {
        for (Node* user : node->users()) {
          EXPECT_FALSE(first_partition.contains(user)) << absl::StreamFormat(
              "%s : %s->%s", benchmark_name, node->GetName(), user->GetName());
        }
      }

      // Try other arbitrary partitions and verify these partitions are greater
      // than or equal to the cost of the minimum cost partition. The arbitrary
      // partitions are generated by splitting a topological sort in two
      // pieces. The split of the topological sort is done such that the
      // parameters are always in the first partition and the return value in
      // the second.
      int64_t last_param_index = 0;
      int64_t return_value_index = 0;
      for (int64_t i = 0; i < nodes_to_partition.size(); ++i) {
        Node* node = nodes_to_partition[i];
        if (node->Is<Param>()) {
          last_param_index = i;
        }
        if (node == f->return_value()) {
          return_value_index = i;
        }
      }
      for (int64_t i = last_param_index; i < return_value_index; ++i) {
        EXPECT_LE(PartitionCost(partition.first, partition.second),
                  PartitionCost(make_span(nodes_to_partition, 0, i),
                                make_span(nodes_to_partition, i + 1,
                                          nodes_to_partition.size())));
      }
    }
  }
}

}  // namespace
}  // namespace sched
}  // namespace xls
