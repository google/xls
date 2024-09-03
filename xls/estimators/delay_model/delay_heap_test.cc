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

#include "xls/estimators/delay_model/delay_heap.h"

#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xls/common/status/matchers.h"
#include "xls/estimators/delay_model/analyze_critical_path.h"
#include "xls/estimators/delay_model/delay_estimator.h"
#include "xls/estimators/delay_model/delay_estimators.h"
#include "xls/examples/sample_packages.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/node.h"
#include "xls/ir/op.h"
#include "xls/ir/package.h"
#include "xls/ir/topo_sort.h"

namespace xls {
namespace sched {
namespace {

using status_testing::IsOkAndHolds;
using ::testing::ElementsAre;

class DelayHeapTest : public IrTestBase {
 protected:
  const DelayEstimator* delay_estimator_ = GetDelayEstimator("unit").value();
};

TEST_F(DelayHeapTest, TrivialFunction) {
  // Sequentially add then remove nodes of simple XLS functionto a delay heap
  // and verify properties of the heap at each step.
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  auto x = fb.Param("x", p->GetBitsType(32));
  auto neg = fb.Negate(x);
  auto rev = fb.Reverse(neg);
  XLS_ASSERT_OK(fb.Build().status());

  DelayHeap heap(Direction::kGrowsTowardUsers, *delay_estimator_);
  EXPECT_EQ(heap.CriticalPathDelay(), 0);
  EXPECT_EQ(heap.size(), 0);
  EXPECT_THAT(heap.CriticalPathDelayAfterAdding(x.node()), IsOkAndHolds(0));

  XLS_ASSERT_OK(heap.Add(x.node()));

  EXPECT_EQ(heap.CriticalPathDelay(), 0);
  EXPECT_EQ(heap.size(), 1);
  EXPECT_THAT(heap.frontier(), ElementsAre(x.node()));
  EXPECT_TRUE(heap.contains(x.node()));
  EXPECT_FALSE(heap.contains(neg.node()));
  EXPECT_FALSE(heap.contains(rev.node()));
  EXPECT_THAT(heap.CriticalPathDelayAfterAdding(neg.node()), IsOkAndHolds(1));

  XLS_ASSERT_OK(heap.Add(neg.node()));

  EXPECT_EQ(heap.CriticalPathDelay(), 1);
  EXPECT_EQ(heap.size(), 2);
  EXPECT_THAT(heap.frontier(), ElementsAre(neg.node()));
  EXPECT_TRUE(heap.contains(x.node()));
  EXPECT_TRUE(heap.contains(neg.node()));
  EXPECT_FALSE(heap.contains(rev.node()));
  EXPECT_THAT(heap.CriticalPathDelayAfterAdding(rev.node()), IsOkAndHolds(2));

  XLS_ASSERT_OK(heap.Add(rev.node()));

  EXPECT_EQ(heap.CriticalPathDelay(), 2);
  EXPECT_EQ(heap.size(), 3);
  EXPECT_THAT(heap.frontier(), ElementsAre(rev.node()));
  EXPECT_TRUE(heap.contains(x.node()));
  EXPECT_TRUE(heap.contains(neg.node()));
  EXPECT_TRUE(heap.contains(rev.node()));
  EXPECT_EQ(heap.CriticalPathDelayAfterRemoving(rev.node()), 1);

  heap.Remove(rev.node());

  EXPECT_EQ(heap.CriticalPathDelay(), 1);
  EXPECT_EQ(heap.size(), 2);
  EXPECT_THAT(heap.frontier(), ElementsAre(neg.node()));
  EXPECT_TRUE(heap.contains(x.node()));
  EXPECT_TRUE(heap.contains(neg.node()));
  EXPECT_FALSE(heap.contains(rev.node()));
  EXPECT_EQ(heap.CriticalPathDelayAfterRemoving(neg.node()), 0);

  heap.Remove(neg.node());

  EXPECT_EQ(heap.CriticalPathDelay(), 0);
  EXPECT_EQ(heap.size(), 1);
  EXPECT_THAT(heap.frontier(), ElementsAre(x.node()));
  EXPECT_TRUE(heap.contains(x.node()));
  EXPECT_FALSE(heap.contains(neg.node()));
  EXPECT_FALSE(heap.contains(rev.node()));
  EXPECT_EQ(heap.CriticalPathDelayAfterRemoving(x.node()), 0);

  heap.Remove(x.node());

  EXPECT_EQ(heap.CriticalPathDelay(), 0);
  EXPECT_EQ(heap.size(), 0);
  EXPECT_TRUE(heap.frontier().empty());
  EXPECT_FALSE(heap.contains(x.node()));
  EXPECT_FALSE(heap.contains(neg.node()));
  EXPECT_FALSE(heap.contains(rev.node()));
}

// The frontier should following properties:
// (1) All nodes on the frontier should be contained in the heap.
// (2) Sorted in decending order by critical path
// (3) No user (operand) of a node in the frontier should be in the heap.
absl::Status VerifyFrontier(const DelayHeap& heap) {
  int64_t last_cp = std::numeric_limits<int64_t>::max();
  for (Node* node : heap.frontier()) {
    EXPECT_TRUE(heap.contains(node));
    EXPECT_GE(last_cp, heap.CriticalPathDelay(node));
    if (heap.direction() == Direction::kGrowsTowardUsers) {
      for (Node* succ : node->users()) {
        EXPECT_FALSE(heap.contains(succ));
      }
    } else {
      for (Node* succ : node->operands()) {
        EXPECT_FALSE(heap.contains(succ));
      }
    }
    last_cp = heap.CriticalPathDelay(node);
  }
  return absl::OkStatus();
}

TEST_F(DelayHeapTest, BenchmarkTest) {
  // Build a heap from the full set of nodes in a benchmark ("sha256") then
  // incrementally remove nodes from the heap verifying properties of the heap
  // at each step.
  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<Package> p,
      sample_packages::GetBenchmark("examples/sha256", /*optimized=*/true));
  std::optional<xls::FunctionBase*> top = p->GetTop();
  ASSERT_TRUE(top.has_value());
  xls::FunctionBase* f = top.value();
  // Run test in both directions.
  for (Direction direction :
       {Direction::kGrowsTowardUsers, Direction::kGrowsTowardOperands}) {
    DelayHeap heap(direction, *delay_estimator_);
    for (Node* node : direction == Direction::kGrowsTowardUsers
                          ? TopoSort(f)
                          : ReverseTopoSort(f)) {
      XLS_ASSERT_OK(heap.Add(node));
      EXPECT_FALSE(heap.frontier().empty());
      XLS_ASSERT_OK(VerifyFrontier(heap));
    }

    // After adding all nodes the delay through the heap should be the
    // critical-path delay through the whole function.
    XLS_ASSERT_OK_AND_ASSIGN(
        std::vector<CriticalPathEntry> cp_entries,
        AnalyzeCriticalPath(f, std::nullopt, *delay_estimator_));
    EXPECT_EQ(heap.CriticalPathDelay(), cp_entries.front().path_delay_ps);

    // Iterate through the frontier removing nodes. This should remove *all*
    // nodes in the heap as the removal of nodes expands the frontier and the
    // newly added nodes to the frontier are necessarily latter in the frontier
    // set than the one that was removed.
    auto it = heap.frontier().begin();
    while (it != heap.frontier().end()) {
      it = heap.Remove(*it);
      XLS_ASSERT_OK(VerifyFrontier(heap));
    }
    EXPECT_EQ(heap.size(), 0);
    EXPECT_EQ(heap.CriticalPathDelay(), 0);
  }
}
}  // namespace
}  // namespace sched
}  // namespace xls
