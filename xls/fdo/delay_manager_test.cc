// Copyright 2023 The XLS Authors
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

#include "xls/fdo/delay_manager.h"

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

#include "gtest/gtest.h"
#include "absl/container/flat_hash_map.h"
#include "xls/common/status/matchers.h"
#include "xls/ir/function.h"
#include "xls/ir/ir_parser.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/node.h"
#include "xls/scheduling/scheduling_options.h"

namespace xls {
namespace {

class DelayManagerTest : public IrTestBase {};

// Smoke test.
TEST_F(DelayManagerTest, DelayManager) {
  std::string ir_text = R"(
package p

fn main(i0: bits[3], i1: bits[3]) -> bits[3] {
  add.1: bits[3] = add(i0, i1)
  sub.2: bits[3] = sub(add.1, i1)
  ret udiv.3: bits[3] = udiv(sub.2, add.1)
}
)";

  XLS_ASSERT_OK_AND_ASSIGN(auto package, Parser::ParsePackage(ir_text));
  XLS_ASSERT_OK_AND_ASSIGN(Function * function, package->GetFunction("main"));
  Node *i0 = FindNode("i0", function);
  Node *i1 = FindNode("i1", function);
  Node *add1 = FindNode("add.1", function);
  Node *sub2 = FindNode("sub.2", function);
  Node *udiv3 = FindNode("udiv.3", function);

  DelayManager dm(function, TestDelayEstimator());

  XLS_ASSERT_OK_AND_ASSIGN(int64_t i0_delay, dm.GetNodeDelay(i0));
  EXPECT_EQ(i0_delay, 0);
  XLS_ASSERT_OK_AND_ASSIGN(int64_t i1_delay, dm.GetNodeDelay(i1));
  EXPECT_EQ(i1_delay, 0);
  XLS_ASSERT_OK_AND_ASSIGN(int64_t add1_delay, dm.GetNodeDelay(add1));
  EXPECT_EQ(add1_delay, 1);
  XLS_ASSERT_OK_AND_ASSIGN(int64_t sub2_delay, dm.GetNodeDelay(sub2));
  EXPECT_EQ(sub2_delay, 1);
  XLS_ASSERT_OK_AND_ASSIGN(int64_t udiv3_delay, dm.GetNodeDelay(udiv3));
  EXPECT_EQ(udiv3_delay, 2);

  XLS_ASSERT_OK_AND_ASSIGN(int64_t i0_add1_delay,
                           dm.GetCriticalPathDelay(i0, add1));
  EXPECT_EQ(i0_add1_delay, 1);
  XLS_ASSERT_OK_AND_ASSIGN(int64_t i1_sub2_delay,
                           dm.GetCriticalPathDelay(i1, sub2));
  EXPECT_EQ(i1_sub2_delay, 2);

  XLS_ASSERT_OK_AND_ASSIGN(int64_t sub2_udiv3_delay,
                           dm.GetCriticalPathDelay(sub2, udiv3));
  EXPECT_EQ(sub2_udiv3_delay, 3);
  XLS_ASSERT_OK_AND_ASSIGN(int64_t add1_udiv3_delay,
                           dm.GetCriticalPathDelay(add1, udiv3));
  EXPECT_EQ(add1_udiv3_delay, 4);

  XLS_ASSERT_OK_AND_ASSIGN(int64_t add1_sub2_delay,
                           dm.GetCriticalPathDelay(add1, sub2));
  EXPECT_EQ(add1_sub2_delay, 2);
  XLS_ASSERT_OK_AND_ASSIGN(int64_t sub2_add1_delay,
                           dm.GetCriticalPathDelay(sub2, add1));
  EXPECT_EQ(sub2_add1_delay, -1);

  XLS_ASSERT_OK_AND_ASSIGN(int64_t i0_udiv3_delay,
                           dm.GetCriticalPathDelay(i0, udiv3));
  EXPECT_EQ(i0_udiv3_delay, 4);
  XLS_ASSERT_OK_AND_ASSIGN(int64_t udiv3_i0_delay,
                           dm.GetCriticalPathDelay(udiv3, i0));
  EXPECT_EQ(udiv3_i0_delay, -1);

  absl::flat_hash_map<Node *, std::vector<Node *>> threshold_result =
      dm.GetPathsOverDelayThreshold(1);
  EXPECT_TRUE(threshold_result.contains(i0));
  EXPECT_TRUE(threshold_result.contains(i1));
  EXPECT_TRUE(threshold_result.contains(add1));
  EXPECT_TRUE(threshold_result.contains(sub2));
  EXPECT_TRUE(threshold_result.contains(udiv3));

  EXPECT_TRUE(threshold_result.at(i0) == std::vector<Node *>({sub2, udiv3}));
  EXPECT_TRUE(threshold_result.at(i1) == std::vector<Node *>({sub2, udiv3}));
  EXPECT_TRUE(threshold_result.at(add1) == std::vector<Node *>({sub2, udiv3}));
  EXPECT_TRUE(threshold_result.at(sub2) == std::vector<Node *>({udiv3}));
  EXPECT_TRUE(threshold_result.at(udiv3) == std::vector<Node *>({udiv3}));

  ScheduleCycleMap cycle_map;
  cycle_map[i0] = 0;
  cycle_map[i1] = 0;
  cycle_map[add1] = 0;
  cycle_map[sub2] = 0;
  cycle_map[udiv3] = 0;

  PathExtractOptions options;
  options.cycle_map = &cycle_map;
  options.exclude_param_source = false;
  options.input_source_only = false;
  options.output_target_only = false;
  options.unique_target_only = false;

  std::vector<PathInfo> sort_result;
  XLS_ASSERT_OK_AND_ASSIGN(sort_result, dm.GetTopNPaths(4, options));
  EXPECT_TRUE(std::find(sort_result.begin(), sort_result.end(),
                        PathInfo(4, i0, udiv3)) != sort_result.end());
  EXPECT_TRUE(std::find(sort_result.begin(), sort_result.end(),
                        PathInfo(4, i1, udiv3)) != sort_result.end());
  EXPECT_TRUE(std::find(sort_result.begin(), sort_result.end(),
                        PathInfo(4, add1, udiv3)) != sort_result.end());
  EXPECT_TRUE(std::find(sort_result.begin(), sort_result.end(),
                        PathInfo(3, sub2, udiv3)) != sort_result.end());

  XLS_EXPECT_OK(dm.SetCriticalPathDelay(add1, sub2, 1));
  dm.PropagateDelays();

  XLS_ASSERT_OK_AND_ASSIGN(int64_t new_i0_delay, dm.GetNodeDelay(i0));
  EXPECT_EQ(new_i0_delay, 0);
  XLS_ASSERT_OK_AND_ASSIGN(int64_t new_i1_delay, dm.GetNodeDelay(i1));
  EXPECT_EQ(new_i1_delay, 0);
  XLS_ASSERT_OK_AND_ASSIGN(int64_t new_add1_delay, dm.GetNodeDelay(add1));
  EXPECT_EQ(new_add1_delay, 1);
  XLS_ASSERT_OK_AND_ASSIGN(int64_t new_sub2_delay, dm.GetNodeDelay(sub2));
  EXPECT_EQ(new_sub2_delay, 1);
  XLS_ASSERT_OK_AND_ASSIGN(int64_t new_udiv3_delay, dm.GetNodeDelay(udiv3));
  EXPECT_EQ(new_udiv3_delay, 2);

  XLS_ASSERT_OK_AND_ASSIGN(int64_t new_i0_add1_delay,
                           dm.GetCriticalPathDelay(i0, add1));
  EXPECT_EQ(new_i0_add1_delay, 1);
  XLS_ASSERT_OK_AND_ASSIGN(int64_t new_i1_sub2_delay,
                           dm.GetCriticalPathDelay(i1, sub2));
  EXPECT_EQ(new_i1_sub2_delay, 1);

  XLS_ASSERT_OK_AND_ASSIGN(int64_t new_sub2_udiv3_delay,
                           dm.GetCriticalPathDelay(sub2, udiv3));
  EXPECT_EQ(new_sub2_udiv3_delay, 3);
  XLS_ASSERT_OK_AND_ASSIGN(int64_t new_add1_udiv3_delay,
                           dm.GetCriticalPathDelay(add1, udiv3));
  EXPECT_EQ(new_add1_udiv3_delay, 3);

  XLS_ASSERT_OK_AND_ASSIGN(int64_t new_add1_sub2_delay,
                           dm.GetCriticalPathDelay(add1, sub2));
  EXPECT_EQ(new_add1_sub2_delay, 1);
  XLS_ASSERT_OK_AND_ASSIGN(int64_t new_sub2_add1_delay,
                           dm.GetCriticalPathDelay(sub2, add1));
  EXPECT_EQ(new_sub2_add1_delay, -1);

  XLS_ASSERT_OK_AND_ASSIGN(int64_t new_i0_udiv3_delay,
                           dm.GetCriticalPathDelay(i0, udiv3));
  EXPECT_EQ(new_i0_udiv3_delay, 3);
  XLS_ASSERT_OK_AND_ASSIGN(int64_t new_udiv3_i0_delay,
                           dm.GetCriticalPathDelay(udiv3, i0));
  EXPECT_EQ(new_udiv3_i0_delay, -1);
}

}  // namespace
}  // namespace xls
