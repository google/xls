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

#include "xls/fdo/node_cut.h"

#include <string>
#include <vector>

#include "gtest/gtest.h"
#include "absl/container/flat_hash_set.h"
#include "xls/common/status/matchers.h"
#include "xls/ir/function.h"
#include "xls/ir/ir_parser.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/node.h"
#include "xls/scheduling/scheduling_options.h"

namespace xls {
namespace {

class NodeCutTest : public IrTestBase {};

TEST_F(NodeCutTest, NodeCutsEnumeration) {
  std::string ir_text = R"(
package p

fn main(i0: bits[3], i1: bits[3]) -> bits[3] {
  add.1: bits[3] = add(i0, i1)
  sub.2: bits[3] = sub(add.1, i1)
  ret or.3: bits[3] = or(sub.2, add.1)
}
)";

  XLS_ASSERT_OK_AND_ASSIGN(auto package, Parser::ParsePackage(ir_text));
  XLS_ASSERT_OK_AND_ASSIGN(Function * function, package->GetFunction("main"));
  Node *i0 = FindNode("i0", function);
  Node *i1 = FindNode("i1", function);
  Node *add1 = FindNode("add.1", function);
  Node *sub2 = FindNode("sub.2", function);
  Node *or3 = FindNode("or.3", function);

  ScheduleCycleMap cycle_map;
  cycle_map[i0] = 0;
  cycle_map[i1] = 0;
  cycle_map[add1] = 0;
  cycle_map[sub2] = 0;
  cycle_map[or3] = 0;

  XLS_ASSERT_OK_AND_ASSIGN(
      NodeCutsMap cuts_map,
      EnumerateCutsInSchedule(function, 1, cycle_map,
                              /*input_leaves_only=*/false));

  EXPECT_FALSE(cuts_map.contains(i0));
  EXPECT_FALSE(cuts_map.contains(i1));

  const std::vector<NodeCut> &add1_cuts = cuts_map.at(add1);
  EXPECT_EQ(add1_cuts.size(), 2);
  if (add1_cuts.size() == 2) {
    EXPECT_TRUE(add1_cuts.at(0).IsTrivial());
    EXPECT_TRUE(add1_cuts.at(1) == NodeCut(add1, {i0, i1}));
  }

  const std::vector<NodeCut> &sub2_cuts = cuts_map.at(sub2);
  EXPECT_EQ(sub2_cuts.size(), 3);
  if (sub2_cuts.size() == 3) {
    EXPECT_TRUE(sub2_cuts.at(0).IsTrivial());
    EXPECT_TRUE(sub2_cuts.at(1) == NodeCut(sub2, {i0, i1}));
    EXPECT_TRUE(sub2_cuts.at(2) == NodeCut(sub2, {add1, i1}));
  }

  const std::vector<NodeCut> &or3_cuts = cuts_map.at(or3);
  EXPECT_EQ(or3_cuts.size(), 4);
  if (or3_cuts.size() == 4) {
    EXPECT_TRUE(or3_cuts.at(0).IsTrivial());
    EXPECT_TRUE(or3_cuts.at(1) == NodeCut(or3, {add1, i1}));
    EXPECT_TRUE(or3_cuts.at(2) == NodeCut(or3, {i0, i1}));
    EXPECT_TRUE(or3_cuts.at(3) == NodeCut(or3, {sub2, add1}));
  }

  EXPECT_TRUE(or3_cuts.at(0).GetNodeCone() ==
              absl::flat_hash_set<Node *>({or3}));
  EXPECT_TRUE(or3_cuts.at(1).GetNodeCone() ==
              absl::flat_hash_set<Node *>({or3, sub2}));
  EXPECT_TRUE(or3_cuts.at(2).GetNodeCone() ==
              absl::flat_hash_set<Node *>({or3, add1, sub2}));
  EXPECT_TRUE(or3_cuts.at(3).GetNodeCone() ==
              absl::flat_hash_set<Node *>({or3}));

  XLS_ASSERT_OK_AND_ASSIGN(NodeCutsMap icuts_map,
                           EnumerateCutsInSchedule(function, 1, cycle_map,
                                                   /*input_leaves_only=*/true));

  EXPECT_FALSE(icuts_map.contains(i0));
  EXPECT_FALSE(icuts_map.contains(i1));

  const std::vector<NodeCut> &add1_icuts = icuts_map.at(add1);
  EXPECT_EQ(add1_icuts.size(), 1);
  if (add1_icuts.size() == 1) {
    EXPECT_TRUE(add1_icuts.at(0) == NodeCut(add1, {i0, i1}));
  }

  const std::vector<NodeCut> &sub2_icuts = icuts_map.at(sub2);
  EXPECT_EQ(sub2_icuts.size(), 1);
  if (sub2_icuts.size() == 1) {
    EXPECT_TRUE(sub2_icuts.at(0) == NodeCut(sub2, {i0, i1}));
  }

  const std::vector<NodeCut> &or3_icuts = icuts_map.at(or3);
  EXPECT_EQ(or3_icuts.size(), 1);
  if (or3_icuts.size() == 1) {
    EXPECT_TRUE(or3_icuts.at(0) == NodeCut(or3, {i0, i1}));
  }
}

}  // namespace
}  // namespace xls
