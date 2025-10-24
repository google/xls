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

#include "xls/passes/folding_graph.h"

#include <utility>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "xls/common/fuzzing/fuzztest.h"
#include "absl/types/span.h"
#include "xls/common/status/matchers.h"
#include "xls/ir/bits.h"
#include "xls/ir/function.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/nodes.h"
#include "xls/ir/op.h"
#include "xls/ir/package.h"

namespace xls {
namespace {

using ::testing::ElementsAre;

class FoldingGraphTest : public IrTestBase {
 protected:
  FoldingGraphTest() = default;
};

TEST_F(FoldingGraphTest, FoldingGraphIsOrdered) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  Type* u32_type = p->GetBitsType(32);
  BValue op = fb.Param("op", u32_type);
  BValue i = fb.Param("i", u32_type);
  BValue j = fb.Param("j", u32_type);
  BValue literal_0 = fb.Literal(UBits(0, 32));
  BValue literal_1 = fb.Literal(UBits(1, 32));
  BValue add = fb.Add(i, j);
  BValue sub = fb.Subtract(i, j);
  BValue mul = fb.UMul(i, j);
  BValue select = fb.PrioritySelect(
      fb.Concat({fb.Eq(op, literal_1), fb.Eq(op, literal_0)}), {add, sub}, mul);

  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.BuildWithReturnValue(select));
  std::vector<std::unique_ptr<BinaryFoldingAction>> foldable_actions;
  // NOTE: node order is reversed to ensure FoldingGraph does sorting.
  for (Node* one : {mul.node(), sub.node(), add.node()}) {
    for (Node* other : {sub.node(), mul.node(), add.node()}) {
      if (one != other) {
        foldable_actions.push_back(std::make_unique<BinaryFoldingAction>(
            one, other, select.node(), 0, 0));
      }
    }
  }
  FoldingGraph graph(f, std::move(foldable_actions));

  EXPECT_EQ(graph.GetNodes().size(), 3);
  EXPECT_THAT(graph.GetNodes(),
              ElementsAre(add.node(), sub.node(), mul.node()));
  EXPECT_EQ(graph.GetEdges().size(), 6);
  EXPECT_EQ(graph.GetEdges()[0]->GetFrom(), add.node());
  EXPECT_EQ(graph.GetEdges()[0]->GetTo(), sub.node());
  EXPECT_EQ(graph.GetEdges()[1]->GetFrom(), add.node());
  EXPECT_EQ(graph.GetEdges()[1]->GetTo(), mul.node());
  EXPECT_EQ(graph.GetEdges()[2]->GetFrom(), sub.node());
  EXPECT_EQ(graph.GetEdges()[2]->GetTo(), add.node());
  EXPECT_EQ(graph.GetEdges()[3]->GetFrom(), sub.node());
  EXPECT_EQ(graph.GetEdges()[3]->GetTo(), mul.node());
  EXPECT_EQ(graph.GetEdges()[4]->GetFrom(), mul.node());
  EXPECT_EQ(graph.GetEdges()[4]->GetTo(), add.node());
  EXPECT_EQ(graph.GetEdges()[5]->GetFrom(), mul.node());
  EXPECT_EQ(graph.GetEdges()[5]->GetTo(), sub.node());
}

}  // namespace
}  // namespace xls
