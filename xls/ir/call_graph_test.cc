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

#include "xls/ir/call_graph.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "xls/common/status/matchers.h"
#include "xls/ir/bits.h"
#include "xls/ir/block.h"
#include "xls/ir/function.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/node.h"
#include "xls/ir/nodes.h"

namespace xls {
namespace {

using ::testing::ElementsAre;
using ::testing::UnorderedElementsAre;

class CallGraphTest : public IrTestBase {};

TEST_F(CallGraphTest, SingleFunction) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  fb.Param("x", p->GetBitsType(32));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  EXPECT_THAT(GetDependentFunctions(f), ElementsAre(f));
  EXPECT_THAT(GetDependentFunctions(f), ElementsAre(f));
  EXPECT_THAT(FunctionsInPostOrder(p.get()), ElementsAre(f));
}

TEST_F(CallGraphTest, FunctionChain) {
  auto p = CreatePackage();
  Type* u32 = p->GetBitsType(32);

  Function* a;
  {
    FunctionBuilder fb("a", p.get());
    fb.Param("x", u32);
    XLS_ASSERT_OK_AND_ASSIGN(a, fb.Build());
  }
  Function* b;
  {
    FunctionBuilder fb("b", p.get());
    BValue x = fb.Param("x", u32);
    fb.Invoke({x}, a);
    XLS_ASSERT_OK_AND_ASSIGN(b, fb.Build());
    EXPECT_THAT(GetDependentFunctions(b), ElementsAre(a, b));
  }
  Function* c;
  {
    FunctionBuilder fb("c", p.get());
    BValue array = fb.Param("array", p->GetArrayType(42, u32));
    fb.Map(array, b);
    XLS_ASSERT_OK_AND_ASSIGN(c, fb.Build());
  }
  EXPECT_THAT(GetDependentFunctions(b), ElementsAre(a, b));
  EXPECT_THAT(GetDependentFunctions(c), ElementsAre(a, b, c));
  EXPECT_THAT(FunctionsInPostOrder(p.get()), ElementsAre(a, b, c));
}

TEST_F(CallGraphTest, CallGraphDiamond) {
  auto p = CreatePackage();
  Type* u32 = p->GetBitsType(32);

  Function* a;
  {
    FunctionBuilder fb("a", p.get());
    fb.Param("x", u32);
    XLS_ASSERT_OK_AND_ASSIGN(a, fb.Build());
  }
  Function* b;
  {
    FunctionBuilder fb("b", p.get());
    BValue x = fb.Param("x", u32);
    fb.Invoke({x}, a);
    XLS_ASSERT_OK_AND_ASSIGN(b, fb.Build());
  }
  Function* c;
  {
    FunctionBuilder fb("c", p.get());
    BValue x = fb.Param("x", u32);
    fb.Invoke({x}, a);
    XLS_ASSERT_OK_AND_ASSIGN(c, fb.Build());
  }
  Function* d;
  {
    FunctionBuilder fb("d", p.get());
    BValue array = fb.Param("array", p->GetArrayType(42, u32));
    fb.Map(array, b);
    fb.Map(array, c);
    XLS_ASSERT_OK_AND_ASSIGN(d, fb.Build());
  }
  EXPECT_EQ(GetDependentFunctions(d).front(), a);
  EXPECT_EQ(GetDependentFunctions(d).back(), d);
  EXPECT_THAT(GetDependentFunctions(d), UnorderedElementsAre(a, b, c, d));

  EXPECT_EQ(FunctionsInPostOrder(p.get()).front(), a);
  EXPECT_EQ(FunctionsInPostOrder(p.get()).back(), d);
  EXPECT_THAT(FunctionsInPostOrder(p.get()), UnorderedElementsAre(a, b, c, d));
}

TEST_F(CallGraphTest, SeveralFunctions) {
  auto p = CreatePackage();
  Type* u32 = p->GetBitsType(32);

  Function* a;
  {
    FunctionBuilder fb("a", p.get());
    fb.Param("x", u32);
    XLS_ASSERT_OK_AND_ASSIGN(a, fb.Build());
  }
  Function* b;
  {
    FunctionBuilder fb("b", p.get());
    BValue x = fb.Param("x", u32);
    fb.Invoke({x}, a);
    XLS_ASSERT_OK_AND_ASSIGN(b, fb.Build());
  }
  Function* c;
  {
    FunctionBuilder fb("c", p.get());
    BValue x = fb.Param("x", u32);
    fb.Invoke({x}, a);
    XLS_ASSERT_OK_AND_ASSIGN(c, fb.Build());
  }
  Function* d;
  {
    FunctionBuilder fb("d", p.get());
    BValue array = fb.Param("array", p->GetArrayType(42, u32));
    fb.Map(array, b);
    XLS_ASSERT_OK_AND_ASSIGN(d, fb.Build());
  }
  EXPECT_EQ(FunctionsInPostOrder(p.get()).back(), d);
  EXPECT_THAT(FunctionsInPostOrder(p.get()), UnorderedElementsAre(a, b, c, d));
}

TEST_F(CallGraphTest, CloneFunctionAndItsDependencies) {
  auto p = CreatePackage();
  Type* u32 = p->GetBitsType(32);

  Function* a;
  {
    FunctionBuilder fb("a", p.get());
    fb.Param("x", u32);
    XLS_ASSERT_OK_AND_ASSIGN(a, fb.Build());
  }
  Function* b;
  {
    FunctionBuilder fb("b", p.get());
    BValue x = fb.Param("x", u32);
    fb.Invoke({x}, a);
    XLS_ASSERT_OK_AND_ASSIGN(b, fb.Build());
    EXPECT_THAT(GetDependentFunctions(b), ElementsAre(a, b));
  }
  Function* c;
  {
    FunctionBuilder fb("c", p.get());
    BValue array = fb.Param("array", p->GetArrayType(42, u32));
    fb.Map(array, b);
    XLS_ASSERT_OK_AND_ASSIGN(c, fb.Build());
  }
  EXPECT_THAT(GetDependentFunctions(c), ElementsAre(a, b, c));
  EXPECT_THAT(p->functions().size(), 3);

  auto p_clone = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(
      Function * c_clone,
      CloneFunctionAndItsDependencies(
          /*to_clone=*/c, /*new_name=*/"c_clone", p_clone.get()));
  EXPECT_THAT(c_clone->name(), "c_clone");
  EXPECT_THAT(p_clone->functions().size(), 3);
}

TEST_F(CallGraphTest, BlockInstantiation) {
  auto p = CreatePackage();

  Block* a;
  Node* a_lit;
  {
    BlockBuilder bb("a", p.get());
    BValue lit = bb.Literal(UBits(1, 8));
    bb.OutputPort("o", lit);
    a_lit = lit.node();
    XLS_ASSERT_OK_AND_ASSIGN(a, bb.Build());
  }
  Block* b;
  Node* b_lit;
  {
    BlockBuilder bb("b", p.get());
    BValue lit = bb.Literal(UBits(1, 8));
    bb.OutputPort("o", lit);
    b_lit = lit.node();
    XLS_ASSERT_OK_AND_ASSIGN(b, bb.Build());
  }
  Block* c;
  {
    BlockBuilder bb("c", p.get());
    bb.OutputPort("o", bb.Literal(UBits(1, 8)));
    XLS_ASSERT_OK_AND_ASSIGN(c, bb.Build());
  }

  // Rewire to have a instantiate b instantiate c
  XLS_ASSERT_OK_AND_ASSIGN(auto* b_inst, a->AddBlockInstantiation("b_inst", b));
  XLS_ASSERT_OK_AND_ASSIGN(auto* c_inst, b->AddBlockInstantiation("c_inst", c));
  XLS_ASSERT_OK(
      a_lit->ReplaceUsesWithNew<InstantiationOutput>(b_inst, "o").status());
  XLS_ASSERT_OK(
      b_lit->ReplaceUsesWithNew<InstantiationOutput>(c_inst, "o").status());

  // Check order.
  EXPECT_THAT(p->GetFunctionBases(), ElementsAre(a, b, c));
  EXPECT_THAT(FunctionsInPostOrder(p.get()), ElementsAre(c, b, a));
}

}  // namespace
}  // namespace xls
