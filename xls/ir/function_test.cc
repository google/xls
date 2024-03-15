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

#include "xls/ir/function.h"

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/container/flat_hash_map.h"
#include "xls/common/status/matchers.h"
#include "xls/ir/bits.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/node_util.h"
#include "xls/ir/nodes.h"
#include "xls/ir/op.h"
#include "xls/ir/package.h"
#include "xls/ir/source_location.h"
#include "xls/ir/topo_sort.h"
#include "xls/ir/type.h"
#include "xls/ir/value.h"

namespace xls {
namespace {

using status_testing::StatusIs;
using ::testing::ElementsAre;
using ::testing::HasSubstr;

class FunctionTest : public IrTestBase {};

TEST_F(FunctionTest, BasicPropertiesTest) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * func, ParseFunction(R"(
fn foo(x: bits[32], y: bits[32]) -> bits[32] {
ret add.3: bits[32] = add(x, y)
}
)",
                                                          p.get()));
  EXPECT_EQ(func->name(), "foo");
  EXPECT_EQ(func->qualified_name(), "BasicPropertiesTest::foo");
  EXPECT_TRUE(func->IsFunction());
  EXPECT_FALSE(func->IsProc());

  func->SetName("bar");
  EXPECT_EQ(func->name(), "bar");
  EXPECT_EQ(func->qualified_name(), "BasicPropertiesTest::bar");
}

TEST_F(FunctionTest, CloneSimpleFunction) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * func, ParseFunction(R"(
fn add(x: bits[32], y: bits[32]) -> bits[32] {
ret add.3: bits[32] = add(x, y)
}
)",
                                                          p.get()));
  XLS_ASSERT_OK_AND_ASSIGN(Function * func_clone, func->Clone("foobar"));
  EXPECT_EQ(func_clone->name(), "foobar");
  EXPECT_EQ(func_clone->node_count(), 3);
  EXPECT_EQ(func_clone->return_value()->op(), Op::kAdd);
}

TEST_F(FunctionTest, CloneSimpleFunctionToDifferentPackage) {
  auto p = CreatePackage();
  FunctionBuilder b("f", p.get());
  auto x = b.Param("x", p->GetBitsType(32));
  auto y = b.Param("y", p->GetBitsType(32));
  auto arr = b.Array({x, y}, x.GetType());
  XLS_ASSERT_OK_AND_ASSIGN(Function * func, b.BuildWithReturnValue(arr));

  auto new_package = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * func_clone,
                           func->Clone("newbar", new_package.get()));
  EXPECT_EQ(func_clone->name(), "newbar");
  EXPECT_EQ(func_clone->node_count(), 3);
  EXPECT_EQ(func_clone->return_value()->op(), Op::kArray);
  EXPECT_EQ(func_clone->package(), new_package.get());
}

TEST_F(FunctionTest, DumpIrWhenParamIsRetval) {
  auto p = CreatePackage();
  FunctionBuilder b("f", p.get());
  auto x = b.Param("x", p->GetBitsType(32));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, b.BuildWithReturnValue(x));
  EXPECT_EQ(f->DumpIr(), R"(fn f(x: bits[32]) -> bits[32] {
  ret x: bits[32] = param(name=x)
}
)");
}

TEST_F(FunctionTest, CloneFunctionWithDeadNodes) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * func, ParseFunction(R"(
fn add(x: bits[32], y: bits[32]) -> bits[32] {
  add.1: bits[32] = add(x, y)
  sub.2: bits[32] = sub(x, y)
  literal.3: bits[32] = literal(value=42)
  umul.4: bits[32] = umul(add.1, literal.3)
  ret neg.5: bits[32] = neg(sub.2)
}
)",
                                                          p.get()));
  XLS_ASSERT_OK_AND_ASSIGN(Function * func_clone, func->Clone("foobar"));

  EXPECT_EQ(func->node_count(), 7);
  EXPECT_EQ(func_clone->node_count(), 7);
}

TEST_F(FunctionTest, IsDefinitelyEqualTo) {
  XLS_ASSERT_OK_AND_ASSIGN(auto p, ParsePackage(R"(
package function_is_equal

fn f1(x: bits[32], y: bits[32]) -> bits[32] {
  add.1: bits[32] = add(x, y)
  sub.2: bits[32] = sub(x, y)
  literal.3: bits[32] = literal(value=42)
  ret umul.4: bits[32] = umul(add.1, literal.3)
}

fn same_as_f1(x: bits[32], y: bits[32]) -> bits[32] {
  add.5: bits[32] = add(x, y)
  sub.6: bits[32] = sub(x, y)
  literal.7: bits[32] = literal(value=42)
  ret umul.8: bits[32] = umul(add.5, literal.7)
}

fn same_as_f1_different_order(x: bits[32], y: bits[32]) -> bits[32] {
  literal.37: bits[32] = literal(value=42)
  sub.36: bits[32] = sub(x, y)
  add.35: bits[32] = add(x, y)
  ret umul.38: bits[32] = umul(add.35, literal.37)
}

fn extra_parameter(x: bits[32], y: bits[32], z: bits[32]) -> bits[32] {
  add.9: bits[32] = add(x, y)
  sub.10: bits[32] = sub(x, y)
  literal.11: bits[32] = literal(value=42)
  ret umul.12: bits[32] = umul(add.9, literal.11)
}

fn different_types(x: bits[16], y: bits[16]) -> bits[16] {
  add.21: bits[16] = add(x, y)
  sub.22: bits[16] = sub(x, y)
  literal.23: bits[16] = literal(value=42)
  ret umul.24: bits[16] = umul(add.21, literal.23)
}
)"));

  auto functions_equal = [&](std::string_view a, std::string_view b) {
    return FindFunction(a, p.get())
        ->IsDefinitelyEqualTo(FindFunction(b, p.get()));
  };

  EXPECT_TRUE(functions_equal("f1", "f1"));
  EXPECT_TRUE(functions_equal("f1", "same_as_f1"));
  EXPECT_TRUE(functions_equal("f1", "same_as_f1_different_order"));
  EXPECT_FALSE(functions_equal("f1", "extra_parameter"));
  EXPECT_FALSE(functions_equal("f1", "different_types"));
}

TEST_F(FunctionTest, RemoveParam) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * func, ParseFunction(R"(
fn id(x: bits[32], y: bits[32]) -> bits[32] {
  ret x: bits[32] = param(name=x)
}
)",
                                                          p.get()));
  XLS_ASSERT_OK_AND_ASSIGN(Param * y, func->GetParamByName("y"));
  Type* u32 = p->GetBitsType(32);
  FunctionType* orig = p->GetFunctionType({u32, u32}, u32);
  FunctionType* updated = p->GetFunctionType({u32}, u32);
  EXPECT_NE(orig, updated);
  EXPECT_EQ(func->GetType(), orig);
  EXPECT_THAT(func->params(),
              ElementsAre(FindNode("x", func), FindNode("y", func)));
  XLS_EXPECT_OK(func->RemoveNode(y));
  EXPECT_THAT(func->params(), ElementsAre(FindNode("x", func)));
  EXPECT_EQ(func->GetType(), updated);
}

TEST_F(FunctionTest, MoveParams) {
  auto p = CreatePackage();
  FunctionBuilder b("f", p.get());
  auto x = b.Param("x", p->GetBitsType(32));
  auto y = b.Param("y", p->GetBitsType(32));
  auto z = b.Param("z", p->GetBitsType(32));
  XLS_ASSERT_OK_AND_ASSIGN(Function * func, b.BuildWithReturnValue(z));

  EXPECT_EQ(func->params()[0]->GetName(), "x");
  EXPECT_EQ(func->params()[1]->GetName(), "y");
  EXPECT_EQ(func->params()[2]->GetName(), "z");

  // Nop move.
  XLS_ASSERT_OK(func->MoveParamToIndex(x.node()->As<Param>(), 0));

  EXPECT_EQ(func->params()[0]->GetName(), "x");
  EXPECT_EQ(func->params()[1]->GetName(), "y");
  EXPECT_EQ(func->params()[2]->GetName(), "z");

  XLS_ASSERT_OK(func->MoveParamToIndex(x.node()->As<Param>(), 1));

  EXPECT_EQ(func->params()[0]->GetName(), "y");
  EXPECT_EQ(func->params()[1]->GetName(), "x");
  EXPECT_EQ(func->params()[2]->GetName(), "z");

  XLS_ASSERT_OK(func->MoveParamToIndex(z.node()->As<Param>(), 0));

  EXPECT_EQ(func->params()[0]->GetName(), "z");
  EXPECT_EQ(func->params()[1]->GetName(), "y");
  EXPECT_EQ(func->params()[2]->GetName(), "x");

  XLS_ASSERT_OK(func->MoveParamToIndex(y.node()->As<Param>(), 2));

  EXPECT_EQ(func->params()[0]->GetName(), "z");
  EXPECT_EQ(func->params()[1]->GetName(), "x");
  EXPECT_EQ(func->params()[2]->GetName(), "y");
}

TEST_F(FunctionTest, MakeInvalidNode) {
  Package p(TestName());
  XLS_ASSERT_OK_AND_ASSIGN(Function * func, ParseFunction(R"(
fn id(x: bits[16], y: bits[32]) -> bits[16] {
  ret x: bits[16] = param(name=x)
}
)",
                                                          &p));

  EXPECT_THAT(
      func->MakeNode<NaryOp>(
          FindNode("x", &p)->loc(),
          std::vector<Node*>{FindNode("x", &p), FindNode("y", &p)}, Op::kXor),
      StatusIs(absl::StatusCode::kInternal,
               HasSubstr("Expected operand 1 of xor.3 to have type bits[16], "
                         "has type bits[32]")));
}

TEST_F(FunctionTest, IsLiteralMask) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  auto seven_3b = fb.Literal(UBits(0b111, 3));
  auto two_3b = fb.Literal(UBits(0b011, 3));
  auto one_1b = fb.Literal(UBits(0b1, 1));
  auto zero_1b = fb.Literal(UBits(0b0, 1));
  auto zero_0b = fb.Literal(UBits(0b0, 0));

  int64_t leading_zeros, trailing_ones;
  EXPECT_TRUE(IsLiteralMask(seven_3b.node(), &leading_zeros, &trailing_ones));
  EXPECT_EQ(0, leading_zeros);
  EXPECT_EQ(3, trailing_ones);

  EXPECT_TRUE(IsLiteralMask(two_3b.node(), &leading_zeros, &trailing_ones));
  EXPECT_EQ(1, leading_zeros);
  EXPECT_EQ(2, trailing_ones);

  EXPECT_TRUE(IsLiteralMask(one_1b.node(), &leading_zeros, &trailing_ones));
  EXPECT_EQ(0, leading_zeros);
  EXPECT_EQ(1, trailing_ones);

  EXPECT_FALSE(IsLiteralMask(zero_1b.node(), &leading_zeros, &trailing_ones));
  EXPECT_FALSE(IsLiteralMask(zero_0b.node(), &leading_zeros, &trailing_ones));
}

TEST_F(FunctionTest, CloneCountedForRemap) {
  auto p = CreatePackage();
  FunctionBuilder fb_body("body_a", p.get());
  fb_body.Param("index", p->GetBitsType(2));
  fb_body.Param("acc", p->GetBitsType(2));
  fb_body.Literal(UBits(0b11, 2));
  XLS_ASSERT_OK_AND_ASSIGN(Function * body_a, fb_body.Build());
  XLS_ASSERT_OK_AND_ASSIGN(Function * body_b, body_a->Clone("body_b"));
  XLS_ASSERT_OK_AND_ASSIGN(Function * body_c, body_a->Clone("body_c"));

  FunctionBuilder fb_main("main", p.get());
  auto init = fb_main.Param("init_acc", p->GetBitsType(2));
  fb_main.CountedFor(init, /*trip_count=*/4, /*stride=*/1, body_a,
                     /*invariant_args=*/{}, SourceInfo(), "counted_a");
  fb_main.CountedFor(init, /*trip_count=*/4, /*stride=*/1, body_b,
                     /*invariant_args=*/{}, SourceInfo(), "counted_b");
  XLS_ASSERT_OK_AND_ASSIGN(Function * main, fb_main.Build());

  absl::flat_hash_map<const Function*, Function*> remap;
  remap[body_a] = body_c;

  XLS_ASSERT_OK_AND_ASSIGN(
      Function * main_clone,
      main->Clone("main_clone", /*target_package=*/p.get(), remap));
  XLS_ASSERT_OK_AND_ASSIGN(Node * node_a_clone,
                           main_clone->GetNode("counted_a"));
  XLS_ASSERT_OK_AND_ASSIGN(Node * node_b_clone,
                           main_clone->GetNode("counted_b"));
  CountedFor* counted_a_clone = node_a_clone->As<CountedFor>();
  CountedFor* counted_b_clone = node_b_clone->As<CountedFor>();
  EXPECT_EQ(counted_a_clone->body(), body_c);
  EXPECT_EQ(counted_b_clone->body(), body_b);
}

TEST_F(FunctionTest, CloneMapRemap) {
  auto p = CreatePackage();
  FunctionBuilder fb_body("apply_a", p.get());
  fb_body.Param("in", p->GetBitsType(2));
  fb_body.Literal(UBits(0b11, 2));
  XLS_ASSERT_OK_AND_ASSIGN(Function * apply_a, fb_body.Build());
  XLS_ASSERT_OK_AND_ASSIGN(Function * apply_b, apply_a->Clone("apply_b"));
  XLS_ASSERT_OK_AND_ASSIGN(Function * apply_c, apply_a->Clone("apply_c"));

  FunctionBuilder fb_main("main", p.get());
  auto input = fb_main.Param("input", p->GetArrayType(2, p->GetBitsType(2)));
  fb_main.Map(input, apply_a, SourceInfo(), "map_a");
  fb_main.Map(input, apply_b, SourceInfo(), "map_b");
  XLS_ASSERT_OK_AND_ASSIGN(Function * main, fb_main.Build());

  absl::flat_hash_map<const Function*, Function*> remap;
  remap[apply_a] = apply_c;

  XLS_ASSERT_OK_AND_ASSIGN(
      Function * main_clone,
      main->Clone("main_clone", /*target_package=*/p.get(), remap));
  XLS_ASSERT_OK_AND_ASSIGN(Node * node_a_clone, main_clone->GetNode("map_a"));
  XLS_ASSERT_OK_AND_ASSIGN(Node * node_b_clone, main_clone->GetNode("map_b"));
  Map* map_a_clone = node_a_clone->As<Map>();
  Map* map_b_clone = node_b_clone->As<Map>();
  EXPECT_EQ(map_a_clone->to_apply(), apply_c);
  EXPECT_EQ(map_b_clone->to_apply(), apply_b);
}

TEST_F(FunctionTest, CloneInvokeForRemap) {
  auto p = CreatePackage();
  FunctionBuilder fb_body("body_a", p.get());
  fb_body.Literal(UBits(0b11, 2));
  XLS_ASSERT_OK_AND_ASSIGN(Function * apply_a, fb_body.Build());
  XLS_ASSERT_OK_AND_ASSIGN(Function * apply_b, apply_a->Clone("apply_b"));
  XLS_ASSERT_OK_AND_ASSIGN(Function * apply_c, apply_a->Clone("apply_c"));

  FunctionBuilder fb_main("main", p.get());
  // TODO finish
  fb_main.Invoke(/*args=*/{}, apply_a, SourceInfo(), "invoke_a");
  fb_main.Invoke(/*args=*/{}, apply_b, SourceInfo(), "invoke_b");
  XLS_ASSERT_OK_AND_ASSIGN(Function * main, fb_main.Build());

  absl::flat_hash_map<const Function*, Function*> remap;
  remap[apply_a] = apply_c;

  XLS_ASSERT_OK_AND_ASSIGN(
      Function * main_clone,
      main->Clone("main_clone", /*target_package=*/p.get(), remap));
  XLS_ASSERT_OK_AND_ASSIGN(Node * node_a_clone,
                           main_clone->GetNode("invoke_a"));
  XLS_ASSERT_OK_AND_ASSIGN(Node * node_b_clone,
                           main_clone->GetNode("invoke_b"));
  Invoke* invoke_a_clone = node_a_clone->As<Invoke>();
  Invoke* invoke_b_clone = node_b_clone->As<Invoke>();
  EXPECT_EQ(invoke_a_clone->to_apply(), apply_c);
  EXPECT_EQ(invoke_b_clone->to_apply(), apply_b);
}

TEST_F(FunctionTest, IrReservedWordIdentifiers) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  auto token = fb.Param("token", p->GetBitsType(32));
  auto reg = fb.Param("reg", p->GetBitsType(32));
  auto rreg = fb.Param("rreg", p->GetBitsType(32));
  XLS_ASSERT_OK(fb.Build().status());

  EXPECT_EQ(token.GetName(), "_token");
  EXPECT_EQ(reg.GetName(), "_reg");
  EXPECT_EQ(rreg.GetName(), "rreg");

  // Serialize to text and verify text is parsable.
  std::string ir_text = p->DumpIr();
  XLS_ASSERT_OK(ParsePackage(ir_text).status());
}

class TestVisitor : public DfsVisitorWithDefault {
 public:
  absl::Status DefaultHandler(Node* node) override { return absl::OkStatus(); }
};

TEST_F(FunctionTest, GraphWithCycle) {
  std::string input = R"(
fn graph(p: bits[42], q: bits[42]) -> bits[42] {
  a: bits[42] = and(p, q)
  b: bits[42] = add(a, q)
  ret c: bits[42] = sub(a, b)
}
)";
  {
    auto p = std::make_unique<Package>(TestName());
    XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(input, p.get()));

    // Introduce a cycle in the graph through two nodes.
    ASSERT_TRUE(
        FindNode("a", f)->ReplaceOperand(FindNode("p", f), FindNode("b", f)));
    TestVisitor v;
    EXPECT_THAT(std::string(f->Accept(&v).message()),
                HasSubstr(std::string("Cycle detected: [a -> b -> a]")));
    EXPECT_DEATH(TopoSort(f), HasSubstr("Cycle detected: [a -> b -> a]"));
  }

  {
    auto p = std::make_unique<Package>(TestName());
    XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(input, p.get()));

    // Introduce a cycle in the graph through one node.
    XLS_ASSERT_OK(FindNode("a", f)->ReplaceOperandNumber(0, FindNode("a", f)));
    TestVisitor v;
    EXPECT_THAT(std::string(f->Accept(&v).message()),
                HasSubstr(std::string("Cycle detected: [a -> a]")));
    EXPECT_DEATH(TopoSort(f), HasSubstr("Cycle detected: [a -> a]"));
  }

  {
    auto p = std::make_unique<Package>(TestName());
    XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(input, p.get()));

    // Introduce a cycle such that no nodes in the graph are userless.
    XLS_ASSERT_OK(FindNode("a", f)->ReplaceOperandNumber(1, FindNode("c", f)));
    TestVisitor v;
    EXPECT_THAT(std::string(f->Accept(&v).message()),
                HasSubstr(std::string("Cycle detected: [a -> c -> a]")));
    EXPECT_DEATH(TopoSort(f), HasSubstr("Cycle detected: [a -> c -> a]"));
  }
}

}  // namespace
}  // namespace xls
