// Copyright 2020 Google LLC
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

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "xls/common/status/matchers.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/ir_test_base.h"

namespace xls {
namespace {

using status_testing::StatusIs;
using ::testing::ElementsAre;
using ::testing::HasSubstr;

class FunctionTest : public IrTestBase {};

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

TEST_F(FunctionTest, DumpIrWhenParamIsRetval) {
  auto p = CreatePackage();
  FunctionBuilder b("f", p.get());
  auto x = b.Param("x", p->GetBitsType(32));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, b.BuildWithReturnValue(x));
  EXPECT_EQ(f->DumpIr(), R"(fn f(x: bits[32]) -> bits[32] {
  ret param.1: bits[32] = param(name=x)
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

  auto functions_equal = [&](absl::string_view a, absl::string_view b) {
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
  ret param.1: bits[32] = param(name=x)
}
)",
                                                          p.get()));
  XLS_ASSERT_OK_AND_ASSIGN(Param * y, func->GetParamByName("y"));
  EXPECT_THAT(func->RemoveNode(y, /*remove_param_ok=*/false),
              StatusIs(absl::StatusCode::kInternal,
                       HasSubstr("Attempting to remove parameter")));
  Type* u32 = p->GetBitsType(32);
  FunctionType* orig = p->GetFunctionType({u32, u32}, u32);
  FunctionType* updated = p->GetFunctionType({u32}, u32);
  EXPECT_NE(orig, updated);
  EXPECT_EQ(func->GetType(), orig);
  EXPECT_THAT(func->params(),
              ElementsAre(FindNode("x", func), FindNode("y", func)));
  XLS_EXPECT_OK(func->RemoveNode(y, /*remove_param_ok=*/true));
  EXPECT_THAT(func->params(), ElementsAre(FindNode("x", func)));
  EXPECT_EQ(func->GetType(), updated);
}

TEST_F(FunctionTest, MakeInvalidNode) {
  Package p(TestName());
  XLS_ASSERT_OK_AND_ASSIGN(Function * func, ParseFunction(R"(
fn id(x: bits[16], y: bits[32]) -> bits[16] {
  ret param.1: bits[16] = param(name=x)
}
)",
                                                          &p));

  EXPECT_THAT(
      func->MakeNode<NaryOp>(
          FindNode("x", &p)->loc(),
          std::vector<Node*>{FindNode("x", &p), FindNode("y", &p)}, Op::kXor),
      StatusIs(absl::StatusCode::kInternal,
               HasSubstr("Type of operand 1 (bits[32] via y) does not "
                         "match type of xor")));
}

}  // namespace
}  // namespace xls
