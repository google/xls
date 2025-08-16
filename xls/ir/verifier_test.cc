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

#include "xls/ir/verifier.h"

#include <memory>
#include <string>
#include <string_view>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/strings/substitute.h"
#include "absl/types/span.h"
#include "xls/common/status/matchers.h"
#include "xls/ir/bits.h"
#include "xls/ir/channel.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/nodes.h"
#include "xls/ir/package.h"
#include "xls/ir/source_location.h"
#include "xls/ir/value.h"

namespace xls {
namespace {

using ::absl_testing::StatusIs;
using ::testing::HasSubstr;

class VerifierTest : public IrTestBase {
 protected:
  VerifierTest() = default;
};

TEST_F(VerifierTest, ArrayIndexOfEmptyArray) {
  std::string input = R"(
package p

fn f(a: bits[8][0], i: bits[3]) -> bits[8] {
  ret array_index.1: bits[8] = array_index(a, indices=[i])
}
)";
  XLS_ASSERT_OK_AND_ASSIGN(auto p, ParsePackageNoVerify(input));
  EXPECT_THAT(
      VerifyPackage(p.get()),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("Array index cannot be applied to an empty array")));
}

TEST_F(VerifierTest, BadSelect) {
  static constexpr std::string_view input = R"(
package subrosa

top fn function_0() -> bits[8][8] {
  name: bits[1] = literal(value=0, id=1)
  name__1: bits[8][8] = literal(value=[0, 0, 0, 0, 0, 0, 0, 0], id=2)
  name__2: bits[64] = literal(value=0x0, id=3)
  ret name__3: bits[8][8] = sel(name, cases=[name__1], default=name__2, id=4)
}
)";
  XLS_ASSERT_OK_AND_ASSIGN(auto p, ParsePackageNoVerify(input));
  ASSERT_THAT(VerifyPackage(p.get()),
              absl_testing::StatusIs(absl::StatusCode::kInternal,
                                     HasSubstr("does not match node type")));
}

TEST_F(VerifierTest, PrioritySelectWithDifferentCaseTypes) {
  std::string input = R"(
package p

fn f(s: bits[2], a: bits[2], b: bits[3]) -> bits[3] {
  ret priority_sel.1: bits[3] = priority_sel(s, cases=[a, b], default=b)
}
)";
  XLS_ASSERT_OK_AND_ASSIGN(auto p, ParsePackageNoVerify(input));
  EXPECT_THAT(
      VerifyPackage(p.get()),
      StatusIs(
          absl::StatusCode::kInternal,
          HasSubstr("All cases in priority select must have the same type")));
}

TEST_F(VerifierTest, PrioritySelectWithDifferentDefaultType) {
  auto p = IrTestBase::CreatePackage();
  FunctionBuilder fb(IrTestBase::TestName(), p.get());
  BValue selector = fb.Param("p0", p->GetBitsType(2));
  BValue case1 = fb.Param("p1", p->GetBitsType(10));
  BValue default_value = fb.Param("p2", p->GetBitsType(20));
  fb.PrioritySelect(selector, {case1, case1}, default_value);
  ASSERT_THAT(fb.Build(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Default value must have the same type")));
}

TEST_F(VerifierTest, OneHotSelectWithDifferentCaseTypes) {
  std::string input = R"(
package p

fn f(s: bits[2], a: bits[2], b: bits[3]) -> bits[2] {
  ret one_hot_sel.1: bits[2] = one_hot_sel(s, cases=[a, b])
}
)";
  XLS_ASSERT_OK_AND_ASSIGN(auto p, ParsePackageNoVerify(input));
  EXPECT_THAT(
      VerifyPackage(p.get()),
      StatusIs(
          absl::StatusCode::kInternal,
          HasSubstr("All cases in one-hot select must have the same type")));
}

TEST_F(VerifierTest, WellFormedPackage) {
  std::string input = R"(
package WellFormedPackage

fn graph(p: bits[42], q: bits[42]) -> bits[42] {
  and.1: bits[42] = and(p, q)
  add.2: bits[42] = add(and.1, q)
  ret sub.3: bits[42] = sub(add.2, add.2)
}

fn graph2(a: bits[16]) -> bits[16] {
  neg.4: bits[16] = neg(a)
  ret not.5: bits[16] = not(neg.4)
}
)";
  XLS_ASSERT_OK_AND_ASSIGN(auto p, ParsePackageNoVerify(input));
  XLS_ASSERT_OK(VerifyPackage(p.get()));
  XLS_ASSERT_OK(VerifyFunction(FindFunction("graph", p.get())));
  XLS_ASSERT_OK(VerifyFunction(FindFunction("graph2", p.get())));
}

TEST_F(VerifierTest, NonUniqueNodeId) {
  std::string input = R"(
package NonUniqueNodeId

fn graph(p: bits[42], q: bits[42]) -> bits[42] {
  and.1: bits[42] = and(p, q)
  add.2: bits[42] = add(and.1, q)
  ret sub.2: bits[42] = sub(add.2, add.2)
}
)";
  XLS_ASSERT_OK_AND_ASSIGN(auto p, ParsePackageNoVerify(input));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, p->GetFunction("graph"));
  EXPECT_THAT(VerifyFunction(f), StatusIs(absl::StatusCode::kInternal,
                                          HasSubstr("ID 2 is not unique")));
}

TEST_F(VerifierTest, KeywordAsFunctionName) {
  auto p = std::make_unique<Package>("KeywordAsFunctionName");
  FunctionBuilder fb("top", p.get(), /*should_verify=*/false);
  Type* u42 = p->GetBitsType(42);
  BValue a = fb.Param("a", u42);
  BValue b = fb.Param("b", u42);
  BValue add = fb.Add(fb.And(absl::MakeConstSpan({a, b})), b);
  fb.Subtract(add, add);
  XLS_ASSERT_OK(fb.Build().status());

  EXPECT_THAT(
      VerifyPackage(p.get()),
      StatusIs(absl::StatusCode::kInternal,
               HasSubstr("Function/proc/block name 'top' is a keyword")));
}

TEST_F(VerifierTest, NonUniqueFunctionName) {
  std::string input = R"(
package NonUniqueFunctionName

fn graph(p: bits[42], q: bits[42]) -> bits[42] {
  and.1: bits[42] = and(p, q)
  add.2: bits[42] = add(and.1, q)
  ret sub.3: bits[42] = sub(add.2, add.2)
}

fn graph(a: bits[16]) -> bits[16] {
  neg.4: bits[16] = neg(a)
  ret not.5: bits[16] = not(neg.4)
}
)";
  XLS_ASSERT_OK_AND_ASSIGN(auto p, ParsePackageNoVerify(input));
  EXPECT_THAT(
      VerifyPackage(p.get()),
      StatusIs(absl::StatusCode::kInternal,
               HasSubstr("Function/proc/block with name graph is not unique")));
}

TEST_F(VerifierTest, NonUniqueFunctionAndBlockName) {
  std::string input = R"(
package NonUniqueFunctionName

fn graph(p: bits[42], q: bits[42]) -> bits[42] {
  and.1: bits[42] = and(p, q)
  add.2: bits[42] = add(and.1, q)
  ret sub.3: bits[42] = sub(add.2, add.2)
}

block graph() {}
)";
  XLS_ASSERT_OK_AND_ASSIGN(auto p, ParsePackageNoVerify(input));
  XLS_ASSERT_OK(VerifyPackage(p.get()));
}

TEST_F(VerifierTest, BinOpOperandTypeMismatch) {
  std::string input = R"(
package BinOpOperandTypeMismatch

fn graph(p: bits[2], q: bits[42], r: bits[42]) -> bits[42] {
  ret and.1: bits[42] = and(q, r)
}
)";
  XLS_ASSERT_OK_AND_ASSIGN(auto p, ParsePackageNoVerify(input));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, p->GetFunction("graph"));
  // Replace lhs of the 'and' with a different bit-width value.
  FindNode("and.1", f)->ReplaceOperand(FindNode("q", f), FindNode("p", f));
  EXPECT_THAT(VerifyFunction(f),
              StatusIs(absl::StatusCode::kInternal,
                       HasSubstr("Expected operand 0 of and.1 to have type "
                                 "bits[42], has type bits[2].")));
  EXPECT_THAT(VerifyPackage(p.get()),
              StatusIs(absl::StatusCode::kInternal,
                       HasSubstr("Expected operand 0 of and.1 to have type "
                                 "bits[42], has type bits[2].")));
}

TEST_F(VerifierTest, SelectWithUselessDefault) {
  std::string input = R"(
package p

fn f(p: bits[1], q: bits[42], r: bits[42]) -> bits[42] {
  literal.1: bits[42] = literal(value=42)
  ret sel.2: bits[42] = sel(p, cases=[q, r], default=literal.1)
}
)";
  XLS_ASSERT_OK_AND_ASSIGN(auto p, ParsePackageNoVerify(input));
  EXPECT_THAT(VerifyPackage(p.get()),
              StatusIs(absl::StatusCode::kInternal,
                       HasSubstr("Select has useless default value")));
}

TEST_F(VerifierTest, SelectWithMissingDefault) {
  std::string input = R"(
package p

fn f(p: bits[2], q: bits[42], r: bits[42], s:bits[42]) -> bits[42] {
  ret sel.1: bits[42] = sel(p, cases=[q, r, s])
}
)";
  XLS_ASSERT_OK_AND_ASSIGN(auto p, ParsePackageNoVerify(input));
  EXPECT_THAT(VerifyPackage(p.get()),
              StatusIs(absl::StatusCode::kInternal,
                       HasSubstr("Select has no default value")));
}

TEST_F(VerifierTest, EmptyConcatIsOk) {
  std::string input = R"(
package p

fn f() -> bits[0] {
  ret concat.1: bits[0] = concat()
}
)";
  XLS_ASSERT_OK_AND_ASSIGN(auto p, ParsePackageNoVerify(input));
  XLS_EXPECT_OK(VerifyPackage(p.get()));
}

TEST_F(VerifierTest, NumericCompareArrayOperands) {
  std::string input = R"(
package p

fn f(a: bits[32][4], b: bits[32][4]) -> bits[1] {
  ret ult.1: bits[1] = ult(a, b)
}
)";
  XLS_ASSERT_OK_AND_ASSIGN(auto p, ParsePackageNoVerify(input));
  EXPECT_THAT(VerifyPackage(p.get()),
              StatusIs(absl::StatusCode::kInternal,
                       HasSubstr("Expected operand 0 of ult.1 to have Bits "
                                 "type, has type bits[32][4]")));
}

TEST_F(VerifierTest, NumericCompareTupleOperands) {
  std::string input = R"(
package p

fn f(a: (bits[32], bits[16]), b: (bits[32], bits[16])) -> bits[1] {
  ret sge.1: bits[1] = sge(a, b)
}
)";
  XLS_ASSERT_OK_AND_ASSIGN(auto p, ParsePackageNoVerify(input));
  EXPECT_THAT(VerifyPackage(p.get()),
              StatusIs(absl::StatusCode::kInternal,
                       HasSubstr("Expected operand 0 of sge.1 to have Bits "
                                 "type, has type (bits[32], bits[16])")));
}

TEST_F(VerifierTest, NumericCompareMismatchedBitWidths) {
  std::string input = R"(
package p

fn f(a: bits[32], b: bits[16]) -> bits[1] {
  ret ugt.1: bits[1] = ugt(a, b)
}
)";
  XLS_ASSERT_OK_AND_ASSIGN(auto p, ParsePackageNoVerify(input));
  EXPECT_THAT(
      VerifyPackage(p.get()),
      StatusIs(absl::StatusCode::kInternal,
               HasSubstr("Expected operand 1 of ugt.1 to have bit "
                         "count 32 (same as operand 0), has bit count 16")));
}

TEST_F(VerifierTest, EqCompareTokenOperands) {
  std::string input = R"(
package p

fn f() -> bits[1] {
  tok_a: token = literal(value=token, id=1)
  tok_b: token = literal(value=token, id=2)
  ret eq_val: bits[1] = eq(tok_a, tok_b, id=3)
}
)";
  XLS_ASSERT_OK_AND_ASSIGN(auto p, ParsePackageNoVerify(input));
  EXPECT_THAT(VerifyPackage(p.get()),
              StatusIs(absl::StatusCode::kInternal,
                       HasSubstr("Operand 0 of eq_val cannot be Token type "
                                 "for this operation")));
}

TEST_F(VerifierTest, NeCompareTokenOperands) {
  std::string input = R"(
package p

fn f() -> bits[1] {
  tok_a: token = literal(value=token, id=1)
  tok_b: token = literal(value=token, id=2)
  ret ne_val: bits[1] = ne(tok_a, tok_b, id=3)
}
)";
  XLS_ASSERT_OK_AND_ASSIGN(auto p, ParsePackageNoVerify(input));
  EXPECT_THAT(VerifyPackage(p.get()),
              StatusIs(absl::StatusCode::kInternal,
                       HasSubstr("Operand 0 of ne_val cannot be Token type "
                                 "for this operation")));
}

TEST_F(VerifierTest, SelectWithTooNarrowSelector) {
  std::string input = R"(
package p

fn f(p: bits[1], q: bits[42], r: bits[42], s:bits[42], t:bits[42]) -> bits[42] {
  ret sel.1: bits[42] = sel(p, cases=[q, r, s, t])
}
)";
  XLS_ASSERT_OK_AND_ASSIGN(auto p, ParsePackageNoVerify(input));
  EXPECT_THAT(
      VerifyPackage(p.get()),
      StatusIs(
          absl::StatusCode::kInternal,
          HasSubstr(
              "Selector must have at least 2 bits to select amongst 4 cases")));
}

TEST_F(VerifierTest, WellFormedProc) {
  std::string input = R"(
package test_package

proc my_proc(s: bits[42], init={45}) {
  next (s)
}

)";
  XLS_ASSERT_OK_AND_ASSIGN(auto p, ParsePackageNoVerify(input));
  XLS_ASSERT_OK(VerifyPackage(p.get()));
  XLS_ASSERT_OK(VerifyProc(FindProc("my_proc", p.get())));
}

TEST_F(VerifierTest, ProcMissingReceive) {
  std::string input = R"(
package test_package

chan ch(bits[32], id=42, kind=streaming, ops=send_receive, flow_control=none)

proc my_proc(t: token, s: bits[32], init={token, 45}) {
  send.1: token = send(t, s, channel=ch)
  next (send.1, s)
}

)";
  XLS_ASSERT_OK_AND_ASSIGN(auto p, ParsePackageNoVerify(input));
  EXPECT_THAT(
      VerifyPackage(p.get(), /*codegen=*/true),
      StatusIs(
          absl::StatusCode::kInternal,
          HasSubstr("Channel 'ch' (id 42) has no associated receive node")));
}

TEST_F(VerifierTest, SendOnReceiveOnlyChannel) {
  std::string input = R"(
package test_package

chan ch(bits[32], id=42, kind=streaming, ops=receive_only, flow_control=none)

proc my_proc(t: token, s: bits[42], init={token, 45}) {
  send.1: token = send(t, s, channel=ch)
  receive.2: (token, bits[32]) = receive(send.1, channel=ch)
  tuple_index.3: token = tuple_index(receive.2, index=0)
  next (tuple_index.3, s)
}

)";
  XLS_ASSERT_OK_AND_ASSIGN(auto p, ParsePackageNoVerify(input));
  EXPECT_THAT(
      VerifyPackage(p.get()),
      StatusIs(
          absl::StatusCode::kInternal,
          HasSubstr("Cannot send over channel ch, send operation: send.1")));
}

TEST_F(VerifierTest, DynamicCountedForBodyParameterCountMismatch) {
  std::string input = R"(
package p

fn body(index: bits[32], accumulator: bits[32], invariant_1: bits[48], invariant_2: bits[64], invariant_3: bits[64]) -> bits[32] {
  ret add.5: bits[32] = add(index, accumulator, id=5)
}

fn main(invariant_1: bits[48], invariant_2: bits[64], stride: bits[16], trip_count: bits[16], init: bits[32]) -> bits[32] {
  ret dynamic_counted_for.11: bits[32] = dynamic_counted_for(init, trip_count, stride, body=body, invariant_args=[invariant_1, invariant_2], id=11)
}
)";
  XLS_ASSERT_OK_AND_ASSIGN(auto p, ParsePackageNoVerify(input));
  EXPECT_THAT(
      VerifyPackage(p.get()),
      StatusIs(absl::StatusCode::kInternal,
               HasSubstr("Function body used as dynamic_counted_for body "
                         "should have 4 parameters, got 5 instead")));
}

TEST_F(VerifierTest, DynamicCountedForBodyIndexParameterNotBits) {
  std::string input = R"(
package p

fn body(index: bits[32][2], accumulator: bits[32], invariant_1: bits[48], invariant_2: bits[64]) -> bits[32] {
  ret add.5: bits[32] = add(accumulator, accumulator, id=5)
}

fn main(invariant_1: bits[48], invariant_2: bits[64], stride: bits[16], trip_count: bits[16], init: bits[32]) -> bits[32] {
  ret dynamic_counted_for.11: bits[32] = dynamic_counted_for(init, trip_count, stride, body=body, invariant_args=[invariant_1, invariant_2], id=11)
}
)";
  XLS_ASSERT_OK_AND_ASSIGN(auto p, ParsePackageNoVerify(input));
  EXPECT_THAT(
      VerifyPackage(p.get()),
      StatusIs(absl::StatusCode::kInternal,
               HasSubstr("Parameter 0 (index) of function body used as "
                         "dynamic_counted_for body should have bits type")));
}

TEST_F(VerifierTest,
       DynamicCountedForBodyLoopFunctionTypeDoesNotMatchDynamicCountedForType) {
  std::string input = R"(
package p

fn body(index: bits[32], accumulator: bits[32], invariant_1: bits[48], invariant_2: bits[64]) -> bits[32] {
  ret add.5: bits[32] = add(index, accumulator, id=5)
}

fn main(invariant_1: bits[48], invariant_2: bits[64], stride: bits[16], trip_count: bits[16], init: bits[32]) -> bits[32] {
  ret dynamic_counted_for.11: bits[32] = dynamic_counted_for(init, trip_count, stride, body=body, invariant_args=[invariant_1, invariant_2], id=11)
}
)";
  XLS_ASSERT_OK_AND_ASSIGN(auto p, ParsePackageNoVerify(input));
  XLS_ASSERT_OK_AND_ASSIGN(Function * main, p->GetFunction("main"));
  Node* main_invariant_1 = FindNode("invariant_1", main);
  Node* main_invariant_2 = FindNode("invariant_2", main);
  Node* main_stride = FindNode("stride", main);
  Node* main_trip_count = FindNode("trip_count", main);
  Node* main_original_for = FindNode("dynamic_counted_for.11", main);
  EXPECT_THAT(
      main_original_for->Clone({main_invariant_1, main_trip_count, main_stride,
                                main_invariant_1, main_invariant_2}),
      StatusIs(
          absl::StatusCode::kInternal,
          HasSubstr("Return type of function body used as dynamic_counted_for "
                    "body should have bits[48] type, got bits[32] instead")));
}

TEST_F(VerifierTest,
       DynamicCountedForBodyAccumulatorTypeDoesNotMatchDynamicCountedForType) {
  std::string input = R"(
package p

fn body(index: bits[32], accumulator: bits[128], invariant_1: bits[48], invariant_2: bits[64]) -> bits[32] {
  ret add.5: bits[32] = add(index, index, id=5)
}

fn main(invariant_1: bits[48], invariant_2: bits[64], stride: bits[16], trip_count: bits[16], init: bits[32]) -> bits[32] {
  ret dynamic_counted_for.11: bits[32] = dynamic_counted_for(init, trip_count, stride, body=body, invariant_args=[invariant_1, invariant_2], id=11)
}
)";
  XLS_ASSERT_OK_AND_ASSIGN(auto p, ParsePackageNoVerify(input));
  EXPECT_THAT(
      VerifyPackage(p.get()),
      StatusIs(absl::StatusCode::kInternal,
               HasSubstr("Parameter 1 (accumulator) of function body used as "
                         "dynamic_counted_for body should have bits[32] type, "
                         "got bits[128] instead")));
}

TEST_F(VerifierTest, DynamicCountedForInvariantDoesNotMatchBodyParamType) {
  std::string input = R"(
package p

fn body(index: bits[32], accumulator: bits[32], invariant_1: bits[48], invariant_2: bits[64]) -> bits[32] {
  ret add.5: bits[32] = add(index, accumulator, id=5)
}

fn main(invariant_1: bits[48], invariant_2: bits[128], stride: bits[16], trip_count: bits[16], init: bits[32]) -> bits[32] {
  ret dynamic_counted_for.11: bits[32] = dynamic_counted_for(init, trip_count, stride, body=body, invariant_args=[invariant_1, invariant_2], id=11)
}
)";
  XLS_ASSERT_OK_AND_ASSIGN(auto p, ParsePackageNoVerify(input));
  EXPECT_THAT(
      VerifyPackage(p.get()),
      StatusIs(
          absl::StatusCode::kInternal,
          HasSubstr("Parameter 3 (invariant_2) of function body used as "
                    "dynamic_counted_for body should have bits[128] type")));
}

TEST_F(VerifierTest, DynamicCountedForTripCountNotBits) {
  std::string input = R"(
package p

fn body(index: bits[32], accumulator: bits[32], invariant_1: bits[48], invariant_2: bits[64]) -> bits[32] {
  ret add.5: bits[32] = add(index, accumulator, id=5)
}

fn main(invariant_1: bits[48], invariant_2: bits[64], stride: bits[16], trip_count: bits[16][2], init: bits[32]) -> bits[32] {
  ret dynamic_counted_for.11: bits[32] = dynamic_counted_for(init, trip_count, stride, body=body, invariant_args=[invariant_1, invariant_2], id=11)
}
)";
  XLS_ASSERT_OK_AND_ASSIGN(auto p, ParsePackageNoVerify(input));
  EXPECT_THAT(
      VerifyPackage(p.get()),
      StatusIs(absl::StatusCode::kInternal,
               HasSubstr("Operand 1 / trip_count of dynamic_counted_for should "
                         "have bits type.")));
}

TEST_F(VerifierTest, DynamicCountedForStrideNotBits) {
  std::string input = R"(
package p

fn body(index: bits[32], accumulator: bits[32], invariant_1: bits[48], invariant_2: bits[64]) -> bits[32] {
  ret add.5: bits[32] = add(index, accumulator, id=5)
}

fn main(invariant_1: bits[48], invariant_2: bits[64], stride: bits[16][2], trip_count: bits[16], init: bits[32]) -> bits[32] {
  ret dynamic_counted_for.11: bits[32] = dynamic_counted_for(init, trip_count, stride, body=body, invariant_args=[invariant_1, invariant_2], id=11)
}
)";
  XLS_ASSERT_OK_AND_ASSIGN(auto p, ParsePackageNoVerify(input));
  EXPECT_THAT(VerifyPackage(p.get()),
              StatusIs(absl::StatusCode::kInternal,
                       HasSubstr("Operand 2 / stride of dynamic_counted_for "
                                 "should have bits type.")));
}

TEST_F(VerifierTest, DynamicCountedForTripCountTooManyBits) {
  std::string input = R"(
package p

fn body(index: bits[32], accumulator: bits[32], invariant_1: bits[48], invariant_2: bits[64]) -> bits[32] {
  ret add.5: bits[32] = add(index, accumulator, id=5)
}

fn main(invariant_1: bits[48], invariant_2: bits[64], stride: bits[16], trip_count: bits[32], init: bits[32]) -> bits[32] {
  ret dynamic_counted_for.11: bits[32] = dynamic_counted_for(init, trip_count, stride, body=body, invariant_args=[invariant_1, invariant_2], id=11)
}
)";
  XLS_ASSERT_OK_AND_ASSIGN(auto p, ParsePackageNoVerify(input));
  EXPECT_THAT(
      VerifyPackage(p.get()),
      StatusIs(absl::StatusCode::kInternal,
               HasSubstr("Operand 1 / trip_count of dynamic_counted_for should "
                         "have < the number of bits of the function body index "
                         "parameter / function body Operand 0")));
}

TEST_F(VerifierTest, DynamicCountedForStrideTooManyBits) {
  std::string input = R"(
package p

fn body(index: bits[32], accumulator: bits[32], invariant_1: bits[48], invariant_2: bits[64]) -> bits[32] {
  ret add.5: bits[32] = add(index, accumulator, id=5)
}

fn main(invariant_1: bits[48], invariant_2: bits[64], stride: bits[33], trip_count: bits[16], init: bits[32]) -> bits[32] {
  ret dynamic_counted_for.11: bits[32] = dynamic_counted_for(init, trip_count, stride, body=body, invariant_args=[invariant_1, invariant_2], id=11)
}
)";
  XLS_ASSERT_OK_AND_ASSIGN(auto p, ParsePackageNoVerify(input));
  EXPECT_THAT(
      VerifyPackage(p.get()),
      StatusIs(absl::StatusCode::kInternal,
               HasSubstr(
                   "Operand 2 / stride of dynamic_counted_for should have <= "
                   "the number of bits of the function body index parameter")));
}

TEST_F(VerifierTest, SimpleBlock) {
  std::string input = R"(
package test_package

block my_block(a: bits[32], b: bits[32], out: bits[32]) {
  a: bits[32] = input_port(name=a)
  b: bits[32] = input_port(name=b)
  sum: bits[32] = add(a, b)
  out: () = output_port(sum, name=out)
}

)";
  XLS_ASSERT_OK_AND_ASSIGN(auto p, ParsePackageNoVerify(input));
  XLS_ASSERT_OK(VerifyPackage(p.get()));
  XLS_ASSERT_OK(VerifyBlock(FindBlock("my_block", p.get())));
}

TEST_F(VerifierTest, BlockWithFifoInstantiation) {
  constexpr std::string_view ir_text_fmt = R"(package test_package

block my_block(push_valid: bits[1], push_data: bits[1], push_ready: bits[1], out:bits[32]) {
  push_valid: bits[1] = input_port(name=push_valid)
  push_data: bits[1] = input_port(name=push_data)
  instantiation my_inst(data_type=bits[32], depth=$0, bypass=true, register_push_outputs=false, register_pop_outputs=false, kind=fifo)
  push_valid_inst_input: () = instantiation_input(push_valid, instantiation=my_inst, port_name=push_valid)
  push_data_inst_input: () = instantiation_input(push_data, instantiation=my_inst, port_name=push_data)
  push_ready_inst_output: bits[1] = instantiation_output(instantiation=my_inst, port_name=push_ready)
  pop_data_inst_output: bits[32] = instantiation_output(instantiation=my_inst, port_name=pop_data)
  push_ready_output_port: () = output_port(push_ready_inst_output, name=push_ready)
  out_output_port: () = output_port(pop_data_inst_output, name=out)
}
)";
  // Set fifo depth to 3.
  std::string valid_ir_text = absl::Substitute(ir_text_fmt, 3);
  XLS_ASSERT_OK_AND_ASSIGN(auto p, ParsePackageNoVerify(valid_ir_text));
  XLS_ASSERT_OK(VerifyPackage(p.get()));

  // depth < 0 is invalid.
  std::string invalid_ir_text = absl::Substitute(ir_text_fmt, -3);
  XLS_ASSERT_OK_AND_ASSIGN(p, ParsePackageNoVerify(invalid_ir_text));
  EXPECT_THAT(VerifyPackage(p.get()),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Expected fifo depth >= 0, got -3")));
}

TEST_F(VerifierTest, MismatchedChannelFlowControl) {
  Package package("p");
  Type* u32 = package.GetBitsType(32);

  Proc* subproc;
  {
    TokenlessProcBuilder pb(NewStyleProc(), "subproc", "tkn", &package);
    XLS_ASSERT_OK(pb.AddInputChannel("ch", u32));
    XLS_ASSERT_OK_AND_ASSIGN(subproc, pb.Build({}));
  }

  {
    TokenlessProcBuilder pb(NewStyleProc(), "the_proc", "tkn", &package);
    XLS_ASSERT_OK_AND_ASSIGN(ChannelWithInterfaces channel1_refs,
                             pb.AddChannel("ch1", u32));
    XLS_ASSERT_OK_AND_ASSIGN(ChannelWithInterfaces channel2_refs,
                             pb.AddChannel("ch2", u32));
    XLS_ASSERT_OK(pb.InstantiateProc("inst1", subproc,
                                     {channel1_refs.receive_interface}));
    XLS_ASSERT_OK(pb.InstantiateProc("inst2", subproc,
                                     {channel2_refs.receive_interface}));
    XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, pb.Build({}));
    XLS_ASSERT_OK(package.SetTop(proc));

    dynamic_cast<StreamingChannel*>(channel1_refs.channel)
        ->SetFlowControl(FlowControl::kReadyValid);
    dynamic_cast<StreamingChannel*>(channel2_refs.channel)
        ->SetFlowControl(FlowControl::kNone);
  }
  EXPECT_THAT(
      VerifyPackage(&package),
      StatusIs(absl::StatusCode::kInternal,
               HasSubstr("ChannelInterface `ch` in proc `subproc` bound to "
                         "channels with different flow control")));
}

TEST_F(VerifierTest, MismatchedInterfaceChannelFlowControl) {
  Package package("p");
  Type* u32 = package.GetBitsType(32);

  Proc* subproc;
  {
    TokenlessProcBuilder pb(NewStyleProc(), "subproc", "tkn", &package);
    XLS_ASSERT_OK(pb.AddInputChannel("ch", u32));
    XLS_ASSERT_OK_AND_ASSIGN(subproc, pb.Build({}));
  }

  {
    TokenlessProcBuilder pb(NewStyleProc(), "the_proc", "tkn", &package);
    XLS_ASSERT_OK_AND_ASSIGN(ReceiveChannelInterface * ch1_ref,
                             pb.AddInputChannel("ch1", u32));
    XLS_ASSERT_OK_AND_ASSIGN(ChannelWithInterfaces channel2_refs,
                             pb.AddChannel("ch2", u32));
    XLS_ASSERT_OK(pb.InstantiateProc("inst1", subproc, {ch1_ref}));
    XLS_ASSERT_OK(pb.InstantiateProc("inst2", subproc,
                                     {channel2_refs.receive_interface}));
    XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, pb.Build({}));
    XLS_ASSERT_OK(package.SetTop(proc));

    // The input `ch1` of `proc` has no associated channel, just a channel
    // reference, because `proc` is the top proc.
    // TODO(https://github.com/google/xls/issues/869): When a channel object is
    // provided for top proc interfaces, set the flow control.
    dynamic_cast<StreamingChannel*>(channel2_refs.channel)
        ->SetFlowControl(FlowControl::kNone);
  }
  EXPECT_THAT(
      VerifyPackage(&package),
      StatusIs(absl::StatusCode::kInternal,
               HasSubstr("ChannelInterface `ch` in proc `subproc` bound to "
                         "channels with different flow control")));
}

TEST_F(VerifierTest, NextNodeWithWrongType) {
  Package package("p");

  ProcBuilder pb("p", &package, /*should_verify=*/false);
  BValue s = pb.StateElement("s", Value(UBits(0, 32)));
  BValue pred = pb.UGt(s, pb.Literal(UBits(10, 32)));
  BValue lit0_width_1 = pb.Literal(UBits(0, 1));
  // Can't use pb.Next() because it checks the type without the verifier before
  // making the node.
  EXPECT_THAT(pb.function()->MakeNode<Next>(SourceInfo(), s.node(),
                                            // This shouldn't verify!
                                            lit0_width_1.node(), pred.node()),
              StatusIs(absl::StatusCode::kInternal,
                       AllOf(HasSubstr("to have type bits[32]"),
                             HasSubstr("has type bits[1]"))));
}

TEST_F(VerifierTest, NextNodeWithWrongTypePredicate) {
  Package package("p");

  ProcBuilder pb("p", &package, /*should_verify=*/false);
  BValue s = pb.StateElement("s", Value(UBits(0, 32)));
  BValue pred = pb.UGt(s, pb.Literal(UBits(10, 32)));
  // This shouldn't verify!
  BValue not_pred = pb.ZeroExtend(pb.Not(pred), 32);
  BValue lit0 = pb.Literal(UBits(0, 32));
  BValue s_plus_one = pb.Add(s, pb.Literal(UBits(1, 32)));
  // Can't use pb.Next() because it checks the type.
  XLS_ASSERT_OK(pb.function()
                    ->MakeNode<Next>(SourceInfo(), s.node(),
                                     // This shouldn't verify!
                                     lit0.node(), pred.node())
                    .status());
  EXPECT_THAT(pb.function()->MakeNode<Next>(SourceInfo(), s.node(),
                                            s_plus_one.node(), not_pred.node()),
              StatusIs(absl::StatusCode::kInternal,
                       AllOf(HasSubstr("to have bit count 1:"),
                             HasSubstr("had 32 bits"))));
}

TEST_F(VerifierTest, NewAndOldStyleProcs) {
  const std::string input = R"(package test

proc my_new_proc<>() {}
proc my_old_proc() {}

)";
  XLS_ASSERT_OK_AND_ASSIGN(auto p, ParsePackageNoVerify(input));
  EXPECT_THAT(
      VerifyPackage(p.get()),
      StatusIs(absl::StatusCode::kInternal,
               HasSubstr("Package has both new style procs (proc-scoped "
                         "channels) and old-style procs")));
}

TEST_F(VerifierTest, NewStyleProcAndGlobalChannels) {
  const std::string input = R"(package test

chan ch(bits[32], id=42, kind=streaming, ops=send_receive, flow_control=none)

proc my_new_proc<>() {}

)";
  XLS_ASSERT_OK_AND_ASSIGN(auto p, ParsePackageNoVerify(input));
  EXPECT_THAT(VerifyPackage(p.get()),
              StatusIs(absl::StatusCode::kInternal,
                       HasSubstr("Package has global channels and procs with "
                                 "proc-scoped channels")));
}

}  // namespace
}  // namespace xls
