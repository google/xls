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

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "xls/common/status/matchers.h"
#include "xls/ir/ir_matcher.h"
#include "xls/ir/ir_test_base.h"

namespace xls {
namespace {

using status_testing::StatusIs;
using ::testing::HasSubstr;

class VerifierTest : public IrTestBase {
 protected:
  VerifierTest() = default;
};

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

proc my_proc(t: token, s: bits[42], init={45}) {
  next (t, s)
}

)";
  XLS_ASSERT_OK_AND_ASSIGN(auto p, ParsePackageNoVerify(input));
  XLS_ASSERT_OK(VerifyPackage(p.get()));
  XLS_ASSERT_OK(VerifyProc(FindProc("my_proc", p.get())));
}

TEST_F(VerifierTest, ProcMissingReceive) {
  std::string input = R"(
package test_package

chan ch(bits[32], id=42, kind=streaming, ops=send_receive, flow_control=none, metadata="""module_port { flopped: true }""")

proc my_proc(t: token, s: bits[32], init={45}) {
  send.1: token = send(t, s, channel_id=42)
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

TEST_F(VerifierTest, MultipleSendNodes) {
  std::string input = R"(
package test_package

chan ch(bits[32], id=42, kind=streaming, ops=send_only, flow_control=none, metadata="""module_port { flopped: true }""")

proc my_proc(t: token, s: bits[32], init={45}) {
  send.1: token = send(t, s, channel_id=42)
  send.2: token = send(send.1, s, channel_id=42)
  next (send.2, s)
}

)";
  XLS_ASSERT_OK_AND_ASSIGN(auto p, ParsePackageNoVerify(input));
  EXPECT_THAT(
      VerifyPackage(p.get(), /*codegen=*/true),
      StatusIs(
          absl::StatusCode::kInternal,
          HasSubstr("Multiple sends associated with the same channel 'ch'")));
}

TEST_F(VerifierTest, DisconnectedSendNode) {
  std::string input = R"(
package test_package

chan ch(bits[32], id=42, kind=streaming, ops=send_only, flow_control=none, metadata="""module_port { flopped: true }""")

proc my_proc(t: token, s: bits[32], init={45}) {
  after_all.1: token = after_all()
  send.2: token = send(after_all.1, s, channel_id=42)
  next (send.2, s)
}

)";
  XLS_ASSERT_OK_AND_ASSIGN(auto p, ParsePackageNoVerify(input));
  EXPECT_THAT(
      VerifyPackage(p.get()),
      StatusIs(
          absl::StatusCode::kInternal,
          HasSubstr("Side-effecting token-typed nodes must be connected to the "
                    "sink token value via a path of tokens")));
}

TEST_F(VerifierTest, DisconnectedReceiveNode) {
  std::string input = R"(
package test_package

chan ch(bits[32], id=42, kind=streaming, ops=receive_only, flow_control=none, metadata="""module_port { flopped: true }""")

proc my_proc(t: token, s: bits[42], init={45}) {
  receive.1: (token, bits[32]) = receive(t, channel_id=42)
  next (t, s)
}

)";
  XLS_ASSERT_OK_AND_ASSIGN(auto p, ParsePackageNoVerify(input));
  EXPECT_THAT(
      VerifyPackage(p.get()),
      StatusIs(
          absl::StatusCode::kInternal,
          HasSubstr("Side-effecting token-typed nodes must be connected to the "
                    "sink token value via a path of tokens")));
}

TEST_F(VerifierTest, DisconnectedReturnValueInProc) {
  std::string input = R"(
package test_package

proc my_proc(t: token, s: bits[42], init={45}) {
  after_all.1: token = after_all()
  next (after_all.1, s)
}

)";
  XLS_ASSERT_OK_AND_ASSIGN(auto p, ParsePackageNoVerify(input));
  EXPECT_THAT(
      VerifyPackage(p.get()),
      StatusIs(absl::StatusCode::kInternal,
               HasSubstr("Side-effecting token-typed nodes must be connected "
                         "to the sink token value via a path of tokens")));
}

TEST_F(VerifierTest, DisconnectedNonSideEffectingTokenOperation) {
  std::string input = R"(
package test_package

proc my_proc(t: token, s: bits[42], init={45}) {
  token_tuple: (token) = tuple(t)
  next (t, s)
}

)";
  XLS_ASSERT_OK_AND_ASSIGN(auto p, ParsePackageNoVerify(input));
  XLS_EXPECT_OK(VerifyPackage(p.get()));
}

TEST_F(VerifierTest, SendOnReceiveOnlyChannel) {
  std::string input = R"(
package test_package

chan ch(bits[32], id=42, kind=streaming, ops=receive_only, flow_control=none, metadata="""module_port { flopped: true }""")

proc my_proc(t: token, s: bits[42], init={45}) {
  send.1: token = send(t, s, channel_id=42)
  receive.2: (token, bits[32]) = receive(send.1, channel_id=42)
  tuple_index.3: token = tuple_index(receive.2, index=0)
  next (tuple_index.3, s)
}

)";
  XLS_ASSERT_OK_AND_ASSIGN(auto p, ParsePackageNoVerify(input));
  EXPECT_THAT(
      VerifyPackage(p.get()),
      StatusIs(
          absl::StatusCode::kInternal,
          HasSubstr(
              "Cannot send over channel ch (42), send operation: send.1")));
}

TEST_F(VerifierTest, DynamicCountedForBodyParamterCountMismatch) {
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
  EXPECT_THAT(VerifyPackage(p.get()),
              StatusIs(absl::StatusCode::kInternal,
                       HasSubstr("Parameter 1 (accumulator) of function body "
                                 "used as dynamic_counted_for body should have "
                                 "bits[32] type, got bits[128] instead")));
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
                    "dynamic_counted_for body should have bits[128] type from "
                    "invariant_2: bits[128] = param(invariant_2, id=7), got "
                    "bits[64] instead")));
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

}  // namespace
}  // namespace xls
