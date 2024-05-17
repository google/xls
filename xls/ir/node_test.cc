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

#include "xls/ir/node.h"

#include <string_view>
#include <type_traits>
#include <vector>

#include "absl/status/status.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "xls/common/status/matchers.h"
#include "xls/ir/bits.h"
#include "xls/ir/channel_ops.h"
#include "xls/ir/function.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/ir_matcher.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/nodes.h"
#include "xls/ir/op.h"
#include "xls/ir/source_location.h"
#include "xls/ir/value.h"
#include "xls/ir/verifier.h"
#include "xls/ir/verify_node.h"

namespace m = ::xls::op_matchers;

namespace xls {
namespace {

using status_testing::IsOkAndHolds;
using status_testing::StatusIs;
using ::testing::HasSubstr;
using ::testing::UnorderedElementsAre;

class NodeTest : public IrTestBase {};

TEST_F(NodeTest, CloneInSameFunction) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * func, ParseFunction(R"(
fn add(x: bits[32], y: bits[32]) -> bits[32] {
ret add.3: bits[32] = add(x, y)
}
)",
                                                          p.get()));

  Node* add = func->return_value();
  ASSERT_EQ(add->op(), Op::kAdd);
  XLS_ASSERT_OK_AND_ASSIGN(Node * add_clone,
                           add->Clone({func->param(1), func->param(0)}));

  EXPECT_EQ(add_clone->operand(0), add->operand(1));
  EXPECT_EQ(add_clone->operand(1), add->operand(0));
}

TEST_F(NodeTest, CloneToNewFunction) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * func_from, ParseFunction(R"(
fn add_from(x: bits[32], y: bits[32]) -> bits[32] {
ret add.3: bits[32] = add(x, y)
}
)",
                                                               p.get()));

  XLS_ASSERT_OK_AND_ASSIGN(Function * func_to, ParseFunction(R"(
fn add_to(x: bits[32], y: bits[32]) -> bits[32] {
ret x: bits[32] = param(name=x)
}
)",
                                                             p.get()));

  Node* add = func_from->return_value();
  XLS_ASSERT_OK_AND_ASSIGN(
      Node * add_clone,
      add->CloneInNewFunction({func_to->param(1), func_to->param(0)}, func_to));
  EXPECT_EQ(add_clone->function_base(), func_to);
}

TEST_F(NodeTest, CloneWithOperandsInDifferentFunctions) {
  Package p(TestName());
  XLS_ASSERT_OK_AND_ASSIGN(Function * func_from, ParseFunction(R"(
fn add_from(x: bits[32], y: bits[32]) -> bits[32] {
ret add.3: bits[32] = add(x, y)
}
)",
                                                               &p));

  XLS_ASSERT_OK_AND_ASSIGN(Function * func_to, ParseFunction(R"(
fn add_to(x: bits[32], y: bits[32]) -> bits[32] {
ret x: bits[32] = param(name=x)
}
)",
                                                             &p));

  Node* add = func_from->return_value();
  EXPECT_THAT(
      add->CloneInNewFunction({func_from->param(0), func_to->param(0)},
                              func_to),
      StatusIs(absl::StatusCode::kInternal,
               HasSubstr("Operand x of node add.6 not in same function")));
}

TEST_F(NodeTest, CloneCountedFor) {
  XLS_ASSERT_OK_AND_ASSIGN(auto p, ParsePackage(R"(
package CountedFor

fn body(x: bits[11], y: bits[11]) -> bits[11] {
  ret add.3: bits[11] = add(x, y)
}

fn main() -> bits[11] {
  literal.4: bits[11] = literal(value=0)
  ret counted_for.5: bits[11] = counted_for(literal.4, trip_count=7, stride=1, body=body)
}
)"));
  Node* counted_for = FindNode("counted_for.5", p.get());
  XLS_ASSERT_OK_AND_ASSIGN(Node * counted_for_clone,
                           counted_for->Clone({counted_for->operand(0)}));
  EXPECT_EQ(counted_for_clone->As<CountedFor>()->body(),
            counted_for->As<CountedFor>()->body());
}

TEST_F(NodeTest, CloneLiteral) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * func, ParseFunction(R"(
fn add() -> bits[32] {
ret literal.1: bits[32] = literal(value=12345)
}
)",
                                                          p.get()));

  Node* literal = func->return_value();
  XLS_ASSERT_OK_AND_ASSIGN(Node * literal_clone, literal->Clone({}));

  EXPECT_EQ(literal_clone->As<Literal>()->value().bits(), UBits(12345, 32));
}

TEST_F(NodeTest, ReplaceOperand) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  auto literal0 = fb.Literal(UBits(123, 32));
  auto literal1 = fb.Literal(UBits(444, 32));
  auto add = fb.Add(literal0, literal0);
  XLS_ASSERT_OK(fb.BuildWithReturnValue(add).status());

  EXPECT_EQ(add.node()->operand(0), literal0.node());
  EXPECT_EQ(add.node()->operand(1), literal0.node());
  EXPECT_TRUE(literal0.node()->HasUser(add.node()));
  EXPECT_FALSE(literal1.node()->HasUser(add.node()));

  add.node()->ReplaceOperand(literal0.node(), literal1.node());

  EXPECT_EQ(add.node()->operand(0), literal1.node());
  EXPECT_EQ(add.node()->operand(1), literal1.node());
  EXPECT_FALSE(literal0.node()->HasUser(add.node()));
  EXPECT_TRUE(literal1.node()->HasUser(add.node()));
}

TEST_F(NodeTest, ReplaceOperandNumberButStillAUser) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  auto literal0 = fb.Literal(UBits(123, 32));
  auto literal1 = fb.Literal(UBits(444, 32));
  auto add = fb.Add(literal0, literal0);
  XLS_ASSERT_OK(fb.BuildWithReturnValue(add).status());

  EXPECT_EQ(add.node()->operand(0), literal0.node());
  EXPECT_EQ(add.node()->operand(1), literal0.node());
  EXPECT_TRUE(literal0.node()->HasUser(add.node()));
  EXPECT_FALSE(literal1.node()->HasUser(add.node()));

  XLS_ASSERT_OK(add.node()->ReplaceOperandNumber(0, literal1.node()));

  EXPECT_EQ(add.node()->operand(0), literal1.node());
  EXPECT_EQ(add.node()->operand(1), literal0.node());
  EXPECT_TRUE(literal0.node()->HasUser(add.node()));
  EXPECT_TRUE(literal1.node()->HasUser(add.node()));
}

TEST_F(NodeTest, ReplaceOperandNumberNoLongerUser) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  auto literal0 = fb.Literal(UBits(123, 32));
  auto literal1 = fb.Literal(UBits(444, 32));
  auto literal2 = fb.Literal(UBits(22, 32));
  auto add = fb.Add(literal0, literal1);
  XLS_ASSERT_OK(fb.BuildWithReturnValue(add).status());

  EXPECT_EQ(add.node()->operand(0), literal0.node());
  EXPECT_EQ(add.node()->operand(1), literal1.node());
  EXPECT_TRUE(literal0.node()->HasUser(add.node()));
  EXPECT_TRUE(literal1.node()->HasUser(add.node()));
  EXPECT_FALSE(literal2.node()->HasUser(add.node()));

  XLS_ASSERT_OK(add.node()->ReplaceOperandNumber(1, literal2.node()));

  EXPECT_EQ(add.node()->operand(0), literal0.node());
  EXPECT_EQ(add.node()->operand(1), literal2.node());
  EXPECT_TRUE(literal0.node()->HasUser(add.node()));
  EXPECT_FALSE(literal1.node()->HasUser(add.node()));
  EXPECT_TRUE(literal2.node()->HasUser(add.node()));
}

TEST_F(NodeTest, ReplaceOperandNumber) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  auto x = fb.Param("x", p->GetBitsType(32));
  fb.Param("y", p->GetBitsType(32));
  fb.Param("z", p->GetBitsType(8));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f,
                           fb.BuildWithReturnValue(fb.Shll(x, x)));
  XLS_ASSERT_OK(f->return_value()->ReplaceOperandNumber(1, FindNode("y", f)));
  ASSERT_THAT(f->return_value()->ReplaceOperandNumber(1, FindNode("z", f)),
              StatusIs(absl::StatusCode::kInternal));
  XLS_ASSERT_OK(f->return_value()->ReplaceOperandNumber(
      1, FindNode("z", f), /*type_must_match=*/false));
}

TEST_F(NodeTest, ReplaceSendChannel) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * ch0, p->CreateStreamingChannel("ch0", ChannelOps::kSendOnly,
                                               p->GetBitsType(32)));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * ch1, p->CreateStreamingChannel("ch1", ChannelOps::kSendOnly,
                                               p->GetBitsType(32)));
  ProcBuilder pb(TestName(), p.get());
  BValue tok = pb.StateElement("tok", Value::Token());
  BValue send_on_c1 =
      pb.StateElement("send_on_c1", Value(UBits(0, /*bit_count=*/1)));
  BValue send0_tok = pb.Send(ch0, tok, pb.Literal(UBits(123, 32)));
  BValue send1_tok =
      pb.SendIf(ch1, send0_tok, send_on_c1, pb.Literal(UBits(456, 32)));
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc,
                           pb.Build({send1_tok, pb.Not(send_on_c1)}));
  Send* send0 = send0_tok.node()->As<Send>();
  Send* send1 = send1_tok.node()->As<Send>();
  EXPECT_NE(send0->channel_name(), ch1->name());
  send0->ReplaceChannel(ch1->name());
  EXPECT_EQ(send0->channel_name(), ch1->name());
  XLS_EXPECT_OK(VerifyNode(send0));
  EXPECT_NE(send1->channel_name(), ch0->name());
  send1->ReplaceChannel(ch0->name());
  EXPECT_EQ(send1->channel_name(), ch0->name());
  XLS_EXPECT_OK(VerifyNode(send1));
  XLS_EXPECT_OK(VerifyProc(proc));
}

TEST_F(NodeTest, ReplaceReceiveChannel) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * ch0, p->CreateStreamingChannel("ch0", ChannelOps::kReceiveOnly,
                                               p->GetBitsType(32)));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * ch1, p->CreateStreamingChannel("ch1", ChannelOps::kReceiveOnly,
                                               p->GetBitsType(32)));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * ch2, p->CreateStreamingChannel("ch2", ChannelOps::kReceiveOnly,
                                               p->GetBitsType(32)));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * ch3, p->CreateStreamingChannel("ch3", ChannelOps::kReceiveOnly,
                                               p->GetBitsType(32)));
  ProcBuilder pb(TestName(), p.get());
  BValue tok = pb.StateElement("tok", Value::Token());
  BValue recv_on_c1 =
      pb.StateElement("recv_on_c1", Value(UBits(0, /*bit_count=*/1)));
  BValue recv_on_c2 =
      pb.StateElement("recv_on_c1", Value(UBits(1, /*bit_count=*/1)));
  BValue recv0 = pb.Receive(ch0, tok);
  BValue recv0_tok = pb.TupleIndex(recv0, /*idx=*/0);
  BValue recv1 = pb.ReceiveIf(ch1, recv0_tok, recv_on_c1);
  BValue recv1_tok = pb.TupleIndex(recv1, /*idx=*/0);
  BValue recv2 = pb.ReceiveIfNonBlocking(ch2, recv1_tok, recv_on_c2);
  BValue recv2_tok = pb.TupleIndex(recv2, /*idx=*/0);
  BValue recv3 = pb.ReceiveNonBlocking(ch3, recv2_tok);
  BValue recv3_tok = pb.TupleIndex(recv3, /*idx=*/0);
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, pb.Build({recv3_tok, pb.Not(recv_on_c1),
                                                  pb.Not(recv_on_c2)}));
  Receive* recv0_node = recv0.node()->As<Receive>();
  Receive* recv1_node = recv1.node()->As<Receive>();
  Receive* recv2_node = recv2.node()->As<Receive>();
  EXPECT_NE(recv0_node->channel_name(), ch1->name());
  recv0_node->ReplaceChannel(ch1->name());
  EXPECT_EQ(recv0_node->channel_name(), ch1->name());
  XLS_EXPECT_OK(VerifyNode(recv0_node));
  EXPECT_NE(recv1_node->channel_name(), ch2->name());
  recv1_node->ReplaceChannel(ch2->name());
  EXPECT_EQ(recv1_node->channel_name(), ch2->name());
  XLS_EXPECT_OK(VerifyNode(recv1_node));
  EXPECT_NE(recv2_node->channel_name(), ch0->name());
  recv2_node->ReplaceChannel(ch0->name());
  EXPECT_EQ(recv2_node->channel_name(), ch0->name());
  XLS_EXPECT_OK(VerifyNode(recv2_node));

  XLS_EXPECT_OK(VerifyProc(proc));
}

TEST_F(NodeTest, IsDefinitelyEqualTo) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
fn EqualityTest(x: bits[8], y: bits[8], z: bits[16]) -> bits[32] {
  and.1: bits[8] = and(x, y)
  and.2: bits[8] = and(y, x)
  xor.3: bits[8] = xor(x, y)

  sign_ext.4: bits[20] = sign_ext(x, new_bit_count=20)
  sign_ext.5: bits[20] = sign_ext(y, new_bit_count=20)
  sign_ext.6: bits[20] = sign_ext(z, new_bit_count=20)
  zero_ext.7: bits[20] = zero_ext(x, new_bit_count=20)

  concat.8: bits[32] = concat(x, y, z)
  concat.9: bits[32] = concat(y, x, z)
  concat.10: bits[32] = concat(x, z, y)

  literal.11: bits[32] = literal(value=12345)
  literal.12: bits[32] = literal(value=42)
  literal.13: bits[16] = literal(value=12345)

  literal.14: bits[16][3] = literal(value=[1,2,3])
  literal.15: bits[16][3] = literal(value=[1,2,3])
  literal.16: bits[16][3] = literal(value=[1,2,30])

  bit_slice.17: bits[3] = bit_slice(x, start=3, width=3)
  bit_slice.18: bits[3] = bit_slice(x, start=2, width=3)
  bit_slice.19: bits[3] = bit_slice(z, start=3, width=3)

  ret literal.20: bits[32] = literal(value=12345)
}
)",
                                                       p.get()));

  auto nodes_equal = [&](std::string_view a, std::string_view b) {
    return FindNode(a, f)->IsDefinitelyEqualTo(FindNode(b, f));
  };
  EXPECT_TRUE(nodes_equal("x", "x"));
  EXPECT_FALSE(nodes_equal("x", "y"));
  EXPECT_FALSE(nodes_equal("x", "z"));
  EXPECT_TRUE(nodes_equal("y", "y"));

  EXPECT_TRUE(nodes_equal("and.1", "and.2"));
  EXPECT_FALSE(nodes_equal("and.1", "xor.3"));

  EXPECT_TRUE(nodes_equal("sign_ext.4", "sign_ext.5"));
  EXPECT_FALSE(nodes_equal("sign_ext.4", "sign_ext.6"));
  EXPECT_FALSE(nodes_equal("sign_ext.4", "zero_ext.7"));

  EXPECT_TRUE(nodes_equal("concat.8", "concat.9"));
  EXPECT_FALSE(nodes_equal("concat.8", "concat.10"));

  EXPECT_TRUE(nodes_equal("literal.11", "literal.11"));
  EXPECT_FALSE(nodes_equal("literal.11", "literal.12"));
  EXPECT_FALSE(nodes_equal("literal.11", "literal.13"));

  EXPECT_TRUE(nodes_equal("literal.14", "literal.15"));
  EXPECT_FALSE(nodes_equal("literal.14", "literal.16"));

  EXPECT_FALSE(nodes_equal("bit_slice.17", "bit_slice.18"));
  EXPECT_FALSE(nodes_equal("bit_slice.17", "bit_slice.19"));
}

TEST_F(NodeTest, CountedForEqualTo) {
  XLS_ASSERT_OK_AND_ASSIGN(auto p, ParsePackage(R"(
package CountedFor

fn body1(x: bits[11], y: bits[11]) -> bits[11] {
  ret add.3: bits[11] = add(x, y)
}

fn same_as_body1(a: bits[11], b: bits[11]) -> bits[11] {
  ret add.102: bits[11] = add(a, b)
}

fn body2(x: bits[11], y: bits[11]) -> bits[11] {
  ret sub.4: bits[11] = sub(x, y)
}

fn nested_body1(x: bits[11], y: bits[11]) -> bits[11] {
  ret invoke.123: bits[11] = invoke(x, y, to_apply=body1)
}

fn same_as_nested_body1(x: bits[11], y: bits[11]) -> bits[11] {
  ret invoke.99: bits[11] = invoke(x, y, to_apply=body1)
}

fn also_same_as_nested_body1(x: bits[11], y: bits[11]) -> bits[11] {
  ret invoke.100: bits[11] = invoke(x, y, to_apply=same_as_body1)
}

fn nested_body2(x: bits[11], y: bits[11]) -> bits[11] {
  ret invoke.101: bits[11] = invoke(x, y, to_apply=body2)
}

fn main() -> bits[11] {
  literal.5: bits[11] = literal(value=0)
  counted_for.6: bits[11] = counted_for(literal.5, trip_count=7, stride=1, body=body1)
  counted_for.7: bits[11] = counted_for(literal.5, trip_count=7, stride=1, body=body1)
  counted_for.8: bits[11] = counted_for(literal.5, trip_count=7, stride=1, body=same_as_body1)
  counted_for.10: bits[11] = counted_for(literal.5, trip_count=7, stride=1, body=body2)
  counted_for.11: bits[11] = counted_for(literal.5, trip_count=9, stride=1, body=body1)
  counted_for.12: bits[11] = counted_for(literal.5, trip_count=7, stride=2, body=body1)

  counted_for.13: bits[11] = counted_for(literal.5, trip_count=7, stride=1, body=nested_body1)
  counted_for.14: bits[11] = counted_for(literal.5, trip_count=7, stride=1, body=same_as_nested_body1)
  counted_for.15: bits[11] = counted_for(literal.5, trip_count=7, stride=1, body=also_same_as_nested_body1)
  ret counted_for.16: bits[11] = counted_for(literal.5, trip_count=7, stride=1, body=nested_body2)
}
)"));

  auto nodes_equal = [&](std::string_view a, std::string_view b) {
    return FindNode(a, p.get())->IsDefinitelyEqualTo(FindNode(b, p.get()));
  };

  EXPECT_TRUE(nodes_equal("counted_for.6", "counted_for.6"));
  EXPECT_TRUE(nodes_equal("counted_for.6", "counted_for.7"));
  EXPECT_TRUE(nodes_equal("counted_for.6", "counted_for.8"));
  EXPECT_FALSE(nodes_equal("counted_for.6", "counted_for.10"));
  EXPECT_FALSE(nodes_equal("counted_for.6", "counted_for.11"));
  EXPECT_FALSE(nodes_equal("counted_for.6", "counted_for.12"));
  EXPECT_FALSE(nodes_equal("counted_for.6", "counted_for.13"));

  EXPECT_TRUE(nodes_equal("counted_for.13", "counted_for.14"));
  EXPECT_TRUE(nodes_equal("counted_for.13", "counted_for.15"));
  EXPECT_FALSE(nodes_equal("counted_for.13", "counted_for.16"));
}

TEST_F(NodeTest, ReplaceUses) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
fn ReplaceUses(x: bits[8], y: bits[8]) -> bits[8] {
  and.1: bits[8] = and(x, y)
  or.2: bits[8] = or(and.1, y)
  literal.4: bits[8] = literal(value=123)
  literal.5: bits[8] = literal(value=42)
  ret add.3: bits[8] = add(or.2, and.1)
}
)",
                                                       p.get()));
  EXPECT_THAT(FindNode("and.1", f)->users(),
              UnorderedElementsAre(FindNode("or.2", f), FindNode("add.3", f)));
  EXPECT_TRUE(FindNode("literal.4", f)->users().empty());

  XLS_ASSERT_OK(
      FindNode("and.1", f)->ReplaceUsesWith(FindNode("literal.4", f)));

  XLS_ASSERT_OK(
      FindNode("and.1", f)->ReplaceUsesWith(FindNode("literal.5", f)));

  EXPECT_THAT(FindNode("literal.4", f)->users(),
              UnorderedElementsAre(FindNode("or.2", f), FindNode("add.3", f)));
  EXPECT_TRUE(FindNode("and.1", f)->users().empty());
}

TEST_F(NodeTest, ReplaceUsesReturnValue) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
fn ReplaceUses(x: bits[8], y: bits[8]) -> bits[8] {
  and.1: bits[8] = and(x, y)
  ret and.2: bits[8] = and(y, x)
}
)",
                                                       p.get()));
  EXPECT_EQ(FindNode("and.2", f), f->return_value());
  XLS_ASSERT_OK(FindNode("and.2", f)->ReplaceUsesWith(FindNode("and.1", f)));
  EXPECT_EQ(FindNode("and.1", f), f->return_value());
}

TEST_F(NodeTest, ReplaceUsesWrongType) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
fn ReplaceUses(x: bits[8], y: bits[8]) -> bits[16] {
  and.1: bits[8] = and(x, y)
  ret concat.2: bits[16] = concat(x, y)
}
)",
                                                       p.get()));
  EXPECT_THAT(FindNode("concat.2", f)->ReplaceUsesWith(FindNode("and.1", f)),
              StatusIs(absl::StatusCode::kInternal));
}

TEST_F(NodeTest, ReplaceUsesWithNewNode) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
fn ReplaceUses(x: bits[8], y: bits[8]) -> bits[16] {
  and.1: bits[8] = and(x, y)
  ret concat.2: bits[16] = concat(x, y)
}
)",
                                                       p.get()));
  EXPECT_EQ(f->return_value()->op(), Op::kConcat);
  XLS_ASSERT_OK(FindNode("concat.2", f)
                    ->ReplaceUsesWithNew<Literal>(Value(UBits(123, 16))));
  EXPECT_EQ(f->return_value()->op(), Op::kLiteral);
}

TEST_F(NodeTest, ReplaceUsesWithInvalidNewNode) {
  Package p(TestName());
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
fn ReplaceUses(x: bits[8], y: bits[16]) -> bits[16] {
  and.1: bits[8] = and(x, x)
  ret concat.2: bits[16] = concat(x, x)
}
)",
                                                       &p));
  EXPECT_THAT(
      FindNode("and.1", f)
          ->ReplaceUsesWithNew<NaryOp>(
              std::vector<Node*>{FindNode("x", f), FindNode("y", f)}, Op::kXor),
      StatusIs(absl::StatusCode::kInternal,
               HasSubstr("Expected operand 1 of xor.5 to have type bits[8], "
                         "has type bits[16].")));
}

TEST_F(NodeTest, ConcatOperandSliceData) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
fn f(x: bits[3], y: bits[2], z: bits[1]) -> bits[6] {
  ret concat.4: bits[6] = concat(x, y, z)
}
)",
                                                       p.get()));
  EXPECT_EQ(f->return_value()->op(), Op::kConcat);
  Concat* concat = FindNode("concat.4", f)->As<Concat>();
  // z operand
  EXPECT_EQ(0, concat->GetOperandSliceData(2).start);
  EXPECT_EQ(1, concat->GetOperandSliceData(2).width);
  // y operand
  EXPECT_EQ(1, concat->GetOperandSliceData(1).start);
  EXPECT_EQ(2, concat->GetOperandSliceData(1).width);
  // x operand
  EXPECT_EQ(3, concat->GetOperandSliceData(0).start);
  EXPECT_EQ(3, concat->GetOperandSliceData(0).width);
}

TEST_F(NodeTest, NodeNames) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  auto literal0 = fb.Literal(UBits(123, 32));
  auto add = fb.Add(literal0, literal0);
  add.SetName("foobar");
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.BuildWithReturnValue(add));

  EXPECT_EQ(f->return_value()->GetName(), "foobar");
  EXPECT_TRUE(f->return_value()->HasAssignedName());

  f->return_value()->ClearName();

  EXPECT_EQ(f->return_value()->GetName(), "add.2");
  EXPECT_FALSE(f->return_value()->HasAssignedName());

  f->return_value()->SetName("foobar");

  EXPECT_EQ(f->return_value()->GetName(), "foobar__1");
  EXPECT_TRUE(f->return_value()->HasAssignedName());
}

TEST_F(NodeTest, IsDead) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
fn ReplaceUses(x: bits[8], y: bits[8]) -> bits[8] {
  so_dead: bits[8] = and(x, x)
  ret n: bits[8] = not(x)
}
)",
                                                       p.get()));
  EXPECT_FALSE(FindNode("x", f)->IsDead());
  EXPECT_FALSE(FindNode("n", f)->IsDead());
  EXPECT_TRUE(FindNode("so_dead", f)->IsDead());
  EXPECT_TRUE(FindNode("y", f)->IsDead());
}

TEST_F(NodeTest, IncorrectOpClass) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue x = fb.Param("x", p->GetBitsType(32));
  BValue y = fb.Param("y", p->GetBitsType(32));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());

  // Cast to void to avoid unused status build error.
  EXPECT_DEATH(
      (void)f->MakeNode<ArithOp>(SourceInfo(), x.node(), y.node(), 32,
                                 Op::kAdd),
      HasSubstr("Op `add` is not a valid op for Node class `ArithOp`"));
  EXPECT_DEATH(
      (void)f->MakeNode<UnOp>(SourceInfo(), x.node(), Op::kAssert),
      HasSubstr("Op `assert` is not a valid op for Node class `UnOp`"));
}

TEST_F(NodeTest, MakeParam) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  fb.Param("x", p->GetBitsType(32));
  fb.Param("y", p->GetBitsType(32));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());

  EXPECT_THAT(f->MakeNode<Param>(SourceInfo(), p->GetBitsType(32)),
              // Node has id=3, so with no name it will be "param.3".
              IsOkAndHolds(m::Param("param.3")));
  EXPECT_THAT(f->MakeNodeWithName<Param>(SourceInfo(), p->GetBitsType(32), "x"),
              // "x" is already taken, so uniquer will choose "x__1".
              IsOkAndHolds(m::Param("x__1")));
}

template <typename T, typename = void>
struct MakeNodeWillSubstituteWith : std::false_type {};

template <typename T>
struct MakeNodeWillSubstituteWith<
    T, std::void_t<decltype(&FunctionBase::MakeNode<T>)>> : std::true_type {};

TEST_F(NodeTest, MakeNodeForType) {
  // These static_asserts() could be anywhere, but we gather them in a test for
  // organizational purposes.
  static_assert(MakeNodeWillSubstituteWith<Param>::value);
  static_assert(!MakeNodeWillSubstituteWith<int>::value);
  static_assert(!MakeNodeWillSubstituteWith<Param*>::value);
  static_assert(!MakeNodeWillSubstituteWith<Block::ClockPort>::value);
  static_assert(!MakeNodeWillSubstituteWith<Block::Port>::value);
}

}  // namespace
}  // namespace xls
