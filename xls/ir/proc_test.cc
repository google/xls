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

#include "xls/ir/proc.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "xls/common/status/matchers.h"
#include "xls/ir/ir_matcher.h"
#include "xls/ir/ir_test_base.h"

namespace m = ::xls::op_matchers;

namespace xls {
namespace {

using status_testing::StatusIs;
using ::testing::ElementsAre;
using ::testing::HasSubstr;

class ProcTest : public IrTestBase {};

TEST_F(ProcTest, SimpleProc) {
  auto p = CreatePackage();
  ProcBuilder pb("p", "tkn", p.get());
  BValue state = pb.StateElement("st", Value(UBits(42, 32)));
  BValue add = pb.Add(pb.Literal(UBits(1, 32)), state);
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, pb.Build(pb.GetTokenParam(), {add}));

  EXPECT_FALSE(proc->IsFunction());
  EXPECT_TRUE(proc->IsProc());
  EXPECT_EQ(proc->GetStateFlatBitCount(), 32);
  EXPECT_EQ(proc->DumpIr(), R"(proc p(tkn: token, st: bits[32], init={42}) {
  literal.3: bits[32] = literal(value=1, id=3)
  add.4: bits[32] = add(literal.3, st, id=4)
  next (tkn, add.4)
}
)");
}

TEST_F(ProcTest, MutateProc) {
  auto p = CreatePackage();
  ProcBuilder pb("p", "tkn", p.get());
  BValue state = pb.StateElement("st", Value(UBits(42, 32)));
  BValue add = pb.Add(pb.Literal(UBits(1, 32)), state);
  BValue after_all = pb.AfterAll({pb.GetTokenParam()});
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, pb.Build(after_all, {add}));

  EXPECT_THAT(proc->NextToken(), m::AfterAll(m::Param("tkn")));
  XLS_ASSERT_OK(proc->SetNextToken(proc->TokenParam()));
  XLS_ASSERT_OK(proc->RemoveNode(after_all.node()));
  EXPECT_THAT(proc->NextToken(), m::Param("tkn"));

  EXPECT_THAT(proc->GetNextStateElement(0), m::Add());
  XLS_ASSERT_OK(proc->SetNextStateElement(0, proc->GetStateParam(0)));
  EXPECT_THAT(proc->GetNextStateElement(0), m::Param("st"));

  // Replace the state with a new type. First need to delete the (dead) use of
  // the existing state param.
  XLS_ASSERT_OK(proc->RemoveNode(add.node()));
  XLS_ASSERT_OK_AND_ASSIGN(
      Node * new_state,
      proc->MakeNode<Literal>(SourceInfo(), Value(UBits(100, 100))));
  XLS_ASSERT_OK(proc->ReplaceStateElement(0, "new_state",
                                          Value(UBits(100, 100)), new_state));

  EXPECT_THAT(proc->GetNextStateElement(0), m::Literal(UBits(100, 100)));
  EXPECT_THAT(proc->GetStateParam(0), m::Type("bits[100]"));
}

TEST_F(ProcTest, AddAndRemoveState) {
  auto p = CreatePackage();
  ProcBuilder pb("p", "tkn", p.get());
  BValue state = pb.StateElement("x", Value(UBits(42, 32)));
  BValue add = pb.Add(pb.Literal(UBits(1, 32)), state, SourceInfo(), "my_add");
  BValue after_all =
      pb.AfterAll({pb.GetTokenParam()}, SourceInfo(), "my_after_all");
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, pb.Build(after_all, {add}));

  EXPECT_EQ(proc->GetStateElementCount(), 1);
  EXPECT_EQ(proc->GetStateParam(0)->GetName(), "x");
  EXPECT_EQ(proc->GetInitValueElement(0), Value(UBits(42, 32)));

  XLS_ASSERT_OK(proc->AppendStateElement("y", Value(UBits(100, 32))));
  EXPECT_EQ(proc->GetStateElementCount(), 2);
  EXPECT_EQ(proc->GetStateParam(0)->GetName(), "x");
  EXPECT_EQ(proc->GetStateParam(1)->GetName(), "y");
  EXPECT_EQ(proc->GetInitValueElement(0), Value(UBits(42, 32)));
  EXPECT_EQ(proc->GetInitValueElement(1), Value(UBits(100, 32)));

  // Add a state element with a specifed next state (the state parameter "x").
  XLS_ASSERT_OK(proc->AppendStateElement(
      "z", Value(UBits(123, 32)), /*next_state=*/proc->GetStateParam(0)));
  EXPECT_EQ(proc->GetStateElementCount(), 3);
  EXPECT_EQ(proc->GetStateParam(0)->GetName(), "x");
  EXPECT_EQ(proc->GetStateParam(1)->GetName(), "y");
  EXPECT_EQ(proc->GetStateParam(2)->GetName(), "z");
  EXPECT_EQ(proc->GetInitValueElement(0), Value(UBits(42, 32)));
  EXPECT_EQ(proc->GetInitValueElement(1), Value(UBits(100, 32)));
  EXPECT_EQ(proc->GetInitValueElement(2), Value(UBits(123, 32)));
  EXPECT_EQ(proc->GetNextStateElement(0)->GetName(), "my_add");
  EXPECT_EQ(proc->GetNextStateElement(1)->GetName(), "y");
  EXPECT_EQ(proc->GetNextStateElement(2)->GetName(), "x");

  XLS_ASSERT_OK(proc->RemoveStateElement(1));
  EXPECT_EQ(proc->GetStateElementCount(), 2);
  EXPECT_EQ(proc->GetStateParam(0)->GetName(), "x");
  EXPECT_EQ(proc->GetStateParam(1)->GetName(), "z");

  XLS_ASSERT_OK(proc->InsertStateElement(0, "foo", Value(UBits(123, 32))));
  EXPECT_EQ(proc->GetStateElementCount(), 3);
  EXPECT_EQ(proc->GetStateParam(0)->GetName(), "foo");
  EXPECT_EQ(proc->GetStateParam(1)->GetName(), "x");
  EXPECT_EQ(proc->GetStateParam(2)->GetName(), "z");

  XLS_ASSERT_OK(proc->InsertStateElement(3, "bar", Value(UBits(1, 64))));
  EXPECT_EQ(proc->GetStateElementCount(), 4);
  EXPECT_EQ(proc->GetStateParam(0)->GetName(), "foo");
  EXPECT_EQ(proc->GetStateParam(1)->GetName(), "x");
  EXPECT_EQ(proc->GetStateParam(2)->GetName(), "z");
  EXPECT_EQ(proc->GetStateParam(3)->GetName(), "bar");

  XLS_ASSERT_OK(proc->ReplaceStateElement(2, "baz", Value::Tuple({})));
  EXPECT_EQ(proc->GetStateElementCount(), 4);
  EXPECT_EQ(proc->GetStateParam(0)->GetName(), "foo");
  EXPECT_EQ(proc->GetStateParam(1)->GetName(), "x");
  EXPECT_EQ(proc->GetStateParam(2)->GetName(), "baz");
  EXPECT_EQ(proc->GetStateParam(3)->GetName(), "bar");

  EXPECT_THAT(proc->DumpIr(),
              HasSubstr("proc p(tkn: token, foo: bits[32], x: bits[32], baz: "
                        "(), bar: bits[64], init={123, 42, (), 1}"));
  EXPECT_THAT(proc->DumpIr(),
              HasSubstr("next (my_after_all, foo, my_add, baz, bar"));
}

TEST_F(ProcTest, ReplaceState) {
  auto p = CreatePackage();
  ProcBuilder pb("p", "tkn", p.get());
  pb.StateElement("x", Value(UBits(42, 32)));
  BValue forty_two = pb.Literal(UBits(42, 32), SourceInfo(), "forty_two");
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc,
                           pb.Build(pb.GetTokenParam(), {forty_two}));
  EXPECT_THAT(proc->GetNextStateIndices(forty_two.node()), ElementsAre(0));

  XLS_ASSERT_OK(proc->ReplaceState(
      {"foo", "bar", "baz"},
      {Value(UBits(1, 32)), Value(UBits(2, 32)), Value(UBits(2, 32))},
      {forty_two.node(), forty_two.node(), forty_two.node()}));
  EXPECT_THAT(proc->GetNextStateIndices(forty_two.node()),
              ElementsAre(0, 1, 2));

  EXPECT_THAT(proc->DumpIr(),
              HasSubstr("proc p(tkn: token, foo: bits[32], bar: bits[32], baz: "
                        "bits[32], init={1, 2, 2})"));
  EXPECT_THAT(proc->DumpIr(),
              HasSubstr("next (tkn, forty_two, forty_two, forty_two)"));
}

TEST_F(ProcTest, StatelessProc) {
  auto p = CreatePackage();
  ProcBuilder pb("p", "tkn", p.get());
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, pb.Build(pb.GetTokenParam(), {}));

  EXPECT_EQ(proc->GetStateElementCount(), 0);
  EXPECT_EQ(proc->GetStateFlatBitCount(), 0);

  EXPECT_THAT(proc->DumpIr(), HasSubstr("proc p(tkn: token, init={})"));
  EXPECT_THAT(proc->DumpIr(), HasSubstr("next (tkn)"));
}

TEST_F(ProcTest, InvalidTokenType) {
  auto p = CreatePackage();
  ProcBuilder pb("p", "tkn", p.get());
  BValue state = pb.StateElement("st", Value(UBits(42, 32)));
  BValue add = pb.Add(pb.Literal(UBits(1, 32)), state);
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc,
                           pb.Build(pb.AfterAll({pb.GetTokenParam()}), {add}));

  // Try setting invalid typed nodes as the next token/state.
  EXPECT_THAT(
      proc->SetNextToken(add.node()),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr(
                   "Cannot set next token to \"add.4\", expected token type")));
}

TEST_F(ProcTest, InvalidStateType) {
  auto p = CreatePackage();
  ProcBuilder pb("p", "tkn", p.get());
  BValue state = pb.StateElement("st", Value(UBits(42, 32)));
  BValue add = pb.Add(pb.Literal(UBits(1, 32)), state);
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc,
                           pb.Build(pb.AfterAll({pb.GetTokenParam()}), {add}));

  EXPECT_THAT(
      proc->SetNextStateElement(0, proc->TokenParam()),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("type token does not match proc state element type")));
}

TEST_F(ProcTest, ReplaceStateThatStillHasUse) {
  // Don't call CreatePackage which creates a VerifiedPackage because we
  // intentionally create a malformed proc.
  Package p(TestName());
  ProcBuilder pb("p", "tkn", &p);
  BValue state = pb.StateElement("st", Value(UBits(42, 32)));
  BValue add = pb.Add(pb.Literal(UBits(1, 32)), state);
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc,
                           pb.Build(pb.AfterAll({pb.GetTokenParam()}), {add}));

  XLS_ASSERT_OK_AND_ASSIGN(
      Node * new_state,
      proc->MakeNode<Literal>(SourceInfo(), Value(UBits(100, 100))));
  EXPECT_THAT(proc->ReplaceStateElement(0, "new_state", Value(UBits(100, 100)),
                                        new_state),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("state param st has uses")));
}

TEST_F(ProcTest, ReplaceStateWithWrongInitValueType) {
  // Don't call CreatePackage which creates a VerifiedPackage because we
  // intentionally create a malformed proc.
  Package p(TestName());
  ProcBuilder pb("p", "tkn", &p);
  pb.StateElement("st", Value(UBits(42, 32)));
  XLS_ASSERT_OK_AND_ASSIGN(
      Proc * proc,
      pb.Build(pb.AfterAll({pb.GetTokenParam()}), {pb.Literal(UBits(1, 32))}));

  XLS_ASSERT_OK_AND_ASSIGN(
      Node * new_state,
      proc->MakeNode<Literal>(SourceInfo(), Value(UBits(100, 100))));
  EXPECT_THAT(
      proc->ReplaceStateElement(0, "new_state",
                                /*init_value=*/Value(UBits(0, 42)), new_state),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("does not match type of initial value")));
}

TEST_F(ProcTest, Clone) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(
      SingleValueChannel * channel,
      p->CreateSingleValueChannel("chan", ChannelOps::kSendReceive,
                                  p->GetBitsType(32)));

  ProcBuilder pb("p", "tkn", p.get());
  BValue state = pb.StateElement("st", Value(UBits(42, 32)));
  BValue recv = pb.Receive(channel, pb.GetTokenParam());
  BValue add1 = pb.Add(pb.Literal(UBits(1, 32)), state);
  BValue add2 = pb.Add(add1, pb.TupleIndex(recv, 1));
  BValue send = pb.Send(channel, pb.TupleIndex(recv, 0), add2);
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, pb.Build(send, {add2}));

  XLS_ASSERT_OK_AND_ASSIGN(
      SingleValueChannel * cloned_channel,
      p->CreateSingleValueChannel("cloned_chan", ChannelOps::kSendReceive,
                                  p->GetBitsType(32)));

  XLS_ASSERT_OK_AND_ASSIGN(
      Proc * clone,
      proc->Clone("cloned", p.get(), {{channel->id(), cloned_channel->id()}}));

  EXPECT_FALSE(clone->IsFunction());
  EXPECT_TRUE(clone->IsProc());

  EXPECT_EQ(clone->DumpIr(),
            R"(proc cloned(tkn: token, st: bits[32], init={42}) {
  literal.12: bits[32] = literal(value=1, id=12)
  receive_3: (token, bits[32]) = receive(tkn, channel_id=1, id=13)
  add.14: bits[32] = add(literal.12, st, id=14)
  tuple_index.15: bits[32] = tuple_index(receive_3, index=1, id=15)
  tuple_index.16: token = tuple_index(receive_3, index=0, id=16)
  add.17: bits[32] = add(add.14, tuple_index.15, id=17)
  send_9: token = send(tuple_index.16, add.17, channel_id=1, id=18)
  next (send_9, add.17)
}
)");
}

TEST_F(ProcTest, JoinNextTokenWith) {
  auto p = CreatePackage();
  ProcBuilder pb("p", "tkn", p.get());
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, pb.Build(pb.GetTokenParam(), {}));
  EXPECT_EQ(proc->node_count(), 1);
  EXPECT_THAT(proc->NextToken(), m::Param());

  XLS_ASSERT_OK(proc->JoinNextTokenWith({}));
  EXPECT_EQ(proc->node_count(), 1);
  EXPECT_THAT(proc->NextToken(), m::Param());

  XLS_ASSERT_OK_AND_ASSIGN(
      Node * tkn_a, proc->MakeNodeWithName<UnOp>(
                        SourceInfo(), proc->TokenParam(), Op::kIdentity, "a"));
  XLS_ASSERT_OK_AND_ASSIGN(
      Node * tkn_b, proc->MakeNodeWithName<UnOp>(
                        SourceInfo(), proc->TokenParam(), Op::kIdentity, "b"));
  XLS_ASSERT_OK_AND_ASSIGN(
      Node * tkn_c, proc->MakeNodeWithName<UnOp>(
                        SourceInfo(), proc->TokenParam(), Op::kIdentity, "c"));

  EXPECT_EQ(proc->node_count(), 4);
  XLS_ASSERT_OK(proc->JoinNextTokenWith({tkn_a}));
  EXPECT_EQ(proc->node_count(), 5);
  EXPECT_THAT(proc->NextToken(), m::AfterAll(m::Param(), m::Name("a")));

  XLS_ASSERT_OK(proc->JoinNextTokenWith({tkn_b, tkn_c}));
  EXPECT_EQ(proc->node_count(), 5);
  EXPECT_THAT(proc->NextToken(), m::AfterAll(m::Param(), m::Name("a"),
                                             m::Name("b"), m::Name("c")));
}

}  // namespace
}  // namespace xls
