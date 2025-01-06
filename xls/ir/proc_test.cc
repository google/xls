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

#include <array>
#include <cstdint>
#include <optional>
#include <string>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "xls/common/status/matchers.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/bits.h"
#include "xls/ir/channel.h"
#include "xls/ir/channel_ops.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/ir_matcher.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/node.h"
#include "xls/ir/nodes.h"
#include "xls/ir/op.h"
#include "xls/ir/package.h"
#include "xls/ir/source_location.h"
#include "xls/ir/value.h"

namespace m = ::xls::op_matchers;

namespace xls {
namespace {

using ::absl_testing::StatusIs;
using ::testing::ElementsAre;
using ::testing::HasSubstr;
using ::testing::IsEmpty;
using ::testing::UnorderedElementsAre;

class ProcTest : public IrTestBase {};

TEST_F(ProcTest, SimpleProc) {
  auto p = CreatePackage();
  ProcBuilder pb("p", p.get());
  BValue state = pb.StateElement("st", Value(UBits(42, 32)));
  BValue add = pb.Add(pb.Literal(UBits(1, 32)), state);
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, pb.Build({add}));

  EXPECT_FALSE(proc->IsFunction());
  EXPECT_TRUE(proc->IsProc());
  EXPECT_EQ(proc->GetStateFlatBitCount(), 32);
  EXPECT_EQ(proc->DumpIr(), R"(proc p(st: bits[32], init={42}) {
  literal.2: bits[32] = literal(value=1, id=2)
  st: bits[32] = state_read(state_element=st, id=1)
  add.3: bits[32] = add(literal.2, st, id=3)
  next_value.4: () = next_value(param=st, value=add.3, id=4)
}
)");
}

TEST_F(ProcTest, MutateProc) {
  auto p = CreatePackage();
  ProcBuilder pb("p", p.get());
  BValue tkn = pb.StateElement("tkn", Value::Token());
  BValue state = pb.StateElement("st", Value(UBits(42, 32)));
  BValue add = pb.Add(pb.Literal(UBits(1, 32)), state);
  BValue after_all = pb.AfterAll({tkn});
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, pb.Build({after_all, add}));

  ASSERT_THAT(proc->next_values(proc->GetStateRead(int64_t{0})),
              ElementsAre(m::Next(m::StateRead("tkn"),
                                  m::AfterAll(m::StateRead("tkn")))));
  Next* next_tkn = *proc->next_values(proc->GetStateRead(int64_t{0})).begin();
  XLS_ASSERT_OK(proc->RemoveNode(next_tkn));
  XLS_ASSERT_OK(proc->RemoveNode(after_all.node()));
  EXPECT_THAT(proc->next_values(proc->GetStateRead(int64_t{0})), IsEmpty());

  ASSERT_THAT(proc->next_values(proc->GetStateRead(1)),
              ElementsAre(m::Next(m::StateRead("st"), m::Add())));
  Next* next_st = *proc->next_values(proc->GetStateRead(1)).begin();
  XLS_ASSERT_OK(proc->RemoveNode(next_st));
  EXPECT_THAT(proc->next_values(proc->GetStateRead(1)), IsEmpty());

  // Replace the state with a new type. First need to delete the (dead) use of
  // the existing state read.
  XLS_ASSERT_OK(proc->RemoveNode(add.node()));
  XLS_ASSERT_OK_AND_ASSIGN(
      Node * new_state,
      proc->MakeNode<Literal>(SourceInfo(), Value(UBits(100, 100))));
  XLS_ASSERT_OK(proc->ReplaceStateElement(1, "new_state",
                                          Value(UBits(100, 100)), new_state));

  EXPECT_THAT(proc->next_values(proc->GetStateRead(1)),
              ElementsAre(m::Next(m::StateRead("new_state"),
                                  m::Literal(UBits(100, 100)))));
  EXPECT_THAT(proc->GetStateRead(1), m::Type("bits[100]"));
}

TEST_F(ProcTest, AddAndRemoveState) {
  auto p = CreatePackage();
  ProcBuilder pb("p", p.get());
  BValue tkn = pb.StateElement("tkn", Value::Token());
  BValue state = pb.StateElement("x", Value(UBits(42, 32)));
  BValue add = pb.Add(pb.Literal(UBits(1, 32)), state, SourceInfo(), "my_add");
  BValue after_all = pb.AfterAll({tkn}, SourceInfo(), "my_after_all");
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, pb.Build({after_all, add}));

  EXPECT_EQ(proc->GetStateElementCount(), 2);
  EXPECT_EQ(proc->GetStateElement(0)->name(), "tkn");
  EXPECT_EQ(proc->GetStateElement(0)->initial_value(), Value::Token());
  EXPECT_EQ(proc->GetStateElement(1)->name(), "x");
  EXPECT_EQ(proc->GetStateElement(1)->initial_value(), Value(UBits(42, 32)));

  XLS_ASSERT_OK(proc->AppendStateElement("y", Value(UBits(100, 32)),
                                         /*read_predicate=*/std::nullopt,
                                         /*next_state=*/std::nullopt));
  EXPECT_EQ(proc->GetStateElementCount(), 3);
  EXPECT_EQ(proc->GetStateElement(0)->name(), "tkn");
  EXPECT_EQ(proc->GetStateElement(1)->name(), "x");
  EXPECT_EQ(proc->GetStateElement(2)->name(), "y");
  EXPECT_EQ(proc->GetStateElement(0)->initial_value(), Value::Token());
  EXPECT_EQ(proc->GetStateElement(1)->initial_value(), Value(UBits(42, 32)));
  EXPECT_EQ(proc->GetStateElement(2)->initial_value(), Value(UBits(100, 32)));

  // Add a state element with a specified next state (the literal 0).
  XLS_ASSERT_OK_AND_ASSIGN(Literal * zero_literal,
                           proc->MakeNodeWithName<Literal>(
                               SourceInfo(), Value(UBits(0, 32)), "zero"));
  XLS_ASSERT_OK(proc->AppendStateElement("z", Value(UBits(123, 32)),
                                         /*read_predicate=*/std::nullopt,
                                         /*next_state=*/zero_literal));
  EXPECT_EQ(proc->GetStateElementCount(), 4);
  EXPECT_EQ(proc->GetStateElement(0)->name(), "tkn");
  EXPECT_EQ(proc->GetStateElement(1)->name(), "x");
  EXPECT_EQ(proc->GetStateElement(2)->name(), "y");
  EXPECT_EQ(proc->GetStateElement(3)->name(), "z");
  EXPECT_EQ(proc->GetStateElement(0)->initial_value(), Value::Token());
  EXPECT_EQ(proc->GetStateElement(1)->initial_value(), Value(UBits(42, 32)));
  EXPECT_EQ(proc->GetStateElement(2)->initial_value(), Value(UBits(100, 32)));
  EXPECT_EQ(proc->GetStateElement(3)->initial_value(), Value(UBits(123, 32)));
  EXPECT_THAT(
      proc->next_values(proc->GetStateRead(int64_t{0})),
      ElementsAre(m::Next(m::StateRead("tkn"), m::Name("my_after_all"))));
  EXPECT_THAT(proc->next_values(proc->GetStateRead(1)),
              ElementsAre(m::Next(m::StateRead("x"), m::Name("my_add"))));
  EXPECT_THAT(proc->next_values(proc->GetStateRead(2)), IsEmpty());
  EXPECT_THAT(proc->next_values(proc->GetStateRead(3)),
              ElementsAre(m::Next(m::StateRead("z"), m::Literal(0))));

  XLS_ASSERT_OK(proc->RemoveStateElement(2));
  EXPECT_EQ(proc->GetStateElementCount(), 3);
  EXPECT_EQ(proc->GetStateElement(0)->name(), "tkn");
  EXPECT_EQ(proc->GetStateElement(1)->name(), "x");
  EXPECT_EQ(proc->GetStateElement(2)->name(), "z");

  XLS_ASSERT_OK(proc->InsertStateElement(0, "foo", Value(UBits(123, 32)),
                                         /*read_predicate=*/std::nullopt,
                                         /*next_state=*/std::nullopt));
  EXPECT_EQ(proc->GetStateElementCount(), 4);
  EXPECT_EQ(proc->GetStateElement(0)->name(), "foo");
  EXPECT_EQ(proc->GetStateElement(1)->name(), "tkn");
  EXPECT_EQ(proc->GetStateElement(2)->name(), "x");
  EXPECT_EQ(proc->GetStateElement(3)->name(), "z");

  XLS_ASSERT_OK(proc->InsertStateElement(4, "bar", Value(UBits(1, 64)),
                                         /*read_predicate=*/std::nullopt,
                                         /*next_state=*/std::nullopt));
  EXPECT_EQ(proc->GetStateElementCount(), 5);
  EXPECT_EQ(proc->GetStateElement(0)->name(), "foo");
  EXPECT_EQ(proc->GetStateElement(1)->name(), "tkn");
  EXPECT_EQ(proc->GetStateElement(2)->name(), "x");
  EXPECT_EQ(proc->GetStateElement(3)->name(), "z");
  EXPECT_EQ(proc->GetStateElement(4)->name(), "bar");

  XLS_ASSERT_OK(
      proc->RemoveNode(*proc->next_values(proc->GetStateRead(3)).begin()));
  XLS_ASSERT_OK(proc->ReplaceStateElement(3, "baz", Value::Tuple({})));
  EXPECT_EQ(proc->GetStateElementCount(), 5);
  EXPECT_EQ(proc->GetStateElement(0)->name(), "foo");
  EXPECT_EQ(proc->GetStateElement(1)->name(), "tkn");
  EXPECT_EQ(proc->GetStateElement(2)->name(), "x");
  EXPECT_EQ(proc->GetStateElement(3)->name(), "baz");
  EXPECT_EQ(proc->GetStateElement(4)->name(), "bar");
  EXPECT_THAT(proc->GetNextStateIndices(zero_literal), IsEmpty());

  EXPECT_THAT(proc->DumpIr(),
              HasSubstr("proc p(foo: bits[32], tkn: token, x: bits[32], baz: "
                        "(), bar: bits[64], init={123, token, 42, (), 1}"));
  EXPECT_THAT(proc->next_values(proc->GetStateRead(int64_t{0})), IsEmpty());
  EXPECT_THAT(
      proc->next_values(proc->GetStateRead(1)),
      ElementsAre(m::Next(m::StateRead("tkn"), m::Name("my_after_all"))));
  EXPECT_THAT(proc->next_values(proc->GetStateRead(2)),
              ElementsAre(m::Next(m::StateRead("x"), m::Name("my_add"))));
  EXPECT_THAT(proc->next_values(proc->GetStateRead(3)), IsEmpty());
  EXPECT_THAT(proc->next_values(proc->GetStateRead(4)), IsEmpty());
}

TEST_F(ProcTest, ReplaceState) {
  auto p = CreatePackage();
  ProcBuilder pb("p", p.get());
  pb.StateElement("x", Value(UBits(42, 32)));
  BValue forty_two = pb.Literal(UBits(42, 32), SourceInfo(), "forty_two");
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, pb.Build({forty_two}));

  XLS_ASSERT_OK(proc->ReplaceState(
      {"foo", "bar", "baz"},
      {Value(UBits(1, 32)), Value(UBits(2, 32)), Value(UBits(2, 32))},
      {std::nullopt, std::nullopt, std::nullopt},
      {forty_two.node(), forty_two.node(), forty_two.node()}));

  EXPECT_THAT(proc->DumpIr(),
              HasSubstr("proc p(foo: bits[32], bar: bits[32], baz: "
                        "bits[32], init={1, 2, 2})"));

  EXPECT_THAT(proc->next_values(proc->GetStateRead(int64_t{0})),
              ElementsAre(m::Next(m::StateRead("foo"), m::Literal(42, 32))));
  EXPECT_THAT(proc->next_values(proc->GetStateRead(1)),
              ElementsAre(m::Next(m::StateRead("bar"), m::Literal(42, 32))));
  EXPECT_THAT(proc->next_values(proc->GetStateRead(2)),
              ElementsAre(m::Next(m::StateRead("baz"), m::Literal(42, 32))));
}

TEST_F(ProcTest, StatelessProc) {
  auto p = CreatePackage();
  ProcBuilder pb("p", p.get());
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, pb.Build({}));

  EXPECT_EQ(proc->GetStateElementCount(), 0);
  EXPECT_EQ(proc->GetStateFlatBitCount(), 0);

  EXPECT_EQ(proc->DumpIr(), "proc p() {\n}\n");
}

TEST_F(ProcTest, InvalidTokenType) {
  auto p = CreatePackage();
  ProcBuilder pb("p", p.get());
  BValue tkn = pb.StateElement("tkn", Value::Token());
  BValue state = pb.StateElement("st", Value(UBits(42, 32)));
  BValue add = pb.Add(pb.Literal(UBits(1, 32)), state);
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, pb.Build({pb.AfterAll({tkn}), add}));

  // Try setting invalid typed nodes as the next token/state.
  EXPECT_THAT(
      proc->SetNextStateElement(0, add.node()),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          HasSubstr(
              "type bits[32] does not match proc state element type token")));
}

TEST_F(ProcTest, InvalidStateType) {
  auto p = CreatePackage();
  ProcBuilder pb("p", p.get());
  BValue tkn = pb.StateElement("tkn", Value::Token());
  BValue state = pb.StateElement("st", Value(UBits(42, 32)));
  BValue add = pb.Add(pb.Literal(UBits(1, 32)), state);
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, pb.Build({pb.AfterAll({tkn}), add}));

  EXPECT_THAT(
      proc->SetNextStateElement(1, tkn.node()),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          HasSubstr(
              "type token does not match proc state element type bits[32]")));
}

TEST_F(ProcTest, ReplaceStateThatStillHasUse) {
  // Don't call CreatePackage which creates a VerifiedPackage because we
  // intentionally create a malformed proc.
  Package p(TestName());
  ProcBuilder pb("p", &p);
  BValue state = pb.StateElement("st", Value(UBits(42, 32)));
  BValue add = pb.Add(pb.Literal(UBits(1, 32)), state);
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, pb.Build({add}));

  XLS_ASSERT_OK_AND_ASSIGN(
      Node * new_state,
      proc->MakeNode<Literal>(SourceInfo(), Value(UBits(100, 100))));
  EXPECT_THAT(proc->ReplaceStateElement(0, "new_state", Value(UBits(100, 100)),
                                        new_state),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("state read st has uses")));
}

TEST_F(ProcTest, RemoveStateThatStillHasUse) {
  // Don't call CreatePackage which creates a VerifiedPackage because we
  // intentionally create a malformed proc.
  Package p(TestName());
  ProcBuilder pb("p", &p);
  BValue state = pb.StateElement("st", Value(UBits(42, 32)));
  BValue add = pb.Add(pb.Literal(UBits(1, 32)), state);
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, pb.Build({add}));

  EXPECT_THAT(proc->RemoveStateElement(0),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("state read st has uses")));
}

TEST_F(ProcTest, ReplaceStateWithWrongInitValueType) {
  // Don't call CreatePackage which creates a VerifiedPackage because we
  // intentionally create a malformed proc.
  Package p(TestName());
  ProcBuilder pb("p", &p);
  pb.StateElement("st", Value(UBits(42, 32)));
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, pb.Build({pb.Literal(UBits(1, 32))}));

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

  ProcBuilder pb("p", p.get());
  BValue tkn = pb.StateElement("tkn", Value::Token());
  BValue state = pb.StateElement("st", Value(UBits(42, 32)));
  BValue recv = pb.Receive(channel, tkn);
  BValue add1 = pb.Add(pb.Literal(UBits(1, 32)), state);
  BValue add2 = pb.Add(add1, pb.TupleIndex(recv, 1));
  BValue send = pb.Send(channel, pb.TupleIndex(recv, 0), add2);
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, pb.Build({send, add2}));

  XLS_ASSERT_OK_AND_ASSIGN(
      SingleValueChannel * cloned_channel,
      p->CreateSingleValueChannel("cloned_chan", ChannelOps::kSendReceive,
                                  p->GetBitsType(32)));

  XLS_ASSERT_OK_AND_ASSIGN(
      Proc * clone, proc->Clone("cloned", p.get(),
                                /*channel_remapping=*/
                                {{std::string{channel->name()},
                                  std::string{cloned_channel->name()}}},
                                /*call_remapping=*/{},
                                /*state_name_remapping=*/{{"st", "state"}}));

  EXPECT_FALSE(clone->IsFunction());
  EXPECT_TRUE(clone->IsProc());

  EXPECT_EQ(clone->DumpIr(),
            R"(proc cloned(tkn: token, state: bits[32], init={token, 42}) {
  tkn: token = state_read(state_element=tkn, id=12)
  literal.14: bits[32] = literal(value=1, id=14)
  state: bits[32] = state_read(state_element=state, id=13)
  receive_3: (token, bits[32]) = receive(tkn, channel=cloned_chan, id=15)
  add.16: bits[32] = add(literal.14, state, id=16)
  tuple_index.17: bits[32] = tuple_index(receive_3, index=1, id=17)
  tuple_index.18: token = tuple_index(receive_3, index=0, id=18)
  add.19: bits[32] = add(add.16, tuple_index.17, id=19)
  send_9: token = send(tuple_index.18, add.19, channel=cloned_chan, id=20)
  next_value.21: () = next_value(param=tkn, value=send_9, id=21)
  next_value.22: () = next_value(param=state, value=add.19, id=22)
}
)");
}

TEST_F(ProcTest, CloneNewStyle) {
  auto p = CreatePackage();
  TokenlessProcBuilder pb(NewStyleProc(), "p", "tkn", p.get());
  XLS_ASSERT_OK_AND_ASSIGN(ReceiveChannelReference * ch_a,
                           pb.AddInputChannel("a", p->GetBitsType(32)));
  XLS_ASSERT_OK_AND_ASSIGN(SendChannelReference * ch_b,
                           pb.AddOutputChannel("b", p->GetBitsType(32)));
  XLS_ASSERT_OK_AND_ASSIGN(ChannelReferences ch_c,
                           pb.AddChannel("c", p->GetBitsType(32), {}));

  BValue state = pb.StateElement("st", Value(UBits(42, 32)));
  BValue recv_a = pb.Receive(ch_a);
  BValue recv_c = pb.Receive(ch_c.receive_ref);
  pb.Send(ch_b, state);
  pb.Send(ch_c.send_ref, state);
  BValue add = pb.Add(recv_a, recv_c);
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, pb.Build({add}));

  XLS_ASSERT_OK_AND_ASSIGN(
      Proc * clone,
      proc->Clone(
          "cloned", p.get(),
          /*channel_remapping=*/{{"a", "foo"}, {"b", "bar"}, {"c", "baz"}},
          /*call_remapping=*/{},
          /*state_name_remapping=*/{{"st", "state"}}));

  EXPECT_FALSE(clone->IsFunction());
  EXPECT_TRUE(clone->IsProc());

  EXPECT_EQ(
      clone->DumpIr(),
      R"(proc cloned<foo: bits[32] in kind=streaming strictness=proven_mutually_exclusive, bar: bits[32] out kind=streaming strictness=proven_mutually_exclusive>(state: bits[32], init={42}) {
  chan baz(bits[32], id=0, kind=streaming, ops=send_receive, flow_control=ready_valid, strictness=proven_mutually_exclusive, metadata="""""")
  tkn: token = literal(value=token, id=14)
  receive_3: (token, bits[32]) = receive(tkn, channel=foo, id=15)
  tuple_index.16: token = tuple_index(receive_3, index=0, id=16)
  receive_6: (token, bits[32]) = receive(tuple_index.16, channel=baz, id=17)
  tuple_index.18: token = tuple_index(receive_6, index=0, id=18)
  state: bits[32] = state_read(state_element=state, id=13)
  tuple_index.19: bits[32] = tuple_index(receive_3, index=1, id=19)
  tuple_index.20: bits[32] = tuple_index(receive_6, index=1, id=20)
  send_9: token = send(tuple_index.18, state, channel=bar, id=21)
  add.22: bits[32] = add(tuple_index.19, tuple_index.20, id=22)
  send_10: token = send(send_9, state, channel=baz, id=23)
  next_value.24: () = next_value(param=state, value=add.22, id=24)
}
)");
}

TEST_F(ProcTest, TransformStateElement) {
  auto p = CreatePackage();
  TokenlessProcBuilder pb(TestName(), "tkn", p.get());
  auto st = pb.StateElement("st", UBits(0b1010, 4));
  auto cond = pb.StateElement("cond", UBits(0, 1));
  auto user = pb.Tuple({st});
  auto add_st = pb.Next(st, pb.Add(st, pb.Literal(UBits(1, 4))), cond);
  auto sub_st =
      pb.Next(st, pb.Subtract(st, pb.Literal(UBits(1, 4))), pb.Not(cond));
  pb.Next(cond, pb.Not(cond));

  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, pb.Build());
  // Test transformer that inverts the param.
  struct TestTransformer : public Proc::StateElementTransformer {
   public:
    absl::StatusOr<Node*> TransformStateRead(
        Proc* proc, StateRead* new_state_read,
        StateRead* old_state_read) override {
      return proc->MakeNode<UnOp>(new_state_read->loc(), new_state_read,
                                  Op::kNeg);
    }
    absl::StatusOr<Node*> TransformNextValue(Proc* proc,
                                             StateRead* new_state_read,
                                             Next* old_next) override {
      return proc->MakeNode<UnOp>(old_next->value()->loc(), old_next->value(),
                                  Op::kNeg);
    }
    absl::StatusOr<std::optional<Node*>> TransformNextPredicate(
        Proc* proc, StateRead* new_state_read, Next* old_next) override {
      XLS_ASSIGN_OR_RETURN(
          Node * true_const,
          proc->MakeNode<Literal>(old_next->loc(), Value::Bool(true)));
      if (old_next->predicate()) {
        return proc->MakeNode<NaryOp>(
            old_next->predicate().value()->loc(),
            std::array<Node*, 2>{true_const, *old_next->predicate()}, Op::kAnd);
      }
      return true_const;
    }
  };
  TestTransformer tt;
  ScopedRecordIr sri(p.get());
  XLS_ASSERT_OK_AND_ASSIGN(
      StateRead * new_st,
      proc->TransformStateElement(st.node()->As<StateRead>(),
                                  Value(UBits(0b0101, 4)), tt));

  // Make sure the st nexts has been identity-ified
  EXPECT_THAT(st.node(), m::StateRead(testing::Not("st")));
  EXPECT_THAT(st.node()->users(),
              UnorderedElementsAre(add_st.node(), sub_st.node()));
  EXPECT_THAT(add_st.node(), m::Next(st.node(), st.node(), cond.node()));
  EXPECT_THAT(sub_st.node(),
              m::Next(st.node(), st.node(), m::Not(cond.node())));

  // Make sure that 'new_state_read' takes over the name and everything.
  EXPECT_THAT(new_st, m::StateRead("st"));
  EXPECT_THAT(
      new_st->users(),
      UnorderedElementsAre(
          m::Neg(new_st),
          m::Next(new_st,
                  m::Neg(m::Add(m::Neg(new_st), m::Literal(UBits(1, 4)))),
                  m::And(m::Literal(UBits(1, 1)), cond.node())),
          m::Next(new_st,
                  m::Neg(m::Sub(m::Neg(new_st), m::Literal(UBits(1, 4)))),
                  m::And(m::Literal(UBits(1, 1)), m::Not(cond.node())))));

  // Make sure that user is updated.
  EXPECT_THAT(user.node(), m::Tuple(m::Neg(new_st)));
}

}  // namespace
}  // namespace xls
