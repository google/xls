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
#include <memory>
#include <optional>
#include <string>
#include <string_view>

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
#include "xls/ir/function_base.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/ir_matcher.h"
#include "xls/ir/ir_parser.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/node.h"
#include "xls/ir/nodes.h"
#include "xls/ir/op.h"
#include "xls/ir/package.h"
#include "xls/ir/proc_instantiation.h"
#include "xls/ir/scheduled_builder.h"
#include "xls/ir/source_location.h"
#include "xls/ir/value.h"

namespace m = ::xls::op_matchers;

namespace xls {
namespace {

using ::absl_testing::IsOkAndHolds;
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
  EXPECT_THAT(
      proc->DumpIr(),
      HasSubstr("proc p(foo: bits[32], tkn: token, x: bits[32], z: "
                "bits[32], bar: bits[64], init={123, token, 42, 123, 1}"));
  EXPECT_THAT(proc->next_values(proc->GetStateRead(int64_t{0})), IsEmpty());
  EXPECT_THAT(
      proc->next_values(proc->GetStateRead(1)),
      ElementsAre(m::Next(m::StateRead("tkn"), m::Name("my_after_all"))));
  EXPECT_THAT(proc->next_values(proc->GetStateRead(2)),
              ElementsAre(m::Next(m::StateRead("x"), m::Name("my_add"))));
  EXPECT_THAT(proc->next_values(proc->GetStateRead(3)), IsEmpty());
  EXPECT_THAT(proc->next_values(proc->GetStateRead(4)), IsEmpty());
}

TEST_F(ProcTest, StatelessProc) {
  auto p = CreatePackage();
  ProcBuilder pb("p", p.get());
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, pb.Build({}));

  EXPECT_EQ(proc->GetStateElementCount(), 0);
  EXPECT_EQ(proc->GetStateFlatBitCount(), 0);

  EXPECT_EQ(proc->DumpIr(), "proc p() {\n}\n");
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

TEST_F(ProcTest, CloneProcScopedChannel) {
  auto p = CreatePackage();

  ProcBuilder pb(NewStyleProc(), "p", p.get());
  XLS_ASSERT_OK_AND_ASSIGN(
      auto inp, pb.AddInputChannel("input_chan", p->GetBitsType(32), {}));
  XLS_ASSERT_OK_AND_ASSIGN(auto out,
                           pb.AddOutputChannel("chan", p->GetBitsType(32), {}));
  BValue tkn = pb.StateElement("tkn", Value::Token());
  BValue state = pb.StateElement("st", Value(UBits(42, 32)));
  BValue recv = pb.Receive(inp, tkn);
  BValue add1 = pb.Add(pb.Literal(UBits(1, 32)), state);
  BValue add2 = pb.Add(add1, pb.TupleIndex(recv, 1));
  BValue send = pb.Send(out, pb.TupleIndex(recv, 0), add2);
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, pb.Build({send, add2}));

  auto p2 = CreatePackage();

  XLS_ASSERT_OK_AND_ASSIGN(
      Proc * clone, proc->Clone("cloned", p2.get(),
                                /*channel_remapping=*/{},
                                /*call_remapping=*/{},
                                /*state_name_remapping=*/{{"st", "state"}}));

  EXPECT_FALSE(clone->IsFunction());
  EXPECT_TRUE(clone->IsProc());

  RecordProperty("p2", p2->DumpIr());

  EXPECT_EQ(p2->DumpIr(),
            R"IR(package CloneProcScopedChannel

proc cloned<input_chan: bits[32] in, chan: bits[32] out>(tkn: token, state: bits[32], init={token, 42}) {
  chan_interface input_chan(direction=receive, kind=streaming, strictness=proven_mutually_exclusive, flow_control=ready_valid, flop_kind=none)
  chan_interface chan(direction=send, kind=streaming, strictness=proven_mutually_exclusive, flow_control=ready_valid, flop_kind=none)
  tkn: token = state_read(state_element=tkn, id=1)
  literal.3: bits[32] = literal(value=1, id=3)
  state: bits[32] = state_read(state_element=state, id=2)
  receive_3: (token, bits[32]) = receive(tkn, channel=input_chan, id=4)
  add.5: bits[32] = add(literal.3, state, id=5)
  tuple_index.6: bits[32] = tuple_index(receive_3, index=1, id=6)
  tuple_index.7: token = tuple_index(receive_3, index=0, id=7)
  add.8: bits[32] = add(add.5, tuple_index.6, id=8)
  send_9: token = send(tuple_index.7, add.8, channel=chan, id=9)
  next_value.10: () = next_value(param=tkn, value=send_9, id=10)
  next_value.11: () = next_value(param=state, value=add.8, id=11)
}
)IR");
}

TEST_F(ProcTest, CloneNewStyle) {
  auto p = CreatePackage();
  TokenlessProcBuilder pb(NewStyleProc(), "p", "tkn", p.get());
  XLS_ASSERT_OK_AND_ASSIGN(ReceiveChannelInterface * ch_a,
                           pb.AddInputChannel("a", p->GetBitsType(32)));
  XLS_ASSERT_OK_AND_ASSIGN(SendChannelInterface * ch_b,
                           pb.AddOutputChannel("b", p->GetBitsType(32)));
  XLS_ASSERT_OK_AND_ASSIGN(ChannelWithInterfaces ch_c,
                           pb.AddChannel("c", p->GetBitsType(32), {}));

  BValue state = pb.StateElement("st", Value(UBits(42, 32)));
  BValue recv_a = pb.Receive(ch_a);
  BValue recv_c = pb.Receive(ch_c.receive_interface);
  pb.Send(ch_b, state);
  pb.Send(ch_c.send_interface, state);
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
      R"(proc cloned<foo: bits[32] in, bar: bits[32] out>(state: bits[32], init={42}) {
  chan_interface foo(direction=receive, kind=streaming, strictness=proven_mutually_exclusive, flow_control=ready_valid, flop_kind=none)
  chan_interface bar(direction=send, kind=streaming, strictness=proven_mutually_exclusive, flow_control=ready_valid, flop_kind=none)
  chan baz(bits[32], id=0, kind=streaming, ops=send_receive, flow_control=ready_valid, strictness=proven_mutually_exclusive)
  chan_interface baz(direction=send, kind=streaming, strictness=proven_mutually_exclusive, flow_control=none, flop_kind=none)
  chan_interface baz(direction=receive, kind=streaming, strictness=proven_mutually_exclusive, flow_control=none, flop_kind=none)
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

TEST_F(ProcTest, CloneProcInstantiation) {
  constexpr std::string_view kIrText = R"(package test
proc spawnee<>(__state: (), init={()}) {
}

top proc main<>(__state: (), init={()}) {
  proc_instantiation spawnee(proc=spawnee)
}
)";
  auto target = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> source,
                           Parser::ParsePackage(kIrText));
  XLS_ASSERT_OK_AND_ASSIGN(Proc * spawnee, source->GetProc("spawnee"));

  XLS_ASSERT_OK_AND_ASSIGN(Proc * cloned_spawnee,
                           spawnee->Clone("cloned_spawnee", target.get()));

  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, source->GetProc("main"));
  XLS_ASSERT_OK(proc->GetProcInstantiation("spawnee"));
  XLS_ASSERT_OK_AND_ASSIGN(
      Proc * clone,
      proc->Clone(
          "cloned", target.get(),
          /*channel_remapping=*/{},
          /*call_remapping=*/{},
          /*state_name_remapping=*/{},
          /*original_node_to_clone=*/std::nullopt,
          /*spawned_proc_name_remapping=*/{{"spawnee", "cloned_spawnee"}}));
  EXPECT_TRUE(clone->IsProc());
  EXPECT_FALSE(clone->GetProcInstantiation("spawnee").status().ok());
  XLS_ASSERT_OK_AND_ASSIGN(ProcInstantiation * instantiation,
                           clone->GetProcInstantiation("cloned_spawnee"));
  EXPECT_EQ(instantiation->name(), "cloned_spawnee");
  EXPECT_EQ(instantiation->proc(), cloned_spawnee);
}

TEST_F(ProcTest, CloneProcInstantiationWithChannels) {
  constexpr std::string_view kIrText = R"(package test
proc spawnee<spin: bits[32] in, spout: bits[32] out>(__state: (), init={()}) {
  chan_interface spin(direction=receive, kind=streaming, strictness=proven_mutually_exclusive, flow_control=ready_valid, flop_kind=none)
  chan_interface spout(direction=send, kind=streaming, strictness=proven_mutually_exclusive, flow_control=ready_valid, flop_kind=none)
}

top proc main<mainin: bits[32] in, mainout: bits[32] out>(__state: (), init={()}) {
  chan_interface mainin(direction=receive, kind=streaming, strictness=proven_mutually_exclusive, flow_control=ready_valid, flop_kind=none)
  chan_interface mainout(direction=send, kind=streaming, strictness=proven_mutually_exclusive, flow_control=ready_valid, flop_kind=none)
  chan declared(bits[32], id=0, kind=streaming, ops=send_receive, flow_control=ready_valid, strictness=proven_mutually_exclusive)
  chan_interface declared(direction=send, kind=streaming, strictness=proven_mutually_exclusive, flow_control=none, flop_kind=none)
  chan_interface declared(direction=receive, kind=streaming, strictness=proven_mutually_exclusive, flow_control=none, flop_kind=none)
  proc_instantiation spawnee(mainin, declared, proc=spawnee)
}
)";
  auto target = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> source,
                           Parser::ParsePackage(kIrText));
  XLS_ASSERT_OK_AND_ASSIGN(Proc * spawnee, source->GetProc("spawnee"));

  XLS_ASSERT_OK_AND_ASSIGN(
      Proc * cloned_spawnee,
      spawnee->Clone("cloned_spawnee", target.get(),
                     /*channel_remapping=*/
                     {{"spin", "newspin"}, {"spout", "newspout"}}));

  XLS_ASSERT_OK_AND_ASSIGN(Proc * main, source->GetProc("main"));
  XLS_ASSERT_OK(main->GetProcInstantiation("spawnee"));
  XLS_ASSERT_OK_AND_ASSIGN(
      Proc * clone,
      main->Clone(
          "cloned", target.get(),
          /*channel_remapping=*/
          {{"declared", "newdeclared"},
           {"mainin", "newmainin"},
           {"mainout", "newmainout"}},
          /*call_remapping=*/{},
          /*state_name_remapping=*/{},
          /*original_node_to_clone=*/std::nullopt,
          /*spawned_proc_name_remapping=*/{{"spawnee", "cloned_spawnee"}}));
  EXPECT_TRUE(clone->IsProc());
  EXPECT_FALSE(clone->GetProcInstantiation("spawnee").status().ok());
  XLS_ASSERT_OK_AND_ASSIGN(ProcInstantiation * instantiation,
                           clone->GetProcInstantiation("cloned_spawnee"));
  EXPECT_EQ(instantiation->name(), "cloned_spawnee");
  EXPECT_EQ(instantiation->proc(), cloned_spawnee);
  EXPECT_EQ(instantiation->channel_args().at(0)->name(), "newmainin");
  EXPECT_EQ(instantiation->channel_args().at(1)->name(), "newdeclared");
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

class ScheduledProcTest : public IrTestBase {
 protected:
  absl::StatusOr<ScheduledProc*> CreateScheduledProc(Package* p) {
    ScheduledProcBuilder pb("p", p);
    pb.StateElement("st", Value(UBits(42, 32)));
    return pb.Build({});
  }
};

TEST_F(ScheduledProcTest, StageAddAndClear) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(ScheduledProc * proc, CreateScheduledProc(p.get()));
  EXPECT_EQ(proc->stages().size(),
            1);  // Starts with one stage from CreateScheduledProc

  Stage stage1;
  proc->AddStage(stage1);
  EXPECT_EQ(proc->stages().size(), 2);

  Stage stage2;
  proc->AddStage(stage2);
  EXPECT_EQ(proc->stages().size(), 3);

  proc->ClearStages();
  EXPECT_TRUE(proc->stages().empty());
  // Re-stage the state element to satisfy the verifier.
  XLS_ASSERT_OK(
      proc->AddNodeToStage(0, proc->GetStateRead(int64_t{0})).status());
}

TEST_F(ScheduledProcTest, AddEmptyStages) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(ScheduledProc * proc, CreateScheduledProc(p.get()));
  EXPECT_EQ(proc->stages().size(), 1);  // Starts with one stage

  proc->AddEmptyStages(3);
  EXPECT_EQ(proc->stages().size(), 4);  // 1 initial + 3 empty
  for (int i = 1; i < proc->stages().size();
       ++i) {  // Check only the added empty stages
    EXPECT_TRUE(proc->stages()[i].active_inputs().empty());
    EXPECT_TRUE(proc->stages()[i].logic().empty());
    EXPECT_TRUE(proc->stages()[i].active_outputs().empty());
  }

  proc->AddEmptyStages(0);
  EXPECT_EQ(proc->stages().size(), 4);
}

TEST_F(ScheduledProcTest, GetStageIndex) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(ScheduledProc * proc, CreateScheduledProc(p.get()));
  // CreateScheduledProc adds "st" to stage 0.
  proc->AddEmptyStages(2);  // Adds stage 1 and 2

  XLS_ASSERT_OK_AND_ASSIGN(Node * x, proc->MakeNodeInStage<Literal>(
                                         1, SourceInfo(), Value(UBits(1, 32))));

  XLS_ASSERT_OK_AND_ASSIGN(
      Node * y, proc->MakeNode<Literal>(SourceInfo(), Value(UBits(2, 32))));
  ASSERT_THAT(proc->AddNodeToStage(2, y), IsOkAndHolds(true));

  XLS_ASSERT_OK_AND_ASSIGN(Node * add,
                           proc->MakeNode<BinOp>(SourceInfo(), x, y, Op::kAdd));

  EXPECT_THAT(proc->GetStageIndex(x), IsOkAndHolds(1));
  EXPECT_THAT(proc->GetStageIndex(y), IsOkAndHolds(2));
  EXPECT_THAT(proc->GetStageIndex(add), StatusIs(absl::StatusCode::kNotFound));
  EXPECT_THAT(proc->GetStageIndex(proc->GetStateRead(int64_t{0})),
              IsOkAndHolds(0));

  // The verifier requires that every node be in a stage before we finish.
  ASSERT_THAT(proc->AddNodeToStage(2, add), IsOkAndHolds(true));
}

TEST_F(ScheduledProcTest, AddNodeToStage) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(ScheduledProc * proc, CreateScheduledProc(p.get()));
  // CreateScheduledProc adds "st" to stage 0.
  proc->AddEmptyStages(1);  // Adds stage 1

  XLS_ASSERT_OK_AND_ASSIGN(
      Node * x, proc->MakeNode<Literal>(SourceInfo(), Value(UBits(1, 32))));
  XLS_ASSERT_OK_AND_ASSIGN(
      Node * y, proc->MakeNode<Literal>(SourceInfo(), Value(UBits(2, 32))));

  XLS_ASSERT_OK_AND_ASSIGN(bool added_x, proc->AddNodeToStage(0, x));
  EXPECT_TRUE(added_x);
  EXPECT_THAT(proc->GetStageIndex(x), IsOkAndHolds(0));
  EXPECT_TRUE(proc->stages()[0].contains(x));

  XLS_ASSERT_OK_AND_ASSIGN(bool added_y, proc->AddNodeToStage(1, y));
  EXPECT_TRUE(added_y);
  EXPECT_EQ(proc->stages().size(), 2);
  EXPECT_THAT(proc->GetStageIndex(y), IsOkAndHolds(1));
  EXPECT_TRUE(proc->stages()[1].contains(y));

  // Adding an existing node should return false.
  EXPECT_THAT(proc->AddNodeToStage(0, x), IsOkAndHolds(false));
}

TEST_F(ScheduledProcTest, MakeNodeInStage) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(ScheduledProc * proc, CreateScheduledProc(p.get()));
  proc->AddEmptyStages(1);

  XLS_ASSERT_OK_AND_ASSIGN(
      Literal * literal,
      proc->MakeNodeInStage<Literal>(0, SourceInfo(), Value(UBits(42, 32))));
  EXPECT_THAT(proc->GetStageIndex(literal), IsOkAndHolds(0));
}

TEST_F(ScheduledProcTest, MakeNodeWithNameInStage) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(ScheduledProc * proc, CreateScheduledProc(p.get()));
  proc->AddEmptyStages(1);

  XLS_ASSERT_OK_AND_ASSIGN(
      Literal * literal, proc->MakeNodeWithNameInStage<Literal>(
                             0, SourceInfo(), Value(UBits(42, 32)), "my_lit"));
  EXPECT_THAT(proc->GetStageIndex(literal), IsOkAndHolds(0));
  EXPECT_EQ(literal->GetName(), "my_lit");
}

// RebuildStageSideTables is implicitly tested by CloneScheduledProc.

TEST_F(ScheduledProcTest, CloneScheduledProc) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(ScheduledProc * proc, CreateScheduledProc(p.get()));
  // CreateScheduledProc adds "st" to stage 0.

  proc->AddEmptyStages(2);  // Adds stage 1 and 2

  XLS_ASSERT_OK(proc->MakeNodeWithNameInStage<Literal>(
      0, SourceInfo(), Value(UBits(1, 32)), "my_x"));
  XLS_ASSERT_OK(proc->MakeNodeWithNameInStage<Literal>(
      0, SourceInfo(), Value(UBits(2, 32)), "my_y"));
  XLS_ASSERT_OK(proc->MakeNodeWithNameInStage<Literal>(
      1, SourceInfo(), Value(UBits(3, 32)), "my_z"));

  XLS_ASSERT_OK_AND_ASSIGN(Proc * cloned_proc,
                           proc->Clone("cloned", p.get(),
                                       /*channel_remapping=*/{},
                                       /*call_remapping=*/{},
                                       /*state_name_remapping=*/{}));

  EXPECT_EQ(cloned_proc->stages().size(), 3);  // 1 initial + 2 added
  EXPECT_TRUE(cloned_proc->IsScheduled());

  Node* cloned_st = FindNode("st", cloned_proc);
  Node* cloned_x = FindNode("my_x", cloned_proc);
  Node* cloned_y = FindNode("my_y", cloned_proc);
  Node* cloned_z = FindNode("my_z", cloned_proc);

  EXPECT_THAT(cloned_proc->GetStageIndex(cloned_st), IsOkAndHolds(0));
  EXPECT_THAT(cloned_proc->GetStageIndex(cloned_x), IsOkAndHolds(0));
  EXPECT_THAT(cloned_proc->GetStageIndex(cloned_y), IsOkAndHolds(0));
  EXPECT_THAT(cloned_proc->GetStageIndex(cloned_z), IsOkAndHolds(1));

  // Verify Stage contents
  EXPECT_TRUE(cloned_proc->stages()[0].contains(cloned_st));
  EXPECT_TRUE(cloned_proc->stages()[0].contains(cloned_x));
  EXPECT_TRUE(cloned_proc->stages()[0].contains(cloned_y));
  EXPECT_FALSE(cloned_proc->stages()[0].contains(cloned_z));
  EXPECT_FALSE(cloned_proc->stages()[1].contains(cloned_st));
  EXPECT_FALSE(cloned_proc->stages()[1].contains(cloned_x));
  EXPECT_FALSE(cloned_proc->stages()[1].contains(cloned_y));
  EXPECT_TRUE(cloned_proc->stages()[1].contains(cloned_z));
}

}  // namespace
}  // namespace xls
