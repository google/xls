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

using status_testing::IsOkAndHolds;
using status_testing::StatusIs;
using ::testing::ElementsAre;
using ::testing::HasSubstr;

class ProcTest : public IrTestBase {};

TEST_F(ProcTest, SimpleProc) {
  auto p = CreatePackage();
  ProcBuilder pb("p", /*init_value=*/Value(UBits(42, 32)), "tkn", "st",
                 p.get());
  BValue add = pb.Add(pb.Literal(UBits(1, 32)), pb.GetStateParam());
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, pb.Build(pb.GetTokenParam(), add));

  EXPECT_FALSE(proc->IsFunction());
  EXPECT_TRUE(proc->IsProc());
  EXPECT_THAT(proc->GetPorts(), IsOkAndHolds(ElementsAre()));

  EXPECT_EQ(proc->DumpIr(), R"(proc p(tkn: token, st: bits[32], init=42) {
  literal.3: bits[32] = literal(value=1, id=3)
  add.4: bits[32] = add(literal.3, st, id=4)
  next (tkn, add.4)
}
)");
}

TEST_F(ProcTest, MutateProc) {
  auto p = CreatePackage();
  ProcBuilder pb("p", /*init_value=*/Value(UBits(42, 32)), "tkn", "st",
                 p.get());
  BValue add = pb.Add(pb.Literal(UBits(1, 32)), pb.GetStateParam());
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc,
                           pb.Build(pb.AfterAll({pb.GetTokenParam()}), add));

  EXPECT_THAT(proc->NextToken(), m::AfterAll(m::Param("tkn")));
  XLS_ASSERT_OK(proc->SetNextToken(proc->TokenParam()));
  EXPECT_THAT(proc->NextToken(), m::Param("tkn"));

  EXPECT_THAT(proc->NextState(), m::Add());
  XLS_ASSERT_OK(proc->SetNextState(proc->StateParam()));
  EXPECT_THAT(proc->NextState(), m::Param("st"));

  // Replace the state with a new type. First need to delete the (dead) use of
  // the existing state param.
  XLS_ASSERT_OK(proc->RemoveNode(add.node()));
  XLS_ASSERT_OK_AND_ASSIGN(
      Node * new_state,
      proc->MakeNode<Literal>(/*loc=*/absl::nullopt, Value(UBits(100, 100))));
  XLS_ASSERT_OK(proc->ReplaceState("new_state", new_state));

  EXPECT_THAT(proc->NextState(), m::Literal(UBits(100, 100)));
  EXPECT_THAT(proc->StateParam(), m::Type("bits[100]"));
}

TEST_F(ProcTest, InvalidTokenType) {
  auto p = CreatePackage();
  ProcBuilder pb("p", /*init_value=*/Value(UBits(42, 32)), "tkn", "st",
                 p.get());
  BValue add = pb.Add(pb.Literal(UBits(1, 32)), pb.GetStateParam());
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc,
                           pb.Build(pb.AfterAll({pb.GetTokenParam()}), add));

  // Try setting invalid typed nodes as the next token/state.
  EXPECT_THAT(
      proc->SetNextToken(add.node()),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr(
                   "Cannot set next token to \"add.4\", expected token type")));
}

TEST_F(ProcTest, InvalidStateType) {
  auto p = CreatePackage();
  ProcBuilder pb("p", /*init_value=*/Value(UBits(42, 32)), "tkn", "st",
                 p.get());
  BValue add = pb.Add(pb.Literal(UBits(1, 32)), pb.GetStateParam());
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc,
                           pb.Build(pb.AfterAll({pb.GetTokenParam()}), add));

  EXPECT_THAT(proc->SetNextState(proc->TokenParam()),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Cannot set next state to \"tkn\"; type token "
                                 "does not match proc state type bits[32]")));
}

TEST_F(ProcTest, ReplaceStateThatStillHasUse) {
  auto p = CreatePackage();
  ProcBuilder pb("p", /*init_value=*/Value(UBits(42, 32)), "tkn", "st",
                 p.get());
  BValue add = pb.Add(pb.Literal(UBits(1, 32)), pb.GetStateParam());
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc,
                           pb.Build(pb.AfterAll({pb.GetTokenParam()}), add));

  XLS_ASSERT_OK_AND_ASSIGN(
      Node * new_state,
      proc->MakeNode<Literal>(/*loc=*/absl::nullopt, Value(UBits(100, 100))));
  EXPECT_THAT(
      proc->ReplaceState("new_state", new_state),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("Existing state param \"st\" still has uses")));
}

TEST_F(ProcTest, TestPorts) {
  Package package(TestName());

  Type* u32 = package.GetBitsType(32);
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * a_ch,
      package.CreatePortChannel("a", ChannelOps::kReceiveOnly, u32));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * b_ch,
      package.CreatePortChannel("b", ChannelOps::kReceiveOnly, u32));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * output_ch,
      package.CreatePortChannel("sum", ChannelOps::kSendOnly, u32));

  TokenlessProcBuilder pb(TestName(), /*init_value=*/Value::Tuple({}),
                          /*token_name=*/"tkn", /*state_name=*/"st", &package);
  BValue a = pb.Receive(a_ch);
  BValue b = pb.Receive(b_ch);
  pb.Send(output_ch, pb.Add(a, b));
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, pb.Build(pb.GetStateParam()));
  XLS_ASSERT_OK_AND_ASSIGN(std::vector<Proc::Port> ports, proc->GetPorts());
  EXPECT_EQ(ports.size(), 3);

  EXPECT_EQ(ports[0].channel, a_ch);
  EXPECT_EQ(ports[0].direction, Proc::PortDirection::kInput);
  EXPECT_THAT(ports[0].node, m::Receive(a_ch));

  EXPECT_EQ(ports[1].channel, b_ch);
  EXPECT_EQ(ports[1].direction, Proc::PortDirection::kInput);
  EXPECT_THAT(ports[1].node, m::Receive(b_ch));

  EXPECT_EQ(ports[2].channel, output_ch);
  EXPECT_EQ(ports[2].direction, Proc::PortDirection::kOutput);
  EXPECT_THAT(ports[2].node, m::Send(output_ch));
}

}  // namespace
}  // namespace xls
