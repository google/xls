// Copyright 2021 The XLS Authors
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

#include "xls/codegen/port_legalization_pass.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "xls/codegen/codegen_pass.h"
#include "xls/common/status/matchers.h"
#include "xls/ir/channel.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/ir_test_base.h"

namespace xls::verilog {
namespace {

using status_testing::IsOkAndHolds;

class PortLegalizationPassTest : public IrTestBase {
 protected:
  absl::StatusOr<bool> Run(Proc* proc) {
    PassResults results;
    CodegenPassUnit unit(proc->package(), proc);
    return PortLegalizationPass().Run(&unit, CodegenPassOptions(), &results);
  }
};

TEST_F(PortLegalizationPassTest, APlusB) {
  auto p = CreatePackage();

  Type* u32 = p->GetBitsType(32);
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * a_ch, p->CreatePortChannel("a", ChannelOps::kReceiveOnly, u32));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * b_ch, p->CreatePortChannel("b", ChannelOps::kReceiveOnly, u32));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * output_ch,
      p->CreatePortChannel("output", ChannelOps::kSendOnly, u32));

  TokenlessProcBuilder pb(TestName(), /*init_value=*/Value::Tuple({}),
                          /*token_name=*/"tkn", /*state_name=*/"st", p.get());
  BValue a = pb.Receive(a_ch);
  BValue b = pb.Receive(b_ch);
  pb.Send(output_ch, pb.Add(a, b));
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, pb.Build(pb.GetStateParam()));

  // There are no zero-width ports so nothing should be removed.
  EXPECT_EQ(proc->GetPorts().value().size(), 3);
  ASSERT_THAT(Run(proc), IsOkAndHolds(false));
  EXPECT_EQ(proc->GetPorts().value().size(), 3);
}

TEST_F(PortLegalizationPassTest, ZeroWidthInputsAndOutput) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * a_ch,
      p->CreatePortChannel("a", ChannelOps::kReceiveOnly, p->GetBitsType(0)));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * b_ch,
      p->CreatePortChannel("b", ChannelOps::kReceiveOnly, p->GetTupleType({})));
  XLS_ASSERT_OK_AND_ASSIGN(Channel * output_ch,
                           p->CreatePortChannel("output", ChannelOps::kSendOnly,
                                                p->GetTupleType({})));

  TokenlessProcBuilder pb(TestName(), /*init_value=*/Value::Tuple({}),
                          /*token_name=*/"tkn", /*state_name=*/"st", p.get());
  pb.Receive(a_ch);
  pb.Send(output_ch, pb.Receive(b_ch));
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, pb.Build(pb.GetStateParam()));

  // All ports should be removed.
  EXPECT_EQ(proc->GetPorts().value().size(), 3);
  ASSERT_THAT(Run(proc), IsOkAndHolds(true));
  EXPECT_EQ(proc->GetPorts().value().size(), 0);
}

TEST_F(PortLegalizationPassTest, ZeroWidthInput) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * a_ch,
      p->CreatePortChannel("a", ChannelOps::kReceiveOnly, p->GetBitsType(0)));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * b_ch,
      p->CreatePortChannel("b", ChannelOps::kReceiveOnly, p->GetBitsType(32)));
  XLS_ASSERT_OK_AND_ASSIGN(Channel * output_ch,
                           p->CreatePortChannel("output", ChannelOps::kSendOnly,
                                                p->GetBitsType(32)));

  TokenlessProcBuilder pb(TestName(), /*init_value=*/Value::Tuple({}),
                          /*token_name=*/"tkn", /*state_name=*/"st", p.get());
  pb.Receive(a_ch);
  pb.Send(output_ch, pb.Receive(b_ch));
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, pb.Build(pb.GetStateParam()));

  // One port should be removed.
  EXPECT_EQ(proc->GetPorts().value().size(), 3);
  ASSERT_THAT(Run(proc), IsOkAndHolds(true));
  XLS_ASSERT_OK_AND_ASSIGN(std::vector<Proc::Port> ports, proc->GetPorts());
  ASSERT_EQ(ports.size(), 2);
  EXPECT_EQ(ports[0].channel->name(), "b");
  EXPECT_EQ(ports[1].channel->name(), "output");
}

TEST_F(PortLegalizationPassTest, ZeroWidthInputWithPosition) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(
      PortChannel * a_ch,
      p->CreatePortChannel("a", ChannelOps::kReceiveOnly, p->GetBitsType(0)));
  a_ch->SetPosition(0);
  XLS_ASSERT_OK_AND_ASSIGN(
      PortChannel * b_ch,
      p->CreatePortChannel("b", ChannelOps::kReceiveOnly, p->GetBitsType(32)));
  b_ch->SetPosition(1);
  XLS_ASSERT_OK_AND_ASSIGN(PortChannel * output_ch,
                           p->CreatePortChannel("output", ChannelOps::kSendOnly,
                                                p->GetBitsType(32)));
  output_ch->SetPosition(2);

  TokenlessProcBuilder pb(TestName(), /*init_value=*/Value::Tuple({}),
                          /*token_name=*/"tkn", /*state_name=*/"st", p.get());
  pb.Receive(a_ch);
  pb.Send(output_ch, pb.Receive(b_ch));
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, pb.Build(pb.GetStateParam()));

  // All ports should be removed.
  EXPECT_EQ(proc->GetPorts().value().size(), 3);
  ASSERT_THAT(Run(proc), IsOkAndHolds(true));
  XLS_ASSERT_OK_AND_ASSIGN(std::vector<Proc::Port> ports, proc->GetPorts());
  ASSERT_EQ(ports.size(), 2);
  EXPECT_EQ(ports[0].channel->name(), "b");
  EXPECT_EQ(ports[0].channel->GetPosition(), 0);
  EXPECT_EQ(ports[1].channel->name(), "output");
  EXPECT_EQ(ports[1].channel->GetPosition(), 1);
}

}  // namespace
}  // namespace xls::verilog
