// Copyright 2025 The XLS Authors
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

#include "xls/codegen_v_1_5/channel_to_port_io_lowering_pass.h"

#include <memory>
#include <optional>
#include <string_view>
#include <utility>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "cppitertools/reversed.hpp"
#include "cppitertools/zip.hpp"
#include "xls/codegen/codegen_options.h"
#include "xls/codegen_v_1_5/block_conversion_pass.h"
#include "xls/codegen_v_1_5/scheduled_block_conversion_pass.h"
#include "xls/common/status/matchers.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/bits.h"
#include "xls/ir/block.h"
#include "xls/ir/channel.h"
#include "xls/ir/channel_ops.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/ir_matcher.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/node.h"
#include "xls/ir/nodes.h"
#include "xls/ir/op.h"
#include "xls/ir/package.h"
#include "xls/ir/scheduled_builder.h"
#include "xls/ir/value.h"
#include "xls/passes/pass_base.h"

namespace xls::codegen {
namespace {

namespace m = ::xls::op_matchers;

using ::absl_testing::IsOk;
using ::absl_testing::IsOkAndHolds;
using ::testing::_;
using ::testing::Contains;
using ::testing::HasSubstr;
using ::testing::Not;
using ::testing::SizeIs;
using ::testing::UnorderedElementsAre;

class ChannelToPortIoLoweringPassTest : public IrTestBase {
 protected:
  absl::StatusOr<bool> Run(Package* p, BlockConversionPassOptions options =
                                           BlockConversionPassOptions()) {
    PassResults results;
    BlockConversionContext context;

    // Set default options if not provided
    if (!options.codegen_options.clock_name().has_value()) {
      options.codegen_options.clock_name("clk");
    }
    if (!options.codegen_options.reset().has_value()) {
      options.codegen_options.reset("rst", false, false, false);
    }

    // 1. Convert scheduled_proc -> scheduled_block
    // We assume the input package has a scheduled proc as top.
    XLS_RETURN_IF_ERROR(ScheduledBlockConversionPass()
                            .Run(p, options, &results, context)
                            .status());

    // 2. Run the pass under test
    return ChannelToPortIoLoweringPass().Run(p, options, &results, context);
  }
};

TEST_F(ChannelToPortIoLoweringPassTest, LowerSingleInputChannel) {
  auto p = std::make_unique<Package>("test");
  ScheduledProcBuilder pb(NewStyleProc(), "test_main", p.get());
  XLS_ASSERT_OK_AND_ASSIGN(auto a_in,
                           pb.AddInputChannel("a_in", p->GetBitsType(32)));
  BValue tkn = pb.Literal(Value::Token());
  pb.Receive(a_in, tkn);
  XLS_ASSERT_OK(pb.Build());

  EXPECT_THAT(Run(p.get()), IsOkAndHolds(true));

  XLS_ASSERT_OK_AND_ASSIGN(Block * block, p->GetBlock("test_main"));

  EXPECT_THAT(block->GetInputPort("a_in"), IsOk());
  EXPECT_THAT(block->GetInputPort("a_in_vld"), IsOk());
  EXPECT_THAT(block->GetOutputPort("a_in_rdy"), IsOk());
}

TEST_F(ChannelToPortIoLoweringPassTest, LowerSingleInputChannelValidData) {
  auto p = std::make_unique<Package>("test");
  ScheduledProcBuilder pb(NewStyleProc(), "test_main", p.get());
  XLS_ASSERT_OK_AND_ASSIGN(
      auto a_in,
      pb.AddInputChannel("a_in", p->GetBitsType(32), ChannelKind::kStreaming,
                         std::nullopt, FlowControl::kValidData));
  BValue tkn = pb.Literal(Value::Token());
  pb.Receive(a_in, tkn);
  XLS_ASSERT_OK(pb.Build());

  EXPECT_THAT(Run(p.get()), IsOkAndHolds(true));

  XLS_ASSERT_OK_AND_ASSIGN(Block * block, p->GetBlock("test_main"));

  EXPECT_THAT(block->GetInputPort("a_in"), IsOk());
  EXPECT_THAT(block->GetInputPort("a_in_vld"), IsOk());
  EXPECT_THAT(block->GetOutputPort("a_in_rdy"), Not(IsOk()));
}

TEST_F(ChannelToPortIoLoweringPassTest, LowerSingleOutputChannel) {
  auto p = std::make_unique<Package>("test");
  ScheduledProcBuilder pb(NewStyleProc(), "test_main", p.get());
  XLS_ASSERT_OK_AND_ASSIGN(auto a_out,
                           pb.AddOutputChannel("a_out", p->GetBitsType(32)));
  BValue tkn = pb.Literal(Value::Token());
  BValue lit = pb.Literal(Value(UBits(123, 32)));
  pb.Send(a_out, tkn, lit);
  XLS_ASSERT_OK(pb.Build());

  EXPECT_THAT(Run(p.get()), IsOkAndHolds(true));

  XLS_ASSERT_OK_AND_ASSIGN(Block * block, p->GetBlock("test_main"));

  EXPECT_THAT(block->GetOutputPort("a_out"), IsOk());
  EXPECT_THAT(block->GetOutputPort("a_out_vld"), IsOk());
  EXPECT_THAT(block->GetInputPort("a_out_rdy"), IsOk());
}

TEST_F(ChannelToPortIoLoweringPassTest, LowerSingleOutputChannelValidData) {
  auto p = std::make_unique<Package>("test");
  ScheduledProcBuilder pb(NewStyleProc(), "test_main", p.get());
  XLS_ASSERT_OK_AND_ASSIGN(
      auto a_out,
      pb.AddOutputChannel("a_out", p->GetBitsType(32), ChannelKind::kStreaming,
                          std::nullopt, FlowControl::kValidData));
  BValue tkn = pb.Literal(Value::Token());
  BValue lit = pb.Literal(Value(UBits(123, 32)));
  pb.Send(a_out, tkn, lit);
  XLS_ASSERT_OK(pb.Build());

  EXPECT_THAT(Run(p.get()), IsOkAndHolds(true));

  XLS_ASSERT_OK_AND_ASSIGN(Block * block, p->GetBlock("test_main"));

  EXPECT_THAT(block->GetOutputPort("a_out"), IsOk());
  EXPECT_THAT(block->GetOutputPort("a_out_vld"), IsOk());
  EXPECT_THAT(block->GetInputPort("a_out_rdy"), Not(IsOk()));
}

TEST_F(ChannelToPortIoLoweringPassTest, LowerInternalLoopbackChannelOldStyle) {
  auto p = std::make_unique<Package>("test");
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * internal_ch,
      p->CreateStreamingChannel("internal_ch", ChannelOps::kSendReceive,
                                p->GetBitsType(32), {},
                                ChannelConfig()
                                    .WithFifoConfig(FifoConfig(
                                        /*depth=*/1,
                                        /*bypass=*/false,
                                        /*register_push_outputs=*/true,
                                        /*register_pop_outputs=*/false))
                                    .WithInputFlopKind(FlopKind::kNone)
                                    .WithOutputFlopKind(FlopKind::kNone)));
  ScheduledProcBuilder pb("test_main", p.get());
  BValue tkn = pb.Literal(Value::Token());
  BValue lit = pb.Literal(Value(UBits(123, 32)));
  pb.Send(internal_ch, tkn, lit);
  pb.Receive(internal_ch, tkn);
  XLS_ASSERT_OK(pb.Build());

  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));

  XLS_ASSERT_OK_AND_ASSIGN(Block * block, p->GetBlock("test_main"));

  // Should NOT have ports for internal channel
  EXPECT_THAT(block->GetInputPort("internal_ch"), Not(IsOk()));
  EXPECT_THAT(block->GetOutputPort("internal_ch"), Not(IsOk()));

  // Should have FIFO instantiation
  EXPECT_THAT(block->nodes(), Contains(AnyOf(m::InstantiationInput(),
                                             m::InstantiationOutput())));
}

TEST_F(ChannelToPortIoLoweringPassTest, MultiOutputWithOneShotLogic) {
  // Two output channels, not mutually exclusive. Should generate one-shot logic
  // (already_done registers).
  auto p = std::make_unique<Package>("test");
  ScheduledProcBuilder pb(NewStyleProc(), "test_main", p.get());
  XLS_ASSERT_OK_AND_ASSIGN(auto ch_a,
                           pb.AddOutputChannel("ch_a", p->GetBitsType(32)));
  XLS_ASSERT_OK_AND_ASSIGN(auto ch_b,
                           pb.AddOutputChannel("ch_b", p->GetBitsType(32)));

  BValue tkn = pb.Literal(Value::Token());
  BValue lit = pb.Literal(Value(UBits(123, 32)));
  pb.Send(ch_a, tkn, lit);
  pb.Send(ch_b, tkn, lit);
  XLS_ASSERT_OK(pb.Build());

  EXPECT_THAT(Run(p.get()), IsOkAndHolds(true));

  XLS_ASSERT_OK_AND_ASSIGN(Block * block, p->GetBlock("test_main"));

  // Check for registers named something like "*already_done_reg"
  EXPECT_THAT(block->nodes(),
              Contains(m::RegisterRead(HasSubstr("already_done_reg"))));
}

TEST_F(ChannelToPortIoLoweringPassTest, NoOneShotLogicForSingleValueChannel) {
  // Three output channels, not mutually exclusive, one of them single-value.
  // Should generate one-shot logic (already_done registers) but only for the
  // streaming ones.
  auto p = std::make_unique<Package>("test");
  ScheduledProcBuilder pb(NewStyleProc(), "test_main", p.get());
  XLS_ASSERT_OK_AND_ASSIGN(auto ch_a,
                           pb.AddOutputChannel("ch_a", p->GetBitsType(32)));
  XLS_ASSERT_OK_AND_ASSIGN(auto ch_b,
                           pb.AddOutputChannel("ch_b", p->GetBitsType(32)));

  XLS_ASSERT_OK_AND_ASSIGN(auto ch_c,
                           pb.AddOutputChannel("ch_c", p->GetBitsType(32),
                                               ChannelKind::kSingleValue));

  BValue tkn = pb.Literal(Value::Token());
  BValue lit = pb.Literal(Value(UBits(123, 32)));
  pb.Send(ch_a, tkn, lit);
  pb.Send(ch_b, tkn, lit);
  pb.Send(ch_c, tkn, lit);
  XLS_ASSERT_OK(pb.Build());

  EXPECT_THAT(Run(p.get()), IsOkAndHolds(true));

  XLS_ASSERT_OK_AND_ASSIGN(Block * block, p->GetBlock("test_main"));

  // Check for registers named something like "*already_done_reg"
  EXPECT_THAT(block->nodes(),
              Contains(m::RegisterRead(HasSubstr("ch_a_already_done_reg"))));
  EXPECT_THAT(block->nodes(),
              Contains(m::RegisterRead(HasSubstr("ch_b_already_done_reg"))));
  EXPECT_THAT(
      block->nodes(),
      Not(Contains(m::RegisterRead(HasSubstr("ch_c_already_done_reg")))));
}

TEST_F(ChannelToPortIoLoweringPassTest,
       MutuallyExclusiveOutputsNoOneShotLogic) {
  // Two output channels, strictly mutually exclusive. Should NOT generate
  // one-shot logic.
  auto p = std::make_unique<Package>("test");
  ScheduledProcBuilder pb(NewStyleProc(), "test_main", p.get());
  XLS_ASSERT_OK_AND_ASSIGN(auto ch_in,
                           pb.AddInputChannel("ch_in", p->GetBitsType(1)));
  XLS_ASSERT_OK_AND_ASSIGN(auto ch_a,
                           pb.AddOutputChannel("ch_a", p->GetBitsType(32)));
  XLS_ASSERT_OK_AND_ASSIGN(auto ch_b,
                           pb.AddOutputChannel("ch_b", p->GetBitsType(32)));

  BValue tkn = pb.Literal(Value::Token());
  BValue lit = pb.Literal(Value(UBits(123, 32)));
  BValue recv = pb.Receive(ch_in, tkn);
  BValue pred = pb.TupleIndex(recv, 1);
  BValue not_pred = pb.Not(pred);
  BValue recv_tkn = pb.TupleIndex(recv, 0);

  pb.SendIf(ch_a, recv_tkn, pred, lit);
  pb.SendIf(ch_b, recv_tkn, not_pred, lit);
  XLS_ASSERT_OK(pb.Build());

  EXPECT_THAT(Run(p.get()), IsOkAndHolds(true));

  XLS_ASSERT_OK_AND_ASSIGN(Block * block, p->GetBlock("test_main"));

  // Check for registers named something like "*already_done_reg"
  EXPECT_THAT(block->nodes(),
              Not(Contains(m::RegisterRead(HasSubstr("already_done_reg")))));
}

TEST_F(ChannelToPortIoLoweringPassTest, LowerSingleValueInputChannel) {
  auto p = std::make_unique<Package>("test");
  ScheduledProcBuilder pb(NewStyleProc(), "test_main", p.get());
  XLS_ASSERT_OK_AND_ASSIGN(auto a_in,
                           pb.AddInputChannel("a_in", p->GetBitsType(32),
                                              ChannelKind::kSingleValue));

  BValue tkn = pb.Literal(Value::Token());
  pb.Receive(a_in, tkn);
  XLS_ASSERT_OK(pb.Build());

  EXPECT_THAT(Run(p.get()), IsOkAndHolds(true));

  XLS_ASSERT_OK_AND_ASSIGN(Block * block, p->GetBlock("test_main"));

  EXPECT_THAT(block->GetInputPort("a_in"), IsOk());
  EXPECT_THAT(block->GetInputPort("a_in_vld"), Not(IsOk()));
  EXPECT_THAT(block->GetOutputPort("a_in_rdy"), Not(IsOk()));
}

TEST_F(ChannelToPortIoLoweringPassTest, LowerSingleValueOutputChannel) {
  auto p = std::make_unique<Package>("test");
  ScheduledProcBuilder pb(NewStyleProc(), "test_main", p.get());
  XLS_ASSERT_OK_AND_ASSIGN(auto a_out,
                           pb.AddOutputChannel("a_out", p->GetBitsType(32),
                                               ChannelKind::kSingleValue));

  BValue tkn = pb.Literal(Value::Token());
  BValue lit = pb.Literal(Value(UBits(123, 32)));
  pb.Send(a_out, tkn, lit);
  XLS_ASSERT_OK(pb.Build());

  EXPECT_THAT(Run(p.get()), IsOkAndHolds(true));

  XLS_ASSERT_OK_AND_ASSIGN(Block * block, p->GetBlock("test_main"));

  EXPECT_THAT(block->GetOutputPort("a_out"), IsOk());
  EXPECT_THAT(block->GetOutputPort("a_out_vld"), Not(IsOk()));
  EXPECT_THAT(block->GetInputPort("a_out_rdy"), Not(IsOk()));
}

TEST_F(ChannelToPortIoLoweringPassTest, LowerMultipleSendsSameChannel) {
  auto p = std::make_unique<Package>("test");
  ScheduledProcBuilder pb(NewStyleProc(), "test_main", p.get());
  XLS_ASSERT_OK_AND_ASSIGN(auto a_out,
                           pb.AddOutputChannel("a_out", p->GetBitsType(32)));

  BValue tkn = pb.Literal(Value::Token());
  BValue lit1 = pb.Literal(Value(UBits(1, 32)));
  BValue lit2 = pb.Literal(Value(UBits(2, 32)));
  BValue p1 = pb.Literal(Value(UBits(1, 1)));
  BValue p2 = pb.Literal(Value(UBits(0, 1)));
  pb.SendIf(a_out, tkn, p1, lit1);
  pb.SendIf(a_out, tkn, p2, lit2);
  XLS_ASSERT_OK(pb.Build());

  EXPECT_THAT(Run(p.get()), IsOkAndHolds(true));

  XLS_ASSERT_OK_AND_ASSIGN(Block * block, p->GetBlock("test_main"));

  // Check that we have the ports.
  EXPECT_THAT(block->GetOutputPort("a_out"), IsOk());

  // Check that the data port is driven by a OneHotSelect (multiplexer)
  Node* data_port_driver =
      block->GetOutputPort("a_out").value()->output_source();
  EXPECT_THAT(data_port_driver, m::OneHotSelect());
}

TEST_F(ChannelToPortIoLoweringPassTest,
       LowerMultipleSendsSameChannelValidData) {
  auto p = std::make_unique<Package>("test");
  ScheduledProcBuilder pb(NewStyleProc(), "test_main", p.get());
  XLS_ASSERT_OK_AND_ASSIGN(
      auto a_out,
      pb.AddOutputChannel("a_out", p->GetBitsType(32), ChannelKind::kStreaming,
                          std::nullopt, FlowControl::kValidData));

  BValue tkn = pb.Literal(Value::Token());
  BValue lit1 = pb.Literal(Value(UBits(1, 32)));
  BValue lit2 = pb.Literal(Value(UBits(2, 32)));
  BValue p1 = pb.Literal(Value(UBits(1, 1)));
  BValue p2 = pb.Literal(Value(UBits(0, 1)));
  pb.SendIf(a_out, tkn, p1, lit1);
  pb.SendIf(a_out, tkn, p2, lit2);
  XLS_ASSERT_OK(pb.Build());

  EXPECT_THAT(Run(p.get()), IsOkAndHolds(true));

  XLS_ASSERT_OK_AND_ASSIGN(Block * block, p->GetBlock("test_main"));

  // Check that we have the ports.
  EXPECT_THAT(block->GetOutputPort("a_out"), IsOk());
  EXPECT_THAT(block->GetOutputPort("a_out_vld"), IsOk());
  EXPECT_THAT(block->GetInputPort("a_out_rdy"), Not(IsOk()));

  // Check that the data port is driven by a OneHotSelect (multiplexer)
  Node* data_port_driver =
      block->GetOutputPort("a_out").value()->output_source();
  EXPECT_THAT(data_port_driver, m::OneHotSelect());

  // Check that valid port is driven by OR of predicates
  Node* valid_port_driver =
      block->GetOutputPort("a_out_vld").value()->output_source();
  EXPECT_THAT(valid_port_driver, m::Or());
}

TEST_F(ChannelToPortIoLoweringPassTest, LowerMultipleReceivesSameChannel) {
  auto p = std::make_unique<Package>("test");
  ScheduledProcBuilder pb(NewStyleProc(), "test_main", p.get());
  XLS_ASSERT_OK_AND_ASSIGN(auto a_in,
                           pb.AddInputChannel("a_in", p->GetBitsType(32)));

  BValue tkn = pb.Literal(Value::Token());
  BValue p1 = pb.Literal(Value(UBits(1, 1)));
  BValue p2 = pb.Literal(Value(UBits(0, 1)));
  pb.ReceiveIf(a_in, tkn, p1);
  pb.ReceiveIf(a_in, tkn, p2);
  XLS_ASSERT_OK(pb.Build());

  EXPECT_THAT(Run(p.get()), IsOkAndHolds(true));

  XLS_ASSERT_OK_AND_ASSIGN(Block * block, p->GetBlock("test_main"));

  EXPECT_THAT(block->GetOutputPort("a_in_rdy"), IsOk());

  // Check that the ready port is driven by an OR of the individual receive
  // ready signals (which include predicates).
  Node* ready_port_driver =
      block->GetOutputPort("a_in_rdy").value()->output_source();
  // It should be an NaryOp(kOr) because we have multiple receives.
  EXPECT_THAT(ready_port_driver, m::Or());
}

TEST_F(ChannelToPortIoLoweringPassTest,
       LowerMultipleReceivesSameChannelValidData) {
  auto p = std::make_unique<Package>("test");
  ScheduledProcBuilder pb(NewStyleProc(), "test_main", p.get());
  XLS_ASSERT_OK_AND_ASSIGN(
      auto a_in,
      pb.AddInputChannel("a_in", p->GetBitsType(32), ChannelKind::kStreaming,
                         std::nullopt, FlowControl::kValidData));

  BValue tkn = pb.Literal(Value::Token());
  BValue p1 = pb.Literal(Value(UBits(1, 1)));
  BValue p2 = pb.Literal(Value(UBits(0, 1)));
  pb.ReceiveIf(a_in, tkn, p1);
  pb.ReceiveIf(a_in, tkn, p2);
  XLS_ASSERT_OK(pb.Build());

  EXPECT_THAT(Run(p.get()), IsOkAndHolds(true));

  XLS_ASSERT_OK_AND_ASSIGN(Block * block, p->GetBlock("test_main"));

  EXPECT_THAT(block->GetInputPort("a_in"), IsOk());
  EXPECT_THAT(block->GetInputPort("a_in_vld"), IsOk());
  EXPECT_THAT(block->GetOutputPort("a_in_rdy"), Not(IsOk()));
}

TEST_F(ChannelToPortIoLoweringPassTest, LowerInternalChannelNewStyle) {
  auto p = std::make_unique<Package>("test");
  ScheduledProcBuilder pb(NewStyleProc(), "test_main", p.get());
  XLS_ASSERT_OK_AND_ASSIGN(auto a_in,
                           pb.AddInputChannel("a_in", p->GetBitsType(32)));
  XLS_ASSERT_OK_AND_ASSIGN(
      ChannelWithInterfaces internal_ch,
      pb.AddChannel("internal_ch", p->GetBitsType(32), ChannelKind::kStreaming,
                    {},
                    ChannelConfig()
                        .WithFifoConfig(FifoConfig(
                            /*depth=*/1,
                            /*bypass=*/false,
                            /*register_push_outputs=*/true,
                            /*register_pop_outputs=*/false))
                        .WithInputFlopKind(FlopKind::kNone)
                        .WithOutputFlopKind(FlopKind::kNone)));

  BValue tkn = pb.Literal(Value::Token());
  pb.Receive(a_in, tkn);
  pb.Receive(internal_ch.receive_interface, tkn);
  BValue lit = pb.Literal(Value(UBits(123, 32)));
  pb.Send(internal_ch.send_interface, tkn, lit);
  XLS_ASSERT_OK(pb.Build());

  EXPECT_THAT(Run(p.get()), IsOkAndHolds(true));

  XLS_ASSERT_OK_AND_ASSIGN(Block * block, p->GetBlock("test_main"));

  // Should NOT have ports for internal channel
  EXPECT_THAT(block->GetInputPort("internal_ch"), Not(IsOk()));
  EXPECT_THAT(block->GetOutputPort("internal_ch"), Not(IsOk()));

  // Should have FIFO instantiation
  EXPECT_THAT(block->nodes(), Contains(AnyOf(m::InstantiationInput(),
                                             m::InstantiationOutput())));
}

TEST_F(ChannelToPortIoLoweringPassTest, StreamingInputFlop) {
  auto p = std::make_unique<Package>("test");
  ScheduledProcBuilder pb(NewStyleProc(), "test_main", p.get());
  XLS_ASSERT_OK_AND_ASSIGN(auto a_in,
                           pb.AddInputChannel("a_in", p->GetBitsType(32)));
  a_in->SetFlopKind(FlopKind::kFlop);

  BValue tkn = pb.Literal(Value::Token());
  pb.Receive(a_in, tkn);
  XLS_ASSERT_OK(pb.Build());

  EXPECT_THAT(Run(p.get()), IsOkAndHolds(true));
  XLS_ASSERT_OK_AND_ASSIGN(Block * block, p->GetBlock("test_main"));

  EXPECT_THAT(block->nodes(),
              Contains(m::RegisterRead(HasSubstr("__a_in_reg"))));
  EXPECT_THAT(block->nodes(),
              Contains(m::RegisterRead(HasSubstr("__a_in_valid_reg"))));
}

TEST_F(ChannelToPortIoLoweringPassTest, StreamingInputFlopValidData) {
  auto p = std::make_unique<Package>("test");
  ScheduledProcBuilder pb(NewStyleProc(), "test_main", p.get());
  XLS_ASSERT_OK_AND_ASSIGN(
      auto a_in,
      pb.AddInputChannel("a_in", p->GetBitsType(32), ChannelKind::kStreaming,
                         std::nullopt, FlowControl::kValidData));
  a_in->SetFlopKind(FlopKind::kFlop);

  BValue tkn = pb.Literal(Value::Token());
  pb.Receive(a_in, tkn);
  XLS_ASSERT_OK(pb.Build());

  EXPECT_THAT(Run(p.get()), IsOkAndHolds(true));
  XLS_ASSERT_OK_AND_ASSIGN(Block * block, p->GetBlock("test_main"));

  // Find the register writes
  RegisterWrite* data_write = nullptr;
  RegisterWrite* valid_write = nullptr;
  for (Node* node : block->nodes()) {
    if (node->Is<RegisterWrite>()) {
      RegisterWrite* w = node->As<RegisterWrite>();
      if (w->GetRegister()->name() == "__a_in_reg") {
        data_write = w;
      } else if (w->GetRegister()->name() == "__a_in_valid_reg") {
        valid_write = w;
      }
    }
  }
  ASSERT_NE(data_write, nullptr);
  ASSERT_NE(valid_write, nullptr);

  ASSERT_TRUE(valid_write->load_enable().has_value());
  EXPECT_THAT(*valid_write->load_enable(), m::Or());

  // Data register should be enabled by the input valid signal
  ASSERT_TRUE(data_write->load_enable().has_value());
  EXPECT_THAT(*data_write->load_enable(), m::And());
}

TEST_F(ChannelToPortIoLoweringPassTest, StreamingInputZeroLatency) {
  auto p = std::make_unique<Package>("test");
  ScheduledProcBuilder pb(NewStyleProc(), "test_main", p.get());
  XLS_ASSERT_OK_AND_ASSIGN(auto a_in,
                           pb.AddInputChannel("a_in", p->GetBitsType(32)));
  a_in->SetFlopKind(FlopKind::kZeroLatency);

  BValue tkn = pb.Literal(Value::Token());
  pb.Receive(a_in, tkn);
  XLS_ASSERT_OK(pb.Build());

  EXPECT_THAT(Run(p.get()), IsOkAndHolds(true));
  XLS_ASSERT_OK_AND_ASSIGN(Block * block, p->GetBlock("test_main"));

  EXPECT_THAT(block->nodes(),
              Contains(m::RegisterRead(HasSubstr("__a_in_skid_reg"))));
  EXPECT_THAT(block->nodes(),
              Contains(m::RegisterRead(HasSubstr("__a_in_valid_skid_reg"))));
}

TEST_F(ChannelToPortIoLoweringPassTest, StreamingInputSkid) {
  auto p = std::make_unique<Package>("test");
  ScheduledProcBuilder pb(NewStyleProc(), "test_main", p.get());
  XLS_ASSERT_OK_AND_ASSIGN(auto a_in,
                           pb.AddInputChannel("a_in", p->GetBitsType(32)));
  a_in->SetFlopKind(FlopKind::kSkid);

  BValue tkn = pb.Literal(Value::Token());
  pb.Receive(a_in, tkn);
  XLS_ASSERT_OK(pb.Build());

  EXPECT_THAT(Run(p.get()), IsOkAndHolds(true));
  XLS_ASSERT_OK_AND_ASSIGN(Block * block, p->GetBlock("test_main"));

  // Expect skid registers
  EXPECT_THAT(block->nodes(),
              Contains(m::RegisterRead(HasSubstr("__a_in_skid_reg"))));
  // Expect pipeline registers
  EXPECT_THAT(block->nodes(),
              Contains(m::RegisterRead(HasSubstr("__a_in_reg"))));
}

TEST_F(ChannelToPortIoLoweringPassTest, StreamingOutputFlop) {
  auto p = std::make_unique<Package>("test");
  ScheduledProcBuilder pb(NewStyleProc(), "test_main", p.get());
  XLS_ASSERT_OK_AND_ASSIGN(auto a_out,
                           pb.AddOutputChannel("a_out", p->GetBitsType(32)));
  a_out->SetFlopKind(FlopKind::kFlop);

  BValue tkn = pb.Literal(Value::Token());
  BValue lit = pb.Literal(Value(UBits(123, 32)));
  pb.Send(a_out, tkn, lit);
  XLS_ASSERT_OK(pb.Build());

  EXPECT_THAT(Run(p.get()), IsOkAndHolds(true));
  XLS_ASSERT_OK_AND_ASSIGN(Block * block, p->GetBlock("test_main"));

  EXPECT_THAT(block->nodes(),
              Contains(m::RegisterRead(HasSubstr("__a_out_reg"))));
  EXPECT_THAT(block->nodes(),
              Contains(m::RegisterRead(HasSubstr("__a_out_valid_reg"))));
}

TEST_F(ChannelToPortIoLoweringPassTest, StreamingOutputFlopValidData) {
  auto p = std::make_unique<Package>("test");
  ScheduledProcBuilder pb(NewStyleProc(), "test_main", p.get());
  XLS_ASSERT_OK_AND_ASSIGN(
      auto a_out,
      pb.AddOutputChannel("a_out", p->GetBitsType(32), ChannelKind::kStreaming,
                          std::nullopt, FlowControl::kValidData));
  a_out->SetFlopKind(FlopKind::kFlop);

  BValue tkn = pb.Literal(Value::Token());
  BValue lit = pb.Literal(Value(UBits(123, 32)));
  pb.Send(a_out, tkn, lit);
  XLS_ASSERT_OK(pb.Build());

  EXPECT_THAT(Run(p.get()), IsOkAndHolds(true));
  XLS_ASSERT_OK_AND_ASSIGN(Block * block, p->GetBlock("test_main"));

  // Find the register writes
  RegisterWrite* data_write = nullptr;
  RegisterWrite* valid_write = nullptr;
  for (Node* node : block->nodes()) {
    if (node->Is<RegisterWrite>()) {
      RegisterWrite* w = node->As<RegisterWrite>();
      if (w->GetRegister()->name() == "__a_out_reg") {
        data_write = w;
      } else if (w->GetRegister()->name() == "__a_out_valid_reg") {
        valid_write = w;
      }
    }
  }
  ASSERT_NE(data_write, nullptr);
  ASSERT_NE(valid_write, nullptr);

  ASSERT_TRUE(valid_write->load_enable().has_value());

  // Data register should be enabled by the valid signal
  ASSERT_TRUE(data_write->load_enable().has_value());
}

TEST_F(ChannelToPortIoLoweringPassTest, StreamingOutputSkid) {
  auto p = std::make_unique<Package>("test");
  ScheduledProcBuilder pb(NewStyleProc(), "test_main", p.get());
  XLS_ASSERT_OK_AND_ASSIGN(auto a_out,
                           pb.AddOutputChannel("a_out", p->GetBitsType(32)));
  a_out->SetFlopKind(FlopKind::kSkid);

  BValue tkn = pb.Literal(Value::Token());
  BValue lit = pb.Literal(Value(UBits(123, 32)));
  pb.Send(a_out, tkn, lit);
  XLS_ASSERT_OK(pb.Build());

  EXPECT_THAT(Run(p.get()), IsOkAndHolds(true));
  XLS_ASSERT_OK_AND_ASSIGN(Block * block, p->GetBlock("test_main"));

  EXPECT_THAT(block->nodes(),
              Contains(m::RegisterRead(HasSubstr("__a_out_skid_reg"))));
  EXPECT_THAT(block->nodes(),
              Contains(m::RegisterRead(HasSubstr("__a_out_reg"))));
}

TEST_F(ChannelToPortIoLoweringPassTest, SingleValueInputFlop) {
  auto p = std::make_unique<Package>("test");
  ScheduledProcBuilder pb(NewStyleProc(), "test_main", p.get());
  XLS_ASSERT_OK_AND_ASSIGN(auto a_in,
                           pb.AddInputChannel("a_in", p->GetBitsType(32),
                                              ChannelKind::kSingleValue));

  BValue tkn = pb.Literal(Value::Token());
  pb.Receive(a_in, tkn);
  XLS_ASSERT_OK(pb.Build());

  BlockConversionPassOptions options;
  options.codegen_options.flop_inputs(true);
  options.codegen_options.flop_single_value_channels(true);

  EXPECT_THAT(Run(p.get(), options), IsOkAndHolds(true));
  XLS_ASSERT_OK_AND_ASSIGN(Block * block, p->GetBlock("test_main"));

  EXPECT_THAT(block->nodes(),
              Contains(m::RegisterRead(HasSubstr("__a_in_reg"))));
}

TEST_F(ChannelToPortIoLoweringPassTest, SingleValueOutputFlop) {
  auto p = std::make_unique<Package>("test");
  ScheduledProcBuilder pb(NewStyleProc(), "test_main", p.get());
  XLS_ASSERT_OK_AND_ASSIGN(auto a_out,
                           pb.AddOutputChannel("a_out", p->GetBitsType(32),
                                               ChannelKind::kSingleValue));

  BValue tkn = pb.Literal(Value::Token());
  BValue lit = pb.Literal(Value(UBits(123, 32)));
  pb.Send(a_out, tkn, lit);
  XLS_ASSERT_OK(pb.Build());

  BlockConversionPassOptions options;
  options.codegen_options.flop_outputs(true);
  options.codegen_options.flop_single_value_channels(true);

  EXPECT_THAT(Run(p.get(), options), IsOkAndHolds(true));
  XLS_ASSERT_OK_AND_ASSIGN(Block * block, p->GetBlock("test_main"));

  EXPECT_THAT(block->nodes(),
              Contains(m::RegisterRead(HasSubstr("__a_out_reg"))));
}

TEST_F(ChannelToPortIoLoweringPassTest, GateRecvsWithPredicate) {
  auto p = std::make_unique<Package>("test");
  ScheduledProcBuilder pb(NewStyleProc(), "test_main", p.get());
  XLS_ASSERT_OK_AND_ASSIGN(auto a_in,
                           pb.AddInputChannel("a_in", p->GetBitsType(32)));
  BValue tkn = pb.Literal(Value::Token());
  BValue pred = pb.Literal(Value(UBits(1, 1)));
  pb.ReceiveIf(a_in, tkn, pred);
  XLS_ASSERT_OK(pb.Build());

  BlockConversionPassOptions options;
  options.codegen_options.gate_recvs(true);

  EXPECT_THAT(Run(p.get(), options), IsOkAndHolds(true));
  XLS_ASSERT_OK_AND_ASSIGN(Block * block, p->GetBlock("test_main"));

  EXPECT_THAT(
      block->nodes(),
      Contains(m::Select(m::Literal(1),
                         /*cases=*/{m::Literal(0), m::InputPort("a_in")})));
}

TEST_F(ChannelToPortIoLoweringPassTest, GateRecvsNonBlocking) {
  auto p = std::make_unique<Package>("test");
  ScheduledProcBuilder pb(NewStyleProc(), "test_main", p.get());
  XLS_ASSERT_OK_AND_ASSIGN(auto a_in,
                           pb.AddInputChannel("a_in", p->GetBitsType(32)));
  BValue tkn = pb.Literal(Value::Token());
  pb.ReceiveNonBlocking(a_in, tkn);
  XLS_ASSERT_OK(pb.Build());

  BlockConversionPassOptions options;
  options.codegen_options.gate_recvs(true);

  EXPECT_THAT(Run(p.get(), options), IsOkAndHolds(true));
  XLS_ASSERT_OK_AND_ASSIGN(Block * block, p->GetBlock("test_main"));

  EXPECT_THAT(
      block->nodes(),
      Contains(m::Select(m::InputPort("a_in_vld"),
                         /*cases=*/{m::Literal(0), m::InputPort("a_in")})));
}

TEST_F(ChannelToPortIoLoweringPassTest, GateRecvsNonBlockingWithPredicate) {
  auto p = std::make_unique<Package>("test");
  ScheduledProcBuilder pb(NewStyleProc(), "test_main", p.get());
  XLS_ASSERT_OK_AND_ASSIGN(auto a_in,
                           pb.AddInputChannel("a_in", p->GetBitsType(32)));
  BValue tkn = pb.Literal(Value::Token());
  BValue pred = pb.Literal(Value(UBits(1, 1)));
  pb.ReceiveIfNonBlocking(a_in, tkn, pred);
  XLS_ASSERT_OK(pb.Build());

  BlockConversionPassOptions options;
  options.codegen_options.gate_recvs(true);

  EXPECT_THAT(Run(p.get(), options), IsOkAndHolds(true));
  XLS_ASSERT_OK_AND_ASSIGN(Block * block, p->GetBlock("test_main"));

  EXPECT_THAT(
      block->nodes(),
      Contains(m::Select(m::And(m::Literal(1), m::InputPort("a_in_vld")),
                         /*cases=*/{m::Literal(0), m::InputPort("a_in")})));
}

TEST_F(ChannelToPortIoLoweringPassTest, GateRecvsBlockingNoPredicate) {
  auto p = std::make_unique<Package>("test");
  ScheduledProcBuilder pb(NewStyleProc(), "test_main", p.get());
  XLS_ASSERT_OK_AND_ASSIGN(auto a_in,
                           pb.AddInputChannel("a_in", p->GetBitsType(32)));
  BValue tkn = pb.Literal(Value::Token());
  pb.Receive(a_in, tkn);
  XLS_ASSERT_OK(pb.Build());

  BlockConversionPassOptions options;
  options.codegen_options.gate_recvs(true);

  EXPECT_THAT(Run(p.get(), options), IsOkAndHolds(true));
  XLS_ASSERT_OK_AND_ASSIGN(Block * block, p->GetBlock("test_main"));

  EXPECT_THAT(block->nodes(), Contains(m::Tuple(m::Literal(Value::Token()),
                                                m::InputPort("a_in"))));
}

TEST_F(ChannelToPortIoLoweringPassTest, NoGateRecvsWithPredicate) {
  auto p = std::make_unique<Package>("test");
  ScheduledProcBuilder pb(NewStyleProc(), "test_main", p.get());
  XLS_ASSERT_OK_AND_ASSIGN(auto a_in,
                           pb.AddInputChannel("a_in", p->GetBitsType(32)));
  BValue tkn = pb.Literal(Value::Token());
  BValue pred = pb.Literal(Value(UBits(1, 1)));
  pb.ReceiveIf(a_in, tkn, pred);
  XLS_ASSERT_OK(pb.Build());

  BlockConversionPassOptions options;
  options.codegen_options.gate_recvs(false);

  EXPECT_THAT(Run(p.get(), options), IsOkAndHolds(true));
  XLS_ASSERT_OK_AND_ASSIGN(Block * block, p->GetBlock("test_main"));

  EXPECT_THAT(block->nodes(), Contains(m::Tuple(m::Literal(Value::Token()),
                                                m::InputPort("a_in"))));
}

TEST_F(ChannelToPortIoLoweringPassTest, NoGateRecvsNonBlocking) {
  auto p = std::make_unique<Package>("test");
  ScheduledProcBuilder pb(NewStyleProc(), "test_main", p.get());
  XLS_ASSERT_OK_AND_ASSIGN(auto a_in,
                           pb.AddInputChannel("a_in", p->GetBitsType(32)));
  BValue tkn = pb.Literal(Value::Token());
  pb.ReceiveNonBlocking(a_in, tkn);
  XLS_ASSERT_OK(pb.Build());

  BlockConversionPassOptions options;
  options.codegen_options.gate_recvs(false);

  EXPECT_THAT(Run(p.get(), options), IsOkAndHolds(true));
  XLS_ASSERT_OK_AND_ASSIGN(Block * block, p->GetBlock("test_main"));

  EXPECT_THAT(block->nodes(), Contains(m::Tuple(m::Literal(Value::Token()),
                                                m::InputPort("a_in"),
                                                m::InputPort("a_in_vld"))));
}

TEST_F(ChannelToPortIoLoweringPassTest, NoGateRecvsNonBlockingWithPredicate) {
  auto p = std::make_unique<Package>("test");
  ScheduledProcBuilder pb(NewStyleProc(), "test_main", p.get());
  XLS_ASSERT_OK_AND_ASSIGN(auto a_in,
                           pb.AddInputChannel("a_in", p->GetBitsType(32)));
  BValue tkn = pb.Literal(Value::Token());
  BValue pred = pb.Literal(Value(UBits(1, 1)));
  pb.ReceiveIfNonBlocking(a_in, tkn, pred);
  XLS_ASSERT_OK(pb.Build());

  BlockConversionPassOptions options;
  options.codegen_options.gate_recvs(false);

  EXPECT_THAT(Run(p.get(), options), IsOkAndHolds(true));
  XLS_ASSERT_OK_AND_ASSIGN(Block * block, p->GetBlock("test_main"));

  EXPECT_THAT(
      block->nodes(),
      Contains(m::Tuple(m::Literal(Value::Token()), m::InputPort("a_in"),
                        m::And(m::InputPort("a_in_vld"), m::Literal(1)))));
}

TEST_F(ChannelToPortIoLoweringPassTest,
       LowerThreeSendsSameChannelSelectorOrdering) {
  auto p = std::make_unique<Package>("test");
  ScheduledProcBuilder pb(NewStyleProc(), "test_main", p.get());
  XLS_ASSERT_OK_AND_ASSIGN(auto p1_ch,
                           pb.AddInputChannel("p1_ch", p->GetBitsType(1)));
  XLS_ASSERT_OK_AND_ASSIGN(auto p2_ch,
                           pb.AddInputChannel("p2_ch", p->GetBitsType(1)));
  XLS_ASSERT_OK_AND_ASSIGN(auto p3_ch,
                           pb.AddInputChannel("p3_ch", p->GetBitsType(1)));
  XLS_ASSERT_OK_AND_ASSIGN(auto a_out,
                           pb.AddOutputChannel("a_out", p->GetBitsType(32)));

  BValue tkn = pb.Literal(Value::Token());
  BValue p1_recv = pb.Receive(p1_ch, tkn);
  BValue p2_recv = pb.Receive(p2_ch, tkn);
  BValue p3_recv = pb.Receive(p3_ch, tkn);
  BValue p1 = pb.TupleIndex(p1_recv, 1);
  BValue p2 = pb.TupleIndex(p2_recv, 1);
  BValue p3 = pb.TupleIndex(p3_recv, 1);
  BValue lit1 = pb.Literal(Value(UBits(1, 32)));
  BValue lit2 = pb.Literal(Value(UBits(2, 32)));
  BValue lit3 = pb.Literal(Value(UBits(3, 32)));
  pb.SendIf(a_out, tkn, p1, lit1);
  pb.SendIf(a_out, tkn, p2, lit2);
  pb.SendIf(a_out, tkn, p3, lit3);
  XLS_ASSERT_OK(pb.Build());

  EXPECT_THAT(Run(p.get()), IsOkAndHolds(true));

  XLS_ASSERT_OK_AND_ASSIGN(Block * block, p->GetBlock("test_main"));

  // Check that we have the ports.
  EXPECT_THAT(block->GetOutputPort("a_out"), IsOk());

  // Check that the data port is driven by a OneHotSelect (multiplexer), where
  // the selector is a concat of the three predicates such that, read in
  // LSB-to-MSB order, they match the three cases.
  Node* output_port_data =
      block->GetOutputPort("a_out").value()->output_source();
  ASSERT_THAT(output_port_data, m::OneHotSelect());
  OneHotSelect* output_data_ohs = output_port_data->As<OneHotSelect>();
  ASSERT_THAT(output_data_ohs->selector(), m::Concat());
  absl::Span<Node* const> predicate_bits =
      output_data_ohs->selector()->As<Concat>()->operands();
  absl::Span<Node* const> cases = output_port_data->As<OneHotSelect>()->cases();
  ASSERT_THAT(cases, SizeIs(predicate_bits.size()));
  auto predicated_cases_it = iter::zip(iter::reversed(predicate_bits), cases);
  std::vector<std::pair<Node*, Node*>> predicated_cases(
      predicated_cases_it.begin(), predicated_cases_it.end());
  EXPECT_THAT(
      predicated_cases,
      UnorderedElementsAre(
          Pair(m::And(_, _,
                      m::TupleIndex(m::Tuple(_, m::InputPort("p1_ch")), 1)),
               m::Literal(1)),
          Pair(m::And(_, _,
                      m::TupleIndex(m::Tuple(_, m::InputPort("p2_ch")), 1)),
               m::Literal(2)),
          Pair(m::And(_, _,
                      m::TupleIndex(m::Tuple(_, m::InputPort("p3_ch")), 1)),
               m::Literal(3))));
}

TEST_F(ChannelToPortIoLoweringPassTest, PeekWithoutRecv) {
  auto p = std::make_unique<Package>("test");
  ScheduledProcBuilder pb(NewStyleProc(), "test_main", p.get());
  XLS_ASSERT_OK_AND_ASSIGN(auto a_in,
                           pb.AddInputChannel("a_in", p->GetBitsType(32)));
  BValue tkn = pb.Literal(Value::Token());
  pb.Peek(a_in, tkn);
  XLS_ASSERT_OK(pb.Build());

  EXPECT_THAT(Run(p.get()), IsOkAndHolds(true));
}

}  // namespace
}  // namespace xls::codegen
