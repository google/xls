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

#include "xls/codegen/passes_ng/block_channel_adapter.h"

#include <cstdint>
#include <string_view>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/types/span.h"
#include "xls/codegen/passes_ng/passes_ng_test_fixtures.h"
#include "xls/common/logging/log_lines.h"
#include "xls/common/status/matchers.h"
#include "xls/ir/bits.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/instantiation.h"
#include "xls/ir/ir_matcher.h"
#include "xls/ir/nodes.h"
#include "xls/ir/op.h"
#include "xls/ir/source_location.h"
#include "xls/ir/value.h"
#include "xls/ir/verifier.h"

namespace m = ::xls::op_matchers;

namespace xls::verilog {
namespace {

using ::absl_testing::IsOkAndHolds;
using ::testing::ElementsAre;
using ::testing::IsSubsetOf;
using ::testing::IsSupersetOf;
using ::testing::Lt;
using ::testing::SizeIs;

TEST_F(SlotTestBase, UseAdaptersTest) {
  XLS_ASSERT_OK_AND_ASSIGN(BlockAndSlots block_and_slots,
                           CreateTestProcAndAssociatedBlock(TestName()));
  Block* block = block_and_slots.block;
  Proc* proc = block_and_slots.proc;
  IrToBlockIrMap& node_map = block_and_slots.node_map;

  Receive* recv_node = FindNode("recv_x", proc)->As<Receive>();
  Send* send_node = FindNode("send_y", proc)->As<Send>();

  XLS_ASSERT_OK_AND_ASSIGN(
      RDVAdapter recv_adapter,
      RDVAdapter::CreateReceiveAdapter(block_and_slots.slot_r, recv_node,
                                       node_map, block));
  XLS_ASSERT_OK_AND_ASSIGN(
      RDVAdapter send_adapter,
      RDVAdapter::CreateSendAdapter(block_and_slots.slot_s, send_node, node_map,
                                    block));

  // Check fields of the adapters.
  EXPECT_EQ(send_adapter.data()->GetName(), "__y_send_buf");
  EXPECT_EQ(recv_adapter.data()->GetName(), "__x_receive_buf");

  EXPECT_EQ(recv_adapter.adapter_type(), RDVAdapter::AdapterType::kReceive);
  EXPECT_EQ(send_adapter.adapter_type(), RDVAdapter::AdapterType::kSend);

  EXPECT_THAT(recv_adapter.channel_op_value(),
              m::Tuple(m::AfterAll(), m::Identity()));
  EXPECT_THAT(recv_adapter.channel_predicate(), m::Literal(1));
  EXPECT_THAT(send_adapter.channel_op_value(), m::Identity(m::AfterAll()));
  EXPECT_THAT(send_adapter.channel_predicate(), m::Literal(1));

  std::vector<uint64_t> input_sequence = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  SimulationResults results;

  // Rewire so that the block is a pass-through.
  XLS_ASSERT_OK(send_adapter.SetData(recv_adapter.data()));
  XLS_VLOG_LINES(1, block->DumpIr());
  {
    XLS_ASSERT_OK_AND_ASSIGN(results,
                             InterpretTestBlock(block, input_sequence, 100));
    ASSERT_EQ(results.sinks.size(), 1);
    EXPECT_THAT(results.sinks[0].GetOutputSequenceAsUint64(),
                IsOkAndHolds(ElementsAre(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)));
  }

  // Rewire so that the sender assumes the receiver is ready.
  // This will mean that the receiver will drop data.
  XLS_ASSERT_OK_AND_ASSIGN(
      Node * literal_1,
      block->MakeNode<xls::Literal>(SourceInfo(), Value(UBits(1, 1))));
  XLS_ASSERT_OK(recv_adapter.SetReady(literal_1));
  {
    XLS_ASSERT_OK_AND_ASSIGN(results,
                             InterpretTestBlock(block, input_sequence, 100));
    ASSERT_EQ(results.sinks.size(), 1);
    XLS_ASSERT_OK_AND_ASSIGN(std::vector<uint64_t> output_sequence,
                             results.sinks[0].GetOutputSequenceAsUint64());
    EXPECT_THAT(output_sequence, IsSubsetOf({1, 2, 3, 4, 5, 6, 7, 8, 9, 10}));
    EXPECT_THAT(output_sequence, SizeIs(Lt(10)));
  }

  // Rewire so that the receiver also assumes that the sender is valid.
  // This will mean that the receiver will pick up random u32:-1 and 0 data.
  XLS_ASSERT_OK(send_adapter.SetValid(literal_1));
  {
    XLS_ASSERT_OK_AND_ASSIGN(results,
                             InterpretTestBlock(block, input_sequence, 100));
    ASSERT_EQ(results.sinks.size(), 1);
    XLS_ASSERT_OK_AND_ASSIGN(std::vector<uint64_t> output_sequence,
                             results.sinks[0].GetOutputSequenceAsUint64());
    EXPECT_THAT(output_sequence, IsSupersetOf<uint64_t>({0, 0xffffffff}));
  }
}

TEST_F(SlotTestBase, UseInterfaceAdaptersTest) {
  XLS_ASSERT_OK_AND_ASSIGN(BlockAndSlots block_and_slots,
                           CreateTestProcAndAssociatedBlock(TestName()));

  Block* block = block_and_slots.block;

  XLS_ASSERT_OK_AND_ASSIGN(
      RDVAdapter recv_adapter,
      RDVAdapter::CreateInterfaceReceiveAdapter(block_and_slots.slot_r, block));
  XLS_ASSERT_OK_AND_ASSIGN(
      RDVAdapter send_adapter,
      RDVAdapter::CreateInterfaceSendAdapter(block_and_slots.slot_s, block));

  std::vector<uint64_t> input_sequence = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  SimulationResults results;

  // Rewire so that the block is a pass-through.
  XLS_ASSERT_OK(send_adapter.SetData(recv_adapter.data()));
  XLS_VLOG_LINES(1, block->DumpIr());
  {
    XLS_ASSERT_OK_AND_ASSIGN(results,
                             InterpretTestBlock(block, input_sequence, 100));
    ASSERT_EQ(results.sinks.size(), 1);
    EXPECT_THAT(results.sinks[0].GetOutputSequenceAsUint64(),
                IsOkAndHolds(ElementsAre(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)));
  }

  // Rewire so that the sender assumes the receiver is ready.
  // This will mean that the receiver will drop data.
  XLS_ASSERT_OK_AND_ASSIGN(
      Node * literal_1,
      block->MakeNode<xls::Literal>(SourceInfo(), Value(UBits(1, 1))));
  XLS_ASSERT_OK(recv_adapter.SetReady(literal_1));
  {
    XLS_ASSERT_OK_AND_ASSIGN(results,
                             InterpretTestBlock(block, input_sequence, 100));
    ASSERT_EQ(results.sinks.size(), 1);
    XLS_ASSERT_OK_AND_ASSIGN(std::vector<uint64_t> output_sequence,
                             results.sinks[0].GetOutputSequenceAsUint64());
    EXPECT_THAT(output_sequence, IsSubsetOf({1, 2, 3, 4, 5, 6, 7, 8, 9, 10}));
    EXPECT_THAT(output_sequence, SizeIs(Lt(10)));
  }

  // Rewire so that the receiver also assumes that the sender is valid.
  // This will mean that the receiver will pick up random u32:-1 and 0 data.
  XLS_ASSERT_OK(send_adapter.SetValid(literal_1));
  {
    XLS_ASSERT_OK_AND_ASSIGN(results,
                             InterpretTestBlock(block, input_sequence, 100));
    ASSERT_EQ(results.sinks.size(), 1);
    XLS_ASSERT_OK_AND_ASSIGN(std::vector<uint64_t> output_sequence,
                             results.sinks[0].GetOutputSequenceAsUint64());
    EXPECT_THAT(output_sequence, IsSupersetOf<uint64_t>({0, 0xffffffff}));
  }
}

TEST_F(SlotTestBase, NonBlockingRecvStructureTest) {
  XLS_ASSERT_OK_AND_ASSIGN(
      BlockAndSlots block_and_slots,
      CreateTestProcAndAssociatedBlock(TestName(), /*use_non_blocking=*/true));
  Block* block = block_and_slots.block;
  Proc* proc = block_and_slots.proc;
  IrToBlockIrMap& node_map = block_and_slots.node_map;

  Receive* recv_node = FindNode("recv_x", proc)->As<Receive>();

  XLS_ASSERT_OK_AND_ASSIGN(
      RDVAdapter recv_adapter,
      RDVAdapter::CreateReceiveAdapter(block_and_slots.slot_r, recv_node,
                                       node_map, block));

  // Check fields of the adapters.
  EXPECT_EQ(recv_adapter.adapter_type(), RDVAdapter::AdapterType::kReceive);

  EXPECT_THAT(recv_adapter.channel_op_value(),
              m::Tuple(m::AfterAll(), m::Identity(m::Type("bits[32]")),
                       m::Identity(m::Type("bits[1]"))));
  EXPECT_THAT(recv_adapter.channel_predicate(), m::Literal(1));
}

TEST_F(PredicatedSlotTestBase, PredicatedChannelOpsStructureTest) {
  XLS_ASSERT_OK_AND_ASSIGN(BlockAndSlots block_and_slots,
                           CreateTestProcAndAssociatedBlock(TestName()));

  Block* block = block_and_slots.block;
  Proc* proc = block_and_slots.proc;
  IrToBlockIrMap& node_map = block_and_slots.node_map;

  Receive* recv_node = FindNode("recv_x", proc)->As<Receive>();
  Send* send_node = FindNode("send_y", proc)->As<Send>();

  XLS_ASSERT_OK_AND_ASSIGN(
      RDVAdapter recv_adapter,
      RDVAdapter::CreateReceiveAdapter(block_and_slots.slot_r, recv_node,
                                       node_map, block));
  XLS_ASSERT_OK_AND_ASSIGN(
      RDVAdapter send_adapter,
      RDVAdapter::CreateSendAdapter(block_and_slots.slot_s, send_node, node_map,
                                    block));

  // Check fields of the adapters.
  EXPECT_EQ(recv_adapter.adapter_type(), RDVAdapter::AdapterType::kReceive);
  EXPECT_EQ(send_adapter.adapter_type(), RDVAdapter::AdapterType::kSend);

  EXPECT_THAT(recv_adapter.channel_op_value(),
              m::Tuple(m::AfterAll(), m::Identity(m::Type("bits[32]"))));
  EXPECT_THAT(recv_adapter.channel_predicate(), m::InputPort("pred"));
  EXPECT_THAT(send_adapter.channel_op_value(), m::Identity(m::AfterAll()));
  EXPECT_THAT(send_adapter.channel_predicate(), m::InputPort("pred"));
}

}  // namespace
}  // namespace xls::verilog
