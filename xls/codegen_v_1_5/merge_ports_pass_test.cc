// Copyright 2026 The XLS Authors
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

#include "xls/codegen_v_1_5/merge_ports_pass.h"

#include <memory>
#include <string>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/statusor.h"
#include "google/protobuf/text_format.h"
#include "xls/codegen/codegen_options.h"
#include "xls/codegen/module_signature.pb.h"
#include "xls/codegen_v_1_5/block_conversion_pass.h"
#include "xls/common/proto_test_utils.h"
#include "xls/common/status/matchers.h"
#include "xls/ir/bits.h"
#include "xls/ir/block.h"
#include "xls/ir/channel.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/nodes.h"
#include "xls/ir/package.h"
#include "xls/passes/pass_base.h"

namespace xls::codegen {
namespace {

using ::absl_testing::IsOkAndHolds;
using ::testing::Optional;
using ::xls::proto_testing::EqualsProto;

class MergePortsPassTest : public IrTestBase {
 protected:
  absl::StatusOr<bool> Run(Package* p, bool preserve_ports = false) {
    verilog::CodegenOptions codegen_options;
    codegen_options.preserve_ports(preserve_ports);
    BlockConversionPassOptions options{
        .codegen_options = codegen_options,
    };
    PassResults results;
    BlockConversionContext context;
    return MergePortsPass().Run(p, options, &results, context);
  }
};

TEST_F(MergePortsPassTest, ShareOutputPortsDisabled) {
  auto p = CreatePackage();
  BlockBuilder bb(TestName(), p.get());
  BValue in = bb.InputPort("in", p->GetBitsType(32));
  bb.OutputPort("out_a", in);
  bb.OutputPort("out_b", in);
  XLS_ASSERT_OK_AND_ASSIGN(Block * block, bb.Build());

  XLS_ASSERT_OK(block->AddChannelPortMetadata(ChannelPortMetadata{
      .channel_name = "out_a",
      .type = p->GetBitsType(32),
      .direction = ChannelDirection::kSend,
      .channel_kind = ChannelKind::kStreaming,
      .flop_kind = FlopKind::kNone,
      .data_port = "out_a",
  }));
  XLS_ASSERT_OK(block->AddChannelPortMetadata(ChannelPortMetadata{
      .channel_name = "out_b",
      .type = p->GetBitsType(32),
      .direction = ChannelDirection::kSend,
      .channel_kind = ChannelKind::kStreaming,
      .flop_kind = FlopKind::kNone,
      .data_port = "out_b",
  }));

  EXPECT_THAT(Run(p.get(), /*preserve_ports=*/true), IsOkAndHolds(false));
  EXPECT_EQ(block->GetOutputPorts().size(), 2);

  XLS_ASSERT_OK_AND_ASSIGN(
      ChannelPortMetadata meta_a,
      block->GetChannelPortMetadata("out_a", ChannelDirection::kSend));
  XLS_ASSERT_OK_AND_ASSIGN(
      ChannelPortMetadata meta_b,
      block->GetChannelPortMetadata("out_b", ChannelDirection::kSend));
  EXPECT_NE(meta_a.data_port, meta_b.data_port);
}

TEST_F(MergePortsPassTest, DistinctSourcesNotMerged) {
  auto p = CreatePackage();
  BlockBuilder bb(TestName(), p.get());
  BValue in_a = bb.InputPort("in_a", p->GetBitsType(32));
  BValue in_b = bb.InputPort("in_b", p->GetBitsType(32));
  bb.OutputPort("out_a", in_a);
  bb.OutputPort("out_b", in_b);
  XLS_ASSERT_OK_AND_ASSIGN(Block * block, bb.Build());

  XLS_ASSERT_OK(block->AddChannelPortMetadata(ChannelPortMetadata{
      .channel_name = "out_a",
      .type = p->GetBitsType(32),
      .direction = ChannelDirection::kSend,
      .channel_kind = ChannelKind::kStreaming,
      .flop_kind = FlopKind::kNone,
      .data_port = "out_a",
  }));
  XLS_ASSERT_OK(block->AddChannelPortMetadata(ChannelPortMetadata{
      .channel_name = "out_b",
      .type = p->GetBitsType(32),
      .direction = ChannelDirection::kSend,
      .channel_kind = ChannelKind::kStreaming,
      .flop_kind = FlopKind::kNone,
      .data_port = "out_b",
  }));

  EXPECT_THAT(Run(p.get()), IsOkAndHolds(false));
  EXPECT_EQ(block->GetOutputPorts().size(), 2);

  XLS_ASSERT_OK_AND_ASSIGN(
      ChannelPortMetadata meta_a,
      block->GetChannelPortMetadata("out_a", ChannelDirection::kSend));
  XLS_ASSERT_OK_AND_ASSIGN(
      ChannelPortMetadata meta_b,
      block->GetChannelPortMetadata("out_b", ChannelDirection::kSend));
  EXPECT_NE(meta_a.data_port, meta_b.data_port);
}

TEST_F(MergePortsPassTest, MergeSendDataPorts) {
  auto p = CreatePackage();
  BlockBuilder bb(TestName(), p.get());
  BValue in = bb.InputPort("in", p->GetBitsType(32));
  bb.OutputPort("out_a_data", in);
  bb.OutputPort("out_b_data", in);
  XLS_ASSERT_OK_AND_ASSIGN(Block * block, bb.Build());

  XLS_ASSERT_OK(block->AddChannelPortMetadata(ChannelPortMetadata{
      .channel_name = "out_a",
      .type = p->GetBitsType(32),
      .direction = ChannelDirection::kSend,
      .channel_kind = ChannelKind::kStreaming,
      .flop_kind = FlopKind::kNone,
      .data_port = "out_a_data",
  }));
  XLS_ASSERT_OK(block->AddChannelPortMetadata(ChannelPortMetadata{
      .channel_name = "out_b",
      .type = p->GetBitsType(32),
      .direction = ChannelDirection::kSend,
      .channel_kind = ChannelKind::kStreaming,
      .flop_kind = FlopKind::kNone,
      .data_port = "out_b_data",
  }));

  EXPECT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_EQ(block->GetOutputPorts().size(), 1);

  XLS_ASSERT_OK_AND_ASSIGN(
      ChannelPortMetadata meta_a,
      block->GetChannelPortMetadata("out_a", ChannelDirection::kSend));
  XLS_ASSERT_OK_AND_ASSIGN(
      ChannelPortMetadata meta_b,
      block->GetChannelPortMetadata("out_b", ChannelDirection::kSend));
  EXPECT_EQ(meta_a.data_port, meta_b.data_port);
}

TEST_F(MergePortsPassTest, MergeSendDataPortsWithoutValid) {
  auto p = CreatePackage();
  BlockBuilder bb(TestName(), p.get());
  BValue in = bb.InputPort("in", p->GetBitsType(32));
  BValue a_vld = bb.InputPort("in_a_vld", p->GetBitsType(1));
  bb.OutputPort("out_a_data", in);
  bb.OutputPort("out_a_vld", a_vld);
  BValue b_vld = bb.InputPort("in_b_vld", p->GetBitsType(1));
  bb.OutputPort("out_b_data", in);
  bb.OutputPort("out_b_vld", b_vld);
  XLS_ASSERT_OK_AND_ASSIGN(Block * block, bb.Build());

  XLS_ASSERT_OK(block->AddChannelPortMetadata(ChannelPortMetadata{
      .channel_name = "out_a",
      .type = p->GetBitsType(32),
      .direction = ChannelDirection::kSend,
      .channel_kind = ChannelKind::kStreaming,
      .flop_kind = FlopKind::kNone,
      .data_port = "out_a_data",
      .valid_port = "out_a_vld",
  }));
  XLS_ASSERT_OK(block->AddChannelPortMetadata(ChannelPortMetadata{
      .channel_name = "out_b",
      .type = p->GetBitsType(32),
      .direction = ChannelDirection::kSend,
      .channel_kind = ChannelKind::kStreaming,
      .flop_kind = FlopKind::kNone,
      .data_port = "out_b_data",
      .valid_port = "out_b_vld",
  }));

  EXPECT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_EQ(block->GetOutputPorts().size(), 3);

  XLS_ASSERT_OK_AND_ASSIGN(
      ChannelPortMetadata meta_a,
      block->GetChannelPortMetadata("out_a", ChannelDirection::kSend));
  XLS_ASSERT_OK_AND_ASSIGN(
      ChannelPortMetadata meta_b,
      block->GetChannelPortMetadata("out_b", ChannelDirection::kSend));
  EXPECT_EQ(meta_a.data_port, meta_b.data_port);
  EXPECT_NE(meta_a.valid_port, meta_b.valid_port);
}

TEST_F(MergePortsPassTest, MergeSendValidPortsWithoutData) {
  auto p = CreatePackage();
  BlockBuilder bb(TestName(), p.get());
  BValue in_vld = bb.InputPort("in_vld", p->GetBitsType(1));
  BValue a_data = bb.InputPort("in_a_data", p->GetBitsType(32));
  bb.OutputPort("out_a_data", a_data);
  bb.OutputPort("out_a_vld", in_vld);
  BValue b_data = bb.InputPort("in_b_data", p->GetBitsType(32));
  bb.OutputPort("out_b_data", b_data);
  bb.OutputPort("out_b_vld", in_vld);
  XLS_ASSERT_OK_AND_ASSIGN(Block * block, bb.Build());

  XLS_ASSERT_OK(block->AddChannelPortMetadata(ChannelPortMetadata{
      .channel_name = "out_a",
      .type = p->GetBitsType(32),
      .direction = ChannelDirection::kSend,
      .channel_kind = ChannelKind::kStreaming,
      .flop_kind = FlopKind::kNone,
      .data_port = "out_a_data",
      .valid_port = "out_a_vld",
  }));
  XLS_ASSERT_OK(block->AddChannelPortMetadata(ChannelPortMetadata{
      .channel_name = "out_b",
      .type = p->GetBitsType(32),
      .direction = ChannelDirection::kSend,
      .channel_kind = ChannelKind::kStreaming,
      .flop_kind = FlopKind::kNone,
      .data_port = "out_b_data",
      .valid_port = "out_b_vld",
  }));

  EXPECT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_EQ(block->GetOutputPorts().size(), 3);

  XLS_ASSERT_OK_AND_ASSIGN(
      ChannelPortMetadata meta_a,
      block->GetChannelPortMetadata("out_a", ChannelDirection::kSend));
  XLS_ASSERT_OK_AND_ASSIGN(
      ChannelPortMetadata meta_b,
      block->GetChannelPortMetadata("out_b", ChannelDirection::kSend));
  EXPECT_NE(meta_a.data_port, meta_b.data_port);
  EXPECT_EQ(meta_a.valid_port, meta_b.valid_port);
}

TEST_F(MergePortsPassTest, MergeSendDataAndValidPorts) {
  auto p = CreatePackage();
  BlockBuilder bb(TestName(), p.get());
  BValue in_data = bb.InputPort("in_data", p->GetBitsType(32));
  BValue in_vld = bb.InputPort("in_vld", p->GetBitsType(1));
  bb.OutputPort("out_a_data", in_data);
  bb.OutputPort("out_a_vld", in_vld);
  bb.OutputPort("out_b_data", in_data);
  bb.OutputPort("out_b_vld", in_vld);
  XLS_ASSERT_OK_AND_ASSIGN(Block * block, bb.Build());

  XLS_ASSERT_OK(block->AddChannelPortMetadata(ChannelPortMetadata{
      .channel_name = "out_a",
      .type = p->GetBitsType(32),
      .direction = ChannelDirection::kSend,
      .channel_kind = ChannelKind::kStreaming,
      .flop_kind = FlopKind::kNone,
      .data_port = "out_a_data",
      .valid_port = "out_a_vld",
  }));
  XLS_ASSERT_OK(block->AddChannelPortMetadata(ChannelPortMetadata{
      .channel_name = "out_b",
      .type = p->GetBitsType(32),
      .direction = ChannelDirection::kSend,
      .channel_kind = ChannelKind::kStreaming,
      .flop_kind = FlopKind::kNone,
      .data_port = "out_b_data",
      .valid_port = "out_b_vld",
  }));

  EXPECT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_EQ(block->GetOutputPorts().size(), 2);

  XLS_ASSERT_OK_AND_ASSIGN(
      ChannelPortMetadata meta_a,
      block->GetChannelPortMetadata("out_a", ChannelDirection::kSend));
  XLS_ASSERT_OK_AND_ASSIGN(
      ChannelPortMetadata meta_b,
      block->GetChannelPortMetadata("out_b", ChannelDirection::kSend));
  EXPECT_EQ(meta_a.data_port, meta_b.data_port);
  EXPECT_EQ(meta_a.valid_port, meta_b.valid_port);
  EXPECT_NE(meta_a.data_port, meta_a.valid_port);
}

TEST_F(MergePortsPassTest, MergeReceiveReadyPorts) {
  auto p = CreatePackage();
  BlockBuilder bb(TestName(), p.get());
  bb.InputPort("in_a_data", p->GetBitsType(32));
  bb.InputPort("in_b_data", p->GetBitsType(32));
  BValue rdy = bb.Literal(UBits(1, 1));
  bb.OutputPort("in_a_rdy", rdy);
  bb.OutputPort("in_b_rdy", rdy);
  XLS_ASSERT_OK_AND_ASSIGN(Block * block, bb.Build());

  XLS_ASSERT_OK(block->AddChannelPortMetadata(ChannelPortMetadata{
      .channel_name = "in_a",
      .type = p->GetBitsType(32),
      .direction = ChannelDirection::kReceive,
      .channel_kind = ChannelKind::kStreaming,
      .flop_kind = FlopKind::kNone,
      .data_port = "in_a_data",
      .ready_port = "in_a_rdy",
  }));
  XLS_ASSERT_OK(block->AddChannelPortMetadata(ChannelPortMetadata{
      .channel_name = "in_b",
      .type = p->GetBitsType(32),
      .direction = ChannelDirection::kReceive,
      .channel_kind = ChannelKind::kStreaming,
      .flop_kind = FlopKind::kNone,
      .data_port = "in_b_data",
      .ready_port = "in_b_rdy",
  }));

  EXPECT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_EQ(block->GetOutputPorts().size(), 1);

  XLS_ASSERT_OK_AND_ASSIGN(
      ChannelPortMetadata meta_a,
      block->GetChannelPortMetadata("in_a", ChannelDirection::kReceive));
  XLS_ASSERT_OK_AND_ASSIGN(
      ChannelPortMetadata meta_b,
      block->GetChannelPortMetadata("in_b", ChannelDirection::kReceive));
  EXPECT_EQ(meta_a.ready_port, meta_b.ready_port);
}

TEST_F(MergePortsPassTest, MergeChannelPortsAcrossRoles) {
  auto p = CreatePackage();
  BlockBuilder bb(TestName(), p.get());
  BValue in_data = bb.InputPort("in_data", p->GetBitsType(32));
  BValue in_vld = bb.InputPort("in_vld", p->GetBitsType(1));
  bb.OutputPort("out_a_data", in_data);
  bb.OutputPort("out_a_vld", in_vld);
  bb.OutputPort("out_b_data", in_vld);
  bb.OutputPort("out_b_vld", in_vld);
  XLS_ASSERT_OK_AND_ASSIGN(Block * block, bb.Build());

  XLS_ASSERT_OK(block->AddChannelPortMetadata(ChannelPortMetadata{
      .channel_name = "out_a",
      .type = p->GetBitsType(32),
      .direction = ChannelDirection::kSend,
      .channel_kind = ChannelKind::kStreaming,
      .flop_kind = FlopKind::kNone,
      .data_port = "out_a_data",
      .valid_port = "out_a_vld",
  }));
  XLS_ASSERT_OK(block->AddChannelPortMetadata(ChannelPortMetadata{
      .channel_name = "out_b",
      .type = p->GetBitsType(1),
      .direction = ChannelDirection::kSend,
      .channel_kind = ChannelKind::kStreaming,
      .flop_kind = FlopKind::kNone,
      .data_port = "out_b_data",
      .valid_port = "out_b_vld",
  }));

  EXPECT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_EQ(block->GetOutputPorts().size(), 2);

  XLS_ASSERT_OK_AND_ASSIGN(
      ChannelPortMetadata meta_a,
      block->GetChannelPortMetadata("out_a", ChannelDirection::kSend));
  XLS_ASSERT_OK_AND_ASSIGN(
      ChannelPortMetadata meta_b,
      block->GetChannelPortMetadata("out_b", ChannelDirection::kSend));
  EXPECT_NE(meta_a.data_port, meta_a.valid_port);
  EXPECT_EQ(meta_b.data_port, meta_a.valid_port);
  EXPECT_EQ(meta_b.valid_port, meta_a.valid_port);
}

TEST_F(MergePortsPassTest, ChannelMergedIntoNonChannelPort) {
  auto p = CreatePackage();
  BlockBuilder bb(TestName(), p.get());
  BValue in = bb.InputPort("in", p->GetBitsType(32));
  bb.OutputPort("non_channel_out", in);
  bb.OutputPort("chan_out", in);
  XLS_ASSERT_OK_AND_ASSIGN(Block * block, bb.Build());

  XLS_ASSERT_OK(block->AddChannelPortMetadata(ChannelPortMetadata{
      .channel_name = "chan",
      .type = p->GetBitsType(32),
      .direction = ChannelDirection::kSend,
      .channel_kind = ChannelKind::kStreaming,
      .flop_kind = FlopKind::kNone,
      .data_port = "chan_out",
  }));

  EXPECT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_EQ(block->GetOutputPorts().size(), 1);

  XLS_ASSERT_OK_AND_ASSIGN(
      ChannelPortMetadata meta,
      block->GetChannelPortMetadata("chan", ChannelDirection::kSend));
  EXPECT_EQ(meta.data_port, "non_channel_out");
}

TEST_F(MergePortsPassTest, ChannelMergedIntoNonChannelPortReverseOrder) {
  auto p = CreatePackage();
  BlockBuilder bb(TestName(), p.get());
  BValue in = bb.InputPort("in", p->GetBitsType(32));
  bb.OutputPort("chan_out", in);
  bb.OutputPort("non_channel_out", in);
  XLS_ASSERT_OK_AND_ASSIGN(Block * block, bb.Build());

  XLS_ASSERT_OK(block->AddChannelPortMetadata(ChannelPortMetadata{
      .channel_name = "chan",
      .type = p->GetBitsType(32),
      .direction = ChannelDirection::kSend,
      .channel_kind = ChannelKind::kStreaming,
      .flop_kind = FlopKind::kNone,
      .data_port = "chan_out",
  }));

  EXPECT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_EQ(block->GetOutputPorts().size(), 1);

  XLS_ASSERT_OK_AND_ASSIGN(
      ChannelPortMetadata meta,
      block->GetChannelPortMetadata("chan", ChannelDirection::kSend));
  EXPECT_EQ(meta.data_port, "non_channel_out");
}

TEST_F(MergePortsPassTest, MultipleNonChannelPortsNotMerged) {
  auto p = CreatePackage();
  BlockBuilder bb(TestName(), p.get());
  BValue in = bb.InputPort("in", p->GetBitsType(32));
  bb.OutputPort("nc1", in);
  bb.OutputPort("nc2", in);
  XLS_ASSERT_OK_AND_ASSIGN(Block * block, bb.Build());

  EXPECT_THAT(Run(p.get()), IsOkAndHolds(false));
  EXPECT_EQ(block->GetOutputPorts().size(), 2);
  EXPECT_NE(block->GetOutputPorts()[0]->name(),
            block->GetOutputPorts()[1]->name());
}

TEST_F(MergePortsPassTest, MergeSingleValueChannels) {
  auto p = CreatePackage();
  BlockBuilder bb(TestName(), p.get());
  BValue in = bb.InputPort("in", p->GetBitsType(32));
  bb.OutputPort("sv_a", in);
  bb.OutputPort("sv_b", in);
  XLS_ASSERT_OK_AND_ASSIGN(Block * block, bb.Build());

  XLS_ASSERT_OK(block->AddChannelPortMetadata(ChannelPortMetadata{
      .channel_name = "sv_a",
      .type = p->GetBitsType(32),
      .direction = ChannelDirection::kSend,
      .channel_kind = ChannelKind::kSingleValue,
      .flop_kind = FlopKind::kNone,
      .data_port = "sv_a",
  }));
  XLS_ASSERT_OK(block->AddChannelPortMetadata(ChannelPortMetadata{
      .channel_name = "sv_b",
      .type = p->GetBitsType(32),
      .direction = ChannelDirection::kSend,
      .channel_kind = ChannelKind::kSingleValue,
      .flop_kind = FlopKind::kNone,
      .data_port = "sv_b",
  }));

  EXPECT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_EQ(block->GetOutputPorts().size(), 1);

  XLS_ASSERT_OK_AND_ASSIGN(
      ChannelPortMetadata meta_a,
      block->GetChannelPortMetadata("sv_a", ChannelDirection::kSend));
  XLS_ASSERT_OK_AND_ASSIGN(
      ChannelPortMetadata meta_b,
      block->GetChannelPortMetadata("sv_b", ChannelDirection::kSend));
  EXPECT_EQ(meta_a.data_port, meta_b.data_port);
}

TEST_F(MergePortsPassTest,
       MultipleChannelsMergedIntoInterleavedNonChannelPort) {
  auto p = CreatePackage();
  BlockBuilder bb(TestName(), p.get());
  BValue in = bb.InputPort("in", p->GetBitsType(32));
  bb.OutputPort("chan_a_out", in);
  bb.OutputPort("non_channel_out", in);
  bb.OutputPort("chan_b_out", in);
  XLS_ASSERT_OK_AND_ASSIGN(Block * block, bb.Build());

  XLS_ASSERT_OK(block->AddChannelPortMetadata(ChannelPortMetadata{
      .channel_name = "chan_a",
      .type = p->GetBitsType(32),
      .direction = ChannelDirection::kSend,
      .channel_kind = ChannelKind::kStreaming,
      .flop_kind = FlopKind::kNone,
      .data_port = "chan_a_out",
  }));
  XLS_ASSERT_OK(block->AddChannelPortMetadata(ChannelPortMetadata{
      .channel_name = "chan_b",
      .type = p->GetBitsType(32),
      .direction = ChannelDirection::kSend,
      .channel_kind = ChannelKind::kStreaming,
      .flop_kind = FlopKind::kNone,
      .data_port = "chan_b_out",
  }));

  EXPECT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_EQ(block->GetOutputPorts().size(), 1);
  EXPECT_EQ(block->GetOutputPorts().front()->name(), "non_channel_out");

  XLS_ASSERT_OK_AND_ASSIGN(
      ChannelPortMetadata meta_a,
      block->GetChannelPortMetadata("chan_a", ChannelDirection::kSend));
  XLS_ASSERT_OK_AND_ASSIGN(
      ChannelPortMetadata meta_b,
      block->GetChannelPortMetadata("chan_b", ChannelDirection::kSend));
  EXPECT_EQ(meta_a.data_port, "non_channel_out");
  EXPECT_EQ(meta_b.data_port, "non_channel_out");
}

TEST_F(MergePortsPassTest, PreservesFlopKindAndStageMetadata) {
  auto p = CreatePackage();
  BlockBuilder bb(TestName(), p.get());
  BValue in = bb.InputPort("in", p->GetBitsType(32));
  bb.OutputPort("out_a_data", in);
  bb.OutputPort("out_b_data", in);
  XLS_ASSERT_OK_AND_ASSIGN(Block * block, bb.Build());

  XLS_ASSERT_OK(block->AddChannelPortMetadata(ChannelPortMetadata{
      .channel_name = "out_a",
      .type = p->GetBitsType(32),
      .direction = ChannelDirection::kSend,
      .channel_kind = ChannelKind::kStreaming,
      .flop_kind = FlopKind::kFlop,
      .data_port = "out_a_data",
      .stage = 2,
  }));
  XLS_ASSERT_OK(block->AddChannelPortMetadata(ChannelPortMetadata{
      .channel_name = "out_b",
      .type = p->GetBitsType(32),
      .direction = ChannelDirection::kSend,
      .channel_kind = ChannelKind::kStreaming,
      .flop_kind = FlopKind::kSkid,
      .data_port = "out_b_data",
      .stage = 3,
  }));

  EXPECT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_EQ(block->GetOutputPorts().size(), 1);

  XLS_ASSERT_OK_AND_ASSIGN(
      ChannelPortMetadata meta_a,
      block->GetChannelPortMetadata("out_a", ChannelDirection::kSend));
  XLS_ASSERT_OK_AND_ASSIGN(
      ChannelPortMetadata meta_b,
      block->GetChannelPortMetadata("out_b", ChannelDirection::kSend));
  EXPECT_EQ(meta_a.data_port, meta_b.data_port);
  EXPECT_EQ(meta_a.flop_kind, FlopKind::kFlop);
  EXPECT_THAT(meta_a.stage, Optional(2));
  EXPECT_EQ(meta_b.flop_kind, FlopKind::kSkid);
  EXPECT_THAT(meta_b.stage, Optional(3));
}

TEST_F(MergePortsPassTest, MultiBlockPackage) {
  auto p = CreatePackage();

  // Block 1 has redundant output ports.
  BlockBuilder bb1("block1", p.get());
  BValue in1 = bb1.InputPort("in1", p->GetBitsType(32));
  bb1.OutputPort("b1_out_a", in1);
  bb1.OutputPort("b1_out_b", in1);
  XLS_ASSERT_OK_AND_ASSIGN(Block * b1, bb1.Build());

  XLS_ASSERT_OK(b1->AddChannelPortMetadata(ChannelPortMetadata{
      .channel_name = "b1_chan_a",
      .type = p->GetBitsType(32),
      .direction = ChannelDirection::kSend,
      .channel_kind = ChannelKind::kStreaming,
      .flop_kind = FlopKind::kNone,
      .data_port = "b1_out_a",
  }));
  XLS_ASSERT_OK(b1->AddChannelPortMetadata(ChannelPortMetadata{
      .channel_name = "b1_chan_b",
      .type = p->GetBitsType(32),
      .direction = ChannelDirection::kSend,
      .channel_kind = ChannelKind::kStreaming,
      .flop_kind = FlopKind::kNone,
      .data_port = "b1_out_b",
  }));

  // Block 2 has distinct output ports.
  BlockBuilder bb2("block2", p.get());
  BValue in2_a = bb2.InputPort("in2_a", p->GetBitsType(32));
  BValue in2_b = bb2.InputPort("in2_b", p->GetBitsType(32));
  bb2.OutputPort("b2_out_a", in2_a);
  bb2.OutputPort("b2_out_b", in2_b);
  XLS_ASSERT_OK_AND_ASSIGN(Block * b2, bb2.Build());

  XLS_ASSERT_OK(b2->AddChannelPortMetadata(ChannelPortMetadata{
      .channel_name = "b2_chan_a",
      .type = p->GetBitsType(32),
      .direction = ChannelDirection::kSend,
      .channel_kind = ChannelKind::kStreaming,
      .flop_kind = FlopKind::kNone,
      .data_port = "b2_out_a",
  }));
  XLS_ASSERT_OK(b2->AddChannelPortMetadata(ChannelPortMetadata{
      .channel_name = "b2_chan_b",
      .type = p->GetBitsType(32),
      .direction = ChannelDirection::kSend,
      .channel_kind = ChannelKind::kStreaming,
      .flop_kind = FlopKind::kNone,
      .data_port = "b2_out_b",
  }));

  EXPECT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_EQ(b1->GetOutputPorts().size(), 1);
  EXPECT_EQ(b2->GetOutputPorts().size(), 2);

  XLS_ASSERT_OK_AND_ASSIGN(
      ChannelPortMetadata b1_meta_a,
      b1->GetChannelPortMetadata("b1_chan_a", ChannelDirection::kSend));
  XLS_ASSERT_OK_AND_ASSIGN(
      ChannelPortMetadata b1_meta_b,
      b1->GetChannelPortMetadata("b1_chan_b", ChannelDirection::kSend));
  EXPECT_EQ(b1_meta_a.data_port, b1_meta_b.data_port);

  XLS_ASSERT_OK_AND_ASSIGN(
      ChannelPortMetadata b2_meta_a,
      b2->GetChannelPortMetadata("b2_chan_a", ChannelDirection::kSend));
  XLS_ASSERT_OK_AND_ASSIGN(
      ChannelPortMetadata b2_meta_b,
      b2->GetChannelPortMetadata("b2_chan_b", ChannelDirection::kSend));
  EXPECT_NE(b2_meta_a.data_port, b2_meta_b.data_port);
}

TEST_F(MergePortsPassTest, AlreadySharedPortsNoOp) {
  auto p = CreatePackage();
  BlockBuilder bb(TestName(), p.get());
  BValue in = bb.InputPort("in", p->GetBitsType(32));
  bb.OutputPort("shared_out", in);
  XLS_ASSERT_OK_AND_ASSIGN(Block * block, bb.Build());

  XLS_ASSERT_OK(block->AddChannelPortMetadata(ChannelPortMetadata{
      .channel_name = "out_a",
      .type = p->GetBitsType(32),
      .direction = ChannelDirection::kSend,
      .channel_kind = ChannelKind::kStreaming,
      .flop_kind = FlopKind::kNone,
      .data_port = "shared_out",
  }));
  XLS_ASSERT_OK(block->AddChannelPortMetadata(ChannelPortMetadata{
      .channel_name = "out_b",
      .type = p->GetBitsType(32),
      .direction = ChannelDirection::kSend,
      .channel_kind = ChannelKind::kStreaming,
      .flop_kind = FlopKind::kNone,
      .data_port = "shared_out",
  }));

  EXPECT_THAT(Run(p.get()), IsOkAndHolds(false));
  EXPECT_EQ(block->GetOutputPorts().size(), 1);

  XLS_ASSERT_OK_AND_ASSIGN(
      ChannelPortMetadata meta_a,
      block->GetChannelPortMetadata("out_a", ChannelDirection::kSend));
  XLS_ASSERT_OK_AND_ASSIGN(
      ChannelPortMetadata meta_b,
      block->GetChannelPortMetadata("out_b", ChannelDirection::kSend));
  EXPECT_EQ(meta_a.data_port, "shared_out");
  EXPECT_EQ(meta_b.data_port, "shared_out");
}

TEST_F(MergePortsPassTest, PartiallySharedPortsMerge) {
  auto p = CreatePackage();
  BlockBuilder bb(TestName(), p.get());
  BValue in = bb.InputPort("in", p->GetBitsType(32));
  bb.OutputPort("shared_out", in);
  bb.OutputPort("out_c_data", in);
  XLS_ASSERT_OK_AND_ASSIGN(Block * block, bb.Build());

  // Channels `out_a` and `out_b` already share `shared_out`.
  XLS_ASSERT_OK(block->AddChannelPortMetadata(ChannelPortMetadata{
      .channel_name = "out_a",
      .type = p->GetBitsType(32),
      .direction = ChannelDirection::kSend,
      .channel_kind = ChannelKind::kStreaming,
      .flop_kind = FlopKind::kNone,
      .data_port = "shared_out",
  }));
  XLS_ASSERT_OK(block->AddChannelPortMetadata(ChannelPortMetadata{
      .channel_name = "out_b",
      .type = p->GetBitsType(32),
      .direction = ChannelDirection::kSend,
      .channel_kind = ChannelKind::kStreaming,
      .flop_kind = FlopKind::kNone,
      .data_port = "shared_out",
  }));

  // Channel `out_c` has its own port `out_c_data` driven by the same source.
  XLS_ASSERT_OK(block->AddChannelPortMetadata(ChannelPortMetadata{
      .channel_name = "out_c",
      .type = p->GetBitsType(32),
      .direction = ChannelDirection::kSend,
      .channel_kind = ChannelKind::kStreaming,
      .flop_kind = FlopKind::kNone,
      .data_port = "out_c_data",
  }));

  EXPECT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_EQ(block->GetOutputPorts().size(), 1);

  XLS_ASSERT_OK_AND_ASSIGN(
      ChannelPortMetadata meta_a,
      block->GetChannelPortMetadata("out_a", ChannelDirection::kSend));
  XLS_ASSERT_OK_AND_ASSIGN(
      ChannelPortMetadata meta_b,
      block->GetChannelPortMetadata("out_b", ChannelDirection::kSend));
  XLS_ASSERT_OK_AND_ASSIGN(
      ChannelPortMetadata meta_c,
      block->GetChannelPortMetadata("out_c", ChannelDirection::kSend));

  // All three channels should now share the exact same data port.
  EXPECT_EQ(meta_a.data_port, meta_b.data_port);
  EXPECT_EQ(meta_b.data_port, meta_c.data_port);
  EXPECT_THAT(meta_a.data_port,
              Optional(std::string{block->GetOutputPorts().front()->name()}));
}

TEST_F(MergePortsPassTest, UpdateSignatureStreamingSendDataPorts) {
  auto p = CreatePackage();
  BlockBuilder bb(TestName(), p.get());
  BValue in = bb.InputPort("in", p->GetBitsType(32));
  bb.OutputPort("out_a_data", in);
  bb.OutputPort("out_b_data", in);
  XLS_ASSERT_OK_AND_ASSIGN(Block * block, bb.Build());

  XLS_ASSERT_OK(block->AddChannelPortMetadata(ChannelPortMetadata{
      .channel_name = "out_a",
      .type = p->GetBitsType(32),
      .direction = ChannelDirection::kSend,
      .channel_kind = ChannelKind::kStreaming,
      .flop_kind = FlopKind::kNone,
      .data_port = "out_a_data",
  }));
  XLS_ASSERT_OK(block->AddChannelPortMetadata(ChannelPortMetadata{
      .channel_name = "out_b",
      .type = p->GetBitsType(32),
      .direction = ChannelDirection::kSend,
      .channel_kind = ChannelKind::kStreaming,
      .flop_kind = FlopKind::kNone,
      .data_port = "out_b_data",
  }));

  verilog::ModuleSignatureProto initial_signature;
  ASSERT_TRUE(google::protobuf::TextFormat::ParseFromString(
      R"pb(
        module_name: "test_module"
        data_ports {
          direction: PORT_DIRECTION_OUTPUT
          name: "out_a_data"
          width: 32
          type { type_enum: BITS bit_count: 32 }
        }
        data_ports {
          direction: PORT_DIRECTION_OUTPUT
          name: "out_b_data"
          width: 32
          type { type_enum: BITS bit_count: 32 }
        }
        channel_interfaces {
          channel_name: "out_a"
          direction: CHANNEL_DIRECTION_SEND
          kind: CHANNEL_KIND_STREAMING
          streaming { data_port_name: "out_a_data" }
        }
        channel_interfaces {
          channel_name: "out_b"
          direction: CHANNEL_DIRECTION_SEND
          kind: CHANNEL_KIND_STREAMING
          streaming { data_port_name: "out_b_data" }
        }
      )pb",
      &initial_signature));
  block->SetSignature(initial_signature);

  EXPECT_THAT(Run(p.get()), IsOkAndHolds(true));

  EXPECT_THAT(block->GetSignature(), Optional(EqualsProto(R"pb(
                module_name: "test_module"
                data_ports {
                  direction: PORT_DIRECTION_OUTPUT
                  name: "out_a_data"
                  width: 32
                  type { type_enum: BITS bit_count: 32 }
                }
                channel_interfaces {
                  channel_name: "out_a"
                  direction: CHANNEL_DIRECTION_SEND
                  kind: CHANNEL_KIND_STREAMING
                  streaming { data_port_name: "out_a_data" }
                }
                channel_interfaces {
                  channel_name: "out_b"
                  direction: CHANNEL_DIRECTION_SEND
                  kind: CHANNEL_KIND_STREAMING
                  streaming { data_port_name: "out_a_data" }
                }
              )pb")));
}

TEST_F(MergePortsPassTest, UpdateSignatureReceiveReadyPorts) {
  auto p = CreatePackage();
  BlockBuilder bb(TestName(), p.get());
  bb.InputPort("in_a_data", p->GetBitsType(32));
  bb.InputPort("in_b_data", p->GetBitsType(32));
  BValue rdy = bb.Literal(UBits(1, 1));
  bb.OutputPort("in_a_rdy", rdy);
  bb.OutputPort("in_b_rdy", rdy);
  XLS_ASSERT_OK_AND_ASSIGN(Block * block, bb.Build());

  XLS_ASSERT_OK(block->AddChannelPortMetadata(ChannelPortMetadata{
      .channel_name = "in_a",
      .type = p->GetBitsType(32),
      .direction = ChannelDirection::kReceive,
      .channel_kind = ChannelKind::kStreaming,
      .flop_kind = FlopKind::kNone,
      .data_port = "in_a_data",
      .ready_port = "in_a_rdy",
  }));
  XLS_ASSERT_OK(block->AddChannelPortMetadata(ChannelPortMetadata{
      .channel_name = "in_b",
      .type = p->GetBitsType(32),
      .direction = ChannelDirection::kReceive,
      .channel_kind = ChannelKind::kStreaming,
      .flop_kind = FlopKind::kNone,
      .data_port = "in_b_data",
      .ready_port = "in_b_rdy",
  }));

  verilog::ModuleSignatureProto initial_signature;
  ASSERT_TRUE(google::protobuf::TextFormat::ParseFromString(
      R"pb(
        module_name: "test_module"
        data_ports {
          direction: PORT_DIRECTION_OUTPUT
          name: "in_a_rdy"
          width: 1
          type { type_enum: BITS bit_count: 1 }
        }
        data_ports {
          direction: PORT_DIRECTION_OUTPUT
          name: "in_b_rdy"
          width: 1
          type { type_enum: BITS bit_count: 1 }
        }
        channel_interfaces {
          channel_name: "in_a"
          direction: CHANNEL_DIRECTION_RECEIVE
          kind: CHANNEL_KIND_STREAMING
          streaming { data_port_name: "in_a_data" ready_port_name: "in_a_rdy" }
        }
        channel_interfaces {
          channel_name: "in_b"
          direction: CHANNEL_DIRECTION_RECEIVE
          kind: CHANNEL_KIND_STREAMING
          streaming { data_port_name: "in_b_data" ready_port_name: "in_b_rdy" }
        }
      )pb",
      &initial_signature));
  block->SetSignature(initial_signature);

  EXPECT_THAT(Run(p.get()), IsOkAndHolds(true));

  EXPECT_THAT(
      block->GetSignature(), Optional(EqualsProto(R"pb(
        module_name: "test_module"
        data_ports {
          direction: PORT_DIRECTION_OUTPUT
          name: "in_a_rdy"
          width: 1
          type { type_enum: BITS bit_count: 1 }
        }
        channel_interfaces {
          channel_name: "in_a"
          direction: CHANNEL_DIRECTION_RECEIVE
          kind: CHANNEL_KIND_STREAMING
          streaming { data_port_name: "in_a_data" ready_port_name: "in_a_rdy" }
        }
        channel_interfaces {
          channel_name: "in_b"
          direction: CHANNEL_DIRECTION_RECEIVE
          kind: CHANNEL_KIND_STREAMING
          streaming { data_port_name: "in_b_data" ready_port_name: "in_a_rdy" }
        }
      )pb")));
}

TEST_F(MergePortsPassTest, UpdateSignatureSingleValueChannels) {
  auto p = CreatePackage();
  BlockBuilder bb(TestName(), p.get());
  BValue in = bb.InputPort("in", p->GetBitsType(32));
  bb.OutputPort("sv_a", in);
  bb.OutputPort("sv_b", in);
  XLS_ASSERT_OK_AND_ASSIGN(Block * block, bb.Build());

  XLS_ASSERT_OK(block->AddChannelPortMetadata(ChannelPortMetadata{
      .channel_name = "sv_a",
      .type = p->GetBitsType(32),
      .direction = ChannelDirection::kSend,
      .channel_kind = ChannelKind::kSingleValue,
      .flop_kind = FlopKind::kNone,
      .data_port = "sv_a",
  }));
  XLS_ASSERT_OK(block->AddChannelPortMetadata(ChannelPortMetadata{
      .channel_name = "sv_b",
      .type = p->GetBitsType(32),
      .direction = ChannelDirection::kSend,
      .channel_kind = ChannelKind::kSingleValue,
      .flop_kind = FlopKind::kNone,
      .data_port = "sv_b",
  }));

  verilog::ModuleSignatureProto initial_signature;
  ASSERT_TRUE(google::protobuf::TextFormat::ParseFromString(
      R"pb(
        module_name: "test_module"
        data_ports {
          direction: PORT_DIRECTION_OUTPUT
          name: "sv_a"
          width: 32
          type { type_enum: BITS bit_count: 32 }
        }
        data_ports {
          direction: PORT_DIRECTION_OUTPUT
          name: "sv_b"
          width: 32
          type { type_enum: BITS bit_count: 32 }
        }
        channel_interfaces {
          channel_name: "sv_a"
          direction: CHANNEL_DIRECTION_SEND
          kind: CHANNEL_KIND_SINGLE_VALUE
          single_value { data_port_name: "sv_a" }
        }
        channel_interfaces {
          channel_name: "sv_b"
          direction: CHANNEL_DIRECTION_SEND
          kind: CHANNEL_KIND_SINGLE_VALUE
          single_value { data_port_name: "sv_b" }
        }
      )pb",
      &initial_signature));
  block->SetSignature(initial_signature);

  EXPECT_THAT(Run(p.get()), IsOkAndHolds(true));

  EXPECT_THAT(block->GetSignature(), Optional(EqualsProto(R"pb(
                module_name: "test_module"
                data_ports {
                  direction: PORT_DIRECTION_OUTPUT
                  name: "sv_a"
                  width: 32
                  type { type_enum: BITS bit_count: 32 }
                }
                channel_interfaces {
                  channel_name: "sv_a"
                  direction: CHANNEL_DIRECTION_SEND
                  kind: CHANNEL_KIND_SINGLE_VALUE
                  single_value { data_port_name: "sv_a" }
                }
                channel_interfaces {
                  channel_name: "sv_b"
                  direction: CHANNEL_DIRECTION_SEND
                  kind: CHANNEL_KIND_SINGLE_VALUE
                  single_value { data_port_name: "sv_a" }
                }
              )pb")));
}

}  // namespace
}  // namespace xls::codegen
