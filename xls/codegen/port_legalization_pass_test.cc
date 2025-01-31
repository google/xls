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

#include <optional>
#include <string>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "xls/codegen/block_conversion.h"
#include "xls/codegen/codegen_options.h"
#include "xls/codegen/codegen_pass.h"
#include "xls/common/status/matchers.h"
#include "xls/ir/bits.h"
#include "xls/ir/block.h"
#include "xls/ir/channel.h"
#include "xls/ir/channel_ops.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/ir_matcher.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/nodes.h"

namespace xls::verilog {
namespace {

using ::absl_testing::IsOkAndHolds;
using ::testing::ElementsAre;
using ::testing::Eq;
using ::testing::IsEmpty;
using ::testing::Optional;
namespace m = ::xls::op_matchers;

class PortLegalizationPassTest : public IrTestBase {
 protected:
  absl::StatusOr<bool> Run(Block* block) {
    CodegenPassResults results;
    CodegenPassUnit unit(block->package(), block);
    return PortLegalizationPass().Run(&unit, CodegenPassOptions(), &results);
  }
};

TEST_F(PortLegalizationPassTest, APlusB) {
  auto p = CreatePackage();

  Type* u32 = p->GetBitsType(32);
  BlockBuilder bb(TestName(), p.get());
  BValue a = bb.InputPort("a", u32);
  BValue b = bb.InputPort("b", u32);
  bb.OutputPort("out", bb.Add(a, b));
  XLS_ASSERT_OK_AND_ASSIGN(Block * block, bb.Build());

  // There are no zero-width ports so nothing should be removed.
  EXPECT_EQ(block->GetPorts().size(), 3);
  EXPECT_THAT(Run(block), IsOkAndHolds(false));
  EXPECT_EQ(block->GetPorts().size(), 3);
}

TEST_F(PortLegalizationPassTest, ZeroWidthInputsAndOutput) {
  auto p = CreatePackage();

  BlockBuilder bb(TestName(), p.get());
  BValue in = bb.InputPort("in", p->GetTupleType({}));
  bb.OutputPort("out", in);
  XLS_ASSERT_OK_AND_ASSIGN(Block * block, bb.Build());

  // All ports should be removed.
  EXPECT_EQ(block->GetPorts().size(), 2);
  ASSERT_THAT(Run(block), IsOkAndHolds(true));
  EXPECT_EQ(block->GetPorts().size(), 0);
}

TEST_F(PortLegalizationPassTest, ZeroWidthInput) {
  auto p = CreatePackage();

  BlockBuilder bb(TestName(), p.get());
  BValue a = bb.InputPort("a", p->GetBitsType(0));
  BValue b = bb.InputPort("b", p->GetBitsType(32));
  bb.OutputPort("out", bb.Concat({a, b}));
  XLS_ASSERT_OK_AND_ASSIGN(Block * block, bb.Build());

  // One port should be removed.
  EXPECT_EQ(block->GetPorts().size(), 3);
  ASSERT_THAT(Run(block), IsOkAndHolds(true));
  ASSERT_EQ(block->GetPorts().size(), 2);
  EXPECT_EQ(std::get<InputPort*>(block->GetPorts()[0])->GetName(), "b");
  EXPECT_EQ(std::get<OutputPort*>(block->GetPorts()[1])->GetName(), "out");
}

TEST_F(PortLegalizationPassTest, MultipleBlocksWithZeroWidthPorts) {
  auto p = CreatePackage();

  BlockBuilder bb0(absl::StrCat(TestName(), "0"), p.get());
  BValue a = bb0.InputPort("a", p->GetBitsType(0));
  BValue b = bb0.InputPort("b", p->GetBitsType(32));
  bb0.OutputPort("out", bb0.Concat({a, b}));
  XLS_ASSERT_OK_AND_ASSIGN(Block * block0, bb0.Build());

  BlockBuilder bb1(absl::StrCat(TestName(), "1"), p.get());
  bb1.InputPort("c", p->GetBitsType(32));
  bb1.InputPort("d", p->GetBitsType(32));
  bb1.OutputPort("out", bb1.Literal(UBits(0, 0)));
  XLS_ASSERT_OK_AND_ASSIGN(Block * block1, bb1.Build());

  EXPECT_THAT(block0->GetInputPorts(),
              ElementsAre(m::InputPort("a"), m::InputPort("b")));
  EXPECT_THAT(block0->GetOutputPorts(),
              ElementsAre(m::OutputPort("out", m::Concat())));
  EXPECT_THAT(block1->GetInputPorts(),
              ElementsAre(m::InputPort("c"), m::InputPort("d")));
  EXPECT_THAT(block1->GetOutputPorts(),
              ElementsAre(m::OutputPort("out", m::Literal())));

  ASSERT_THAT(Run(block0), IsOkAndHolds(true));

  EXPECT_THAT(block0->GetInputPorts(), ElementsAre(m::InputPort("b")));
  EXPECT_THAT(block0->GetOutputPorts(),
              ElementsAre(m::OutputPort("out", m::Concat())));
  EXPECT_THAT(block1->GetInputPorts(),
              ElementsAre(m::InputPort("c"), m::InputPort("d")));
  EXPECT_THAT(block1->GetOutputPorts(), IsEmpty());
}

TEST_F(PortLegalizationPassTest, InstantiatedBlocksWithZeroWidthPorts) {
  auto p = CreatePackage();

  BlockBuilder empty_input_bb(absl::StrCat(TestName(), "_empty_input"),
                              p.get());
  BValue a = empty_input_bb.InputPort("a", p->GetBitsType(0));
  BValue b = empty_input_bb.InputPort("b", p->GetBitsType(32));
  empty_input_bb.OutputPort("out", empty_input_bb.Concat({a, b}));

  BlockBuilder empty_output_bb(absl::StrCat(TestName(), "_empty_output"),

                               p.get());

  BValue c = empty_output_bb.InputPort("c", p->GetBitsType(0));
  BValue d = empty_output_bb.InputPort("d", p->GetBitsType(32));
  empty_output_bb.OutputPort("out0", c);
  empty_output_bb.OutputPort("out1", d);

  XLS_ASSERT_OK_AND_ASSIGN(Block * empty_input_block, empty_input_bb.Build());
  XLS_ASSERT_OK_AND_ASSIGN(Block * empty_output_block, empty_output_bb.Build());

  BlockBuilder top_bb(absl::StrCat(TestName(), "_top"), p.get());
  XLS_ASSERT_OK_AND_ASSIGN(
      xls::Instantiation * empty_input_instantiation,
      top_bb.block()->AddBlockInstantiation(
          absl::StrFormat("%s_inst0", empty_input_block->name()),
          empty_input_block));
  XLS_ASSERT_OK_AND_ASSIGN(
      xls::Instantiation * empty_output_instantiation,
      top_bb.block()->AddBlockInstantiation(
          absl::StrFormat("%s_inst1", empty_output_block->name()),
          empty_output_block));
  BValue e = top_bb.InputPort("e", p->GetBitsType(0));
  BValue f = top_bb.InputPort("f", p->GetBitsType(32));

  top_bb.InstantiationInput(empty_output_instantiation, "c", e);
  top_bb.InstantiationInput(empty_output_instantiation, "d", f);
  BValue passed_e =
      top_bb.InstantiationOutput(empty_output_instantiation, "out0");
  BValue passed_f =
      top_bb.InstantiationOutput(empty_output_instantiation, "out1");

  top_bb.InstantiationInput(empty_input_instantiation, "a", passed_e);
  top_bb.InstantiationInput(empty_input_instantiation, "b", passed_f);
  BValue ef = top_bb.InstantiationOutput(empty_input_instantiation, "out");
  top_bb.OutputPort("out", ef);

  XLS_ASSERT_OK_AND_ASSIGN(Block * top, top_bb.Build());

  EXPECT_THAT(empty_input_block->GetInputPorts(),
              ElementsAre(m::InputPort("a"), m::InputPort("b")));
  EXPECT_THAT(empty_input_block->GetOutputPorts(),
              ElementsAre(m::OutputPort("out", m::Concat())));

  EXPECT_THAT(empty_output_block->GetInputPorts(),
              ElementsAre(m::InputPort("c"), m::InputPort("d")));
  EXPECT_THAT(empty_output_block->GetOutputPorts(),
              ElementsAre(m::OutputPort("out0", m::InputPort("c")),
                          m::OutputPort("out1", m::InputPort("d"))));

  EXPECT_THAT(top->GetInputPorts(),
              ElementsAre(m::InputPort("e"), m::InputPort("f")));
  EXPECT_THAT(top->GetOutputPorts(),
              ElementsAre(m::OutputPort("out", m::InstantiationOutput())));

  ASSERT_THAT(Run(top), IsOkAndHolds(true));

  EXPECT_EQ(empty_input_block->GetPorts().size(), 2);
  EXPECT_EQ(empty_output_block->GetPorts().size(), 2);
  EXPECT_EQ(top->GetPorts().size(), 2);

  EXPECT_THAT(empty_input_block->GetInputPorts(),
              ElementsAre(m::InputPort("b")));
  EXPECT_THAT(empty_input_block->GetOutputPorts(),
              ElementsAre(m::OutputPort("out", m::Concat())));

  EXPECT_THAT(empty_output_block->GetInputPorts(),
              ElementsAre(m::InputPort("d")));
  EXPECT_THAT(empty_output_block->GetOutputPorts(),
              ElementsAre(m::OutputPort("out1", m::InputPort("d"))));

  EXPECT_THAT(top->GetInputPorts(), ElementsAre(m::InputPort("f")));
  EXPECT_THAT(top->GetOutputPorts(),
              ElementsAre(m::OutputPort("out", m::InstantiationOutput())));
}

TEST_F(PortLegalizationPassTest, ZeroWidthChannelMetadata) {
  auto p = CreatePackage();

  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * ch_in, p->CreateStreamingChannel("in", ChannelOps::kReceiveOnly,
                                                 p->GetTupleType({})));

  TokenlessProcBuilder pb(TestName(), /*token_name=*/"tkn", p.get());
  pb.Receive(ch_in);
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, pb.Build({}));

  XLS_ASSERT_OK_AND_ASSIGN(CodegenPassUnit unit,
                           ProcToCombinationalBlock(proc, CodegenOptions()));
  Block* block = FindBlock(TestName(), p.get());

  {
    XLS_ASSERT_OK_AND_ASSIGN(ChannelPortMetadata in_metadata,
                             block->GetChannelPortMetadata("in"));
    EXPECT_EQ(in_metadata.channel_name, "in");
    EXPECT_THAT(in_metadata.data_port, Optional(std::string{"in"}));
    EXPECT_THAT(in_metadata.valid_port, Optional(std::string{"in_vld"}));
    EXPECT_THAT(in_metadata.ready_port, Optional(std::string{"in_rdy"}));

    EXPECT_THAT(block->GetDataPortForChannel("in"),
                IsOkAndHolds(Optional(m::InputPort("in"))));
    EXPECT_THAT(block->GetValidPortForChannel("in"),
                IsOkAndHolds(Optional(m::InputPort("in_vld"))));
    EXPECT_THAT(block->GetReadyPortForChannel("in"),
                IsOkAndHolds(Optional(m::OutputPort("in_rdy"))));
  }
  ASSERT_THAT(Run(block), IsOkAndHolds(true));

  {
    XLS_ASSERT_OK_AND_ASSIGN(ChannelPortMetadata in_metadata,
                             block->GetChannelPortMetadata("in"));
    // The data port should have been removed.
    EXPECT_THAT(in_metadata.data_port, Eq(std::nullopt));
    EXPECT_THAT(in_metadata.valid_port, Optional(std::string{"in_vld"}));
    EXPECT_THAT(in_metadata.ready_port, Optional(std::string{"in_rdy"}));

    EXPECT_THAT(block->GetDataPortForChannel("in"), IsOkAndHolds(std::nullopt));
    EXPECT_THAT(block->GetValidPortForChannel("in"),
                IsOkAndHolds(Optional(m::InputPort("in_vld"))));
    EXPECT_THAT(block->GetReadyPortForChannel("in"),
                IsOkAndHolds(Optional(m::OutputPort("in_rdy"))));
  }
}

}  // namespace
}  // namespace xls::verilog
