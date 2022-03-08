// Copyright 2022 The XLS Authors
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

#include "xls/codegen/block_metrics.h"

#include "gtest/gtest.h"
#include "xls/codegen/block_conversion.h"
#include "xls/codegen/codegen_options.h"
#include "xls/codegen/xls_metrics.pb.h"
#include "xls/common/status/matchers.h"
#include "xls/delay_model/delay_estimators.h"
#include "xls/ir/block.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/ir_parser.h"
#include "xls/ir/package.h"
#include "xls/ir/type.h"
#include "xls/scheduling/pipeline_schedule.h"

namespace xls {
namespace verilog {
namespace {

TEST(BlockMetricsGeneratorTest, ZeroRegisters) {
  Package package("test");

  Type* u32 = package.GetBitsType(32);
  BlockBuilder bb("test_block", &package);
  BValue a = bb.InputPort("a", u32);
  BValue b = bb.InputPort("b", u32);
  bb.OutputPort("z", bb.Subtract(a, b));

  XLS_ASSERT_OK_AND_ASSIGN(Block * block, bb.Build());

  XLS_ASSERT_OK_AND_ASSIGN(BlockMetricsProto proto,
                           GenerateBlockMetrics(block));

  EXPECT_EQ(proto.flop_count(), 0);
}

TEST(BlockMetricsGeneratorTest, PipelineRegisters) {
  Package package("test");

  Type* u32 = package.GetBitsType(32);
  BlockBuilder bb("test_block", &package);

  XLS_ASSERT_OK(bb.block()->AddClockPort("clk"));
  BValue rst = bb.InputPort("rst", package.GetBitsType(1));

  BValue a = bb.InputPort("a", u32);
  BValue b = bb.InputPort("b", u32);
  BValue c = bb.Subtract(a, b);

  BValue p0_c = bb.InsertRegister("p0_c", c, rst,
                                  xls::Reset{.reset_value = Value(UBits(0, 32)),
                                             .asynchronous = false,
                                             .active_low = false});

  BValue p1_c = bb.InsertRegister("p1_c", p0_c, rst,
                                  xls::Reset{.reset_value = Value(UBits(0, 32)),
                                             .asynchronous = false,
                                             .active_low = false});

  bb.OutputPort("z", p1_c);

  XLS_ASSERT_OK_AND_ASSIGN(Block * block, bb.Build());

  XLS_ASSERT_OK_AND_ASSIGN(BlockMetricsProto proto,
                           GenerateBlockMetrics(block));

  EXPECT_EQ(proto.flop_count(), 64);
}

TEST(BlockMetricsGeneratorTest, PipelineRegistersCount) {
  Package package("test");

  FunctionBuilder fb("test_func", &package);
  BValue x = fb.Param("x", package.GetBitsType(32));
  BValue y = fb.Param("y", package.GetBitsType(32));
  XLS_ASSERT_OK_AND_ASSIGN(
      Function * f, fb.BuildWithReturnValue(fb.Negate(fb.Not(fb.Add(x, y)))));

  XLS_ASSERT_OK_AND_ASSIGN(const DelayEstimator* delay_estimator,
                           GetDelayEstimator("unit"));

  XLS_ASSERT_OK_AND_ASSIGN(
      PipelineSchedule schedule,
      PipelineSchedule::Run(f, *delay_estimator,
                            SchedulingOptions().pipeline_stages(3)));

  XLS_ASSERT_OK_AND_ASSIGN(
      Block * block,
      FunctionToPipelinedBlock(
          schedule,
          CodegenOptions().flop_inputs(false).flop_outputs(false).clock_name(
              "clk"),
          f));

  XLS_ASSERT_OK_AND_ASSIGN(BlockMetricsProto proto,
                           GenerateBlockMetrics(block));

  EXPECT_EQ(proto.flop_count(), schedule.CountFinalInteriorPipelineRegisters());
}

TEST(BlockMetricsGeneratorTest, CombinationalPath) {
  Package package("test");
  Type* u32 = package.GetBitsType(32);
  XLS_ASSERT_OK_AND_ASSIGN(DelayEstimator * delay_estimator,
                           GetDelayEstimator("unit"));

  {
    // Pass through from input to output.
    BlockBuilder bb("pass_thru", &package);
    bb.OutputPort("out", bb.InputPort("in", u32));
    XLS_ASSERT_OK_AND_ASSIGN(Block * block, bb.Build());
    XLS_ASSERT_OK_AND_ASSIGN(BlockMetricsProto proto,
                             GenerateBlockMetrics(block, delay_estimator));
    EXPECT_TRUE(proto.feedthrough_path_exists());
    EXPECT_TRUE(proto.has_max_feedthrough_path_delay_ps());
    EXPECT_EQ(proto.max_feedthrough_path_delay_ps(), 0);
  }

  {
    // Empty block.
    BlockBuilder bb("empty", &package);
    XLS_ASSERT_OK_AND_ASSIGN(Block * block, bb.Build());
    XLS_ASSERT_OK_AND_ASSIGN(BlockMetricsProto proto,
                             GenerateBlockMetrics(block, delay_estimator));
    EXPECT_FALSE(proto.feedthrough_path_exists());
    EXPECT_FALSE(proto.has_max_feedthrough_path_delay_ps());
  }

  {
    // Combinational logic between input and output.
    BlockBuilder bb("combo_path", &package);
    bb.OutputPort("out", bb.Not(bb.Add(bb.InputPort("in", u32),
                                       bb.Literal(UBits(42, 32)))));
    XLS_ASSERT_OK_AND_ASSIGN(Block * block, bb.Build());
    XLS_ASSERT_OK_AND_ASSIGN(BlockMetricsProto proto,
                             GenerateBlockMetrics(block, delay_estimator));
    EXPECT_TRUE(proto.feedthrough_path_exists());
    EXPECT_TRUE(proto.has_max_feedthrough_path_delay_ps());
    EXPECT_EQ(proto.max_feedthrough_path_delay_ps(), 2);
  }

  {
    // Multiple combo paths.
    BlockBuilder bb("combo_path", &package);
    BValue in0 = bb.InputPort("in0", u32);
    BValue in1 = bb.InputPort("in1", u32);
    bb.OutputPort("out0", bb.Not(bb.Add(in0, in1)));
    bb.OutputPort("out1", bb.Not(bb.Not(bb.Not(in1))));
    XLS_ASSERT_OK_AND_ASSIGN(Block * block, bb.Build());
    XLS_ASSERT_OK_AND_ASSIGN(BlockMetricsProto proto,
                             GenerateBlockMetrics(block, delay_estimator));
    EXPECT_TRUE(proto.feedthrough_path_exists());
    EXPECT_TRUE(proto.has_max_feedthrough_path_delay_ps());
    EXPECT_EQ(proto.max_feedthrough_path_delay_ps(), 3);
  }

  {
    // Flopped path between input and output.
    BlockBuilder bb("flopped_path", &package);
    bb.OutputPort("out", bb.InsertRegister("foo_reg", bb.InputPort("in", u32)));
    XLS_ASSERT_OK(bb.block()->AddClockPort("clk"));
    XLS_ASSERT_OK_AND_ASSIGN(Block * block, bb.Build());
    XLS_ASSERT_OK_AND_ASSIGN(BlockMetricsProto proto,
                             GenerateBlockMetrics(block, delay_estimator));
    EXPECT_FALSE(proto.feedthrough_path_exists());
    EXPECT_FALSE(proto.has_max_feedthrough_path_delay_ps());
  }

  {
    // Flopped and combo path between input and output.
    BlockBuilder bb("flopped_and_combo_path", &package);
    BValue in = bb.InputPort("in", u32);
    bb.OutputPort("out", bb.Add(bb.InsertRegister("foo_reg", in), in));
    XLS_ASSERT_OK(bb.block()->AddClockPort("clk"));
    XLS_ASSERT_OK_AND_ASSIGN(Block * block, bb.Build());
    XLS_ASSERT_OK_AND_ASSIGN(BlockMetricsProto proto,
                             GenerateBlockMetrics(block, delay_estimator));
    EXPECT_TRUE(proto.feedthrough_path_exists());
    EXPECT_TRUE(proto.has_max_feedthrough_path_delay_ps());
    EXPECT_EQ(proto.max_feedthrough_path_delay_ps(), 1);
  }
}

TEST(BlockMetricsGeneratorTest, DelayMetrics) {
  Package package("test");
  Type* u32 = package.GetBitsType(32);
  Type* u1 = package.GetBitsType(1);
  XLS_ASSERT_OK_AND_ASSIGN(DelayEstimator * delay_estimator,
                           GetDelayEstimator("unit"));

  {
    // No registers.
    BlockBuilder bb("no_regs", &package);
    bb.OutputPort("out", bb.InputPort("in", u32));
    XLS_ASSERT_OK_AND_ASSIGN(Block * block, bb.Build());
    XLS_ASSERT_OK_AND_ASSIGN(BlockMetricsProto proto,
                             GenerateBlockMetrics(block, delay_estimator));
    EXPECT_FALSE(proto.has_max_reg_to_reg_delay_ps());
    EXPECT_FALSE(proto.has_max_input_to_reg_delay_ps());
    EXPECT_FALSE(proto.has_max_reg_to_output_delay_ps());
    EXPECT_TRUE(proto.has_max_feedthrough_path_delay_ps());
    EXPECT_EQ(proto.max_feedthrough_path_delay_ps(), 0);
  }
  {
    // Input flopped then output.
    BlockBuilder bb("input_flopped", &package);
    bb.OutputPort("out", bb.InsertRegister("foo_reg", bb.InputPort("in", u32)));
    XLS_ASSERT_OK(bb.block()->AddClockPort("clk"));
    XLS_ASSERT_OK_AND_ASSIGN(Block * block, bb.Build());
    XLS_ASSERT_OK_AND_ASSIGN(BlockMetricsProto proto,
                             GenerateBlockMetrics(block, delay_estimator));
    EXPECT_FALSE(proto.has_max_reg_to_reg_delay_ps());
    EXPECT_TRUE(proto.has_max_input_to_reg_delay_ps());
    EXPECT_EQ(proto.max_input_to_reg_delay_ps(), 0);
    EXPECT_TRUE(proto.has_max_reg_to_output_delay_ps());
    EXPECT_EQ(proto.max_reg_to_output_delay_ps(), 0);
    EXPECT_FALSE(proto.has_max_feedthrough_path_delay_ps());
  }
  {
    // Input flopped after logic.
    BlockBuilder bb("input_logic", &package);
    bb.OutputPort(
        "out", bb.InsertRegister("foo_reg", bb.Not(bb.InputPort("in", u32))));
    XLS_ASSERT_OK(bb.block()->AddClockPort("clk"));
    XLS_ASSERT_OK_AND_ASSIGN(Block * block, bb.Build());
    XLS_ASSERT_OK_AND_ASSIGN(BlockMetricsProto proto,
                             GenerateBlockMetrics(block, delay_estimator));
    EXPECT_FALSE(proto.has_max_reg_to_reg_delay_ps());
    EXPECT_TRUE(proto.has_max_input_to_reg_delay_ps());
    EXPECT_EQ(proto.max_input_to_reg_delay_ps(), 1);
    EXPECT_TRUE(proto.has_max_reg_to_output_delay_ps());
    EXPECT_EQ(proto.max_reg_to_output_delay_ps(), 0);
    EXPECT_FALSE(proto.has_max_feedthrough_path_delay_ps());
  }
  {
    // Logic before output.
    BlockBuilder bb("output_logic", &package);
    bb.OutputPort(
        "out",
        bb.Not(bb.Not(bb.InsertRegister("foo_reg", bb.InputPort("in", u32)))));
    XLS_ASSERT_OK(bb.block()->AddClockPort("clk"));
    XLS_ASSERT_OK_AND_ASSIGN(Block * block, bb.Build());
    XLS_ASSERT_OK_AND_ASSIGN(BlockMetricsProto proto,
                             GenerateBlockMetrics(block, delay_estimator));
    EXPECT_FALSE(proto.has_max_reg_to_reg_delay_ps());
    EXPECT_TRUE(proto.has_max_input_to_reg_delay_ps());
    EXPECT_EQ(proto.max_input_to_reg_delay_ps(), 0);
    EXPECT_TRUE(proto.has_max_reg_to_output_delay_ps());
    EXPECT_EQ(proto.max_reg_to_output_delay_ps(), 2);
    EXPECT_FALSE(proto.has_max_feedthrough_path_delay_ps());
  }
  {
    // Input port to load enable with logic.
    BlockBuilder bb("load_en", &package);
    BValue in = bb.InputPort("in", u32);
    BValue le = bb.InputPort("le", u1);
    bb.OutputPort("out", bb.InsertRegister("foo_reg", in,
                                           /*load_enable=*/bb.Not(bb.Not(le))));
    XLS_ASSERT_OK(bb.block()->AddClockPort("clk"));
    XLS_ASSERT_OK_AND_ASSIGN(Block * block, bb.Build());
    XLS_ASSERT_OK_AND_ASSIGN(BlockMetricsProto proto,
                             GenerateBlockMetrics(block, delay_estimator));
    EXPECT_FALSE(proto.has_max_reg_to_reg_delay_ps());
    EXPECT_TRUE(proto.has_max_input_to_reg_delay_ps());
    EXPECT_EQ(proto.max_input_to_reg_delay_ps(), 2);
    EXPECT_TRUE(proto.has_max_reg_to_output_delay_ps());
    EXPECT_EQ(proto.max_reg_to_output_delay_ps(), 0);
    EXPECT_FALSE(proto.has_max_feedthrough_path_delay_ps());
  }
  {
    // Input double flopped then output.
    BlockBuilder bb("double_flopped", &package);
    bb.OutputPort(
        "out", bb.InsertRegister(
                   "reg1", bb.InsertRegister("reg0", bb.InputPort("in", u32))));
    XLS_ASSERT_OK(bb.block()->AddClockPort("clk"));
    XLS_ASSERT_OK_AND_ASSIGN(Block * block, bb.Build());
    XLS_ASSERT_OK_AND_ASSIGN(BlockMetricsProto proto,
                             GenerateBlockMetrics(block, delay_estimator));
    EXPECT_TRUE(proto.has_max_reg_to_reg_delay_ps());
    EXPECT_EQ(proto.max_reg_to_reg_delay_ps(), 0);
    EXPECT_TRUE(proto.has_max_input_to_reg_delay_ps());
    EXPECT_EQ(proto.max_input_to_reg_delay_ps(), 0);
    EXPECT_TRUE(proto.has_max_reg_to_output_delay_ps());
    EXPECT_EQ(proto.max_reg_to_output_delay_ps(), 0);
    EXPECT_FALSE(proto.has_max_feedthrough_path_delay_ps());
  }
  {
    // Combo logic between registers.
    BlockBuilder bb("logic_between_regs", &package);
    BValue x_d = bb.InsertRegister("x_d", bb.InputPort("x", u32));
    BValue y_d = bb.InsertRegister("y_d", bb.InputPort("y", u32));
    BValue sum = bb.InsertRegister("sum", bb.Add(x_d, y_d));
    BValue sum_not = bb.InsertRegister("sum_not", bb.Add(x_d, bb.Not(y_d)));
    bb.OutputPort("sum_out", sum);
    bb.OutputPort("sum_not_out", sum_not);
    XLS_ASSERT_OK(bb.block()->AddClockPort("clk"));
    XLS_ASSERT_OK_AND_ASSIGN(Block * block, bb.Build());
    XLS_ASSERT_OK_AND_ASSIGN(BlockMetricsProto proto,
                             GenerateBlockMetrics(block, delay_estimator));
    EXPECT_TRUE(proto.has_max_reg_to_reg_delay_ps());
    EXPECT_EQ(proto.max_reg_to_reg_delay_ps(), 2);
    EXPECT_TRUE(proto.has_max_input_to_reg_delay_ps());
    EXPECT_EQ(proto.max_input_to_reg_delay_ps(), 0);
    EXPECT_TRUE(proto.has_max_reg_to_output_delay_ps());
    EXPECT_EQ(proto.max_reg_to_output_delay_ps(), 0);
    EXPECT_FALSE(proto.has_max_feedthrough_path_delay_ps());
  }
}

}  // namespace
}  // namespace verilog
}  // namespace xls
