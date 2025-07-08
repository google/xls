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
#include "xls/codegen/codegen_pass.h"
#include "xls/codegen/xls_metrics.pb.h"
#include "xls/common/status/matchers.h"
#include "xls/estimators/delay_model/delay_estimator.h"
#include "xls/estimators/delay_model/delay_estimators.h"
#include "xls/ir/bits.h"
#include "xls/ir/block.h"
#include "xls/ir/fileno.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/op.h"
#include "xls/ir/package.h"
#include "xls/ir/register.h"
#include "xls/ir/source_location.h"
#include "xls/ir/type.h"
#include "xls/ir/value.h"
#include "xls/scheduling/pipeline_schedule.h"
#include "xls/scheduling/run_pipeline_schedule.h"
#include "xls/scheduling/scheduling_options.h"

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
  BValue rst = bb.ResetPort(
      "rst", ResetBehavior{.asynchronous = false, .active_low = false});

  BValue a = bb.InputPort("a", u32);
  BValue b = bb.InputPort("b", u32);
  BValue c = bb.Subtract(a, b);

  BValue p0_c = bb.InsertRegister("p0_c", c, rst, Value(UBits(0, 32)));

  BValue p1_c = bb.InsertRegister("p1_c", p0_c, rst, Value(UBits(0, 32)));

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
      RunPipelineSchedule(f, *delay_estimator,
                          SchedulingOptions().pipeline_stages(3)));

  XLS_ASSERT_OK_AND_ASSIGN(
      CodegenContext context,
      FunctionBaseToPipelinedBlock(
          schedule,
          CodegenOptions().flop_inputs(false).flop_outputs(false).clock_name(
              "clk"),
          f));

  XLS_ASSERT_OK_AND_ASSIGN(BlockMetricsProto proto,
                           GenerateBlockMetrics(context.top_block()));

  EXPECT_EQ(proto.flop_count(), schedule.CountFinalInteriorPipelineRegisters());
}

TEST(BlockMetricsGeneratorTest, DelayModel) {
  Package package("test");
  BlockBuilder bb("pass_thru", &package);
  bb.OutputPort("out", bb.InputPort("in", package.GetBitsType(32)));
  XLS_ASSERT_OK_AND_ASSIGN(Block * block, bb.Build());

  XLS_ASSERT_OK_AND_ASSIGN(DelayEstimator * delay_estimator,
                           GetDelayEstimator("unit"));
  XLS_ASSERT_OK_AND_ASSIGN(BlockMetricsProto proto,
                           GenerateBlockMetrics(block, delay_estimator));
  EXPECT_EQ(proto.delay_model(), "unit");
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

TEST(BlockMetricsGeneratorTest, BillOfMaterials) {
  Package package("test");
  SourceLocation loc1 = package.AddSourceLocation("foo", Lineno(20), Colno(8));
  SourceLocation loc2 = package.AddSourceLocation("bar", Lineno(25), Colno(12));
  SourceLocation loc3(Fileno(5), Lineno(60), Colno(4));

  FunctionBuilder fb("test_func", &package);
  BValue x = fb.Param("x", package.GetBitsType(24));
  BValue y = fb.Param("y", package.GetBitsType(16));
  BValue mac =
      fb.Add(fb.UMul(x, y, 32, SourceInfo(loc1)),
             fb.Literal(UBits(53, 32), SourceInfo(loc3)), SourceInfo(loc2));

  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.BuildWithReturnValue(mac));

  XLS_ASSERT_OK_AND_ASSIGN(const DelayEstimator* delay_estimator,
                           GetDelayEstimator("unit"));

  XLS_ASSERT_OK_AND_ASSIGN(
      PipelineSchedule schedule,
      RunPipelineSchedule(f, *delay_estimator,
                          SchedulingOptions().pipeline_stages(1)));

  XLS_ASSERT_OK_AND_ASSIGN(
      CodegenContext context,
      FunctionBaseToPipelinedBlock(
          schedule,
          CodegenOptions().flop_inputs(false).flop_outputs(false).clock_name(
              "clk"),
          f));

  XLS_ASSERT_OK_AND_ASSIGN(BlockMetricsProto proto,
                           GenerateBlockMetrics(context.top_block()));

  EXPECT_EQ(proto.bill_of_materials_size(), 6);
  EXPECT_EQ(proto.bill_of_materials(0).op(), ToOpProto(Op::kInputPort));
  EXPECT_EQ(proto.bill_of_materials(0).kind(), BOM_KIND_MISC);
  EXPECT_EQ(proto.bill_of_materials(0).output_width(), 24);
  EXPECT_EQ(proto.bill_of_materials(0).maximum_input_width(), 0);
  EXPECT_EQ(proto.bill_of_materials(0).number_of_arguments(), 0);
  EXPECT_EQ(proto.bill_of_materials(0).location_size(), 0);

  EXPECT_EQ(proto.bill_of_materials(1).op(), ToOpProto(Op::kInputPort));
  EXPECT_EQ(proto.bill_of_materials(1).kind(), BOM_KIND_MISC);
  EXPECT_EQ(proto.bill_of_materials(1).output_width(), 16);
  EXPECT_EQ(proto.bill_of_materials(1).maximum_input_width(), 0);
  EXPECT_EQ(proto.bill_of_materials(1).number_of_arguments(), 0);
  EXPECT_EQ(proto.bill_of_materials(1).location_size(), 0);

  EXPECT_EQ(proto.bill_of_materials(2).op(), ToOpProto(Op::kUMul));
  EXPECT_EQ(proto.bill_of_materials(2).kind(), BOM_KIND_MULTIPLIER);
  EXPECT_EQ(proto.bill_of_materials(2).output_width(), 32);
  EXPECT_EQ(proto.bill_of_materials(2).maximum_input_width(), 24);
  EXPECT_EQ(proto.bill_of_materials(2).number_of_arguments(), 2);
  EXPECT_EQ(proto.bill_of_materials(2).location_size(), 1);
  EXPECT_EQ(proto.bill_of_materials(2).location(0).file(), "foo");
  EXPECT_EQ(proto.bill_of_materials(2).location(0).line(), 20);
  EXPECT_EQ(proto.bill_of_materials(2).location(0).col(), 8);

  EXPECT_EQ(proto.bill_of_materials(3).op(), ToOpProto(Op::kLiteral));
  EXPECT_EQ(proto.bill_of_materials(3).kind(), BOM_KIND_INSIGNIFICANT);
  EXPECT_EQ(proto.bill_of_materials(3).output_width(), 32);
  EXPECT_EQ(proto.bill_of_materials(3).maximum_input_width(), 0);
  EXPECT_EQ(proto.bill_of_materials(3).number_of_arguments(), 0);
  EXPECT_EQ(proto.bill_of_materials(3).location_size(), 1);
  EXPECT_FALSE(proto.bill_of_materials(3).location(0).has_file());
  EXPECT_TRUE(proto.bill_of_materials(3).location(0).has_line());
  EXPECT_TRUE(proto.bill_of_materials(3).location(0).has_col());
  EXPECT_EQ(proto.bill_of_materials(3).location(0).line(), 60);
  EXPECT_EQ(proto.bill_of_materials(3).location(0).col(), 4);

  EXPECT_EQ(proto.bill_of_materials(4).op(), ToOpProto(Op::kAdd));
  EXPECT_EQ(proto.bill_of_materials(4).kind(), BOM_KIND_ADDER);
  EXPECT_EQ(proto.bill_of_materials(4).output_width(), 32);
  EXPECT_EQ(proto.bill_of_materials(4).maximum_input_width(), 32);
  EXPECT_EQ(proto.bill_of_materials(4).number_of_arguments(), 2);
  EXPECT_EQ(proto.bill_of_materials(4).location_size(), 1);
  EXPECT_EQ(proto.bill_of_materials(4).location(0).file(), "bar");
  EXPECT_EQ(proto.bill_of_materials(4).location(0).line(), 25);
  EXPECT_EQ(proto.bill_of_materials(4).location(0).col(), 12);

  EXPECT_EQ(proto.bill_of_materials(5).op(), ToOpProto(Op::kOutputPort));
  EXPECT_EQ(proto.bill_of_materials(5).kind(), BOM_KIND_MISC);
  EXPECT_EQ(proto.bill_of_materials(5).output_width(), 0);
  EXPECT_EQ(proto.bill_of_materials(5).maximum_input_width(), 32);
  EXPECT_EQ(proto.bill_of_materials(5).number_of_arguments(), 1);
  EXPECT_EQ(proto.bill_of_materials(5).location_size(), 0);
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
