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

#include "xls/codegen/signature_generator.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "xls/codegen/codegen_options.h"
#include "xls/codegen/module_signature.pb.h"
#include "xls/common/status/matchers.h"
#include "xls/delay_model/delay_estimators.h"
#include "xls/ir/channel.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/package.h"
#include "xls/ir/type.h"
#include "xls/scheduling/pipeline_schedule.h"

namespace xls {
namespace verilog {
namespace {

TEST(SignatureGeneratorTest, CombinationalBlock) {
  Package package("test");
  FunctionBuilder fb("test", &package);
  auto a = fb.Param("a", package.GetBitsType(8));
  auto b = fb.Param("b", package.GetBitsType(32));
  fb.Param("c", package.GetBitsType(0));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f,
                           fb.BuildWithReturnValue(fb.Concat({a, b})));

  // Default options.
  XLS_ASSERT_OK_AND_ASSIGN(ModuleSignature sig,
                           GenerateSignature(CodegenOptions(), f));

  ASSERT_EQ(sig.data_inputs().size(), 3);

  EXPECT_EQ(sig.data_inputs()[0].direction(), DIRECTION_INPUT);
  EXPECT_EQ(sig.data_inputs()[0].name(), "a");
  EXPECT_EQ(sig.data_inputs()[0].width(), 8);

  EXPECT_EQ(sig.data_inputs()[1].direction(), DIRECTION_INPUT);
  EXPECT_EQ(sig.data_inputs()[1].name(), "b");
  EXPECT_EQ(sig.data_inputs()[1].width(), 32);

  EXPECT_EQ(sig.data_inputs()[2].direction(), DIRECTION_INPUT);
  EXPECT_EQ(sig.data_inputs()[2].name(), "c");
  EXPECT_EQ(sig.data_inputs()[2].width(), 0);

  ASSERT_EQ(sig.data_outputs().size(), 1);

  EXPECT_EQ(sig.data_outputs()[0].direction(), DIRECTION_OUTPUT);
  EXPECT_EQ(sig.data_outputs()[0].name(), "out");
  EXPECT_EQ(sig.data_outputs()[0].width(), 40);

  ASSERT_TRUE(sig.proto().has_combinational());
}

TEST(SignatureGeneratorTest, PipelinedFunction) {
  Package package("test");
  FunctionBuilder fb("test", &package);
  auto a = fb.Param("a", package.GetBitsType(32));
  auto b = fb.Param("b", package.GetBitsType(32));
  XLS_ASSERT_OK_AND_ASSIGN(
      Function * f, fb.BuildWithReturnValue(fb.Not(fb.Negate(fb.Add(a, b)))));

  XLS_ASSERT_OK_AND_ASSIGN(DelayEstimator * estimator,
                           GetDelayEstimator("unit"));
  XLS_ASSERT_OK_AND_ASSIGN(
      PipelineSchedule schedule,
      PipelineSchedule::Run(f, *estimator,
                            SchedulingOptions().pipeline_stages(4)));

  {
    XLS_ASSERT_OK_AND_ASSIGN(
        ModuleSignature sig,
        GenerateSignature(
            CodegenOptions()
                .module_name("foobar")
                .reset("rst_n", /*asynchronous=*/false, /*active_low=*/true)
                .clock_name("the_clock"),
            f, schedule));

    ASSERT_EQ(sig.data_inputs().size(), 2);

    EXPECT_EQ(sig.data_inputs()[0].direction(), DIRECTION_INPUT);
    EXPECT_EQ(sig.data_inputs()[0].name(), "a");
    EXPECT_EQ(sig.data_inputs()[0].width(), 32);

    EXPECT_EQ(sig.data_inputs()[1].direction(), DIRECTION_INPUT);
    EXPECT_EQ(sig.data_inputs()[1].name(), "b");
    EXPECT_EQ(sig.data_inputs()[1].width(), 32);

    ASSERT_EQ(sig.data_outputs().size(), 1);

    EXPECT_EQ(sig.data_outputs()[0].direction(), DIRECTION_OUTPUT);
    EXPECT_EQ(sig.data_outputs()[0].name(), "out");
    EXPECT_EQ(sig.data_outputs()[0].width(), 32);

    EXPECT_EQ(sig.proto().reset().name(), "rst_n");
    EXPECT_FALSE(sig.proto().reset().asynchronous());
    EXPECT_TRUE(sig.proto().reset().active_low());
    EXPECT_EQ(sig.proto().clock_name(), "the_clock");

    ASSERT_TRUE(sig.proto().has_pipeline());
    EXPECT_EQ(sig.proto().pipeline().latency(), 3);
  }

  {
    // Adding flopping of the inputs should increase latency by one.
    XLS_ASSERT_OK_AND_ASSIGN(ModuleSignature sig,
                             GenerateSignature(CodegenOptions()
                                                   .module_name("foobar")
                                                   .clock_name("the_clock")
                                                   .flop_inputs(true),
                                               f, schedule));
    ASSERT_TRUE(sig.proto().has_pipeline());
    EXPECT_EQ(sig.proto().pipeline().latency(), 4);
  }

  {
    // Adding flopping of the outputs should increase latency by one.
    XLS_ASSERT_OK_AND_ASSIGN(ModuleSignature sig,
                             GenerateSignature(CodegenOptions()
                                                   .module_name("foobar")
                                                   .clock_name("the_clock")
                                                   .flop_outputs(true),
                                               f, schedule));
    ASSERT_TRUE(sig.proto().has_pipeline());
    EXPECT_EQ(sig.proto().pipeline().latency(), 4);
  }

  {
    // Adding flopping of both inputs and outputs should increase latency by
    // two.
    XLS_ASSERT_OK_AND_ASSIGN(ModuleSignature sig,
                             GenerateSignature(CodegenOptions()
                                                   .module_name("foobar")
                                                   .clock_name("the_clock")
                                                   .flop_inputs(true)
                                                   .flop_outputs(true),
                                               f, schedule));
    ASSERT_TRUE(sig.proto().has_pipeline());
    EXPECT_EQ(sig.proto().pipeline().latency(), 5);
  }
}

}  // namespace
}  // namespace verilog
}  // namespace xls
