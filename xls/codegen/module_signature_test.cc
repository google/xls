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

#include "xls/codegen/module_signature.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "xls/common/status/matchers.h"
#include "xls/ir/bits.h"
#include "xls/ir/value.h"

namespace xls {
namespace verilog {
namespace {

using status_testing::StatusIs;
using ::testing::HasSubstr;

std::string TestName() {
  return ::testing::UnitTest::GetInstance()->current_test_info()->name();
}

TEST(ModuleSignatureTest, SimpledFixedLatencyInterface) {
  ModuleSignatureBuilder b(TestName());

  b.AddDataInput("x", 42).AddDataOutput("y", 2).WithFixedLatencyInterface(123);

  XLS_ASSERT_OK_AND_ASSIGN(ModuleSignature signature, b.Build());
  ASSERT_EQ(signature.data_inputs().size(), 1);
  EXPECT_EQ(signature.data_inputs().front().width(), 42);
  EXPECT_EQ(signature.data_inputs().front().name(), "x");
  EXPECT_EQ(signature.TotalDataInputBits(), 42);

  ASSERT_EQ(signature.data_outputs().size(), 1);
  EXPECT_EQ(signature.data_outputs().front().width(), 2);
  EXPECT_EQ(signature.data_outputs().front().name(), "y");
  EXPECT_EQ(signature.TotalDataOutputBits(), 2);

  ASSERT_TRUE(signature.proto().has_fixed_latency());
  EXPECT_EQ(signature.proto().fixed_latency().latency(), 123);
}

TEST(ModuleSignatureTest, ReadyValidInterface) {
  ModuleSignatureBuilder b(TestName());

  b.WithReadyValidInterface("input_rdy", "input_vld", "output_rdy",
                            "output_vld")
      .WithClock("the_clk")
      .WithReset("reset_me", /*asynchronous=*/true, /*active_low=*/false)
      .AddDataInput("x", 42)
      .AddDataInput("y", 2)
      .AddDataInput("z", 44444)
      .AddDataOutput("o1", 1)
      .AddDataOutput("o2", 3);

  XLS_ASSERT_OK_AND_ASSIGN(ModuleSignature signature, b.Build());
  ASSERT_TRUE(signature.proto().has_ready_valid());
  EXPECT_EQ(signature.proto().ready_valid().input_ready(), "input_rdy");
  EXPECT_EQ(signature.proto().ready_valid().input_valid(), "input_vld");
  EXPECT_EQ(signature.proto().ready_valid().output_ready(), "output_rdy");
  EXPECT_EQ(signature.proto().ready_valid().output_valid(), "output_vld");

  EXPECT_EQ(signature.TotalDataInputBits(), 44488);
  EXPECT_EQ(signature.TotalDataOutputBits(), 4);

  EXPECT_EQ(signature.proto().clock_name(), "the_clk");
  EXPECT_TRUE(signature.proto().has_reset());
  EXPECT_EQ(signature.proto().reset().name(), "reset_me");
  EXPECT_TRUE(signature.proto().reset().asynchronous());
  EXPECT_FALSE(signature.proto().reset().active_low());

  EXPECT_EQ(signature.data_inputs().size(), 3);
  EXPECT_EQ(signature.data_outputs().size(), 2);
}

TEST(ModuleSignatureTest, PipelineInterface) {
  ModuleSignatureBuilder b(TestName());

  b.WithPipelineInterface(/*latency=*/2, /*initiation_interval=*/3)
      .WithClock("clk")
      .AddDataInput("in", 4)
      .AddDataOutput("out", 5);

  XLS_ASSERT_OK_AND_ASSIGN(ModuleSignature signature, b.Build());
  ASSERT_TRUE(signature.proto().has_pipeline());
  EXPECT_EQ(signature.proto().pipeline().latency(), 2);
  EXPECT_EQ(signature.proto().pipeline().initiation_interval(), 3);
}

TEST(ModuleSignatureTest, PipelineInterfaceMissingClock) {
  ModuleSignatureBuilder b(TestName());

  b.WithPipelineInterface(/*latency=*/2, /*initiation_interval=*/3)
      .AddDataInput("in", 4)
      .AddDataOutput("out", 5);

  EXPECT_THAT(b.Build(), StatusIs(absl::StatusCode::kInvalidArgument,
                                  HasSubstr("Missing clock")));
}

TEST(ModuleSignatureTest, ToKwargs) {
  ModuleSignatureBuilder b(TestName());
  b.AddDataInput("x", 42)
      .AddDataInput("y", 2)
      .AddDataOutput("z", 32)
      .WithFixedLatencyInterface(123);
  XLS_ASSERT_OK_AND_ASSIGN(ModuleSignature signature, b.Build());

  absl::flat_hash_map<std::string, Value> kwargs;
  XLS_ASSERT_OK_AND_ASSIGN(
      kwargs, signature.ToKwargs({Value(UBits(7, 42)), Value(UBits(0, 2))}));
  EXPECT_THAT(kwargs, testing::UnorderedElementsAre(
                          testing::Pair("x", Value(UBits(7, 42))),
                          testing::Pair("y", Value(UBits(0, 2)))));
}

}  // namespace
}  // namespace verilog
}  // namespace xls
