// Copyright 2020 Google LLC
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

#include "xls/codegen/sequential_generator.h"

#include <cstdio>
#include <memory>
#include <optional>
#include <string>
#include <type_traits>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "xls/codegen/module_signature.h"
#include "xls/codegen/pipeline_generator.h"
#include "xls/common/status/matchers.h"
#include "xls/common/status/status_macros.h"
#include "xls/common/status/statusor.h"
#include "xls/delay_model/delay_estimator.h"
#include "xls/ir/bits.h"
#include "xls/ir/function.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/ir_parser.h"
#include "xls/ir/nodes.h"
#include "xls/ir/package.h"
#include "xls/ir/value.h"
#include "xls/scheduling/pipeline_schedule.h"
#include "xls/simulation/module_simulator.h"
#include "xls/simulation/module_testbench.h"
#include "xls/simulation/verilog_test_base.h"

namespace xls {
namespace verilog {
namespace {

using status_testing::IsOkAndHolds;

constexpr char kTestName[] = "sequential_generator_test";
constexpr char kTestdataPath[] = "xls/codegen/testdata";

class TestDelayEstimator : public DelayEstimator {
 public:
  xabsl::StatusOr<int64> GetOperationDelayInPs(Node* node) const override {
    switch (node->op()) {
      case Op::kParam:
      case Op::kLiteral:
      case Op::kBitSlice:
      case Op::kConcat:
        return 0;
      default:
        return 1;
    }
  }
};

class SequentialGeneratorTest : public VerilogTestBase {};

TEST_P(SequentialGeneratorTest, LoopBodyPipelineTest) {
  std::string text = R"(
package LoopBodyPipelineTest

fn ____LoopBodyPipelineTest__main_counted_for_0_body(index: bits[32], acc: bits[32]) -> bits[32] {
  add.5: bits[32] = add(acc, index, pos=0,2,8)
  literal.6: bits[32] = literal(value=3, pos=0,2,22)
  add.7: bits[32] = add(add.5, literal.6, pos=0,2,16)
  literal.8: bits[32] = literal(value=4, pos=0,2,30)
  ret add.9: bits[32] = add(add.7, literal.8, pos=0,2,24)
}

fn __LoopBodyPipelineTest__main() -> bits[32] {
  literal.1: bits[32] = literal(value=0, pos=0,3,8)
  literal.2: bits[32] = literal(value=4, pos=0,1,51)
  ret counted_for.10: bits[32] = counted_for(literal.1, trip_count=4, stride=1, body=____LoopBodyPipelineTest__main_counted_for_0_body, pos=0,1,5)
}

)";
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> package,
                           Parser::ParsePackage(text));
  XLS_ASSERT_OK_AND_ASSIGN(Function * main, package->EntryFunction());

  // Grab loop node.
  CountedFor* loop = nullptr;
  for (auto* node : main->nodes()) {
    if (node->Is<CountedFor>()) {
      loop = node->As<CountedFor>();
    }
  }
  EXPECT_NE(loop, nullptr);

  // Generate pipeline for loop body.
  xabsl::StatusOr<std::unique_ptr<ModuleGeneratorResult>> loop_body_status =
      GenerateLoopBodyPipeline(loop, UseSystemVerilog(),
                               SchedulingOptions().pipeline_stages(3),
                               TestDelayEstimator());
  EXPECT_TRUE(loop_body_status.ok());
  std::unique_ptr<ModuleGeneratorResult> loop_body =
      std::move(loop_body_status.value());
  ExpectVerilogEqualToGoldenFile(GoldenFilePath(kTestName, kTestdataPath),
                                 loop_body->verilog_text);
  EXPECT_TRUE(loop_body->signature.proto().has_pipeline());
  EXPECT_EQ(loop_body->signature.proto().pipeline().initiation_interval(), 1);
  EXPECT_EQ(loop_body->signature.proto().pipeline().latency(), 2);

  // Check functionality.
  ModuleSimulator simulator(loop_body->signature, loop_body->verilog_text,
                            GetSimulator());
  EXPECT_THAT(simulator.Run({{"acc", Value(UBits(10, 32))},
                             {"index", Value(UBits(1, 32))}}),
              IsOkAndHolds(Value(UBits(18, 32))));
}

INSTANTIATE_TEST_SUITE_P(SequentialGeneratorTestInstantiation,
                         SequentialGeneratorTest,
                         testing::ValuesIn(kDefaultSimulationTargets),
                         ParameterizedTestName<SequentialGeneratorTest>);

}  // namespace
}  // namespace verilog
}  // namespace xls
