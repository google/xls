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
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "xls/codegen/module_builder.h"
#include "xls/codegen/module_signature.h"
#include "xls/codegen/module_signature.pb.h"
#include "xls/codegen/pipeline_generator.h"
#include "xls/codegen/vast.h"
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
  XLS_ASSERT_OK_AND_ASSIGN(Node * node_loop, main->GetNode("counted_for.10"));
  CountedFor* loop = node_loop->As<CountedFor>();

  // Build the builder.
  SequentialModuleBuilder builder(
      SequentialOptions().use_system_verilog(UseSystemVerilog()), loop);

  // Generate pipeline for loop body.
  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<ModuleGeneratorResult> loop_body,
      builder.GenerateLoopBodyPipeline(SchedulingOptions().pipeline_stages(3),
                                       TestDelayEstimator()));
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

// TODO(jbaileyhandle): Test reset signature for pipeline.

TEST_P(SequentialGeneratorTest, ModuleSignatureTestSimple) {
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
  XLS_ASSERT_OK_AND_ASSIGN(Node * node_loop, main->GetNode("counted_for.10"));
  CountedFor* loop = node_loop->As<CountedFor>();

  // Build the builder.
  SequentialOptions sequential_options;
  SequentialModuleBuilder builder(sequential_options, loop);

  // Generate module signature.
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<ModuleSignature> signature,
                           builder.GenerateModuleSignature());

  // Generate expected signature.
  ModuleSignatureBuilder oracle_builder("counted_for_10_sequential_module");
  oracle_builder.AddDataInput("literal_1_in", 32);
  oracle_builder.AddDataOutput("counted_for_10_out", 32);
  oracle_builder.WithClock("clk");
  oracle_builder.WithReadyValidInterface("ready_in", "valid_in", "ready_out",
                                         "valid_out");
  XLS_ASSERT_OK_AND_ASSIGN(ModuleSignature expected, oracle_builder.Build());

  EXPECT_EQ(signature->proto().DebugString(), expected.proto().DebugString());
}

TEST_P(SequentialGeneratorTest, ModuleSignatureTestCustomModuleName) {
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
  XLS_ASSERT_OK_AND_ASSIGN(Node * node_loop, main->GetNode("counted_for.10"));
  CountedFor* loop = node_loop->As<CountedFor>();

  // Build the builder.
  SequentialOptions sequential_options;
  sequential_options.module_name("foobar");
  SequentialModuleBuilder builder(sequential_options, loop);

  // Generate module signature.
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<ModuleSignature> signature,
                           builder.GenerateModuleSignature());

  // Generate expected signature.
  ModuleSignatureBuilder oracle_builder("foobar");
  oracle_builder.AddDataInput("literal_1_in", 32);
  oracle_builder.AddDataOutput("counted_for_10_out", 32);
  oracle_builder.WithClock("clk");
  oracle_builder.WithReadyValidInterface("ready_in", "valid_in", "ready_out",
                                         "valid_out");
  XLS_ASSERT_OK_AND_ASSIGN(ModuleSignature expected, oracle_builder.Build());

  EXPECT_EQ(signature->proto().DebugString(), expected.proto().DebugString());
}

TEST_P(SequentialGeneratorTest, ModuleSignatureTestInvariants) {
  // CountedFor with 2 invariants plus an extra node defined outside the loop
  // but not consumed in the loop (should be ignored).
  std::string text = R"(
package ModuleSignatureTestInvariants

fn ____ModuleSignatureTestInvariants__main_counted_for_0_body(index: bits[32], acc: bits[32], invar_a: bits[32], invar_b: bits[32]) -> bits[32] {
  add.10: bits[32] = add(acc, index, pos=0,5,8)
  add.11: bits[32] = add(add.10, invar_a, pos=0,5,16)
  ret add.12: bits[32] = add(add.11, invar_b, pos=0,5,26)
}

fn __ModuleSignatureTestInvariants__main() -> bits[32] {
  literal.4: bits[32] = literal(value=0, pos=0,6,8)
  literal.1: bits[32] = literal(value=3, pos=0,1,26)
  literal.2: bits[32] = literal(value=4, pos=0,2,26)
  literal.3: bits[32] = literal(value=5, pos=0,3,30)
  literal.5: bits[32] = literal(value=4, pos=0,4,51)
  ret counted_for.13: bits[32] = counted_for(literal.4, trip_count=4, stride=1, body=____ModuleSignatureTestInvariants__main_counted_for_0_body, invariant_args=[literal.1, literal.2], pos=0,4,5)
}
)";
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> package,
                           Parser::ParsePackage(text));
  XLS_ASSERT_OK_AND_ASSIGN(Function * main, package->EntryFunction());

  // Grab loop node.
  XLS_ASSERT_OK_AND_ASSIGN(Node * node_loop, main->GetNode("counted_for.13"));
  CountedFor* loop = node_loop->As<CountedFor>();

  // Build the builder.
  SequentialOptions sequential_options;
  SequentialModuleBuilder builder(sequential_options, loop);

  // Generate module signature.
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<ModuleSignature> signature,
                           builder.GenerateModuleSignature());

  // Generate expected signature.
  ModuleSignatureBuilder oracle_builder("counted_for_13_sequential_module");
  oracle_builder.AddDataInput("literal_4_in", 32);
  oracle_builder.AddDataInput("literal_1_in", 32);
  oracle_builder.AddDataInput("literal_2_in", 32);
  oracle_builder.AddDataOutput("counted_for_13_out", 32);
  oracle_builder.WithClock("clk");
  oracle_builder.WithReadyValidInterface("ready_in", "valid_in", "ready_out",
                                         "valid_out");
  XLS_ASSERT_OK_AND_ASSIGN(ModuleSignature expected, oracle_builder.Build());

  EXPECT_EQ(signature->proto().DebugString(), expected.proto().DebugString());
}

TEST_P(SequentialGeneratorTest, ModuleSignatureSynchronousResetActiveHigh) {
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
  XLS_ASSERT_OK_AND_ASSIGN(Node * node_loop, main->GetNode("counted_for.10"));
  CountedFor* loop = node_loop->As<CountedFor>();

  // Build the builder.
  SequentialOptions sequential_options;
  ResetProto reset;
  reset.set_name("reset_me_A");
  reset.set_asynchronous(true);
  reset.set_active_low(false);
  sequential_options.reset(reset);
  SequentialModuleBuilder builder(sequential_options, loop);

  // Generate module signature.
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<ModuleSignature> signature,
                           builder.GenerateModuleSignature());

  // Generate expected signature.
  ModuleSignatureBuilder oracle_builder("counted_for_10_sequential_module");
  oracle_builder.AddDataInput("literal_1_in", 32);
  oracle_builder.AddDataOutput("counted_for_10_out", 32);
  oracle_builder.WithClock("clk");
  oracle_builder.WithReadyValidInterface("ready_in", "valid_in", "ready_out",
                                         "valid_out");
  oracle_builder.WithReset(reset.name(), reset.asynchronous(),
                           reset.active_low());
  XLS_ASSERT_OK_AND_ASSIGN(ModuleSignature expected, oracle_builder.Build());

  EXPECT_EQ(signature->proto().DebugString(), expected.proto().DebugString());
}

TEST_P(SequentialGeneratorTest, ModuleSignatureAsynchronousActiveLowReset) {
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
  XLS_ASSERT_OK_AND_ASSIGN(Node * node_loop, main->GetNode("counted_for.10"));
  CountedFor* loop = node_loop->As<CountedFor>();

  // Build the builder.
  SequentialOptions sequential_options;
  ResetProto reset;
  reset.set_name("reset_me_B");
  reset.set_asynchronous(false);
  reset.set_active_low(true);
  sequential_options.reset(reset);
  SequentialModuleBuilder builder(sequential_options, loop);

  // Generate module signature.
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<ModuleSignature> signature,
                           builder.GenerateModuleSignature());

  // Generate expected signature.
  ModuleSignatureBuilder oracle_builder("counted_for_10_sequential_module");
  oracle_builder.AddDataInput("literal_1_in", 32);
  oracle_builder.AddDataOutput("counted_for_10_out", 32);
  oracle_builder.WithClock("clk");
  oracle_builder.WithReadyValidInterface("ready_in", "valid_in", "ready_out",
                                         "valid_out");
  oracle_builder.WithReset(reset.name(), reset.asynchronous(),
                           reset.active_low());
  XLS_ASSERT_OK_AND_ASSIGN(ModuleSignature expected, oracle_builder.Build());

  EXPECT_EQ(signature->proto().DebugString(), expected.proto().DebugString());
}

TEST_P(SequentialGeneratorTest, ModuleHeaderTestSimple) {
  // Generate signature.
  ModuleSignatureBuilder signature_builder("counted_for_10_sequential_module");
  signature_builder.AddDataInput("literal_1_in", 32);
  signature_builder.AddDataOutput("counted_for_10_out", 32);
  signature_builder.WithClock("clk");
  signature_builder.WithReadyValidInterface("ready_in", "valid_in", "ready_out",
                                            "valid_out");
  XLS_ASSERT_OK_AND_ASSIGN(ModuleSignature signature,
                           signature_builder.Build());

  // Generate module with header.
  SequentialOptions sequential_options;
  sequential_options.use_system_verilog(UseSystemVerilog());
  SequentialModuleBuilder builder(sequential_options, nullptr);
  XLS_ASSERT_OK(builder.InitializeModuleBuilder(signature));

  auto protos_to_strings = [](absl::Span<const PortProto> ports) {
    absl::flat_hash_set<std::string> result;
    for (auto& port : ports) {
      result.insert(port.DebugString());
    }
    return result;
  };

  auto ports_to_strings = [](absl::Span<const Port> ports) {
    absl::flat_hash_set<std::string> result;
    for (auto& port : ports) {
      result.insert(port.ToProto()->DebugString());
    }
    return result;
  };

  std::vector<PortProto> expected_protos;
  expected_protos.push_back(PortProto());
  expected_protos.back().set_name("clk");
  expected_protos.back().set_width(1);
  expected_protos.back().set_direction(DIRECTION_INPUT);
  expected_protos.push_back(PortProto());
  expected_protos.back().set_name("ready_in");
  expected_protos.back().set_width(1);
  expected_protos.back().set_direction(DIRECTION_OUTPUT);
  expected_protos.push_back(PortProto());
  expected_protos.back().set_name("valid_in");
  expected_protos.back().set_width(1);
  expected_protos.back().set_direction(DIRECTION_INPUT);
  expected_protos.push_back(PortProto());
  expected_protos.back().set_name("ready_out");
  expected_protos.back().set_width(1);
  expected_protos.back().set_direction(DIRECTION_INPUT);
  expected_protos.push_back(PortProto());
  expected_protos.back().set_name("valid_out");
  expected_protos.back().set_width(1);
  expected_protos.back().set_direction(DIRECTION_OUTPUT);
  expected_protos.push_back(PortProto());
  expected_protos.back().set_name("literal_1_in");
  expected_protos.back().set_width(32);
  expected_protos.back().set_direction(DIRECTION_INPUT);
  expected_protos.push_back(PortProto());
  expected_protos.back().set_name("counted_for_10_out");
  expected_protos.back().set_width(32);
  expected_protos.back().set_direction(DIRECTION_OUTPUT);

  EXPECT_THAT(ports_to_strings(builder.module()->ports()),
              testing::ContainerEq(protos_to_strings(expected_protos)));
  EXPECT_EQ(builder.module()->name(), "counted_for_10_sequential_module");
}

TEST_P(SequentialGeneratorTest, ModuleHeaderTestCustomModuleName) {
  // Generate signature.
  ModuleSignatureBuilder signature_builder("foobar");
  signature_builder.AddDataInput("literal_1_in", 32);
  signature_builder.AddDataOutput("counted_for_10_out", 32);
  signature_builder.WithClock("clk");
  signature_builder.WithReadyValidInterface("ready_in", "valid_in", "ready_out",
                                            "valid_out");
  XLS_ASSERT_OK_AND_ASSIGN(ModuleSignature signature,
                           signature_builder.Build());

  // Generate module with header.
  SequentialOptions sequential_options;
  sequential_options.use_system_verilog(UseSystemVerilog());
  SequentialModuleBuilder builder(sequential_options, nullptr);
  XLS_ASSERT_OK(builder.InitializeModuleBuilder(signature));

  auto protos_to_strings = [](absl::Span<const PortProto> ports) {
    absl::flat_hash_set<std::string> result;
    for (auto& port : ports) {
      result.insert(port.DebugString());
    }
    return result;
  };

  auto ports_to_strings = [](absl::Span<const Port> ports) {
    absl::flat_hash_set<std::string> result;
    for (auto& port : ports) {
      result.insert(port.ToProto()->DebugString());
    }
    return result;
  };

  std::vector<PortProto> expected_protos;
  expected_protos.push_back(PortProto());
  expected_protos.back().set_name("clk");
  expected_protos.back().set_width(1);
  expected_protos.back().set_direction(DIRECTION_INPUT);
  expected_protos.push_back(PortProto());
  expected_protos.back().set_name("ready_in");
  expected_protos.back().set_width(1);
  expected_protos.back().set_direction(DIRECTION_OUTPUT);
  expected_protos.push_back(PortProto());
  expected_protos.back().set_name("valid_in");
  expected_protos.back().set_width(1);
  expected_protos.back().set_direction(DIRECTION_INPUT);
  expected_protos.push_back(PortProto());
  expected_protos.back().set_name("ready_out");
  expected_protos.back().set_width(1);
  expected_protos.back().set_direction(DIRECTION_INPUT);
  expected_protos.push_back(PortProto());
  expected_protos.back().set_name("valid_out");
  expected_protos.back().set_width(1);
  expected_protos.back().set_direction(DIRECTION_OUTPUT);
  expected_protos.push_back(PortProto());
  expected_protos.back().set_name("literal_1_in");
  expected_protos.back().set_width(32);
  expected_protos.back().set_direction(DIRECTION_INPUT);
  expected_protos.push_back(PortProto());
  expected_protos.back().set_name("counted_for_10_out");
  expected_protos.back().set_width(32);
  expected_protos.back().set_direction(DIRECTION_OUTPUT);

  EXPECT_THAT(ports_to_strings(builder.module()->ports()),
              testing::ContainerEq(protos_to_strings(expected_protos)));
  EXPECT_EQ(builder.module()->name(), "foobar");
}

TEST_P(SequentialGeneratorTest, ModuleHeaderTestReset) {
  // Generate signature.
  ModuleSignatureBuilder signature_builder("counted_for_10_sequential_module");
  signature_builder.AddDataInput("literal_1_in", 32);
  signature_builder.AddDataOutput("counted_for_10_out", 32);
  signature_builder.WithClock("clk");
  signature_builder.WithReadyValidInterface("ready_in", "valid_in", "ready_out",
                                            "valid_out");
  signature_builder.WithReset("rst", false, false);
  XLS_ASSERT_OK_AND_ASSIGN(ModuleSignature signature,
                           signature_builder.Build());

  // Generate module with header.
  SequentialOptions sequential_options;
  sequential_options.use_system_verilog(UseSystemVerilog());
  SequentialModuleBuilder builder(sequential_options, nullptr);
  XLS_ASSERT_OK(builder.InitializeModuleBuilder(signature));

  auto protos_to_strings = [](absl::Span<const PortProto> ports) {
    absl::flat_hash_set<std::string> result;
    for (auto& port : ports) {
      result.insert(port.DebugString());
    }
    return result;
  };

  auto ports_to_strings = [](absl::Span<const Port> ports) {
    absl::flat_hash_set<std::string> result;
    for (auto& port : ports) {
      result.insert(port.ToProto()->DebugString());
    }
    return result;
  };

  std::vector<PortProto> expected_protos;
  expected_protos.push_back(PortProto());
  expected_protos.back().set_name("rst");
  expected_protos.back().set_width(1);
  expected_protos.back().set_direction(DIRECTION_INPUT);
  expected_protos.push_back(PortProto());
  expected_protos.back().set_name("clk");
  expected_protos.back().set_width(1);
  expected_protos.back().set_direction(DIRECTION_INPUT);
  expected_protos.push_back(PortProto());
  expected_protos.back().set_name("ready_in");
  expected_protos.back().set_width(1);
  expected_protos.back().set_direction(DIRECTION_OUTPUT);
  expected_protos.push_back(PortProto());
  expected_protos.back().set_name("valid_in");
  expected_protos.back().set_width(1);
  expected_protos.back().set_direction(DIRECTION_INPUT);
  expected_protos.push_back(PortProto());
  expected_protos.back().set_name("ready_out");
  expected_protos.back().set_width(1);
  expected_protos.back().set_direction(DIRECTION_INPUT);
  expected_protos.push_back(PortProto());
  expected_protos.back().set_name("valid_out");
  expected_protos.back().set_width(1);
  expected_protos.back().set_direction(DIRECTION_OUTPUT);
  expected_protos.push_back(PortProto());
  expected_protos.back().set_name("literal_1_in");
  expected_protos.back().set_width(32);
  expected_protos.back().set_direction(DIRECTION_INPUT);
  expected_protos.push_back(PortProto());
  expected_protos.back().set_name("counted_for_10_out");
  expected_protos.back().set_width(32);
  expected_protos.back().set_direction(DIRECTION_OUTPUT);

  EXPECT_THAT(ports_to_strings(builder.module()->ports()),
              testing::ContainerEq(protos_to_strings(expected_protos)));
  EXPECT_EQ(builder.module()->name(), "counted_for_10_sequential_module");
}

TEST_P(SequentialGeneratorTest, ModuleHeaderTestInvariants) {
  // Generate signature.
  ModuleSignatureBuilder signature_builder("counted_for_10_sequential_module");
  signature_builder.AddDataInput("literal_4_in", 32);
  signature_builder.AddDataInput("literal_1_in", 16);
  signature_builder.AddDataInput("literal_2_in", 8);
  signature_builder.AddDataOutput("counted_for_10_out", 32);
  signature_builder.WithClock("clk");
  signature_builder.WithReadyValidInterface("ready_in", "valid_in", "ready_out",
                                            "valid_out");
  XLS_ASSERT_OK_AND_ASSIGN(ModuleSignature signature,
                           signature_builder.Build());

  // Generate module with header.
  SequentialOptions sequential_options;
  sequential_options.use_system_verilog(UseSystemVerilog());
  SequentialModuleBuilder builder(sequential_options, nullptr);
  XLS_ASSERT_OK(builder.InitializeModuleBuilder(signature));

  auto protos_to_strings = [](absl::Span<const PortProto> ports) {
    absl::flat_hash_set<std::string> result;
    for (auto& port : ports) {
      result.insert(port.DebugString());
    }
    return result;
  };

  auto ports_to_strings = [](absl::Span<const Port> ports) {
    absl::flat_hash_set<std::string> result;
    for (auto& port : ports) {
      result.insert(port.ToProto()->DebugString());
    }
    return result;
  };

  std::vector<PortProto> expected_protos;
  expected_protos.push_back(PortProto());
  expected_protos.back().set_name("rst");
  expected_protos.back().set_width(1);
  expected_protos.back().set_direction(DIRECTION_INPUT);
  expected_protos.back().set_name("clk");
  expected_protos.back().set_width(1);
  expected_protos.back().set_direction(DIRECTION_INPUT);
  expected_protos.push_back(PortProto());
  expected_protos.back().set_name("ready_in");
  expected_protos.back().set_width(1);
  expected_protos.back().set_direction(DIRECTION_OUTPUT);
  expected_protos.push_back(PortProto());
  expected_protos.back().set_name("valid_in");
  expected_protos.back().set_width(1);
  expected_protos.back().set_direction(DIRECTION_INPUT);
  expected_protos.push_back(PortProto());
  expected_protos.back().set_name("ready_out");
  expected_protos.back().set_width(1);
  expected_protos.back().set_direction(DIRECTION_INPUT);
  expected_protos.push_back(PortProto());
  expected_protos.back().set_name("valid_out");
  expected_protos.back().set_width(1);
  expected_protos.back().set_direction(DIRECTION_OUTPUT);
  expected_protos.push_back(PortProto());
  expected_protos.back().set_name("literal_4_in");
  expected_protos.back().set_width(32);
  expected_protos.back().set_direction(DIRECTION_INPUT);
  expected_protos.push_back(PortProto());
  expected_protos.back().set_name("literal_1_in");
  expected_protos.back().set_width(16);
  expected_protos.back().set_direction(DIRECTION_INPUT);
  expected_protos.push_back(PortProto());
  expected_protos.back().set_name("literal_2_in");
  expected_protos.back().set_width(8);
  expected_protos.back().set_direction(DIRECTION_INPUT);
  expected_protos.push_back(PortProto());
  expected_protos.back().set_name("counted_for_10_out");
  expected_protos.back().set_width(32);
  expected_protos.back().set_direction(DIRECTION_OUTPUT);

  EXPECT_THAT(ports_to_strings(builder.module()->ports()),
              testing::ContainerEq(protos_to_strings(expected_protos)));
  EXPECT_EQ(builder.module()->name(), "counted_for_10_sequential_module");
}

// TODO(jbaileyhandle): Test module reset (active high and active low).

INSTANTIATE_TEST_SUITE_P(SequentialGeneratorTestInstantiation,
                         SequentialGeneratorTest,
                         testing::ValuesIn(kDefaultSimulationTargets),
                         ParameterizedTestName<SequentialGeneratorTest>);

}  // namespace
}  // namespace verilog
}  // namespace xls
