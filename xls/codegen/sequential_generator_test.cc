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
#include "absl/status/statusor.h"
#include "xls/codegen/module_builder.h"
#include "xls/codegen/module_signature.h"
#include "xls/codegen/module_signature.pb.h"
#include "xls/codegen/pipeline_generator.h"
#include "xls/codegen/vast.h"
#include "xls/common/status/matchers.h"
#include "xls/common/status/status_macros.h"
#include "xls/delay_model/delay_estimator.h"
#include "xls/ir/bits.h"
#include "xls/ir/function.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/ir_parser.h"
#include "xls/ir/nodes.h"
#include "xls/ir/package.h"
#include "xls/ir/type.h"
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
  absl::StatusOr<int64_t> GetOperationDelayInPs(Node* node) const override {
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
class PipelinedSequentialGeneratorTest : public SequentialGeneratorTest {};

TEST_P(SequentialGeneratorTest, LoopBodyPipelineTest) {
  std::string text = R"(
package LoopBodyPipelineTest

fn ____LoopBodyPipelineTest__main_counted_for_0_body(index: bits[32], acc: bits[32]) -> bits[32] {
  add.5: bits[32] = add(acc, index, pos=[(0,2,8)])
  literal.6: bits[32] = literal(value=3, pos=[(0,2,22)])
  add.7: bits[32] = add(add.5, literal.6, pos=[(0,2,16)])
  literal.8: bits[32] = literal(value=4, pos=[(0,2,30)])
  ret add.9: bits[32] = add(add.7, literal.8, pos=[(0,2,24)])
}

top fn __LoopBodyPipelineTest__main() -> bits[32] {
  literal.1: bits[32] = literal(value=0, pos=[(0,3,8)])
  literal.2: bits[32] = literal(value=4, pos=[(0,1,51)])
  ret counted_for.10: bits[32] = counted_for(literal.1, trip_count=4, stride=1, body=____LoopBodyPipelineTest__main_counted_for_0_body, pos=[(0,1,5)])
}

)";
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> package,
                           Parser::ParsePackage(text));
  std::optional<FunctionBase*> main = package->GetTop();
  ASSERT_TRUE(main.has_value());

  // Grab loop node.
  XLS_ASSERT_OK_AND_ASSIGN(Node * node_loop,
                           main.value()->GetNode("counted_for.10"));
  CountedFor* loop = node_loop->As<CountedFor>();

  // Build the builder.
  SequentialOptions sequential_options;
  sequential_options.use_system_verilog(UseSystemVerilog());
  sequential_options.pipeline_scheduling_options(
      SchedulingOptions().pipeline_stages(3));
  SequentialModuleBuilder builder(sequential_options, loop);

  // Generate pipeline for loop body.
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<ModuleGeneratorResult> loop_body,
                           builder.GenerateLoopBodyPipeline());
  ExpectVerilogEqualToGoldenFile(GoldenFilePath(kTestName, kTestdataPath),
                                 loop_body->verilog_text);
  EXPECT_TRUE(loop_body->signature.proto().has_pipeline());
  EXPECT_EQ(loop_body->signature.proto().pipeline().initiation_interval(), 1);
  EXPECT_EQ(loop_body->signature.proto().pipeline().latency(), 2);

  // Check functionality.
  ModuleSimulator simulator =
      NewModuleSimulator(loop_body->verilog_text, loop_body->signature);
  EXPECT_THAT(simulator.Run({{"acc", Value(UBits(10, 32))},
                             {"index", Value(UBits(1, 32))}}),
              IsOkAndHolds(Value(UBits(18, 32))));
}

// TODO(jbaileyhandle): Test reset signature for pipeline.

TEST_P(SequentialGeneratorTest, ModuleSignatureTestSimple) {
  std::string text = R"(
package LoopBodyPipelineTest

fn ____LoopBodyPipelineTest__main_counted_for_0_body(index: bits[32], acc: bits[32]) -> bits[32] {
  add.5: bits[32] = add(acc, index, pos=[(0,2,8)])
  literal.6: bits[32] = literal(value=3, pos=[(0,2,22)])
  add.7: bits[32] = add(add.5, literal.6, pos=[(0,2,16)])
  literal.8: bits[32] = literal(value=4, pos=[(0,2,30)])
  ret add.9: bits[32] = add(add.7, literal.8, pos=[(0,2,24)])
}

top fn __LoopBodyPipelineTest__main() -> bits[32] {
  literal.1: bits[32] = literal(value=0, pos=[(0,3,8)])
  literal.2: bits[32] = literal(value=4, pos=[(0,1,51)])
  ret counted_for.10: bits[32] = counted_for(literal.1, trip_count=4, stride=1, body=____LoopBodyPipelineTest__main_counted_for_0_body, pos=[(0,1,5)])
}

)";
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> package,
                           Parser::ParsePackage(text));
  std::optional<FunctionBase*> main = package->GetTop();
  ASSERT_TRUE(main.has_value());

  // Grab loop node.
  XLS_ASSERT_OK_AND_ASSIGN(Node * node_loop,
                           main.value()->GetNode("counted_for.10"));
  CountedFor* loop = node_loop->As<CountedFor>();

  // Build the builder.
  SequentialOptions sequential_options;
  SequentialModuleBuilder builder(sequential_options, loop);

  // Generate module signature.
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<ModuleSignature> signature,
                           builder.GenerateModuleSignature());

  // Generate expected signature.
  // Set function type.
  ModuleSignatureBuilder oracle_builder("counted_for_10_sequential_module");
  oracle_builder.AddDataInput("literal_1_in", 32);
  oracle_builder.AddDataOutput("counted_for_10_out", 32);
  oracle_builder.WithClock("clk");
  oracle_builder.WithReadyValidInterface("ready_in", "valid_in", "ready_out",
                                         "valid_out");
  auto bit_type = [&package](int64_t num_bits) {
    return package->GetBitsType(num_bits);
  };
  FunctionType func({bit_type(32)}, bit_type(32));
  oracle_builder.WithFunctionType(&func);
  XLS_ASSERT_OK_AND_ASSIGN(ModuleSignature expected, oracle_builder.Build());

  EXPECT_EQ(signature->proto().DebugString(), expected.proto().DebugString());
}

TEST_P(SequentialGeneratorTest, ModuleSignatureTestCustomModuleName) {
  std::string text = R"(
package LoopBodyPipelineTest

fn ____LoopBodyPipelineTest__main_counted_for_0_body(index: bits[32], acc: bits[32]) -> bits[32] {
  add.5: bits[32] = add(acc, index, pos=[(0,2,8)])
  literal.6: bits[32] = literal(value=3, pos=[(0,2,22)])
  add.7: bits[32] = add(add.5, literal.6, pos=[(0,2,16)])
  literal.8: bits[32] = literal(value=4, pos=[(0,2,30)])
  ret add.9: bits[32] = add(add.7, literal.8, pos=[(0,2,24)])
}

top fn __LoopBodyPipelineTest__main() -> bits[32] {
  literal.1: bits[32] = literal(value=0, pos=[(0,3,8)])
  literal.2: bits[32] = literal(value=4, pos=[(0,1,51)])
  ret counted_for.10: bits[32] = counted_for(literal.1, trip_count=4, stride=1, body=____LoopBodyPipelineTest__main_counted_for_0_body, pos=[(0,1,5)])
}

)";
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> package,
                           Parser::ParsePackage(text));
  std::optional<FunctionBase*> main = package->GetTop();
  ASSERT_TRUE(main.has_value());

  // Grab loop node.
  XLS_ASSERT_OK_AND_ASSIGN(Node * node_loop,
                           main.value()->GetNode("counted_for.10"));
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
  auto bit_type = [&package](int64_t num_bits) {
    return package->GetBitsType(num_bits);
  };
  FunctionType func({bit_type(32)}, bit_type(32));
  oracle_builder.WithFunctionType(&func);
  XLS_ASSERT_OK_AND_ASSIGN(ModuleSignature expected, oracle_builder.Build());

  EXPECT_EQ(signature->proto().DebugString(), expected.proto().DebugString());
}

TEST_P(SequentialGeneratorTest, ModuleSignatureTestInvariants) {
  // CountedFor with 2 invariants plus an extra node defined outside the loop
  // but not consumed in the loop (should be ignored).
  std::string text = R"(
package ModuleSignatureTestInvariants

fn ____ModuleSignatureTestInvariants__main_counted_for_0_body(index: bits[32], acc: bits[32], invar_a: bits[32], invar_b: bits[32]) -> bits[32] {
  add.10: bits[32] = add(acc, index, pos=[(0,5,8)])
  add.11: bits[32] = add(add.10, invar_a, pos=[(0,5,16)])
  ret add.12: bits[32] = add(add.11, invar_b, pos=[(0,5,26)])
}

top fn __ModuleSignatureTestInvariants__main() -> bits[32] {
  literal.4: bits[32] = literal(value=0, pos=[(0,6,8)])
  literal.1: bits[32] = literal(value=3, pos=[(0,1,26)])
  literal.2: bits[32] = literal(value=4, pos=[(0,2,26)])
  literal.3: bits[32] = literal(value=5, pos=[(0,3,30)])
  literal.5: bits[32] = literal(value=4, pos=[(0,4,51)])
  ret counted_for.13: bits[32] = counted_for(literal.4, trip_count=4, stride=1, body=____ModuleSignatureTestInvariants__main_counted_for_0_body, invariant_args=[literal.1, literal.2], pos=[(0,4,5)])
}
)";
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> package,
                           Parser::ParsePackage(text));
  std::optional<FunctionBase*> main = package->GetTop();
  ASSERT_TRUE(main.has_value());

  // Grab loop node.
  XLS_ASSERT_OK_AND_ASSIGN(Node * node_loop,
                           main.value()->GetNode("counted_for.13"));
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
  auto bit_type = [&package](int64_t num_bits) {
    return package->GetBitsType(num_bits);
  };
  FunctionType func({bit_type(32), bit_type(32), bit_type(32)}, bit_type(32));
  oracle_builder.WithFunctionType(&func);
  XLS_ASSERT_OK_AND_ASSIGN(ModuleSignature expected, oracle_builder.Build());

  EXPECT_EQ(signature->proto().DebugString(), expected.proto().DebugString());
}

TEST_P(SequentialGeneratorTest, ModuleSignatureSynchronousResetActiveHigh) {
  std::string text = R"(
package LoopBodyPipelineTest

fn ____LoopBodyPipelineTest__main_counted_for_0_body(index: bits[32], acc: bits[32]) -> bits[32] {
  add.5: bits[32] = add(acc, index, pos=[(0,2,8)])
  literal.6: bits[32] = literal(value=3, pos=[(0,2,22)])
  add.7: bits[32] = add(add.5, literal.6, pos=[(0,2,16)])
  literal.8: bits[32] = literal(value=4, pos=[(0,2,30)])
  ret add.9: bits[32] = add(add.7, literal.8, pos=[(0,2,24)])
}

top fn __LoopBodyPipelineTest__main() -> bits[32] {
  literal.1: bits[32] = literal(value=0, pos=[(0,3,8)])
  literal.2: bits[32] = literal(value=4, pos=[(0,1,51)])
  ret counted_for.10: bits[32] = counted_for(literal.1, trip_count=4, stride=1, body=____LoopBodyPipelineTest__main_counted_for_0_body, pos=[(0,1,5)])
}

)";
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> package,
                           Parser::ParsePackage(text));
  std::optional<FunctionBase*> main = package->GetTop();
  ASSERT_TRUE(main.has_value());

  // Grab loop node.
  XLS_ASSERT_OK_AND_ASSIGN(Node * node_loop,
                           main.value()->GetNode("counted_for.10"));
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
  auto bit_type = [&package](int64_t num_bits) {
    return package->GetBitsType(num_bits);
  };
  FunctionType func({bit_type(32)}, bit_type(32));
  oracle_builder.WithFunctionType(&func);
  XLS_ASSERT_OK_AND_ASSIGN(ModuleSignature expected, oracle_builder.Build());

  EXPECT_EQ(signature->proto().DebugString(), expected.proto().DebugString());
}

TEST_P(SequentialGeneratorTest, ModuleSignatureAsynchronousActiveLowReset) {
  std::string text = R"(
package LoopBodyPipelineTest

fn ____LoopBodyPipelineTest__main_counted_for_0_body(index: bits[32], acc: bits[32]) -> bits[32] {
  add.5: bits[32] = add(acc, index, pos=[(0,2,8)])
  literal.6: bits[32] = literal(value=3, pos=[(0,2,22)])
  add.7: bits[32] = add(add.5, literal.6, pos=[(0,2,16)])
  literal.8: bits[32] = literal(value=4, pos=[(0,2,30)])
  ret add.9: bits[32] = add(add.7, literal.8, pos=[(0,2,24)])
}

top fn __LoopBodyPipelineTest__main() -> bits[32] {
  literal.1: bits[32] = literal(value=0, pos=[(0,3,8)])
  literal.2: bits[32] = literal(value=4, pos=[(0,1,51)])
  ret counted_for.10: bits[32] = counted_for(literal.1, trip_count=4, stride=1, body=____LoopBodyPipelineTest__main_counted_for_0_body, pos=[(0,1,5)])
}

)";
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> package,
                           Parser::ParsePackage(text));
  std::optional<FunctionBase*> main = package->GetTop();
  ASSERT_TRUE(main.has_value());

  // Grab loop node.
  XLS_ASSERT_OK_AND_ASSIGN(Node * node_loop,
                           main.value()->GetNode("counted_for.10"));
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
  auto bit_type = [&package](int64_t num_bits) {
    return package->GetBitsType(num_bits);
  };
  FunctionType func({bit_type(32)}, bit_type(32));
  oracle_builder.WithFunctionType(&func);
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

TEST_P(SequentialGeneratorTest, StaticStridedCounterNegativeMax) {
  // Build the builder.
  SequentialModuleBuilder builder(
      SequentialOptions().use_system_verilog(UseSystemVerilog()), nullptr);
  EXPECT_EQ(builder
                .AddStaticStridedCounter("m_counter", 1, -3, nullptr, nullptr,
                                         nullptr)
                .status(),
            absl::UnimplementedError(
                "Tried to generate static strided counter with non-positive "
                "value_limit_exlusive - not currently supported."));
}

TEST_P(SequentialGeneratorTest, StaticStridedCounterZeroMax) {
  // Build the builder.
  SequentialModuleBuilder builder(
      SequentialOptions().use_system_verilog(UseSystemVerilog()), nullptr);
  EXPECT_EQ(builder
                .AddStaticStridedCounter("m_counter", 1, -3, nullptr, nullptr,
                                         nullptr)
                .status(),
            absl::UnimplementedError(
                "Tried to generate static strided counter with non-positive "
                "value_limit_exlusive - not currently supported."));
}

TEST_P(SequentialGeneratorTest, StaticStridedCounterNegativeStride) {
  // Build the builder.
  SequentialModuleBuilder builder(
      SequentialOptions().use_system_verilog(UseSystemVerilog()), nullptr);
  EXPECT_EQ(
      builder
          .AddStaticStridedCounter("m_counter", -1, 3, nullptr, nullptr,
                                   nullptr)
          .status(),
      absl::UnimplementedError(
          "Tried to generate static strided counter with non-positive stride - "
          "not currently supported."));
}

TEST_P(SequentialGeneratorTest, StaticStridedCounterZeroStride) {
  // Build the builder.
  SequentialModuleBuilder builder(
      SequentialOptions().use_system_verilog(UseSystemVerilog()), nullptr);
  EXPECT_EQ(
      builder
          .AddStaticStridedCounter("m_counter", -1, 3, nullptr, nullptr,
                                   nullptr)
          .status(),
      absl::UnimplementedError(
          "Tried to generate static strided counter with non-positive stride - "
          "not currently supported."));
}

TEST_P(SequentialGeneratorTest, StaticStridedCounterSimple) {
  // Make counter signature.
  constexpr int64_t num_counter_bits = 2;
  ModuleSignatureBuilder signature_builder("static_strided_counter_signature");
  signature_builder.AddDataInput("set_zero", 1);
  signature_builder.AddDataInput("increment", 1);
  signature_builder.AddDataOutput("value", num_counter_bits);
  signature_builder.AddDataOutput("holds_max_inclusive_value", 1);
  signature_builder.WithClock("clk");
  signature_builder.WithReadyValidInterface("ready_in", "valid_in", "ready_out",
                                            "valid_out");
  XLS_ASSERT_OK_AND_ASSIGN(ModuleSignature signature,
                           signature_builder.Build());
  // Build the builder.
  SequentialModuleBuilder builder(
      SequentialOptions().use_system_verilog(UseSystemVerilog()), nullptr);
  XLS_ASSERT_OK(builder.InitializeModuleBuilder(signature));
  const SequentialModuleBuilder::PortReferences* ports = builder.ports();

  // Add Counter.
  XLS_ASSERT_OK_AND_ASSIGN(
      SequentialModuleBuilder::StridedCounterReferences counter_refs,
      builder.AddStaticStridedCounter("m_counter", 1, 3, ports->clk,
                                      ports->data_in[0], ports->data_in[1]));
  builder.AddContinuousAssignment(
      ports->data_out[0], counter_refs.value);
  builder.AddContinuousAssignment(
      ports->data_out[1], counter_refs.holds_max_inclusive_value);

  // Test counter module.
  ModuleTestbench tb(builder.module(), GetSimulator(), "clk");
  tb.Set("set_zero", 1);
  tb.NextCycle().ExpectEq("value", 0).ExpectEq("holds_max_inclusive_value", 0);
  tb.Set("set_zero", 0);
  tb.Set("increment", 1);
  tb.NextCycle().ExpectEq("value", 1).ExpectEq("holds_max_inclusive_value", 0);
  tb.NextCycle().ExpectEq("value", 2).ExpectEq("holds_max_inclusive_value", 1);
  XLS_ASSERT_OK(tb.Run());
}

TEST_P(SequentialGeneratorTest, StaticStridedCounterIntermittentIncrement) {
  // Make counter signature.
  constexpr int64_t num_counter_bits = 2;
  ModuleSignatureBuilder signature_builder("static_strided_counter_signature");
  signature_builder.AddDataInput("set_zero", 1);
  signature_builder.AddDataInput("increment", 1);
  signature_builder.AddDataOutput("value", num_counter_bits);
  signature_builder.AddDataOutput("holds_max_inclusive_value", 1);
  signature_builder.WithClock("clk");
  signature_builder.WithReadyValidInterface("ready_in", "valid_in", "ready_out",
                                            "valid_out");
  XLS_ASSERT_OK_AND_ASSIGN(ModuleSignature signature,
                           signature_builder.Build());
  // Build the builder.
  SequentialModuleBuilder builder(
      SequentialOptions().use_system_verilog(UseSystemVerilog()), nullptr);
  XLS_ASSERT_OK(builder.InitializeModuleBuilder(signature));
  const SequentialModuleBuilder::PortReferences* ports = builder.ports();

  // Add Counter.
  XLS_ASSERT_OK_AND_ASSIGN(
      SequentialModuleBuilder::StridedCounterReferences counter_refs,
      builder.AddStaticStridedCounter("m_counter", 1, 3, ports->clk,
                                      ports->data_in[0], ports->data_in[1]));
  builder.AddContinuousAssignment(
      ports->data_out[0], counter_refs.value);
  builder.AddContinuousAssignment(
      ports->data_out[1], counter_refs.holds_max_inclusive_value);

  // Test counter module.
  ModuleTestbench tb(builder.module(), GetSimulator(), "clk");
  tb.Set("set_zero", 1);
  tb.NextCycle().ExpectEq("value", 0).ExpectEq("holds_max_inclusive_value", 0);
  tb.Set("set_zero", 0)
      .Set("increment", 0)
      .AdvanceNCycles(100)
      .Set("increment", 1);
  tb.NextCycle().ExpectEq("value", 1).ExpectEq("holds_max_inclusive_value", 0);
  tb.Set("increment", 0).AdvanceNCycles(100).Set("increment", 1);
  tb.NextCycle().ExpectEq("value", 2).ExpectEq("holds_max_inclusive_value", 1);
  XLS_ASSERT_OK(tb.Run());
}

TEST_P(SequentialGeneratorTest, StaticStridedCounterNonOneStride) {
  // Make counter signature.
  constexpr int64_t num_counter_bits = 3;
  ModuleSignatureBuilder signature_builder("static_strided_counter_signature");
  signature_builder.AddDataInput("set_zero", 1);
  signature_builder.AddDataInput("increment", 1);
  signature_builder.AddDataOutput("value", num_counter_bits);
  signature_builder.AddDataOutput("holds_max_inclusive_value", 1);
  signature_builder.WithClock("clk");
  signature_builder.WithReadyValidInterface("ready_in", "valid_in", "ready_out",
                                            "valid_out");
  XLS_ASSERT_OK_AND_ASSIGN(ModuleSignature signature,
                           signature_builder.Build());
  // Build the builder.
  SequentialModuleBuilder builder(
      SequentialOptions().use_system_verilog(UseSystemVerilog()), nullptr);
  XLS_ASSERT_OK(builder.InitializeModuleBuilder(signature));
  const SequentialModuleBuilder::PortReferences* ports = builder.ports();

  // Add Counter.
  XLS_ASSERT_OK_AND_ASSIGN(
      SequentialModuleBuilder::StridedCounterReferences counter_refs,
      builder.AddStaticStridedCounter("m_counter", 3, 7, ports->clk,
                                      ports->data_in[0], ports->data_in[1]));
  builder.AddContinuousAssignment(
      ports->data_out[0], counter_refs.value);
  builder.AddContinuousAssignment(
      ports->data_out[1], counter_refs.holds_max_inclusive_value);

  // Test counter module.
  ModuleTestbench tb(builder.module(), GetSimulator(), "clk");
  tb.Set("set_zero", 1);
  tb.NextCycle().ExpectEq("value", 0).ExpectEq("holds_max_inclusive_value", 0);
  tb.Set("set_zero", 0);
  tb.Set("increment", 1);
  tb.NextCycle().ExpectEq("value", 3).ExpectEq("holds_max_inclusive_value", 0);
  tb.NextCycle().ExpectEq("value", 6).ExpectEq("holds_max_inclusive_value", 1);
  XLS_ASSERT_OK(tb.Run());
}

TEST_P(SequentialGeneratorTest, StaticStridedCounterStrideMultipleLimit) {
  // Make counter signature.
  constexpr int64_t num_counter_bits = 2;
  ModuleSignatureBuilder signature_builder("static_strided_counter_signature");
  signature_builder.AddDataInput("set_zero", 1);
  signature_builder.AddDataInput("increment", 1);
  signature_builder.AddDataOutput("value", num_counter_bits);
  signature_builder.AddDataOutput("holds_max_inclusive_value", 1);
  signature_builder.WithClock("clk");
  signature_builder.WithReadyValidInterface("ready_in", "valid_in", "ready_out",
                                            "valid_out");
  XLS_ASSERT_OK_AND_ASSIGN(ModuleSignature signature,
                           signature_builder.Build());
  // Build the builder.
  SequentialModuleBuilder builder(
      SequentialOptions().use_system_verilog(UseSystemVerilog()), nullptr);
  XLS_ASSERT_OK(builder.InitializeModuleBuilder(signature));
  const SequentialModuleBuilder::PortReferences* ports = builder.ports();

  // Add Counter.
  XLS_ASSERT_OK_AND_ASSIGN(
      SequentialModuleBuilder::StridedCounterReferences counter_refs,
      builder.AddStaticStridedCounter("m_counter", 3, 6, ports->clk,
                                      ports->data_in[0], ports->data_in[1]));
  builder.AddContinuousAssignment(
      ports->data_out[0], counter_refs.value);
  builder.AddContinuousAssignment(
      ports->data_out[1], counter_refs.holds_max_inclusive_value);

  // Test counter module.
  ModuleTestbench tb(builder.module(), GetSimulator(), "clk");
  tb.Set("set_zero", 1);
  tb.NextCycle().ExpectEq("value", 0).ExpectEq("holds_max_inclusive_value", 0);
  tb.Set("set_zero", 0);
  tb.Set("increment", 1);
  tb.NextCycle().ExpectEq("value", 3).ExpectEq("holds_max_inclusive_value", 1);
  XLS_ASSERT_OK(tb.Run());
}

TEST_P(SequentialGeneratorTest,
       StaticStridedCounterOneLessThanStrideMultipleLimit) {
  // Make counter signature.
  constexpr int64_t num_counter_bits = 2;
  ModuleSignatureBuilder signature_builder("static_strided_counter_signature");
  signature_builder.AddDataInput("set_zero", 1);
  signature_builder.AddDataInput("increment", 1);
  signature_builder.AddDataOutput("value", num_counter_bits);
  signature_builder.AddDataOutput("holds_max_inclusive_value", 1);
  signature_builder.WithClock("clk");
  signature_builder.WithReadyValidInterface("ready_in", "valid_in", "ready_out",
                                            "valid_out");
  XLS_ASSERT_OK_AND_ASSIGN(ModuleSignature signature,
                           signature_builder.Build());
  // Build the builder.
  SequentialModuleBuilder builder(
      SequentialOptions().use_system_verilog(UseSystemVerilog()), nullptr);
  XLS_ASSERT_OK(builder.InitializeModuleBuilder(signature));
  const SequentialModuleBuilder::PortReferences* ports = builder.ports();

  // Add Counter.
  XLS_ASSERT_OK_AND_ASSIGN(
      SequentialModuleBuilder::StridedCounterReferences counter_refs,
      builder.AddStaticStridedCounter("m_counter", 3, 5, ports->clk,
                                      ports->data_in[0], ports->data_in[1]));
  builder.AddContinuousAssignment(
      ports->data_out[0], counter_refs.value);
  builder.AddContinuousAssignment(
      ports->data_out[1], counter_refs.holds_max_inclusive_value);

  // Test counter module.
  ModuleTestbench tb(builder.module(), GetSimulator(), "clk");
  tb.Set("set_zero", 1);
  tb.NextCycle().ExpectEq("value", 0).ExpectEq("holds_max_inclusive_value", 0);
  tb.Set("set_zero", 0);
  tb.Set("increment", 1);
  tb.NextCycle().ExpectEq("value", 3).ExpectEq("holds_max_inclusive_value", 1);
  XLS_ASSERT_OK(tb.Run());
}

TEST_P(SequentialGeneratorTest, StaticStridedCounterClearValue) {
  // Make counter signature.
  constexpr int64_t num_counter_bits = 2;
  ModuleSignatureBuilder signature_builder("static_strided_counter_signature");
  signature_builder.AddDataInput("set_zero", 1);
  signature_builder.AddDataInput("increment", 1);
  signature_builder.AddDataOutput("value", num_counter_bits);
  signature_builder.AddDataOutput("holds_max_inclusive_value", 1);
  signature_builder.WithClock("clk");
  signature_builder.WithReadyValidInterface("ready_in", "valid_in", "ready_out",
                                            "valid_out");
  XLS_ASSERT_OK_AND_ASSIGN(ModuleSignature signature,
                           signature_builder.Build());
  // Build the builder.
  SequentialModuleBuilder builder(
      SequentialOptions().use_system_verilog(UseSystemVerilog()), nullptr);
  XLS_ASSERT_OK(builder.InitializeModuleBuilder(signature));
  const SequentialModuleBuilder::PortReferences* ports = builder.ports();

  // Add Counter.
  XLS_ASSERT_OK_AND_ASSIGN(
      SequentialModuleBuilder::StridedCounterReferences counter_refs,
      builder.AddStaticStridedCounter("m_counter", 1, 3, ports->clk,
                                      ports->data_in[0], ports->data_in[1]));
  builder.AddContinuousAssignment(
      ports->data_out[0], counter_refs.value);
  builder.AddContinuousAssignment(
      ports->data_out[1], counter_refs.holds_max_inclusive_value);

  // Test counter module.
  ModuleTestbench tb(builder.module(), GetSimulator(), "clk");
  tb.Set("set_zero", 1);
  tb.NextCycle().ExpectEq("value", 0).ExpectEq("holds_max_inclusive_value", 0);
  tb.Set("set_zero", 0);
  tb.Set("increment", 1);
  tb.NextCycle().ExpectEq("value", 1).ExpectEq("holds_max_inclusive_value", 0);
  tb.Set("set_zero", 1);
  tb.NextCycle().ExpectEq("value", 0).ExpectEq("holds_max_inclusive_value", 0);
  XLS_ASSERT_OK(tb.Run());
}

TEST_P(SequentialGeneratorTest, FsmSimple) {
  // Make counter signature.
  ModuleSignatureBuilder signature_builder("fsm_signature");
  signature_builder.AddDataInput("index_holds_max_inclusive_value_port", 1);
  signature_builder.AddDataOutput("last_pipeline_cycle_port", 1);
  signature_builder.WithClock("clk");
  signature_builder.WithReadyValidInterface("ready_in", "valid_in", "ready_out",
                                            "valid_out");
  ResetProto reset;
  reset.set_name("reset");
  reset.set_asynchronous(false);
  reset.set_active_low(false);
  signature_builder.WithReset(reset.name(), reset.asynchronous(),
                              reset.active_low());
  XLS_ASSERT_OK_AND_ASSIGN(ModuleSignature signature,
                           signature_builder.Build());

  // Would be better to do this as a parameterized test, but this
  // conflicts with paramterizing based on the simulation target
  // defined by VerilogTestBase.
  for (int64_t latency = 0; latency < 3; ++latency) {
    // Build the builder.
    SequentialOptions sequential_options;
    sequential_options.use_system_verilog(UseSystemVerilog());
    sequential_options.reset(reset);
    SequentialModuleBuilder builder(sequential_options, nullptr);
    XLS_ASSERT_OK(builder.InitializeModuleBuilder(signature));
    const SequentialModuleBuilder::PortReferences* ports = builder.ports();

    // Add FSM.
    LogicRef* index_holds_max_inclusive_value = ports->data_in[0];
    LogicRef* last_pipeline_cycle = ports->data_out[0];
    XLS_ASSERT_OK(builder.AddFsm(/*pipeline_latency=*/latency,
                                 index_holds_max_inclusive_value,
                                 last_pipeline_cycle));

    ModuleTestbench tb(builder.module(), GetSimulator(), "clk");
    // Reset.
    tb.Set("reset", 1)
        .Set("valid_in", 0)
        .Set("ready_out", 0)
        .Set("index_holds_max_inclusive_value_port", 0);
    tb.NextCycle();

    // Ready.
    tb.ExpectEq("ready_in", 1)
        .ExpectEq("valid_out", 0)
        .ExpectEq("last_pipeline_cycle_port", 0);
    tb.Set("reset", 0)
        .Set("valid_in", 0)
        .Set("ready_out", 0)
        .Set("index_holds_max_inclusive_value_port", 0);

    // Hold ready.
    for (int i = 0; i < 4 * latency; ++i) {
      tb.NextCycle();
      tb.ExpectEq("ready_in", 1)
          .ExpectEq("valid_out", 0)
          .ExpectEq("last_pipeline_cycle_port", 0);
      tb.Set("reset", 0)
          .Set("valid_in", 0)
          .Set("ready_out", 0)
          .Set("index_holds_max_inclusive_value_port", 0);
    }

    // Valid in, transition to running state.
    tb.Set("reset", 0)
        .Set("valid_in", 1)
        .Set("ready_out", 0)
        .Set("index_holds_max_inclusive_value_port", 0);

    // Iterate a few pipeline iterations.
    for (int i = 0; i < 3; ++i) {
      for (int cycle = 0; cycle < latency + 1; ++cycle) {
        tb.NextCycle();
        tb.ExpectEq("ready_in", 0).ExpectEq("valid_out", 0);
        // Set "last_pipeline_cycle_port" high for the
        // last cycle.
        if (cycle == latency) {
          tb.ExpectEq("last_pipeline_cycle_port", 1);
        } else {
          tb.ExpectEq("last_pipeline_cycle_port", 0);
        }
        tb.Set("reset", 0)
            .Set("valid_in", 0)
            .Set("ready_out", 0)
            .Set("index_holds_max_inclusive_value_port", 0);
      }
    }

    // "index_holds_max_inclusive_value_port" set externally, final pipeline
    // iteration.
    for (int cycle = 0; cycle < latency + 1; ++cycle) {
      tb.NextCycle();
      tb.ExpectEq("ready_in", 0).ExpectEq("valid_out", 0);
      // Set "last_pipeline_cycle_port" high for the
      // last cycle.
      if (cycle == latency) {
        tb.ExpectEq("last_pipeline_cycle_port", 1);
      } else {
        tb.ExpectEq("last_pipeline_cycle_port", 0);
      }
      tb.Set("reset", 0)
          .Set("valid_in", 0)
          .Set("ready_out", 0)
          .Set("index_holds_max_inclusive_value_port", 1);
    }

    // Done state.
    tb.NextCycle();
    tb.ExpectEq("ready_in", 0)
        .ExpectEq("valid_out", 1)
        .ExpectEq("last_pipeline_cycle_port", 0);
    tb.Set("reset", 0)
        .Set("valid_in", 0)
        .Set("ready_out", 0)
        .Set("index_holds_max_inclusive_value_port", 1);

    // Wait for ready out, hold done state.
    for (int i = 0; i < 4 * latency; ++i) {
      tb.NextCycle();
      tb.ExpectEq("ready_in", 0)
          .ExpectEq("valid_out", 1)
          .ExpectEq("last_pipeline_cycle_port", 0);
      tb.Set("reset", 0)
          .Set("valid_in", 0)
          .Set("ready_out", 0)
          .Set("index_holds_max_inclusive_value_port", 1);
    }

    // Ready out set, transition to ready state again.
    tb.Set("reset", 0)
        .Set("valid_in", 0)
        .Set("ready_out", 1)
        .Set("index_holds_max_inclusive_value_port", 1);
    tb.NextCycle();
    tb.ExpectEq("ready_in", 1)
        .ExpectEq("valid_out", 0)
        .ExpectEq("last_pipeline_cycle_port", 0);

    XLS_ASSERT_OK(tb.Run());
  }
}

TEST_P(SequentialGeneratorTest, FsmActiveLowReset) {
  // Make counter signature.
  ModuleSignatureBuilder signature_builder("fsm_signature");
  signature_builder.AddDataInput("index_holds_max_inclusive_value_port", 1);
  signature_builder.AddDataOutput("last_pipeline_cycle_port", 1);
  signature_builder.WithClock("clk");
  signature_builder.WithReadyValidInterface("ready_in", "valid_in", "ready_out",
                                            "valid_out");
  ResetProto reset;
  reset.set_name("reset");
  reset.set_asynchronous(false);
  reset.set_active_low(true);
  signature_builder.WithReset(reset.name(), reset.asynchronous(),
                              reset.active_low());
  XLS_ASSERT_OK_AND_ASSIGN(ModuleSignature signature,
                           signature_builder.Build());

  // Would be better to do this as a parameterized test, but this
  // conflicts with paramterizing based on the simulation target
  // defined by VerilogTestBase.
  for (int64_t latency = 0; latency < 3; ++latency) {
    // Build the builder.
    SequentialOptions sequential_options;
    sequential_options.use_system_verilog(UseSystemVerilog());
    sequential_options.reset(reset);
    SequentialModuleBuilder builder(sequential_options, nullptr);
    XLS_ASSERT_OK(builder.InitializeModuleBuilder(signature));
    const SequentialModuleBuilder::PortReferences* ports = builder.ports();

    // Add FSM.
    LogicRef* index_holds_max_inclusive_value = ports->data_in[0];
    LogicRef* last_pipeline_cycle = ports->data_out[0];
    XLS_ASSERT_OK(builder.AddFsm(/*pipeline_latency=*/latency,
                                 index_holds_max_inclusive_value,
                                 last_pipeline_cycle));

    ModuleTestbench tb(builder.module(), GetSimulator(), "clk");
    // Reset.
    tb.Set("reset", 0)
        .Set("valid_in", 0)
        .Set("ready_out", 0)
        .Set("index_holds_max_inclusive_value_port", 0);
    tb.NextCycle();

    // Ready.
    tb.ExpectEq("ready_in", 1)
        .ExpectEq("valid_out", 0)
        .ExpectEq("last_pipeline_cycle_port", 0);
    tb.Set("reset", 1)
        .Set("valid_in", 0)
        .Set("ready_out", 0)
        .Set("index_holds_max_inclusive_value_port", 0);

    // Hold ready.
    for (int i = 0; i < 4 * latency; ++i) {
      tb.NextCycle();
      tb.ExpectEq("ready_in", 1)
          .ExpectEq("valid_out", 0)
          .ExpectEq("last_pipeline_cycle_port", 0);
      tb.Set("reset", 1)
          .Set("valid_in", 0)
          .Set("ready_out", 0)
          .Set("index_holds_max_inclusive_value_port", 0);
    }

    // Valid in, transition to running state.
    tb.Set("reset", 1)
        .Set("valid_in", 1)
        .Set("ready_out", 0)
        .Set("index_holds_max_inclusive_value_port", 0);

    // Iterate a few pipeline iterations.
    for (int i = 0; i < 3; ++i) {
      for (int cycle = 0; cycle < latency + 1; ++cycle) {
        tb.NextCycle();
        tb.ExpectEq("ready_in", 0).ExpectEq("valid_out", 0);
        // Set "last_pipeline_cycle_port" high for the
        // last cycle.
        if (cycle == latency) {
          tb.ExpectEq("last_pipeline_cycle_port", 1);
        } else {
          tb.ExpectEq("last_pipeline_cycle_port", 0);
        }
        tb.Set("reset", 1)
            .Set("valid_in", 0)
            .Set("ready_out", 0)
            .Set("index_holds_max_inclusive_value_port", 0);
      }
    }

    // "index_holds_max_inclusive_value_port" set externally, final pipeline
    // iteration.
    for (int cycle = 0; cycle < latency + 1; ++cycle) {
      tb.NextCycle();
      tb.ExpectEq("ready_in", 0).ExpectEq("valid_out", 0);
      // Set "last_pipeline_cycle_port" high for the
      // last cycle.
      if (cycle == latency) {
        tb.ExpectEq("last_pipeline_cycle_port", 1);
      } else {
        tb.ExpectEq("last_pipeline_cycle_port", 0);
      }
      tb.Set("reset", 1)
          .Set("valid_in", 0)
          .Set("ready_out", 0)
          .Set("index_holds_max_inclusive_value_port", 1);
    }

    // Done state.
    tb.NextCycle();
    tb.ExpectEq("ready_in", 0)
        .ExpectEq("valid_out", 1)
        .ExpectEq("last_pipeline_cycle_port", 0);
    tb.Set("reset", 1)
        .Set("valid_in", 0)
        .Set("ready_out", 0)
        .Set("index_holds_max_inclusive_value_port", 1);

    // Wait for ready out, hold done state.
    for (int i = 0; i < 4 * latency; ++i) {
      tb.NextCycle();
      tb.ExpectEq("ready_in", 0)
          .ExpectEq("valid_out", 1)
          .ExpectEq("last_pipeline_cycle_port", 0);
      tb.Set("reset", 1)
          .Set("valid_in", 0)
          .Set("ready_out", 0)
          .Set("index_holds_max_inclusive_value_port", 1);
    }

    // Ready out set, transition to ready state again.
    tb.Set("reset", 1)
        .Set("valid_in", 0)
        .Set("ready_out", 1)
        .Set("index_holds_max_inclusive_value_port", 1);
    tb.NextCycle();
    tb.ExpectEq("ready_in", 1)
        .ExpectEq("valid_out", 0)
        .ExpectEq("last_pipeline_cycle_port", 0);

    XLS_ASSERT_OK(tb.Run());
  }
}

TEST_P(SequentialGeneratorTest, FsmIgnoreMaxValueUnlessRunning) {
  // Make counter signature.
  ModuleSignatureBuilder signature_builder("fsm_signature");
  signature_builder.AddDataInput("index_holds_max_inclusive_value_port", 1);
  signature_builder.AddDataOutput("last_pipeline_cycle_port", 1);
  signature_builder.WithClock("clk");
  signature_builder.WithReadyValidInterface("ready_in", "valid_in", "ready_out",
                                            "valid_out");
  ResetProto reset;
  reset.set_name("reset");
  reset.set_asynchronous(false);
  reset.set_active_low(false);
  signature_builder.WithReset(reset.name(), reset.asynchronous(),
                              reset.active_low());
  XLS_ASSERT_OK_AND_ASSIGN(ModuleSignature signature,
                           signature_builder.Build());

  // Would be better to do this as a parameterized test, but this
  // conflicts with paramterizing based on the simulation target
  // defined by VerilogTestBase.
  for (int64_t latency = 0; latency < 3; ++latency) {
    // Build the builder.
    SequentialOptions sequential_options;
    sequential_options.use_system_verilog(UseSystemVerilog());
    sequential_options.reset(reset);
    SequentialModuleBuilder builder(sequential_options, nullptr);
    XLS_ASSERT_OK(builder.InitializeModuleBuilder(signature));
    const SequentialModuleBuilder::PortReferences* ports = builder.ports();

    // Add FSM.
    LogicRef* index_holds_max_inclusive_value = ports->data_in[0];
    LogicRef* last_pipeline_cycle = ports->data_out[0];
    XLS_ASSERT_OK(builder.AddFsm(/*pipeline_latency=*/latency,
                                 index_holds_max_inclusive_value,
                                 last_pipeline_cycle));

    ModuleTestbench tb(builder.module(), GetSimulator(), "clk");
    // Reset.
    tb.Set("reset", 1)
        .Set("valid_in", 0)
        .Set("ready_out", 0)
        .Set("index_holds_max_inclusive_value_port", 1);
    tb.NextCycle();

    // Ready.
    tb.ExpectEq("ready_in", 1)
        .ExpectEq("valid_out", 0)
        .ExpectEq("last_pipeline_cycle_port", 0);
    tb.Set("reset", 0)
        .Set("valid_in", 0)
        .Set("ready_out", 0)
        .Set("index_holds_max_inclusive_value_port", 1);

    // Hold ready.
    for (int i = 0; i < 4 * latency; ++i) {
      tb.NextCycle();
      tb.ExpectEq("ready_in", 1)
          .ExpectEq("valid_out", 0)
          .ExpectEq("last_pipeline_cycle_port", 0);
      tb.Set("reset", 0)
          .Set("valid_in", 0)
          .Set("ready_out", 0)
          .Set("index_holds_max_inclusive_value_port", 1);
    }

    // Valid in, transition to running state.
    tb.Set("reset", 0)
        .Set("valid_in", 1)
        .Set("ready_out", 0)
        .Set("index_holds_max_inclusive_value_port", 1);

    // Iterate a few pipeline iterations.
    for (int i = 0; i < 3; ++i) {
      for (int cycle = 0; cycle < latency + 1; ++cycle) {
        tb.NextCycle();
        tb.ExpectEq("ready_in", 0).ExpectEq("valid_out", 0);
        // Set "last_pipeline_cycle_port" high for the
        // last cycle.
        if (cycle == latency) {
          tb.ExpectEq("last_pipeline_cycle_port", 1);
        } else {
          tb.ExpectEq("last_pipeline_cycle_port", 0);
        }
        tb.Set("reset", 0)
            .Set("valid_in", 0)
            .Set("ready_out", 0)
            .Set("index_holds_max_inclusive_value_port", 0);
      }
    }

    // "index_holds_max_inclusive_value_port" set externally, final pipeline
    // iteration.
    for (int cycle = 0; cycle < latency + 1; ++cycle) {
      tb.NextCycle();
      tb.ExpectEq("ready_in", 0).ExpectEq("valid_out", 0);
      // Set "last_pipeline_cycle_port" high for the
      // last cycle.
      if (cycle == latency) {
        tb.ExpectEq("last_pipeline_cycle_port", 1);
      } else {
        tb.ExpectEq("last_pipeline_cycle_port", 0);
      }
      tb.Set("reset", 0)
          .Set("valid_in", 0)
          .Set("ready_out", 0)
          .Set("index_holds_max_inclusive_value_port", 1);
    }

    // Done state.
    tb.NextCycle();
    tb.ExpectEq("ready_in", 0)
        .ExpectEq("valid_out", 1)
        .ExpectEq("last_pipeline_cycle_port", 0);
    tb.Set("reset", 0)
        .Set("valid_in", 0)
        .Set("ready_out", 0)
        .Set("index_holds_max_inclusive_value_port", 1);

    XLS_ASSERT_OK(tb.Run());
  }
}

TEST_P(SequentialGeneratorTest, FsmIgnoreValidInUnlessReady) {
  // Make counter signature.
  ModuleSignatureBuilder signature_builder("fsm_signature");
  signature_builder.AddDataInput("index_holds_max_inclusive_value_port", 1);
  signature_builder.AddDataOutput("last_pipeline_cycle_port", 1);
  signature_builder.WithClock("clk");
  signature_builder.WithReadyValidInterface("ready_in", "valid_in", "ready_out",
                                            "valid_out");
  ResetProto reset;
  reset.set_name("reset");
  reset.set_asynchronous(false);
  reset.set_active_low(false);
  signature_builder.WithReset(reset.name(), reset.asynchronous(),
                              reset.active_low());
  XLS_ASSERT_OK_AND_ASSIGN(ModuleSignature signature,
                           signature_builder.Build());

  // Would be better to do this as a parameterized test, but this
  // conflicts with paramterizing based on the simulation target
  // defined by VerilogTestBase.
  for (int64_t latency = 0; latency < 3; ++latency) {
    // Build the builder.
    SequentialOptions sequential_options;
    sequential_options.use_system_verilog(UseSystemVerilog());
    sequential_options.reset(reset);
    SequentialModuleBuilder builder(sequential_options, nullptr);
    XLS_ASSERT_OK(builder.InitializeModuleBuilder(signature));
    const SequentialModuleBuilder::PortReferences* ports = builder.ports();

    // Add FSM.
    LogicRef* index_holds_max_inclusive_value = ports->data_in[0];
    LogicRef* last_pipeline_cycle = ports->data_out[0];
    XLS_ASSERT_OK(builder.AddFsm(/*pipeline_latency=*/latency,
                                 index_holds_max_inclusive_value,
                                 last_pipeline_cycle));

    ModuleTestbench tb(builder.module(), GetSimulator(), "clk");
    // Reset.
    tb.Set("reset", 1)
        .Set("valid_in", 1)
        .Set("ready_out", 0)
        .Set("index_holds_max_inclusive_value_port", 0);
    tb.NextCycle();

    // Ready.
    tb.ExpectEq("ready_in", 1)
        .ExpectEq("valid_out", 0)
        .ExpectEq("last_pipeline_cycle_port", 0);
    tb.Set("reset", 0)
        .Set("valid_in", 1)
        .Set("ready_out", 0)
        .Set("index_holds_max_inclusive_value_port", 0);

    // Valid in, transition to running state.
    tb.Set("reset", 0)
        .Set("valid_in", 1)
        .Set("ready_out", 0)
        .Set("index_holds_max_inclusive_value_port", 0);

    // Iterate a few pipeline iterations.
    for (int i = 0; i < 3; ++i) {
      for (int cycle = 0; cycle < latency + 1; ++cycle) {
        tb.NextCycle();
        tb.ExpectEq("ready_in", 0).ExpectEq("valid_out", 0);
        // Set "last_pipeline_cycle_port" high for the
        // last cycle.
        if (cycle == latency) {
          tb.ExpectEq("last_pipeline_cycle_port", 1);
        } else {
          tb.ExpectEq("last_pipeline_cycle_port", 0);
        }
        tb.Set("reset", 0)
            .Set("valid_in", 1)
            .Set("ready_out", 0)
            .Set("index_holds_max_inclusive_value_port", 0);
      }
    }

    // "index_holds_max_inclusive_value_port" set externally, final pipeline
    // iteration.
    for (int cycle = 0; cycle < latency + 1; ++cycle) {
      tb.NextCycle();
      tb.ExpectEq("ready_in", 0).ExpectEq("valid_out", 0);
      // Set "last_pipeline_cycle_port" high for the
      // last cycle.
      if (cycle == latency) {
        tb.ExpectEq("last_pipeline_cycle_port", 1);
      } else {
        tb.ExpectEq("last_pipeline_cycle_port", 0);
      }
      tb.Set("reset", 0)
          .Set("valid_in", 1)
          .Set("ready_out", 0)
          .Set("index_holds_max_inclusive_value_port", 1);
    }

    // Done state.
    tb.NextCycle();
    tb.ExpectEq("ready_in", 0)
        .ExpectEq("valid_out", 1)
        .ExpectEq("last_pipeline_cycle_port", 0);
    tb.Set("reset", 0)
        .Set("valid_in", 1)
        .Set("ready_out", 0)
        .Set("index_holds_max_inclusive_value_port", 1);

    // Wait for ready out, hold done state.
    for (int i = 0; i < 4 * latency; ++i) {
      tb.NextCycle();
      tb.ExpectEq("ready_in", 0)
          .ExpectEq("valid_out", 1)
          .ExpectEq("last_pipeline_cycle_port", 0);
      tb.Set("reset", 0)
          .Set("valid_in", 1)
          .Set("ready_out", 0)
          .Set("index_holds_max_inclusive_value_port", 1);
    }

    // Ready out set, transition to ready state again.
    tb.Set("reset", 0)
        .Set("valid_in", 1)
        .Set("ready_out", 1)
        .Set("index_holds_max_inclusive_value_port", 1);
    tb.NextCycle();
    tb.ExpectEq("ready_in", 1)
        .ExpectEq("valid_out", 0)
        .ExpectEq("last_pipeline_cycle_port", 0);

    // Transition to running.
    tb.Set("reset", 0)
        .Set("valid_in", 1)
        .Set("ready_out", 0)
        .Set("index_holds_max_inclusive_value_port", 0);
    tb.NextCycle();
    tb.ExpectEq("ready_in", 0).ExpectEq("valid_out", 0);

    XLS_ASSERT_OK(tb.Run());
  }
}

TEST_P(SequentialGeneratorTest, FsmNoReset) {
  // Make counter signature.
  ModuleSignatureBuilder signature_builder("fsm_signature");
  signature_builder.AddDataInput("index_holds_max_inclusive_value_port", 1);
  signature_builder.AddDataOutput("last_pipeline_cycle_port", 1);
  signature_builder.WithClock("clk");
  signature_builder.WithReadyValidInterface("ready_in", "valid_in", "ready_out",
                                            "valid_out");
  ResetProto reset;
  reset.set_name("reset");
  reset.set_asynchronous(false);
  reset.set_active_low(false);
  signature_builder.WithReset(reset.name(), reset.asynchronous(),
                              reset.active_low());
  XLS_ASSERT_OK_AND_ASSIGN(ModuleSignature signature,
                           signature_builder.Build());

  // Build the builder.
  SequentialOptions sequential_options;
  sequential_options.use_system_verilog(UseSystemVerilog());
  SequentialModuleBuilder builder(sequential_options, nullptr);
  XLS_ASSERT_OK(builder.InitializeModuleBuilder(signature));
  const SequentialModuleBuilder::PortReferences* ports = builder.ports();

  // Add FSM.
  LogicRef* index_holds_max_inclusive_value = ports->data_in[0];
  LogicRef* last_pipeline_cycle = ports->data_out[0];
  EXPECT_EQ(
      builder.AddFsm(/*pipeline_latency=*/1, index_holds_max_inclusive_value,
                     last_pipeline_cycle),
      absl::InvalidArgumentError("Tried to create FSM without specifying reset "
                                 "in SequentialOptions."));
}

TEST_P(SequentialGeneratorTest, SequentialModuleSimple) {
  std::string text = R"(
package SequentialModuleSimple

fn ____SequentialModuleSimple__main_counted_for_0_body(index: bits[32], acc: bits[32]) -> bits[32] {
  ret add.5: bits[32] = add(acc, index, pos=[(0,2,8)])
}

top fn __SequentialModuleSimple__main(init_acc: bits[32]) -> bits[32] {
  literal.2: bits[32] = literal(value=4, pos=[(0,1,51)])
  ret counted_for.6: bits[32] = counted_for(init_acc, trip_count=4, stride=1, body=____SequentialModuleSimple__main_counted_for_0_body, pos=[(0,1,5)])
}
)";
  // Would be better to do this as a parameterized test, but this
  // conflicts with paramterizing based on the simulation target
  // defined by VerilogTestBase.
  for (int64_t latency = 0; latency < 3; ++latency) {
    XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> package,
                             Parser::ParsePackage(text));
    std::optional<FunctionBase*> main = package->GetTop();
    ASSERT_TRUE(main.has_value());

    // Grab loop node.
    XLS_ASSERT_OK_AND_ASSIGN(Node * node_loop,
                             main.value()->GetNode("counted_for.6"));
    CountedFor* loop = node_loop->As<CountedFor>();

    // Build the builder.
    ResetProto reset;
    reset.set_name("reset");
    reset.set_asynchronous(false);
    reset.set_active_low(false);
    SequentialOptions sequential_options;
    sequential_options.use_system_verilog(UseSystemVerilog());
    sequential_options.reset(reset);
    sequential_options.pipeline_scheduling_options().pipeline_stages(latency +
                                                                     1);
    XLS_ASSERT_OK_AND_ASSIGN(ModuleGeneratorResult result,
                             ToSequentialModuleText(sequential_options, loop));

    // Check functionality.
    ModuleSimulator simulator =
        NewModuleSimulator(result.verilog_text, result.signature);
    EXPECT_THAT(simulator.Run({{"init_acc_in", Value(UBits(100, 32))}}),
                IsOkAndHolds(Value(UBits(106, 32))));
  }
}

TEST_P(SequentialGeneratorTest, SequentialModuleInvariants) {
  std::string text = R"(
package SequentialModuleInvariants

fn ____SequentialModuleInvariants__main_counted_for_0_body(index: bits[32], acc: bits[32], invara: bits[32], invarb: bits[32]) -> bits[32] {
  add.9: bits[32] = add(acc, index, pos=[(0,2,8)])
  add.10: bits[32] = add(add.9, invara, pos=[(0,2,16)])
  ret add.11: bits[32] = add(add.10, invarb, pos=[(0,2,25)])
}

top fn __SequentialModuleInvariants__main(init_acc: bits[32], invara: bits[32], invarb: bits[32]) -> bits[32] {
  literal.4: bits[32] = literal(value=4, pos=[(0,1,51)])
  ret counted_for.12: bits[32] = counted_for(init_acc, trip_count=4, stride=1, body=____SequentialModuleInvariants__main_counted_for_0_body, invariant_args=[invara, invarb], pos=[(0,1,5)])
}
)";
  // Would be better to do this as a parameterized test, but this
  // conflicts with paramterizing based on the simulation target
  // defined by VerilogTestBase.
  for (int64_t latency = 0; latency < 3; ++latency) {
    XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> package,
                             Parser::ParsePackage(text));
    std::optional<FunctionBase*> main = package->GetTop();
    ASSERT_TRUE(main.has_value());

    // Grab loop node.
    XLS_ASSERT_OK_AND_ASSIGN(Node * node_loop,
                             main.value()->GetNode("counted_for.12"));
    CountedFor* loop = node_loop->As<CountedFor>();

    // Build the builder.
    ResetProto reset;
    reset.set_name("reset");
    reset.set_asynchronous(false);
    reset.set_active_low(false);
    SequentialOptions sequential_options;
    sequential_options.use_system_verilog(UseSystemVerilog());
    sequential_options.reset(reset);
    sequential_options.pipeline_scheduling_options().pipeline_stages(latency +
                                                                     1);
    XLS_ASSERT_OK_AND_ASSIGN(ModuleGeneratorResult result,
                             ToSequentialModuleText(sequential_options, loop));

    // Check functionality.
    ModuleSimulator simulator =
        NewModuleSimulator(result.verilog_text, result.signature);
    EXPECT_THAT(simulator.Run({{"init_acc_in", Value(UBits(100, 32))},
                               {"invara_in", Value(UBits(1000, 32))},
                               {"invarb_in", Value(UBits(10000, 32))}}),
                IsOkAndHolds(Value(UBits(44106, 32))));
  }
}

TEST_P(SequentialGeneratorTest, SequentialModuleReadyValidTiming) {
  std::string text = R"(
package SequentialModuleSimple

fn ____SequentialModuleSimple__main_counted_for_0_body(index: bits[32], acc: bits[32]) -> bits[32] {
  ret add.5: bits[32] = add(acc, index, pos=[(0,2,8)])
}

top fn __SequentialModuleSimple__main(init_acc: bits[32]) -> bits[32] {
  literal.2: bits[32] = literal(value=4, pos=[(0,1,51)])
  ret counted_for.6: bits[32] = counted_for(init_acc, trip_count=4, stride=1, body=____SequentialModuleSimple__main_counted_for_0_body, pos=[(0,1,5)])
}
)";
  // Would be better to do this as a parameterized test, but this
  // conflicts with paramterizing based on the simulation target
  // defined by VerilogTestBase.
  for (int64_t latency = 0; latency < 3; ++latency) {
    XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> package,
                             Parser::ParsePackage(text));
    std::optional<FunctionBase*> main = package->GetTop();
    ASSERT_TRUE(main.has_value());

    // Grab loop node.
    XLS_ASSERT_OK_AND_ASSIGN(Node * node_loop,
                             main.value()->GetNode("counted_for.6"));
    CountedFor* loop = node_loop->As<CountedFor>();

    // Build the builder.
    ResetProto reset;
    reset.set_name("reset");
    reset.set_asynchronous(false);
    reset.set_active_low(false);
    SequentialOptions sequential_options;
    sequential_options.use_system_verilog(UseSystemVerilog());
    sequential_options.reset(reset);
    sequential_options.pipeline_scheduling_options().pipeline_stages(latency +
                                                                     1);
    XLS_ASSERT_OK_AND_ASSIGN(ModuleGeneratorResult result,
                             ToSequentialModuleText(sequential_options, loop));

    // Check functionality.
    ModuleTestbench tb(result.verilog_text, GetFileType(), result.signature,
                       GetSimulator());

    // Reset.
    tb.ExpectEq("ready_in", 0).ExpectEq("valid_out", 0);
    tb.Set("reset", 1)
        .Set("valid_in", 1)
        .Set("ready_out", 0)
        .Set("init_acc_in", 0);
    tb.NextCycle();

    // Ready.
    tb.ExpectEq("ready_in", 1).ExpectEq("valid_out", 0);
    tb.Set("reset", 0)
        .Set("valid_in", 0)
        .Set("ready_out", 0)
        .Set("init_acc_in", 0);

    // Hold ready.
    for (int i = 0; i < 4 * latency; ++i) {
      tb.NextCycle();
      tb.ExpectEq("ready_in", 1).ExpectEq("valid_out", 0);
      tb.Set("reset", 0)
          .Set("valid_in", 0)
          .Set("ready_out", 0)
          .Set("init_acc_in", i);
    }

    // Valid in, transition to running state.
    tb.Set("reset", 0)
        .Set("valid_in", 1)
        .Set("ready_out", 0)
        .Set("init_acc_in", 100);

    // Stop driving init_acc_in (should have loaded on clock edge).
    // Keep driving valid_in (should be ignored until ready again).
    tb.NextCycle();
    tb.ExpectEq("ready_in", 0).ExpectEq("valid_out", 0);
    tb.Set("reset", 0)
        .Set("valid_in", 1)
        .Set("ready_out", 0)
        .Set("init_acc_in", 0);

    // Run until done.
    tb.WaitFor("valid_out");
    tb.ExpectEq("ready_in", 0)
        .ExpectEq("valid_out", 1)
        .ExpectEq("counted_for_6_out", 106);

    // Wait for ready out, hold done state.
    for (int i = 0; i < 4 * latency; ++i) {
      tb.NextCycle();
      tb.ExpectEq("ready_in", 0)
          .ExpectEq("valid_out", 1)
          .ExpectEq("counted_for_6_out", 106);
      tb.Set("reset", 0)
          .Set("valid_in", 1)
          .Set("ready_out", 0)
          .Set("init_acc_in", 0);
    }

    // Ready out set, transition to ready state again.
    tb.Set("reset", 0)
        .Set("valid_in", 1)
        .Set("ready_out", 1)
        .Set("init_acc_in", 0);
    tb.NextCycle();
    tb.ExpectEq("ready_in", 1).ExpectEq("valid_out", 0);

    XLS_ASSERT_OK(tb.Run());
  }
}

TEST_P(SequentialGeneratorTest, SequentialModuleMultipleUses) {
  std::string text = R"(
package SequentialModuleSimple

fn ____SequentialModuleSimple__main_counted_for_0_body(index: bits[32], acc: bits[32]) -> bits[32] {
  ret add.5: bits[32] = add(acc, index, pos=[(0,2,8)])
}

top fn __SequentialModuleSimple__main(init_acc: bits[32]) -> bits[32] {
  literal.2: bits[32] = literal(value=4, pos=[(0,1,51)])
  ret counted_for.6: bits[32] = counted_for(init_acc, trip_count=4, stride=1, body=____SequentialModuleSimple__main_counted_for_0_body, pos=[(0,1,5)])
}
)";
  // Would be better to do this as a parameterized test, but this
  // conflicts with paramterizing based on the simulation target
  // defined by VerilogTestBase.
  for (int64_t latency = 0; latency < 3; ++latency) {
    XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> package,
                             Parser::ParsePackage(text));
    std::optional<FunctionBase*> main = package->GetTop();
    ASSERT_TRUE(main.has_value());

    // Grab loop node.
    XLS_ASSERT_OK_AND_ASSIGN(Node * node_loop,
                             main.value()->GetNode("counted_for.6"));
    CountedFor* loop = node_loop->As<CountedFor>();

    // Build the builder.
    ResetProto reset;
    reset.set_name("reset");
    reset.set_asynchronous(false);
    reset.set_active_low(false);
    SequentialOptions sequential_options;
    sequential_options.use_system_verilog(UseSystemVerilog());
    sequential_options.reset(reset);
    sequential_options.pipeline_scheduling_options().pipeline_stages(latency +
                                                                     1);
    XLS_ASSERT_OK_AND_ASSIGN(ModuleGeneratorResult result,
                             ToSequentialModuleText(sequential_options, loop));

    // Check functionality.
    ModuleTestbench tb(result.verilog_text, GetFileType(), result.signature,
                       GetSimulator());

    // Reset.
    tb.ExpectEq("ready_in", 0).ExpectEq("valid_out", 0);
    tb.Set("reset", 1)
        .Set("valid_in", 1)
        .Set("ready_out", 0)
        .Set("init_acc_in", 0);
    tb.NextCycle();

    // Ready.
    // Valid in, transition to running state.
    tb.ExpectEq("ready_in", 1).ExpectEq("valid_out", 0);
    tb.Set("reset", 0)
        .Set("valid_in", 1)
        .Set("ready_out", 0)
        .Set("init_acc_in", 100);

    // Run until done.
    tb.WaitFor("valid_out");
    tb.ExpectEq("ready_in", 0)
        .ExpectEq("valid_out", 1)
        .ExpectEq("counted_for_6_out", 106);

    // Ready out set, transition to ready state again.
    tb.Set("reset", 0)
        .Set("valid_in", 0)
        .Set("ready_out", 1)
        .Set("init_acc_in", 0);
    tb.NextCycle();

    // Ready.
    // Valid in, transition to running state.
    tb.ExpectEq("ready_in", 1).ExpectEq("valid_out", 0);
    tb.Set("reset", 0)
        .Set("valid_in", 1)
        .Set("ready_out", 0)
        .Set("init_acc_in", 50);

    // Run until done.
    tb.WaitFor("valid_out");
    tb.ExpectEq("ready_in", 0)
        .ExpectEq("valid_out", 1)
        .ExpectEq("counted_for_6_out", 56);

    // Ready out set, transition to ready state again.
    tb.Set("reset", 0)
        .Set("valid_in", 0)
        .Set("ready_out", 1)
        .Set("init_acc_in", 0);
    tb.NextCycle();

    // Ready.
    tb.ExpectEq("ready_in", 1).ExpectEq("valid_out", 0);

    XLS_ASSERT_OK(tb.Run());
  }
}

// TODO(jbaileyhandle): Test module reset (active high and active low).

INSTANTIATE_TEST_SUITE_P(SequentialGeneratorTestInstantiation,
                         SequentialGeneratorTest,
                         testing::ValuesIn(kDefaultSimulationTargets),
                         ParameterizedTestName<SequentialGeneratorTest>);

}  // namespace
}  // namespace verilog
}  // namespace xls
