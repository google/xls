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

#include "xls/codegen/pipeline_generator.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/strings/substitute.h"
#include "xls/codegen/module_signature.h"
#include "xls/codegen/module_signature.pb.h"
#include "xls/common/status/matchers.h"
#include "xls/estimators/delay_model/delay_estimator.h"
#include "xls/ir/bits.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/ir_parser.h"
#include "xls/ir/op.h"
#include "xls/ir/package.h"
#include "xls/ir/value.h"
#include "xls/scheduling/pipeline_schedule.h"
#include "xls/scheduling/run_pipeline_schedule.h"
#include "xls/scheduling/scheduling_options.h"
#include "xls/simulation/module_simulator.h"
#include "xls/simulation/module_testbench.h"
#include "xls/simulation/module_testbench_thread.h"
#include "xls/simulation/verilog_test_base.h"

namespace xls {
namespace verilog {
namespace {

using ::absl_testing::IsOkAndHolds;
using ::absl_testing::StatusIs;
using ::testing::ContainsRegex;
using ::testing::HasSubstr;
using ::testing::Not;

constexpr char kTestName[] = "pipeline_generator_test";
constexpr char kTestdataPath[] = "xls/codegen/testdata";

class TestDelayEstimator : public DelayEstimator {
 public:
  TestDelayEstimator() : DelayEstimator("test") {}

  absl::StatusOr<int64_t> GetOperationDelayInPs(Node* node) const override {
    switch (node->op()) {
      case Op::kParam:
      case Op::kStateRead:
      case Op::kLiteral:
      case Op::kBitSlice:
      case Op::kConcat:
        return 0;
      default:
        return 1;
    }
  }
};

absl::StatusOr<ModuleSignature> StripResetFromSignature(
    const ModuleSignature& signature) {
  ModuleSignatureProto proto = signature.proto();
  proto.clear_reset();
  PortProto* reset_as_data_port = proto.add_data_ports();
  reset_as_data_port->set_direction(PORT_DIRECTION_INPUT);
  reset_as_data_port->set_name(signature.proto().reset().name());
  reset_as_data_port->set_width(1);
  return ModuleSignature::FromProto(proto);
}

class PipelineGeneratorTest : public VerilogTestBase {};

TEST_P(PipelineGeneratorTest, TrivialFunction) {
  Package package(TestBaseName());
  FunctionBuilder fb(TestBaseName(), &package);
  fb.Param("x", package.GetBitsType(8));
  XLS_ASSERT_OK_AND_ASSIGN(Function * func, fb.Build());

  XLS_ASSERT_OK_AND_ASSIGN(
      PipelineSchedule schedule,
      RunPipelineSchedule(func, TestDelayEstimator(),
                          SchedulingOptions().clock_period_ps(1)));

  XLS_ASSERT_OK_AND_ASSIGN(
      ModuleGeneratorResult result,
      ToPipelineModuleText(
          schedule, func,
          BuildPipelineOptions().use_system_verilog(UseSystemVerilog())));

  ExpectVerilogEqualToGoldenFile(GoldenFilePath(kTestName, kTestdataPath),
                                 result.verilog_text);

  EXPECT_EQ(result.signature.proto().pipeline().initiation_interval(), 1);
  EXPECT_EQ(result.signature.proto().pipeline().latency(), 2);
}

TEST_P(PipelineGeneratorTest, ReturnLiteral) {
  Package package(TestBaseName());
  FunctionBuilder fb(TestBaseName(), &package);
  fb.Literal(UBits(42, 32));
  XLS_ASSERT_OK_AND_ASSIGN(Function * func, fb.Build());

  XLS_ASSERT_OK_AND_ASSIGN(
      PipelineSchedule schedule,
      RunPipelineSchedule(func, TestDelayEstimator(),
                          SchedulingOptions().pipeline_stages(5)));

  XLS_ASSERT_OK_AND_ASSIGN(
      ModuleGeneratorResult result,
      ToPipelineModuleText(schedule, func,
                           BuildPipelineOptions()
                               .flop_inputs(false)
                               .flop_outputs(false)
                               .use_system_verilog(UseSystemVerilog())));

  EXPECT_EQ(result.signature.proto().pipeline().initiation_interval(), 1);
  EXPECT_EQ(result.signature.proto().pipeline().latency(), 4);

  ExpectVerilogEqualToGoldenFile(GoldenFilePath(kTestName, kTestdataPath),
                                 result.verilog_text);

  ModuleSimulator simulator =
      NewModuleSimulator(result.verilog_text, result.signature);
  EXPECT_THAT(simulator.RunAndReturnSingleOutput(ModuleSimulator::BitsMap()),
              IsOkAndHolds(UBits(42, 32)));
}

TEST_P(PipelineGeneratorTest, ReturnTupleLiteral) {
  Package package(TestBaseName());
  FunctionBuilder fb(TestBaseName(), &package);
  fb.Literal(Value::Tuple({Value(UBits(0, 1)), Value(UBits(123, 32))}));
  XLS_ASSERT_OK_AND_ASSIGN(Function * func, fb.Build());

  XLS_ASSERT_OK_AND_ASSIGN(
      PipelineSchedule schedule,
      RunPipelineSchedule(func, TestDelayEstimator(),
                          SchedulingOptions().pipeline_stages(5)));

  XLS_ASSERT_OK_AND_ASSIGN(
      ModuleGeneratorResult result,
      ToPipelineModuleText(schedule, func,
                           BuildPipelineOptions()
                               .flop_inputs(false)
                               .flop_outputs(false)
                               .use_system_verilog(UseSystemVerilog())));

  ExpectVerilogEqualToGoldenFile(GoldenFilePath(kTestName, kTestdataPath),
                                 result.verilog_text);

  ModuleSimulator simulator =
      NewModuleSimulator(result.verilog_text, result.signature);
  EXPECT_THAT(
      simulator.RunFunction(absl::flat_hash_map<std::string, Value>()),
      IsOkAndHolds(Value::Tuple({Value(UBits(0, 1)), Value(UBits(123, 32))})));
}

TEST_P(PipelineGeneratorTest, ReturnEmptyTuple) {
  Package package(TestBaseName());
  FunctionBuilder fb(TestBaseName(), &package);
  fb.Literal(Value::Tuple({}));
  XLS_ASSERT_OK_AND_ASSIGN(Function * func, fb.Build());

  XLS_ASSERT_OK_AND_ASSIGN(
      PipelineSchedule schedule,
      RunPipelineSchedule(func, TestDelayEstimator(),
                          SchedulingOptions().pipeline_stages(5)));

  XLS_ASSERT_OK_AND_ASSIGN(
      ModuleGeneratorResult result,
      ToPipelineModuleText(
          schedule, func,
          BuildPipelineOptions().use_system_verilog(UseSystemVerilog())));

  ModuleSimulator simulator =
      NewModuleSimulator(result.verilog_text, result.signature);
  EXPECT_THAT(simulator.RunFunction(absl::flat_hash_map<std::string, Value>()),
              IsOkAndHolds(Value::Tuple({})));
}

TEST_P(PipelineGeneratorTest, NestedEmptyTuple) {
  Package package(TestBaseName());
  FunctionBuilder fb(TestBaseName(), &package);
  fb.Tuple({fb.Literal(Value::Tuple({}))});
  XLS_ASSERT_OK_AND_ASSIGN(Function * func, fb.Build());

  XLS_ASSERT_OK_AND_ASSIGN(
      PipelineSchedule schedule,
      RunPipelineSchedule(func, TestDelayEstimator(),
                          SchedulingOptions().pipeline_stages(5)));

  XLS_ASSERT_OK_AND_ASSIGN(
      ModuleGeneratorResult result,
      ToPipelineModuleText(
          schedule, func,
          BuildPipelineOptions().use_system_verilog(UseSystemVerilog())));

  ModuleSimulator simulator =
      NewModuleSimulator(result.verilog_text, result.signature);
  EXPECT_THAT(simulator.RunFunction(absl::flat_hash_map<std::string, Value>()),
              IsOkAndHolds(Value::Tuple({Value::Tuple({})})));
}

TEST_P(PipelineGeneratorTest, TakesEmptyTuple) {
  Package package(TestBaseName());
  FunctionBuilder fb(TestBaseName(), &package);
  Type* u8 = package.GetBitsType(8);
  auto a = fb.Param("a", u8);
  fb.Param("b", package.GetTupleType({}));
  auto c = fb.Param("c", u8);
  fb.Add(a, c);
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());

  XLS_ASSERT_OK_AND_ASSIGN(
      PipelineSchedule schedule,
      RunPipelineSchedule(f, TestDelayEstimator(),
                          SchedulingOptions().pipeline_stages(5)));
  XLS_ASSERT_OK_AND_ASSIGN(
      ModuleGeneratorResult result,
      ToPipelineModuleText(
          schedule, f,
          BuildPipelineOptions().use_system_verilog(UseSystemVerilog())));

  ModuleSimulator simulator =
      NewModuleSimulator(result.verilog_text, result.signature);
  EXPECT_THAT(simulator.RunFunction({{"a", Value(UBits(42, 8))},
                                     {"b", Value::Tuple({})},
                                     {"c", Value(UBits(100, 8))}}),
              IsOkAndHolds(Value(UBits(142, 8))));
}

TEST_P(PipelineGeneratorTest, PassesEmptyTuple) {
  Package package(TestBaseName());
  FunctionBuilder fb(TestBaseName(), &package);
  fb.Param("x", package.GetTupleType({}));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());

  XLS_ASSERT_OK_AND_ASSIGN(
      PipelineSchedule schedule,
      RunPipelineSchedule(f, TestDelayEstimator(),
                          SchedulingOptions().pipeline_stages(5)));
  XLS_ASSERT_OK_AND_ASSIGN(
      ModuleGeneratorResult result,
      ToPipelineModuleText(
          schedule, f,
          BuildPipelineOptions().use_system_verilog(UseSystemVerilog())));

  ModuleSimulator simulator =
      NewModuleSimulator(result.verilog_text, result.signature);
  EXPECT_THAT(simulator.RunFunction({{"x", Value::Tuple({})}}),
              IsOkAndHolds(Value::Tuple({})));
}

TEST_P(PipelineGeneratorTest, ReturnArrayLiteral) {
  Package package(TestBaseName());
  FunctionBuilder fb(TestBaseName(), &package);
  fb.Literal(Value::ArrayOrDie({Value(UBits(0, 1)), Value(UBits(1, 1))}));
  XLS_ASSERT_OK_AND_ASSIGN(Function * func, fb.Build());

  XLS_ASSERT_OK_AND_ASSIGN(
      PipelineSchedule schedule,
      RunPipelineSchedule(func, TestDelayEstimator(),
                          SchedulingOptions().pipeline_stages(5)));

  XLS_ASSERT_OK_AND_ASSIGN(
      ModuleGeneratorResult result,
      ToPipelineModuleText(schedule, func,
                           BuildPipelineOptions()
                               .flop_inputs(false)
                               .flop_outputs(false)
                               .use_system_verilog(UseSystemVerilog())));

  ExpectVerilogEqualToGoldenFile(GoldenFilePath(kTestName, kTestdataPath),
                                 result.verilog_text);

  ModuleSimulator simulator =
      NewModuleSimulator(result.verilog_text, result.signature);
  EXPECT_THAT(simulator.RunFunction(absl::flat_hash_map<std::string, Value>()),
              IsOkAndHolds(
                  Value::ArrayOrDie({Value(UBits(0, 1)), Value(UBits(1, 1))})));
}

TEST_P(PipelineGeneratorTest, SingleNegate) {
  Package package(TestBaseName());
  FunctionBuilder fb("negate", &package);
  auto x = fb.Param("x", package.GetBitsType(8));
  fb.Negate(x);

  XLS_ASSERT_OK_AND_ASSIGN(Function * func, fb.Build());

  XLS_ASSERT_OK_AND_ASSIGN(
      PipelineSchedule schedule,
      RunPipelineSchedule(func, TestDelayEstimator(),
                          SchedulingOptions().clock_period_ps(40)));

  XLS_ASSERT_OK_AND_ASSIGN(
      ModuleGeneratorResult result,
      ToPipelineModuleText(
          schedule, func,
          BuildPipelineOptions().use_system_verilog(UseSystemVerilog())));

  ExpectVerilogEqualToGoldenFile(GoldenFilePath(kTestName, kTestdataPath),
                                 result.verilog_text);
}

TEST_P(PipelineGeneratorTest, PassThroughArray) {
  Package package(TestBaseName());
  FunctionBuilder fb(TestBaseName(), &package);
  fb.Param("x", package.GetArrayType(3, package.GetBitsType(8)));
  XLS_ASSERT_OK_AND_ASSIGN(Function * func, fb.Build());

  XLS_ASSERT_OK_AND_ASSIGN(
      PipelineSchedule schedule,
      RunPipelineSchedule(func, TestDelayEstimator(),
                          SchedulingOptions().pipeline_stages(3)));

  XLS_ASSERT_OK_AND_ASSIGN(
      ModuleGeneratorResult result,
      ToPipelineModuleText(
          schedule, func,
          BuildPipelineOptions().use_system_verilog(UseSystemVerilog())));

  ExpectVerilogEqualToGoldenFile(GoldenFilePath(kTestName, kTestdataPath),
                                 result.verilog_text);
  ModuleSimulator simulator =
      NewModuleSimulator(result.verilog_text, result.signature);
  Value array = Value::ArrayOrDie(
      {Value(UBits(123, 8)), Value(UBits(42, 8)), Value(UBits(33, 8))});
  EXPECT_THAT(simulator.RunFunction({{"x", array}}), IsOkAndHolds(array));
}

TEST_P(PipelineGeneratorTest, TupleOfArrays) {
  // Function takes a tuple of arrays and produces another tuple with the
  // elements interchanged.
  Package package(TestBaseName());
  FunctionBuilder fb(TestBaseName(), &package);
  BValue x = fb.Param(
      "x",
      package.GetTupleType({package.GetArrayType(3, package.GetBitsType(8)),
                            package.GetArrayType(2, package.GetBitsType(16))}));
  fb.Tuple({fb.TupleIndex(x, 1), fb.TupleIndex(x, 0)});
  XLS_ASSERT_OK_AND_ASSIGN(Function * func, fb.Build());

  XLS_ASSERT_OK_AND_ASSIGN(
      PipelineSchedule schedule,
      RunPipelineSchedule(func, TestDelayEstimator(),
                          SchedulingOptions().pipeline_stages(3)));

  XLS_ASSERT_OK_AND_ASSIGN(
      ModuleGeneratorResult result,
      ToPipelineModuleText(
          schedule, func,
          BuildPipelineOptions().use_system_verilog(UseSystemVerilog())));

  ExpectVerilogEqualToGoldenFile(GoldenFilePath(kTestName, kTestdataPath),
                                 result.verilog_text);
  ModuleSimulator simulator =
      NewModuleSimulator(result.verilog_text, result.signature);
  Value array_0 = Value::ArrayOrDie(
      {Value(UBits(123, 8)), Value(UBits(42, 8)), Value(UBits(33, 8))});
  Value array_1 = Value::ArrayOrDie({Value(UBits(4, 16)), Value(UBits(5, 16))});
  EXPECT_THAT(simulator.RunFunction({{"x", Value::Tuple({array_0, array_1})}}),
              IsOkAndHolds(Value::Tuple({array_1, array_0})));
}

TEST_P(PipelineGeneratorTest, MultidimensionalArray) {
  Package package(TestBaseName());
  FunctionBuilder fb(TestBaseName(), &package);
  BValue a = fb.Param(
      "a",
      package.GetArrayType(2, package.GetArrayType(3, package.GetBitsType(8))));
  BValue index = fb.Param("index", package.GetBitsType(16));
  fb.ArrayIndex(a, {index});
  XLS_ASSERT_OK_AND_ASSIGN(Function * func, fb.Build());

  XLS_ASSERT_OK_AND_ASSIGN(
      PipelineSchedule schedule,
      RunPipelineSchedule(func, TestDelayEstimator(),
                          SchedulingOptions().pipeline_stages(3)));

  XLS_ASSERT_OK_AND_ASSIGN(
      ModuleGeneratorResult result,
      ToPipelineModuleText(
          schedule, func,
          BuildPipelineOptions().use_system_verilog(UseSystemVerilog())));

  ExpectVerilogEqualToGoldenFile(GoldenFilePath(kTestName, kTestdataPath),
                                 result.verilog_text);

  ModuleSimulator simulator =
      NewModuleSimulator(result.verilog_text, result.signature);
  Value inner_array_0 = Value::ArrayOrDie(
      {Value(UBits(123, 8)), Value(UBits(42, 8)), Value(UBits(33, 8))});
  Value inner_array_1 = Value::ArrayOrDie(
      {Value(UBits(44, 8)), Value(UBits(22, 8)), Value(UBits(11, 8))});
  Value array = Value::ArrayOrDie({inner_array_0, inner_array_1});
  EXPECT_THAT(
      simulator.RunFunction({{"a", array}, {"index", Value(UBits(0, 16))}}),
      IsOkAndHolds(inner_array_0));
  EXPECT_THAT(
      simulator.RunFunction({{"a", array}, {"index", Value(UBits(1, 16))}}),
      IsOkAndHolds(inner_array_1));
}

TEST_P(PipelineGeneratorTest, TreeOfAdds) {
  Package package(TestBaseName());
  FunctionBuilder fb("x_plus_y_plus_z", &package);
  Type* u32 = package.GetBitsType(32);
  auto a = fb.Param("a", u32);
  auto b = fb.Param("b", u32);
  auto c = fb.Param("c", u32);
  auto d = fb.Param("d", u32);
  auto e = fb.Param("e", u32);
  auto out = a + b + c + d + e;

  XLS_ASSERT_OK_AND_ASSIGN(Function * func, fb.BuildWithReturnValue(out));

  XLS_ASSERT_OK_AND_ASSIGN(
      int64_t add_delay_in_ps,
      TestDelayEstimator().GetOperationDelayInPs(out.node()));
  XLS_ASSERT_OK_AND_ASSIGN(
      PipelineSchedule schedule,
      RunPipelineSchedule(
          func, TestDelayEstimator(),
          SchedulingOptions().clock_period_ps(add_delay_in_ps * 2)));

  XLS_ASSERT_OK_AND_ASSIGN(
      ModuleGeneratorResult result,
      ToPipelineModuleText(
          schedule, func,
          BuildPipelineOptions().use_system_verilog(UseSystemVerilog())));

  ExpectVerilogEqualToGoldenFile(GoldenFilePath(kTestName, kTestdataPath),
                                 result.verilog_text);

  EXPECT_EQ(result.signature.proto().pipeline().initiation_interval(), 1);
  EXPECT_EQ(result.signature.proto().pipeline().latency(), 3);
  EXPECT_EQ(result.signature.data_inputs().size(), 5);
  EXPECT_EQ(result.signature.data_outputs().size(), 1);
  for (const PortProto& input_port : result.signature.data_inputs()) {
    XLS_ASSERT_OK_AND_ASSIGN(Type * input_type,
                             package.GetTypeFromProto(input_port.type()));
    EXPECT_EQ(input_type->ToString(), "bits[32]");
  }
  XLS_ASSERT_OK_AND_ASSIGN(
      Type * output_type,
      package.GetTypeFromProto(result.signature.data_inputs()[0].type()));
  EXPECT_EQ(output_type->ToString(), "bits[32]");
}

TEST_P(PipelineGeneratorTest, BigExpressionInOneStage) {
  Package package(TestBaseName());
  FunctionBuilder fb("x_plus_y_plus_z", &package);
  Type* u32 = package.GetBitsType(32);
  auto a = fb.Param("a", u32);
  auto b = fb.Param("b", u32);
  auto c = fb.Param("c", u32);
  auto d = fb.Param("d", u32);
  auto tmp0 = a + b;
  auto tmp1 = fb.Xor(a, tmp0);
  auto tmp2 = fb.Not(tmp0) + c;
  auto tmp3 = tmp1 - tmp2;
  auto tmp4 = d + tmp1;
  auto out = tmp3 - tmp4;

  XLS_ASSERT_OK_AND_ASSIGN(Function * func, fb.BuildWithReturnValue(out));

  XLS_ASSERT_OK_AND_ASSIGN(
      PipelineSchedule schedule,
      RunPipelineSchedule(func, TestDelayEstimator(),
                          SchedulingOptions().clock_period_ps(4000)));

  XLS_ASSERT_OK_AND_ASSIGN(
      ModuleGeneratorResult result,
      ToPipelineModuleText(
          schedule, func,
          BuildPipelineOptions().use_system_verilog(UseSystemVerilog())));

  ExpectVerilogEqualToGoldenFile(GoldenFilePath(kTestName, kTestdataPath),
                                 result.verilog_text);

  EXPECT_EQ(result.signature.proto().pipeline().initiation_interval(), 1);
  EXPECT_EQ(result.signature.proto().pipeline().latency(), 2);
}

TEST_P(PipelineGeneratorTest, IdentityOfMul) {
  Package package(TestBaseName());
  FunctionBuilder fb("identity_of_mul", &package);
  auto x = fb.Param("x", package.GetBitsType(8));
  auto y = fb.Param("y", package.GetBitsType(8));
  fb.Identity(fb.UMul(x, y));

  XLS_ASSERT_OK_AND_ASSIGN(Function * func, fb.Build());

  XLS_ASSERT_OK_AND_ASSIGN(
      PipelineSchedule schedule,
      RunPipelineSchedule(func, TestDelayEstimator(),
                          SchedulingOptions().clock_period_ps(50)));

  XLS_ASSERT_OK_AND_ASSIGN(
      ModuleGeneratorResult result,
      ToPipelineModuleText(
          schedule, func,
          BuildPipelineOptions().use_system_verilog(UseSystemVerilog())));

  ExpectVerilogEqualToGoldenFile(GoldenFilePath(kTestName, kTestdataPath),
                                 result.verilog_text);
}

TEST_P(PipelineGeneratorTest, RequiredNamedIntermediates) {
  // Tests that nodes (such as bit slice) which require named intermediates are
  // properly emitted.
  Package package(TestBaseName());
  FunctionBuilder fb(TestBaseName(), &package);
  auto x = fb.Param("x", package.GetBitsType(8));
  auto y = fb.Param("y", package.GetBitsType(8));
  fb.BitSlice(x + y, /*start=*/0, /*width=*/5);

  XLS_ASSERT_OK_AND_ASSIGN(Function * func, fb.Build());

  // Choose a large clock period such that all nodes are scheduled in the same
  // stage.
  XLS_ASSERT_OK_AND_ASSIGN(
      PipelineSchedule schedule,
      RunPipelineSchedule(func, TestDelayEstimator(),
                          SchedulingOptions().clock_period_ps(400)));

  XLS_ASSERT_OK_AND_ASSIGN(
      ModuleGeneratorResult result,
      ToPipelineModuleText(
          schedule, func,
          BuildPipelineOptions().use_system_verilog(UseSystemVerilog())));

  ExpectVerilogEqualToGoldenFile(GoldenFilePath(kTestName, kTestdataPath),
                                 result.verilog_text);
}

TEST_P(PipelineGeneratorTest, BinarySelect) {
  // Tests that nodes (such as bit slice) which require named intermediates are
  // properly emitted.
  Package package(TestBaseName());
  FunctionBuilder fb(TestBaseName(), &package);
  auto s = fb.Param("s", package.GetBitsType(1));
  auto x = fb.Param("x", package.GetBitsType(8));
  auto y = fb.Param("y", package.GetBitsType(8));
  fb.Select(s, x, y);

  XLS_ASSERT_OK_AND_ASSIGN(Function * func, fb.Build());

  // Choose a large clock period such that all nodes are scheduled in the same
  // stage.
  XLS_ASSERT_OK_AND_ASSIGN(
      PipelineSchedule schedule,
      RunPipelineSchedule(func, TestDelayEstimator(),
                          SchedulingOptions().clock_period_ps(400)));

  XLS_ASSERT_OK_AND_ASSIGN(
      ModuleGeneratorResult result,
      ToPipelineModuleText(
          schedule, func,
          BuildPipelineOptions().use_system_verilog(UseSystemVerilog())));

  ExpectVerilogEqualToGoldenFile(GoldenFilePath(kTestName, kTestdataPath),
                                 result.verilog_text);
}

TEST_P(PipelineGeneratorTest, TwoBitSelector) {
  Package package(TestBaseName());
  FunctionBuilder fb(TestBaseName(), &package);
  auto s = fb.Param("s", package.GetBitsType(2));
  auto x = fb.Param("x", package.GetBitsType(8));
  auto y = fb.Param("y", package.GetBitsType(8));
  auto z = fb.Param("z", package.GetBitsType(8));
  fb.Select(s, {x, y}, z);

  XLS_ASSERT_OK_AND_ASSIGN(Function * func, fb.Build());

  // Choose a large clock period such that all nodes are scheduled in the same
  // stage.
  XLS_ASSERT_OK_AND_ASSIGN(
      PipelineSchedule schedule,
      RunPipelineSchedule(func, TestDelayEstimator(),
                          SchedulingOptions().clock_period_ps(400)));

  XLS_ASSERT_OK_AND_ASSIGN(
      ModuleGeneratorResult result,
      ToPipelineModuleText(
          schedule, func,
          BuildPipelineOptions().use_system_verilog(UseSystemVerilog())));

  ExpectVerilogEqualToGoldenFile(GoldenFilePath(kTestName, kTestdataPath),
                                 result.verilog_text);
}

TEST_P(PipelineGeneratorTest, TwoBitSelectorAllCasesPopulated) {
  Package package(TestBaseName());
  FunctionBuilder fb(TestBaseName(), &package);
  auto s = fb.Param("s", package.GetBitsType(2));
  auto x = fb.Param("x", package.GetBitsType(8));
  auto y = fb.Param("y", package.GetBitsType(8));
  auto z = fb.Param("z", package.GetBitsType(8));
  auto a = fb.Param("a", package.GetBitsType(8));
  fb.Select(s, {x, y, z, a}, /*default_value=*/std::nullopt);

  XLS_ASSERT_OK_AND_ASSIGN(Function * func, fb.Build());

  // Choose a large clock period such that all nodes are scheduled in the same
  // stage.
  XLS_ASSERT_OK_AND_ASSIGN(
      PipelineSchedule schedule,
      RunPipelineSchedule(func, TestDelayEstimator(),
                          SchedulingOptions().clock_period_ps(400)));

  XLS_ASSERT_OK_AND_ASSIGN(
      ModuleGeneratorResult result,
      ToPipelineModuleText(
          schedule, func,
          BuildPipelineOptions().use_system_verilog(UseSystemVerilog())));

  ExpectVerilogEqualToGoldenFile(GoldenFilePath(kTestName, kTestdataPath),
                                 result.verilog_text);
}

TEST_P(PipelineGeneratorTest, ValidSignalWithoutReset) {
  Package package(TestBaseName());
  FunctionBuilder fb(TestBaseName(), &package);
  auto s = fb.Param("s", package.GetBitsType(2));
  auto x = fb.Param("x", package.GetBitsType(8));
  auto y = fb.Param("y", package.GetBitsType(8));
  auto z = fb.Param("z", package.GetBitsType(8));
  auto a = fb.Param("a", package.GetBitsType(8));
  fb.Select(s, {x, y, z, a}, /*default_value=*/std::nullopt);

  XLS_ASSERT_OK_AND_ASSIGN(Function * func, fb.Build());

  // Choose a large clock period such that all nodes are scheduled in the same
  // stage.
  XLS_ASSERT_OK_AND_ASSIGN(
      PipelineSchedule schedule,
      RunPipelineSchedule(func, TestDelayEstimator(),
                          SchedulingOptions().clock_period_ps(400)));

  EXPECT_THAT(
      ToPipelineModuleText(schedule, func,
                           BuildPipelineOptions()
                               .valid_control("in_valid", "out_valid")
                               .use_system_verilog(UseSystemVerilog())),
      StatusIs(absl::StatusCode::kInternal,
               testing::HasSubstr("has valid signal output but no reset")));
}

TEST_P(PipelineGeneratorTest, ValidSignalWithoutResetAndWithoutOutputValid) {
  Package package(TestBaseName());
  FunctionBuilder fb(TestBaseName(), &package);
  auto s = fb.Param("s", package.GetBitsType(2));
  auto x = fb.Param("x", package.GetBitsType(8));
  auto y = fb.Param("y", package.GetBitsType(8));
  auto z = fb.Param("z", package.GetBitsType(8));
  auto a = fb.Param("a", package.GetBitsType(8));
  fb.Select(s, {x, y, z, a}, /*default_value=*/std::nullopt);

  XLS_ASSERT_OK_AND_ASSIGN(Function * func, fb.Build());

  // Choose a large clock period such that all nodes are scheduled in the same
  // stage.
  XLS_ASSERT_OK_AND_ASSIGN(
      PipelineSchedule schedule,
      RunPipelineSchedule(func, TestDelayEstimator(),
                          SchedulingOptions().pipeline_stages(3)));

  XLS_ASSERT_OK_AND_ASSIGN(
      ModuleGeneratorResult result,
      ToPipelineModuleText(schedule, func,
                           BuildPipelineOptions()
                               .valid_control("in_valid", std::nullopt)
                               .use_system_verilog(UseSystemVerilog())));

  ExpectVerilogEqualToGoldenFile(GoldenFilePath(kTestName, kTestdataPath),
                                 result.verilog_text);
}

TEST_P(PipelineGeneratorTest, ValidPipelineControlWithSimulation) {
  // Verify the valid signalling works as expected by driving the module with a
  // ModuleTestBench.
  Package package(TestBaseName());
  FunctionBuilder fb(TestBaseName(), &package);
  auto x = fb.Param("x", package.GetBitsType(64));
  auto y = fb.Param("y", package.GetBitsType(64));
  auto z = fb.Param("z", package.GetBitsType(64));
  auto a = fb.UMul(x, y);
  fb.UMul(a, z);

  XLS_ASSERT_OK_AND_ASSIGN(Function * func, fb.Build());

  XLS_ASSERT_OK_AND_ASSIGN(
      PipelineSchedule schedule,
      RunPipelineSchedule(func, TestDelayEstimator(),
                          SchedulingOptions().pipeline_stages(5)));

  ResetProto reset;
  reset.set_name("rst");
  reset.set_asynchronous(false);
  reset.set_active_low(false);
  reset.set_reset_data_path(false);
  XLS_ASSERT_OK_AND_ASSIGN(
      ModuleGeneratorResult result,
      ToPipelineModuleText(
          schedule, func,
          BuildPipelineOptions()
              .valid_control("in_valid", "out_valid")
              .use_system_verilog(UseSystemVerilog())
              .reset(reset.name(), reset.asynchronous(), reset.active_low(),
                     reset.reset_data_path())));

  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<ModuleTestbench> tb,
      ModuleTestbench::CreateFromVerilogText(result.verilog_text, GetFileType(),
                                             result.signature, GetSimulator()));
  XLS_ASSERT_OK_AND_ASSIGN(
      ModuleTestbenchThread * tbt,
      tb->CreateThreadDrivingAllInputs("main", /*default_value=*/ZeroOrX::kX));
  SequentialBlock& seq = tbt->MainBlock();
  seq.Set("in_valid", 0);
  seq.AtEndOfCycleWhenNotX("out_valid").ExpectEq("out_valid", 0);

  // Send in a valid value on one cycle.
  seq.Set("in_valid", 1).Set("x", 2).Set("y", 3).Set("z", 4);
  seq.NextCycle();
  // Then don't send any more valid values.
  seq.Set("in_valid", 0);

  // Wait until the output goes valid.
  const int kExpected = 2 * 3 * 4;
  seq.AtEndOfCycleWhen("out_valid")
      .ExpectEq("out_valid", 1)
      .ExpectEq("out", kExpected);

  // Output will be invalid in all subsequent cycles.
  seq.AtEndOfCycle().ExpectEq("out_valid", 0);
  seq.AtEndOfCycle().ExpectEq("out_valid", 0);

  // Now change the input and observe that the output never changes (because we
  // don't correspondingly set input_valid).
  seq.Set("z", 7);
  int64_t latency = result.signature.proto().pipeline().latency();
  ASSERT_GT(latency, 0);
  for (int64_t i = 0; i < 2 * latency; ++i) {
    seq.AtEndOfCycle().ExpectEq("out", kExpected).ExpectEq("out_valid", 0);
  }

  XLS_ASSERT_OK(tb->Run());
}

TEST_P(PipelineGeneratorTest, ValidSignalWithReset) {
  // Test with both active low and active high signals.
  for (bool active_low : {false, true}) {
    Package package(TestBaseName());
    FunctionBuilder fb(TestBaseName(), &package);
    auto x = fb.Param("x", package.GetBitsType(64));
    auto y = fb.Param("y", package.GetBitsType(64));
    auto z = fb.Param("z", package.GetBitsType(64));
    auto a = fb.UMul(x, y);
    fb.UMul(a, z);

    XLS_ASSERT_OK_AND_ASSIGN(Function * func, fb.Build());

    // Choose a large clock period such that all nodes are scheduled in the same
    // stage.
    XLS_ASSERT_OK_AND_ASSIGN(
        PipelineSchedule schedule,
        RunPipelineSchedule(func, TestDelayEstimator(),
                            SchedulingOptions().clock_period_ps(400)));

    const int64_t kAssertReset = active_low ? 0 : 1;
    const int64_t kDeassertReset = active_low ? 1 : 0;
    const std::string kResetSignal = active_low ? "the_rst_n" : "the_rst";

    ResetProto reset;
    reset.set_name(kResetSignal);
    reset.set_asynchronous(false);
    reset.set_active_low(active_low);
    reset.set_reset_data_path(false);
    XLS_ASSERT_OK_AND_ASSIGN(
        ModuleGeneratorResult result,
        ToPipelineModuleText(
            schedule, func,
            BuildPipelineOptions()
                .valid_control("in_valid", "out_valid")
                .use_system_verilog(UseSystemVerilog())
                .reset(reset.name(), reset.asynchronous(), reset.active_low(),
                       reset.reset_data_path())));
    // The reset signal is synchronous so the edge sensitivity ("posedge foo" or
    // "negedge foo" should only contain the clock, i.e. "posedge clk").
    EXPECT_THAT(result.verilog_text, ContainsRegex(R"(edge\s+clk)"));
    EXPECT_THAT(
        result.verilog_text,
        Not(ContainsRegex(absl::StrFormat(R"(edge\s+%s)", kResetSignal))));
    // Verilog should have an "if (rst)" of "if (!rst_n)"  conditional.
    EXPECT_THAT(result.verilog_text,
                HasSubstr(absl::StrFormat("if (%s%s)", active_low ? "!" : "",
                                          kResetSignal)));

    EXPECT_EQ(result.signature.proto().reset().name(), kResetSignal);
    EXPECT_FALSE(result.signature.proto().reset().asynchronous());
    EXPECT_EQ(result.signature.proto().reset().active_low(), active_low);
    EXPECT_FALSE(result.signature.proto().reset().reset_data_path());

    // We directly manipulate the reset line so strip reset from the signature
    // and add a regular port so the testbench does not drive the reset line.
    XLS_ASSERT_OK_AND_ASSIGN(ModuleSignature resetless_signature,
                             StripResetFromSignature(result.signature));

    XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<ModuleTestbench> tb,
                             ModuleTestbench::CreateFromVerilogText(
                                 result.verilog_text, GetFileType(),
                                 resetless_signature, GetSimulator()));
    XLS_ASSERT_OK_AND_ASSIGN(ModuleTestbenchThread * tbt,
                             tb->CreateThreadDrivingAllInputs(
                                 "main", /*default_value=*/ZeroOrX::kX));
    SequentialBlock& seq = tbt->MainBlock();
    // One cycle after reset the output control signal should be zero.
    seq.Set(kResetSignal, kAssertReset).Set("in_valid", 0);
    seq.NextCycle();
    seq.AtEndOfCycle().ExpectEq("out_valid", 0);

    // Even with in_valid one, out_valid should never be one because reset is
    // asserted.
    seq.Set("in_valid", 1);
    seq.AdvanceNCycles(100);
    seq.AtEndOfCycle().ExpectEq("out_valid", 0);

    // Deassert rst and set inputs.
    seq.Set(kResetSignal, kDeassertReset).Set("x", 2).Set("y", 3).Set("z", 4);

    // Wait until the output goes valid.
    const int kExpected = 2 * 3 * 4;
    seq.AtEndOfCycleWhen("out_valid")
        .ExpectEq("out_valid", 1)
        .ExpectEq("out", kExpected);

    // Output will remain valid in subsequent cycles.
    seq.AtEndOfCycle().ExpectEq("out_valid", 1);
    seq.AtEndOfCycle().ExpectEq("out_valid", 1);

    // Assert reset and verify out_valid is always zero.
    seq.Set(kResetSignal, kAssertReset);
    seq.NextCycle();
    seq.AtEndOfCycle().ExpectEq("out_valid", 0);
    seq.AtEndOfCycle().ExpectEq("out_valid", 0);

    // Deassert reset and in_valid and change the input and observe that the
    // output never changes (because we don't correspondingly set input_valid).
    seq.Set("z", 7).Set(kResetSignal, kDeassertReset).Set("in_valid", 0);
    int64_t latency = result.signature.proto().pipeline().latency();
    ASSERT_GT(latency, 0);
    for (int64_t i = 0; i < 2 * latency; ++i) {
      seq.AtEndOfCycle().ExpectEq("out", kExpected).ExpectEq("out_valid", 0);
    }

    XLS_ASSERT_OK(tb->Run());
  }
}

TEST_P(PipelineGeneratorTest, CustomModuleName) {
  Package package(TestBaseName());
  FunctionBuilder fb(TestBaseName(), &package);
  fb.Param("x", package.GetBitsType(8));
  XLS_ASSERT_OK_AND_ASSIGN(Function * func, fb.Build());

  XLS_ASSERT_OK_AND_ASSIGN(
      PipelineSchedule schedule,
      RunPipelineSchedule(func, TestDelayEstimator(),
                          SchedulingOptions().clock_period_ps(40)));

  XLS_ASSERT_OK_AND_ASSIGN(
      ModuleGeneratorResult result,
      ToPipelineModuleText(
          schedule, func,
          BuildPipelineOptions().module_name("foobar").use_system_verilog(
              UseSystemVerilog())));

  EXPECT_THAT(result.verilog_text, HasSubstr("module foobar("));
}

TEST_P(PipelineGeneratorTest, AddNegateFlopInputsAndOutputs) {
  Package package(TestBaseName());
  FunctionBuilder fb(TestBaseName(), &package);
  auto x = fb.Param("x", package.GetBitsType(8));
  auto y = fb.Param("y", package.GetBitsType(8));
  fb.Negate(fb.Add(x, y));
  XLS_ASSERT_OK_AND_ASSIGN(Function * func, fb.Build());

  XLS_ASSERT_OK_AND_ASSIGN(
      PipelineSchedule schedule,
      RunPipelineSchedule(func, TestDelayEstimator(),
                          SchedulingOptions().pipeline_stages(2)));

  XLS_ASSERT_OK_AND_ASSIGN(
      ModuleGeneratorResult result,
      ToPipelineModuleText(schedule, func,
                           BuildPipelineOptions()
                               .use_system_verilog(UseSystemVerilog())
                               .flop_inputs(true)
                               .flop_outputs(true)));

  ExpectVerilogEqualToGoldenFile(GoldenFilePath(kTestName, kTestdataPath),
                                 result.verilog_text);
  EXPECT_EQ(result.signature.proto().pipeline().latency(), 3);

  ModuleSimulator simulator =
      NewModuleSimulator(result.verilog_text, result.signature);
  EXPECT_THAT(simulator.RunAndReturnSingleOutput(
                  {{"x", UBits(123, 8)}, {"y", UBits(42, 8)}}),
              IsOkAndHolds(UBits(91, 8)));
}

TEST_P(PipelineGeneratorTest, AddNegateFlopInputsNotOutputs) {
  Package package(TestBaseName());
  FunctionBuilder fb(TestBaseName(), &package);
  auto x = fb.Param("x", package.GetBitsType(8));
  auto y = fb.Param("y", package.GetBitsType(8));
  fb.Negate(fb.Add(x, y));
  XLS_ASSERT_OK_AND_ASSIGN(Function * func, fb.Build());

  XLS_ASSERT_OK_AND_ASSIGN(
      PipelineSchedule schedule,
      RunPipelineSchedule(func, TestDelayEstimator(),
                          SchedulingOptions().pipeline_stages(2)));

  XLS_ASSERT_OK_AND_ASSIGN(
      ModuleGeneratorResult result,
      ToPipelineModuleText(schedule, func,
                           BuildPipelineOptions()
                               .use_system_verilog(UseSystemVerilog())
                               .flop_inputs(true)
                               .flop_outputs(false)));

  ExpectVerilogEqualToGoldenFile(GoldenFilePath(kTestName, kTestdataPath),
                                 result.verilog_text);
  EXPECT_EQ(result.signature.proto().pipeline().latency(), 2);

  ModuleSimulator simulator =
      NewModuleSimulator(result.verilog_text, result.signature);
  EXPECT_THAT(simulator.RunAndReturnSingleOutput(
                  {{"x", UBits(123, 8)}, {"y", UBits(42, 8)}}),
              IsOkAndHolds(UBits(91, 8)));
}

TEST_P(PipelineGeneratorTest, AddNegateFlopOutputsNotInputs) {
  Package package(TestBaseName());
  FunctionBuilder fb(TestBaseName(), &package);
  auto x = fb.Param("x", package.GetBitsType(8));
  auto y = fb.Param("y", package.GetBitsType(8));
  fb.Negate(fb.Add(x, y));
  XLS_ASSERT_OK_AND_ASSIGN(Function * func, fb.Build());

  XLS_ASSERT_OK_AND_ASSIGN(
      PipelineSchedule schedule,
      RunPipelineSchedule(func, TestDelayEstimator(),
                          SchedulingOptions().pipeline_stages(2)));

  XLS_ASSERT_OK_AND_ASSIGN(
      ModuleGeneratorResult result,
      ToPipelineModuleText(schedule, func,
                           BuildPipelineOptions()
                               .use_system_verilog(UseSystemVerilog())
                               .flop_inputs(false)
                               .flop_outputs(true)));

  ExpectVerilogEqualToGoldenFile(GoldenFilePath(kTestName, kTestdataPath),
                                 result.verilog_text);
  EXPECT_EQ(result.signature.proto().pipeline().latency(), 2);

  ModuleSimulator simulator =
      NewModuleSimulator(result.verilog_text, result.signature);
  EXPECT_THAT(simulator.RunAndReturnSingleOutput(
                  {{"x", UBits(123, 8)}, {"y", UBits(42, 8)}}),
              IsOkAndHolds(UBits(91, 8)));
}

TEST_P(PipelineGeneratorTest, AddNegateFlopNeitherInputsNorOutputs) {
  Package package(TestBaseName());
  FunctionBuilder fb(TestBaseName(), &package);
  auto x = fb.Param("x", package.GetBitsType(8));
  auto y = fb.Param("y", package.GetBitsType(8));
  fb.Negate(fb.Add(x, y));
  XLS_ASSERT_OK_AND_ASSIGN(Function * func, fb.Build());

  XLS_ASSERT_OK_AND_ASSIGN(
      PipelineSchedule schedule,
      RunPipelineSchedule(func, TestDelayEstimator(),
                          SchedulingOptions().pipeline_stages(2)));

  XLS_ASSERT_OK_AND_ASSIGN(
      ModuleGeneratorResult result,
      ToPipelineModuleText(schedule, func,
                           BuildPipelineOptions()
                               .use_system_verilog(UseSystemVerilog())
                               .flop_inputs(false)
                               .flop_outputs(false)));

  ExpectVerilogEqualToGoldenFile(GoldenFilePath(kTestName, kTestdataPath),
                                 result.verilog_text);
  EXPECT_EQ(result.signature.proto().pipeline().latency(), 1);

  ModuleSimulator simulator =
      NewModuleSimulator(result.verilog_text, result.signature);
  EXPECT_THAT(simulator.RunAndReturnSingleOutput(
                  {{"x", UBits(123, 8)}, {"y", UBits(42, 8)}}),
              IsOkAndHolds(UBits(91, 8)));
}

TEST_P(PipelineGeneratorTest, EmitsCoverpoints) {
  Package package(TestBaseName());
  FunctionBuilder fb(TestBaseName(), &package);
  auto x = fb.Param("x", package.GetBitsType(8));
  auto y = fb.Param("y", package.GetBitsType(8));
  auto sum = fb.Add(x, y);
  auto val_128 = fb.Literal(UBits(128, 8));
  auto gt_128 = fb.UGt(sum, val_128);
  fb.Cover(gt_128, "my_coverpoint");
  XLS_ASSERT_OK_AND_ASSIGN(Function * func, fb.BuildWithReturnValue(sum));

  XLS_ASSERT_OK_AND_ASSIGN(
      PipelineSchedule schedule,
      RunPipelineSchedule(func, TestDelayEstimator(),
                          SchedulingOptions().pipeline_stages(1)));

  XLS_ASSERT_OK_AND_ASSIGN(
      ModuleGeneratorResult result,
      ToPipelineModuleText(
          schedule, func,
          BuildPipelineOptions().use_system_verilog(UseSystemVerilog())));

  EXPECT_EQ(result.signature.data_outputs().size(), 1);
  EXPECT_EQ(result.signature.data_outputs()[0].name(), "out");

  ExpectVerilogEqualToGoldenFile(GoldenFilePath(kTestName, kTestdataPath),
                                 result.verilog_text);
}

TEST_P(PipelineGeneratorTest, ValidPipelineControlWithResetSimulation) {
  // Verify the valid signaling works as expected by driving the module with a
  // ModuleTestBench when a reset is present.
  Package package(TestBaseName());
  FunctionBuilder fb(TestBaseName(), &package);
  auto x = fb.Param("x", package.GetBitsType(64));
  auto y = fb.Param("y", package.GetBitsType(64));
  auto z = fb.Param("z", package.GetBitsType(64));
  auto a = fb.UMul(x, y);
  fb.UMul(a, z);

  XLS_ASSERT_OK_AND_ASSIGN(Function * func, fb.Build());

  XLS_ASSERT_OK_AND_ASSIGN(
      PipelineSchedule schedule,
      RunPipelineSchedule(func, TestDelayEstimator(),
                          SchedulingOptions().pipeline_stages(5)));

  ResetProto reset_proto;
  reset_proto.set_name("rst");
  reset_proto.set_asynchronous(false);
  reset_proto.set_active_low(false);

  XLS_ASSERT_OK_AND_ASSIGN(
      ModuleGeneratorResult result,
      ToPipelineModuleText(
          schedule, func,
          BuildPipelineOptions()
              .valid_control("in_valid", "out_valid")
              .reset(reset_proto.name(), reset_proto.asynchronous(),
                     reset_proto.active_low(), reset_proto.reset_data_path())
              .use_system_verilog(UseSystemVerilog())));

  // We directly manipulate the reset line so strip reset from the signature
  // and add a regular port so the testbench does not drive the reset line.
  XLS_ASSERT_OK_AND_ASSIGN(ModuleSignature resetless_signature,
                           StripResetFromSignature(result.signature));

  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<ModuleTestbench> tb,
                           ModuleTestbench::CreateFromVerilogText(
                               result.verilog_text, GetFileType(),
                               resetless_signature, GetSimulator()));
  XLS_ASSERT_OK_AND_ASSIGN(
      ModuleTestbenchThread * tbt,
      tb->CreateThreadDrivingAllInputs("main", /*default_value=*/ZeroOrX::kX));
  SequentialBlock& seq = tbt->MainBlock();
  seq.Set("in_valid", 0).Set("rst", 1);
  seq.NextCycle();
  seq.Set("rst", 0).NextCycle();

  seq.AtEndOfCycleWhenNotX("out_valid").ExpectEq("out_valid", 0);

  // Send in a valid value on one cycle.
  seq.Set("in_valid", 1).Set("x", 2).Set("y", 3).Set("z", 4);
  seq.NextCycle();

  // Then don't send any more valid values.
  seq.Set("in_valid", 0);

  // Wait until the output goes valid.
  const int kExpected = 2 * 3 * 4;
  seq.AtEndOfCycleWhen("out_valid")
      .ExpectEq("out_valid", 1)
      .ExpectEq("out", kExpected);

  // Output will be invalid in all subsequent cycles.
  seq.AtEndOfCycle().ExpectEq("out_valid", 0);
  seq.AtEndOfCycle().ExpectEq("out_valid", 0);

  // Now change the input and observe that the output never changes (because
  // we don't correspondingly set input_valid).
  seq.Set("z", 7);
  int64_t latency = result.signature.proto().pipeline().latency();
  ASSERT_GT(latency, 0);
  for (int64_t i = 0; i < 2 * latency; ++i) {
    seq.AtEndOfCycle().ExpectEq("out", kExpected).ExpectEq("out_valid", 0);
  }

  // Asserting reset should flush the pipeline after the pipeline latency even
  // without in_valid asserted.
  seq.Set("rst", 1).Set("in_valid", 0).Set("x", 0).Set("y", 0).Set("z", 0);
  for (int64_t i = 0; i < latency; ++i) {
    seq.AtEndOfCycle().ExpectEq("out", kExpected);
  }
  for (int64_t i = 0; i < latency; ++i) {
    seq.AtEndOfCycle().ExpectEq("out", 0);
  }

  XLS_ASSERT_OK(tb->Run());
}

TEST_P(PipelineGeneratorTest, IIGreaterThanOne) {
  const std::string ir_text = absl::Substitute(R"(package $0
chan in(bits[32], id=0, kind=streaming, ops=receive_only, flow_control=ready_valid)
chan out(bits[32], id=1, kind=streaming, ops=send_only, flow_control=ready_valid)
chan in_out(bits[32], id=2, kind=streaming, ops=send_only, flow_control=ready_valid)

#[initiation_interval(2)]
proc ii_greater_than_one(st: bits[32], init={0}) {
  tkn: token = literal(value=token, id=1000)
  send.1: token = send(tkn, st, channel=out, id=1)
  min_delay.2: token = min_delay(send.1, delay=1, id=2)
  receive.3: (token, bits[32]) = receive(min_delay.2, channel=in, id=3)
  tuple_index.4: token = tuple_index(receive.3, index=0, id=4)
  tuple_index.5: bits[32] = tuple_index(receive.3, index=1, id=5)
  send.6: token = send(tuple_index.4, tuple_index.5, channel=in_out, id=6)
  next_st: () = next_value(param=st, value=tuple_index.5)
}
)",
                                               TestBaseName());

  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> package,
                           xls::Parser::ParsePackage(ir_text));

  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc,
                           package->GetProc("ii_greater_than_one"));

  XLS_ASSERT_OK_AND_ASSIGN(
      PipelineSchedule schedule,
      RunPipelineSchedule(
          proc, TestDelayEstimator(),
          SchedulingOptions().clock_period_ps(50).pipeline_stages(2)));

  ResetProto reset_proto;
  reset_proto.set_name("rst");
  reset_proto.set_asynchronous(false);
  reset_proto.set_active_low(false);

  XLS_ASSERT_OK_AND_ASSIGN(
      ModuleGeneratorResult result,
      ToPipelineModuleText(
          schedule, proc,
          BuildPipelineOptions()
              .reset(reset_proto.name(), reset_proto.asynchronous(),
                     reset_proto.active_low(), reset_proto.reset_data_path())
              .use_system_verilog(UseSystemVerilog())));

  ExpectVerilogEqualToGoldenFile(GoldenFilePath(kTestName, kTestdataPath),
                                 result.verilog_text);
}

INSTANTIATE_TEST_SUITE_P(PipelineGeneratorTestInstantiation,
                         PipelineGeneratorTest,
                         testing::ValuesIn(kDefaultSimulationTargets),
                         ParameterizedTestName<PipelineGeneratorTest>);

}  // namespace
}  // namespace verilog
}  // namespace xls
