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

#include "xls/codegen/combinational_generator.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <random>
#include <string>
#include <string_view>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xls/codegen/vast/vast.h"
#include "xls/common/status/matchers.h"
#include "xls/examples/sample_packages.h"
#include "xls/interpreter/function_interpreter.h"
#include "xls/interpreter/random_value.h"
#include "xls/ir/bits.h"
#include "xls/ir/channel.h"
#include "xls/ir/events.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/ir_parser.h"
#include "xls/ir/package.h"
#include "xls/ir/source_location.h"
#include "xls/ir/value.h"
#include "xls/simulation/module_simulator.h"
#include "xls/simulation/module_testbench.h"
#include "xls/simulation/module_testbench_thread.h"
#include "xls/simulation/verilog_test_base.h"

namespace xls {
namespace verilog {
namespace {

using ::absl_testing::IsOkAndHolds;

constexpr char kTestName[] = "combinational_generator_test";
constexpr char kTestdataPath[] = "xls/codegen/testdata";

class CombinationalGeneratorTest : public VerilogTestBase {};

TEST_P(CombinationalGeneratorTest, RrotToCombinationalText) {
  auto rrot32_data = sample_packages::BuildRrot32();
  Function* f = rrot32_data.second;
  XLS_ASSERT_OK_AND_ASSIGN(auto result,
                           GenerateCombinationalModule(f, codegen_options()));

  ExpectVerilogEqualToGoldenFile(GoldenFilePath(kTestName, kTestdataPath),
                                 result.verilog_text);

  ModuleSimulator simulator =
      NewModuleSimulator(result.verilog_text, result.signature);
  EXPECT_THAT(simulator.RunAndReturnSingleOutput(
                  {{"x", UBits(0x12345678ULL, 32)}, {"y", UBits(4, 32)}}),
              IsOkAndHolds(UBits(0x81234567, 32)));
}

TEST_P(CombinationalGeneratorTest, RandomExpression) {
  Package package(TestBaseName());
  FunctionBuilder fb(TestBaseName(), &package);
  Type* u8 = package.GetBitsType(8);
  auto a = fb.Param("a", u8);
  auto b = fb.Param("b", u8);
  auto c = fb.Param("c", u8);
  auto a_minus_b = fb.Subtract(a, b, SourceInfo(), /*name=*/"diff");
  auto lhs = (a_minus_b * a_minus_b);
  auto rhs = (c * a_minus_b);
  auto out = fb.Add(lhs, rhs, SourceInfo(), /*name=*/"the_output");
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.BuildWithReturnValue(out));
  XLS_ASSERT_OK_AND_ASSIGN(auto result,
                           GenerateCombinationalModule(f, codegen_options()));

  ExpectVerilogEqualToGoldenFile(GoldenFilePath(kTestName, kTestdataPath),
                                 result.verilog_text);

  ModuleSimulator simulator =
      NewModuleSimulator(result.verilog_text, result.signature);

  // Value should be: (7-2)*(7-2) + 3*(7-2) = 40
  EXPECT_THAT(simulator.RunAndReturnSingleOutput(
                  {{"a", UBits(7, 8)}, {"b", UBits(2, 8)}, {"c", UBits(3, 8)}}),
              IsOkAndHolds(UBits(40, 8)));
}

TEST_P(CombinationalGeneratorTest, ReturnsLiteral) {
  Package package(TestBaseName());
  FunctionBuilder fb(TestBaseName(), &package);
  fb.Literal(UBits(123, 8));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  XLS_ASSERT_OK_AND_ASSIGN(auto result,
                           GenerateCombinationalModule(f, codegen_options()));
  ModuleSimulator simulator =
      NewModuleSimulator(result.verilog_text, result.signature);
  EXPECT_THAT(simulator.RunAndReturnSingleOutput(ModuleSimulator::BitsMap()),
              IsOkAndHolds(UBits(123, 8)));
}

TEST_P(CombinationalGeneratorTest, ReturnsTupleLiteral) {
  Package package(TestBaseName());
  FunctionBuilder fb(TestBaseName(), &package);
  fb.Literal(Value::Tuple({Value(UBits(123, 8)), Value(UBits(42, 32))}));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  XLS_ASSERT_OK_AND_ASSIGN(auto result,
                           GenerateCombinationalModule(f, codegen_options()));
  ModuleSimulator simulator =
      NewModuleSimulator(result.verilog_text, result.signature);
  EXPECT_THAT(
      simulator.RunFunction(absl::flat_hash_map<std::string, Value>()),
      IsOkAndHolds(Value::Tuple({Value(UBits(123, 8)), Value(UBits(42, 32))})));
}

TEST_P(CombinationalGeneratorTest, ReturnsEmptyTuple) {
  Package package(TestBaseName());
  FunctionBuilder fb(TestBaseName(), &package);
  fb.Literal(Value::Tuple({}));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  XLS_ASSERT_OK_AND_ASSIGN(auto result,
                           GenerateCombinationalModule(f, codegen_options()));
  ModuleSimulator simulator =
      NewModuleSimulator(result.verilog_text, result.signature);
  EXPECT_THAT(simulator.RunFunction(absl::flat_hash_map<std::string, Value>()),
              IsOkAndHolds(Value::Tuple({})));
}

TEST_P(CombinationalGeneratorTest, PassesEmptyTuple) {
  Package package(TestBaseName());
  FunctionBuilder fb(TestBaseName(), &package);
  fb.Param("x", package.GetTupleType({}));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  XLS_ASSERT_OK_AND_ASSIGN(auto result,
                           GenerateCombinationalModule(f, codegen_options()));
  ModuleSimulator simulator =
      NewModuleSimulator(result.verilog_text, result.signature);
  EXPECT_THAT(simulator.RunFunction({{"x", Value::Tuple({})}}),
              IsOkAndHolds(Value::Tuple({})));
}

TEST_P(CombinationalGeneratorTest, TakesEmptyTuple) {
  Package package(TestBaseName());
  FunctionBuilder fb(TestBaseName(), &package);
  Type* u8 = package.GetBitsType(8);
  auto a = fb.Param("a", u8);
  fb.Param("b", package.GetTupleType({}));
  auto c = fb.Param("c", u8);
  fb.Add(a, c, SourceInfo(), /*name=*/"sum");
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  XLS_ASSERT_OK_AND_ASSIGN(auto result,
                           GenerateCombinationalModule(f, codegen_options()));
  ModuleSimulator simulator =
      NewModuleSimulator(result.verilog_text, result.signature);
  EXPECT_THAT(simulator.RunFunction({{"a", Value(UBits(42, 8))},
                                     {"b", Value::Tuple({})},
                                     {"c", Value(UBits(100, 8))}}),
              IsOkAndHolds(Value(UBits(142, 8))));
}

TEST_P(CombinationalGeneratorTest, ReturnsParam) {
  Package package(TestBaseName());
  FunctionBuilder fb(TestBaseName(), &package);
  Type* u8 = package.GetBitsType(8);
  fb.Param("a", u8);
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  XLS_ASSERT_OK_AND_ASSIGN(auto result,
                           GenerateCombinationalModule(f, codegen_options()));
  ModuleSimulator simulator =
      NewModuleSimulator(result.verilog_text, result.signature);
  EXPECT_THAT(simulator.RunAndReturnSingleOutput({{"a", UBits(0x42, 8)}}),
              IsOkAndHolds(UBits(0x42, 8)));
}

TEST_P(CombinationalGeneratorTest, ExpressionWhichRequiresNamedIntermediate) {
  Package package(TestBaseName());
  FunctionBuilder fb(TestBaseName(), &package);
  Type* u8 = package.GetBitsType(8);
  auto a = fb.Param("a", u8);
  auto b = fb.Param("b", u8);
  auto a_plus_b = a + b;
  auto out = fb.BitSlice(a_plus_b, /*start=*/3, /*width=*/4, SourceInfo(),
                         /*name=*/"slice_n_dice");
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.BuildWithReturnValue(out));
  XLS_ASSERT_OK_AND_ASSIGN(auto result,
                           GenerateCombinationalModule(f, codegen_options()));

  ExpectVerilogEqualToGoldenFile(GoldenFilePath(kTestName, kTestdataPath),
                                 result.verilog_text);

  ModuleSimulator simulator =
      NewModuleSimulator(result.verilog_text, result.signature);
  EXPECT_THAT(simulator.RunAndReturnSingleOutput(
                  {{"a", UBits(0x42, 8)}, {"b", UBits(0x33, 8)}}),
              IsOkAndHolds(UBits(14, 4)));
}

TEST_P(CombinationalGeneratorTest, ExpressionsOfTuples) {
  Package package(TestBaseName());
  FunctionBuilder fb(TestBaseName(), &package);
  Type* u8 = package.GetBitsType(8);
  Type* u10 = package.GetBitsType(10);
  Type* u16 = package.GetBitsType(16);
  Type* tuple_u10_u16 = package.GetTupleType({u10, u16});
  auto a = fb.Param("a", u8);
  auto b = fb.Param("b", u10);
  auto c = fb.Param("c", tuple_u10_u16);

  // Glom all the inputs together into a big tuple.
  auto a_b_c = fb.Tuple({a, b, c}, SourceInfo(), /*name=*/"big_tuple");

  // Then extract some elements and perform some arithmetic operations on them
  // after zero-extending them to the same width (16-bits).
  auto a_plus_b = fb.ZeroExtend(fb.TupleIndex(a_b_c, 0), 16) +
                  fb.ZeroExtend(fb.TupleIndex(a_b_c, 1), 16);
  auto c_tmp = fb.TupleIndex(a_b_c, 2);
  auto c0_minus_c1 =
      fb.ZeroExtend(fb.TupleIndex(c_tmp, 0), 16) - fb.TupleIndex(c_tmp, 1);

  // Result should be a two-tuple containing {a + b, c[0] - c[1]}
  auto return_value = fb.Tuple({a_plus_b, c0_minus_c1});
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.BuildWithReturnValue(return_value));
  XLS_ASSERT_OK_AND_ASSIGN(auto result,
                           GenerateCombinationalModule(f, codegen_options()));

  ExpectVerilogEqualToGoldenFile(GoldenFilePath(kTestName, kTestdataPath),
                                 result.verilog_text);

  ModuleSimulator simulator =
      NewModuleSimulator(result.verilog_text, result.signature);
  EXPECT_THAT(
      simulator.RunFunction({{"a", Value(UBits(42, 8))},
                             {"b", Value(UBits(123, 10))},
                             {"c", Value::Tuple({Value(UBits(333, 10)),
                                                 Value(UBits(222, 16))})}}),
      IsOkAndHolds(
          Value::Tuple({Value(UBits(165, 16)), Value(UBits(111, 16))})));
}

TEST_P(CombinationalGeneratorTest, TupleLiterals) {
  std::string text = R"(
package TupleLiterals

top fn main(x: bits[123]) -> bits[123] {
  literal.1: (bits[123], bits[123], bits[123]) = literal(value=(0x10000, 0x2000, 0x300))
  tuple_index.2: bits[123] = tuple_index(literal.1, index=0)
  tuple_index.3: bits[123] = tuple_index(literal.1, index=1)
  tuple_index.4: bits[123] = tuple_index(literal.1, index=2)
  sum1: bits[123] = add(tuple_index.2, tuple_index.3)
  sum2: bits[123] = add(tuple_index.4, x)
  ret total: bits[123] = add(sum1, sum2)
}
)";
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> package,
                           Parser::ParsePackage(text));

  std::optional<FunctionBase*> top = package->GetTop();
  ASSERT_TRUE(top.has_value());
  XLS_ASSERT_OK_AND_ASSIGN(
      auto result, GenerateCombinationalModule(top.value(), codegen_options()));

  ExpectVerilogEqualToGoldenFile(GoldenFilePath(kTestName, kTestdataPath),
                                 result.verilog_text);

  ModuleSimulator simulator =
      NewModuleSimulator(result.verilog_text, result.signature);
  EXPECT_THAT(simulator.RunFunction({{"x", Value(UBits(0x40, 123))}}),
              IsOkAndHolds(Value(UBits(0x12340, 123))));
}

TEST_P(CombinationalGeneratorTest, ArrayLiteral) {
  std::string text = R"(
package ArrayLiterals

top fn main(x: bits[32], y: bits[32]) -> bits[44] {
  literal.1: bits[44][3][2] = literal(value=[[1, 2, 3], [4, 5, 6]])
  array_index.2: bits[44][3] = array_index(literal.1, indices=[x])
  ret result: bits[44] = array_index(array_index.2, indices=[y])
}
)";
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> package,
                           Parser::ParsePackage(text));

  std::optional<FunctionBase*> top = package->GetTop();
  ASSERT_TRUE(top.has_value());
  XLS_ASSERT_OK_AND_ASSIGN(
      auto result, GenerateCombinationalModule(top.value(), codegen_options()));

  ExpectVerilogEqualToGoldenFile(GoldenFilePath(kTestName, kTestdataPath),
                                 result.verilog_text);

  ModuleSimulator simulator =
      NewModuleSimulator(result.verilog_text, result.signature);
  EXPECT_THAT(simulator.RunFunction(
                  {{"x", Value(UBits(0, 32))}, {"y", Value(UBits(1, 32))}}),
              IsOkAndHolds(Value(UBits(2, 44))));
  EXPECT_THAT(simulator.RunFunction(
                  {{"x", Value(UBits(1, 32))}, {"y", Value(UBits(0, 32))}}),
              IsOkAndHolds(Value(UBits(4, 44))));
}

TEST_P(CombinationalGeneratorTest, OneHot) {
  std::string text = R"(
package OneHot

top fn main(x: bits[3]) -> bits[4] {
  ret one_hot.1: bits[4] = one_hot(x, lsb_prio=true)
}
)";
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> package,
                           Parser::ParsePackage(text));

  std::optional<FunctionBase*> top = package->GetTop();
  ASSERT_TRUE(top.has_value());
  XLS_ASSERT_OK_AND_ASSIGN(
      auto result, GenerateCombinationalModule(top.value(), codegen_options()));

  ExpectVerilogEqualToGoldenFile(GoldenFilePath(kTestName, kTestdataPath),
                                 result.verilog_text);

  ModuleSimulator simulator =
      NewModuleSimulator(result.verilog_text, result.signature);
  EXPECT_THAT(simulator.RunFunction({{"x", Value(UBits(0b000, 3))}}),
              IsOkAndHolds(Value(UBits(0b1000, 4))));
  EXPECT_THAT(simulator.RunFunction({{"x", Value(UBits(0b001, 3))}}),
              IsOkAndHolds(Value(UBits(0b0001, 4))));
  EXPECT_THAT(simulator.RunFunction({{"x", Value(UBits(0b010, 3))}}),
              IsOkAndHolds(Value(UBits(0b0010, 4))));
  EXPECT_THAT(simulator.RunFunction({{"x", Value(UBits(0b011, 3))}}),
              IsOkAndHolds(Value(UBits(0b0001, 4))));
  EXPECT_THAT(simulator.RunFunction({{"x", Value(UBits(0b100, 3))}}),
              IsOkAndHolds(Value(UBits(0b0100, 4))));
  EXPECT_THAT(simulator.RunFunction({{"x", Value(UBits(0b101, 3))}}),
              IsOkAndHolds(Value(UBits(0b0001, 4))));
  EXPECT_THAT(simulator.RunFunction({{"x", Value(UBits(0b110, 3))}}),
              IsOkAndHolds(Value(UBits(0b0010, 4))));
  EXPECT_THAT(simulator.RunFunction({{"x", Value(UBits(0b111, 3))}}),
              IsOkAndHolds(Value(UBits(0b0001, 4))));
}

TEST_P(CombinationalGeneratorTest, OneHotSelect) {
  std::string text = R"(
package OneHotSelect

top fn main(p: bits[2], x: bits[16], y: bits[16]) -> bits[16] {
  ret one_hot_sel.1: bits[16] = one_hot_sel(p, cases=[x, y])
}
)";
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> package,
                           Parser::ParsePackage(text));

  std::optional<FunctionBase*> top = package->GetTop();
  ASSERT_TRUE(top.has_value());
  XLS_ASSERT_OK_AND_ASSIGN(
      auto result, GenerateCombinationalModule(top.value(), codegen_options()));

  ExpectVerilogEqualToGoldenFile(GoldenFilePath(kTestName, kTestdataPath),
                                 result.verilog_text);

  ModuleSimulator simulator =
      NewModuleSimulator(result.verilog_text, result.signature);
  absl::flat_hash_map<std::string, Value> args = {
      {"x", Value(UBits(0x00ff, 16))}, {"y", Value(UBits(0xf0f0, 16))}};
  args["p"] = Value(UBits(0b00, 2));
  EXPECT_THAT(simulator.RunFunction(args),
              IsOkAndHolds(Value(UBits(0x0000, 16))));
  args["p"] = Value(UBits(0b01, 2));
  EXPECT_THAT(simulator.RunFunction(args),
              IsOkAndHolds(Value(UBits(0x00ff, 16))));
  args["p"] = Value(UBits(0b10, 2));
  EXPECT_THAT(simulator.RunFunction(args),
              IsOkAndHolds(Value(UBits(0xf0f0, 16))));
  args["p"] = Value(UBits(0b11, 2));
  EXPECT_THAT(simulator.RunFunction(args),
              IsOkAndHolds(Value(UBits(0xf0ff, 16))));
}

TEST_P(CombinationalGeneratorTest, OneHotSelectNonBits) {
  std::string text = R"(
package OneHotSelect

top fn main(p: bits[2], x: (bits[16], bits[16]), y: (bits[16], bits[16])) -> (bits[16], bits[16]) {
  ret one_hot_sel.1: (bits[16], bits[16]) = one_hot_sel(p, cases=[x, y])
}
)";
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> package,
                           Parser::ParsePackage(text));

  std::optional<FunctionBase*> top = package->GetTop();
  ASSERT_TRUE(top.has_value());
  XLS_ASSERT_OK_AND_ASSIGN(
      auto result, GenerateCombinationalModule(top.value(), codegen_options()));

  ExpectVerilogEqualToGoldenFile(GoldenFilePath(kTestName, kTestdataPath),
                                 result.verilog_text);

  ModuleSimulator simulator =
      NewModuleSimulator(result.verilog_text, result.signature);
  absl::flat_hash_map<std::string, Value> args = {
      {"x", Value::Tuple({Value(UBits(0x00ff, 16)), Value(UBits(0xff00, 16))})},
      {"y",
       Value::Tuple({Value(UBits(0xf0f0, 16)), Value(UBits(0x0f0f, 16))})}};
  args["p"] = Value(UBits(0b00, 2));
  EXPECT_THAT(simulator.RunFunction(args),
              IsOkAndHolds(Value::Tuple(
                  {Value(UBits(0x0000, 16)), Value(UBits(0x0000, 16))})));
  args["p"] = Value(UBits(0b01, 2));
  EXPECT_THAT(simulator.RunFunction(args),
              IsOkAndHolds(Value::Tuple(
                  {Value(UBits(0x00ff, 16)), Value(UBits(0xff00, 16))})));
  args["p"] = Value(UBits(0b10, 2));
  EXPECT_THAT(simulator.RunFunction(args),
              IsOkAndHolds(Value::Tuple(
                  {Value(UBits(0xf0f0, 16)), Value(UBits(0x0f0f, 16))})));
  args["p"] = Value(UBits(0b11, 2));
  EXPECT_THAT(simulator.RunFunction(args),
              IsOkAndHolds(Value::Tuple(
                  {Value(UBits(0xf0ff, 16)), Value(UBits(0xff0f, 16))})));
}

TEST_P(CombinationalGeneratorTest, PrioritySelect) {
  std::string text = R"(
package PrioritySelect

top fn main(p: bits[2], x: bits[16], y: bits[16], d: bits[16]) -> bits[16] {
  ret priority_sel.1: bits[16] = priority_sel(p, cases=[x, y], default=d)
}
)";
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> package,
                           Parser::ParsePackage(text));

  std::optional<FunctionBase*> top = package->GetTop();
  ASSERT_TRUE(top.has_value());
  XLS_ASSERT_OK_AND_ASSIGN(
      auto result, GenerateCombinationalModule(top.value(), codegen_options()));

  ExpectVerilogEqualToGoldenFile(GoldenFilePath(kTestName, kTestdataPath),
                                 result.verilog_text);

  ModuleSimulator simulator =
      NewModuleSimulator(result.verilog_text, result.signature);
  absl::flat_hash_map<std::string, Value> args = {
      {"x", Value(UBits(0x00ff, 16))},
      {"y", Value(UBits(0xf0f0, 16))},
      {"d", Value(UBits(0xff00, 16))}};
  args["p"] = Value(UBits(0b00, 2));
  EXPECT_THAT(simulator.RunFunction(args),
              IsOkAndHolds(Value(UBits(0xff00, 16))));
  args["p"] = Value(UBits(0b01, 2));
  EXPECT_THAT(simulator.RunFunction(args),
              IsOkAndHolds(Value(UBits(0x00ff, 16))));
  args["p"] = Value(UBits(0b10, 2));
  EXPECT_THAT(simulator.RunFunction(args),
              IsOkAndHolds(Value(UBits(0xf0f0, 16))));
  args["p"] = Value(UBits(0b11, 2));
  EXPECT_THAT(simulator.RunFunction(args),
              IsOkAndHolds(Value(UBits(0x00ff, 16))));
}

TEST_P(CombinationalGeneratorTest, PrioritySelectMultipleAreMerged) {
  std::string text = R"(
package PrioritySelect

top fn main(p: bits[2], x: bits[16], y: bits[16], d: bits[16]) -> bits[16] {
  priority_sel.1: bits[16] = priority_sel(p, cases=[x, y], default=d)
  priority_sel.2: bits[16] = priority_sel(p, cases=[y, x], default=d)
  ret add.3: bits[16] = add(priority_sel.1, priority_sel.2)
}
)";
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> package,
                           Parser::ParsePackage(text));

  std::optional<FunctionBase*> top = package->GetTop();
  ASSERT_TRUE(top.has_value());
  XLS_ASSERT_OK_AND_ASSIGN(
      auto result, GenerateCombinationalModule(top.value(), codegen_options()));

  ExpectVerilogEqualToGoldenFile(GoldenFilePath(kTestName, kTestdataPath),
                                 result.verilog_text);

  ModuleSimulator simulator =
      NewModuleSimulator(result.verilog_text, result.signature);
  absl::flat_hash_map<std::string, Value> args = {
      {"x", Value(UBits(0x00ff, 16))},
      {"y", Value(UBits(0xf0f0, 16))},
      {"d", Value(UBits(0xff00, 16))}};
  args["p"] = Value(UBits(0b00, 2));
  // both priority selects return 0xff00, 0xff00 + 0xff00 = 0xfe00
  EXPECT_THAT(simulator.RunFunction(args),
              IsOkAndHolds(Value(UBits(0xfe00, 16))));
  args["p"] = Value(UBits(0b01, 2));
  // sum = 0x00ff + 0xf0f0 = 0xf1ef
  EXPECT_THAT(simulator.RunFunction(args),
              IsOkAndHolds(Value(UBits(0xf1ef, 16))));
  args["p"] = Value(UBits(0b10, 2));
  EXPECT_THAT(simulator.RunFunction(args),
              IsOkAndHolds(Value(UBits(0xf1ef, 16))));
  args["p"] = Value(UBits(0b11, 2));
  EXPECT_THAT(simulator.RunFunction(args),
              IsOkAndHolds(Value(UBits(0xf1ef, 16))));
}

TEST_P(CombinationalGeneratorTest, PrioritySelectSelectorNeverZero) {
  std::string text = R"(
package PrioritySelectSelectorNeverZero

top fn main(p: bits[2], x: bits[16], y: bits[16], d: bits[16]) -> bits[16] {
  literal.1: bits[2] = literal(value=1)
  or.2: bits[2] = or(p, literal.1)
  ret priority_sel.3: bits[16] = priority_sel(or.2, cases=[x, y], default=d)
}
)";
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> package,
                           Parser::ParsePackage(text));

  std::optional<FunctionBase*> top = package->GetTop();
  ASSERT_TRUE(top.has_value());
  XLS_ASSERT_OK_AND_ASSIGN(
      auto result, GenerateCombinationalModule(top.value(), codegen_options()));

  ExpectVerilogEqualToGoldenFile(GoldenFilePath(kTestName, kTestdataPath),
                                 result.verilog_text);

  ModuleSimulator simulator =
      NewModuleSimulator(result.verilog_text, result.signature);
  absl::flat_hash_map<std::string, Value> args = {
      {"x", Value(UBits(0x00ff, 16))},
      {"y", Value(UBits(0xf0f0, 16))},
      {"d", Value(UBits(0xff00, 16))}};
  args["p"] = Value(UBits(0b00, 2));
  EXPECT_THAT(simulator.RunFunction(args),
              IsOkAndHolds(Value(UBits(0x00ff, 16))));
  args["p"] = Value(UBits(0b01, 2));
  EXPECT_THAT(simulator.RunFunction(args),
              IsOkAndHolds(Value(UBits(0x00ff, 16))));
  args["p"] = Value(UBits(0b10, 2));
  EXPECT_THAT(simulator.RunFunction(args),
              IsOkAndHolds(Value(UBits(0x00ff, 16))));
  args["p"] = Value(UBits(0b11, 2));
  EXPECT_THAT(simulator.RunFunction(args),
              IsOkAndHolds(Value(UBits(0x00ff, 16))));
}

TEST_P(CombinationalGeneratorTest, PrioritySelectWithAndWithoutDefault) {
  // Tests that the different specialized ways of codegen'ing priority selects
  // are applied in the right context, e.g. we don't make a no-default priority
  // select function and call it later for a needs-default priority select.
  std::string text = R"(
package PrioritySelect

top fn main(p: bits[2], x: bits[16], y: bits[16], d: bits[16]) -> (bits[16], bits[16]) {
  literal.1: bits[2] = literal(value=1)
  or.2: bits[2] = or(p, literal.1)
  priority_sel.3: bits[16] = priority_sel(or.2, cases=[x, y], default=d)
  priority_sel.4: bits[16] = priority_sel(p, cases=[x, y], default=d)
  ret tuple.5: (bits[16], bits[16]) = tuple(priority_sel.3, priority_sel.4)
}
)";
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> package,
                           Parser::ParsePackage(text));

  std::optional<FunctionBase*> top = package->GetTop();
  ASSERT_TRUE(top.has_value());
  XLS_ASSERT_OK_AND_ASSIGN(
      auto result, GenerateCombinationalModule(top.value(), codegen_options()));

  ExpectVerilogEqualToGoldenFile(GoldenFilePath(kTestName, kTestdataPath),
                                 result.verilog_text);

  ModuleSimulator simulator =
      NewModuleSimulator(result.verilog_text, result.signature);
  absl::flat_hash_map<std::string, Value> args = {
      {"x", Value(UBits(0x00ff, 16))},
      {"y", Value(UBits(0xf0f0, 16))},
      {"d", Value(UBits(0xff00, 16))}};
  args["p"] = Value(UBits(0b00, 2));
  EXPECT_THAT(simulator.RunFunction(args),
              IsOkAndHolds(Value::Tuple(
                  {Value(UBits(0x00ff, 16)), Value(UBits(0xff00, 16))})));
  args["p"] = Value(UBits(0b01, 2));
  EXPECT_THAT(simulator.RunFunction(args),
              IsOkAndHolds(Value::Tuple(
                  {Value(UBits(0x00ff, 16)), Value(UBits(0x00ff, 16))})));
  args["p"] = Value(UBits(0b10, 2));
  EXPECT_THAT(simulator.RunFunction(args),
              IsOkAndHolds(Value::Tuple(
                  {Value(UBits(0x00ff, 16)), Value(UBits(0xf0f0, 16))})));
  args["p"] = Value(UBits(0b11, 2));
  EXPECT_THAT(simulator.RunFunction(args),
              IsOkAndHolds(Value::Tuple(
                  {Value(UBits(0x00ff, 16)), Value(UBits(0x00ff, 16))})));
}

TEST_P(CombinationalGeneratorTest, PrioritySelectOneHot) {
  std::string text = R"(
package PrioritySelectOneHot

top fn main(p: bits[1], x: bits[16], y: bits[16], d: bits[16]) -> bits[16] {
  one_hot.1: bits[2] = one_hot(p, lsb_prio=true)
  ret priority_sel.2: bits[16] = priority_sel(one_hot.1, cases=[x, y], default=d)
}
)";
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> package,
                           Parser::ParsePackage(text));

  std::optional<FunctionBase*> top = package->GetTop();
  ASSERT_TRUE(top.has_value());
  XLS_ASSERT_OK_AND_ASSIGN(
      auto result, GenerateCombinationalModule(top.value(), codegen_options()));

  ExpectVerilogEqualToGoldenFile(GoldenFilePath(kTestName, kTestdataPath),
                                 result.verilog_text);

  ModuleSimulator simulator =
      NewModuleSimulator(result.verilog_text, result.signature);
  absl::flat_hash_map<std::string, Value> args = {
      {"x", Value(UBits(0x00ff, 16))},
      {"y", Value(UBits(0xf0f0, 16))},
      {"d", Value(UBits(0xff00, 16))}};
  args["p"] = Value(UBits(0b0, 1));
  EXPECT_THAT(simulator.RunFunction(args),
              IsOkAndHolds(Value(UBits(0xf0f0, 16))));
  args["p"] = Value(UBits(0b1, 1));
  EXPECT_THAT(simulator.RunFunction(args),
              IsOkAndHolds(Value(UBits(0x00ff, 16))));
}

TEST_P(CombinationalGeneratorTest, PrioritySelectArray) {
  std::string text = R"(
package PrioritySelectArray

top fn main(p: bits[2], x: bits[16][4], y: bits[16][4], d: bits[16][4]) -> bits[16][4] {
  ret priority_sel.1: bits[16][4] = priority_sel(p, cases=[x, y], default=d)
}
)";
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> package,
                           Parser::ParsePackage(text));

  std::optional<FunctionBase*> top = package->GetTop();
  ASSERT_TRUE(top.has_value());
  XLS_ASSERT_OK_AND_ASSIGN(
      auto result, GenerateCombinationalModule(top.value(), codegen_options()));

  ExpectVerilogEqualToGoldenFile(GoldenFilePath(kTestName, kTestdataPath),
                                 result.verilog_text);

  ModuleSimulator simulator =
      NewModuleSimulator(result.verilog_text, result.signature);
  XLS_ASSERT_OK_AND_ASSIGN(
      Value x_value, Value::UBitsArray({0x00ff, 0x00ff, 0x00ff, 0x00ff}, 16));
  XLS_ASSERT_OK_AND_ASSIGN(
      Value y_value, Value::UBitsArray({0xf0f0, 0xf0f0, 0xf0f0, 0xf0f0}, 16));
  XLS_ASSERT_OK_AND_ASSIGN(
      Value d_value, Value::UBitsArray({0xff00, 0xff00, 0xff00, 0xff00}, 16));
  absl::flat_hash_map<std::string, Value> args = {
      {"x", x_value}, {"y", y_value}, {"d", d_value}};
  args["p"] = Value(UBits(0b00, 2));
  EXPECT_THAT(simulator.RunFunction(args), IsOkAndHolds(d_value));
  args["p"] = Value(UBits(0b01, 2));
  EXPECT_THAT(simulator.RunFunction(args), IsOkAndHolds(x_value));
  args["p"] = Value(UBits(0b10, 2));
  EXPECT_THAT(simulator.RunFunction(args), IsOkAndHolds(y_value));
  args["p"] = Value(UBits(0b11, 2));
  EXPECT_THAT(simulator.RunFunction(args), IsOkAndHolds(x_value));
}

TEST_P(CombinationalGeneratorTest,
       PrioritySelectMultipleArraysWithSameElementType) {
  std::string text = R"(
package PrioritySelectArraysWithSameElementType

top fn main(p: bits[2], x_short: bits[16][2], x_long: bits[16][4], y_short: bits[16][2], y_long: bits[16][4], d_short: bits[16][2], d_long: bits[16][4]) -> (bits[16][2], bits[16][4]) {
  priority_sel.1: bits[16][2] = priority_sel(p, cases=[x_short, y_short], default=d_short)
  priority_sel.2: bits[16][4] = priority_sel(p, cases=[x_long, y_long], default=d_long)
  ret tuple.3: (bits[16][2], bits[16][4]) = tuple(priority_sel.1, priority_sel.2)
}
)";
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> package,
                           Parser::ParsePackage(text));

  std::optional<FunctionBase*> top = package->GetTop();
  ASSERT_TRUE(top.has_value());
  XLS_ASSERT_OK_AND_ASSIGN(
      auto result, GenerateCombinationalModule(top.value(), codegen_options()));

  ExpectVerilogEqualToGoldenFile(GoldenFilePath(kTestName, kTestdataPath),
                                 result.verilog_text);

  ModuleSimulator simulator =
      NewModuleSimulator(result.verilog_text, result.signature);
  XLS_ASSERT_OK_AND_ASSIGN(Value x_short_value,
                           Value::UBitsArray({0x00ff, 0x00ff}, 16));
  XLS_ASSERT_OK_AND_ASSIGN(Value y_short_value,
                           Value::UBitsArray({0xf0f0, 0xf0f0}, 16));
  XLS_ASSERT_OK_AND_ASSIGN(Value d_short_value,
                           Value::UBitsArray({0xff00, 0xff00}, 16));
  XLS_ASSERT_OK_AND_ASSIGN(
      Value x_long_value,
      Value::UBitsArray({0x00ff, 0x00ff, 0x00ff, 0x00ff}, 16));
  XLS_ASSERT_OK_AND_ASSIGN(
      Value y_long_value,
      Value::UBitsArray({0xf0f0, 0xf0f0, 0xf0f0, 0xf0f0}, 16));
  XLS_ASSERT_OK_AND_ASSIGN(
      Value d_long_value,
      Value::UBitsArray({0xff00, 0xff00, 0xff00, 0xff00}, 16));
  absl::flat_hash_map<std::string, Value> args = {
      {"x_short", x_short_value}, {"y_short", y_short_value},
      {"d_short", d_short_value}, {"x_long", x_long_value},
      {"y_long", y_long_value},   {"d_long", d_long_value}};
  args["p"] = Value(UBits(0b00, 2));
  EXPECT_THAT(simulator.RunFunction(args),
              IsOkAndHolds(Value::Tuple({d_short_value, d_long_value})));
  args["p"] = Value(UBits(0b01, 2));
  EXPECT_THAT(simulator.RunFunction(args),
              IsOkAndHolds(Value::Tuple({x_short_value, x_long_value})));
  args["p"] = Value(UBits(0b10, 2));
  EXPECT_THAT(simulator.RunFunction(args),
              IsOkAndHolds(Value::Tuple({y_short_value, y_long_value})));
  args["p"] = Value(UBits(0b11, 2));
  EXPECT_THAT(simulator.RunFunction(args),
              IsOkAndHolds(Value::Tuple({x_short_value, x_long_value})));
}

TEST_P(CombinationalGeneratorTest, PrioritySelectArraySelectorNeverZero) {
  std::string text = R"(
package PrioritySelectArraySelectorNeverZero

top fn main(p: bits[2], x: bits[16][4], y: bits[16][4], d: bits[16][4]) -> bits[16][4] {

  literal.1: bits[2] = literal(value=1)
  or.2: bits[2] = or(p, literal.1)
  ret priority_sel.3: bits[16][4] = priority_sel(or.2, cases=[x, y], default=d)
}
)";
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> package,
                           Parser::ParsePackage(text));

  std::optional<FunctionBase*> top = package->GetTop();
  ASSERT_TRUE(top.has_value());
  XLS_ASSERT_OK_AND_ASSIGN(
      auto result, GenerateCombinationalModule(top.value(), codegen_options()));

  ExpectVerilogEqualToGoldenFile(GoldenFilePath(kTestName, kTestdataPath),
                                 result.verilog_text);

  ModuleSimulator simulator =
      NewModuleSimulator(result.verilog_text, result.signature);
  XLS_ASSERT_OK_AND_ASSIGN(
      Value x_value, Value::UBitsArray({0x00ff, 0x00ff, 0x00ff, 0x00ff}, 16));
  XLS_ASSERT_OK_AND_ASSIGN(
      Value y_value, Value::UBitsArray({0xf0f0, 0xf0f0, 0xf0f0, 0xf0f0}, 16));
  XLS_ASSERT_OK_AND_ASSIGN(
      Value d_value, Value::UBitsArray({0xff00, 0xff00, 0xff00, 0xff00}, 16));
  absl::flat_hash_map<std::string, Value> args = {
      {"x", x_value}, {"y", y_value}, {"d", d_value}};
  args["p"] = Value(UBits(0b00, 2));
  EXPECT_THAT(simulator.RunFunction(args), IsOkAndHolds(x_value));
  args["p"] = Value(UBits(0b01, 2));
  EXPECT_THAT(simulator.RunFunction(args), IsOkAndHolds(x_value));
  args["p"] = Value(UBits(0b10, 2));
  EXPECT_THAT(simulator.RunFunction(args), IsOkAndHolds(x_value));
  args["p"] = Value(UBits(0b11, 2));
  EXPECT_THAT(simulator.RunFunction(args), IsOkAndHolds(x_value));
}

TEST_P(CombinationalGeneratorTest, PrioritySelectArrayOneHot) {
  std::string text = R"(
package PrioritySelectArrayOneHot

top fn main(p: bits[1], x: bits[16][4], y: bits[16][4], d: bits[16][4]) -> bits[16][4] {
  one_hot.1: bits[2] = one_hot(p, lsb_prio=true)
  ret priority_sel.2: bits[16][4] = priority_sel(one_hot.1, cases=[x, y], default=d)
}
)";
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> package,
                           Parser::ParsePackage(text));

  std::optional<FunctionBase*> top = package->GetTop();
  ASSERT_TRUE(top.has_value());
  XLS_ASSERT_OK_AND_ASSIGN(
      auto result, GenerateCombinationalModule(top.value(), codegen_options()));

  ExpectVerilogEqualToGoldenFile(GoldenFilePath(kTestName, kTestdataPath),
                                 result.verilog_text);

  ModuleSimulator simulator =
      NewModuleSimulator(result.verilog_text, result.signature);
  XLS_ASSERT_OK_AND_ASSIGN(
      Value x_value, Value::UBitsArray({0x00ff, 0x00ff, 0x00ff, 0x00ff}, 16));
  XLS_ASSERT_OK_AND_ASSIGN(
      Value y_value, Value::UBitsArray({0xf0f0, 0xf0f0, 0xf0f0, 0xf0f0}, 16));
  XLS_ASSERT_OK_AND_ASSIGN(
      Value d_value, Value::UBitsArray({0xff00, 0xff00, 0xff00, 0xff00}, 16));
  absl::flat_hash_map<std::string, Value> args = {
      {"x", x_value}, {"y", y_value}, {"d", d_value}};
  args["p"] = Value(UBits(0b0, 1));
  EXPECT_THAT(simulator.RunFunction(args), IsOkAndHolds(y_value));
  args["p"] = Value(UBits(0b1, 1));
  EXPECT_THAT(simulator.RunFunction(args), IsOkAndHolds(x_value));
}

TEST_P(CombinationalGeneratorTest, PrioritySelectNonBits) {
  constexpr std::string_view text = R"(
package PrioritySelect

top fn main(p: bits[2], x: (bits[16], bits[16]), y: (bits[16], bits[16]), d: (bits[16], bits[16])) -> (bits[16], bits[16]) {
  ret priority_sel.1: (bits[16], bits[16]) = priority_sel(p, cases=[x, y], default=d)
}
)";
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> package,
                           Parser::ParsePackage(text));

  std::optional<FunctionBase*> top = package->GetTop();
  ASSERT_TRUE(top.has_value());
  XLS_ASSERT_OK_AND_ASSIGN(
      auto result, GenerateCombinationalModule(top.value(), codegen_options()));

  ExpectVerilogEqualToGoldenFile(GoldenFilePath(kTestName, kTestdataPath),
                                 result.verilog_text);

  ModuleSimulator simulator =
      NewModuleSimulator(result.verilog_text, result.signature);
  absl::flat_hash_map<std::string, Value> args = {
      {"x", Value::Tuple({Value(UBits(0x00ff, 16)), Value(UBits(0xff00, 16))})},
      {"y", Value::Tuple({Value(UBits(0xf0f0, 16)), Value(UBits(0x0f0f, 16))})},
      {"d", Value::Tuple({Value(UBits(0xff00, 16)), Value(UBits(0x00ff, 16))})},
  };
  args["p"] = Value(UBits(0b00, 2));
  EXPECT_THAT(simulator.RunFunction(args),
              IsOkAndHolds(Value::Tuple(
                  {Value(UBits(0xff00, 16)), Value(UBits(0x00ff, 16))})));
  args["p"] = Value(UBits(0b01, 2));
  EXPECT_THAT(simulator.RunFunction(args),
              IsOkAndHolds(Value::Tuple(
                  {Value(UBits(0x00ff, 16)), Value(UBits(0xff00, 16))})));
  args["p"] = Value(UBits(0b10, 2));
  EXPECT_THAT(simulator.RunFunction(args),
              IsOkAndHolds(Value::Tuple(
                  {Value(UBits(0xf0f0, 16)), Value(UBits(0x0f0f, 16))})));
  args["p"] = Value(UBits(0b11, 2));
  EXPECT_THAT(simulator.RunFunction(args),
              IsOkAndHolds(Value::Tuple(
                  {Value(UBits(0x00ff, 16)), Value(UBits(0xff00, 16))})));
}

TEST_P(CombinationalGeneratorTest, PrioritySelectMixedBitsAndNonBits) {
  constexpr std::string_view text = R"(
package PrioritySelect

top fn main(p: bits[2], x: (bits[16], bits[16]), y: (bits[16], bits[16]), d: (bits[16], bits[16])) -> (bits[32], bits[16], bits[16]) {
  tuple_index.1: bits[16] = tuple_index(x, index=0)
  tuple_index.2: bits[16] = tuple_index(x, index=1)
  tuple_index.3: bits[16] = tuple_index(y, index=0)
  tuple_index.4: bits[16] = tuple_index(y, index=1)
  tuple_index.5: bits[16] = tuple_index(d, index=0)
  tuple_index.6: bits[16] = tuple_index(d, index=1)
  concat.7: bits[32] = concat(tuple_index.1, tuple_index.2)
  concat.8: bits[32] = concat(tuple_index.3, tuple_index.4)
  concat.9: bits[32] = concat(tuple_index.5, tuple_index.6)
  priority_sel.10: bits[32] = priority_sel(p, cases=[concat.7, concat.8], default=concat.9)
  priority_sel.11: (bits[16], bits[16]) = priority_sel(p, cases=[x, y], default=d)
  tuple_index.12: bits[16] = tuple_index(priority_sel.11, index=0)
  tuple_index.13: bits[16] = tuple_index(priority_sel.11, index=1)
  ret tuple.14: (bits[32], bits[16], bits[16]) = tuple(priority_sel.10, tuple_index.12, tuple_index.13)
}
)";
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> package,
                           Parser::ParsePackage(text));

  std::optional<FunctionBase*> top = package->GetTop();
  ASSERT_TRUE(top.has_value());
  XLS_ASSERT_OK_AND_ASSIGN(
      auto result, GenerateCombinationalModule(top.value(), codegen_options()));

  ExpectVerilogEqualToGoldenFile(GoldenFilePath(kTestName, kTestdataPath),
                                 result.verilog_text);

  ModuleSimulator simulator =
      NewModuleSimulator(result.verilog_text, result.signature);
  absl::flat_hash_map<std::string, Value> args = {
      {"x", Value::Tuple({Value(UBits(0x00ff, 16)), Value(UBits(0xff00, 16))})},
      {"y", Value::Tuple({Value(UBits(0xf0f0, 16)), Value(UBits(0x0f0f, 16))})},
      {"d", Value::Tuple({Value(UBits(0xff00, 16)), Value(UBits(0x00ff, 16))})},
  };
  args["p"] = Value(UBits(0b00, 2));
  EXPECT_THAT(simulator.RunFunction(args),
              IsOkAndHolds(Value::Tuple({Value(UBits(0xff0000ff, 32)),
                                         Value(UBits(0xff00, 16)),
                                         Value(UBits(0x00ff, 16))})));
  args["p"] = Value(UBits(0b01, 2));
  EXPECT_THAT(simulator.RunFunction(args),
              IsOkAndHolds(Value::Tuple({Value(UBits(0x00ffff00, 32)),
                                         Value(UBits(0x00ff, 16)),
                                         Value(UBits(0xff00, 16))})));
  args["p"] = Value(UBits(0b10, 2));
  EXPECT_THAT(simulator.RunFunction(args),
              IsOkAndHolds(Value::Tuple({Value(UBits(0xf0f00f0f, 32)),
                                         Value(UBits(0xf0f0, 16)),
                                         Value(UBits(0x0f0f, 16))})));
  args["p"] = Value(UBits(0b11, 2));
  EXPECT_THAT(simulator.RunFunction(args),
              IsOkAndHolds(Value::Tuple({Value(UBits(0x00ffff00, 32)),
                                         Value(UBits(0x00ff, 16)),
                                         Value(UBits(0xff00, 16))})));
}

TEST_P(CombinationalGeneratorTest, UncommonParameterTypes) {
  std::string text = R"(
package UncommonParameterTypes

top fn main(a: bits[32],
        b: (bits[32], ()),
        c: bits[32][3],
        d: (bits[32], bits[32])[1],
        e: (bits[32][2], (), ()),
        f: bits[0],
        g: bits[1]) -> bits[32] {
  tuple_index.1: bits[32] = tuple_index(b, index=0)
  literal.2: bits[32] = literal(value=0)
  array_index.3: bits[32] = array_index(c, indices=[g])
  array_index.4: (bits[32], bits[32]) = array_index(d, indices=[literal.2])
  tuple_index.5: bits[32] = tuple_index(array_index.4, index=1)
  tuple_index.6: bits[32][2] = tuple_index(e, index=0)
  array_index.7: bits[32] = array_index(tuple_index.6, indices=[g])
  ret or.8: bits[32] = or(a, tuple_index.1, array_index.3, tuple_index.5, array_index.7)
}
)";
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> package,
                           Parser::ParsePackage(text));

  std::optional<FunctionBase*> top = package->GetTop();
  ASSERT_TRUE(top.has_value());
  XLS_ASSERT_OK_AND_ASSIGN(
      auto result, GenerateCombinationalModule(top.value(), codegen_options()));
  ExpectVerilogEqualToGoldenFile(GoldenFilePath(kTestName, kTestdataPath),
                                 result.verilog_text);

  ModuleSimulator simulator =
      NewModuleSimulator(result.verilog_text, result.signature);

  std::minstd_rand engine;
  ASSERT_TRUE(top.value()->IsFunction());
  Function* fn = top.value()->AsFunctionOrDie();
  std::vector<Value> arguments = RandomFunctionArguments(fn, engine);
  XLS_ASSERT_OK_AND_ASSIGN(
      Value expected, DropInterpreterEvents(InterpretFunction(fn, arguments)));
  EXPECT_THAT(simulator.RunFunction(arguments), IsOkAndHolds(expected));
}

TEST_P(CombinationalGeneratorTest, ArrayIndexWithBoundsCheck) {
  Package package(TestBaseName());
  FunctionBuilder fb(TestBaseName(), &package);
  Type* u8 = package.GetBitsType(8);
  Type* array_u8 = package.GetArrayType(3, u8);
  auto a = fb.Param("A", array_u8);
  auto index = fb.Param("index", u8);
  fb.ArrayIndex(a, {index});

  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  XLS_ASSERT_OK_AND_ASSIGN(
      auto result, GenerateCombinationalModule(
                       f, codegen_options().array_index_bounds_checking(true)));

  ExpectVerilogEqualToGoldenFile(GoldenFilePath(kTestName, kTestdataPath),
                                 result.verilog_text);

  ModuleSimulator simulator =
      NewModuleSimulator(result.verilog_text, result.signature);
  EXPECT_THAT(
      simulator.RunFunction({{"A", Value::UBitsArray({30, 40, 50}, 8).value()},
                             {"index", Value(UBits(1, 8))}}),
      IsOkAndHolds(Value(UBits(40, 8))));
  EXPECT_THAT(
      simulator.RunFunction({{"A", Value::UBitsArray({30, 40, 50}, 8).value()},
                             {"index", Value(UBits(3, 8))}}),
      IsOkAndHolds(Value(UBits(50, 8))));
  EXPECT_THAT(
      simulator.RunFunction({{"A", Value::UBitsArray({30, 40, 50}, 8).value()},
                             {"index", Value(UBits(42, 8))}}),
      IsOkAndHolds(Value(UBits(50, 8))));

  // The out of bounds value should return the highest index value.
  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<ModuleTestbench> tb,
      ModuleTestbench::CreateFromVerilogText(result.verilog_text, GetFileType(),
                                             result.signature, GetSimulator()));
  XLS_ASSERT_OK_AND_ASSIGN(
      ModuleTestbenchThread * tbt,
      tb->CreateThreadDrivingAllInputs("main", /*default_value=*/ZeroOrX::kX));
  SequentialBlock& seq = tbt->MainBlock();
  seq.Set("A", UBits(0xabcdef, 24));
  seq.Set("index", UBits(42, 8));
  seq.AtEndOfCycle().ExpectEq("out", 0xab);
  XLS_EXPECT_OK(tb->Run());
}

TEST_P(CombinationalGeneratorTest, ArrayIndexWithoutBoundsCheck) {
  Package package(TestBaseName());
  FunctionBuilder fb(TestBaseName(), &package);
  Type* u8 = package.GetBitsType(8);
  Type* array_u8 = package.GetArrayType(3, u8);
  auto a = fb.Param("A", array_u8);
  auto index = fb.Param("index", u8);
  fb.ArrayIndex(a, {index});

  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  XLS_ASSERT_OK_AND_ASSIGN(
      auto result,
      GenerateCombinationalModule(
          f, codegen_options().array_index_bounds_checking(false)));

  ExpectVerilogEqualToGoldenFile(GoldenFilePath(kTestName, kTestdataPath),
                                 result.verilog_text);

  ModuleSimulator simulator =
      NewModuleSimulator(result.verilog_text, result.signature);
  EXPECT_THAT(
      simulator.RunFunction({{"A", Value::UBitsArray({30, 40, 50}, 8).value()},
                             {"index", Value(UBits(1, 8))}}),
      IsOkAndHolds(Value(UBits(40, 8))));
  EXPECT_THAT(
      simulator.RunFunction({{"A", Value::UBitsArray({30, 40, 50}, 8).value()},
                             {"index", Value(UBits(2, 8))}}),
      IsOkAndHolds(Value(UBits(50, 8))));

  // The out of bounds value should return X.
  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<ModuleTestbench> tb,
      ModuleTestbench::CreateFromVerilogText(result.verilog_text, GetFileType(),
                                             result.signature, GetSimulator()));
  XLS_ASSERT_OK_AND_ASSIGN(
      ModuleTestbenchThread * tbt,
      tb->CreateThreadDrivingAllInputs("main", /*default_value=*/ZeroOrX::kX));
  SequentialBlock& seq = tbt->MainBlock();
  seq.Set("A", UBits(0xabcdef, 24));
  seq.Set("index", UBits(3, 8));
  seq.AtEndOfCycle().ExpectX("out");
  XLS_EXPECT_OK(tb->Run());
}

TEST_P(CombinationalGeneratorTest, TwoDArray) {
  // Build up a two dimensional array from scalars, then deconstruct it and do
  // something with the elements.
  Package package(TestBaseName());
  FunctionBuilder fb(TestBaseName(), &package);
  Type* u8 = package.GetBitsType(8);
  auto a = fb.Param("a", u8);
  auto b = fb.Param("b", u8);
  auto c = fb.Param("c", u8);
  auto row_0 = fb.Array({a, b, c}, a.GetType());
  auto row_1 = fb.Array({a, b, c}, a.GetType());
  auto two_d = fb.Array({row_0, row_1}, row_0.GetType());
  fb.Add(fb.ArrayIndex(fb.ArrayIndex(two_d, {fb.Literal(UBits(0, 8))}),
                       {fb.Literal(UBits(2, 8))}),
         fb.ArrayIndex(fb.ArrayIndex(two_d, {fb.Literal(UBits(1, 8))}),
                       {fb.Literal(UBits(1, 8))}));

  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  XLS_ASSERT_OK_AND_ASSIGN(auto result,
                           GenerateCombinationalModule(f, codegen_options()));
  ModuleSimulator simulator =
      NewModuleSimulator(result.verilog_text, result.signature);
  EXPECT_THAT(simulator.RunFunction({{"a", Value(UBits(123, 8))},
                                     {"b", Value(UBits(42, 8))},
                                     {"c", Value(UBits(100, 8))}}),
              IsOkAndHolds(Value(UBits(142, 8))));
}

TEST_P(CombinationalGeneratorTest, ReturnTwoDArray) {
  // Build up a two dimensional array from scalars and return it.
  Package package(TestBaseName());
  FunctionBuilder fb(TestBaseName(), &package);
  Type* u8 = package.GetBitsType(8);
  auto a = fb.Param("a", u8);
  auto b = fb.Param("b", u8);
  auto row_0 = fb.Array({a, b}, a.GetType());
  auto row_1 = fb.Array({b, a}, a.GetType());
  fb.Array({row_0, row_1}, row_0.GetType());

  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  XLS_ASSERT_OK_AND_ASSIGN(auto result,
                           GenerateCombinationalModule(f, codegen_options()));
  ModuleSimulator simulator =
      NewModuleSimulator(result.verilog_text, result.signature);
  EXPECT_THAT(
      simulator.RunFunction(
          {{"a", Value(UBits(123, 8))}, {"b", Value(UBits(42, 8))}}),
      IsOkAndHolds(Value::ArrayOrDie({
          Value::ArrayOrDie({Value(UBits(123, 8)), Value(UBits(42, 8))}),
          Value::ArrayOrDie({Value(UBits(42, 8)), Value(UBits(123, 8))}),
      })));
}

TEST_P(CombinationalGeneratorTest, ArrayUpdateBitElements) {
  std::string text = R"(
package ArrayUpdate

top fn main(idx: bits[2]) -> bits[32][3] {
  literal.5: bits[32][3] = literal(value=[1, 2, 3])
  literal.6: bits[32] = literal(value=99)
  ret updated_array: bits[32][3] = array_update(literal.5, literal.6, indices=[idx])
}
)";
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> package,
                           Parser::ParsePackage(text));

  std::optional<FunctionBase*> top = package->GetTop();
  ASSERT_TRUE(top.has_value());
  XLS_ASSERT_OK_AND_ASSIGN(
      auto result, GenerateCombinationalModule(top.value(), codegen_options()));

  ModuleSimulator simulator =
      NewModuleSimulator(result.verilog_text, result.signature);

  auto make_array = [](absl::Span<const int64_t> values) {
    std::vector<Value> elements;
    for (auto v : values) {
      elements.push_back(Value(UBits(v, 32)));
    }
    absl::StatusOr<Value> array = Value::Array(elements);
    EXPECT_TRUE(array.ok());
    return array.value();
  };

  EXPECT_THAT(simulator.RunFunction({{"idx", Value(UBits(0b00, 2))}}),
              IsOkAndHolds(make_array({99, 2, 3})));
  EXPECT_THAT(simulator.RunFunction({{"idx", Value(UBits(0b01, 2))}}),
              IsOkAndHolds(make_array({1, 99, 3})));
  EXPECT_THAT(simulator.RunFunction({{"idx", Value(UBits(0b10, 2))}}),
              IsOkAndHolds(make_array({1, 2, 99})));
  EXPECT_THAT(simulator.RunFunction({{"idx", Value(UBits(0b11, 2))}}),
              IsOkAndHolds(make_array({1, 2, 3})));
}

TEST_P(CombinationalGeneratorTest, ArrayUpdateArrayElements) {
  std::string text = R"(
package ArrayUpdate

top fn main(idx: bits[2]) -> bits[32][2][3] {
  literal.17: bits[32][2][3] = literal(value=[[1, 2], [3, 4], [5, 6]])
  literal.14: bits[32][2] = literal(value=[98, 99])
  ret updated_array: bits[32][2][3] = array_update(literal.17, literal.14, indices=[idx])
}

)";
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> package,
                           Parser::ParsePackage(text));

  std::optional<FunctionBase*> top = package->GetTop();
  ASSERT_TRUE(top.has_value());
  XLS_ASSERT_OK_AND_ASSIGN(
      auto result, GenerateCombinationalModule(top.value(), codegen_options()));

  ModuleSimulator simulator =
      NewModuleSimulator(result.verilog_text, result.signature);

  auto make_array = [](absl::Span<const int64_t> values) {
    std::vector<Value> elements;
    for (auto v : values) {
      elements.push_back(Value(UBits(v, 32)));
    }
    absl::StatusOr<Value> array = Value::Array(elements);
    EXPECT_TRUE(array.ok());
    return array.value();
  };

  auto make_array_of_values = [&](absl::Span<const Value> values) {
    std::vector<Value> elements;
    for (const auto& array : values) {
      elements.push_back(array);
    }
    absl::StatusOr<Value> array_of_values = Value::Array(elements);
    EXPECT_TRUE(array_of_values.ok());
    return array_of_values.value();
  };

  EXPECT_THAT(
      simulator.RunFunction({{"idx", Value(UBits(0b00, 2))}}),
      IsOkAndHolds(make_array_of_values(
          {make_array({98, 99}), make_array({3, 4}), make_array({5, 6})})));
  EXPECT_THAT(
      simulator.RunFunction({{"idx", Value(UBits(0b01, 2))}}),
      IsOkAndHolds(make_array_of_values(
          {make_array({1, 2}), make_array({98, 99}), make_array({5, 6})})));
  EXPECT_THAT(
      simulator.RunFunction({{"idx", Value(UBits(0b10, 2))}}),
      IsOkAndHolds(make_array_of_values(
          {make_array({1, 2}), make_array({3, 4}), make_array({98, 99})})));
  EXPECT_THAT(
      simulator.RunFunction({{"idx", Value(UBits(0b11, 2))}}),
      IsOkAndHolds(make_array_of_values(
          {make_array({1, 2}), make_array({3, 4}), make_array({5, 6})})));
}

TEST_P(CombinationalGeneratorTest, ArrayUpdateTupleElements) {
  std::string text = R"(
package ArrayUpdate

top fn main(idx: bits[2]) -> (bits[32], bits[32])[3] {
  literal.17: (bits[32], bits[32])[3] = literal(value=[(1,2),(3,4),(5,6)])
  literal.14: (bits[32], bits[32]) = literal(value=(98, 99))
  ret array_update.15: (bits[32], bits[32])[3] = array_update(literal.17, literal.14, indices=[idx])
}

)";
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> package,
                           Parser::ParsePackage(text));

  std::optional<FunctionBase*> top = package->GetTop();
  ASSERT_TRUE(top.has_value());
  XLS_ASSERT_OK_AND_ASSIGN(
      auto result, GenerateCombinationalModule(top.value(), codegen_options()));

  ModuleSimulator simulator =
      NewModuleSimulator(result.verilog_text, result.signature);

  auto make_tuple = [](absl::Span<const int64_t> values) {
    std::vector<Value> elements;
    for (auto v : values) {
      elements.push_back(Value(UBits(v, 32)));
    }
    absl::StatusOr<Value> tuple = Value::Tuple(elements);
    EXPECT_TRUE(tuple.ok());
    return tuple.value();
  };

  auto make_array_of_values = [&](absl::Span<const Value> values) {
    std::vector<Value> elements;
    for (const auto& array : values) {
      elements.push_back(array);
    }
    absl::StatusOr<Value> array_of_values = Value::Array(elements);
    EXPECT_TRUE(array_of_values.ok());
    return array_of_values.value();
  };

  EXPECT_THAT(
      simulator.RunFunction({{"idx", Value(UBits(0b00, 2))}}),
      IsOkAndHolds(make_array_of_values(
          {make_tuple({98, 99}), make_tuple({3, 4}), make_tuple({5, 6})})));
  EXPECT_THAT(
      simulator.RunFunction({{"idx", Value(UBits(0b01, 2))}}),
      IsOkAndHolds(make_array_of_values(
          {make_tuple({1, 2}), make_tuple({98, 99}), make_tuple({5, 6})})));
  EXPECT_THAT(
      simulator.RunFunction({{"idx", Value(UBits(0b10, 2))}}),
      IsOkAndHolds(make_array_of_values(
          {make_tuple({1, 2}), make_tuple({3, 4}), make_tuple({98, 99})})));
  EXPECT_THAT(
      simulator.RunFunction({{"idx", Value(UBits(0b11, 2))}}),
      IsOkAndHolds(make_array_of_values(
          {make_tuple({1, 2}), make_tuple({3, 4}), make_tuple({5, 6})})));
}

TEST_P(CombinationalGeneratorTest, ArrayUpdateTupleWithArrayElements) {
  std::string text = R"(
package ArrayUpdate

top fn main(idx: bits[2]) -> (bits[32], bits[8][2])[2] {
  literal.17: (bits[32], bits[8][2])[2] = literal(value=[(1,[2,3]),(4,[5,6])])
  literal.14: (bits[32], bits[8][2]) = literal(value=(98, [99, 100]))
  ret array_update.15: (bits[32], bits[8][2])[2] = array_update(literal.17, literal.14, indices=[idx])
}

)";
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> package,
                           Parser::ParsePackage(text));

  std::optional<FunctionBase*> top = package->GetTop();
  ASSERT_TRUE(top.has_value());
  XLS_ASSERT_OK_AND_ASSIGN(
      auto result, GenerateCombinationalModule(top.value(), codegen_options()));

  ModuleSimulator simulator =
      NewModuleSimulator(result.verilog_text, result.signature);

  auto make_array = [](absl::Span<const int64_t> values) {
    std::vector<Value> elements;
    for (auto v : values) {
      elements.push_back(Value(UBits(v, 8)));
    }
    absl::StatusOr<Value> array = Value::Array(elements);
    EXPECT_TRUE(array.ok());
    return array.value();
  };

  auto make_tuple = [](absl::Span<const Value> values) {
    std::vector<Value> elements;
    for (const auto& v : values) {
      elements.push_back(v);
    }
    absl::StatusOr<Value> tuple = Value::Tuple(elements);
    EXPECT_TRUE(tuple.ok());
    return tuple.value();
  };

  auto make_array_of_values = [&](absl::Span<const Value> values) {
    std::vector<Value> elements;
    for (const auto& array : values) {
      elements.push_back(array);
    }
    absl::StatusOr<Value> array_of_values = Value::Array(elements);
    EXPECT_TRUE(array_of_values.ok());
    return array_of_values.value();
  };

  EXPECT_THAT(
      simulator.RunFunction({{"idx", Value(UBits(0b01, 2))}}),
      IsOkAndHolds(make_array_of_values(
          {make_tuple({Value(UBits(1, 32)), make_array({2, 3})}),
           make_tuple({Value(UBits(98, 32)), make_array({99, 100})})})));
}

TEST_P(CombinationalGeneratorTest, BuildComplicatedType) {
  Package package(TestBaseName());
  FunctionBuilder fb(TestBaseName(), &package);
  Type* u8 = package.GetBitsType(8);
  // Construct some terrible abomination of tuples and arrays.
  auto a = fb.Param("a", u8);
  auto b = fb.Param("b", u8);
  auto c = fb.Param("c", u8);
  auto row_0 = fb.Array({a, b}, a.GetType());
  auto row_1 = fb.Array({b, a}, a.GetType());
  auto ar = fb.Array({row_0, row_1}, row_0.GetType());
  auto tuple = fb.Tuple({ar, a});
  // Deconstruct it and return some scalar element.
  fb.ArrayIndex(fb.ArrayIndex(fb.TupleIndex(tuple, 0), {a}), {c});

  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  XLS_ASSERT_OK_AND_ASSIGN(auto result,
                           GenerateCombinationalModule(f, codegen_options()));
  ModuleSimulator simulator =
      NewModuleSimulator(result.verilog_text, result.signature);
  EXPECT_THAT(simulator.RunFunction({{"a", Value(UBits(0, 8))},
                                     {"b", Value(UBits(42, 8))},
                                     {"c", Value(UBits(1, 8))}}),
              IsOkAndHolds(Value(UBits(42, 8))));
}

TEST_P(CombinationalGeneratorTest, ArrayShapedSel) {
  Package package(TestBaseName());
  FunctionBuilder fb(TestBaseName(), &package);
  BValue p = fb.Param("p", package.GetBitsType(8));
  BValue x = fb.Param("x", package.GetArrayType(3, package.GetBitsType(8)));
  BValue y = fb.Param("y", package.GetArrayType(3, package.GetBitsType(8)));
  BValue z = fb.Param("z", package.GetArrayType(3, package.GetBitsType(8)));
  BValue d = fb.Param("d", package.GetArrayType(3, package.GetBitsType(8)));
  fb.Select(p, {x, y, z}, /*default_value=*/d);

  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  XLS_ASSERT_OK_AND_ASSIGN(auto result,
                           GenerateCombinationalModule(f, codegen_options()));
  ModuleSimulator simulator =
      NewModuleSimulator(result.verilog_text, result.signature);
  XLS_ASSERT_OK_AND_ASSIGN(
      Value x_in,
      Parser::ParseTypedValue("[bits[8]:0xa, bits[8]:0xb, bits[8]:0xc]"));
  XLS_ASSERT_OK_AND_ASSIGN(
      Value y_in,
      Parser::ParseTypedValue("[bits[8]:0x1, bits[8]:0x2, bits[8]:0x3]"));
  XLS_ASSERT_OK_AND_ASSIGN(
      Value z_in,
      Parser::ParseTypedValue("[bits[8]:0x4, bits[8]:0x5, bits[8]:0x6]"));
  XLS_ASSERT_OK_AND_ASSIGN(
      Value d_in,
      Parser::ParseTypedValue("[bits[8]:0x7, bits[8]:0x8, bits[8]:0x9]"));
  EXPECT_THAT(simulator.RunFunction({{"p", Value(UBits(0, 8))},
                                     {"x", x_in},
                                     {"y", y_in},
                                     {"z", z_in},
                                     {"d", d_in}}),
              IsOkAndHolds(x_in));
  EXPECT_THAT(simulator.RunFunction({{"p", Value(UBits(1, 8))},
                                     {"x", x_in},
                                     {"y", y_in},
                                     {"z", z_in},
                                     {"d", d_in}}),
              IsOkAndHolds(y_in));
  EXPECT_THAT(simulator.RunFunction({{"p", Value(UBits(2, 8))},
                                     {"x", x_in},
                                     {"y", y_in},
                                     {"z", z_in},
                                     {"d", d_in}}),
              IsOkAndHolds(z_in));
  EXPECT_THAT(simulator.RunFunction({{"p", Value(UBits(3, 8))},
                                     {"x", x_in},
                                     {"y", y_in},
                                     {"z", z_in},
                                     {"d", d_in}}),
              IsOkAndHolds(d_in));
  EXPECT_THAT(simulator.RunFunction({{"p", Value(UBits(100, 8))},
                                     {"x", x_in},
                                     {"y", y_in},
                                     {"z", z_in},
                                     {"d", d_in}}),
              IsOkAndHolds(d_in));
}

TEST_P(CombinationalGeneratorTest, ArrayShapedSelNoDefault) {
  Package package(TestBaseName());
  FunctionBuilder fb(TestBaseName(), &package);
  BValue p = fb.Param("p", package.GetBitsType(1));
  BValue x = fb.Param("x", package.GetArrayType(3, package.GetBitsType(8)));
  BValue y = fb.Param("y", package.GetArrayType(3, package.GetBitsType(8)));
  fb.Select(p, {x, y});

  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  XLS_ASSERT_OK_AND_ASSIGN(auto result,
                           GenerateCombinationalModule(f, codegen_options()));
  ModuleSimulator simulator =
      NewModuleSimulator(result.verilog_text, result.signature);
  XLS_ASSERT_OK_AND_ASSIGN(
      Value x_in,
      Parser::ParseTypedValue("[bits[8]:0xa, bits[8]:0xb, bits[8]:0xc]"));
  XLS_ASSERT_OK_AND_ASSIGN(
      Value y_in,
      Parser::ParseTypedValue("[bits[8]:0x1, bits[8]:0x2, bits[8]:0x3]"));
  EXPECT_THAT(simulator.RunFunction(
                  {{"p", Value(UBits(0, 1))}, {"x", x_in}, {"y", y_in}}),
              IsOkAndHolds(x_in));
  EXPECT_THAT(simulator.RunFunction(
                  {{"p", Value(UBits(1, 1))}, {"x", x_in}, {"y", y_in}}),
              IsOkAndHolds(y_in));
}

TEST_P(CombinationalGeneratorTest, ArrayShapedOneHotSelect) {
  Package package(TestBaseName());
  FunctionBuilder fb(TestBaseName(), &package);
  BValue s = fb.Param("s", package.GetBitsType(2));
  BValue x = fb.Param("x", package.GetArrayType(2, package.GetBitsType(8)));
  BValue y = fb.Param("y", package.GetArrayType(2, package.GetBitsType(8)));
  fb.OneHotSelect(s, {x, y});

  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  XLS_ASSERT_OK_AND_ASSIGN(auto result,
                           GenerateCombinationalModule(f, codegen_options()));
  ModuleSimulator simulator =
      NewModuleSimulator(result.verilog_text, result.signature);
  XLS_ASSERT_OK_AND_ASSIGN(
      Value x_in, Parser::ParseTypedValue("[bits[8]:0x0f, bits[8]:0xf0]"));
  XLS_ASSERT_OK_AND_ASSIGN(
      Value y_in, Parser::ParseTypedValue("[bits[8]:0xab, bits[8]:0xcd]"));
  EXPECT_THAT(simulator.RunFunction(
                  {{"s", Value(UBits(0b00, 2))}, {"x", x_in}, {"y", y_in}}),
              IsOkAndHolds(Value::UBitsArray({0x0, 0x0}, 8).value()));
  EXPECT_THAT(simulator.RunFunction(
                  {{"s", Value(UBits(0b01, 2))}, {"x", x_in}, {"y", y_in}}),
              IsOkAndHolds(Value::UBitsArray({0x0f, 0xf0}, 8).value()));
  EXPECT_THAT(simulator.RunFunction(
                  {{"s", Value(UBits(0b10, 2))}, {"x", x_in}, {"y", y_in}}),
              IsOkAndHolds(Value::UBitsArray({0xab, 0xcd}, 8).value()));
  EXPECT_THAT(simulator.RunFunction(
                  {{"s", Value(UBits(0b11, 2))}, {"x", x_in}, {"y", y_in}}),
              IsOkAndHolds(Value::UBitsArray({0xaf, 0xfd}, 8).value()));
}

TEST_P(CombinationalGeneratorTest, ArrayConcatArrayOfBits) {
  Package package(TestBaseName());

  std::string ir_text = R"(
  fn f(a0: bits[32][2], a1: bits[32][3]) -> bits[32][7] {
    array_concat.3: bits[32][5] = array_concat(a0, a1)
    ret array_concat.4: bits[32][7] = array_concat(array_concat.3, a0)
  }
  )";
  XLS_ASSERT_OK_AND_ASSIGN(Function * function,
                           Parser::ParseFunction(ir_text, &package));

  XLS_ASSERT_OK_AND_ASSIGN(
      auto result, GenerateCombinationalModule(function, codegen_options()));

  ModuleSimulator simulator =
      NewModuleSimulator(result.verilog_text, result.signature);

  XLS_ASSERT_OK_AND_ASSIGN(Value a0, Value::UBitsArray({1, 2}, 32));
  XLS_ASSERT_OK_AND_ASSIGN(Value a1, Value::UBitsArray({3, 4, 5}, 32));
  XLS_ASSERT_OK_AND_ASSIGN(Value ret,
                           Value::UBitsArray({1, 2, 3, 4, 5, 1, 2}, 32));

  EXPECT_THAT(simulator.RunFunction({{"a0", a0}, {"a1", a1}}),
              IsOkAndHolds(ret));
}

TEST_P(CombinationalGeneratorTest, ArrayConcatArrayOfBitsMixedOperands) {
  Package package(TestBaseName());

  std::string ir_text = R"(
  fn f(a0: bits[32][2], a1: bits[32][3], a2: bits[32][1]) -> bits[32][7] {
    array_concat.4: bits[32][1] = array_concat(a2)
    array_concat.5: bits[32][2] = array_concat(array_concat.4, array_concat.4)
    array_concat.6: bits[32][7] = array_concat(a0, array_concat.5, a1)
    ret array_concat.7: bits[32][7] = array_concat(array_concat.6)
  }
  )";
  XLS_ASSERT_OK_AND_ASSIGN(Function * function,
                           Parser::ParseFunction(ir_text, &package));

  XLS_ASSERT_OK_AND_ASSIGN(
      auto result, GenerateCombinationalModule(function, codegen_options()));

  ModuleSimulator simulator =
      NewModuleSimulator(result.verilog_text, result.signature);

  XLS_ASSERT_OK_AND_ASSIGN(Value a0, Value::UBitsArray({1, 2}, 32));
  XLS_ASSERT_OK_AND_ASSIGN(Value a1, Value::UBitsArray({3, 4, 5}, 32));
  XLS_ASSERT_OK_AND_ASSIGN(Value a2, Value::SBitsArray({-1}, 32));
  XLS_ASSERT_OK_AND_ASSIGN(Value ret,
                           Value::SBitsArray({1, 2, -1, -1, 3, 4, 5}, 32));

  EXPECT_THAT(simulator.RunFunction({{"a0", a0}, {"a1", a1}, {"a2", a2}}),
              IsOkAndHolds(ret));
}

TEST_P(CombinationalGeneratorTest, InterpretArrayConcatArraysOfArrays) {
  Package package(TestBaseName());

  std::string ir_text = R"(
  fn f() -> bits[32][2][3] {
    literal.1: bits[32][2][2] = literal(value=[[1, 2], [3, 4]])
    literal.2: bits[32][2][1] = literal(value=[[5, 6]])

    ret array_concat.3: bits[32][2][3] = array_concat(literal.2, literal.1)
  }
  )";
  XLS_ASSERT_OK_AND_ASSIGN(Function * function,
                           Parser::ParseFunction(ir_text, &package));

  XLS_ASSERT_OK_AND_ASSIGN(
      auto result, GenerateCombinationalModule(function, codegen_options()));

  ModuleSimulator simulator =
      NewModuleSimulator(result.verilog_text, result.signature);

  XLS_ASSERT_OK_AND_ASSIGN(Value ret,
                           Value::SBits2DArray({{5, 6}, {1, 2}, {3, 4}}, 32));

  std::vector<Value> args;
  EXPECT_THAT(simulator.RunFunction(args), IsOkAndHolds(ret));
}

TEST_P(CombinationalGeneratorTest, ArrayIndexSimpleArray) {
  Package package(TestBaseName());
  FunctionBuilder fb(TestBaseName(), &package);
  Type* u8 = package.GetBitsType(8);
  Type* u16 = package.GetBitsType(16);
  auto a = fb.Param("a", package.GetArrayType(3, u8));
  auto idx = fb.Param("idx", u16);
  auto ret = fb.ArrayIndex(a, {idx});
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.BuildWithReturnValue(ret));
  XLS_ASSERT_OK_AND_ASSIGN(auto result,
                           GenerateCombinationalModule(f, codegen_options()));

  ExpectVerilogEqualToGoldenFile(GoldenFilePath(kTestName, kTestdataPath),
                                 result.verilog_text);

  ModuleSimulator simulator =
      NewModuleSimulator(result.verilog_text, result.signature);
  EXPECT_THAT(
      simulator.RunFunction({{"a", Value::UBitsArray({11, 22, 33}, 8).value()},
                             {"idx", Value(UBits(2, 16))}}),
      IsOkAndHolds(Value(UBits(33, 8))));

  // OOB access should return the last element.
  EXPECT_THAT(
      simulator.RunFunction({{"a", Value::UBitsArray({11, 22, 33}, 8).value()},
                             {"idx", Value(UBits(42, 16))}}),
      IsOkAndHolds(Value(UBits(33, 8))));
}

TEST_P(CombinationalGeneratorTest, ArrayIndexWithNarrowIndex) {
  // An array index with a sufficiently narrow index that OOB access is not
  // possible.
  Package package(TestBaseName());
  FunctionBuilder fb(TestBaseName(), &package);
  Type* u8 = package.GetBitsType(8);
  Type* u2 = package.GetBitsType(2);
  auto a = fb.Param("a", package.GetArrayType(4, u8));
  auto idx = fb.Param("idx", u2);
  auto ret = fb.ArrayIndex(a, {idx});
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.BuildWithReturnValue(ret));
  XLS_ASSERT_OK_AND_ASSIGN(auto result,
                           GenerateCombinationalModule(f, codegen_options()));

  ExpectVerilogEqualToGoldenFile(GoldenFilePath(kTestName, kTestdataPath),
                                 result.verilog_text);

  ModuleSimulator simulator =
      NewModuleSimulator(result.verilog_text, result.signature);
  EXPECT_THAT(simulator.RunFunction(
                  {{"a", Value::UBitsArray({11, 22, 33, 44}, 8).value()},
                   {"idx", Value(UBits(1, 2))}}),
              IsOkAndHolds(Value(UBits(22, 8))));
}

TEST_P(CombinationalGeneratorTest, ArrayIndexWithLiteralIndex) {
  Package package(TestBaseName());
  FunctionBuilder fb(TestBaseName(), &package);
  Type* u8 = package.GetBitsType(8);
  auto a = fb.Param("a", package.GetArrayType(4, u8));
  auto idx = fb.Literal(UBits(3, 42));
  auto ret = fb.ArrayIndex(a, {idx});
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.BuildWithReturnValue(ret));
  XLS_ASSERT_OK_AND_ASSIGN(auto result,
                           GenerateCombinationalModule(f, codegen_options()));

  ExpectVerilogEqualToGoldenFile(GoldenFilePath(kTestName, kTestdataPath),
                                 result.verilog_text);

  ModuleSimulator simulator =
      NewModuleSimulator(result.verilog_text, result.signature);
  EXPECT_THAT(simulator.RunFunction(
                  {{"a", Value::UBitsArray({11, 22, 33, 44}, 8).value()}}),
              IsOkAndHolds(Value(UBits(44, 8))));
}

TEST_P(CombinationalGeneratorTest, ArrayIndexNilIndex) {
  Package package(TestBaseName());
  FunctionBuilder fb(TestBaseName(), &package);
  Type* u8 = package.GetBitsType(8);
  auto a = fb.Param("a", package.GetArrayType(3, u8));
  auto ret = fb.ArrayIndex(a, {});
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.BuildWithReturnValue(ret));
  XLS_ASSERT_OK_AND_ASSIGN(auto result,
                           GenerateCombinationalModule(f, codegen_options()));

  ExpectVerilogEqualToGoldenFile(GoldenFilePath(kTestName, kTestdataPath),
                                 result.verilog_text);

  ModuleSimulator simulator =
      NewModuleSimulator(result.verilog_text, result.signature);
  EXPECT_THAT(simulator.RunFunction(
                  {{"a", Value::UBitsArray({11, 22, 33}, 8).value()}}),
              IsOkAndHolds(Value::UBitsArray({11, 22, 33}, 8).value()));
}

TEST_P(CombinationalGeneratorTest, ArrayIndex2DArrayIndexSingleElement) {
  Package package(TestBaseName());
  FunctionBuilder fb(TestBaseName(), &package);
  Type* u8 = package.GetBitsType(8);
  Type* u16 = package.GetBitsType(16);
  auto a = fb.Param("a", package.GetArrayType(2, package.GetArrayType(3, u8)));
  auto idx0 = fb.Param("idx0", u16);
  auto idx1 = fb.Param("idx1", u16);
  auto ret = fb.ArrayIndex(a, {idx0, idx1});
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.BuildWithReturnValue(ret));
  XLS_ASSERT_OK_AND_ASSIGN(auto result,
                           GenerateCombinationalModule(f, codegen_options()));

  ExpectVerilogEqualToGoldenFile(GoldenFilePath(kTestName, kTestdataPath),
                                 result.verilog_text);

  ModuleSimulator simulator =
      NewModuleSimulator(result.verilog_text, result.signature);
  EXPECT_THAT(
      simulator.RunFunction(
          {{"a", Value::UBits2DArray({{11, 22, 33}, {44, 55, 66}}, 8).value()},
           {"idx0", Value(UBits(0, 16))},
           {"idx1", Value(UBits(1, 16))}}),
      IsOkAndHolds(Value(UBits(22, 8))));
}

TEST_P(CombinationalGeneratorTest, ArrayIndex2DArrayIndexSubArray) {
  Package package(TestBaseName());
  FunctionBuilder fb(TestBaseName(), &package);
  Type* u8 = package.GetBitsType(8);
  Type* u16 = package.GetBitsType(16);
  auto a = fb.Param("a", package.GetArrayType(2, package.GetArrayType(3, u8)));
  auto idx = fb.Param("idx", u16);
  auto ret = fb.ArrayIndex(a, {idx});
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.BuildWithReturnValue(ret));
  XLS_ASSERT_OK_AND_ASSIGN(auto result,
                           GenerateCombinationalModule(f, codegen_options()));

  ExpectVerilogEqualToGoldenFile(GoldenFilePath(kTestName, kTestdataPath),
                                 result.verilog_text);
  ModuleSimulator simulator =
      NewModuleSimulator(result.verilog_text, result.signature);
  EXPECT_THAT(
      simulator.RunFunction(
          {{"a", Value::UBits2DArray({{11, 22, 33}, {44, 55, 66}}, 8).value()},
           {"idx", Value(UBits(0, 16))}}),
      IsOkAndHolds(Value::UBitsArray({11, 22, 33}, 8).value()));
  EXPECT_THAT(
      simulator.RunFunction(
          {{"a", Value::UBits2DArray({{11, 22, 33}, {44, 55, 66}}, 8).value()},
           {"idx", Value(UBits(1, 16))}}),
      IsOkAndHolds(Value::UBitsArray({44, 55, 66}, 8).value()));
  EXPECT_THAT(
      simulator.RunFunction(
          {{"a", Value::UBits2DArray({{11, 22, 33}, {44, 55, 66}}, 8).value()},
           {"idx", Value(UBits(42, 16))}}),
      IsOkAndHolds(Value::UBitsArray({44, 55, 66}, 8).value()));
}

TEST_P(CombinationalGeneratorTest, ArrayUpdateLiteralIndex) {
  Package package(TestBaseName());
  FunctionBuilder fb(TestBaseName(), &package);
  Type* u8 = package.GetBitsType(8);
  auto a = fb.Param("a", package.GetArrayType(3, u8));
  auto update_value = fb.Param("value", u8);
  auto idx = fb.Literal(UBits(1, 16));
  auto ret = fb.ArrayUpdate(a, update_value, {idx});
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.BuildWithReturnValue(ret));
  XLS_ASSERT_OK_AND_ASSIGN(auto result,
                           GenerateCombinationalModule(f, codegen_options()));

  ExpectVerilogEqualToGoldenFile(GoldenFilePath(kTestName, kTestdataPath),
                                 result.verilog_text);

  ModuleSimulator simulator =
      NewModuleSimulator(result.verilog_text, result.signature);
  EXPECT_THAT(
      simulator.RunFunction({{"a", Value::UBitsArray({11, 22, 33}, 8).value()},
                             {"value", Value(UBits(123, 8))}}),
      IsOkAndHolds(Value::UBitsArray({11, 123, 33}, 8).value()));
}

TEST_P(CombinationalGeneratorTest, ArrayUpdateVariableIndex) {
  Package package(TestBaseName());
  FunctionBuilder fb(TestBaseName(), &package);
  Type* u8 = package.GetBitsType(8);
  auto a = fb.Param("a", package.GetArrayType(3, u8));
  auto update_value = fb.Param("value", u8);
  auto idx = fb.Param("idx", package.GetBitsType(32));
  auto ret = fb.ArrayUpdate(a, update_value, {idx});
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.BuildWithReturnValue(ret));
  XLS_ASSERT_OK_AND_ASSIGN(auto result,
                           GenerateCombinationalModule(f, codegen_options()));

  ExpectVerilogEqualToGoldenFile(GoldenFilePath(kTestName, kTestdataPath),
                                 result.verilog_text);
  ModuleSimulator simulator =
      NewModuleSimulator(result.verilog_text, result.signature);
  EXPECT_THAT(
      simulator.RunFunction({{"a", Value::UBitsArray({11, 22, 33}, 8).value()},
                             {"idx", Value(UBits(0, 32))},
                             {"value", Value(UBits(123, 8))}}),
      IsOkAndHolds(Value::UBitsArray({123, 22, 33}, 8).value()));
  // Out-of-bounds should just return the original array.
  EXPECT_THAT(
      simulator.RunFunction({{"a", Value::UBitsArray({11, 22, 33}, 8).value()},
                             {"idx", Value(UBits(3, 32))},
                             {"value", Value(UBits(123, 8))}}),
      IsOkAndHolds(Value::UBitsArray({11, 22, 33}, 8).value()));
}

TEST_P(CombinationalGeneratorTest, ArrayUpdate2DLiteralIndex) {
  Package package(TestBaseName());
  FunctionBuilder fb(TestBaseName(), &package);
  Type* u8 = package.GetBitsType(8);
  auto a = fb.Param("a", package.GetArrayType(2, package.GetArrayType(3, u8)));
  auto update_value = fb.Param("value", u8);
  auto idx0 = fb.Literal(UBits(0, 32));
  auto idx1 = fb.Literal(UBits(2, 14));
  auto ret = fb.ArrayUpdate(a, update_value, {idx0, idx1});
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.BuildWithReturnValue(ret));
  XLS_ASSERT_OK_AND_ASSIGN(auto result,
                           GenerateCombinationalModule(f, codegen_options()));

  ExpectVerilogEqualToGoldenFile(GoldenFilePath(kTestName, kTestdataPath),
                                 result.verilog_text);
  ModuleSimulator simulator =
      NewModuleSimulator(result.verilog_text, result.signature);
  EXPECT_THAT(
      simulator.RunFunction(
          {{"a", Value::UBits2DArray({{11, 22, 33}, {44, 55, 66}}, 8).value()},
           {"value", Value(UBits(123, 8))}}),
      IsOkAndHolds(
          Value::UBits2DArray({{11, 22, 123}, {44, 55, 66}}, 8).value()));
}

TEST_P(CombinationalGeneratorTest, ArrayUpdate2DVariableIndex) {
  Package package(TestBaseName());
  FunctionBuilder fb(TestBaseName(), &package);
  Type* u8 = package.GetBitsType(8);
  auto a = fb.Param("a", package.GetArrayType(2, package.GetArrayType(3, u8)));
  auto update_value = fb.Param("value", u8);
  auto idx0 = fb.Param("idx0", package.GetBitsType(32));
  auto idx1 = fb.Param("idx1", package.GetBitsType(32));
  auto ret = fb.ArrayUpdate(a, update_value, {idx0, idx1});
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.BuildWithReturnValue(ret));
  XLS_ASSERT_OK_AND_ASSIGN(auto result,
                           GenerateCombinationalModule(f, codegen_options()));

  ExpectVerilogEqualToGoldenFile(GoldenFilePath(kTestName, kTestdataPath),
                                 result.verilog_text);
  ModuleSimulator simulator =
      NewModuleSimulator(result.verilog_text, result.signature);
  EXPECT_THAT(
      simulator.RunFunction(
          {{"a", Value::UBits2DArray({{11, 22, 33}, {44, 55, 66}}, 8).value()},
           {"value", Value(UBits(123, 8))},
           {"idx0", Value(UBits(1, 32))},
           {"idx1", Value(UBits(0, 32))}}),
      IsOkAndHolds(
          Value::UBits2DArray({{11, 22, 33}, {123, 55, 66}}, 8).value()));
  // Out-of-bounds should just return the original array.
  EXPECT_THAT(
      simulator.RunFunction(
          {{"a", Value::UBits2DArray({{11, 22, 33}, {44, 55, 66}}, 8).value()},
           {"value", Value(UBits(123, 8))},
           {"idx0", Value(UBits(1, 32))},
           {"idx1", Value(UBits(44, 32))}}),
      IsOkAndHolds(
          Value::UBits2DArray({{11, 22, 33}, {44, 55, 66}}, 8).value()));
  EXPECT_THAT(
      simulator.RunFunction(
          {{"a", Value::UBits2DArray({{11, 22, 33}, {44, 55, 66}}, 8).value()},
           {"value", Value(UBits(123, 8))},
           {"idx0", Value(UBits(11, 32))},
           {"idx1", Value(UBits(0, 32))}}),
      IsOkAndHolds(
          Value::UBits2DArray({{11, 22, 33}, {44, 55, 66}}, 8).value()));
}

TEST_P(CombinationalGeneratorTest, ArrayUpdate2DLiteralAndVariableIndex) {
  Package package(TestBaseName());
  FunctionBuilder fb(TestBaseName(), &package);
  Type* u8 = package.GetBitsType(8);
  auto a = fb.Param("a", package.GetArrayType(2, package.GetArrayType(3, u8)));
  auto update_value = fb.Param("value", u8);
  auto idx0 = fb.Param("idx", package.GetBitsType(32));
  auto idx1 = fb.Literal(UBits(2, 14));
  auto ret = fb.ArrayUpdate(a, update_value, {idx0, idx1});
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.BuildWithReturnValue(ret));
  XLS_ASSERT_OK_AND_ASSIGN(auto result,
                           GenerateCombinationalModule(f, codegen_options()));

  ModuleSimulator simulator =
      NewModuleSimulator(result.verilog_text, result.signature);
  EXPECT_THAT(
      simulator.RunFunction(
          {{"a", Value::UBits2DArray({{11, 22, 33}, {44, 55, 66}}, 8).value()},
           {"value", Value(UBits(123, 8))},
           {"idx", Value(UBits(0, 32))}}),
      IsOkAndHolds(
          Value::UBits2DArray({{11, 22, 123}, {44, 55, 66}}, 8).value()));
  // Out-of-bounds should just return the original array.
  EXPECT_THAT(
      simulator.RunFunction(
          {{"a", Value::UBits2DArray({{11, 22, 33}, {44, 55, 66}}, 8).value()},
           {"value", Value(UBits(123, 8))},
           {"idx", Value(UBits(10, 32))}}),
      IsOkAndHolds(
          Value::UBits2DArray({{11, 22, 33}, {44, 55, 66}}, 8).value()));
}

TEST_P(CombinationalGeneratorTest, ArrayUpdate2DUpdateArrayLiteralIndex) {
  Package package(TestBaseName());
  FunctionBuilder fb(TestBaseName(), &package);
  Type* u8 = package.GetBitsType(8);
  auto a = fb.Param("a", package.GetArrayType(2, package.GetArrayType(3, u8)));
  auto update_value = fb.Param("value", package.GetArrayType(3, u8));
  auto idx = fb.Literal(UBits(1, 14));
  auto ret = fb.ArrayUpdate(a, update_value, {idx});
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.BuildWithReturnValue(ret));
  XLS_ASSERT_OK_AND_ASSIGN(auto result,
                           GenerateCombinationalModule(f, codegen_options()));

  ExpectVerilogEqualToGoldenFile(GoldenFilePath(kTestName, kTestdataPath),
                                 result.verilog_text);
  ModuleSimulator simulator =
      NewModuleSimulator(result.verilog_text, result.signature);
  EXPECT_THAT(
      simulator.RunFunction(
          {{"a", Value::UBits2DArray({{11, 22, 33}, {44, 55, 66}}, 8).value()},
           {"value", Value::UBitsArray({101, 102, 103}, 8).value()}}),
      IsOkAndHolds(
          Value::UBits2DArray({{11, 22, 33}, {101, 102, 103}}, 8).value()));
}

TEST_P(CombinationalGeneratorTest, ArrayUpdate2DUpdateArrayVariableIndex) {
  Package package(TestBaseName());
  FunctionBuilder fb(TestBaseName(), &package);
  Type* u8 = package.GetBitsType(8);
  auto a = fb.Param("a", package.GetArrayType(2, package.GetArrayType(3, u8)));
  auto update_value = fb.Param("value", package.GetArrayType(3, u8));
  auto idx = fb.Param("idx", package.GetBitsType(37));
  auto ret = fb.ArrayUpdate(a, update_value, {idx});
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.BuildWithReturnValue(ret));
  XLS_ASSERT_OK_AND_ASSIGN(auto result,
                           GenerateCombinationalModule(f, codegen_options()));

  ExpectVerilogEqualToGoldenFile(GoldenFilePath(kTestName, kTestdataPath),
                                 result.verilog_text);
  ModuleSimulator simulator =
      NewModuleSimulator(result.verilog_text, result.signature);
  EXPECT_THAT(
      simulator.RunFunction(
          {{"a", Value::UBits2DArray({{11, 22, 33}, {44, 55, 66}}, 8).value()},
           {"value", Value::UBitsArray({101, 102, 103}, 8).value()},
           {"idx", Value(UBits(1, 37))}}),
      IsOkAndHolds(
          Value::UBits2DArray({{11, 22, 33}, {101, 102, 103}}, 8).value()));
  // Out-of-bounds should just return the original array.
  EXPECT_THAT(
      simulator.RunFunction(
          {{"a", Value::UBits2DArray({{11, 22, 33}, {44, 55, 66}}, 8).value()},
           {"value", Value::UBitsArray({101, 102, 103}, 8).value()},
           {"idx", Value(UBits(2, 37))}}),
      IsOkAndHolds(
          Value::UBits2DArray({{11, 22, 33}, {44, 55, 66}}, 8).value()));
}

TEST_P(CombinationalGeneratorTest, ArrayUpdate2DUpdateArrayNilIndex) {
  Package package(TestBaseName());
  FunctionBuilder fb(TestBaseName(), &package);
  Type* u8 = package.GetBitsType(8);
  auto a = fb.Param("a", package.GetArrayType(2, package.GetArrayType(3, u8)));
  auto update_value =
      fb.Param("value", package.GetArrayType(2, package.GetArrayType(3, u8)));
  auto ret = fb.ArrayUpdate(a, update_value, {});
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.BuildWithReturnValue(ret));
  XLS_ASSERT_OK_AND_ASSIGN(auto result,
                           GenerateCombinationalModule(f, codegen_options()));

  ExpectVerilogEqualToGoldenFile(GoldenFilePath(kTestName, kTestdataPath),
                                 result.verilog_text);

  ModuleSimulator simulator =
      NewModuleSimulator(result.verilog_text, result.signature);
  EXPECT_THAT(
      simulator.RunFunction(
          {{"a", Value::UBits2DArray({{11, 22, 33}, {44, 55, 66}}, 8).value()},
           {"value", Value::UBits2DArray({{101, 102, 103}, {104, 105, 106}}, 8)
                         .value()}}),
      IsOkAndHolds(
          Value::UBits2DArray({{101, 102, 103}, {104, 105, 106}}, 8).value()));
}

TEST_P(CombinationalGeneratorTest, ArrayUpdateBitsNilIndex) {
  Package package(TestBaseName());
  FunctionBuilder fb(TestBaseName(), &package);
  Type* u8 = package.GetBitsType(8);
  auto a = fb.Param("a", u8);
  auto update_value = fb.Param("value", u8);
  auto ret = fb.ArrayUpdate(a, update_value, {});
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.BuildWithReturnValue(ret));
  XLS_ASSERT_OK_AND_ASSIGN(auto result,
                           GenerateCombinationalModule(f, codegen_options()));

  ExpectVerilogEqualToGoldenFile(GoldenFilePath(kTestName, kTestdataPath),
                                 result.verilog_text);

  ModuleSimulator simulator =
      NewModuleSimulator(result.verilog_text, result.signature);
  EXPECT_THAT(simulator.RunAndReturnSingleOutput(
                  {{"a", UBits(11, 8)}, {"value", UBits(22, 8)}}),
              IsOkAndHolds(UBits(22, 8)));
}

TEST_P(CombinationalGeneratorTest, ArrayUpdateWithDifferentTypesIndices) {
  Package package(TestBaseName());
  FunctionBuilder fb(TestBaseName(), &package);
  Type* u32 = package.GetBitsType(32);
  BValue i0 = fb.Param("i0", package.GetBitsType(4));
  BValue i1 = fb.Param("i1", package.GetBitsType(5));
  BValue a =
      fb.Param("a", package.GetArrayType(2, package.GetArrayType(3, u32)));
  BValue value = fb.Param("value", u32);
  fb.ArrayUpdate(a, value, {i0, i1});

  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  XLS_ASSERT_OK_AND_ASSIGN(auto result,
                           GenerateCombinationalModule(f, codegen_options()));

  ExpectVerilogEqualToGoldenFile(GoldenFilePath(kTestName, kTestdataPath),
                                 result.verilog_text);
}

TEST_P(CombinationalGeneratorTest, ArrayUpdateWithNarrowIndex) {
  Package package(TestBaseName());
  FunctionBuilder fb(TestBaseName(), &package);
  Type* u32 = package.GetBitsType(32);
  BValue a = fb.Param("a", package.GetArrayType(10, u32));
  BValue idx = fb.Param("idx", package.GetBitsType(2));
  BValue value = fb.Param("v", u32);
  XLS_ASSERT_OK_AND_ASSIGN(
      Function * f, fb.BuildWithReturnValue(fb.ArrayUpdate(a, value, {idx})));
  XLS_ASSERT_OK_AND_ASSIGN(auto result,
                           GenerateCombinationalModule(f, codegen_options()));

  ExpectVerilogEqualToGoldenFile(GoldenFilePath(kTestName, kTestdataPath),
                                 result.verilog_text);
}

TEST_P(CombinationalGeneratorTest, ArraySliceWithNarrowStart) {
  Package package(TestBaseName());
  FunctionBuilder fb(TestBaseName(), &package);
  Type* u32 = package.GetBitsType(32);
  BValue a = fb.Param("a", package.GetArrayType(5, u32));
  BValue start = fb.Param("start", package.GetBitsType(1));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.BuildWithReturnValue(fb.ArraySlice(
                                             a, start, /*width=*/3)));
  XLS_ASSERT_OK_AND_ASSIGN(auto result,
                           GenerateCombinationalModule(f, codegen_options()));

  ModuleSimulator simulator =
      NewModuleSimulator(result.verilog_text, result.signature);
  XLS_ASSERT_OK_AND_ASSIGN(Value a_value,
                           Value::UBitsArray({1, 2, 3, 4, 5}, 32));
  EXPECT_THAT(
      simulator.RunFunction({{"a", a_value}, {"start", Value(UBits(0, 1))}}),
      IsOkAndHolds(Value::UBitsArray({1, 2, 3}, 32).value()));
  EXPECT_THAT(
      simulator.RunFunction({{"a", a_value}, {"start", Value(UBits(1, 1))}}),
      IsOkAndHolds(Value::UBitsArray({2, 3, 4}, 32).value()));

  ExpectVerilogEqualToGoldenFile(GoldenFilePath(kTestName, kTestdataPath),
                                 result.verilog_text);
}

TEST_P(CombinationalGeneratorTest, ArraySliceWithWideStart) {
  Package package(TestBaseName());
  FunctionBuilder fb(TestBaseName(), &package);
  Type* u32 = package.GetBitsType(32);
  BValue a = fb.Param("a", package.GetArrayType(5, u32));
  BValue start = fb.Param("start", package.GetBitsType(100));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.BuildWithReturnValue(fb.ArraySlice(
                                             a, start, /*width=*/3)));
  XLS_ASSERT_OK_AND_ASSIGN(auto result,
                           GenerateCombinationalModule(f, codegen_options()));

  ModuleSimulator simulator =
      NewModuleSimulator(result.verilog_text, result.signature);
  XLS_ASSERT_OK_AND_ASSIGN(Value a_value,
                           Value::UBitsArray({1, 2, 3, 4, 5}, 32));
  EXPECT_THAT(
      simulator.RunFunction({{"a", a_value}, {"start", Value(UBits(1, 100))}}),
      IsOkAndHolds(Value::UBitsArray({2, 3, 4}, 32).value()));
  EXPECT_THAT(simulator.RunFunction(
                  {{"a", a_value}, {"start", Value(Bits::AllOnes(100))}}),
              IsOkAndHolds(Value::UBitsArray({5, 5, 5}, 32).value()));
}

TEST_P(CombinationalGeneratorTest, ArraySliceWiderThanInputArray) {
  Package package(TestBaseName());
  FunctionBuilder fb(TestBaseName(), &package);
  Type* u32 = package.GetBitsType(32);
  BValue a = fb.Param("a", package.GetArrayType(3, u32));
  BValue start = fb.Param("start", package.GetBitsType(32));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.BuildWithReturnValue(fb.ArraySlice(
                                             a, start, /*width=*/5)));
  XLS_ASSERT_OK_AND_ASSIGN(auto result,
                           GenerateCombinationalModule(f, codegen_options()));

  ModuleSimulator simulator =
      NewModuleSimulator(result.verilog_text, result.signature);
  XLS_ASSERT_OK_AND_ASSIGN(Value a_value, Value::UBitsArray({1, 2, 3}, 32));
  EXPECT_THAT(
      simulator.RunFunction({{"a", a_value}, {"start", Value(UBits(0, 32))}}),
      IsOkAndHolds(Value::UBitsArray({1, 2, 3, 3, 3}, 32).value()));
  EXPECT_THAT(
      simulator.RunFunction({{"a", a_value}, {"start", Value(UBits(1, 32))}}),
      IsOkAndHolds(Value::UBitsArray({2, 3, 3, 3, 3}, 32).value()));
  EXPECT_THAT(
      simulator.RunFunction({{"a", a_value}, {"start", Value(UBits(2, 32))}}),
      IsOkAndHolds(Value::UBitsArray({3, 3, 3, 3, 3}, 32).value()));
  EXPECT_THAT(simulator.RunFunction(
                  {{"a", a_value}, {"start", Value(UBits(123456, 32))}}),
              IsOkAndHolds(Value::UBitsArray({3, 3, 3, 3, 3}, 32).value()));

  ExpectVerilogEqualToGoldenFile(GoldenFilePath(kTestName, kTestdataPath),
                                 result.verilog_text);
}

TEST_P(CombinationalGeneratorTest, TwoDArraySlice) {
  VerilogFile file(codegen_options().use_system_verilog()
                       ? FileType::kSystemVerilog
                       : FileType::kVerilog);
  Package package(TestBaseName());
  FunctionBuilder fb(TestBaseName(), &package);
  Type* u32 = package.GetBitsType(32);
  Type* a2_u32 = package.GetArrayType(2, u32);
  Type* a2x3_u32 = package.GetArrayType(3, a2_u32);
  BValue a = fb.Param("a", a2x3_u32);
  BValue start = fb.Param("start", package.GetBitsType(16));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.BuildWithReturnValue(fb.ArraySlice(
                                             a, start, /*width=*/2)));
  XLS_ASSERT_OK_AND_ASSIGN(auto result,
                           GenerateCombinationalModule(f, codegen_options()));

  ModuleSimulator simulator(result.signature, result.verilog_text,
                            GetFileType(), GetSimulator());
  XLS_ASSERT_OK_AND_ASSIGN(Value a_value,
                           Value::UBits2DArray({{1, 2}, {3, 4}, {5, 6}}, 32));

  EXPECT_THAT(
      simulator.RunFunction({{"a", a_value}, {"start", Value(UBits(0, 16))}}),
      IsOkAndHolds(Value::UBits2DArray({{1, 2}, {3, 4}}, 32).value()));

  EXPECT_THAT(
      simulator.RunFunction({{"a", a_value}, {"start", Value(UBits(1, 16))}}),
      IsOkAndHolds(Value::UBits2DArray({{3, 4}, {5, 6}}, 32).value()));

  EXPECT_THAT(
      simulator.RunFunction({{"a", a_value}, {"start", Value(UBits(2, 16))}}),
      IsOkAndHolds(Value::UBits2DArray({{5, 6}, {5, 6}}, 32).value()));

  EXPECT_THAT(
      simulator.RunFunction({{"a", a_value}, {"start", Value(UBits(10, 16))}}),
      IsOkAndHolds(Value::UBits2DArray({{5, 6}, {5, 6}}, 32).value()));

  ExpectVerilogEqualToGoldenFile(GoldenFilePath(kTestName, kTestdataPath),
                                 result.verilog_text);
}

TEST_P(CombinationalGeneratorTest, UDiv) {
  Package package(TestBaseName());
  FunctionBuilder fb(TestBaseName(), &package);
  BValue x = fb.Param("x", package.GetBitsType(32));
  BValue y = fb.Param("y", package.GetBitsType(32));
  fb.UDiv(x, y);
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  XLS_ASSERT_OK_AND_ASSIGN(auto result,
                           GenerateCombinationalModule(f, codegen_options()));

  ModuleSimulator simulator =
      NewModuleSimulator(result.verilog_text, result.signature);
  EXPECT_THAT(simulator.RunAndReturnSingleOutput(
                  {{"x", UBits(42, 32)}, {"y", UBits(7, 32)}}),
              IsOkAndHolds(UBits(6, 32)));
  // Should round toward zero.
  EXPECT_THAT(simulator.RunAndReturnSingleOutput(
                  {{"x", UBits(10, 32)}, {"y", UBits(7, 32)}}),
              IsOkAndHolds(UBits(1, 32)));
  // Div by zero should return max value.
  EXPECT_THAT(simulator.RunAndReturnSingleOutput(
                  {{"x", UBits(0, 32)}, {"y", UBits(0, 32)}}),
              IsOkAndHolds(Bits::AllOnes(32)));
  EXPECT_THAT(simulator.RunAndReturnSingleOutput(
                  {{"x", UBits(12345, 32)}, {"y", UBits(0, 32)}}),
              IsOkAndHolds(Bits::AllOnes(32)));

  ExpectVerilogEqualToGoldenFile(GoldenFilePath(kTestName, kTestdataPath),
                                 result.verilog_text);
}

TEST_P(CombinationalGeneratorTest, SDiv) {
  Package package(TestBaseName());
  FunctionBuilder fb(TestBaseName(), &package);
  BValue x = fb.Param("x", package.GetBitsType(32));
  BValue y = fb.Param("y", package.GetBitsType(32));
  fb.SDiv(x, y);
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  XLS_ASSERT_OK_AND_ASSIGN(auto result,
                           GenerateCombinationalModule(f, codegen_options()));

  ModuleSimulator simulator =
      NewModuleSimulator(result.verilog_text, result.signature);
  EXPECT_THAT(simulator.RunAndReturnSingleOutput(
                  {{"x", SBits(42, 32)}, {"y", SBits(7, 32)}}),
              IsOkAndHolds(SBits(6, 32)));
  EXPECT_THAT(simulator.RunAndReturnSingleOutput(
                  {{"x", SBits(-42, 32)}, {"y", SBits(7, 32)}}),
              IsOkAndHolds(SBits(-6, 32)));
  EXPECT_THAT(simulator.RunAndReturnSingleOutput(
                  {{"x", SBits(42, 32)}, {"y", SBits(-7, 32)}}),
              IsOkAndHolds(SBits(-6, 32)));
  EXPECT_THAT(simulator.RunAndReturnSingleOutput(
                  {{"x", SBits(-42, 32)}, {"y", SBits(-7, 32)}}),
              IsOkAndHolds(SBits(6, 32)));

  // Should round toward zero.
  EXPECT_THAT(simulator.RunAndReturnSingleOutput(
                  {{"x", SBits(10, 32)}, {"y", SBits(7, 32)}}),
              IsOkAndHolds(SBits(1, 32)));
  EXPECT_THAT(simulator.RunAndReturnSingleOutput(
                  {{"x", SBits(-10, 32)}, {"y", SBits(7, 32)}}),
              IsOkAndHolds(SBits(-1, 32)));

  // Test overflow condition.
  EXPECT_THAT(simulator.RunAndReturnSingleOutput(
                  {{"x", Bits::MinSigned(32)}, {"y", SBits(-1, 32)}}),
              IsOkAndHolds(Bits::MinSigned(32)));

  // Div by zero should return max/min signed value.
  EXPECT_THAT(simulator.RunAndReturnSingleOutput(
                  {{"x", SBits(0, 32)}, {"y", SBits(0, 32)}}),
              IsOkAndHolds(Bits::MaxSigned(32)));
  EXPECT_THAT(simulator.RunAndReturnSingleOutput(
                  {{"x", SBits(12345, 32)}, {"y", SBits(0, 32)}}),
              IsOkAndHolds(Bits::MaxSigned(32)));
  EXPECT_THAT(simulator.RunAndReturnSingleOutput(
                  {{"x", SBits(-12345, 32)}, {"y", SBits(0, 32)}}),
              IsOkAndHolds(Bits::MinSigned(32)));

  ExpectVerilogEqualToGoldenFile(GoldenFilePath(kTestName, kTestdataPath),
                                 result.verilog_text);
}

TEST_P(CombinationalGeneratorTest, OneBitTupleIndex) {
  Package package(TestBaseName());
  FunctionBuilder fb(TestBaseName(), &package);
  BValue x = fb.Param(
      "x",
      package.GetArrayType(1, package.GetTupleType({package.GetBitsType(1)})));
  fb.TupleIndex(fb.ArrayIndex(x, {fb.Literal(UBits(0, 32))}), 0);
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  XLS_ASSERT_OK_AND_ASSIGN(auto result,
                           GenerateCombinationalModule(f, codegen_options()));

  ModuleSimulator simulator =
      NewModuleSimulator(result.verilog_text, result.signature);
  XLS_ASSERT_OK_AND_ASSIGN(Value x_value,
                           Value::Array({Value::Tuple({Value(UBits(1, 1))})}));
  EXPECT_THAT(simulator.RunFunction({{"x", x_value}}),
              IsOkAndHolds(Value(UBits(1, 1))));

  ExpectVerilogEqualToGoldenFile(GoldenFilePath(kTestName, kTestdataPath),
                                 result.verilog_text);
}

TEST_P(CombinationalGeneratorTest, ArrayEq) {
  Package package(TestBaseName());
  FunctionBuilder fb(TestBaseName(), &package);
  Type* array_type = package.GetArrayType(5, package.GetBitsType(32));
  BValue x = fb.Param("x", array_type);
  BValue y = fb.Param("y", array_type);
  fb.Eq(x, y);
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  XLS_ASSERT_OK_AND_ASSIGN(auto result,
                           GenerateCombinationalModule(f, codegen_options()));

  ModuleSimulator simulator =
      NewModuleSimulator(result.verilog_text, result.signature);
  Value a = Value::UBitsArray({1, 2, 3, 4, 5}, 32).value();
  Value b = Value::UBitsArray({1, 20, 3, 4, 5}, 32).value();
  EXPECT_THAT(simulator.RunFunction({{"x", a}, {"y", b}}),
              IsOkAndHolds(Value(UBits(0, 1))));
  EXPECT_THAT(simulator.RunFunction({{"x", a}, {"y", a}}),
              IsOkAndHolds(Value(UBits(1, 1))));

  ExpectVerilogEqualToGoldenFile(GoldenFilePath(kTestName, kTestdataPath),
                                 result.verilog_text);
}

TEST_P(CombinationalGeneratorTest, TwoDArrayNe) {
  Package package(TestBaseName());
  FunctionBuilder fb(TestBaseName(), &package);
  Type* array_type =
      package.GetArrayType(2, package.GetArrayType(3, package.GetBitsType(1)));
  BValue x = fb.Param("x", array_type);
  BValue y = fb.Param("y", array_type);
  fb.Ne(x, y);
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  XLS_ASSERT_OK_AND_ASSIGN(auto result,
                           GenerateCombinationalModule(f, codegen_options()));

  ModuleSimulator simulator =
      NewModuleSimulator(result.verilog_text, result.signature);
  Value a = Value::UBits2DArray({{1, 0, 1}, {1, 1, 0}}, 1).value();
  Value b = Value::UBits2DArray({{1, 1, 1}, {1, 1, 0}}, 1).value();
  EXPECT_THAT(simulator.RunFunction({{"x", a}, {"y", b}}),
              IsOkAndHolds(Value(UBits(1, 1))));
  EXPECT_THAT(simulator.RunFunction({{"x", a}, {"y", a}}),
              IsOkAndHolds(Value(UBits(0, 1))));

  ExpectVerilogEqualToGoldenFile(GoldenFilePath(kTestName, kTestdataPath),
                                 result.verilog_text);
}

TEST_P(CombinationalGeneratorTest, SingleProcWithProcScopedChannels) {
  Package package(TestBaseName());

  TokenlessProcBuilder pb(NewStyleProc(), "myleaf", "tkn", &package);
  XLS_ASSERT_OK_AND_ASSIGN(ReceiveChannelInterface * in,
                           pb.AddInputChannel("in", package.GetBitsType(32)));
  XLS_ASSERT_OK_AND_ASSIGN(SendChannelInterface * out,
                           pb.AddOutputChannel("out", package.GetBitsType(32)));

  pb.Send(out, pb.Add(pb.Receive(in), pb.Literal(UBits(1, 32))));
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, pb.Build());
  XLS_ASSERT_OK(package.SetTop(proc));

  XLS_ASSERT_OK_AND_ASSIGN(
      auto result, GenerateCombinationalModule(proc, codegen_options()));

  ExpectVerilogEqualToGoldenFile(GoldenFilePath(kTestName, kTestdataPath),
                                 result.verilog_text);
}

INSTANTIATE_TEST_SUITE_P(CombinationalGeneratorTestInstantiation,
                         CombinationalGeneratorTest,
                         testing::ValuesIn(kDefaultSimulationTargets),
                         ParameterizedTestName<CombinationalGeneratorTest>);

}  // namespace
}  // namespace verilog
}  // namespace xls
