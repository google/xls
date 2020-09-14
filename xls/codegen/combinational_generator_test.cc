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

#include "xls/codegen/combinational_generator.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "xls/common/status/matchers.h"
#include "xls/examples/sample_packages.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/ir_interpreter.h"
#include "xls/ir/ir_parser.h"
#include "xls/ir/package.h"
#include "xls/ir/value_helpers.h"
#include "xls/simulation/module_simulator.h"
#include "xls/simulation/verilog_simulators.h"
#include "xls/simulation/verilog_test_base.h"

namespace xls {
namespace verilog {
namespace {

using status_testing::IsOkAndHolds;

constexpr char kTestName[] = "combinational_generator_test";
constexpr char kTestdataPath[] = "xls/codegen/testdata";

class CombinationalGeneratorTest : public VerilogTestBase {};

TEST_P(CombinationalGeneratorTest, RrotToCombinationalText) {
  auto rrot32_data = sample_packages::BuildRrot32();
  Function* f = rrot32_data.second;
  XLS_ASSERT_OK_AND_ASSIGN(auto result,
                           ToCombinationalModuleText(f, UseSystemVerilog()));

  ExpectVerilogEqualToGoldenFile(GoldenFilePath(kTestName, kTestdataPath),
                                 result.verilog_text);

  ModuleSimulator simulator(result.signature, result.verilog_text,
                            GetSimulator());
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
  auto a_minus_b = a - b;
  auto lhs = (a_minus_b * a_minus_b);
  auto rhs = (c * a_minus_b);
  auto out = lhs + rhs;
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.BuildWithReturnValue(out));
  XLS_ASSERT_OK_AND_ASSIGN(auto result,
                           ToCombinationalModuleText(f, UseSystemVerilog()));

  ExpectVerilogEqualToGoldenFile(GoldenFilePath(kTestName, kTestdataPath),
                                 result.verilog_text);

  ModuleSimulator simulator(result.signature, result.verilog_text,
                            GetSimulator());
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
                           ToCombinationalModuleText(f, UseSystemVerilog()));
  ModuleSimulator simulator(result.signature, result.verilog_text,
                            GetSimulator());
  EXPECT_THAT(simulator.RunAndReturnSingleOutput(ModuleSimulator::BitsMap()),
              IsOkAndHolds(UBits(123, 8)));
}

TEST_P(CombinationalGeneratorTest, ReturnsTupleLiteral) {
  Package package(TestBaseName());
  FunctionBuilder fb(TestBaseName(), &package);
  fb.Literal(Value::Tuple({Value(UBits(123, 8)), Value(UBits(42, 32))}));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  XLS_ASSERT_OK_AND_ASSIGN(auto result,
                           ToCombinationalModuleText(f, UseSystemVerilog()));
  ModuleSimulator simulator(result.signature, result.verilog_text,
                            GetSimulator());
  EXPECT_THAT(
      simulator.Run(absl::flat_hash_map<std::string, Value>()),
      IsOkAndHolds(Value::Tuple({Value(UBits(123, 8)), Value(UBits(42, 32))})));
}

TEST_P(CombinationalGeneratorTest, ReturnsEmptyTuple) {
  Package package(TestBaseName());
  FunctionBuilder fb(TestBaseName(), &package);
  fb.Literal(Value::Tuple({}));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  XLS_ASSERT_OK_AND_ASSIGN(auto result,
                           ToCombinationalModuleText(f, UseSystemVerilog()));
  ModuleSimulator simulator(result.signature, result.verilog_text,
                            GetSimulator());
  EXPECT_THAT(simulator.Run(absl::flat_hash_map<std::string, Value>()),
              IsOkAndHolds(Value::Tuple({})));
}

TEST_P(CombinationalGeneratorTest, PassesEmptyTuple) {
  Package package(TestBaseName());
  FunctionBuilder fb(TestBaseName(), &package);
  fb.Param("x", package.GetTupleType({}));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  XLS_ASSERT_OK_AND_ASSIGN(auto result,
                           ToCombinationalModuleText(f, UseSystemVerilog()));
  ModuleSimulator simulator(result.signature, result.verilog_text,
                            GetSimulator());
  EXPECT_THAT(simulator.Run({{"x", Value::Tuple({})}}),
              IsOkAndHolds(Value::Tuple({})));
}

TEST_P(CombinationalGeneratorTest, TakesEmptyTuple) {
  Package package(TestBaseName());
  FunctionBuilder fb(TestBaseName(), &package);
  Type* u8 = package.GetBitsType(8);
  auto a = fb.Param("a", u8);
  fb.Param("b", package.GetTupleType({}));
  auto c = fb.Param("c", u8);
  fb.Add(a, c);
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  XLS_ASSERT_OK_AND_ASSIGN(auto result,
                           ToCombinationalModuleText(f, UseSystemVerilog()));
  ModuleSimulator simulator(result.signature, result.verilog_text,
                            GetSimulator());
  EXPECT_THAT(simulator.Run({{"a", Value(UBits(42, 8))},
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
                           ToCombinationalModuleText(f, UseSystemVerilog()));
  ModuleSimulator simulator(result.signature, result.verilog_text,
                            GetSimulator());
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
  auto out = fb.BitSlice(a_plus_b, /*start=*/3, /*width=*/4);
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.BuildWithReturnValue(out));
  XLS_ASSERT_OK_AND_ASSIGN(auto result,
                           ToCombinationalModuleText(f, UseSystemVerilog()));

  ExpectVerilogEqualToGoldenFile(GoldenFilePath(kTestName, kTestdataPath),
                                 result.verilog_text);

  ModuleSimulator simulator(result.signature, result.verilog_text,
                            GetSimulator());
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
  auto a_b_c = fb.Tuple({a, b, c});

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
                           ToCombinationalModuleText(f, UseSystemVerilog()));

  ExpectVerilogEqualToGoldenFile(GoldenFilePath(kTestName, kTestdataPath),
                                 result.verilog_text);

  ModuleSimulator simulator(result.signature, result.verilog_text,
                            GetSimulator());
  EXPECT_THAT(simulator.Run({{"a", Value(UBits(42, 8))},
                             {"b", Value(UBits(123, 10))},
                             {"c", Value::Tuple({Value(UBits(333, 10)),
                                                 Value(UBits(222, 16))})}}),
              IsOkAndHolds(Value::Tuple(
                  {Value(UBits(165, 16)), Value(UBits(111, 16))})));
}

TEST_P(CombinationalGeneratorTest, TupleLiterals) {
  std::string text = R"(
package TupleLiterals

fn main(x: bits[123]) -> bits[123] {
  literal.1: (bits[123], bits[123], bits[123]) = literal(value=(0x10000, 0x2000, 0x300))
  tuple_index.2: bits[123] = tuple_index(literal.1, index=0)
  tuple_index.3: bits[123] = tuple_index(literal.1, index=1)
  tuple_index.4: bits[123] = tuple_index(literal.1, index=2)
  add.6: bits[123] = add(tuple_index.2, tuple_index.3)
  add.7: bits[123] = add(tuple_index.4, x)
  ret add.8: bits[123] = add(add.6, add.7)
}
)";
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> package,
                           Parser::ParsePackage(text));

  XLS_ASSERT_OK_AND_ASSIGN(Function * entry, package->EntryFunction());
  XLS_ASSERT_OK_AND_ASSIGN(
      auto result, ToCombinationalModuleText(entry, UseSystemVerilog()));

  ExpectVerilogEqualToGoldenFile(GoldenFilePath(kTestName, kTestdataPath),
                                 result.verilog_text);

  ModuleSimulator simulator(result.signature, result.verilog_text,
                            GetSimulator());
  EXPECT_THAT(simulator.Run({{"x", Value(UBits(0x40, 123))}}),
              IsOkAndHolds(Value(UBits(0x12340, 123))));
}

TEST_P(CombinationalGeneratorTest, ArrayLiteral) {
  std::string text = R"(
package ArrayLiterals

fn main(x: bits[32], y: bits[32]) -> bits[44] {
  literal.1: bits[44][3][2] = literal(value=[[1, 2, 3], [4, 5, 6]])
  array_index.2: bits[44][3] = array_index(literal.1, x)
  ret array_index.3: bits[44] = array_index(array_index.2, y)
}
)";
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> package,
                           Parser::ParsePackage(text));

  XLS_ASSERT_OK_AND_ASSIGN(Function * entry, package->EntryFunction());
  XLS_ASSERT_OK_AND_ASSIGN(
      auto result, ToCombinationalModuleText(entry, UseSystemVerilog()));

  ExpectVerilogEqualToGoldenFile(GoldenFilePath(kTestName, kTestdataPath),
                                 result.verilog_text);

  ModuleSimulator simulator(result.signature, result.verilog_text,
                            GetSimulator());
  EXPECT_THAT(
      simulator.Run({{"x", Value(UBits(0, 32))}, {"y", Value(UBits(1, 32))}}),
      IsOkAndHolds(Value(UBits(2, 44))));
  EXPECT_THAT(
      simulator.Run({{"x", Value(UBits(1, 32))}, {"y", Value(UBits(0, 32))}}),
      IsOkAndHolds(Value(UBits(4, 44))));
}

TEST_P(CombinationalGeneratorTest, OneHot) {
  std::string text = R"(
package OneHot

fn main(x: bits[3]) -> bits[4] {
  ret one_hot.1: bits[4] = one_hot(x, lsb_prio=true)
}
)";
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> package,
                           Parser::ParsePackage(text));

  XLS_ASSERT_OK_AND_ASSIGN(Function * entry, package->EntryFunction());
  XLS_ASSERT_OK_AND_ASSIGN(
      auto result, ToCombinationalModuleText(entry, UseSystemVerilog()));

  ExpectVerilogEqualToGoldenFile(GoldenFilePath(kTestName, kTestdataPath),
                                 result.verilog_text);

  ModuleSimulator simulator(result.signature, result.verilog_text,
                            GetSimulator());
  EXPECT_THAT(simulator.Run({{"x", Value(UBits(0b000, 3))}}),
              IsOkAndHolds(Value(UBits(0b1000, 4))));
  EXPECT_THAT(simulator.Run({{"x", Value(UBits(0b001, 3))}}),
              IsOkAndHolds(Value(UBits(0b0001, 4))));
  EXPECT_THAT(simulator.Run({{"x", Value(UBits(0b010, 3))}}),
              IsOkAndHolds(Value(UBits(0b0010, 4))));
  EXPECT_THAT(simulator.Run({{"x", Value(UBits(0b011, 3))}}),
              IsOkAndHolds(Value(UBits(0b0001, 4))));
  EXPECT_THAT(simulator.Run({{"x", Value(UBits(0b100, 3))}}),
              IsOkAndHolds(Value(UBits(0b0100, 4))));
  EXPECT_THAT(simulator.Run({{"x", Value(UBits(0b101, 3))}}),
              IsOkAndHolds(Value(UBits(0b0001, 4))));
  EXPECT_THAT(simulator.Run({{"x", Value(UBits(0b110, 3))}}),
              IsOkAndHolds(Value(UBits(0b0010, 4))));
  EXPECT_THAT(simulator.Run({{"x", Value(UBits(0b111, 3))}}),
              IsOkAndHolds(Value(UBits(0b0001, 4))));
}

TEST_P(CombinationalGeneratorTest, OneHotSelect) {
  std::string text = R"(
package OneHotSelect

fn main(p: bits[2], x: bits[16], y: bits[16]) -> bits[16] {
  ret one_hot_sel.1: bits[16] = one_hot_sel(p, cases=[x, y])
}
)";
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> package,
                           Parser::ParsePackage(text));

  XLS_ASSERT_OK_AND_ASSIGN(Function * entry, package->EntryFunction());
  XLS_ASSERT_OK_AND_ASSIGN(
      auto result, ToCombinationalModuleText(entry, UseSystemVerilog()));

  ExpectVerilogEqualToGoldenFile(GoldenFilePath(kTestName, kTestdataPath),
                                 result.verilog_text);

  ModuleSimulator simulator(result.signature, result.verilog_text,
                            GetSimulator());
  absl::flat_hash_map<std::string, Value> args = {
      {"x", Value(UBits(0x00ff, 16))}, {"y", Value(UBits(0xf0f0, 16))}};
  args["p"] = Value(UBits(0b00, 2));
  EXPECT_THAT(simulator.Run(args), IsOkAndHolds(Value(UBits(0x0000, 16))));
  args["p"] = Value(UBits(0b01, 2));
  EXPECT_THAT(simulator.Run(args), IsOkAndHolds(Value(UBits(0x00ff, 16))));
  args["p"] = Value(UBits(0b10, 2));
  EXPECT_THAT(simulator.Run(args), IsOkAndHolds(Value(UBits(0xf0f0, 16))));
  args["p"] = Value(UBits(0b11, 2));
  EXPECT_THAT(simulator.Run(args), IsOkAndHolds(Value(UBits(0xf0ff, 16))));
}

TEST_P(CombinationalGeneratorTest, CrazyParameterTypes) {
  std::string text = R"(
package CrazyParameterTypes

fn main(a: bits[32],
        b: (bits[32], ()),
        c: bits[32][3],
        d: (bits[32], bits[32])[1],
        e: (bits[32][2], (), ()),
        f: bits[0],
        g: bits[1]) -> bits[32] {
  tuple_index.1: bits[32] = tuple_index(b, index=0)
  literal.2: bits[32] = literal(value=0)
  array_index.3: bits[32] = array_index(c, g)
  array_index.4: (bits[32], bits[32]) = array_index(d, literal.2)
  tuple_index.5: bits[32] = tuple_index(array_index.4, index=1)
  tuple_index.6: bits[32][2] = tuple_index(e, index=0)
  array_index.7: bits[32] = array_index(tuple_index.6, g)
  ret or.8: bits[32] = or(a, tuple_index.1, array_index.3, tuple_index.5, array_index.7)
}
)";
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> package,
                           Parser::ParsePackage(text));

  XLS_ASSERT_OK_AND_ASSIGN(Function * entry, package->EntryFunction());
  XLS_ASSERT_OK_AND_ASSIGN(
      auto result, ToCombinationalModuleText(entry, UseSystemVerilog()));

  ExpectVerilogEqualToGoldenFile(GoldenFilePath(kTestName, kTestdataPath),
                                 result.verilog_text);

  ModuleSimulator simulator(result.signature, result.verilog_text,
                            GetSimulator());

  std::minstd_rand engine;
  std::vector<Value> arguments = RandomFunctionArguments(entry, &engine);
  XLS_ASSERT_OK_AND_ASSIGN(Value expected,
                           IrInterpreter::Run(entry, arguments));
  EXPECT_THAT(simulator.Run(arguments), IsOkAndHolds(expected));
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
  fb.Add(fb.ArrayIndex(fb.ArrayIndex(two_d, fb.Literal(UBits(0, 8))),
                       fb.Literal(UBits(2, 8))),
         fb.ArrayIndex(fb.ArrayIndex(two_d, fb.Literal(UBits(1, 8))),
                       fb.Literal(UBits(1, 8))));

  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  XLS_ASSERT_OK_AND_ASSIGN(auto result,
                           ToCombinationalModuleText(f, UseSystemVerilog()));
  ModuleSimulator simulator(result.signature, result.verilog_text,
                            GetSimulator());
  EXPECT_THAT(simulator.Run({{"a", Value(UBits(123, 8))},
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
                           ToCombinationalModuleText(f, UseSystemVerilog()));
  ModuleSimulator simulator(result.signature, result.verilog_text,
                            GetSimulator());
  EXPECT_THAT(
      simulator.Run({{"a", Value(UBits(123, 8))}, {"b", Value(UBits(42, 8))}}),
      IsOkAndHolds(Value::ArrayOrDie({
          Value::ArrayOrDie({Value(UBits(123, 8)), Value(UBits(42, 8))}),
          Value::ArrayOrDie({Value(UBits(42, 8)), Value(UBits(123, 8))}),
      })));
}

TEST_P(CombinationalGeneratorTest, ArrayUpdateBitElements) {
  std::string text = R"(
package ArrayUpdate

fn main(idx: bits[2]) -> bits[32][3] {
  literal.5: bits[32][3] = literal(value=[1, 2, 3])
  literal.6: bits[32] = literal(value=99)
  ret array_update.7: bits[32][3] = array_update(literal.5, idx, literal.6)
}
)";
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> package,
                           Parser::ParsePackage(text));

  XLS_ASSERT_OK_AND_ASSIGN(Function * entry, package->EntryFunction());
  XLS_ASSERT_OK_AND_ASSIGN(
      auto result, ToCombinationalModuleText(entry, UseSystemVerilog()));

  ModuleSimulator simulator(result.signature, result.verilog_text,
                            GetSimulator());

  auto make_array = [](absl::Span<const int64> values) {
    std::vector<Value> elements;
    for (auto v : values) {
      elements.push_back(Value(UBits(v, 32)));
    }
    xabsl::StatusOr<Value> array = Value::Array(elements);
    EXPECT_TRUE(array.ok());
    return array.value();
  };

  EXPECT_THAT(simulator.Run({{"idx", Value(UBits(0b00, 2))}}),
              IsOkAndHolds(make_array({99, 2, 3})));
  EXPECT_THAT(simulator.Run({{"idx", Value(UBits(0b01, 2))}}),
              IsOkAndHolds(make_array({1, 99, 3})));
  EXPECT_THAT(simulator.Run({{"idx", Value(UBits(0b10, 2))}}),
              IsOkAndHolds(make_array({1, 2, 99})));
  EXPECT_THAT(simulator.Run({{"idx", Value(UBits(0b11, 2))}}),
              IsOkAndHolds(make_array({1, 2, 3})));
}

TEST_P(CombinationalGeneratorTest, ArrayUpdateArrayElements) {
  std::string text = R"(
package ArrayUpdate

fn main(idx: bits[2]) -> bits[32][2][3] {
  literal.17: bits[32][2][3] = literal(value=[[1, 2], [3, 4], [5, 6]])
  literal.14: bits[32][2] = literal(value=[98, 99])
  ret array_update.15: bits[32][2][3] = array_update(literal.17, idx, literal.14)
}

)";
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> package,
                           Parser::ParsePackage(text));

  XLS_ASSERT_OK_AND_ASSIGN(Function * entry, package->EntryFunction());
  XLS_ASSERT_OK_AND_ASSIGN(
      auto result, ToCombinationalModuleText(entry, UseSystemVerilog()));

  ModuleSimulator simulator(result.signature, result.verilog_text,
                            GetSimulator());

  auto make_array = [](absl::Span<const int64> values) {
    std::vector<Value> elements;
    for (auto v : values) {
      elements.push_back(Value(UBits(v, 32)));
    }
    xabsl::StatusOr<Value> array = Value::Array(elements);
    EXPECT_TRUE(array.ok());
    return array.value();
  };

  auto make_array_of_values = [&](absl::Span<const Value> values) {
    std::vector<Value> elements;
    for (auto array : values) {
      elements.push_back(array);
    }
    xabsl::StatusOr<Value> array_of_values = Value::Array(elements);
    EXPECT_TRUE(array_of_values.ok());
    return array_of_values.value();
  };

  EXPECT_THAT(
      simulator.Run({{"idx", Value(UBits(0b00, 2))}}),
      IsOkAndHolds(make_array_of_values(
          {make_array({98, 99}), make_array({3, 4}), make_array({5, 6})})));
  EXPECT_THAT(
      simulator.Run({{"idx", Value(UBits(0b01, 2))}}),
      IsOkAndHolds(make_array_of_values(
          {make_array({1, 2}), make_array({98, 99}), make_array({5, 6})})));
  EXPECT_THAT(
      simulator.Run({{"idx", Value(UBits(0b10, 2))}}),
      IsOkAndHolds(make_array_of_values(
          {make_array({1, 2}), make_array({3, 4}), make_array({98, 99})})));
  EXPECT_THAT(
      simulator.Run({{"idx", Value(UBits(0b11, 2))}}),
      IsOkAndHolds(make_array_of_values(
          {make_array({1, 2}), make_array({3, 4}), make_array({5, 6})})));
}

TEST_P(CombinationalGeneratorTest, ArrayUpdateTupleElements) {
  std::string text = R"(
package ArrayUpdate

fn main(idx: bits[2]) -> (bits[32], bits[32])[3] {
  literal.17: (bits[32], bits[32])[3] = literal(value=[(1,2),(3,4),(5,6)])
  literal.14: (bits[32], bits[32]) = literal(value=(98, 99))
  ret array_update.15: (bits[32], bits[32])[3] = array_update(literal.17, idx, literal.14)
}

)";
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> package,
                           Parser::ParsePackage(text));

  XLS_ASSERT_OK_AND_ASSIGN(Function * entry, package->EntryFunction());
  XLS_ASSERT_OK_AND_ASSIGN(
      auto result, ToCombinationalModuleText(entry, UseSystemVerilog()));

  ModuleSimulator simulator(result.signature, result.verilog_text,
                            GetSimulator());

  auto make_tuple = [](absl::Span<const int64> values) {
    std::vector<Value> elements;
    for (auto v : values) {
      elements.push_back(Value(UBits(v, 32)));
    }
    xabsl::StatusOr<Value> tuple = Value::Tuple(elements);
    EXPECT_TRUE(tuple.ok());
    return tuple.value();
  };

  auto make_array_of_values = [&](absl::Span<const Value> values) {
    std::vector<Value> elements;
    for (auto array : values) {
      elements.push_back(array);
    }
    xabsl::StatusOr<Value> array_of_values = Value::Array(elements);
    EXPECT_TRUE(array_of_values.ok());
    return array_of_values.value();
  };

  EXPECT_THAT(
      simulator.Run({{"idx", Value(UBits(0b00, 2))}}),
      IsOkAndHolds(make_array_of_values(
          {make_tuple({98, 99}), make_tuple({3, 4}), make_tuple({5, 6})})));
  EXPECT_THAT(
      simulator.Run({{"idx", Value(UBits(0b01, 2))}}),
      IsOkAndHolds(make_array_of_values(
          {make_tuple({1, 2}), make_tuple({98, 99}), make_tuple({5, 6})})));
  EXPECT_THAT(
      simulator.Run({{"idx", Value(UBits(0b10, 2))}}),
      IsOkAndHolds(make_array_of_values(
          {make_tuple({1, 2}), make_tuple({3, 4}), make_tuple({98, 99})})));
  EXPECT_THAT(
      simulator.Run({{"idx", Value(UBits(0b11, 2))}}),
      IsOkAndHolds(make_array_of_values(
          {make_tuple({1, 2}), make_tuple({3, 4}), make_tuple({5, 6})})));
}

TEST_P(CombinationalGeneratorTest, ArrayUpdateTupleWithArrayElements) {
  std::string text = R"(
package ArrayUpdate

fn main(idx: bits[2]) -> (bits[32], bits[8][2])[2] {
  literal.17: (bits[32], bits[8][2])[2] = literal(value=[(1,[2,3]),(4,[5,6])])
  literal.14: (bits[32], bits[8][2]) = literal(value=(98, [99, 100]))
  ret array_update.15: (bits[32], bits[8][2])[2] = array_update(literal.17, idx, literal.14)
}

)";
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> package,
                           Parser::ParsePackage(text));

  XLS_ASSERT_OK_AND_ASSIGN(Function * entry, package->EntryFunction());
  XLS_ASSERT_OK_AND_ASSIGN(
      auto result, ToCombinationalModuleText(entry, UseSystemVerilog()));

  ModuleSimulator simulator(result.signature, result.verilog_text,
                            GetSimulator());

  auto make_array = [](absl::Span<const int64> values) {
    std::vector<Value> elements;
    for (auto v : values) {
      elements.push_back(Value(UBits(v, 8)));
    }
    xabsl::StatusOr<Value> array = Value::Array(elements);
    EXPECT_TRUE(array.ok());
    return array.value();
  };

  auto make_tuple = [](absl::Span<const Value> values) {
    std::vector<Value> elements;
    for (auto v : values) {
      elements.push_back(v);
    }
    xabsl::StatusOr<Value> tuple = Value::Tuple(elements);
    EXPECT_TRUE(tuple.ok());
    return tuple.value();
  };

  auto make_array_of_values = [&](absl::Span<const Value> values) {
    std::vector<Value> elements;
    for (auto array : values) {
      elements.push_back(array);
    }
    xabsl::StatusOr<Value> array_of_values = Value::Array(elements);
    EXPECT_TRUE(array_of_values.ok());
    return array_of_values.value();
  };

  EXPECT_THAT(
      simulator.Run({{"idx", Value(UBits(0b01, 2))}}),
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
  fb.ArrayIndex(fb.ArrayIndex(fb.TupleIndex(tuple, 0), a), c);

  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  XLS_ASSERT_OK_AND_ASSIGN(auto result,
                           ToCombinationalModuleText(f, UseSystemVerilog()));
  ModuleSimulator simulator(result.signature, result.verilog_text,
                            GetSimulator());
  EXPECT_THAT(simulator.Run({{"a", Value(UBits(0, 8))},
                             {"b", Value(UBits(42, 8))},
                             {"c", Value(UBits(1, 8))}}),
              IsOkAndHolds(Value(UBits(42, 8))));
}

TEST_P(CombinationalGeneratorTest, ArrayShapedSel) {
  VerilogFile file;
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
                           ToCombinationalModuleText(f, UseSystemVerilog()));
  ModuleSimulator simulator(result.signature, result.verilog_text,
                            GetSimulator());
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
  EXPECT_THAT(simulator.Run({{"p", Value(UBits(0, 8))},
                             {"x", x_in},
                             {"y", y_in},
                             {"z", z_in},
                             {"d", d_in}}),
              IsOkAndHolds(x_in));
  EXPECT_THAT(simulator.Run({{"p", Value(UBits(1, 8))},
                             {"x", x_in},
                             {"y", y_in},
                             {"z", z_in},
                             {"d", d_in}}),
              IsOkAndHolds(y_in));
  EXPECT_THAT(simulator.Run({{"p", Value(UBits(2, 8))},
                             {"x", x_in},
                             {"y", y_in},
                             {"z", z_in},
                             {"d", d_in}}),
              IsOkAndHolds(z_in));
  EXPECT_THAT(simulator.Run({{"p", Value(UBits(3, 8))},
                             {"x", x_in},
                             {"y", y_in},
                             {"z", z_in},
                             {"d", d_in}}),
              IsOkAndHolds(d_in));
  EXPECT_THAT(simulator.Run({{"p", Value(UBits(100, 8))},
                             {"x", x_in},
                             {"y", y_in},
                             {"z", z_in},
                             {"d", d_in}}),
              IsOkAndHolds(d_in));
}

TEST_P(CombinationalGeneratorTest, ArrayShapedSelNoDefault) {
  VerilogFile file;
  Package package(TestBaseName());
  FunctionBuilder fb(TestBaseName(), &package);
  BValue p = fb.Param("p", package.GetBitsType(1));
  BValue x = fb.Param("x", package.GetArrayType(3, package.GetBitsType(8)));
  BValue y = fb.Param("y", package.GetArrayType(3, package.GetBitsType(8)));
  fb.Select(p, {x, y});

  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  XLS_ASSERT_OK_AND_ASSIGN(auto result,
                           ToCombinationalModuleText(f, UseSystemVerilog()));
  ModuleSimulator simulator(result.signature, result.verilog_text,
                            GetSimulator());
  XLS_ASSERT_OK_AND_ASSIGN(
      Value x_in,
      Parser::ParseTypedValue("[bits[8]:0xa, bits[8]:0xb, bits[8]:0xc]"));
  XLS_ASSERT_OK_AND_ASSIGN(
      Value y_in,
      Parser::ParseTypedValue("[bits[8]:0x1, bits[8]:0x2, bits[8]:0x3]"));
  EXPECT_THAT(
      simulator.Run({{"p", Value(UBits(0, 1))}, {"x", x_in}, {"y", y_in}}),
      IsOkAndHolds(x_in));
  EXPECT_THAT(
      simulator.Run({{"p", Value(UBits(1, 1))}, {"x", x_in}, {"y", y_in}}),
      IsOkAndHolds(y_in));
}

INSTANTIATE_TEST_SUITE_P(CombinationalGeneratorTestInstantiation,
                         CombinationalGeneratorTest,
                         testing::ValuesIn(kDefaultSimulationTargets),
                         ParameterizedTestName<CombinationalGeneratorTest>);

}  // namespace
}  // namespace verilog
}  // namespace xls
