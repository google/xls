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

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "xls/common/status/matchers.h"
#include "xls/examples/sample_packages.h"
#include "xls/interpreter/ir_interpreter.h"
#include "xls/ir/function_builder.h"
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
                           GenerateCombinationalModule(f, UseSystemVerilog()));

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
  auto a_minus_b = fb.Subtract(a, b, /*loc=*/absl::nullopt, /*name=*/"diff");
  auto lhs = (a_minus_b * a_minus_b);
  auto rhs = (c * a_minus_b);
  auto out = fb.Add(lhs, rhs, /*loc=*/absl::nullopt, /*name=*/"the_output");
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.BuildWithReturnValue(out));
  XLS_ASSERT_OK_AND_ASSIGN(auto result,
                           GenerateCombinationalModule(f, UseSystemVerilog()));

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
                           GenerateCombinationalModule(f, UseSystemVerilog()));
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
                           GenerateCombinationalModule(f, UseSystemVerilog()));
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
                           GenerateCombinationalModule(f, UseSystemVerilog()));
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
                           GenerateCombinationalModule(f, UseSystemVerilog()));
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
  fb.Add(a, c, /*loc=*/absl::nullopt, /*name=*/"sum");
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  XLS_ASSERT_OK_AND_ASSIGN(auto result,
                           GenerateCombinationalModule(f, UseSystemVerilog()));
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
                           GenerateCombinationalModule(f, UseSystemVerilog()));
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
  auto out = fb.BitSlice(a_plus_b, /*start=*/3, /*width=*/4,
                         /*loc=*/absl::nullopt, /*name=*/"slice_n_dice");
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.BuildWithReturnValue(out));
  XLS_ASSERT_OK_AND_ASSIGN(auto result,
                           GenerateCombinationalModule(f, UseSystemVerilog()));

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
  auto a_b_c = fb.Tuple({a, b, c}, /*loc=*/absl::nullopt, /*name=*/"big_tuple");

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
                           GenerateCombinationalModule(f, UseSystemVerilog()));

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
  sum1: bits[123] = add(tuple_index.2, tuple_index.3)
  sum2: bits[123] = add(tuple_index.4, x)
  ret total: bits[123] = add(sum1, sum2)
}
)";
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> package,
                           Parser::ParsePackage(text));

  XLS_ASSERT_OK_AND_ASSIGN(Function * entry, package->EntryFunction());
  XLS_ASSERT_OK_AND_ASSIGN(
      auto result, GenerateCombinationalModule(entry, UseSystemVerilog()));

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
  ret result: bits[44] = array_index(array_index.2, y)
}
)";
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> package,
                           Parser::ParsePackage(text));

  XLS_ASSERT_OK_AND_ASSIGN(Function * entry, package->EntryFunction());
  XLS_ASSERT_OK_AND_ASSIGN(
      auto result, GenerateCombinationalModule(entry, UseSystemVerilog()));

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
      auto result, GenerateCombinationalModule(entry, UseSystemVerilog()));

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
      auto result, GenerateCombinationalModule(entry, UseSystemVerilog()));

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
      auto result, GenerateCombinationalModule(entry, UseSystemVerilog()));

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
                           GenerateCombinationalModule(f, UseSystemVerilog()));
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
                           GenerateCombinationalModule(f, UseSystemVerilog()));
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
  ret updated_array: bits[32][3] = array_update(literal.5, idx, literal.6)
}
)";
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> package,
                           Parser::ParsePackage(text));

  XLS_ASSERT_OK_AND_ASSIGN(Function * entry, package->EntryFunction());
  XLS_ASSERT_OK_AND_ASSIGN(
      auto result, GenerateCombinationalModule(entry, UseSystemVerilog()));

  ModuleSimulator simulator(result.signature, result.verilog_text,
                            GetSimulator());

  auto make_array = [](absl::Span<const int64> values) {
    std::vector<Value> elements;
    for (auto v : values) {
      elements.push_back(Value(UBits(v, 32)));
    }
    absl::StatusOr<Value> array = Value::Array(elements);
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
  ret updated_array: bits[32][2][3] = array_update(literal.17, idx, literal.14)
}

)";
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> package,
                           Parser::ParsePackage(text));

  XLS_ASSERT_OK_AND_ASSIGN(Function * entry, package->EntryFunction());
  XLS_ASSERT_OK_AND_ASSIGN(
      auto result, GenerateCombinationalModule(entry, UseSystemVerilog()));

  ModuleSimulator simulator(result.signature, result.verilog_text,
                            GetSimulator());

  auto make_array = [](absl::Span<const int64> values) {
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
    for (auto array : values) {
      elements.push_back(array);
    }
    absl::StatusOr<Value> array_of_values = Value::Array(elements);
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
      auto result, GenerateCombinationalModule(entry, UseSystemVerilog()));

  ModuleSimulator simulator(result.signature, result.verilog_text,
                            GetSimulator());

  auto make_tuple = [](absl::Span<const int64> values) {
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
    for (auto array : values) {
      elements.push_back(array);
    }
    absl::StatusOr<Value> array_of_values = Value::Array(elements);
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
      auto result, GenerateCombinationalModule(entry, UseSystemVerilog()));

  ModuleSimulator simulator(result.signature, result.verilog_text,
                            GetSimulator());

  auto make_array = [](absl::Span<const int64> values) {
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
    for (auto v : values) {
      elements.push_back(v);
    }
    absl::StatusOr<Value> tuple = Value::Tuple(elements);
    EXPECT_TRUE(tuple.ok());
    return tuple.value();
  };

  auto make_array_of_values = [&](absl::Span<const Value> values) {
    std::vector<Value> elements;
    for (auto array : values) {
      elements.push_back(array);
    }
    absl::StatusOr<Value> array_of_values = Value::Array(elements);
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
                           GenerateCombinationalModule(f, UseSystemVerilog()));
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
                           GenerateCombinationalModule(f, UseSystemVerilog()));
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
                           GenerateCombinationalModule(f, UseSystemVerilog()));
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
      auto result, GenerateCombinationalModule(function, UseSystemVerilog()));

  ModuleSimulator simulator(result.signature, result.verilog_text,
                            GetSimulator());

  XLS_ASSERT_OK_AND_ASSIGN(Value a0, Value::UBitsArray({1, 2}, 32));
  XLS_ASSERT_OK_AND_ASSIGN(Value a1, Value::UBitsArray({3, 4, 5}, 32));
  XLS_ASSERT_OK_AND_ASSIGN(Value ret,
                           Value::UBitsArray({1, 2, 3, 4, 5, 1, 2}, 32));

  EXPECT_THAT(simulator.Run({{"a0", a0}, {"a1", a1}}), IsOkAndHolds(ret));
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
      auto result, GenerateCombinationalModule(function, UseSystemVerilog()));

  ModuleSimulator simulator(result.signature, result.verilog_text,
                            GetSimulator());

  XLS_ASSERT_OK_AND_ASSIGN(Value a0, Value::UBitsArray({1, 2}, 32));
  XLS_ASSERT_OK_AND_ASSIGN(Value a1, Value::UBitsArray({3, 4, 5}, 32));
  XLS_ASSERT_OK_AND_ASSIGN(Value a2, Value::SBitsArray({-1}, 32));
  XLS_ASSERT_OK_AND_ASSIGN(Value ret,
                           Value::SBitsArray({1, 2, -1, -1, 3, 4, 5}, 32));

  EXPECT_THAT(simulator.Run({{"a0", a0}, {"a1", a1}, {"a2", a2}}),
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
      auto result, GenerateCombinationalModule(function, UseSystemVerilog()));

  ModuleSimulator simulator(result.signature, result.verilog_text,
                            GetSimulator());

  XLS_ASSERT_OK_AND_ASSIGN(Value ret,
                           Value::SBits2DArray({{5, 6}, {1, 2}, {3, 4}}, 32));

  std::vector<Value> args;
  EXPECT_THAT(simulator.Run(args), IsOkAndHolds(ret));
}

TEST_P(CombinationalGeneratorTest, SimpleProc) {
  const std::string ir_text = R"(package test

chan in(my_in: bits[32], id=0, kind=receive_only,
        metadata="""module_port { flopped: false,  port_order: 1 }""")
chan out(my_out: bits[32], id=1, kind=send_only,
         metadata="""module_port { flopped: false,  port_order: 0 }""")

proc my_proc(my_token: token, my_state: (), init=()) {
  rcv: (token, bits[32]) = receive(my_token, channel_id=0)
  data: bits[32] = tuple_index(rcv, index=1)
  negate: bits[32] = neg(data)
  rcv_token: token = tuple_index(rcv, index=0)
  send: token = send(rcv_token, data=[negate], channel_id=1)
  ret next: (token, ()) = tuple(send, my_state)
}
)";
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> package,
                           Parser::ParsePackage(ir_text));

  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, package->GetProc("my_proc"));
  XLS_ASSERT_OK_AND_ASSIGN(auto result, GenerateCombinationalModuleFromProc(
                                            proc, UseSystemVerilog()));

  ExpectVerilogEqualToGoldenFile(GoldenFilePath(kTestName, kTestdataPath),
                                 result.verilog_text);

  ModuleSimulator simulator(result.signature, result.verilog_text,
                            GetSimulator());
  EXPECT_THAT(simulator.RunAndReturnSingleOutput({{"my_in", SBits(10, 32)}}),
              IsOkAndHolds(SBits(-10, 32)));
  EXPECT_THAT(simulator.RunAndReturnSingleOutput({{"my_in", SBits(0, 32)}}),
              IsOkAndHolds(SBits(0, 32)));
}

TEST_P(CombinationalGeneratorTest, ProcWithMultipleInputChannels) {
  const std::string ir_text = R"(package test

chan in0(my_in0: bits[32], id=0, kind=receive_only,
        metadata="""module_port { flopped: false,  port_order: 0 }""")
chan in1(my_in1: bits[32], id=1, kind=receive_only,
        metadata="""module_port { flopped: false,  port_order: 2 }""")
chan in2(my_in2: bits[32], id=2, kind=receive_only,
        metadata="""module_port { flopped: false,  port_order: 1 }""")
chan out(my_out: bits[32], id=3, kind=send_only,
         metadata="""module_port { flopped: false,  port_order: 0 }""")

proc my_proc(my_token: token, my_state: (), init=()) {
  rcv0: (token, bits[32]) = receive(my_token, channel_id=0)
  rcv0_token: token = tuple_index(rcv0, index=0)
  rcv1: (token, bits[32]) = receive(rcv0_token, channel_id=1)
  rcv1_token: token = tuple_index(rcv1, index=0)
  rcv2: (token, bits[32]) = receive(rcv1_token, channel_id=2)
  rcv2_token: token = tuple_index(rcv2, index=0)
  data0: bits[32] = tuple_index(rcv0, index=1)
  data1: bits[32] = tuple_index(rcv1, index=1)
  data2: bits[32] = tuple_index(rcv2, index=1)
  neg_data1: bits[32] = neg(data1)
  two: bits[32] = literal(value=2)
  data2_times_two: bits[32] = umul(data2, two)
  tmp: bits[32] = add(neg_data1, data2_times_two)
  sum: bits[32] = add(tmp, data0)
  send: token = send(rcv2_token, data=[sum], channel_id=3)
  ret next: (token, ()) = tuple(send, my_state)
}
)";
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> package,
                           Parser::ParsePackage(ir_text));

  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, package->GetProc("my_proc"));
  XLS_ASSERT_OK_AND_ASSIGN(auto result, GenerateCombinationalModuleFromProc(
                                            proc, UseSystemVerilog()));

  ModuleSimulator simulator(result.signature, result.verilog_text,
                            GetSimulator());
  // The computed expression is: my_out = my_in0 - my_in1 + 2 * my_in2
  EXPECT_THAT(simulator.RunAndReturnSingleOutput({{"my_in0", UBits(10, 32)},
                                                  {"my_in1", SBits(7, 32)},
                                                  {"my_in2", SBits(42, 32)}}),
              IsOkAndHolds(UBits(87, 32)));
}

TEST_P(CombinationalGeneratorTest, ProcWithMultipleOutputChannels) {
  const std::string ir_text = R"(package test

chan in(my_in: bits[32], id=0, kind=receive_only,
        metadata="""module_port { flopped: false,  port_order: 1 }""")
chan out0(my_out0: bits[32], id=1, kind=send_only,
          metadata="""module_port { flopped: false,  port_order: 0 }""")
chan out1(my_out1: bits[32], id=2, kind=send_only,
          metadata="""module_port { flopped: false,  port_order: 2 }""")

proc my_proc(my_token: token, my_state: (), init=()) {
  rcv: (token, bits[32]) = receive(my_token, channel_id=0)
  data: bits[32] = tuple_index(rcv, index=1)
  negate: bits[32] = neg(data)
  rcv_token: token = tuple_index(rcv, index=0)
  send0: token = send(rcv_token, data=[data], channel_id=1)
  send1: token = send(send0, data=[negate], channel_id=2)
  ret next: (token, ()) = tuple(send1, my_state)
}
)";
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> package,
                           Parser::ParsePackage(ir_text));

  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, package->GetProc("my_proc"));
  XLS_ASSERT_OK_AND_ASSIGN(auto result, GenerateCombinationalModuleFromProc(
                                            proc, UseSystemVerilog()));

  ModuleSimulator simulator(result.signature, result.verilog_text,
                            GetSimulator());
  EXPECT_THAT(simulator.Run({{"my_in", SBits(10, 32)}}),
              IsOkAndHolds(ModuleSimulator::BitsMap(
                  {{"my_out0", SBits(10, 32)}, {"my_out1", SBits(-10, 32)}})));
}

TEST_P(CombinationalGeneratorTest, NToOneMuxProc) {
  Package package(TestBaseName());
  ProcBuilder pb(TestBaseName(), /*init_value=*/Value::Tuple({}),
                 /*token_name=*/"tkn", /*state_name=*/"st", &package);

  const int64 kInputCount = 4;
  const int64 kSelectorBitCount = 2;

  Type* data_type = package.GetBitsType(32);
  Type* bit_type = package.GetBitsType(1);
  Type* selector_type = package.GetBitsType(kSelectorBitCount);

  int64 port_order = 0;
  auto make_channel_metadata = [&port_order]() {
    ChannelMetadataProto metadata;
    metadata.mutable_module_port()->set_flopped(false);
    metadata.mutable_module_port()->set_port_order(port_order++);
    return metadata;
  };

  BValue token = pb.GetTokenParam();

  // Sends the given data over the given channel. Threads 'token' through the
  // Send operation.
  auto make_send = [&](Channel* ch, BValue data) {
    BValue send = pb.Send(ch, token, {data});
    token = send;
    return send;
  };

  // Adds a receive instruction and returns the data BValue. Threads 'token'
  // through the Receive operation.
  auto make_receive = [&](Channel* ch) {
    BValue receive = pb.Receive(ch, token);
    token = pb.TupleIndex(receive, 0);
    return pb.TupleIndex(receive, 1);
  };

  // Add the selector module port which selects which input to forward.
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * selector_channel,
      package.CreateChannel(
          "selector", ChannelKind::kReceiveOnly,
          {DataElement{.name = "selector", .type = selector_type}},
          make_channel_metadata()));
  BValue selector = make_receive(selector_channel);

  // Add the output ready channel. It's an input and will be used to generate
  // the input ready outputs.
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * out_ready_channel,
      package.CreateChannel("out_rdy", ChannelKind::kReceiveOnly,
                            {DataElement{.name = "out_rdy", .type = bit_type}},
                            make_channel_metadata()));
  BValue output_ready = make_receive(out_ready_channel);

  // Generate all the input ports and their ready/valid signals.
  std::vector<BValue> input_datas;
  std::vector<BValue> input_valids;
  for (int64 i = 0; i < kInputCount; ++i) {
    XLS_ASSERT_OK_AND_ASSIGN(
        Channel * data_channel,
        package.CreateChannel(absl::StrFormat("in_%d", i),
                              ChannelKind::kReceiveOnly,
                              {DataElement{.name = absl::StrFormat("in_%d", i),
                                           .type = data_type}},
                              make_channel_metadata()));
    input_datas.push_back(make_receive(data_channel));

    XLS_ASSERT_OK_AND_ASSIGN(
        Channel * valid_channel,
        package.CreateChannel(
            absl::StrFormat("in_%d_vld", i), ChannelKind::kReceiveOnly,
            {DataElement{.name = absl::StrFormat("in_%d_vld", i),
                         .type = bit_type}},
            make_channel_metadata()));
    input_valids.push_back(make_receive(valid_channel));

    BValue ready = pb.And(
        output_ready, pb.Eq(selector, pb.Literal(UBits(i, kSelectorBitCount))));
    XLS_ASSERT_OK_AND_ASSIGN(
        Channel * ready_channel,
        package.CreateChannel(
            absl::StrFormat("in_%d_rdy", i), ChannelKind::kSendOnly,
            {DataElement{.name = absl::StrFormat("in_%d_rdy", i),
                         .type = bit_type}},
            make_channel_metadata()));
    make_send(ready_channel, ready);
  }

  // Output data is a select amongst the input data.
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * out_data_channel,
      package.CreateChannel("out", ChannelKind::kSendOnly,
                            {DataElement{.name = "out", .type = data_type}},
                            make_channel_metadata()));
  make_send(out_data_channel, pb.Select(selector, /*cases=*/input_datas));

  // Output valid is a select amongs the input valid signals.
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * out_valid_channel,
      package.CreateChannel("out_vld", ChannelKind::kSendOnly,
                            {DataElement{.name = "out_vld", .type = bit_type}},
                            make_channel_metadata()));
  make_send(out_valid_channel, pb.Select(selector, /*cases=*/input_valids));

  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, pb.Build(token, pb.GetStateParam()));
  XLS_ASSERT_OK_AND_ASSIGN(auto result, GenerateCombinationalModuleFromProc(
                                            proc, UseSystemVerilog()));
  ExpectVerilogEqualToGoldenFile(GoldenFilePath(kTestName, kTestdataPath),
                                 result.verilog_text);
}

TEST_P(CombinationalGeneratorTest, OneToNMuxProc) {
  Package package(TestBaseName());
  ProcBuilder pb(TestBaseName(), /*init_value=*/Value::Tuple({}),
                 /*token_name=*/"tkn", /*state_name=*/"st", &package);

  const int64 kOutputCount = 4;
  const int64 kSelectorBitCount = 2;

  Type* data_type = package.GetBitsType(32);
  Type* bit_type = package.GetBitsType(1);
  Type* selector_type = package.GetBitsType(kSelectorBitCount);

  int64 port_order = 0;
  auto make_channel_metadata = [&port_order]() {
    ChannelMetadataProto metadata;
    metadata.mutable_module_port()->set_flopped(false);
    metadata.mutable_module_port()->set_port_order(port_order++);
    return metadata;
  };

  BValue token = pb.GetTokenParam();

  // Sends the given data over the given channel. Threads 'token' through the
  // Send operation.
  auto make_send = [&](Channel* ch, BValue data) {
    BValue send = pb.Send(ch, token, {data});
    token = send;
    return send;
  };

  // Adds a receive instruction and returns the data BValue. Threads 'token'
  // through the Receive operation.
  auto make_receive = [&](Channel* ch) {
    BValue receive = pb.Receive(ch, token);
    token = pb.TupleIndex(receive, 0);
    return pb.TupleIndex(receive, 1);
  };

  // Add the selector module port which selects which input to forward.
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * selector_channel,
      package.CreateChannel(
          "selector", ChannelKind::kReceiveOnly,
          {DataElement{.name = "selector", .type = selector_type}},
          make_channel_metadata()));
  BValue selector = make_receive(selector_channel);

  // Add the input data channel.
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * input_data_channel,
      package.CreateChannel("in", ChannelKind::kReceiveOnly,
                            {DataElement{.name = "in", .type = data_type}},
                            make_channel_metadata()));
  BValue input = make_receive(input_data_channel);

  // Add the input valid channel. It's an input and will be used to generate
  // the output valid outputs.
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * input_valid_channel,
      package.CreateChannel("in_vld", ChannelKind::kReceiveOnly,
                            {DataElement{.name = "in_vld", .type = bit_type}},
                            make_channel_metadata()));
  BValue input_valid = make_receive(input_valid_channel);

  // Generate all the output ports and their ready/valid signals.
  std::vector<BValue> output_readys;
  for (int64 i = 0; i < kOutputCount; ++i) {
    XLS_ASSERT_OK_AND_ASSIGN(
        Channel * data_channel,
        package.CreateChannel(absl::StrFormat("out_%d", i),
                              ChannelKind::kSendOnly,
                              {DataElement{.name = absl::StrFormat("out_%d", i),
                                           .type = data_type}},
                              make_channel_metadata()));
    make_send(data_channel, input);

    XLS_ASSERT_OK_AND_ASSIGN(
        Channel * ready_channel,
        package.CreateChannel(
            absl::StrFormat("out_%d_rdy", i), ChannelKind::kReceiveOnly,
            {DataElement{.name = absl::StrFormat("out_%d_rdy", i),
                         .type = bit_type}},
            make_channel_metadata()));
    output_readys.push_back(make_receive(ready_channel));

    BValue valid = pb.And(
        input_valid, pb.Eq(selector, pb.Literal(UBits(i, kSelectorBitCount))));
    XLS_ASSERT_OK_AND_ASSIGN(
        Channel * valid_channel,
        package.CreateChannel(
            absl::StrFormat("out_%d_vld", i), ChannelKind::kSendOnly,
            {DataElement{.name = absl::StrFormat("out_%d_vld", i),
                         .type = bit_type}},
            make_channel_metadata()));
    make_send(valid_channel, valid);
  }

  // Output ready is a select amongs the input ready signals.
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * input_ready_channel,
      package.CreateChannel("in_rdy", ChannelKind::kSendOnly,
                            {DataElement{.name = "in_rdy", .type = bit_type}},
                            make_channel_metadata()));
  make_send(input_ready_channel, pb.Select(selector, /*cases=*/output_readys));

  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, pb.Build(token, pb.GetStateParam()));
  XLS_ASSERT_OK_AND_ASSIGN(auto result, GenerateCombinationalModuleFromProc(
                                            proc, UseSystemVerilog()));
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
