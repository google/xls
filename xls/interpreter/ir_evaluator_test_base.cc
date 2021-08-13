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

#include "xls/interpreter/ir_evaluator_test_base.h"

#include "absl/status/statusor.h"
#include "absl/strings/substitute.h"
#include "xls/common/logging/logging.h"
#include "xls/common/math_util.h"
#include "xls/common/status/matchers.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/bits.h"
#include "xls/ir/keyword_args.h"
#include "xls/ir/value_helpers.h"

namespace xls {

using status_testing::IsOkAndHolds;
using status_testing::StatusIs;
using ::testing::HasSubstr;

using ArgMap = absl::flat_hash_map<std::string, Value>;

TEST_P(IrEvaluatorTestBase, InterpretLiteral) {
  Package package("my_package");
  XLS_ASSERT_OK_AND_ASSIGN(Function * function,
                           ParseAndGetFunction(&package, R"(
  fn literal() -> bits[34] {
    ret result: bits[34] = literal(value=42)
  }
  )"));

  IrEvaluatorTestParam param = GetParam();
  EXPECT_THAT(RunWithValues(function, {}), IsOkAndHolds(Value(UBits(42, 34))));
}

TEST_P(IrEvaluatorTestBase, InterpretWideLiteral) {
  Package package("my_package");
  XLS_ASSERT_OK_AND_ASSIGN(Function * function,
                           ParseAndGetFunction(&package, R"(
  fn literal() -> bits[123] {
    ret result: bits[123] = literal(value=0xabcd_1234_5678_0011_dead_beef_bcde)
  }
  )"));

  IrEvaluatorTestParam param = GetParam();
  EXPECT_THAT(RunWithValues(function, {}),
              IsOkAndHolds(Parser::ParseTypedValue(
                               "bits[123]:0xabcd_1234_5678_0011_dead_beef_bcde")
                               .value()));
}

TEST_P(IrEvaluatorTestBase, InterpretIdentity) {
  Package package("my_package");
  XLS_ASSERT_OK_AND_ASSIGN(Function * function,
                           ParseAndGetFunction(&package, R"(
  fn get_identity(x: bits[8]) -> bits[8] {
    ret identity.1: bits[8] = identity(x)
  }
  )"));

  IrEvaluatorTestParam param = GetParam();
  EXPECT_THAT(RunWithValues(function, {Value(UBits(2, 8))}),
              IsOkAndHolds(Value(UBits(2, 8))));
}

TEST_P(IrEvaluatorTestBase, InterpretIdentityTuple) {
  Package package("my_package");
  XLS_ASSERT_OK_AND_ASSIGN(Function * function,
                           ParseAndGetFunction(&package, R"(
  fn get_identity(x: (bits[5], bits[32])) -> (bits[5], bits[32]) {
    ret identity.1: (bits[5], bits[32]) = identity(x)
  }
  )"));

  // Use a big ol' value so data dumps might contain helpful patterns.
  const Value x =
      Value::Tuple({Value(UBits(3, 5)), Value(UBits(2051583859, 32))});
  IrEvaluatorTestParam param = GetParam();
  EXPECT_THAT(RunWithValues(function, {x}), IsOkAndHolds(x));
}

TEST_P(IrEvaluatorTestBase, InterpretIdentityArrayOfTuple) {
  Package package("my_package");
  XLS_ASSERT_OK_AND_ASSIGN(Function * function,
                           ParseAndGetFunction(&package, R"(
  fn get_identity(x: (bits[8], bits[32])[8]) -> (bits[8], bits[32])[8] {
    ret identity.1: (bits[8], bits[32])[8] = identity(x)
  }
  )"));

  const Value t = Value::Tuple({Value(UBits(2, 8)), Value(UBits(7, 32))});
  std::vector<Value> v(8, t);
  XLS_ASSERT_OK_AND_ASSIGN(const Value a, Value::Array(v));
  EXPECT_THAT(RunWithValues(function, {a}), IsOkAndHolds(a));
}

TEST_P(IrEvaluatorTestBase, InterpretNegPositiveValue) {
  // Positive values should become negative.
  Package package("my_package");
  XLS_ASSERT_OK_AND_ASSIGN(Function * function, get_neg_function(&package));

  EXPECT_THAT(RunWithValues(function, {Value(UBits(1, 4))}),
              IsOkAndHolds(Value(SBits(-1, 4))));
}

TEST_P(IrEvaluatorTestBase, InterpretNegNegativeValue) {
  // Negative values should become positive.
  Package package("my_package");
  XLS_ASSERT_OK_AND_ASSIGN(Function * function, get_neg_function(&package));
  EXPECT_THAT(RunWithValues(function, {Value(SBits(-1, 4))}),
              IsOkAndHolds(Value(UBits(1, 4))));
}

TEST_P(IrEvaluatorTestBase, InterpretNegZeroValue) {
  // Zero should stay zero.
  Package package("my_package");
  XLS_ASSERT_OK_AND_ASSIGN(Function * function, get_neg_function(&package));
  EXPECT_THAT(RunWithValues(function, {Value(UBits(0, 4))}),
              IsOkAndHolds(Value(UBits(0, 4))));
}

TEST_P(IrEvaluatorTestBase, InterpretNegMaxPositiveValue) {
  // Test the maximum positive 2s-complement value that fits in four bits.
  Package package("my_package");
  XLS_ASSERT_OK_AND_ASSIGN(Function * function, get_neg_function(&package));
  EXPECT_THAT(RunWithValues(function, {Value(UBits(7, 4))}),
              IsOkAndHolds(Value(UBits(0b1001, 4))));
}

TEST_P(IrEvaluatorTestBase, InterpretNegMaxMinValue) {
  // Max minimum 2s-complement value that fits in four bits.
  Package package("my_package");
  XLS_ASSERT_OK_AND_ASSIGN(Function * function, get_neg_function(&package));
  XLS_ASSERT_OK_AND_ASSIGN(Value result,
                           RunWithValues(function, {Value(SBits(-8, 4))}));
  EXPECT_EQ(result, Value(UBits(0b1000, 4))) << "Actual: " << result;
}

TEST_P(IrEvaluatorTestBase, InterpretNot) {
  Package package("my_package");
  XLS_ASSERT_OK_AND_ASSIGN(Function * function,
                           ParseAndGetFunction(&package, R"(
  fn invert_bits(a: bits[4]) -> bits[4] {
    ret not.1: bits[4] = not(a)
  }
  )"));

  EXPECT_THAT(RunWithValues(function, {Value(UBits(0b1111, 4))}),
              IsOkAndHolds(Value(UBits(0b0000, 4))));
  EXPECT_THAT(RunWithValues(function, {Value(UBits(0b0000, 4))}),
              IsOkAndHolds(Value(UBits(0b1111, 4))));
  EXPECT_THAT(RunWithValues(function, {Value(UBits(0b1010, 4))}),
              IsOkAndHolds(Value(UBits(0b0101, 4))));
}

TEST_P(IrEvaluatorTestBase, InterpretSixAndThree) {
  Package package("my_package");
  XLS_ASSERT_OK_AND_ASSIGN(Function * function,
                           ParseAndGetFunction(&package, R"(
  fn six_or_three() -> bits[8] {
    literal.1: bits[8] = literal(value=6)
    literal.2: bits[8] = literal(value=3)
    ret and.3: bits[8] = and(literal.1, literal.2)
  }
  )"));

  EXPECT_THAT(RunWithBits(function, /*args=*/{}), IsOkAndHolds(UBits(2, 8)));
}

TEST_P(IrEvaluatorTestBase, InterpretOr) {
  Package package("my_package");
  XLS_ASSERT_OK_AND_ASSIGN(Function * function,
                           ParseAndGetFunction(&package, R"(
  fn six_or_three() -> bits[8] {
    literal.1: bits[8] = literal(value=6)
    literal.2: bits[8] = literal(value=3)
    ret or.3: bits[8] = or(literal.1, literal.2)
  }
  )"));

  EXPECT_THAT(RunWithValues(function, /*args=*/{}),
              IsOkAndHolds(Value(UBits(0b111, 8))));
}

TEST_P(IrEvaluatorTestBase, InterpretXor) {
  Package package("my_package");
  XLS_ASSERT_OK_AND_ASSIGN(Function * function,
                           ParseAndGetFunction(&package, R"(
  fn six_xor_three() -> bits[8] {
    literal.1: bits[8] = literal(value=6)
    literal.2: bits[8] = literal(value=3)
    ret xor.3: bits[8] = xor(literal.1, literal.2)
  }
  )"));

  EXPECT_THAT(RunWithValues(function, /*args=*/{}),
              IsOkAndHolds(Value(UBits(0b101, 8))));
}

TEST_P(IrEvaluatorTestBase, InterpretNaryXor) {
  Package package("my_package");
  XLS_ASSERT_OK_AND_ASSIGN(Function * function,
                           ParseAndGetFunction(&package, R"(
  fn six_xor_three_xor_one() -> bits[8] {
    literal.1: bits[8] = literal(value=6)
    literal.2: bits[8] = literal(value=3)
    literal.3: bits[8] = literal(value=1)
    ret xor.4: bits[8] = xor(literal.1, literal.2, literal.3)
  }
  )"));

  EXPECT_THAT(RunWithValues(function, /*args=*/{}),
              IsOkAndHolds(Value(UBits(0b100, 8))));
}

TEST_P(IrEvaluatorTestBase, InterpretNaryOr) {
  Package package("my_package");
  XLS_ASSERT_OK_AND_ASSIGN(Function * function,
                           ParseAndGetFunction(&package, R"(
  fn f() -> bits[8] {
    literal.1: bits[8] = literal(value=4)
    literal.2: bits[8] = literal(value=3)
    literal.3: bits[8] = literal(value=1)
    ret or.4: bits[8] = or(literal.1, literal.2, literal.3)
  }
  )"));

  EXPECT_THAT(RunWithValues(function, /*args=*/{}),
              IsOkAndHolds(Value(UBits(0b111, 8))));
}

TEST_P(IrEvaluatorTestBase, InterpretNaryNor) {
  Package package("my_package");
  XLS_ASSERT_OK_AND_ASSIGN(Function * function,
                           ParseAndGetFunction(&package, R"(
  fn f() -> bits[8] {
    literal.1: bits[8] = literal(value=4)
    literal.2: bits[8] = literal(value=2)
    literal.3: bits[8] = literal(value=0)
    ret nor.4: bits[8] = nor(literal.1, literal.2, literal.3)
  }
  )"));

  EXPECT_THAT(RunWithValues(function, /*args=*/{}),
              IsOkAndHolds(Value(UBits(0b11111001, 8))));
}

TEST_P(IrEvaluatorTestBase, InterpretNaryAnd) {
  Package package("my_package");
  XLS_ASSERT_OK_AND_ASSIGN(Function * function,
                           ParseAndGetFunction(&package, R"(
  fn f() -> bits[8] {
    literal.1: bits[8] = literal(value=6)
    literal.2: bits[8] = literal(value=3)
    literal.3: bits[8] = literal(value=2)
    ret and.4: bits[8] = and(literal.1, literal.2, literal.3)
  }
  )"));

  EXPECT_THAT(RunWithValues(function, /*args=*/{}),
              IsOkAndHolds(Value(UBits(0b010, 8))));
}

TEST_P(IrEvaluatorTestBase, InterpretNaryNand) {
  Package package("my_package");
  XLS_ASSERT_OK_AND_ASSIGN(Function * function,
                           ParseAndGetFunction(&package, R"(
  fn f() -> bits[8] {
    literal.1: bits[8] = literal(value=6)
    literal.2: bits[8] = literal(value=3)
    literal.3: bits[8] = literal(value=2)
    ret nand.4: bits[8] = nand(literal.1, literal.2, literal.3)
  }
  )"));

  EXPECT_THAT(RunWithValues(function, /*args=*/{}),
              IsOkAndHolds(Value(UBits(0b11111101, 8))));
}

absl::Status RunZeroExtendTest(const IrEvaluatorTestParam& param,
                               int64_t input_size, int64_t output_size) {
  constexpr absl::string_view ir_text = R"(
  package test

  fn zero_extend() -> bits[$0] {
    literal.1: bits[$1] = literal(value=3)
    ret zero_ext.2: bits[$0] = zero_ext(literal.1, new_bit_count=$0)
  }
  )";

  std::string formatted_ir = absl::Substitute(ir_text, output_size, input_size);
  XLS_ASSIGN_OR_RETURN(auto package, Parser::ParsePackage(formatted_ir));
  XLS_ASSIGN_OR_RETURN(Function * function, package->EntryFunction());

  EXPECT_THAT(param.evaluator(function, /*args=*/{}),
              IsOkAndHolds(Value(UBits(3, output_size))));

  return absl::OkStatus();
}

TEST_P(IrEvaluatorTestBase, InterpretZeroExtend) {
  XLS_ASSERT_OK(RunZeroExtendTest(GetParam(), 3, 8));
  XLS_ASSERT_OK(RunZeroExtendTest(GetParam(), 3, 1024));
  XLS_ASSERT_OK(RunZeroExtendTest(GetParam(), 1024, 1025));
}

absl::Status RunSignExtendTest(const IrEvaluatorTestParam& param,
                               const Bits& input, int new_bit_count) {
  constexpr absl::string_view ir_text = R"(
  package test

  fn concatenate() -> bits[$0] {
    literal.1: bits[$1] = literal(value=$2)
    ret sign_ext.2: bits[$0] = sign_ext(literal.1, new_bit_count=$0)
  }
  )";

  std::string formatted_text = absl::Substitute(
      ir_text, new_bit_count, input.bit_count(), input.ToString());
  XLS_ASSIGN_OR_RETURN(auto package, Parser::ParsePackage(formatted_text));
  XLS_ASSIGN_OR_RETURN(Function * function, package->EntryFunction());
  Value expected(bits_ops::SignExtend(input, new_bit_count));
  EXPECT_THAT(param.evaluator(function, {}), IsOkAndHolds(expected));
  return absl::OkStatus();
}

TEST_P(IrEvaluatorTestBase, InterpretSignExtend) {
  XLS_ASSERT_OK(RunSignExtendTest(GetParam(), UBits(0b11, 2), 8));
  XLS_ASSERT_OK(RunSignExtendTest(GetParam(), UBits(0b01, 2), 8));
  XLS_ASSERT_OK(RunSignExtendTest(GetParam(), UBits(0b11, 57), 1023));
  XLS_ASSERT_OK(RunSignExtendTest(GetParam(), UBits(0b11, 64), 128));
  XLS_ASSERT_OK(RunSignExtendTest(GetParam(), UBits(0b01, 64), 128));
}

TEST_P(IrEvaluatorTestBase, InterpretTwoAddTwo) {
  Package package("my_package");
  // Big values to help debugging, again.
  XLS_ASSERT_OK_AND_ASSIGN(Function * function,
                           ParseAndGetFunction(&package, R"(
  fn two_add_two() -> bits[32] {
    literal.1: bits[32] = literal(value=127)
    literal.2: bits[32] = literal(value=255)
    ret add.3: bits[32] = add(literal.1, literal.2)
  }
  )"));

  EXPECT_THAT(RunWithValues(function, /*args=*/{}),
              IsOkAndHolds(Value(UBits(382, 32))));
}

TEST_P(IrEvaluatorTestBase, InterpretMaxAddTwo) {
  Package package("my_package");
  XLS_ASSERT_OK_AND_ASSIGN(Function * function,
                           ParseAndGetFunction(&package, R"(
  fn max_add_two() -> bits[3] {
    literal.1: bits[3] = literal(value=7)
    literal.2: bits[3] = literal(value=2)
    ret add.3: bits[3] = add(literal.1, literal.2)
  }
  )"));

  EXPECT_THAT(RunWithValues(function, /*args=*/{}),
              IsOkAndHolds(Value(UBits(1, 3))));
}

TEST_P(IrEvaluatorTestBase, InterpretEq) {
  Package package("my_package");
  XLS_ASSERT_OK_AND_ASSIGN(Function * function,
                           ParseAndGetFunction(&package, R"(
  fn equal(a: bits[32], b: bits[32]) -> bits[1] {
    ret eq.1: bits[1] = eq(a, b)
  }
  )"));

  absl::flat_hash_map<std::string, Value> args;

  // a > b
  args = {{"a", Value(UBits(4, 32))}, {"b", Value(UBits(1, 32))}};
  EXPECT_THAT(GetParam().kwargs_evaluator(function, args),
              IsOkAndHolds(Value(UBits(0, 1))));

  // a < b
  args = {{"a", Value(UBits(1, 32))}, {"b", Value(UBits(4, 32))}};
  EXPECT_THAT(GetParam().kwargs_evaluator(function, args),
              IsOkAndHolds(Value(UBits(0, 1))));

  // a ==  b
  args = {{"a", Value(UBits(4, 32))}, {"b", Value(UBits(4, 32))}};
  EXPECT_THAT(GetParam().kwargs_evaluator(function, args),
              IsOkAndHolds(Value(UBits(1, 1))));
}

TEST_P(IrEvaluatorTestBase, InterpretULt) {
  Package package("my_package");
  XLS_ASSERT_OK_AND_ASSIGN(Function * function,
                           ParseAndGetFunction(&package, R"(
  fn less_than(a: bits[32], b: bits[32]) -> bits[1] {
    ret ult.1: bits[1] = ult(a, b)
  }
  )"));

  absl::flat_hash_map<std::string, Value> args;

  // a > b
  args = {{"a", Value(UBits(4, 32))}, {"b", Value(UBits(1, 32))}};
  EXPECT_THAT(GetParam().kwargs_evaluator(function, args),
              IsOkAndHolds(Value(UBits(0, 1))));

  // a < b
  args = {{"a", Value(UBits(1, 32))}, {"b", Value(UBits(4, 32))}};
  EXPECT_THAT(GetParam().kwargs_evaluator(function, args),
              IsOkAndHolds(Value(UBits(1, 1))));

  // a ==  b
  args = {{"a", Value(UBits(4, 32))}, {"b", Value(UBits(4, 32))}};
  EXPECT_THAT(GetParam().kwargs_evaluator(function, args),
              IsOkAndHolds(Value(UBits(0, 1))));
}

TEST_P(IrEvaluatorTestBase, InterpretULe) {
  Package package("my_package");
  XLS_ASSERT_OK_AND_ASSIGN(Function * function,
                           ParseAndGetFunction(&package, R"(
  fn less_than_or_equal(a: bits[32], b: bits[32]) -> bits[1] {
    ret ule.1: bits[1] = ule(a, b)
  }
  )"));

  absl::flat_hash_map<std::string, Value> args;

  // a > b
  args = {{"a", Value(UBits(4, 32))}, {"b", Value(UBits(1, 32))}};
  EXPECT_THAT(GetParam().kwargs_evaluator(function, args),
              IsOkAndHolds(Value(UBits(0, 1))));

  // a < b
  args = {{"a", Value(UBits(1, 32))}, {"b", Value(UBits(4, 32))}};
  EXPECT_THAT(GetParam().kwargs_evaluator(function, args),
              IsOkAndHolds(Value(UBits(1, 1))));

  // a ==  b
  args = {{"a", Value(UBits(4, 32))}, {"b", Value(UBits(4, 32))}};
  EXPECT_THAT(GetParam().kwargs_evaluator(function, args),
              IsOkAndHolds(Value(UBits(1, 1))));
}

TEST_P(IrEvaluatorTestBase, InterpretUGt) {
  Package package("my_package");
  XLS_ASSERT_OK_AND_ASSIGN(Function * function,
                           ParseAndGetFunction(&package, R"(
  fn greater_than(a: bits[32], b: bits[32]) -> bits[1] {
    ret ugt.1: bits[1] = ugt(a, b)
  }
  )"));

  absl::flat_hash_map<std::string, Value> args;

  // a > b
  args = {{"a", Value(UBits(4, 32))}, {"b", Value(UBits(1, 32))}};
  EXPECT_THAT(GetParam().kwargs_evaluator(function, args),
              IsOkAndHolds(Value(UBits(1, 1))));

  // a < b
  args = {{"a", Value(UBits(1, 32))}, {"b", Value(UBits(4, 32))}};
  EXPECT_THAT(GetParam().kwargs_evaluator(function, args),
              IsOkAndHolds(Value(UBits(0, 1))));

  // a ==  b
  args = {{"a", Value(UBits(4, 32))}, {"b", Value(UBits(4, 32))}};
  EXPECT_THAT(GetParam().kwargs_evaluator(function, args),
              IsOkAndHolds(Value(UBits(0, 1))));
}

TEST_P(IrEvaluatorTestBase, InterpretUGe) {
  Package package("my_package");
  XLS_ASSERT_OK_AND_ASSIGN(Function * function,
                           ParseAndGetFunction(&package, R"(
  fn greater_than_or_equal(a: bits[32], b: bits[32]) -> bits[1] {
    ret uge.1: bits[1] = uge(a, b)
  }
  )"));

  absl::flat_hash_map<std::string, Value> args;

  // a > b
  args = {{"a", Value(UBits(4, 32))}, {"b", Value(UBits(1, 32))}};
  EXPECT_THAT(GetParam().kwargs_evaluator(function, args),
              IsOkAndHolds(Value(UBits(1, 1))));

  // a < b
  args = {{"a", Value(UBits(1, 32))}, {"b", Value(UBits(4, 32))}};
  EXPECT_THAT(GetParam().kwargs_evaluator(function, args),
              IsOkAndHolds(Value(UBits(0, 1))));

  // a ==  b
  args = {{"a", Value(UBits(4, 32))}, {"b", Value(UBits(4, 32))}};
  EXPECT_THAT(GetParam().kwargs_evaluator(function, args),
              IsOkAndHolds(Value(UBits(1, 1))));
}

TEST_P(IrEvaluatorTestBase, InterpretSLt) {
  Package package("my_package");
  XLS_ASSERT_OK_AND_ASSIGN(Function * function,
                           ParseAndGetFunction(&package, R"(
  fn less_than(a: bits[32], b: bits[32]) -> bits[1] {
    ret slt.1: bits[1] = slt(a, b)
  }
  )"));

  absl::flat_hash_map<std::string, Value> args;

  // a > b
  args = {{"a", Value(SBits(4, 32))}, {"b", Value(SBits(1, 32))}};
  EXPECT_THAT(GetParam().kwargs_evaluator(function, args),
              IsOkAndHolds(Value(UBits(0, 1))));

  // a < b
  args = {{"a", Value(SBits(-1, 32))}, {"b", Value(SBits(4, 32))}};
  EXPECT_THAT(GetParam().kwargs_evaluator(function, args),
              IsOkAndHolds(Value(UBits(1, 1))));

  // a ==  b
  args = {{"a", Value(SBits(-4, 32))}, {"b", Value(SBits(-4, 32))}};
  EXPECT_THAT(GetParam().kwargs_evaluator(function, args),
              IsOkAndHolds(Value(UBits(0, 1))));
}

TEST_P(IrEvaluatorTestBase, InterpretSLe) {
  Package package("my_package");
  XLS_ASSERT_OK_AND_ASSIGN(Function * function,
                           ParseAndGetFunction(&package, R"(
  fn less_than_or_equal(a: bits[32], b: bits[32]) -> bits[1] {
    ret sle.1: bits[1] = sle(a, b)
  }
  )"));

  absl::flat_hash_map<std::string, Value> args;

  // a > b
  args = {{"a", Value(SBits(-1, 32))}, {"b", Value(SBits(-4, 32))}};
  EXPECT_THAT(GetParam().kwargs_evaluator(function, args),
              IsOkAndHolds(Value(UBits(0, 1))));

  // a < b
  args = {{"a", Value(SBits(-1, 32))}, {"b", Value(SBits(4, 32))}};
  EXPECT_THAT(GetParam().kwargs_evaluator(function, args),
              IsOkAndHolds(Value(UBits(1, 1))));

  // a ==  b
  args = {{"a", Value(SBits(4, 32))}, {"b", Value(SBits(4, 32))}};
  EXPECT_THAT(GetParam().kwargs_evaluator(function, args),
              IsOkAndHolds(Value(UBits(1, 1))));
}

TEST_P(IrEvaluatorTestBase, InterpretSGt) {
  Package package("my_package");
  XLS_ASSERT_OK_AND_ASSIGN(Function * function,
                           ParseAndGetFunction(&package, R"(
  fn greater_than(a: bits[32], b: bits[32]) -> bits[1] {
    ret sgt.1: bits[1] = sgt(a, b)
  }
  )"));

  absl::flat_hash_map<std::string, Value> args;

  // a > b
  args = {{"a", Value(SBits(-123, 32))}, {"b", Value(SBits(-442, 32))}};
  EXPECT_THAT(GetParam().kwargs_evaluator(function, args),
              IsOkAndHolds(Value(UBits(1, 1))));

  // a < b
  args = {{"a", Value(SBits(-1, 32))}, {"b", Value(SBits(400, 32))}};
  EXPECT_THAT(GetParam().kwargs_evaluator(function, args),
              IsOkAndHolds(Value(UBits(0, 1))));

  // a ==  b
  args = {{"a", Value(SBits(4, 32))}, {"b", Value(SBits(4, 32))}};
  EXPECT_THAT(GetParam().kwargs_evaluator(function, args),
              IsOkAndHolds(Value(UBits(0, 1))));
}

TEST_P(IrEvaluatorTestBase, InterpretSGe) {
  Package package("my_package");
  XLS_ASSERT_OK_AND_ASSIGN(Function * function,
                           ParseAndGetFunction(&package, R"(
  fn greater_than_or_equal(a: bits[32], b: bits[32]) -> bits[1] {
    ret sge.1: bits[1] = sge(a, b)
  }
  )"));

  absl::flat_hash_map<std::string, Value> args;

  // a > b
  args = {{"a", Value(SBits(4, 32))}, {"b", Value(SBits(-1, 32))}};
  EXPECT_THAT(GetParam().kwargs_evaluator(function, args),
              IsOkAndHolds(Value(UBits(1, 1))));

  // a < b
  args = {{"a", Value(SBits(-10, 32))}, {"b", Value(SBits(4, 32))}};
  EXPECT_THAT(GetParam().kwargs_evaluator(function, args),
              IsOkAndHolds(Value(UBits(0, 1))));

  // a ==  b
  args = {{"a", Value(SBits(400, 32))}, {"b", Value(SBits(400, 32))}};
  EXPECT_THAT(GetParam().kwargs_evaluator(function, args),
              IsOkAndHolds(Value(UBits(1, 1))));
}

TEST_P(IrEvaluatorTestBase, InterpretTwoMulThree) {
  Package package("my_package");
  XLS_ASSERT_OK_AND_ASSIGN(Function * function,
                           ParseAndGetFunction(&package, R"(
  fn two_umul_three() -> bits[32] {
    literal.1: bits[32] = literal(value=2)
    literal.2: bits[32] = literal(value=3)
    ret umul.3: bits[32] = umul(literal.1, literal.2)
  }
  )"));

  EXPECT_THAT(RunWithValues(function, /*args=*/{}),
              IsOkAndHolds(Value(UBits(6, 32))));
}

TEST_P(IrEvaluatorTestBase, InterpretMaxMulTwo) {
  Package package("my_package");
  XLS_ASSERT_OK_AND_ASSIGN(Function * function,
                           ParseAndGetFunction(&package, R"(
  fn max_plus_two() -> bits[3] {
    literal.1: bits[3] = literal(value=7)
    literal.2: bits[3] = literal(value=2)
    ret umul.3: bits[3] = umul(literal.1, literal.2)
  }
  )"));

  EXPECT_THAT(RunWithValues(function, /*args=*/{}),
              IsOkAndHolds(Value(UBits(6, 3))));
}

TEST_P(IrEvaluatorTestBase, MixedWidthMultiplication) {
  Package package("my_package");
  XLS_ASSERT_OK_AND_ASSIGN(Function * function,
                           ParseAndGetFunction(&package, R"(
  fn mixed_mul(x: bits[7], y: bits[3]) -> bits[5] {
    ret umul.1: bits[5] = umul(x, y)
  }
  )"));

  EXPECT_THAT(RunWithUint64s(function, {0, 0}), IsOkAndHolds(0));
  EXPECT_THAT(RunWithUint64s(function, {127, 7}), IsOkAndHolds(25));
  EXPECT_THAT(RunWithUint64s(function, {3, 5}), IsOkAndHolds(15));
}

TEST_P(IrEvaluatorTestBase, MixedWidthMultiplicationExtraWideResult) {
  Package package("my_package");
  XLS_ASSERT_OK_AND_ASSIGN(Function * function,
                           ParseAndGetFunction(&package, R"(
  fn mixed_mul(x: bits[5], y: bits[4]) -> bits[20] {
    ret umul.1: bits[20] = umul(x, y)
  }
  )"));

  EXPECT_THAT(RunWithUint64s(function, {0, 0}), IsOkAndHolds(0));
  EXPECT_THAT(RunWithUint64s(function, {31, 15}), IsOkAndHolds(465));
  EXPECT_THAT(RunWithUint64s(function, {11, 5}), IsOkAndHolds(55));
}

TEST_P(IrEvaluatorTestBase, MixedWidthMultiplicationExhaustive) {
  // Exhaustively check all bit width and value combinations of a mixed-width
  // unsigned multiply up to a small constant bit width.
  constexpr absl::string_view ir_text = R"(
  fn mixed_mul(x: bits[$0], y: bits[$1]) -> bits[$2] {
    ret umul.1: bits[$2] = umul(x, y)
  }
  )";

  const int64_t kMaxWidth = 3;
  for (int x_width = 1; x_width <= kMaxWidth; ++x_width) {
    for (int y_width = 1; y_width <= kMaxWidth; ++y_width) {
      for (int result_width = 1; result_width <= kMaxWidth; ++result_width) {
        std::string formatted_ir =
            absl::Substitute(ir_text, x_width, y_width, result_width);
        Package package("my_package");
        XLS_ASSERT_OK_AND_ASSIGN(Function * function,
                                 ParseAndGetFunction(&package, formatted_ir));
        for (int x = 0; x < 1 << x_width; ++x) {
          for (int y = 0; y < 1 << y_width; ++y) {
            Bits x_bits = UBits(x, x_width);
            Bits y_bits = UBits(y, y_width);
            // The expected result width may be narrower or wider than the
            // result produced by bits_ops::Umul so just sign extend to a wide
            // value then slice it to size.
            Bits expected = bits_ops::ZeroExtend(bits_ops::UMul(x_bits, y_bits),
                                                 2 * kMaxWidth)
                                .Slice(0, result_width);
            XLS_ASSERT_OK_AND_ASSIGN(Bits actual,
                                     RunWithBits(function, {x_bits, y_bits}));
            EXPECT_EQ(expected, actual)
                << absl::StreamFormat("umul(bits[%d]: %s, bits[%d]: %s)",
                                      x_width, x_bits.ToString(), y_width,
                                      y_bits.ToString())
                << "Expected: " << expected << "Actual: " << actual;
          }
        }
      }
    }
  }
}

TEST_P(IrEvaluatorTestBase, MixedWidthSignedMultiplication) {
  Package package("my_package");
  XLS_ASSERT_OK_AND_ASSIGN(Function * function,
                           ParseAndGetFunction(&package, R"(
  fn mixed_smul(x: bits[7], y: bits[3]) -> bits[6] {
    ret smul.1: bits[6] = smul(x, y)
  }
  )"));

  XLS_ASSERT_OK_AND_ASSIGN(Bits actual,
                           RunWithBits(function, {SBits(0, 7), SBits(0, 3)}));
  EXPECT_EQ(actual, SBits(0, 6));
  XLS_ASSERT_OK_AND_ASSIGN(actual,
                           RunWithBits(function, {SBits(-5, 7), SBits(-2, 3)}));
  EXPECT_EQ(actual, SBits(10, 6));
  XLS_ASSERT_OK_AND_ASSIGN(actual,
                           RunWithBits(function, {SBits(10, 7), SBits(-2, 3)}));
  EXPECT_EQ(actual, SBits(-20, 6));
  XLS_ASSERT_OK_AND_ASSIGN(
      actual, RunWithBits(function, {SBits(-50, 7), SBits(-2, 3)}));
  EXPECT_EQ(actual, SBits(-28, 6));
}

TEST_P(IrEvaluatorTestBase, MixedWidthSignedMultiplicationExhaustive) {
  // Exhaustively check all bit width and value combinations of a mixed-width
  // unsigned multiply up to a small constant bit width.
  constexpr absl::string_view ir_text = R"(
  fn mixed_mul(x: bits[$0], y: bits[$1]) -> bits[$2] {
    ret smul.1: bits[$2] = smul(x, y)
  }
  )";

  const int64_t kMaxWidth = 3;
  for (int x_width = 1; x_width <= kMaxWidth; ++x_width) {
    for (int y_width = 1; y_width <= kMaxWidth; ++y_width) {
      for (int result_width = 1; result_width <= kMaxWidth; ++result_width) {
        std::string formatted_ir =
            absl::Substitute(ir_text, x_width, y_width, result_width);
        Package package("my_package");
        XLS_ASSERT_OK_AND_ASSIGN(Function * function,
                                 ParseAndGetFunction(&package, formatted_ir));
        for (int x = 0; x < 1 << x_width; ++x) {
          for (int y = 0; y < 1 << y_width; ++y) {
            Bits x_bits = UBits(x, x_width);
            Bits y_bits = UBits(y, y_width);
            // The expected result width may be narrower or wider than the
            // result produced by bits_ops::Umul so just sign extend to a wide
            // value then slice it to size.
            Bits expected = bits_ops::SignExtend(bits_ops::SMul(x_bits, y_bits),
                                                 2 * kMaxWidth)
                                .Slice(0, result_width);
            EXPECT_THAT(RunWithBits(function, {x_bits, y_bits}),
                        IsOkAndHolds(expected))
                << absl::StreamFormat("smul(bits[%d]: %s, bits[%d]: %s)",
                                      x_width, x_bits.ToString(), y_width,
                                      y_bits.ToString());
          }
        }
      }
    }
  }
}

TEST_P(IrEvaluatorTestBase, InterpretUDiv) {
  Package package("my_package");
  XLS_ASSERT_OK_AND_ASSIGN(Function * function,
                           ParseAndGetFunction(&package, R"(
  fn shift_left_logical(a: bits[8], b: bits[8]) -> bits[8] {
    ret udiv.1: bits[8] = udiv(a, b)
  }
  )"));

  absl::flat_hash_map<std::string, Value> args;

  args = {{"a", Value(UBits(33, 8))}, {"b", Value(UBits(11, 8))}};
  EXPECT_THAT(GetParam().kwargs_evaluator(function, args),
              IsOkAndHolds(Value(UBits(3, 8))));

  // Should round to zero.
  args = {{"a", Value(UBits(4, 8))}, {"b", Value(UBits(3, 8))}};
  EXPECT_THAT(GetParam().kwargs_evaluator(function, args),
              IsOkAndHolds(Value(UBits(1, 8))));

  // Divide by zero.
  args = {{"a", Value(UBits(123, 8))}, {"b", Value(UBits(0, 8))}};
  EXPECT_THAT(GetParam().kwargs_evaluator(function, args),
              IsOkAndHolds(Value(UBits(0xff, 8))));
}

TEST_P(IrEvaluatorTestBase, InterpretUMod) {
  Package package("my_package");
  XLS_ASSERT_OK_AND_ASSIGN(Function * function,
                           ParseAndGetFunction(&package, R"(
  fn shift_left_logical(a: bits[8], b: bits[8]) -> bits[8] {
    ret umod.1: bits[8] = umod(a, b)
  }
  )"));

  absl::flat_hash_map<std::string, Value> args;

  args = {{"a", Value(UBits(33, 8))}, {"b", Value(UBits(11, 8))}};
  EXPECT_THAT(GetParam().kwargs_evaluator(function, args),
              IsOkAndHolds(Value(UBits(0, 8))));

  args = {{"a", Value(UBits(42, 8))}, {"b", Value(UBits(5, 8))}};
  EXPECT_THAT(GetParam().kwargs_evaluator(function, args),
              IsOkAndHolds(Value(UBits(2, 8))));

  // Modulo by zero.
  args = {{"a", Value(UBits(123, 8))}, {"b", Value(UBits(0, 8))}};
  EXPECT_THAT(GetParam().kwargs_evaluator(function, args),
              IsOkAndHolds(Value(UBits(0, 8))));
}

TEST_P(IrEvaluatorTestBase, InterpretSDiv) {
  Package package("my_package");
  XLS_ASSERT_OK_AND_ASSIGN(Function * function,
                           ParseAndGetFunction(&package, R"(
  fn signed_div(a: bits[8], b: bits[8]) -> bits[8] {
    ret sdiv.1: bits[8] = sdiv(a, b)
  }
  )"));

  absl::flat_hash_map<std::string, Value> args;

  args = {{"a", Value(SBits(33, 8))}, {"b", Value(SBits(-11, 8))}};
  EXPECT_THAT(GetParam().kwargs_evaluator(function, args),
              IsOkAndHolds(Value(SBits(-3, 8))));

  // Should round to zero.
  args = {{"a", Value(SBits(4, 8))}, {"b", Value(SBits(3, 8))}};
  EXPECT_THAT(GetParam().kwargs_evaluator(function, args),
              IsOkAndHolds(Value(SBits(1, 8))));
  args = {{"a", Value(SBits(4, 8))}, {"b", Value(SBits(-3, 8))}};
  EXPECT_THAT(GetParam().kwargs_evaluator(function, args),
              IsOkAndHolds(Value(SBits(-1, 8))));

  // Divide by zero.
  args = {{"a", Value(SBits(123, 8))}, {"b", Value(SBits(0, 8))}};
  EXPECT_THAT(GetParam().kwargs_evaluator(function, args),
              IsOkAndHolds(Value(SBits(127, 8))));
  args = {{"a", Value(SBits(-10, 8))}, {"b", Value(SBits(0, 8))}};
  EXPECT_THAT(GetParam().kwargs_evaluator(function, args),
              IsOkAndHolds(Value(SBits(-128, 8))));
}

TEST_P(IrEvaluatorTestBase, InterpretSMod) {
  Package package("my_package");
  XLS_ASSERT_OK_AND_ASSIGN(Function * function,
                           ParseAndGetFunction(&package, R"(
  fn signed_mod(a: bits[8], b: bits[8]) -> bits[8] {
    ret smod.1: bits[8] = smod(a, b)
  }
  )"));

  absl::flat_hash_map<std::string, Value> args;

  args = {{"a", Value(SBits(33, 8))}, {"b", Value(SBits(-11, 8))}};
  EXPECT_THAT(GetParam().kwargs_evaluator(function, args),
              IsOkAndHolds(Value(SBits(0, 8))));

  args = {{"a", Value(SBits(4, 8))}, {"b", Value(SBits(3, 8))}};
  EXPECT_THAT(GetParam().kwargs_evaluator(function, args),
              IsOkAndHolds(Value(SBits(1, 8))));
  args = {{"a", Value(SBits(4, 8))}, {"b", Value(SBits(-3, 8))}};
  EXPECT_THAT(GetParam().kwargs_evaluator(function, args),
              IsOkAndHolds(Value(SBits(1, 8))));
  args = {{"a", Value(SBits(-4, 8))}, {"b", Value(SBits(3, 8))}};
  EXPECT_THAT(GetParam().kwargs_evaluator(function, args),
              IsOkAndHolds(Value(SBits(-1, 8))));
  args = {{"a", Value(SBits(-4, 8))}, {"b", Value(SBits(-3, 8))}};
  EXPECT_THAT(GetParam().kwargs_evaluator(function, args),
              IsOkAndHolds(Value(SBits(-1, 8))));

  // Modulus by zero.
  args = {{"a", Value(SBits(123, 8))}, {"b", Value(SBits(0, 8))}};
  EXPECT_THAT(GetParam().kwargs_evaluator(function, args),
              IsOkAndHolds(Value(SBits(0, 8))));
  args = {{"a", Value(SBits(-10, 8))}, {"b", Value(SBits(0, 8))}};
  EXPECT_THAT(GetParam().kwargs_evaluator(function, args),
              IsOkAndHolds(Value(SBits(0, 8))));
}

TEST_P(IrEvaluatorTestBase, InterpretShll) {
  Package package("my_package");
  XLS_ASSERT_OK_AND_ASSIGN(Function * function,
                           ParseAndGetFunction(&package, R"(
  fn shift_left_logical(a: bits[8], b: bits[8]) -> bits[8] {
    ret shll.1: bits[8] = shll(a, b)
  }
  )"));

  absl::flat_hash_map<std::string, Value> args;

  args = {{"a", Value(UBits(4, 8))}, {"b", Value(UBits(3, 8))}};
  EXPECT_THAT(GetParam().kwargs_evaluator(function, args),
              IsOkAndHolds(Value(UBits(32, 8))));

  // No shift.
  args = {{"a", Value(UBits(4, 8))}, {"b", Value(UBits(0, 8))}};
  EXPECT_THAT(GetParam().kwargs_evaluator(function, args),
              IsOkAndHolds(Value(UBits(4, 8))));

  // Max shift amount, does not wrap.
  args = {{"a", Value(UBits(0b11111111, 8))}, {"b", Value(UBits(7, 8))}};
  EXPECT_THAT(GetParam().kwargs_evaluator(function, args),
              IsOkAndHolds(Value(UBits(0b10000000, 8))));

  // Overshift.
  args = {{"a", Value(UBits(0b11111111, 8))}, {"b", Value(UBits(8, 8))}};
  EXPECT_THAT(GetParam().kwargs_evaluator(function, args),
              IsOkAndHolds(Value(UBits(0, 8))));
  args = {{"a", Value(UBits(0b11111111, 8))}, {"b", Value(UBits(128, 8))}};
  EXPECT_THAT(GetParam().kwargs_evaluator(function, args),
              IsOkAndHolds(Value(UBits(0, 8))));
}

TEST_P(IrEvaluatorTestBase, InterpretShllNarrowShiftAmount) {
  Package package("my_package");
  XLS_ASSERT_OK_AND_ASSIGN(Function * function,
                           ParseAndGetFunction(&package, R"(
  fn shift_left_logical(a: bits[8], b: bits[3]) -> bits[8] {
    ret shll.1: bits[8] = shll(a, b)
  }
  )"));

  absl::flat_hash_map<std::string, Value> args;

  args = {{"a", Value(UBits(4, 8))}, {"b", Value(UBits(3, 3))}};
  EXPECT_THAT(GetParam().kwargs_evaluator(function, args),
              IsOkAndHolds(Value(UBits(32, 8))));

  // No shift.
  args = {{"a", Value(UBits(4, 8))}, {"b", Value(UBits(0, 3))}};
  EXPECT_THAT(GetParam().kwargs_evaluator(function, args),
              IsOkAndHolds(Value(UBits(4, 8))));

  // Max shift amount, does not wrap.
  args = {{"a", Value(UBits(0b11111111, 8))}, {"b", Value(UBits(7, 3))}};
  EXPECT_THAT(GetParam().kwargs_evaluator(function, args),
              IsOkAndHolds(Value(UBits(0b10000000, 8))));
}

TEST_P(IrEvaluatorTestBase, InterpretShll64) {
  Package package("my_package");
  XLS_ASSERT_OK_AND_ASSIGN(Function * function,
                           ParseAndGetFunction(&package, R"(
  fn shift_left_logical(a: bits[64], b: bits[64]) -> bits[64] {
    ret shll.1: bits[64] = shll(a, b)
  }
  )"));

  absl::flat_hash_map<std::string, Value> args = {{"a", Value(UBits(1, 64))},
                                                  {"b", Value(UBits(63, 64))}};
  XLS_ASSERT_OK_AND_ASSIGN(auto result,
                           GetParam().kwargs_evaluator(function, args));
  EXPECT_EQ(result.bits(), UBits(0x8000000000000000, 64));
}

TEST_P(IrEvaluatorTestBase, InterpretShrl) {
  Package package("my_package");
  XLS_ASSERT_OK_AND_ASSIGN(Function * function,
                           ParseAndGetFunction(&package, R"(
  fn shift_right_logical(a: bits[8], b: bits[8]) -> bits[8] {
    ret shrl.1: bits[8] = shrl(a, b)
  }
  )"));

  absl::flat_hash_map<std::string, Value> args;

  args = {{"a", Value(UBits(0b00000100, 8))}, {"b", Value(UBits(2, 8))}};
  EXPECT_THAT(GetParam().kwargs_evaluator(function, args),
              IsOkAndHolds(Value(UBits(1, 8))));

  // No shift.
  args = {{"a", Value(UBits(0b00000100, 8))}, {"b", Value(UBits(0, 8))}};
  EXPECT_THAT(GetParam().kwargs_evaluator(function, args),
              IsOkAndHolds(Value(UBits(4, 8))));

  // Max shift amount, does not wrap.
  args = {{"a", Value(UBits(0b11111111, 8))}, {"b", Value(UBits(7, 8))}};
  EXPECT_THAT(GetParam().kwargs_evaluator(function, args),
              IsOkAndHolds(Value(UBits(0b00000001, 8))));

  // Overshift.
  args = {{"a", Value(UBits(0b11111111, 8))}, {"b", Value(UBits(8, 8))}};
  EXPECT_THAT(GetParam().kwargs_evaluator(function, args),
              IsOkAndHolds(Value(UBits(0, 8))));
  args = {{"a", Value(UBits(0b11111111, 8))}, {"b", Value(UBits(200, 8))}};
  EXPECT_THAT(GetParam().kwargs_evaluator(function, args),
              IsOkAndHolds(Value(UBits(0, 8))));
}

TEST_P(IrEvaluatorTestBase, InterpretShrlBits64) {
  Package package("my_package");
  XLS_ASSERT_OK_AND_ASSIGN(Function * function,
                           ParseAndGetFunction(&package, R"(
  fn shift_right_logical(a: bits[64], b: bits[64]) -> bits[64] {
    ret shrl.1: bits[64] = shrl(a, b)
  }
  )"));

  // Max shift amount is bit_count() - 1; should be one bit left over.
  absl::flat_hash_map<std::string, Value> args = {{"a", Value(SBits(-1, 64))},
                                                  {"b", Value(UBits(63, 64))}};
  XLS_ASSERT_OK_AND_ASSIGN(auto result,
                           GetParam().kwargs_evaluator(function, args));
  EXPECT_EQ(result.bits(), UBits(1, 64));
}

TEST_P(IrEvaluatorTestBase, InterpretShra) {
  Package package("my_package");
  XLS_ASSERT_OK_AND_ASSIGN(Function * function,
                           ParseAndGetFunction(&package, R"(
  fn shift_right_arithmetic(a: bits[8], b: bits[8]) -> bits[8] {
    ret shra.1: bits[8] = shra(a, b)
  }
  )"));

  absl::flat_hash_map<std::string, Value> args;

  // Zero padding from left.
  args = {{"a", Value(UBits(0b00000100, 8))}, {"b", Value(UBits(2, 8))}};
  EXPECT_THAT(GetParam().kwargs_evaluator(function, args),
              IsOkAndHolds(Value(UBits(1, 8))));

  // Ones padding from left.
  args = {{"a", Value(UBits(0b10000100, 8))}, {"b", Value(UBits(2, 8))}};
  EXPECT_THAT(GetParam().kwargs_evaluator(function, args),
              IsOkAndHolds(Value(UBits(0b11100001, 8))));

  // No shift.
  args = {{"a", Value(UBits(4, 8))}, {"b", Value(UBits(0, 8))}};
  EXPECT_THAT(GetParam().kwargs_evaluator(function, args),
              IsOkAndHolds(Value(UBits(4, 8))));

  // Max shift amount, does not wrap.
  args = {{"a", Value(UBits(0b11111111, 8))}, {"b", Value(UBits(7, 8))}};
  EXPECT_THAT(GetParam().kwargs_evaluator(function, args),
              IsOkAndHolds(Value(UBits(0b11111111, 8))));

  // Overshift.
  args = {{"a", Value(UBits(0b11111111, 8))}, {"b", Value(UBits(100, 8))}};
  EXPECT_THAT(GetParam().kwargs_evaluator(function, args),
              IsOkAndHolds(Value(UBits(0b11111111, 8))));
  args = {{"a", Value(UBits(0b01111111, 8))}, {"b", Value(UBits(100, 8))}};
  EXPECT_THAT(GetParam().kwargs_evaluator(function, args),
              IsOkAndHolds(Value(UBits(0, 8))));
}

TEST_P(IrEvaluatorTestBase, InterpretShraNarrowShiftAmount) {
  Package package("my_package");
  XLS_ASSERT_OK_AND_ASSIGN(Function * function,
                           ParseAndGetFunction(&package, R"(
  fn shift_right_arithmetic(a: bits[8], b: bits[2]) -> bits[8] {
    ret shra.1: bits[8] = shra(a, b)
  }
  )"));

  absl::flat_hash_map<std::string, Value> args;

  // Zero padding from left.
  args = {{"a", Value(UBits(0b00000100, 8))}, {"b", Value(UBits(2, 2))}};
  EXPECT_THAT(GetParam().kwargs_evaluator(function, args),
              IsOkAndHolds(Value(UBits(1, 8))));

  // Ones padding from left.
  args = {{"a", Value(UBits(0b10000100, 8))}, {"b", Value(UBits(2, 2))}};
  EXPECT_THAT(GetParam().kwargs_evaluator(function, args),
              IsOkAndHolds(Value(UBits(0b11100001, 8))));

  // No shift.
  args = {{"a", Value(UBits(4, 8))}, {"b", Value(UBits(0, 2))}};
  EXPECT_THAT(GetParam().kwargs_evaluator(function, args),
              IsOkAndHolds(Value(UBits(4, 8))));

  // Max shift amount, does not wrap.
  args = {{"a", Value(UBits(0b11111111, 8))}, {"b", Value(UBits(3, 2))}};
  EXPECT_THAT(GetParam().kwargs_evaluator(function, args),
              IsOkAndHolds(Value(UBits(0b11111111, 8))));
}

TEST_P(IrEvaluatorTestBase, InterpretShraBits64) {
  Package package("my_package");
  XLS_ASSERT_OK_AND_ASSIGN(Function * function,
                           ParseAndGetFunction(&package, R"(
  fn shift_right_arithmetic(a: bits[64], b: bits[64]) -> bits[64] {
    ret shra.1: bits[64] = shra(a, b)
  }
  )"));

  // Max shift amount is bit_count() - 1; should be all one's after shift.
  absl::flat_hash_map<std::string, Value> args = {{"a", Value(SBits(-1, 64))},
                                                  {"b", Value(UBits(63, 64))}};
  XLS_ASSERT_OK_AND_ASSIGN(auto result,
                           GetParam().kwargs_evaluator(function, args));
  EXPECT_EQ(result.bits(), SBits(-1, 64));

  args = {{"a", Value(UBits(0xF000000000000000ULL, 64))},
          {"b", Value(UBits(4, 64))}};
  XLS_ASSERT_OK_AND_ASSIGN(result, GetParam().kwargs_evaluator(function, args));
  EXPECT_EQ(result.bits(), UBits(0xFF00000000000000ULL, 64));
}

TEST_P(IrEvaluatorTestBase, InterpretFourSubOne) {
  Package package("my_package");
  XLS_ASSERT_OK_AND_ASSIGN(Function * function,
                           ParseAndGetFunction(&package, R"(
  fn sub(a: bits[32], b: bits[32]) -> bits[32] {
    ret sub.3: bits[32] = sub(a, b)
  }
  )"));

  absl::flat_hash_map<std::string, Value> args = {{"a", Value(UBits(4, 32))},
                                                  {"b", Value(UBits(1, 32))}};
  EXPECT_THAT(GetParam().kwargs_evaluator(function, args),
              IsOkAndHolds(Value(UBits(3, 32))));
}

absl::Status RunConcatTest(const IrEvaluatorTestParam& param,
                           const ArgMap& kwargs) {
  constexpr absl::string_view ir_text = R"(
  package test

  fn concatenate(a: bits[$0], b: bits[$1]) -> bits[$2] {
    ret concat.1: bits[$2] = concat(a, b)
  }
  )";

  std::string bytes_str = "0x";
  std::vector<uint8_t> bytes;

  Value a = kwargs.at("a");
  Value b = kwargs.at("b");
  int64_t a_width = a.GetFlatBitCount();
  int64_t b_width = b.GetFlatBitCount();
  std::string formatted_text =
      absl::Substitute(ir_text, a_width, b_width, a_width + b_width);
  XLS_ASSIGN_OR_RETURN(auto package, Parser::ParsePackage(formatted_text));
  XLS_ASSIGN_OR_RETURN(Function * function, package->EntryFunction());

  Value expected(bits_ops::Concat({a.bits(), b.bits()}));
  EXPECT_THAT(param.kwargs_evaluator(function, kwargs), IsOkAndHolds(expected));
  return absl::OkStatus();
}

TEST_P(IrEvaluatorTestBase, InterpretConcat) {
  ArgMap args = {{"a", Value(UBits(1, 8))}, {"b", Value(UBits(4, 8))}};
  XLS_ASSERT_OK(RunConcatTest(GetParam(), args));

  args = {{"a", Value(UBits(4, 8))}, {"b", Value(UBits(1, 8))}};
  XLS_ASSERT_OK(RunConcatTest(GetParam(), args));

  args = {{"a", Value(SBits(-1, 8))}, {"b", Value(SBits(-1, 8))}};
  XLS_ASSERT_OK(RunConcatTest(GetParam(), args));

  args = {{"a", Value(SBits(-1, 128))}, {"b", Value(SBits(-1, 128))}};
  XLS_ASSERT_OK(RunConcatTest(GetParam(), args));

  args = {{"a", Value(SBits(512, 513))}, {"b", Value(SBits(511, 513))}};
  XLS_ASSERT_OK(RunConcatTest(GetParam(), args));

  args = {{"a", Value(SBits(512, 1025))}, {"b", Value(SBits(511, 1025))}};
  XLS_ASSERT_OK(RunConcatTest(GetParam(), args));
}

TEST_P(IrEvaluatorTestBase, InterpretOneHot) {
  Package package("my_package");
  XLS_ASSERT_OK_AND_ASSIGN(Function * function,
                           ParseAndGetFunction(&package, R"(
  fn one_hot(x: bits[3]) -> bits[4] {
    ret one_hot.1: bits[4] = one_hot(x, lsb_prio=true)
  }
  )"));

  struct Example {
    Bits input;
    Bits output;
  };
  std::vector<Example> examples = {
      {UBits(0b000, 3), UBits(0b1000, 4)}, {UBits(0b001, 3), UBits(0b0001, 4)},
      {UBits(0b010, 3), UBits(0b0010, 4)}, {UBits(0b011, 3), UBits(0b0001, 4)},
      {UBits(0b100, 3), UBits(0b0100, 4)}, {UBits(0b101, 3), UBits(0b0001, 4)},
      {UBits(0b110, 3), UBits(0b0010, 4)}, {UBits(0b111, 3), UBits(0b0001, 4)},
  };

  for (const auto& example : examples) {
    XLS_VLOG(2) << "input: "
                << example.input.ToString(FormatPreference::kBinary, true)
                << " expected: "
                << example.output.ToString(FormatPreference::kBinary, true);
    EXPECT_THAT(RunWithValues(function, {Value(example.input)}),
                IsOkAndHolds(Value(example.output)));
  }
}

TEST_P(IrEvaluatorTestBase, InterpretOneHotMsbPrio) {
  Package package("my_package");
  XLS_ASSERT_OK_AND_ASSIGN(Function * function,
                           ParseAndGetFunction(&package, R"(
  fn one_hot(x: bits[3]) -> bits[4] {
    ret one_hot.1: bits[4] = one_hot(x, lsb_prio=false)
  }
  )"));

  struct Example {
    Bits input;
    Bits output;
  };
  std::vector<Example> examples = {
      {UBits(0b000, 3), UBits(0b1000, 4)}, {UBits(0b001, 3), UBits(0b0001, 4)},
      {UBits(0b010, 3), UBits(0b0010, 4)}, {UBits(0b011, 3), UBits(0b0010, 4)},
      {UBits(0b100, 3), UBits(0b0100, 4)}, {UBits(0b101, 3), UBits(0b0100, 4)},
      {UBits(0b110, 3), UBits(0b0100, 4)}, {UBits(0b111, 3), UBits(0b0100, 4)},
  };

  for (const auto& example : examples) {
    XLS_VLOG(2) << "input: "
                << example.input.ToString(FormatPreference::kBinary, true)
                << " expected: "
                << example.output.ToString(FormatPreference::kBinary, true);
    EXPECT_THAT(RunWithValues(function, {Value(example.input)}),
                IsOkAndHolds(Value(example.output)));
  }
}

TEST_P(IrEvaluatorTestBase, InterpretOneBitOneHot) {
  Package package("my_package");
  XLS_ASSERT_OK_AND_ASSIGN(Function * function,
                           ParseAndGetFunction(&package, R"(
  fn one_hot(x: bits[1]) -> bits[2] {
    ret one_hot.1: bits[2] = one_hot(x, lsb_prio=true)
  }
  )"));
  struct Example {
    Bits input;
    Bits output;
  };
  std::vector<Example> examples = {
      {UBits(0, 1), UBits(2, 2)},
      {UBits(1, 1), UBits(1, 2)},
  };
  for (const auto& example : examples) {
    EXPECT_THAT(RunWithValues(function, {Value(example.input)}),
                IsOkAndHolds(Value(example.output)));
  }
}

TEST_P(IrEvaluatorTestBase, InterpretOneHotSelect) {
  Package package("my_package");
  XLS_ASSERT_OK_AND_ASSIGN(Function * function,
                           ParseAndGetFunction(&package, R"(
  fn one_hot(p: bits[3], x: bits[8], y: bits[8], z: bits[8]) -> bits[8] {
    ret one_hot_sel.1: bits[8] = one_hot_sel(p, cases=[x, y, z])
  }
  )"));
  absl::flat_hash_map<std::string, Value> args_map = {
      {"x", Value(UBits(0b00001111, 8))},
      {"y", Value(UBits(0b00110011, 8))},
      {"z", Value(UBits(0b01010101, 8))}};

  struct Example {
    Bits p;
    Bits output;
  };

  std::vector<Example> examples = {
      {UBits(0b000, 3), UBits(0b00000000, 8)},
      {UBits(0b001, 3), UBits(0b00001111, 8)},
      {UBits(0b010, 3), UBits(0b00110011, 8)},
      {UBits(0b011, 3), UBits(0b00111111, 8)},
      {UBits(0b100, 3), UBits(0b01010101, 8)},
      {UBits(0b101, 3), UBits(0b01011111, 8)},
      {UBits(0b110, 3), UBits(0b01110111, 8)},
      {UBits(0b111, 3), UBits(0b01111111, 8)},
  };

  for (const auto& example : examples) {
    std::vector<Value> args = {Value(example.p)};
    args.push_back(args_map["x"]);
    args.push_back(args_map["y"]);
    args.push_back(args_map["z"]);
    EXPECT_THAT(RunWithValues(function, args),
                IsOkAndHolds(Value(example.output)));
  }
}

TEST_P(IrEvaluatorTestBase, InterpretOneHotSelectNestedTuple) {
  Package package("my_package");
  XLS_ASSERT_OK_AND_ASSIGN(Function * function,
                           ParseAndGetFunction(&package, R"(
  fn one_hot(p: bits[2], x: (bits[3], (bits[8])), y: (bits[3], (bits[8]))) -> (bits[3], (bits[8])) {
    ret one_hot_sel.1: (bits[3], (bits[8])) = one_hot_sel(p, cases=[x, y])
  }
  )"));
  absl::flat_hash_map<std::string, Value> args = {
      {"x", AsValue("(bits[3]:0b101, (bits[8]:0b11001100))")},
      {"y", AsValue("(bits[3]:0b001, (bits[8]:0b00001111))")}};

  args["p"] = AsValue("bits[2]: 0b00");
  EXPECT_THAT(GetParam().kwargs_evaluator(function, args),
              IsOkAndHolds(AsValue("(bits[3]:0b000, (bits[8]:0b00000000))")));
  args["p"] = AsValue("bits[2]: 0b01");
  EXPECT_THAT(GetParam().kwargs_evaluator(function, args),
              IsOkAndHolds(AsValue("(bits[3]:0b101, (bits[8]:0b11001100))")));
  args["p"] = AsValue("bits[2]: 0b10");
  EXPECT_THAT(GetParam().kwargs_evaluator(function, args),
              IsOkAndHolds(AsValue("(bits[3]:0b001, (bits[8]:0b00001111))")));
  args["p"] = AsValue("bits[2]: 0b11");
  EXPECT_THAT(GetParam().kwargs_evaluator(function, args),
              IsOkAndHolds(AsValue("(bits[3]:0b101, (bits[8]:0b11001111))")));
}

TEST_P(IrEvaluatorTestBase, InterpretOneHotSelectArray) {
  Package package("my_package");
  XLS_ASSERT_OK_AND_ASSIGN(Function * function,
                           ParseAndGetFunction(&package, R"(
  fn one_hot(p: bits[2], x: bits[4][3], y: bits[4][3]) -> bits[4][3] {
    ret one_hot_sel.1: bits[4][3] = one_hot_sel(p, cases=[x, y])
  }
  )"));
  absl::flat_hash_map<std::string, Value> args = {
      {"x", AsValue("[bits[4]:0b0001, bits[4]:0b0010, bits[4]:0b0100]")},
      {"y", AsValue("[bits[4]:0b1100, bits[4]:0b0101, bits[4]:0b1110]")}};

  args["p"] = AsValue("bits[2]: 0b00");
  EXPECT_THAT(GetParam().kwargs_evaluator(function, args),
              IsOkAndHolds(
                  AsValue("[bits[4]:0b0000, bits[4]:0b0000, bits[4]:0b0000]")));
  args["p"] = AsValue("bits[2]: 0b01");
  EXPECT_THAT(GetParam().kwargs_evaluator(function, args),
              IsOkAndHolds(
                  AsValue("[bits[4]:0b0001, bits[4]:0b0010, bits[4]:0b0100]")));
  args["p"] = AsValue("bits[2]: 0b10");
  EXPECT_THAT(GetParam().kwargs_evaluator(function, args),
              IsOkAndHolds(
                  AsValue("[bits[4]:0b1100, bits[4]:0b0101, bits[4]:0b1110]")));
  args["p"] = AsValue("bits[2]: 0b11");
  EXPECT_THAT(GetParam().kwargs_evaluator(function, args),
              IsOkAndHolds(
                  AsValue("[bits[4]:0b1101, bits[4]:0b0111, bits[4]:0b1110]")));
}

TEST_P(IrEvaluatorTestBase, InterpretBinarySel) {
  Package package("my_package");
  XLS_ASSERT_OK_AND_ASSIGN(Function * function,
                           ParseAndGetFunction(&package, R"(
  fn select(cond: bits[1], if_true: bits[8], if_false: bits[8]) -> bits[8] {
    ret sel.1: bits[8] = sel(cond, cases=[if_false, if_true])
  }
  )"));

  absl::flat_hash_map<std::string, Value> args = {
      {"if_true", Value(UBits(0xA, 8))}, {"if_false", Value(UBits(0xB, 8))}};

  args["cond"] = Value(UBits(1, 1));
  EXPECT_THAT(GetParam().kwargs_evaluator(function, args),
              IsOkAndHolds(Value(UBits(0xA, 8))));

  args["cond"] = Value(UBits(0, 1));
  EXPECT_THAT(GetParam().kwargs_evaluator(function, args),
              IsOkAndHolds(Value(UBits(0xB, 8))));
}

TEST_P(IrEvaluatorTestBase, InterpretBinarySelCompoundType) {
  Package package("my_package");
  XLS_ASSERT_OK_AND_ASSIGN(Function * function,
                           ParseAndGetFunction(&package, R"(
  fn select(cond: bits[1],
            if_true: (bits[1], bits[3])[2],
            if_false: (bits[1], bits[3])[2]) -> (bits[1], bits[3])[2] {
    ret result: (bits[1], bits[3])[2] = sel(cond, cases=[if_false, if_true])
  }
  )"));

  XLS_ASSERT_OK_AND_ASSIGN(
      Value a,
      Value::Array({Value::Tuple({Value(UBits(1, 1)), Value(UBits(2, 3))}),
                    Value::Tuple({Value(UBits(0, 1)), Value(UBits(7, 3))})}));
  XLS_ASSERT_OK_AND_ASSIGN(
      Value b,
      Value::Array({Value::Tuple({Value(UBits(0, 1)), Value(UBits(4, 3))}),
                    Value::Tuple({Value(UBits(1, 1)), Value(UBits(5, 3))})}));
  EXPECT_THAT(RunWithValues(function, {Value(UBits(0, 1)), a, b}),
              IsOkAndHolds(b));
  EXPECT_THAT(RunWithValues(function, {Value(UBits(1, 1)), a, b}),
              IsOkAndHolds(a));
}

TEST_P(IrEvaluatorTestBase, Interpret4WaySel) {
  Package package("my_package");
  XLS_ASSERT_OK_AND_ASSIGN(Function * function,
                           ParseAndGetFunction(&package, R"(
  fn select(p: bits[2], w:bits[8], x: bits[8], y: bits[8], z:bits[8]) -> bits[8] {
    ret sel.1: bits[8] = sel(p, cases=[w, x, y, z])
  }
  )"));

  auto u2 = [](int64_t i) { return Value(UBits(i, 2)); };
  auto u8 = [](int64_t i) { return Value(UBits(i, 8)); };
  absl::flat_hash_map<std::string, Value> args = {
      {"w", u8(0xa)}, {"x", u8(0xb)}, {"y", u8(0xc)}, {"z", u8(0xd)}};
  args["p"] = u2(0);
  EXPECT_THAT(GetParam().kwargs_evaluator(function, args),
              IsOkAndHolds(u8(0xa)));
  args["p"] = u2(1);
  EXPECT_THAT(GetParam().kwargs_evaluator(function, args),
              IsOkAndHolds(u8(0xb)));
  args["p"] = u2(2);
  EXPECT_THAT(GetParam().kwargs_evaluator(function, args),
              IsOkAndHolds(u8(0xc)));
  args["p"] = u2(3);
  EXPECT_THAT(GetParam().kwargs_evaluator(function, args),
              IsOkAndHolds(u8(0xd)));
}

TEST_P(IrEvaluatorTestBase, InterpretManyWaySelWithDefault) {
  Package package("my_package");
  XLS_ASSERT_OK_AND_ASSIGN(Function * function,
                           ParseAndGetFunction(&package, R"(
  fn select(p: bits[1024], w:bits[32], x: bits[32], y: bits[32]) -> bits[32] {
    literal.1: bits[32] = literal(value=0xdefa17)
    ret sel.2: bits[32] = sel(p, cases=[w, x, y], default=literal.1)
  }
  )"));

  auto u1024 = [](int64_t i) { return Value(UBits(i, 1024)); };
  auto u32 = [](int64_t i) { return Value(UBits(i, 32)); };
  absl::flat_hash_map<std::string, Value> args = {
      {"w", u32(0xa)}, {"x", u32(0xb)}, {"y", u32(0xc)}};
  args["p"] = u1024(0);
  EXPECT_THAT(GetParam().kwargs_evaluator(function, args),
              IsOkAndHolds(u32(0xa)));
  args["p"] = u1024(1);
  EXPECT_THAT(GetParam().kwargs_evaluator(function, args),
              IsOkAndHolds(u32(0xb)));
  args["p"] = u1024(2);
  EXPECT_THAT(GetParam().kwargs_evaluator(function, args),
              IsOkAndHolds(u32(0xc)));
  args["p"] = u1024(3);
  EXPECT_THAT(GetParam().kwargs_evaluator(function, args),
              IsOkAndHolds(u32(0xdefa17)));
  XLS_ASSERT_OK_AND_ASSIGN(
      args["p"],
      Parser::ParseTypedValue(
          "bits[1024]:0xabcd_1111_cccc_1234_ffff_eeee_3333_7777_0000_9876"));
  EXPECT_THAT(GetParam().kwargs_evaluator(function, args),
              IsOkAndHolds(u32(0xdefa17)));
}

TEST_P(IrEvaluatorTestBase, InterpretMap) {
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> package,
                           Parser::ParsePackage(R"(
  package SimpleMap

  fn to_apply(element: bits[16]) -> bits[1] {
    literal.2: bits[16] = literal(value=2)
    ret ult.3: bits[1] = ult(element, literal.2)
  }

  fn main(input: bits[16][2]) -> bits[1][2] {
    ret map.5: bits[1][2] = map(input, to_apply=to_apply)
  }
  )"));

  XLS_ASSERT_OK(VerifyPackage(package.get()));
  XLS_ASSERT_OK_AND_ASSIGN(Function * function, package->EntryFunction());
  XLS_ASSERT_OK_AND_ASSIGN(
      auto input_array,
      Value::Array({Value(UBits(1, 16)), Value(UBits(2, 16))}));
  XLS_ASSERT_OK_AND_ASSIGN(Value result,
                           RunWithValues(function, {input_array}));
  EXPECT_EQ(result.elements().at(0), Value(UBits(1, 1)));
  EXPECT_EQ(result.elements().at(1), Value(UBits(0, 1)));
}

TEST_P(IrEvaluatorTestBase, InterpretTwoLevelInvoke) {
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> package,
                           Parser::ParsePackage(R"(
  package SimpleMap

  fn create_tuple(element: bits[16]) -> (bits[1], bits[16]) {
    literal.2: bits[16] = literal(value=2)
    ult.3: bits[1] = ult(element, literal.2)
    ret tuple.4: (bits[1], bits[16]) = tuple(ult.3, element)
  }

  fn nest_tuple(element: (bits[1], bits[16])) -> ((bits[1], bits[1]), bits[16]) {
    tuple_index.5: bits[1] = tuple_index(element, index=0)
    tuple_index.6: bits[16] = tuple_index(element, index=1)
    tuple.7: (bits[1], bits[1]) = tuple(tuple_index.5, tuple_index.5)
    ret tuple.8: ((bits[1], bits[1]), bits[16]) = tuple(tuple.7, tuple_index.6)
  }

  fn to_apply(element: bits[16]) -> ((bits[1], bits[1]), bits[16], bits[1]) {
    invoke.9: (bits[1], bits[16]) = invoke(element, to_apply=create_tuple)
    invoke.10: ((bits[1], bits[1]), bits[16]) = invoke(invoke.9, to_apply=nest_tuple)
    tuple_index.11: (bits[1], bits[1]) = tuple_index(invoke.10, index=0)
    tuple_index.12: bits[16] = tuple_index(invoke.10, index=1)
    tuple_index.13: bits[1] = tuple_index(tuple_index.11, index=0)
    ret tuple.14: ((bits[1], bits[1]), bits[16], bits[1]) = tuple(tuple_index.11, tuple_index.12, tuple_index.13)
  }

  fn main(input: bits[16][2]) -> ((bits[1], bits[1]), bits[16], bits[1])[2] {
    ret map.15: ((bits[1], bits[1]), bits[16], bits[1])[2] = map(input, to_apply=to_apply)
  }
  )"));

  XLS_ASSERT_OK(VerifyPackage(package.get()));
  XLS_ASSERT_OK_AND_ASSIGN(Function * function, package->EntryFunction());
  XLS_ASSERT_OK_AND_ASSIGN(
      auto input_array,
      Value::Array({Value(UBits(1, 16)), Value(UBits(2, 16))}));
  XLS_ASSERT_OK_AND_ASSIGN(Value result,
                           RunWithValues(function, {input_array}));

  Value expected =
      Value::Tuple({Value::Tuple({Value(UBits(1, 1)), Value(UBits(1, 1))}),
                    Value(UBits(1, 16)), Value(UBits(1, 1))});

  EXPECT_EQ(result.elements().at(0), expected);

  expected =
      Value::Tuple({Value::Tuple({Value(UBits(0, 1)), Value(UBits(0, 1))}),
                    Value(UBits(2, 16)), Value(UBits(0, 1))});
  EXPECT_EQ(result.elements().at(1), expected);
}

absl::Status RunReverseTest(const IrEvaluatorTestParam& param,
                            const Bits& bits) {
  constexpr absl::string_view ir_text = R"(
  package test

  fn main(x: bits[$0]) -> bits[$0] {
    ret reverse.1: bits[$0] = reverse(x)
  }
  )";

  std::string formatted_ir = absl::Substitute(ir_text, bits.bit_count());
  XLS_ASSIGN_OR_RETURN(auto package, Parser::ParsePackage(formatted_ir));
  XLS_ASSIGN_OR_RETURN(Function * function, package->EntryFunction());

  Value expected(bits_ops::Reverse(bits));
  EXPECT_THAT(param.evaluator(function, {Value(bits)}), IsOkAndHolds(expected));

  return absl::OkStatus();
}
TEST_P(IrEvaluatorTestBase, InterpretReverse) {
  XLS_ASSERT_OK(RunReverseTest(GetParam(), UBits(0, 1)));
  XLS_ASSERT_OK(RunReverseTest(GetParam(), UBits(1, 1)));
  XLS_ASSERT_OK(RunReverseTest(GetParam(), UBits(0b10000000, 8)));
  XLS_ASSERT_OK(RunReverseTest(GetParam(), UBits(0b11000101, 8)));
  XLS_ASSERT_OK(RunReverseTest(GetParam(), UBits(0b11000101, 9)));
  XLS_ASSERT_OK(RunReverseTest(GetParam(), UBits(0b011000101, 9)));
  XLS_ASSERT_OK(RunReverseTest(GetParam(), UBits(0b011000101, 65)));
  XLS_ASSERT_OK(RunReverseTest(GetParam(), UBits(0b011000101, 129)));
}

TEST_P(IrEvaluatorTestBase, InterpretCountedFor) {
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> package,
                           Parser::ParsePackage(R"(
  package test

  fn body(iv: bits[3], y: bits[11]) -> bits[11] {
    zero_ext.1: bits[11] = zero_ext(iv, new_bit_count=11)
    ret add.3: bits[11] = add(zero_ext.1, y)
  }

  fn main() -> bits[11] {
    literal.4: bits[11] = literal(value=0)
    ret counted_for.5: bits[11] = counted_for(literal.4, trip_count=7, stride=1, body=body)
  }
  )"));
  XLS_ASSERT_OK(VerifyPackage(package.get()));

  // Expected execution behavior:
  //  initial_value = 0, trip_count = 7, stride = 1
  //  iteration 0: body(iv = 0, x =  0) ->  0
  //  iteration 1: body(iv = 1, x =  0) ->  1
  //  iteration 2: body(iv = 2, x =  1) ->  3
  //  iteration 3: body(iv = 3, x =  3) ->  6
  //  iteration 4: body(iv = 4, x =  6) -> 10
  //  iteration 5: body(iv = 5, x = 10) -> 15
  //  iteration 6: body(iv = 6, x = 15) -> 21
  XLS_ASSERT_OK_AND_ASSIGN(Function * function, package->EntryFunction());
  EXPECT_THAT(RunWithValues(function, /*args=*/{}),
              IsOkAndHolds(Value(UBits(21, 11))));
}

TEST_P(IrEvaluatorTestBase, InterpretCountedForStride2) {
  XLS_ASSERT_OK_AND_ASSIGN(auto package, Parser::ParsePackage(R"(
  package test

  fn body(x: bits[11], y: bits[11]) -> bits[11] {
    ret add.3: bits[11] = add(x, y)
  }

  fn main() -> bits[11] {
    literal.4: bits[11] = literal(value=0)
    ret counted_for.5: bits[11] = counted_for(literal.4, trip_count=7, stride=2, body=body)
  }
  )"));

  // TODO(dmlockhart): is this the stride of 2 behavior we want?
  // Expected execution behavior:
  //  initial_value = 0, trip_count = 7, stride = 2
  //  iteration 0: body(x =  0, y =  0) ->  0
  //  iteration 1: body(x =  0, y =  2) ->  2
  //  iteration 2: body(x =  2, y =  4) ->  6
  //  iteration 3: body(x =  6, y =  6) -> 12
  //  iteration 4: body(x = 12, y =  8) -> 20
  //  iteration 5: body(x = 20, y = 10) -> 30
  //  iteration 6: body(x = 30, y = 12) -> 42
  XLS_ASSERT_OK_AND_ASSIGN(Function * function, package->EntryFunction());
  EXPECT_THAT(RunWithValues(function, /*args=*/{}),
              IsOkAndHolds(Value(UBits(42, 11))));
}

TEST_P(IrEvaluatorTestBase, InterpretCountedForStaticArgs) {
  XLS_ASSERT_OK_AND_ASSIGN(auto package, Parser::ParsePackage(R"(
package test

fn body(x: bits[32], y: bits[32], z: bits[32], unused: bits[32]) -> bits[32] {
  add.3: bits[32] = add(x, y)
  ret shll.4: bits[32] = shll(add.3, z)
}

fn main() -> bits[32] {
  literal.0: bits[32] = literal(value=0)
  literal.1: bits[32] = literal(value=1)
  ret counted_for.5: bits[32] = counted_for(literal.0, trip_count=4, body=body, invariant_args=[literal.1, literal.0])
}
)"));
  XLS_ASSERT_OK_AND_ASSIGN(Function * function, package->EntryFunction());
  // The body function executed via counted_for above should be equivalent to
  // the following:
  int64_t expected = 0;
  for (int64_t i = 0; i < 4; ++i) {
    expected += i;
    expected <<= 1;
  }

  // Expected execution behavior:
  //  initial_value = 0, trip_count = 4, stride = 1, invariant_args=[1,0]
  //  iteration 0: body(x =  0, y =  0, z = 1, unused = 0) ->  0
  //  iteration 1: body(x =  0, y =  1, z = 1, unused = 0) ->  2
  //  iteration 2: body(x =  2, y =  2, z = 1, unused = 0) ->  8
  //  iteration 3: body(x =  8, y =  3, z = 1, unused = 0) -> 11
  EXPECT_THAT(RunWithValues(function, /*args=*/{}),
              IsOkAndHolds(Value(UBits(expected, 32))));
}

TEST_P(IrEvaluatorTestBase, InterpretDynamicCountedFor) {
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> package,
                           Parser::ParsePackage(R"(
  package test

  fn body(iv: bits[16], y: bits[16], invar: bits[16]) -> bits[16] {
    sign_ext.1: bits[16] = sign_ext(iv, new_bit_count=16)
    add.3: bits[16] = add(sign_ext.1, y)
    ret add.9: bits[16] = add(add.3, invar)
  }

  fn main() -> bits[16] {
    literal.4: bits[16] = literal(value=0)
    literal.5: bits[8] = literal(value=4)
    literal.6: bits[8] = literal(value=1)
    literal.7: bits[16] = literal(value=1)
    ret dynamic_counted_for.8: bits[16] = dynamic_counted_for(literal.4, literal.5, literal.6, body=body, invariant_args=[literal.7])
  }
  )"));
  XLS_ASSERT_OK(VerifyPackage(package.get()));

  // Expected execution behavior:
  //  initial_value = 0, trip_count = 6, stride = 1
  //  iteration 0: body(iv = 0, x =  0) ->  1
  //  iteration 1: body(iv = 1, x =  1) ->  3
  //  iteration 2: body(iv = 2, x =  3) ->  6
  //  iteration 3: body(iv = 3, x =  6) ->  10
  XLS_ASSERT_OK_AND_ASSIGN(Function * function, package->EntryFunction());
  EXPECT_THAT(RunWithValues(function, /*args=*/{}),
              IsOkAndHolds(Value(SBits(10, 16))));
}

TEST_P(IrEvaluatorTestBase, InterpretDynamicCountedForZeroTrip) {
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> package,
                           Parser::ParsePackage(R"(
  package test

  fn body(iv: bits[16], y: bits[16], invar: bits[16]) -> bits[16] {
    sign_ext.1: bits[16] = sign_ext(iv, new_bit_count=16)
    add.3: bits[16] = add(sign_ext.1, y)
    ret add.9: bits[16] = add(add.3, invar)
  }

  fn main() -> bits[16] {
    literal.4: bits[16] = literal(value=0)
    literal.5: bits[8] = literal(value=0)
    literal.6: bits[8] = literal(value=1)
    literal.7: bits[16] = literal(value=1)
    ret dynamic_counted_for.8: bits[16] = dynamic_counted_for(literal.4, literal.5, literal.6, body=body, invariant_args=[literal.7])
  }
  )"));
  XLS_ASSERT_OK(VerifyPackage(package.get()));

  // Expected execution behavior:
  //  initial_value = 0, trip_count = 0, stride = 1 -> 0
  XLS_ASSERT_OK_AND_ASSIGN(Function * function, package->EntryFunction());
  EXPECT_THAT(RunWithValues(function, /*args=*/{}),
              IsOkAndHolds(Value(SBits(0, 16))));
}

TEST_P(IrEvaluatorTestBase, InterpretDynamicCountedForMultiStride) {
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> package,
                           Parser::ParsePackage(R"(
  package test

  fn body(iv: bits[16], y: bits[16], invar: bits[16]) -> bits[16] {
    sign_ext.1: bits[16] = sign_ext(iv, new_bit_count=16)
    add.3: bits[16] = add(sign_ext.1, y)
    ret add.9: bits[16] = add(add.3, invar)
  }

  fn main() -> bits[16] {
    literal.4: bits[16] = literal(value=0)
    literal.5: bits[8] = literal(value=4)
    literal.6: bits[8] = literal(value=2)
    literal.7: bits[16] = literal(value=1)
    ret dynamic_counted_for.8: bits[16] = dynamic_counted_for(literal.4, literal.5, literal.6, body=body, invariant_args=[literal.7])
  }
  )"));
  XLS_ASSERT_OK(VerifyPackage(package.get()));

  // Expected execution behavior:
  //  initial_value = 0, trip_count = 6, stride = 1
  //  iteration 0: body(iv = 0, x =  0) ->  1
  //  iteration 1: body(iv = 2, x =  1) ->  4
  //  iteration 2: body(iv = 4, x =  4) ->  9
  //  iteration 3: body(iv = 6, x =  9) ->  16
  XLS_ASSERT_OK_AND_ASSIGN(Function * function, package->EntryFunction());
  EXPECT_THAT(RunWithValues(function, /*args=*/{}),
              IsOkAndHolds(Value(SBits(16, 16))));
}

TEST_P(IrEvaluatorTestBase,
       InterpretDynamicCountedForMaxTripAndStrideBitsValues) {
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> package,
                           Parser::ParsePackage(R"(
  package test

  fn body(iv: bits[5], y: bits[16], invar: bits[16]) -> bits[16] {
    sign_ext.1: bits[16] = sign_ext(iv, new_bit_count=16)
    add.3: bits[16] = add(sign_ext.1, y)
    ret add.9: bits[16] = add(add.3, invar)
  }

  fn main() -> bits[16] {
    literal.4: bits[16] = literal(value=0)
    literal.5: bits[2] = literal(value=3)
    literal.6: bits[3] = literal(value=3)
    literal.7: bits[16] = literal(value=1)
    ret dynamic_counted_for.8: bits[16] = dynamic_counted_for(literal.4, literal.5, literal.6, body=body, invariant_args=[literal.7])
  }
  )"));
  XLS_ASSERT_OK(VerifyPackage(package.get()));

  // Expected execution behavior:
  //  initial_value = 0, trip_count = 3, stride = 3
  //  iteration 0: body(iv = 0, x =  0) ->  1
  //  iteration 1: body(iv = 3, x =  1) ->  5
  //  iteration 2: body(iv = 6, x =  5) ->  12
  XLS_ASSERT_OK_AND_ASSIGN(Function * function, package->EntryFunction());
  EXPECT_THAT(RunWithValues(function, /*args=*/{}),
              IsOkAndHolds(Value(SBits(12, 16))));
}

TEST_P(IrEvaluatorTestBase, InterpretDynamicCountedForNegativeStride) {
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> package,
                           Parser::ParsePackage(R"(
  package test

  fn body(iv: bits[16], y: bits[16], invar: bits[16]) -> bits[16] {
    sign_ext.1: bits[16] = sign_ext(iv, new_bit_count=16)
    add.3: bits[16] = add(sign_ext.1, y)
    ret add.9: bits[16] = add(add.3, invar)
  }

  fn main() -> bits[16] {
    literal.4: bits[16] = literal(value=0)
    literal.5: bits[8] = literal(value=4)
    literal.6: bits[8] = literal(value=-2)
    literal.7: bits[16] = literal(value=1)
    ret dynamic_counted_for.8: bits[16] = dynamic_counted_for(literal.4, literal.5, literal.6, body=body, invariant_args=[literal.7])
  }
  )"));
  XLS_ASSERT_OK(VerifyPackage(package.get()));

  // Expected execution behavior:
  //  initial_value = 0, trip_count = 6, stride = 1
  //  iteration 0: body(iv = 0, x =  0) ->  1
  //  iteration 1: body(iv = -2, x =  1) ->  0
  //  iteration 2: body(iv = -4, x =  0) ->  -3
  //  iteration 3: body(iv = -6, x = -3) ->  -8
  XLS_ASSERT_OK_AND_ASSIGN(Function * function, package->EntryFunction());
  EXPECT_THAT(RunWithValues(function, /*args=*/{}),
              IsOkAndHolds(Value(SBits(-8, 16))));
}

TEST_P(IrEvaluatorTestBase, InterpretTuple) {
  Package package("my_package");
  XLS_ASSERT_OK_AND_ASSIGN(Function * function,
                           ParseAndGetFunction(&package, R"(
  fn make_tuple(x: bits[32]) -> (bits[32], bits[64], bits[128]) {
     zero_ext.1: bits[64] = zero_ext(x, new_bit_count=64)
     zero_ext.2: bits[128] = zero_ext(x, new_bit_count=128)
     ret tuple.3: (bits[32], bits[64], bits[128]) = tuple(x, zero_ext.1, zero_ext.2)
  }
  )"));

  absl::flat_hash_map<std::string, Value> args = {{"x", Value(UBits(4, 32))}};

  EXPECT_THAT(GetParam().kwargs_evaluator(function, args),
              IsOkAndHolds(Value::Tuple({
                  Value(UBits(4, 32)),
                  Value(UBits(4, 64)),
                  Value(UBits(4, 128)),
              })));
}

TEST_P(IrEvaluatorTestBase, InterpretTupleLiteral) {
  Package package("my_package");
  XLS_ASSERT_OK_AND_ASSIGN(Function * function,
                           ParseAndGetFunction(&package, R"(
  fn tuple_literal() -> (bits[8], bits[8], bits[8]) {
     literal.1: bits[8] = literal(value=3)
     literal.2: bits[8] = literal(value=2)
     literal.3: bits[8] = literal(value=1)
     ret tuple.4: (bits[8], bits[8], bits[8]) = tuple(literal.1, literal.2,
     literal.3)
  }
  )"));

  EXPECT_THAT(RunWithValues(function, /*args=*/{}), IsOkAndHolds(Value::Tuple({
                                                        Value(UBits(3, 8)),
                                                        Value(UBits(2, 8)),
                                                        Value(UBits(1, 8)),
                                                    })));
}

TEST_P(IrEvaluatorTestBase, InterpretTupleIndexReturnsBits) {
  Package package("my_package");
  XLS_ASSERT_OK_AND_ASSIGN(Function * function,
                           ParseAndGetFunction(&package, R"(
  fn tuple_index(input: (bits[32], bits[33])) -> bits[33] {
    ret tuple_index.4: bits[33] = tuple_index(input, index=1)
  }
  )"));

  EXPECT_THAT(RunWithValues(function, /*args=*/{Value::Tuple(
                                {Value(UBits(11, 32)), Value(UBits(22, 33))})}),
              IsOkAndHolds(Value(UBits(22, 33))));
}

TEST_P(IrEvaluatorTestBase, InterpretTupleIndexReturnsTuple) {
  Package package("my_package");
  XLS_ASSERT_OK_AND_ASSIGN(Function * function,
                           ParseAndGetFunction(&package, R"(
  fn tuple_index() -> (bits[33], bits[34]) {
    literal.1: bits[32] = literal(value=5)
    literal.2: bits[33] = literal(value=123)
    literal.3: bits[34] = literal(value=7)
    tuple.4: (bits[33], bits[34]) = tuple(literal.2, literal.3)
    tuple.5: (bits[32], (bits[33], bits[34])) = tuple(literal.1, tuple.4)
    ret tuple_index.6: (bits[33], bits[34]) = tuple_index(tuple.5, index=1)
  }
  )"));

  EXPECT_THAT(RunWithValues(function, /*args=*/{}), IsOkAndHolds(Value::Tuple({
                                                        Value(UBits(123, 33)),
                                                        Value(UBits(7, 34)),
                                                    })));
}

TEST_P(IrEvaluatorTestBase, InterpretTupleIndexOfLiteralReturnsTuple) {
  Package package("my_package");
  XLS_ASSERT_OK_AND_ASSIGN(Function * function,
                           ParseAndGetFunction(&package, R"(
  fn tuple_index() -> (bits[37], bits[38]) {
    literal.1: (bits[32][3], bits[33], bits[34], (bits[35], bits[36], (bits[37], bits[38]))) = literal(value=([0, 1, 2], 3, 4, (5, 6, (7, 8))))
    tuple_index.2: (bits[35], bits[36], (bits[37], bits[38])) = tuple_index(literal.1, index=3)
    ret tuple_index.3: (bits[37], bits[38]) = tuple_index(tuple_index.2, index=2)
  }
  )"));

  EXPECT_THAT(RunWithValues(function, /*args=*/{}), IsOkAndHolds(Value::Tuple({
                                                        Value(UBits(7, 37)),
                                                        Value(UBits(8, 38)),
                                                    })));
}

TEST_P(IrEvaluatorTestBase, InterpretArrayLiteral) {
  Package package("my_package");
  XLS_ASSERT_OK_AND_ASSIGN(Function * function,
                           ParseAndGetFunction(&package, R"(
  fn array_literal() -> bits[32][3] {
    ret literal.1: bits[32][3] = literal(value=[0, 1, 2])
  }
  )"));

  // Note: Array constructor returns a StatusOr, so we need to get at the value.
  XLS_ASSERT_OK_AND_ASSIGN(Value array_value, Value::Array({
                                                  Value(UBits(0, 32)),
                                                  Value(UBits(1, 32)),
                                                  Value(UBits(2, 32)),
                                              }));
  EXPECT_THAT(RunWithValues(function, /*args=*/{}), IsOkAndHolds(array_value));
}

TEST_P(IrEvaluatorTestBase, Interpret2DArrayLiteral) {
  Package package("my_package");
  XLS_ASSERT_OK_AND_ASSIGN(Function * function,
                           ParseAndGetFunction(&package, R"(
  fn literal() -> bits[17][3][2] {
    ret result: bits[17][3][2] = literal(value=[[11,22,33], [55,44,77]])
  }
  )"));

  IrEvaluatorTestParam param = GetParam();
  EXPECT_THAT(
      RunWithValues(function, {}),
      IsOkAndHolds(
          Value::UBits2DArray({{11, 22, 33}, {55, 44, 77}}, 17).value()));
}

TEST_P(IrEvaluatorTestBase, InterpretArrayIndex) {
  Package package("my_package");
  XLS_ASSERT_OK_AND_ASSIGN(Function * function,
                           ParseAndGetFunction(&package, R"(
  fn array_index() -> bits[32] {
    literal.1: bits[32][2] = literal(value=[5, 123])
    literal.2: bits[3] = literal(value=0)
    literal.3: bits[3] = literal(value=1)
    array_index.4: bits[32] = array_index(literal.1, indices=[literal.2])
    ret array_index.5: bits[32] = array_index(literal.1, indices=[literal.3])
  }
  )"));

  EXPECT_THAT(RunWithValues(function, /*args=*/{}),
              IsOkAndHolds(Value(UBits(123, 32))));
}
TEST_P(IrEvaluatorTestBase, InterpretArraySlice) {
  Package package("my_package");
  XLS_ASSERT_OK_AND_ASSIGN(Function * function,
                           ParseAndGetFunction(&package, R"(
  fn array_slice(start: bits[32]) -> bits[32][4] {
    array: bits[32][8] = literal(value=[5, 6, 7, 8, 9, 10, 11, 12])
    ret result: bits[32][4] = array_slice(array, start, width=4)
  }
  )"));

  {
    XLS_ASSERT_OK_AND_ASSIGN(Value correct_result,
                             Value::UBitsArray({8, 9, 10, 11}, 32));
    EXPECT_THAT(RunWithValues(function, {Value(UBits(3, 32))}),
                IsOkAndHolds(correct_result));
  }

  {
    XLS_ASSERT_OK_AND_ASSIGN(Value correct_result,
                             Value::UBitsArray({11, 12, 12, 12}, 32));
    EXPECT_THAT(RunWithValues(function, {Value(UBits(6, 32))}),
                IsOkAndHolds(correct_result));
  }

  {
    XLS_ASSERT_OK_AND_ASSIGN(Value correct_result,
                             Value::UBitsArray({12, 12, 12, 12}, 32));
    EXPECT_THAT(RunWithValues(function, {Value(UBits(100, 32))}),
                IsOkAndHolds(correct_result));
  }
}

TEST_P(IrEvaluatorTestBase, InterpretArraySliceWideStart) {
  Package package("my_package");
  XLS_ASSERT_OK_AND_ASSIGN(Function * function,
                           ParseAndGetFunction(&package, R"(
  fn array_slice(start: bits[256]) -> bits[32][4] {
    array: bits[32][8] = literal(value=[5, 6, 7, 8, 9, 10, 11, 12])
    ret result: bits[32][4] = array_slice(array, start, width=4)
  }
  )"));

  {
    XLS_ASSERT_OK_AND_ASSIGN(Value correct_result,
                             Value::UBitsArray({5, 6, 7, 8}, 32));
    EXPECT_THAT(RunWithValues(function, {Value(UBits(0, 256))}),
                IsOkAndHolds(correct_result));
  }

  {
    XLS_ASSERT_OK_AND_ASSIGN(Value correct_result,
                             Value::UBitsArray({12, 12, 12, 12}, 32));
    XLS_ASSERT_OK_AND_ASSIGN(
        Value start,
        Parser::ParseTypedValue(/*random*/
                                "bits[256]:0xc910_72a8_1cd9_5fce_db32"));
    EXPECT_THAT(RunWithValues(function, {start}), IsOkAndHolds(correct_result));
  }
}

TEST_P(IrEvaluatorTestBase, InterpretArrayOfArrayIndex) {
  Package package("my_package");
  XLS_ASSERT_OK_AND_ASSIGN(Function * function,
                           ParseAndGetFunction(&package, R"(
  fn array_index() -> bits[32] {
    literal.1: bits[32][2][2] = literal(value=[[1, 2], [3, 4]])
    literal.2: bits[3] = literal(value=1)
    array_index.3: bits[32][2] = array_index(literal.1, indices=[literal.2])
    ret array_index.4: bits[32] = array_index(array_index.3, indices=[literal.2])
  }
  )"));

  EXPECT_THAT(RunWithValues(function, /*args=*/{}),
              IsOkAndHolds(Value(UBits(4, 32))));
}

TEST_P(IrEvaluatorTestBase, InterpretArrayUpdateInBounds) {
  auto make_array = [](absl::Span<const int64_t> values) {
    std::vector<Value> elements;
    for (auto v : values) {
      elements.push_back(Value(UBits(v, 32)));
    }
    absl::StatusOr<Value> array = Value::Array(elements);
    EXPECT_TRUE(array.ok());
    return array.value();
  };

  Package package("my_package");
  XLS_ASSERT_OK_AND_ASSIGN(Function * function,
                           ParseAndGetFunction(&package, R"(
  fn array_update(array: bits[32][3], idx: bits[32], new_value: bits[32]) -> bits[32][3] {
    ret array_update.4: bits[32][3] = array_update(array, new_value, indices=[idx])
  }
  )"));

  // Index 0
  EXPECT_THAT(GetParam().kwargs_evaluator(
                  function,
                  /*args=*/{{"array", make_array({1, 2, 3})},
                            {"idx", Value(UBits(0, 32))},
                            {"new_value", Value(UBits(99, 32))}}),
              IsOkAndHolds(make_array({99, 2, 3})));
  // Index 1
  EXPECT_THAT(GetParam().kwargs_evaluator(
                  function,
                  /*args=*/{{"array", make_array({1, 2, 3})},
                            {"idx", Value(UBits(1, 32))},
                            {"new_value", Value(UBits(99, 32))}}),
              IsOkAndHolds(make_array({1, 99, 3})));
  // Index 2
  EXPECT_THAT(GetParam().kwargs_evaluator(
                  function,
                  /*args=*/{{"array", make_array({1, 2, 3})},
                            {"idx", Value(UBits(2, 32))},
                            {"new_value", Value(UBits(99, 32))}}),
              IsOkAndHolds(make_array({1, 2, 99})));
}

TEST_P(IrEvaluatorTestBase, InterpretArrayUpdateOutOfBounds) {
  Package package("my_package");
  XLS_ASSERT_OK_AND_ASSIGN(Function * function,
                           ParseAndGetFunction(&package, R"(
  fn array_update() -> bits[32][3] {
    literal.1: bits[32][3] = literal(value=[1, 2, 3])
    literal.2: bits[2] = literal(value=3)
    literal.3: bits[32] = literal(value=99)
    ret array_update.4: bits[32][3] = array_update(literal.1, literal.3, indices=[literal.2])
  }
  )"));

  XLS_ASSERT_OK_AND_ASSIGN(Value array_value, Value::Array({
                                                  Value(UBits(1, 32)),
                                                  Value(UBits(2, 32)),
                                                  Value(UBits(3, 32)),
                                              }));
  EXPECT_THAT(RunWithValues(function, /*args=*/{}), IsOkAndHolds(array_value));
}

TEST_P(IrEvaluatorTestBase, InterpretArrayOfArraysUpdate) {
  Package package("my_package");
  XLS_ASSERT_OK_AND_ASSIGN(Function * function,
                           ParseAndGetFunction(&package, R"(
  fn array_update() -> bits[32][2][2] {
    literal.1: bits[32][2][2] = literal(value=[[1, 2], [3, 4]])
    literal.2: bits[2] = literal(value=1)
    literal.3: bits[32][2] = literal(value=[98,99])
    ret array_update.4: bits[32][2][2] = array_update(literal.1, literal.3, indices=[literal.2])
  }
  )"));

  XLS_ASSERT_OK_AND_ASSIGN(Value index_0, Value::Array({
                                              Value(UBits(1, 32)),
                                              Value(UBits(2, 32)),
                                          }));
  XLS_ASSERT_OK_AND_ASSIGN(Value index_1, Value::Array({
                                              Value(UBits(98, 32)),
                                              Value(UBits(99, 32)),
                                          }));
  XLS_ASSERT_OK_AND_ASSIGN(Value array_value, Value::Array({
                                                  index_0,
                                                  index_1,
                                              }));
  EXPECT_THAT(RunWithValues(function, /*args=*/{}), IsOkAndHolds(array_value));
}

TEST_P(IrEvaluatorTestBase, InterpretArrayOfTuplesUpdate) {
  Package package("my_package");
  XLS_ASSERT_OK_AND_ASSIGN(Function * function,
                           ParseAndGetFunction(&package, R"(
  fn array_update() -> (bits[2], bits[32])[2] {
    literal.1: (bits[2], bits[32])[2] = literal(value=[(1, 20), (3, 40)])
    literal.2: bits[2] = literal(value=1)
    literal.3: (bits[2], bits[32]) = literal(value=(0,99))
    ret array_update.4: (bits[2], bits[32])[2] = array_update(literal.1, literal.3, indices=[literal.2])
  }
  )"));

  Value index_0 = Value::Tuple({
      Value(UBits(1, 2)),
      Value(UBits(20, 32)),
  });
  Value index_1 = Value::Tuple({
      Value(UBits(0, 2)),
      Value(UBits(99, 32)),
  });
  XLS_ASSERT_OK_AND_ASSIGN(Value array_value, Value::Array({
                                                  index_0,
                                                  index_1,
                                              }));
  EXPECT_THAT(RunWithValues(function, /*args=*/{}), IsOkAndHolds(array_value));
}

TEST_P(IrEvaluatorTestBase, InterpretArrayUpdateOriginalNotMutated) {
  Package package("my_package");
  XLS_ASSERT_OK_AND_ASSIGN(Function * function,
                           ParseAndGetFunction(&package, R"(
  fn array_update() -> (bits[32][3], bits[32][3]) {
    literal.1: bits[32][3] = literal(value=[1, 2, 3])
    literal.2: bits[2] = literal(value=0)
    literal.3: bits[32] = literal(value=99)
    array_update.4: bits[32][3] = array_update(literal.1, literal.3, indices=[literal.2])
    ret tuple.5: (bits[32][3], bits[32][3]) =  tuple(literal.1, array_update.4)
  }
  )"));

  XLS_ASSERT_OK_AND_ASSIGN(Value array_value_original, Value::Array({
                                                           Value(UBits(1, 32)),
                                                           Value(UBits(2, 32)),
                                                           Value(UBits(3, 32)),
                                                       }));
  XLS_ASSERT_OK_AND_ASSIGN(Value array_value_updated, Value::Array({
                                                          Value(UBits(99, 32)),
                                                          Value(UBits(2, 32)),
                                                          Value(UBits(3, 32)),
                                                      }));
  EXPECT_THAT(
      RunWithValues(function, /*args=*/{}),
      IsOkAndHolds(Value::Tuple({array_value_original, array_value_updated})));
}

TEST_P(IrEvaluatorTestBase, InterpretArrayUpdateIndex) {
  Package package("my_package");
  XLS_ASSERT_OK_AND_ASSIGN(Function * function,
                           ParseAndGetFunction(&package, R"(
  fn array_update() -> bits[32] {
    literal.1: bits[32][3] = literal(value=[1, 2, 3])
    literal.2: bits[2] = literal(value=0)
    literal.3: bits[32] = literal(value=99)
    array_update.4: bits[32][3] = array_update(literal.1, literal.3, indices=[literal.2])
    ret array_index.5: bits[32] = array_index(array_update.4, indices=[literal.2])
  }
  )"));

  EXPECT_THAT(RunWithValues(function, /*args=*/{}),
              IsOkAndHolds(Value(UBits(99, 32))));
}

TEST_P(IrEvaluatorTestBase, InterpretArrayUpdateUpdate) {
  Package package("my_package");
  XLS_ASSERT_OK_AND_ASSIGN(Function * function,
                           ParseAndGetFunction(&package, R"(
  fn array_update() -> bits[32][3] {
    literal.1: bits[32][3] = literal(value=[1, 2, 3])
    literal.2: bits[2] = literal(value=0)
    literal.3: bits[32] = literal(value=99)
    array_update.4: bits[32][3] = array_update(literal.1, literal.3, indices=[literal.2])
    literal.5: bits[2] = literal(value=2)
    ret array_update.6: bits[32][3] = array_update(array_update.4, literal.3, indices=[literal.5])
  }
  )"));

  XLS_ASSERT_OK_AND_ASSIGN(Value array_value, Value::Array({
                                                  Value(UBits(99, 32)),
                                                  Value(UBits(2, 32)),
                                                  Value(UBits(99, 32)),
                                              }));
  EXPECT_THAT(RunWithValues(function, /*args=*/{}), IsOkAndHolds(array_value));
}

TEST_P(IrEvaluatorTestBase, InterpretArrayUpdateWideElements) {
  Package package("my_package");
  XLS_ASSERT_OK_AND_ASSIGN(Function * function,
                           ParseAndGetFunction(&package, R"(
  fn array_update() -> bits[1000][3] {
    literal.1: bits[1000][3] = literal(value=[1, 2, 3])
    literal.2: bits[2] = literal(value=0)
    literal.3: bits[1000] = literal(value=99)
    ret array_update.4: bits[1000][3] = array_update(literal.1, literal.3, indices=[literal.2])
  }
  )"));

  XLS_ASSERT_OK_AND_ASSIGN(Value array_value, Value::Array({
                                                  Value(UBits(99, 1000)),
                                                  Value(UBits(2, 1000)),
                                                  Value(UBits(3, 1000)),
                                              }));
  EXPECT_THAT(RunWithValues(function, /*args=*/{}), IsOkAndHolds(array_value));
}

TEST_P(IrEvaluatorTestBase, InterpretArrayUpdateWideIndexInbounds) {
  Package package("my_package");
  XLS_ASSERT_OK_AND_ASSIGN(Function * function,
                           ParseAndGetFunction(&package, R"(
  fn array_update(index: bits[1000]) -> bits[32][3] {
    literal.1: bits[32][3] = literal(value=[1, 2, 3])
    literal.2: bits[32] = literal(value=99)
    ret array_update.3: bits[32][3] = array_update(literal.1, literal.2, indices=[index])
  }
  )"));

  XLS_ASSERT_OK_AND_ASSIGN(Value array_value, Value::Array({
                                                  Value(UBits(1, 32)),
                                                  Value(UBits(99, 32)),
                                                  Value(UBits(3, 32)),
                                              }));
  Value index = Value(UBits(1, 1000));
  EXPECT_THAT(GetParam().kwargs_evaluator(function,
                                          /*args=*/{{"index", index}}),
              IsOkAndHolds(array_value));
}

TEST_P(IrEvaluatorTestBase, InterpretArrayUpdateWideIndexOutOfBounds) {
  Package package("my_package");
  XLS_ASSERT_OK_AND_ASSIGN(Function * function,
                           ParseAndGetFunction(&package, R"(
  fn array_update(index: bits[1000]) -> bits[32][3] {
    literal.1: bits[32][3] = literal(value=[1, 2, 3])
    literal.2: bits[32] = literal(value=99)
    ret array_update.3: bits[32][3] = array_update(literal.1, literal.2, indices=[index])
  }
  )"));

  XLS_ASSERT_OK_AND_ASSIGN(Value array_value, Value::Array({
                                                  Value(UBits(1, 32)),
                                                  Value(UBits(2, 32)),
                                                  Value(UBits(3, 32)),
                                              }));
  Value index = Value(Bits::PowerOfTwo(900, 1000));
  EXPECT_THAT(GetParam().kwargs_evaluator(function,
                                          /*args=*/{{"index", index}}),
              IsOkAndHolds(array_value));
}

TEST_P(IrEvaluatorTestBase, InterpretArrayConcatArraysOfBits) {
  Package package("my_package");
  XLS_ASSERT_OK_AND_ASSIGN(Function * function,
                           ParseAndGetFunction(&package, R"(
  fn f(a0: bits[32][2], a1: bits[32][3]) -> bits[32][7] {
    array_concat.3: bits[32][5] = array_concat(a0, a1)
    ret array_concat.4: bits[32][7] = array_concat(array_concat.3, a0)
  }
  )"));

  XLS_ASSERT_OK_AND_ASSIGN(Value a0, Value::UBitsArray({1, 2}, 32));
  XLS_ASSERT_OK_AND_ASSIGN(Value a1, Value::UBitsArray({3, 4, 5}, 32));
  XLS_ASSERT_OK_AND_ASSIGN(Value ret,
                           Value::UBitsArray({1, 2, 3, 4, 5, 1, 2}, 32));

  EXPECT_THAT(GetParam().kwargs_evaluator(function,
                                          /*args=*/{{"a0", a0}, {"a1", a1}}),
              IsOkAndHolds(ret));
}

TEST_P(IrEvaluatorTestBase, InterpretArrayConcatArraysOfBitsMixedOperands) {
  Package package("my_package");
  XLS_ASSERT_OK_AND_ASSIGN(Function * function,
                           ParseAndGetFunction(&package, R"(
  fn f(a0: bits[32][2], a1: bits[32][3], a2: bits[32][1]) -> bits[32][7] {
    array_concat.4: bits[32][1] = array_concat(a2)
    array_concat.5: bits[32][2] = array_concat(array_concat.4, array_concat.4)
    array_concat.6: bits[32][7] = array_concat(a0, array_concat.5, a1)
    ret array_concat.7: bits[32][7] = array_concat(array_concat.6)
  }
  )"));

  XLS_ASSERT_OK_AND_ASSIGN(Value a0, Value::UBitsArray({1, 2}, 32));
  XLS_ASSERT_OK_AND_ASSIGN(Value a1, Value::UBitsArray({3, 4, 5}, 32));
  XLS_ASSERT_OK_AND_ASSIGN(Value a2, Value::SBitsArray({-1}, 32));
  XLS_ASSERT_OK_AND_ASSIGN(Value ret,
                           Value::SBitsArray({1, 2, -1, -1, 3, 4, 5}, 32));

  EXPECT_THAT(GetParam().kwargs_evaluator(
                  function,
                  /*args=*/{{"a0", a0}, {"a1", a1}, {"a2", a2}}),
              IsOkAndHolds(ret));
}

TEST_P(IrEvaluatorTestBase, InterpretArrayConcatArraysOfArrays) {
  Package package("my_package");
  XLS_ASSERT_OK_AND_ASSIGN(Function * function,
                           ParseAndGetFunction(&package, R"(
  fn f() -> bits[32][2][3] {
    literal.1: bits[32][2][2] = literal(value=[[1, 2], [3, 4]])
    literal.2: bits[32][2][1] = literal(value=[[5, 6]])

    ret array_concat.3: bits[32][2][3] = array_concat(literal.2, literal.1)
  }
  )"));

  XLS_ASSERT_OK_AND_ASSIGN(Value ret,
                           Value::UBits2DArray({{5, 6}, {1, 2}, {3, 4}}, 32));

  EXPECT_THAT(RunWithValues(function, /*args=*/{}), IsOkAndHolds(ret));
}

TEST_P(IrEvaluatorTestBase, InterpretInvokeZeroArgs) {
  XLS_ASSERT_OK_AND_ASSIGN(auto package, Parser::ParsePackage(R"(
package test

fn val() -> bits[32] {
  ret literal.1: bits[32] = literal(value=42)
}

fn main() -> bits[32] {
  ret invoke.5: bits[32] = invoke(to_apply=val)
}
)"));
  XLS_ASSERT_OK_AND_ASSIGN(Function * function, package->EntryFunction());
  EXPECT_THAT(RunWithValues(function, /*args=*/{}),
              IsOkAndHolds(Value(UBits(42, 32))));
}

TEST_P(IrEvaluatorTestBase, InterpretInvokeMultipleArgs) {
  XLS_ASSERT_OK_AND_ASSIGN(auto package, Parser::ParsePackage(R"(
package test

fn add_wrapper(x: bits[32], y: bits[32]) -> bits[32] {
  ret add.4: bits[32] = add(x, y)
}

fn main() -> bits[32] {
  literal.0: bits[32] = literal(value=2)
  literal.1: bits[32] = literal(value=3)
  ret invoke.5: bits[32] = invoke(literal.0, literal.1, to_apply=add_wrapper)
}
)"));
  XLS_ASSERT_OK_AND_ASSIGN(Function * function, package->EntryFunction());
  EXPECT_THAT(RunWithValues(function, /*args=*/{}),
              IsOkAndHolds(Value(UBits(5, 32))));
}

TEST_P(IrEvaluatorTestBase, InterpretInvokeMultipleArgsDepth2) {
  XLS_ASSERT_OK_AND_ASSIGN(auto package, Parser::ParsePackage(R"(
package test

fn add_wrapper(x: bits[32], y: bits[32]) -> bits[32] {
  ret add.4: bits[32] = add(x, y)
}

fn middleman(x: bits[32], y: bits[32]) -> bits[32] {
  ret invoke.2: bits[32] = invoke(x, y, to_apply=add_wrapper)
}

fn main() -> bits[32] {
  literal.0: bits[32] = literal(value=2)
  literal.1: bits[32] = literal(value=3)
  ret invoke.5: bits[32] = invoke(literal.0, literal.1, to_apply=middleman)
}
)"));
  XLS_ASSERT_OK_AND_ASSIGN(Function * function, package->EntryFunction());
  EXPECT_THAT(RunWithValues(function, /*args=*/{}),
              IsOkAndHolds(Value(UBits(5, 32))));
}

TEST_P(IrEvaluatorTestBase, WideAdd) {
  XLS_ASSERT_OK_AND_ASSIGN(auto package, Parser::ParsePackage(R"(
  package wide_add

  fn main(a: bits[128], b: bits[128]) -> bits[128] {
    ret add.1: bits[128] = add(a, b)
  }
  )"));
  absl::flat_hash_map<std::string, Value> args = {
      {"a", Value(UBits(42, 128))}, {"b", Value(UBits(123, 128))}};

  XLS_ASSERT_OK_AND_ASSIGN(Function * function, package->EntryFunction());
  EXPECT_THAT(GetParam().kwargs_evaluator(function, args),
              IsOkAndHolds(Value(UBits(165, 128))));
}

TEST_P(IrEvaluatorTestBase, WideNegate) {
  XLS_ASSERT_OK_AND_ASSIGN(auto package, Parser::ParsePackage(R"(
  package wide_negate

  fn main(a: bits[128]) -> bits[128] {
    ret neg.1: bits[128] = neg(a)
  }
  )"));
  absl::flat_hash_map<std::string, Value> args = {
      {"a", Value(UBits(0x42, 128))}};
  XLS_ASSERT_OK_AND_ASSIGN(Function * function, package->EntryFunction());
  XLS_ASSERT_OK_AND_ASSIGN(Value result,
                           GetParam().kwargs_evaluator(function, args));
  EXPECT_EQ(result.bits().ToString(FormatPreference::kHex),
            "0xffff_ffff_ffff_ffff_ffff_ffff_ffff_ffbe");
}

TEST_P(IrEvaluatorTestBase, WideLogicOperator) {
  XLS_ASSERT_OK_AND_ASSIGN(auto package, Parser::ParsePackage(R"(
  package wide_add

  fn main(a: bits[128], b: bits[128]) -> bits[128] {
    xor.1: bits[128] = xor(a, b)
    ret not.2: bits[128] = not(xor.1)
  }
  )"));
  // Build the 128-bit arguments by concating together 64-bit values.  The
  // 64-bit half-arguments are palindromes, so the result should be similarly
  // structured.
  absl::flat_hash_map<std::string, Value> args = {
      {"a", Value(bits_ops::Concat({UBits(0xdeadbeeffeebdaedULL, 64),
                                    UBits(0x1234567887654321ULL, 64)}))},
      {"b", Value(bits_ops::Concat({UBits(0xf0f0f0f00f0f0f0fULL, 64),
                                    UBits(0x5a5a5a5aa5a5a5a5ULL, 64)}))}};

  XLS_ASSERT_OK_AND_ASSIGN(Function * function, package->EntryFunction());
  XLS_ASSERT_OK_AND_ASSIGN(Value result,
                           GetParam().kwargs_evaluator(function, args));
  EXPECT_EQ(result.bits().ToString(FormatPreference::kHex),
            "0xd1a2_b1e0_0e1b_2a1d_b791_f3dd_dd3f_197b");
}

TEST_P(IrEvaluatorTestBase, OptimizedParamReturn) {
  XLS_ASSERT_OK_AND_ASSIGN(auto package, Parser::ParsePackage(R"(
  package test

  fn main(x0: bits[1], x1: bits[2]) -> bits[1] {
    ret param.1: bits[1] = param(name=x0)
  }
  )"));
  absl::flat_hash_map<std::string, Value> args = {
      {"x0", Value(UBits(1, 1))},
      {"x1", Value(UBits(0, 2))},
  };

  XLS_ASSERT_OK_AND_ASSIGN(Function * function, package->EntryFunction());
  XLS_ASSERT_OK_AND_ASSIGN(Value result,
                           GetParam().kwargs_evaluator(function, args));
  EXPECT_EQ(result.bits().ToString(FormatPreference::kBinary), "0b1");
}

TEST_P(IrEvaluatorTestBase, AfterAllWithOtherOps) {
  XLS_ASSERT_OK_AND_ASSIGN(auto package, Parser::ParsePackage(R"(
  package test

  fn main() -> bits[8] {
    after_all.1: token = after_all()
    literal.2: bits[8] = literal(value=6)
    after_all.3: token = after_all()
    literal.4: bits[8] = literal(value=3)
    after_all.5: token = after_all()
    literal.6: bits[8] = literal(value=2)
    after_all.7: token = after_all()
    ret nand.8: bits[8] = nand(literal.2, literal.4, literal.6)
  }
  )"));
  XLS_ASSERT_OK_AND_ASSIGN(Function * function, package->EntryFunction());
  EXPECT_THAT(RunWithValues(function, /*args=*/{}),
              IsOkAndHolds(Value(UBits(0b11111101, 8))));
}

TEST_P(IrEvaluatorTestBase, AfterAllReturnToken) {
  XLS_ASSERT_OK_AND_ASSIGN(auto package, Parser::ParsePackage(R"(
  package test

  fn main() -> token {
    ret after_all.1: token = after_all()
  }
  )"));
  XLS_ASSERT_OK_AND_ASSIGN(Function * function, package->EntryFunction());
  EXPECT_THAT(RunWithValues(function, /*args=*/{}),
              IsOkAndHolds(Value::Token()));
}

TEST_P(IrEvaluatorTestBase, AfterAllTokenArgs) {
  XLS_ASSERT_OK_AND_ASSIGN(auto package, Parser::ParsePackage(R"(
  package test

  fn main(t1: token, t2: token) -> token {
    ret after_all.1: token = after_all(t1, t2)
  }
  )"));
  XLS_ASSERT_OK_AND_ASSIGN(Function * function, package->EntryFunction());
  EXPECT_THAT(
      GetParam().kwargs_evaluator(
          function, /*args=*/{{"t1", Value::Token()}, {"t2", Value::Token()}}),
      IsOkAndHolds(Value::Token()));
}

TEST_P(IrEvaluatorTestBase, ArrayOperation) {
  XLS_ASSERT_OK_AND_ASSIGN(auto package, Parser::ParsePackage(R"(
  package test

  fn main(x0: bits[16], x1: bits[16]) -> bits[16][2] {
    ret array.1: bits[16][2] = array(x0, x1)
  }
  )"));
  XLS_ASSERT_OK_AND_ASSIGN(Function * function, package->EntryFunction());
  XLS_ASSERT_OK_AND_ASSIGN(
      Value result,
      GetParam().kwargs_evaluator(function, {{"x0", Value(UBits(34, 16))},
                                             {"x1", Value(UBits(56, 16))}}));
  EXPECT_EQ(result,
            Value::ArrayOrDie({Value(UBits(34, 16)), Value(UBits(56, 16))}));
}

TEST_P(IrEvaluatorTestBase, Decode) {
  XLS_ASSERT_OK_AND_ASSIGN(auto package, Parser::ParsePackage(R"(
  package test

  fn main(x: bits[3]) -> bits[8] {
    ret decode.1: bits[8] = decode(x, width=8)
  }
  )"));
  XLS_ASSERT_OK_AND_ASSIGN(Function * function, package->EntryFunction());
  EXPECT_THAT(RunWithUint64s(function, {0}), IsOkAndHolds(1));
  EXPECT_THAT(RunWithUint64s(function, {1}), IsOkAndHolds(2));
  EXPECT_THAT(RunWithUint64s(function, {2}), IsOkAndHolds(4));
  EXPECT_THAT(RunWithUint64s(function, {3}), IsOkAndHolds(8));
  EXPECT_THAT(RunWithUint64s(function, {4}), IsOkAndHolds(16));
  EXPECT_THAT(RunWithUint64s(function, {5}), IsOkAndHolds(32));
  EXPECT_THAT(RunWithUint64s(function, {6}), IsOkAndHolds(64));
  EXPECT_THAT(RunWithUint64s(function, {7}), IsOkAndHolds(128));
}

TEST_P(IrEvaluatorTestBase, NarrowedDecode) {
  XLS_ASSERT_OK_AND_ASSIGN(auto package, Parser::ParsePackage(R"(
  package test

  fn main(x: bits[3]) -> bits[5] {
    ret decode.1: bits[5] = decode(x, width=5)
  }
  )"));
  XLS_ASSERT_OK_AND_ASSIGN(Function * function, package->EntryFunction());
  EXPECT_THAT(RunWithUint64s(function, {0}), IsOkAndHolds(1));
  EXPECT_THAT(RunWithUint64s(function, {1}), IsOkAndHolds(2));
  EXPECT_THAT(RunWithUint64s(function, {2}), IsOkAndHolds(4));
  EXPECT_THAT(RunWithUint64s(function, {3}), IsOkAndHolds(8));
  EXPECT_THAT(RunWithUint64s(function, {4}), IsOkAndHolds(16));
  EXPECT_THAT(RunWithUint64s(function, {5}), IsOkAndHolds(0));
  EXPECT_THAT(RunWithUint64s(function, {6}), IsOkAndHolds(0));
  EXPECT_THAT(RunWithUint64s(function, {7}), IsOkAndHolds(0));
}

TEST_P(IrEvaluatorTestBase, Encode) {
  XLS_ASSERT_OK_AND_ASSIGN(auto package, Parser::ParsePackage(R"(
  package test

  fn main(x: bits[5]) -> bits[3] {
    ret encode.1: bits[3] = encode(x)
  }
  )"));
  XLS_ASSERT_OK_AND_ASSIGN(Function * function, package->EntryFunction());
  EXPECT_THAT(RunWithUint64s(function, {0}), IsOkAndHolds(0));

  // Explicitly test all the one-hot values.
  EXPECT_THAT(RunWithUint64s(function, {1}), IsOkAndHolds(0));
  EXPECT_THAT(RunWithUint64s(function, {2}), IsOkAndHolds(1));
  EXPECT_THAT(RunWithUint64s(function, {4}), IsOkAndHolds(2));
  EXPECT_THAT(RunWithUint64s(function, {8}), IsOkAndHolds(3));
  EXPECT_THAT(RunWithUint64s(function, {16}), IsOkAndHolds(4));

  // Test a few random non-one-hot values.
  EXPECT_THAT(RunWithUint64s(function, {3}), IsOkAndHolds(1));
  EXPECT_THAT(RunWithUint64s(function, {18}), IsOkAndHolds(5));

  // Test all values in a loop.
  for (uint64_t i = 0; i < 31; ++i) {
    uint64_t expected = 0;
    for (int64_t j = 0; j < 5; ++j) {
      if (i & (1 << j)) {
        expected |= j;
      }
    }
    EXPECT_THAT(RunWithUint64s(function, {i}), IsOkAndHolds(expected));
  }
}

TEST_P(IrEvaluatorTestBase, RunMismatchedType) {
  XLS_ASSERT_OK_AND_ASSIGN(auto package, Parser::ParsePackage(R"(
  package test

  fn main(x: bits[16]) -> bits[16] {
    ret param.2: bits[16] = param(name=x)
  }
  )"));
  XLS_ASSERT_OK_AND_ASSIGN(Function * function, package->EntryFunction());
  EXPECT_THAT(
      RunWithValues(function, {Value(UBits(42, 17))}),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("Got argument bits[17]:42 for parameter 0 which is "
                         "not of type bits[16]")));
}

absl::Status RunBitSliceTest(const IrEvaluatorTestParam& param,
                             int64_t literal_width, int64_t slice_start,
                             int64_t slice_width) {
  constexpr absl::string_view ir_text = R"(
  package test

  fn main() -> bits[$0] {
    literal.1: bits[$1] = literal(value=$2)
    ret bit_slice.2: bits[$0] = bit_slice(literal.1, start=$3, width=$0)
  }
  )";

  std::string bytes_str = "0x";
  std::vector<uint8_t> bytes;
  for (int i = 0;
       i < CeilOfRatio(literal_width, static_cast<int64_t>(CHAR_BIT)); i++) {
    absl::StrAppend(&bytes_str, absl::Hex(i % 256, absl::kZeroPad2));
    bytes.push_back(i % 256);
  }

  std::string formatted_ir = absl::Substitute(
      ir_text, slice_width, literal_width, bytes_str, slice_start);
  XLS_ASSIGN_OR_RETURN(auto package, Parser::ParsePackage(formatted_ir));
  XLS_ASSIGN_OR_RETURN(Function * function, package->EntryFunction());

  Value expected(
      Bits::FromBytes(bytes, literal_width).Slice(slice_start, slice_width));
  EXPECT_THAT(param.evaluator(function, {}), IsOkAndHolds(expected));

  return absl::OkStatus();
}

TEST_P(IrEvaluatorTestBase, BitSlice) {
  XLS_ASSERT_OK(RunBitSliceTest(GetParam(), 27, 9, 3));
  XLS_ASSERT_OK(RunBitSliceTest(GetParam(), 32, 9, 3));
  XLS_ASSERT_OK(RunBitSliceTest(GetParam(), 64, 15, 27));
  XLS_ASSERT_OK(RunBitSliceTest(GetParam(), 128, 24, 50));
  XLS_ASSERT_OK(RunBitSliceTest(GetParam(), 1024, 747, 32));
  XLS_ASSERT_OK(RunBitSliceTest(GetParam(), 65536, 8192, 32768));
}

absl::Status RunDynamicBitSliceTest(const IrEvaluatorTestParam& param,
                                    int64_t literal_width, int64_t slice_start,
                                    int64_t start_width, int64_t slice_width) {
  constexpr absl::string_view ir_text = R"(
  package test

  fn main() -> bits[$0] {
    literal.1: bits[$1] = literal(value=$2)
    literal.2: bits[$3] = literal(value=$4)
    ret dynamic_bit_slice.3: bits[$0] = dynamic_bit_slice(literal.1,
                                                          literal.2, width=$0)
  }
  )";

  std::string bytes_str = "0x";
  std::string start_bytes_str = "0x";
  std::vector<uint8_t> bytes;

  for (int i = 0;
       i < CeilOfRatio(literal_width, static_cast<int64_t>(CHAR_BIT)); i++) {
    absl::StrAppend(&bytes_str, absl::Hex(i % 256, absl::kZeroPad2));
    bytes.push_back(i % 256);
  }

  absl::StrAppend(&start_bytes_str, absl::Hex(slice_start, absl::kZeroPad2));
  std::string formatted_ir =
      absl::Substitute(ir_text, slice_width, literal_width, bytes_str,
                       start_width, start_bytes_str);
  XLS_ASSIGN_OR_RETURN(auto package, Parser::ParsePackage(formatted_ir));
  XLS_ASSIGN_OR_RETURN(Function * function, package->EntryFunction());

  Value expected;
  if (slice_start > literal_width) {
    expected = Value(Bits(slice_width));
  } else {
    Bits operand = Bits::FromBytes(bytes, literal_width);
    Bits shifted = bits_ops::ShiftRightLogical(operand, slice_start);
    Bits truncated = shifted.Slice(0, slice_width);
    expected = Value(truncated);
  }
  EXPECT_THAT(param.evaluator(function, {}), IsOkAndHolds(expected));

  return absl::OkStatus();
}

absl::Status RunDynamicBitSliceTestLargeStart(const IrEvaluatorTestParam& param,
                                              int64_t literal_width,
                                              int64_t slice_width) {
  constexpr absl::string_view ir_text = R"(
  package test

  fn main() -> bits[$0] {
    literal.1: bits[$1] = literal(value=$2)
    literal.2: bits[$3] = literal(value=$4)
    ret dynamic_bit_slice.3: bits[$0] = dynamic_bit_slice(literal.1,
                                                          literal.2, width=$0)
  }
  )";

  std::string bytes_str = "0x";
  std::string start_bytes_str = "0x";
  std::vector<uint8_t> bytes;
  std::vector<uint8_t> start_bytes;

  for (int i = 0;
       i < CeilOfRatio(literal_width, static_cast<int64_t>(CHAR_BIT)); i++) {
    absl::StrAppend(&bytes_str, absl::Hex(i % 256, absl::kZeroPad2));
    bytes.push_back(i % 256);
  }

  // Set start to be much larger than the operand.
  for (int i = 0;
       i < CeilOfRatio(2 * literal_width, static_cast<int64_t>(CHAR_BIT));
       i++) {
    absl::StrAppend(&start_bytes_str, absl::Hex(255, absl::kZeroPad2));
    start_bytes.push_back(255);
  }
  std::string formatted_ir =
      absl::Substitute(ir_text, slice_width, literal_width, bytes_str,
                       2 * literal_width, start_bytes_str);
  XLS_ASSIGN_OR_RETURN(auto package, Parser::ParsePackage(formatted_ir));
  XLS_ASSIGN_OR_RETURN(Function * function, package->EntryFunction());

  // Clearly out of bounds
  Value expected = Value(Bits(slice_width));
  EXPECT_THAT(param.evaluator(function, {}), IsOkAndHolds(expected));

  return absl::OkStatus();
}

TEST_P(IrEvaluatorTestBase, DynamicBitSlice) {
  XLS_ASSERT_OK(RunDynamicBitSliceTestLargeStart(GetParam(), 64, 25));
  XLS_ASSERT_OK(RunDynamicBitSliceTestLargeStart(GetParam(), 200, 20));
  XLS_ASSERT_OK(RunDynamicBitSliceTest(GetParam(), 16, 24, 8, 3));
  XLS_ASSERT_OK(RunDynamicBitSliceTest(GetParam(), 16, 0, 1, 8));
  XLS_ASSERT_OK(RunDynamicBitSliceTest(GetParam(), 16, 15, 4, 3));
  XLS_ASSERT_OK(RunDynamicBitSliceTest(GetParam(), 27, 9, 8, 3));
  XLS_ASSERT_OK(RunDynamicBitSliceTest(GetParam(), 32, 9, 16, 3));
  XLS_ASSERT_OK(RunDynamicBitSliceTest(GetParam(), 64, 15, 32, 27));
  XLS_ASSERT_OK(RunDynamicBitSliceTest(GetParam(), 128, 100, 32, 50));
  XLS_ASSERT_OK(RunDynamicBitSliceTest(GetParam(), 1024, 747, 32, 32));
  XLS_ASSERT_OK(RunDynamicBitSliceTest(GetParam(), 65536, 8192, 200, 32768));
}
// Test driven by b/148608161.
TEST_P(IrEvaluatorTestBase, FunnyShapedArrays) {
  XLS_ASSERT_OK_AND_ASSIGN(auto package, Parser::ParsePackage(R"(
  package test

  fn main() -> bits[20][2] {
    ret literal.1: bits[20][2] = literal(value=[0xabcde, 0x12345])
  }
  )"));
  XLS_ASSERT_OK_AND_ASSIGN(Function * function, package->EntryFunction());
  Value expected =
      Value::ArrayOrDie({Value(UBits(0xabcde, 20)), Value(UBits(0x12345, 20))});
  EXPECT_THAT(RunWithValues(function, {}), IsOkAndHolds(expected));
}

absl::Status RunBitwiseReduceTest(
    const IrEvaluatorTestParam& param, const std::string& reduce_op,
    std::function<Value(const Bits&)> generate_expected, const Bits& bits) {
  constexpr absl::string_view ir_text = R"(
  package test

  fn main(x: bits[$1]) -> bits[1] {
    ret $0.1: bits[1] = $0(x)
  }
  )";

  std::string formatted_ir =
      absl::Substitute(ir_text, reduce_op, bits.bit_count());
  XLS_ASSIGN_OR_RETURN(auto package, Parser::ParsePackage(formatted_ir));
  XLS_ASSIGN_OR_RETURN(Function * function, package->EntryFunction());

  EXPECT_THAT(param.evaluator(function, {Value(bits)}),
              IsOkAndHolds(generate_expected(bits)));
  return absl::OkStatus();
}

TEST_P(IrEvaluatorTestBase, InterpretAndReduce) {
  auto gen_expected = [](const Bits& bits) {
    return Value(bits_ops::AndReduce(bits));
  };
  std::vector<Bits> test_cases = {
      UBits(0, 1),        UBits(1, 1),       UBits(0xFF, 8),
      UBits(0x7FF, 11),   UBits(0x0FF, 11),  UBits(0x1FFFFFFFF, 33),
      Bits::AllOnes(31),  Bits::AllOnes(63), Bits::AllOnes(127),
      Bits::AllOnes(255),
  };
  for (const auto& test_case : test_cases) {
    XLS_ASSERT_OK(RunBitwiseReduceTest(GetParam(), "and_reduce", gen_expected,
                                       test_case));
  }
}

TEST_P(IrEvaluatorTestBase, InterpretOrReduce) {
  auto gen_expected = [](const Bits& bits) {
    return Value(bits_ops::OrReduce(bits));
  };
  std::vector<Bits> test_cases = {
      UBits(0, 1),         UBits(1, 1),        UBits(0xFF, 8),
      UBits(0x7FF, 11),    UBits(0x0FF, 11),   UBits(0x1FFFFFFFF, 33),
      Bits::AllOnes(255),  Bits::AllOnes(256), Bits::AllOnes(257),
      Bits::AllOnes(2048),
  };
  for (const auto& test_case : test_cases) {
    XLS_ASSERT_OK(
        RunBitwiseReduceTest(GetParam(), "or_reduce", gen_expected, test_case));
  }
}

TEST_P(IrEvaluatorTestBase, InterpretXorReduce) {
  auto gen_expected = [](const Bits& bits) {
    return Value(bits_ops::XorReduce(bits));
  };
  std::vector<Bits> test_cases = {
      UBits(0, 1),         UBits(1, 1),        UBits(0xFF, 8),
      UBits(0x7FF, 11),    UBits(0x0FF, 11),   UBits(0x1FFFFFFFF, 33),
      Bits::AllOnes(255),  Bits::AllOnes(256), Bits::AllOnes(257),
      Bits::AllOnes(2048),
  };
  for (const auto& test_case : test_cases) {
    XLS_ASSERT_OK(RunBitwiseReduceTest(GetParam(), "xor_reduce", gen_expected,
                                       test_case));
  }
}

TEST_P(IrEvaluatorTestBase, ArrayIndex) {
  auto package = CreatePackage();
  std::string input = R"(
  fn main(a: bits[13][3], idx: bits[3]) -> bits[13] {
    ret result: bits[13] = array_index(a, indices=[idx])
  }
  )";
  XLS_ASSERT_OK_AND_ASSIGN(Function * function,
                           ParseFunction(input, package.get()));

  EXPECT_THAT(
      RunWithValues(function,
                    {Value::UBitsArray({11, 22, 33}, /*bit_count=*/13).value(),
                     Value(UBits(1, 3))}),
      IsOkAndHolds(Value(UBits(22, 13))));
  EXPECT_THAT(
      RunWithValues(function,
                    {Value::UBitsArray({11, 22, 33}, /*bit_count=*/13).value(),
                     Value(UBits(2, 3))}),
      IsOkAndHolds(Value(UBits(33, 13))));
  // Out of bounds access should return last element.
  EXPECT_THAT(
      RunWithValues(function,
                    {Value::UBitsArray({11, 22, 33}, /*bit_count=*/13).value(),
                     Value(UBits(6, 3))}),
      IsOkAndHolds(Value(UBits(33, 13))));
}

TEST_P(IrEvaluatorTestBase, ArrayIndex2DArray) {
  auto package = CreatePackage();
  std::string input = R"(
  fn main(a: bits[13][3][2], idx0: bits[3], idx1: bits[7]) -> bits[13] {
    ret result: bits[13] = array_index(a, indices=[idx0, idx1])
  }
  )";
  XLS_ASSERT_OK_AND_ASSIGN(Function * function,
                           ParseFunction(input, package.get()));

  EXPECT_THAT(
      RunWithValues(function, {Value::UBits2DArray({{44, 55, 66}, {11, 22, 33}},
                                                   /*bit_count=*/13)
                                   .value(),
                               Value(UBits(1, 3)), Value(UBits(2, 7))}),
      IsOkAndHolds(Value(UBits(33, 13))));
  EXPECT_THAT(
      RunWithValues(function, {Value::UBits2DArray({{44, 55, 66}, {11, 22, 33}},
                                                   /*bit_count=*/13)
                                   .value(),
                               Value(UBits(0, 3)), Value(UBits(1, 7))}),
      IsOkAndHolds(Value(UBits(55, 13))));

  // Out of bounds access should return last element of respective array.
  EXPECT_THAT(
      RunWithValues(function, {Value::UBits2DArray({{44, 55, 66}, {11, 22, 33}},
                                                   /*bit_count=*/13)
                                   .value(),
                               Value(UBits(6, 3)), Value(UBits(1, 7))}),
      IsOkAndHolds(Value(UBits(22, 13))));
  EXPECT_THAT(
      RunWithValues(function, {Value::UBits2DArray({{44, 55, 66}, {11, 22, 33}},
                                                   /*bit_count=*/13)
                                   .value(),
                               Value(UBits(0, 3)), Value(UBits(100, 7))}),
      IsOkAndHolds(Value(UBits(66, 13))));
}

TEST_P(IrEvaluatorTestBase, ArrayIndex2DArrayReturnArray) {
  auto package = CreatePackage();
  std::string input = R"(
  fn main(a: bits[13][3][2], idx: bits[3]) -> bits[13][3] {
    ret result: bits[13][3] = array_index(a, indices=[idx])
  }
  )";
  XLS_ASSERT_OK_AND_ASSIGN(Function * function,
                           ParseFunction(input, package.get()));

  EXPECT_THAT(
      RunWithValues(function, {Value::UBits2DArray({{44, 55, 66}, {11, 22, 33}},
                                                   /*bit_count=*/13)
                                   .value(),
                               Value(UBits(1, 3))}),
      IsOkAndHolds(Value::UBitsArray({11, 22, 33}, /*bit_count=*/13).value()));
  EXPECT_THAT(
      RunWithValues(function, {Value::UBits2DArray({{44, 55, 66}, {11, 22, 33}},
                                                   /*bit_count=*/13)
                                   .value(),
                               Value(UBits(0, 3))}),
      IsOkAndHolds(Value::UBitsArray({44, 55, 66}, /*bit_count=*/13).value()));

  // Out of bounds access should return last element of respective array.
  EXPECT_THAT(
      RunWithValues(function, {Value::UBits2DArray({{44, 55, 66}, {11, 22, 33}},
                                                   /*bit_count=*/13)
                                   .value(),
                               Value(UBits(7, 3))}),
      IsOkAndHolds(Value::UBitsArray({11, 22, 33}, /*bit_count=*/13).value()));
}

TEST_P(IrEvaluatorTestBase, ArrayIndex2DArrayNilIndex) {
  auto package = CreatePackage();
  std::string input = R"(
  fn main(a: bits[13][3][2]) -> bits[13][3][2] {
    ret result: bits[13][3][2] = array_index(a, indices=[])
  }
  )";
  XLS_ASSERT_OK_AND_ASSIGN(Function * function,
                           ParseFunction(input, package.get()));

  XLS_ASSERT_OK_AND_ASSIGN(
      Value array,
      Value::UBits2DArray({{44, 55, 66}, {11, 22, 33}}, /*bit_count=*/13));
  EXPECT_THAT(RunWithValues(function, {array}), IsOkAndHolds(array));
}

TEST_P(IrEvaluatorTestBase, ArrayUpdate) {
  auto package = CreatePackage();
  std::string input = R"(
  fn main(a: bits[13][3], idx: bits[7], value: bits[13]) -> bits[13][3] {
    ret result: bits[13][3] = array_update(a, value, indices=[idx])
 }
  )";
  XLS_ASSERT_OK_AND_ASSIGN(Function * function,
                           ParseFunction(input, package.get()));

  EXPECT_THAT(
      RunWithValues(function,
                    {Value::UBitsArray({11, 22, 33}, /*bit_count=*/13).value(),
                     Value(UBits(1, 7)), Value(UBits(123, 13))}),
      IsOkAndHolds(Value::UBitsArray({11, 123, 33}, /*bit_count=*/13).value()));
  EXPECT_THAT(
      RunWithValues(function,
                    {Value::UBitsArray({11, 22, 33}, /*bit_count=*/13).value(),
                     Value(UBits(2, 7)), Value(UBits(123, 13))}),
      IsOkAndHolds(Value::UBitsArray({11, 22, 123},
                                     /*bit_count=*/13)
                       .value()));
  // Out of bounds access should update no element.
  EXPECT_THAT(
      RunWithValues(function,
                    {Value::UBitsArray({11, 22, 33}, /*bit_count=*/13).value(),
                     Value(UBits(3, 7)), Value(UBits(123, 13))}),
      IsOkAndHolds(Value::UBitsArray({11, 22, 33},
                                     /*bit_count=*/13)
                       .value()));
  EXPECT_THAT(
      RunWithValues(function,
                    {Value::UBitsArray({11, 22, 33}, /*bit_count=*/13).value(),
                     Value(UBits(55, 7)), Value(UBits(123, 13))}),
      IsOkAndHolds(Value::UBitsArray({11, 22, 33},
                                     /*bit_count=*/13)
                       .value()));
}

TEST_P(IrEvaluatorTestBase, ArrayUpdate2DArray) {
  auto package = CreatePackage();
  std::string input = R"(
  fn main(a: bits[13][3][2], idx0: bits[3], idx1: bits[7], value: bits[13]) -> bits[13][3][2] {
    ret result: bits[13][3][2] = array_update(a, value, indices=[idx0, idx1])
  }
  )";
  XLS_ASSERT_OK_AND_ASSIGN(Function * function,
                           ParseFunction(input, package.get()));

  EXPECT_THAT(
      RunWithValues(
          function,
          {Value::UBits2DArray({{44, 55, 66}, {11, 22, 33}}, /*bit_count=*/13)
               .value(),
           Value(UBits(1, 3)), Value(UBits(2, 7)), Value(UBits(999, 13))}),
      IsOkAndHolds(
          Value::UBits2DArray({{44, 55, 66}, {11, 22, 999}}, /*bit_count=*/13)
              .value()));
  EXPECT_THAT(
      RunWithValues(
          function,
          {Value::UBits2DArray({{44, 55, 66}, {11, 22, 33}}, /*bit_count=*/13)
               .value(),
           Value(UBits(0, 3)), Value(UBits(1, 7)), Value(UBits(999, 13))}),
      IsOkAndHolds(
          Value::UBits2DArray({{44, 999, 66}, {11, 22, 33}}, /*bit_count=*/13)
              .value()));
  // Out of bounds on either index should update no element.
  EXPECT_THAT(
      RunWithValues(
          function,
          {Value::UBits2DArray({{44, 55, 66}, {11, 22, 33}}, /*bit_count=*/13)
               .value(),
           Value(UBits(4, 3)), Value(UBits(1, 7)), Value(UBits(999, 13))}),
      IsOkAndHolds(
          Value::UBits2DArray({{44, 55, 66}, {11, 22, 33}}, /*bit_count=*/13)
              .value()));
  EXPECT_THAT(
      RunWithValues(
          function,
          {Value::UBits2DArray({{44, 55, 66}, {11, 22, 33}}, /*bit_count=*/13)
               .value(),
           Value(UBits(1, 3)), Value(UBits(3, 7)), Value(UBits(999, 13))}),
      IsOkAndHolds(
          Value::UBits2DArray({{44, 55, 66}, {11, 22, 33}}, /*bit_count=*/13)
              .value()));
}

TEST_P(IrEvaluatorTestBase, ArrayUpdate2DArrayWithArray) {
  auto package = CreatePackage();
  std::string input = R"(
  fn main(a: bits[13][3][2], idx: bits[3], value: bits[13][3]) -> bits[13][3][2] {
    ret result: bits[13][3][2] = array_update(a, value, indices=[idx])
  }
  )";
  XLS_ASSERT_OK_AND_ASSIGN(Function * function,
                           ParseFunction(input, package.get()));

  EXPECT_THAT(
      RunWithValues(
          function,
          {Value::UBits2DArray({{44, 55, 66}, {11, 22, 33}}, /*bit_count=*/13)
               .value(),
           Value(UBits(1, 3)),
           Value::UBitsArray({999, 888, 777}, /*bit_count=*/13).value()}),
      IsOkAndHolds(
          Value::UBits2DArray({{44, 55, 66}, {999, 888, 777}}, /*bit_count=*/13)
              .value()));

  // Out of bounds should update no element.
  EXPECT_THAT(
      RunWithValues(
          function,
          {Value::UBits2DArray({{44, 55, 66}, {11, 22, 33}}, /*bit_count=*/13)
               .value(),
           Value(UBits(2, 3)),
           Value::UBitsArray({999, 888, 777}, /*bit_count=*/13).value()}),
      IsOkAndHolds(
          Value::UBits2DArray({{44, 55, 66}, {11, 22, 33}}, /*bit_count=*/13)
              .value()));
}

TEST_P(IrEvaluatorTestBase, ArrayUpdate2DArrayNilIndex) {
  auto package = CreatePackage();
  std::string input = R"(
  fn main(a: bits[13][3][2], value: bits[13][3][2]) -> bits[13][3][2] {
    ret result: bits[13][3][2] = array_update(a, value, indices=[])
  }
  )";
  XLS_ASSERT_OK_AND_ASSIGN(Function * function,
                           ParseFunction(input, package.get()));

  EXPECT_THAT(
      RunWithValues(
          function,
          {Value::UBits2DArray({{44, 55, 66}, {11, 22, 33}}, /*bit_count=*/13)
               .value(),
           Value::UBits2DArray({{99, 123, 4}, {432, 7, 42}}, /*bit_count=*/13)
               .value()}),
      IsOkAndHolds(
          Value::UBits2DArray({{99, 123, 4}, {432, 7, 42}}, /*bit_count=*/13)
              .value()));
}

TEST_P(IrEvaluatorTestBase, ArrayUpdateBitsValueNilIndex) {
  auto package = CreatePackage();
  std::string input = R"(
  fn main(a: bits[1234], value: bits[1234]) -> bits[1234] {
    ret result: bits[1234] = array_update(a, value, indices=[])
  }
  )";
  XLS_ASSERT_OK_AND_ASSIGN(Function * function,
                           ParseFunction(input, package.get()));

  EXPECT_THAT(RunWithValues(function,
                            {Value(UBits(42, 1234)), Value(UBits(888, 1234))}),
              IsOkAndHolds(Value(UBits(888, 1234))));
}

TEST_P(IrEvaluatorTestBase, BitSliceUpdate) {
  auto package = CreatePackage();
  std::string input = R"(
  fn main(a: bits[32], start: bits[32], value: bits[8]) -> bits[32] {
    ret result: bits[32] = bit_slice_update(a, start, value)
  }
  )";
  XLS_ASSERT_OK_AND_ASSIGN(Function * function,
                           ParseFunction(input, package.get()));

  EXPECT_THAT(RunWithUint64s(function, {0x1234abcd, 0, 0xef}),
              IsOkAndHolds(0x1234abef));
  EXPECT_THAT(RunWithUint64s(function, {0x1234abcd, 4, 0xef}),
              IsOkAndHolds(0x1234aefd));
  EXPECT_THAT(RunWithUint64s(function, {0x1234abcd, 16, 0xef}),
              IsOkAndHolds(0x12efabcd));
  EXPECT_THAT(RunWithUint64s(function, {0x1234abcd, 31, 0xef}),
              IsOkAndHolds(0x9234abcd));
  EXPECT_THAT(RunWithUint64s(function, {0x1234abcd, 32, 0xef}),
              IsOkAndHolds(0x1234abcd));
  EXPECT_THAT(RunWithUint64s(function, {0x1234abcd, 1234567, 0xef}),
              IsOkAndHolds(0x1234abcd));
}

TEST_P(IrEvaluatorTestBase, BitSliceUpdateWideUpdateValue) {
  auto package = CreatePackage();
  std::string input = R"(
  fn main(a: bits[16], start: bits[32], value: bits[157]) -> bits[16] {
    ret result: bits[16] = bit_slice_update(a, start, value)
  }
  )";
  XLS_ASSERT_OK_AND_ASSIGN(Function * function,
                           ParseFunction(input, package.get()));

  EXPECT_THAT(RunWithUint64s(function, {0x1234, 0, 0xabcdef}),
              IsOkAndHolds(0xcdef));
  EXPECT_THAT(RunWithUint64s(function, {0x1234, 5, 0xabcdef}),
              IsOkAndHolds(0xbdf4));
  EXPECT_THAT(RunWithUint64s(function, {0x1234, 44, 0xabcdef}),
              IsOkAndHolds(0x1234));
}

TEST_P(IrEvaluatorTestBase, NestedEmptyTuple) {
  auto package = CreatePackage();
  std::string input = R"(
  fn main(x: (())) -> () {
    ret result: () = tuple_index(x, index=0)
  }
  )";
  XLS_ASSERT_OK_AND_ASSIGN(Function * function,
                           ParseFunction(input, package.get()));
  EXPECT_THAT(RunWithValues(function, {Value::Tuple({Value::Tuple({})})}),
              IsOkAndHolds(Value::Tuple({})));
}

TEST_P(IrEvaluatorTestBase, NestedNestedEmptyTuple) {
  auto package = CreatePackage();
  std::string input = R"(
  fn main(x: ((()))) -> (()) {
    ret result: (()) = tuple_index(x, index=0)
  }
  )";
  XLS_ASSERT_OK_AND_ASSIGN(Function * function,
                           ParseFunction(input, package.get()));
  EXPECT_THAT(RunWithValues(function,
                            {Value::Tuple({Value::Tuple({Value::Tuple({})})})}),
              IsOkAndHolds(Value::Tuple({Value::Tuple({})})));
}

TEST_P(IrEvaluatorTestBase, NestedEmptyTupleWithMultipleElements) {
  auto package = CreatePackage();
  std::string input = R"(
  fn main(x: ((), ())) -> () {
    ret result: () = tuple_index(x, index=1)
  }
  )";
  XLS_ASSERT_OK_AND_ASSIGN(Function * function,
                           ParseFunction(input, package.get()));
  EXPECT_THAT(
      RunWithValues(function,
                    {Value::Tuple({Value::Tuple({}), Value::Tuple({})})}),
      IsOkAndHolds(Value::Tuple({})));
}

TEST_P(IrEvaluatorTestBase, ReturnEmptyTupleEmptyTuple) {
  auto package = CreatePackage();
  std::string input = R"(
  fn main(x: ()) -> ((), ()) {
    ret result: ((), ()) = tuple(x, x)
  }
  )";
  XLS_ASSERT_OK_AND_ASSIGN(Function * function,
                           ParseFunction(input, package.get()));
  EXPECT_THAT(RunWithValues(function, {Value::Tuple({})}),
              IsOkAndHolds(Value::Tuple({Value::Tuple({}), Value::Tuple({})})));
}

TEST_P(IrEvaluatorTestBase, ArrayOfEmptyTuples) {
  auto package = CreatePackage();
  std::string input = R"(
  fn main(x: ()[2]) -> () {
    index: bits[32] = literal(value=0)
    ret result: () = array_index(x, indices=[index])
  }
  )";
  XLS_ASSERT_OK_AND_ASSIGN(Function * function,
                           ParseFunction(input, package.get()));
  XLS_ASSERT_OK_AND_ASSIGN(Value arg,
                           Value::Array({Value::Tuple({}), Value::Tuple({})}));
  EXPECT_THAT(RunWithValues(function, {arg}), IsOkAndHolds(Value::Tuple({})));
}

TEST_P(IrEvaluatorTestBase, AssertTest) {
  auto p = CreatePackage();
  FunctionBuilder b(TestName(), p.get());
  auto p0 = b.Param("tkn", p->GetTokenType());
  auto p1 = b.Param("cond", p->GetBitsType(1));
  b.Assert(p0, p1, "the assertion error message", {});
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, b.Build());

  EXPECT_THAT(RunWithValues(f, {Value::Token(), Value(UBits(1, 1))}),
              IsOkAndHolds(Value::Token()));
  EXPECT_THAT(RunWithValues(f, {Value::Token(), Value(UBits(0, 1))}),
              StatusIs(absl::StatusCode::kAborted,
                       HasSubstr("the assertion error message")));
}

TEST_P(IrEvaluatorTestBase, GateBitsTypeTest) {
  auto p = CreatePackage();
  FunctionBuilder b(TestName(), p.get());
  auto cond = b.Param("cond", p->GetBitsType(1));
  auto data = b.Param("data", p->GetBitsType(32));
  b.Gate(cond, data);
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, b.Build());
  EXPECT_THAT(RunWithUint64s(f, {0, 42}), 42);
  EXPECT_THAT(RunWithUint64s(f, {1, 42}), 0);
}

TEST_P(IrEvaluatorTestBase, GateCompoundTypeTest) {
  auto p = CreatePackage();
  FunctionBuilder b(TestName(), p.get());
  auto cond = b.Param("cond", p->GetBitsType(1));
  auto data = b.Param(
      "data", p->GetArrayType(2, p->GetTupleType({p->GetBitsType(32)})));
  b.Gate(cond, data);
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, b.Build());

  XLS_ASSERT_OK_AND_ASSIGN(
      Value a, Value::Array({Value::Tuple({Value(UBits(42, 32))}),
                             Value::Tuple({Value(UBits(123, 32))})}));

  EXPECT_THAT(RunWithValues(f, {Value(UBits(0, 1)), a}), a);
  EXPECT_THAT(RunWithValues(f, {Value(UBits(1, 1)), a}),
              ZeroOfType(data.node()->GetType()));
}

}  // namespace xls
