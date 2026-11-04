// Copyright 2026 The XLS Authors
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

#include "xls/dslx/tabulate.h"

#include <string>
#include <string_view>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "xls/common/status/matchers.h"
#include "xls/common/status/status_macros.h"
#include "xls/dslx/create_import_data.h"
#include "xls/dslx/import_data.h"
#include "xls/dslx/parse_and_typecheck.h"

namespace xls::dslx {
namespace {

using ::absl_testing::StatusIs;
using ::testing::HasSubstr;

absl::StatusOr<std::string> ParseAndTabulate(std::string_view program,
                                             std::string_view fn_name) {
  ImportData import_data = CreateImportDataForTest();
  XLS_ASSIGN_OR_RETURN(
      TypecheckedModule tm,
      ParseAndTypecheck(program, "test.x", "test", &import_data));
  return TabulateFunction(&import_data, tm, fn_name);
}

TEST(TabulateTest, SimpleEnumAndBoolTable) {
  constexpr std::string_view kProgram = R"(
enum Op : u1 {
    ADD = 0,
    SUB = 1,
}

#[tabulate]
fn compute(op: Op, x: u1) -> u2 {
    match op {
        Op::ADD => (x as u2) + 1,
        Op::SUB => 2 - (x as u2),
    }
}
)";
  XLS_ASSERT_OK_AND_ASSIGN(std::string md,
                           ParseAndTabulate(kProgram, "compute"));
  EXPECT_EQ(md,
            "| op  | x     | result |\n"
            "| --- | ----- | ------ |\n"
            "| ADD | false | 1      |\n"
            "| ADD | true  | 2      |\n"
            "| SUB | false | 2      |\n"
            "| SUB | true  | 1      |\n");
}

TEST(TabulateTest, StructReturnTable) {
  constexpr std::string_view kProgram = R"(
enum AluOp : u1 {
    ADD = 0,
    XOR = 1,
}

struct AluOutput { value: u2, zero: bool }

#[tabulate]
fn eval_alu(op: AluOp, carry_in: bool) -> AluOutput {
    let value = match op {
        AluOp::ADD => 1 + (carry_in as u2),
        AluOp::XOR => 1 ^ (carry_in as u2),
    };
    AluOutput { value, zero: value == 0 }
}
)";
  XLS_ASSERT_OK_AND_ASSIGN(std::string md,
                           ParseAndTabulate(kProgram, "eval_alu"));
  EXPECT_EQ(md,
            "| op  | carry_in | value | zero  |\n"
            "| --- | -------- | ----- | ----- |\n"
            "| ADD | false    | 1     | false |\n"
            "| ADD | true     | 2     | false |\n"
            "| XOR | false    | 1     | false |\n"
            "| XOR | true     | 0     | true  |\n");
}

TEST(TabulateTest, TupleReturnWithSignedIntsTable) {
  constexpr std::string_view kProgram = R"(
#[tabulate]
fn saturate_inc(offset: s2) -> (s2, bool) {
    if offset == 1 { (1, true) } else { (offset + 1, false) }
}
)";
  XLS_ASSERT_OK_AND_ASSIGN(std::string md,
                           ParseAndTabulate(kProgram, "saturate_inc"));
  EXPECT_EQ(md,
            "| offset | result_0 | result_1 |\n"
            "| ------ | -------- | -------- |\n"
            "| -2     | -1       | false    |\n"
            "| -1     | 0        | false    |\n"
            "| 0      | 1        | false    |\n"
            "| 1      | 1        | true     |\n");
}

TEST(TabulateTest, UnitToUnitFunctionError) {
  constexpr std::string_view kProgram = R"(
#[tabulate]
fn nop() {
}
)";
  EXPECT_THAT(ParseAndTabulate(kProgram, "nop"),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("has no parameters")));
}

TEST(TabulateTest, ZeroParamFunctionError) {
  constexpr std::string_view kProgram = R"(
#[tabulate]
fn default_alu_mode() -> u2 {
    2
}
)";
  EXPECT_THAT(ParseAndTabulate(kProgram, "default_alu_mode"),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("has no parameters")));
}

TEST(TabulateTest, UnitReturnFunctionError) {
  constexpr std::string_view kProgram = R"(
#[tabulate]
fn consume_opcode(opcode: u2) {
}
)";
  EXPECT_THAT(ParseAndTabulate(kProgram, "consume_opcode"),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("has an empty return type")));
}

TEST(TabulateTest, MissingTabulateAttributeError) {
  constexpr std::string_view kProgram = R"(
fn plain(x: u2) -> u2 { x }
)";
  EXPECT_THAT(ParseAndTabulate(kProgram, "plain"),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("does not have a #[tabulate] attribute")));
}

TEST(TabulateTest, FunctionNotFoundError) {
  constexpr std::string_view kProgram = R"(
#[tabulate]
fn plain(x: u2) -> u2 { x }
)";
  EXPECT_THAT(
      ParseAndTabulate(kProgram, "nonexistent"),
      StatusIs(absl::StatusCode::kNotFound, HasSubstr("not found in module")));
}

TEST(TabulateTest, ParametricFunctionError) {
  constexpr std::string_view kProgram = R"(
#[tabulate]
fn passthrough<WIDTH: u32>(operand: uN[WIDTH]) -> uN[WIDTH] {
    operand
}
)";
  EXPECT_THAT(
      ParseAndTabulate(kProgram, "passthrough"),
      StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("is parametric")));
}

TEST(TabulateTest, UnsupportedParamTypeError) {
  constexpr std::string_view kProgram = R"(
#[tabulate]
fn add_pair(operands: (u2, u2)) -> u2 {
    operands.0 + operands.1
}
)";
  EXPECT_THAT(ParseAndTabulate(kProgram, "add_pair"),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("unsupported type")));
}

TEST(TabulateTest, ParamWidthExceedsLimitError) {
  constexpr std::string_view kProgram = R"(
#[tabulate]
fn is_zero_pc(pc: u11) -> bool {
    pc == 0
}
)";
  EXPECT_THAT(
      ParseAndTabulate(kProgram, "is_zero_pc"),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("exceeds the maximum single-parameter width")));
}

TEST(TabulateTest, DomainSizeExceedsLimitError) {
  constexpr std::string_view kProgram = R"(
#[tabulate]
fn addr_match(tag_a: u6, tag_b: u6) -> bool {
    tag_a == tag_b
}
)";
  EXPECT_THAT(ParseAndTabulate(kProgram, "addr_match"),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("exceeds maximum size")));
}

}  // namespace
}  // namespace xls::dslx
