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

TEST(TabulateTest, DomainsRangesOnWideAndSignedInts) {
  constexpr std::string_view kProgram = R"(
#[tabulate(domains=`u32:100..102, s16:-1..=1`)]
pub fn add_offset(base: u32, delta: s16) -> s32 {
    (base as s32) + (delta as s32)
}
)";
  ImportData import_data = CreateImportDataForTest();
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule tm,
      ParseAndTypecheck(kProgram, "test.x", "test", &import_data));
  EXPECT_TRUE(tm.warnings.empty());
  XLS_ASSERT_OK_AND_ASSIGN(std::string md,
                           TabulateFunction(&import_data, tm, "add_offset"));
  EXPECT_EQ(md,
            "| base | delta | result |\n"
            "| ---- | ----- | ------ |\n"
            "| 100  | -1    | 99     |\n"
            "| 100  | 0     | 100    |\n"
            "| 100  | 1     | 101    |\n"
            "| 101  | -1    | 100    |\n"
            "| 101  | 0     | 101    |\n"
            "| 101  | 1     | 102    |\n");
}

TEST(TabulateTest, DomainsFullWidthInclusiveRangeDoesNotWrap) {
  constexpr std::string_view kProgram = R"(
#[tabulate(domains=`u2:0..=3`)]
fn passthrough(x: u2) -> u2 {
    x
}
)";
  XLS_ASSERT_OK_AND_ASSIGN(std::string md,
                           ParseAndTabulate(kProgram, "passthrough"));
  EXPECT_EQ(md,
            "| x   | result |\n"
            "| --- | ------ |\n"
            "| 0   | 0      |\n"
            "| 1   | 1      |\n"
            "| 2   | 2      |\n"
            "| 3   | 3      |\n");
}

TEST(TabulateTest, DomainsElementOfArrayAndArbitraryEmptyTuple) {
  constexpr std::string_view kProgram = R"(
enum Op : u2 {
    NOP = 0,
    ADD = 1,
    SUB = 2,
}

#[tabulate(domains=`[Op::SUB, Op::ADD], [u32:10, u32:25], ()`)]
fn exec(op: Op, lhs: u32, inc: bool) -> u32 {
    let adjusted = lhs + (inc as u32);
    match op {
        Op::NOP => u32:0,
        Op::ADD => adjusted + u32:1,
        Op::SUB => adjusted - u32:1,
    }
}
)";
  XLS_ASSERT_OK_AND_ASSIGN(std::string md, ParseAndTabulate(kProgram, "exec"));
  EXPECT_EQ(md,
            "| op  | lhs | inc   | result |\n"
            "| --- | --- | ----- | ------ |\n"
            "| SUB | 10  | false | 9      |\n"
            "| SUB | 10  | true  | 10     |\n"
            "| SUB | 25  | false | 24     |\n"
            "| SUB | 25  | true  | 25     |\n"
            "| ADD | 10  | false | 11     |\n"
            "| ADD | 10  | true  | 12     |\n"
            "| ADD | 25  | false | 26     |\n"
            "| ADD | 25  | true  | 27     |\n");
}

TEST(TabulateTest, DomainsConstHelperAndImportedModule) {
  constexpr std::string_view kImported = R"(
pub enum Mode : u1 {
    FAST = 0,
    SAFE = 1,
}

pub const X_DOMAIN = u32:5..7;
)";
  constexpr std::string_view kProgram = R"(
import imported;

fn mode_domain() -> imported::Mode[1] {
    [imported::Mode::SAFE]
}

#[tabulate(domains=`imported::X_DOMAIN, mode_domain()`)]
fn scale(x: u32, mode: imported::Mode) -> u32 {
    match mode {
        imported::Mode::FAST => x,
        imported::Mode::SAFE => x * u32:2,
    }
}
)";
  ImportData import_data = CreateImportDataForTest();
  XLS_ASSERT_OK(
      ParseAndTypecheck(kImported, "imported.x", "imported", &import_data));
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule tm,
      ParseAndTypecheck(kProgram, "test.x", "test", &import_data));
  XLS_ASSERT_OK_AND_ASSIGN(std::string md,
                           TabulateFunction(&import_data, tm, "scale"));
  EXPECT_EQ(md,
            "| x   | mode | result |\n"
            "| --- | ---- | ------ |\n"
            "| 5   | SAFE | 10     |\n"
            "| 6   | SAFE | 12     |\n");
}

TEST(TabulateTest, DomainsTupleAndArrayParameters) {
  constexpr std::string_view kProgram = R"(
#[tabulate(domains=`(u32:1..3, [false, true]), ([u1:1],)`)]
fn combine(pair: (u32, bool), flags: u1[2]) -> u32[2] {
    let base = if pair.1 { pair.0 } else { u32:0 };
    [base + (flags[0] as u32), base + (flags[1] as u32)]
}
)";
  XLS_ASSERT_OK_AND_ASSIGN(std::string md,
                           ParseAndTabulate(kProgram, "combine"));
  EXPECT_EQ(md,
            "| pair       | flags         | result |\n"
            "| ---------- | ------------- | ------ |\n"
            "| (1, false) | [true, false] | [1, 0] |\n"
            "| (1, false) | [true, true]  | [1, 1] |\n"
            "| (1, true)  | [true, false] | [2, 1] |\n"
            "| (1, true)  | [true, true]  | [2, 2] |\n"
            "| (2, false) | [true, false] | [1, 0] |\n"
            "| (2, false) | [true, true]  | [1, 1] |\n"
            "| (2, true)  | [true, false] | [3, 2] |\n"
            "| (2, true)  | [true, true]  | [3, 3] |\n");
}

TEST(TabulateTest, DomainsInlineStructAndDerivedFuzzDomainStruct) {
  constexpr std::string_view kProgram = R"(
struct Config {
    gain: u32,
    enable: bool,
}

#[fuzz_domain("PacketDomain")]
struct Packet {
    payload: u16,
    valid: bool,
}

fn make_packet_domain() -> PacketDomain {
    PacketDomain { payload: [u16:10, u16:20], valid: [true] }
}

#[tabulate(domains=`Config { gain: u32:2..4 }, make_packet_domain()`)]
fn process(cfg: Config, pkt: Packet) -> u32 {
    if cfg.enable && pkt.valid {
        cfg.gain * (pkt.payload as u32)
    } else {
        u32:0
    }
}
)";
  XLS_ASSERT_OK_AND_ASSIGN(std::string md,
                           ParseAndTabulate(kProgram, "process"));
  EXPECT_EQ(md,
            "| cfg                               | pkt                         "
            "        | result |\n"
            "| --------------------------------- | "
            "----------------------------------- | ------ |\n"
            "| Config { gain: 2, enable: false } | Packet { payload: 10, "
            "valid: true } | 0      |\n"
            "| Config { gain: 2, enable: false } | Packet { payload: 20, "
            "valid: true } | 0      |\n"
            "| Config { gain: 2, enable: true }  | Packet { payload: 10, "
            "valid: true } | 20     |\n"
            "| Config { gain: 2, enable: true }  | Packet { payload: 20, "
            "valid: true } | 40     |\n"
            "| Config { gain: 3, enable: false } | Packet { payload: 10, "
            "valid: true } | 0      |\n"
            "| Config { gain: 3, enable: false } | Packet { payload: 20, "
            "valid: true } | 0      |\n"
            "| Config { gain: 3, enable: true }  | Packet { payload: 10, "
            "valid: true } | 30     |\n"
            "| Config { gain: 3, enable: true }  | Packet { payload: 20, "
            "valid: true } | 60     |\n");
}

TEST(TabulateTest, DomainsWithLambdaMap) {
  constexpr std::string_view kProgram = R"(
#[tabulate(domains=`map(u8:0..u8:3, |i: u8| { u8:2 * i })`)]
fn plus_one(x: u8) -> u8 {
    x + u8:1
}
)";
  XLS_ASSERT_OK_AND_ASSIGN(std::string md,
                           ParseAndTabulate(kProgram, "plus_one"));
  EXPECT_EQ(md,
            "| x   | result |\n"
            "| --- | ------ |\n"
            "| 0   | 1      |\n"
            "| 2   | 3      |\n"
            "| 4   | 5      |\n");
}

TEST(TabulateTest, DomainsCountMismatchError) {
  constexpr std::string_view kProgram = R"(
#[tabulate(domains=`u32:0..2`)]
fn add(x: u32, y: u32) -> u32 {
    x + y
}
)";
  EXPECT_THAT(ParseAndTabulate(kProgram, "add"),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("tabulate attribute has 1 domain argument, "
                                 "but function `add` has 2 parameters")));
}

TEST(TabulateTest, DomainsEmptyRangeError) {
  constexpr std::string_view kProgram = R"(
#[tabulate(domains=`u32:5..5`)]
fn id(x: u32) -> u32 {
    x
}
)";
  EXPECT_THAT(ParseAndTabulate(kProgram, "id"),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Parameter `x` has an empty domain")));
}

TEST(TabulateTest, DomainsHugeRangeExceedsLimitWithoutExpanding) {
  constexpr std::string_view kProgram = R"(
#[tabulate(domains=`u32:0..u32:0xFFFFFFFF`)]
fn id(x: u32) -> u32 {
    x
}
)";
  EXPECT_THAT(ParseAndTabulate(kProgram, "id"),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("exceeds maximum size")));
}

TEST(TabulateTest, UnconstrainedAndConstrainedTabulateInSameModule) {
  constexpr std::string_view kProgram = R"(
#[tabulate]
fn full_space(x: u2) -> u2 {
    x + u2:1
}

#[tabulate(domains=`u32:10..12`)]
fn constrained_space(x: u32) -> u32 {
    x + u32:1
}
)";
  ImportData import_data = CreateImportDataForTest();
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule tm,
      ParseAndTypecheck(kProgram, "test.x", "test", &import_data));
  XLS_ASSERT_OK_AND_ASSIGN(std::string full_md,
                           TabulateFunction(&import_data, tm, "full_space"));
  EXPECT_EQ(full_md,
            "| x   | result |\n"
            "| --- | ------ |\n"
            "| 0   | 1      |\n"
            "| 1   | 2      |\n"
            "| 2   | 3      |\n"
            "| 3   | 0      |\n");

  XLS_ASSERT_OK_AND_ASSIGN(
      std::string constrained_md,
      TabulateFunction(&import_data, tm, "constrained_space"));
  EXPECT_EQ(constrained_md,
            "| x   | result |\n"
            "| --- | ------ |\n"
            "| 10  | 11     |\n"
            "| 11  | 12     |\n");
}

TEST(TabulateTest, CombinedWithFuzzTestDivergingDomains) {
  constexpr std::string_view kProgram = R"(
struct Config {
    base: u2,
    scale: u32,
}

#[fuzz_test(domains=`Config { scale: u32:0..1000000 }, u32:0..1000000`)]
#[tabulate(domains=`Config { scale: [u32:10] }, u32:1..3`)]
fn compute(cfg: Config, offset: u32) -> u32 {
    (cfg.base as u32) * cfg.scale + offset
}

#[fuzz_test(domains=`[u2:1]`)]
#[tabulate]
fn full_tabulate_constrained_fuzz(x: u2) -> u2 {
    x + u2:1
}

#[tabulate(domains=`map(u32:0..3, |i: u32| i + u32:5)`)]
#[fuzz_test]
fn constrained_tabulate_unconstrained_fuzz(x: u32) -> u32 {
    x * u32:2
}
)";
  ImportData import_data = CreateImportDataForTest();
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule tm,
      ParseAndTypecheck(kProgram, "test.x", "test", &import_data));

  XLS_ASSERT_OK_AND_ASSIGN(std::string compute_md,
                           TabulateFunction(&import_data, tm, "compute"));
  EXPECT_EQ(compute_md,
            "| cfg                           | offset | result |\n"
            "| ----------------------------- | ------ | ------ |\n"
            "| Config { base: 0, scale: 10 } | 1      | 1      |\n"
            "| Config { base: 0, scale: 10 } | 2      | 2      |\n"
            "| Config { base: 1, scale: 10 } | 1      | 11     |\n"
            "| Config { base: 1, scale: 10 } | 2      | 12     |\n"
            "| Config { base: 2, scale: 10 } | 1      | 21     |\n"
            "| Config { base: 2, scale: 10 } | 2      | 22     |\n"
            "| Config { base: 3, scale: 10 } | 1      | 31     |\n"
            "| Config { base: 3, scale: 10 } | 2      | 32     |\n");

  XLS_ASSERT_OK_AND_ASSIGN(
      std::string full_tab_md,
      TabulateFunction(&import_data, tm, "full_tabulate_constrained_fuzz"));
  EXPECT_EQ(full_tab_md,
            "| x   | result |\n"
            "| --- | ------ |\n"
            "| 0   | 1      |\n"
            "| 1   | 2      |\n"
            "| 2   | 3      |\n"
            "| 3   | 0      |\n");

  XLS_ASSERT_OK_AND_ASSIGN(
      std::string constrained_tab_md,
      TabulateFunction(&import_data, tm,
                       "constrained_tabulate_unconstrained_fuzz"));
  EXPECT_EQ(constrained_tab_md,
            "| x   | result |\n"
            "| --- | ------ |\n"
            "| 5   | 10     |\n"
            "| 6   | 12     |\n"
            "| 7   | 14     |\n");
}

}  // namespace
}  // namespace xls::dslx
