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

#include "xls/ir/ir_parser.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>

#include "absl/status/status.h"
#include "absl/strings/ascii.h"
#include "absl/strings/str_format.h"
#include "absl/strings/substitute.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "xls/common/casts.h"
#include "xls/common/source_location.h"
#include "xls/common/status/matchers.h"
#include "xls/ir/bits.h"
#include "xls/ir/bits_ops.h"
#include "xls/ir/channel.h"
#include "xls/ir/channel.pb.h"
#include "xls/ir/channel_ops.h"
#include "xls/ir/nodes.h"
#include "xls/ir/op.h"
#include "xls/ir/package.h"
#include "xls/ir/value.h"

namespace xls {

using status_testing::StatusIs;
using ::testing::ElementsAre;
using ::testing::Eq;
using ::testing::HasSubstr;
using ::testing::IsEmpty;
using ::testing::Not;
using ::testing::Optional;

// EXPECTS that the two given strings are similar modulo extra whitespace.
static void ExpectStringsSimilar(
    std::string_view a, std::string_view b,
    xabsl::SourceLocation loc = xabsl::SourceLocation::current()) {
  std::string a_string(a);
  std::string b_string(b);
  testing::ScopedTrace trace(loc.file_name(), loc.line(),
                             "ExpectStringsSimilar failed");

  // After dumping remove any extra leading, trailing, and consecutive internal
  // whitespace verify that strings are the same.
  absl::RemoveExtraAsciiWhitespace(&a_string);
  absl::RemoveExtraAsciiWhitespace(&b_string);

  EXPECT_EQ(a_string, b_string);
}

// Parses the given string as a function, dumps the IR and compares that the
// dumped string and input string are the same modulo whitespace.
static void ParseFunctionAndCheckDump(
    std::string_view in,
    xabsl::SourceLocation loc = xabsl::SourceLocation::current()) {
  testing::ScopedTrace trace(loc.file_name(), loc.line(),
                             "ParseFunctionAndCheckDump failed");
  Package p("my_package");
  XLS_ASSERT_OK_AND_ASSIGN(auto function, Parser::ParseFunction(in, &p));
  ExpectStringsSimilar(function->DumpIr(), in, loc);
}

// Parses the given string as a package, dumps the IR and compares that the
// dumped string and input string are the same modulo whitespace.
static void ParsePackageAndCheckDump(
    std::string_view in,
    xabsl::SourceLocation loc = xabsl::SourceLocation::current()) {
  testing::ScopedTrace trace(loc.file_name(), loc.line(),
                             "ParsePackageAndCheckDump failed");
  XLS_ASSERT_OK_AND_ASSIGN(auto package, Parser::ParsePackage(in));
  ExpectStringsSimilar(package->DumpIr(), in, loc);
}

TEST(IrParserTest, ParseBitsLiteral) {
  ParseFunctionAndCheckDump(R"(fn f() -> bits[37] {
  ret literal.1: bits[37] = literal(value=42, id=1)
})");
}

TEST(IrParserTest, ParseTokenLiteral) {
  ParseFunctionAndCheckDump(R"(fn f() -> token {
  ret literal.1: token = literal(value=token, id=1)
})");
}

TEST(IrParserTest, ParseWideLiteral) {
  ParseFunctionAndCheckDump(R"(fn f() -> bits[96] {
  ret literal.1: bits[96] = literal(value=0xaaaa_bbbb_1234_5678_90ab_cdef, id=1)
})");
}

TEST(IrParserTest, ParseVariousBitsLiterals) {
  const char tmplate[] = R"(fn f() -> bits[$0] {
  ret literal.1: bits[$0] = literal(value=$1)
})";
  struct TestCase {
    int64_t width;
    std::string literal;
    Bits expected;
  };
  for (const TestCase& test_case :
       {TestCase{1, "-1", UBits(1, 1)}, TestCase{8, "-1", UBits(0xff, 8)},
        TestCase{8, "-128", UBits(0x80, 8)},
        TestCase{32, "0xffffffff", UBits(0xffffffffULL, 32)},
        TestCase{32, "-0x80000000", UBits(0x80000000ULL, 32)},
        TestCase{32, "0x80000000", UBits(0x80000000ULL, 32)}}) {
    Package p("my_package");
    XLS_ASSERT_OK_AND_ASSIGN(
        auto function,
        Parser::ParseFunction(
            absl::Substitute(tmplate, test_case.width, test_case.literal), &p));
    EXPECT_EQ(function->return_value()->As<Literal>()->value().bits(),
              test_case.expected);
  }
}

TEST(IrParserTest, ParseTupleLiterals) {
  std::string text = R"(fn f() -> (bits[16], bits[96]) {
  ret literal.1: (bits[16], bits[96]) = literal(value=(1234, 0xdeadbeefdeadbeefdeadbeef))
})";
  Package p("my_package");
  XLS_ASSERT_OK_AND_ASSIGN(auto function, Parser::ParseFunction(text, &p));
  Bits deadbeef = UBits(0xdeadbeefULL, 32);
  EXPECT_EQ(
      function->return_value()->As<Literal>()->value(),
      Value::Tuple({Value(UBits(1234, 16)),
                    Value(bits_ops::Concat({deadbeef, deadbeef, deadbeef}))}));
}

TEST(IrParserTest, ParseVariousLiteralsTooFewBits) {
  const char tmplate[] = R"(fn f() -> bits[$0] {
  ret literal.1: bits[$0] = literal(value=$1)
})";
  struct TestCase {
    int64_t width;
    std::string literal;
  };
  for (const TestCase& test_case :
       {TestCase{1, "-2"}, TestCase{3, "42"}, TestCase{3, "-5"},
        TestCase{8, "-129"}, TestCase{64, "0x1_ffff_ffff_ffff_ffff"},
        TestCase{32, "-0x80000001"}}) {
    Package p("my_package");
    EXPECT_THAT(
        Parser::ParseFunction(
            absl::Substitute(tmplate, test_case.width, test_case.literal), &p)
            .status(),
        StatusIs(absl::StatusCode::kInvalidArgument,
                 HasSubstr("is not representable")));
  }
}

TEST(IrParserTest, DuplicateKeywordArgs) {
  Package p("my_package");
  const std::string input =
      R"(fn f() -> bits[37] {
  ret literal.1: bits[37] = literal(value=42, value=123)
})";
  EXPECT_THAT(Parser::ParseFunction(input, &p).status(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Duplicate keyword argument `value`")));
}

TEST(IrParserTest, WrongDeclaredNodeType) {
  Package p("my_package");
  const std::string input =
      R"(
fn less_than(a: bits[32], b: bits[32]) -> bits[1] {
  ret ult.3: bits[32] = ult(a, b)
})";
  EXPECT_THAT(
      Parser::ParseFunction(input, &p).status(),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          HasSubstr(
              "Declared type bits[32] does not match expected type bits[1]")));
}

TEST(IrParserTest, WrongFunctionReturnType) {
  Package p("my_package");
  const std::string input =
      R"(
fn less_than(a: bits[32], b: bits[32]) -> bits[32] {
  ret ult.3: bits[1] = ult(a, b)
})";
  EXPECT_THAT(Parser::ParseFunction(input, &p).status(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Type of return value bits[1] does not match "
                                 "declared function return type bits[32]")));
}

TEST(IrParserTest, MissingMandatoryKeyword) {
  Package p("my_package");
  const std::string input =
      R"(fn f() -> bits[37] {
  ret literal.1: bits[37] = literal()
})";
  EXPECT_THAT(
      Parser::ParseFunction(input, &p).status(),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("Mandatory keyword argument `value` not found")));
}

TEST(IrParserTest, ParsePosition) {
  ParseFunctionAndCheckDump(
      R"(
fn f(x: bits[42], y: bits[42]) -> bits[42] {
  ret and.1: bits[42] = and(x, y, id=1, pos=[(0,1,3)])
}
)");
}

TEST(IrParserTest, UndefinedOperand) {
  Package p("my_package");
  std::string input =
      R"(
fn f(x: bits[42]) -> bits[42] {
  ret and.1: bits[42] = and(x, z)
}
)";
  EXPECT_THAT(Parser::ParseFunction(input, &p).status(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("was not previously defined: \"z\"")));
}

TEST(IrParserTest, InvalidOp) {
  Package p("my_package");
  std::string input =
      R"(
fn f(x: bits[42]) -> bits[42] {
  ret foo_op.1: bits[42] = foo_op(x, z)
}
)";
  EXPECT_THAT(
      Parser::ParseFunction(input, &p).status(),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          HasSubstr("Unknown operation for string-to-op conversion: foo_op")));
}

TEST(IrParserTest, PositionalArgumentAfterKeywordArgument) {
  Package p("my_package");
  std::string input =
      R"(
fn f(x: bits[42], y: bits[42]) -> bits[42] {
  ret and.1: bits[42] = and(x, pos=[(0,1,3)], y)
}
)";
  EXPECT_THAT(Parser::ParseFunction(input, &p).status(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Expected token of type \"=\"")));
}

TEST(IrParserTest, ExtraOperands) {
  Package p("my_package");
  const std::string input =
      R"(
fn f(x: bits[42], y: bits[42], z: bits[42]) -> bits[42] {
  ret add.1: bits[42] = add(x, y, z)
}
})";
  EXPECT_THAT(Parser::ParseFunction(input, &p).status(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Expected 2 operands, got 3")));
}

TEST(IrParserTest, TooFewOperands) {
  Package p("my_package");
  const std::string input =
      R"(
fn f(x: bits[42]) -> bits[42] {
  ret add.1: bits[42] = add(x, id=1)
}
})";
  EXPECT_THAT(Parser::ParseFunction(input, &p).status(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Expected 2 operands, got 1")));
}

TEST(IrParserTest, DuplicateName) {
  Package p("my_package");
  const std::string input =
      R"(
fn f(x: bits[42]) -> bits[42] {
 and.1: bits[42] = and(x, x, id=1)
 ret and.1: bits[42] = and(and.1, and.1)
}
})";
  EXPECT_THAT(Parser::ParseFunction(input, &p).status(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Name 'and.1' has already been defined")));
}

TEST(IrParserTest, ParseNode) {
  ParseFunctionAndCheckDump(
      R"(
fn f() -> bits[32] {
  literal.1: bits[32] = literal(value=3, id=1, pos=[(0,3,11)])
  ret sub.2: bits[32] = sub(literal.1, literal.1, id=2)
})");
}

TEST(IrParserTest, ParseFunction) {
  ParseFunctionAndCheckDump(
      R"(
fn simple_arith(a: bits[32], b: bits[32]) -> bits[32] {
  ret sub.3: bits[32] = sub(a, b, id=3)
})");
}

TEST(IrParserTest, ParseFunctionWithFFI) {
  ParsePackageAndCheckDump(
      R"(package test

#[ffi_proto("""code_template: "verilog_module {fn} (.in({a}));"
""")]
fn fun(a: bits[23]) -> bits[42] {
  ret umul.1: bits[42] = umul(a, a, id=1)
})");
}

TEST(IrParserTest, ParseFunctionWithNewlineInFFI) {
  ParsePackageAndCheckDump(
      R"(package test

#[ffi_proto("""code_template: "verilog_module {fn} (\n.in({a})\n);"
""")]
fn fun(a: bits[23]) -> bits[42] {
  ret umul.1: bits[42] = umul(a, a, id=1)
})");
}

TEST(IrParserTest, ParseULessThan) {
  ParseFunctionAndCheckDump(
      R"(
fn less_than(a: bits[32], b: bits[32]) -> bits[1] {
  ret ult.3: bits[1] = ult(a, b, id=3)
})");
}

TEST(IrParserTest, ParseSLessThan) {
  ParseFunctionAndCheckDump(
      R"(
fn less_than(a: bits[32], b: bits[32]) -> bits[1] {
  ret slt.3: bits[1] = slt(a, b, id=3)
})");
}

TEST(IrParserTest, ParseTwoPlusTwo) {
  std::string program = R"(
fn two_plus_two() -> bits[32] {
  literal.1: bits[32] = literal(value=2, id=1)
  literal.2: bits[32] = literal(value=2, id=2)
  ret add.3: bits[32] = add(literal.1, literal.2, id=3)
}
)";
  ParseFunctionAndCheckDump(program);
}

TEST(IrParserTest, ParseTwoPlusThreeCustomIdentifiers) {
  std::string program = R"(
fn two_plus_two() -> bits[32] {
  literal.1: bits[32] = literal(value=2, id=1)
  literal.2: bits[32] = literal(value=3, id=2)
  ret add.3: bits[32] = add(literal.1, literal.2, id=3)
}
)";
  Package p("my_package");
  XLS_ASSERT_OK_AND_ASSIGN(auto function, Parser::ParseFunction(program, &p));
  // The nodes are given canonical names when we dump because we don't note the
  // original names.
  ExpectStringsSimilar(function->DumpIr(), R"(
fn two_plus_two() -> bits[32] {
  literal.1: bits[32] = literal(value=2, id=1)
  literal.2: bits[32] = literal(value=3, id=2)
  ret add.3: bits[32] = add(literal.1, literal.2, id=3)
})");
}

TEST(IrParserTest, CountedFor) {
  std::string program = R"(
package CountedFor

fn body(x: bits[11], y: bits[11]) -> bits[11] {
  ret add.3: bits[11] = add(x, y, id=3)
}

fn main() -> bits[11] {
  literal.4: bits[11] = literal(value=0, id=4)
  ret counted_for.5: bits[11] = counted_for(literal.4, trip_count=7, stride=1, body=body, id=5)
}
)";
  ParsePackageAndCheckDump(program);
}

TEST(IrParserTest, CountedForMissingBody) {
  std::string program = R"(
package CountedForMissingBody

fn body(i: bits[11], x: bits[11], y: bits[11]) -> bits[11] {
  ret add.3: bits[11] = add(x, y, id=3)
}

fn main() -> bits[11] {
  literal.4: bits[11] = literal(value=0, id=4)
  ret counted_for.5: bits[11] = counted_for(literal.4, trip_count=7, stride=1)
}
)";
  EXPECT_THAT(
      Parser::ParsePackage(program).status(),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("Mandatory keyword argument `body` not found")));
}

TEST(IrParserTest, CountedForInvariantArgs) {
  std::string program = R"(
package CountedFor

fn body(i: bits[11], x: bits[11], y: bits[11]) -> bits[11] {
  ret add.3: bits[11] = add(x, y, id=3)
}

fn main() -> bits[11] {
  literal.4: bits[11] = literal(value=0, id=4)
  literal.5: bits[11] = literal(value=1, id=5)
  ret counted_for.6: bits[11] = counted_for(literal.4, trip_count=7, stride=1, body=body, invariant_args=[literal.5], id=6)
}
)";
  ParsePackageAndCheckDump(program);
}
TEST(IrParserTest, CountedForBodyParamCountTooMany0) {
  std::string program = R"(
package test

fn loop_fn(i: bits[4], data: bits[16], x: bits[16], y: bits[16]) -> bits[16] {
  ret literal.10: bits[16] = literal(value=0)
}

fn f(x: bits[16]) -> bits[16] {
  literal.2: bits[16] = literal(value=2, id=2)
  literal.3: bits[16] = literal(value=3)

  ret counted_for.100: bits[16] = counted_for(x, trip_count=2, body=loop_fn,
      invariant_args=[literal.2])
}
)";

  EXPECT_THAT(Parser::ParsePackage(program).status(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("counted_for body should have 3 parameters, "
                                 "got 4 instead")));
}

TEST(IrParserTest, CountedForBodyParamCountTooMany1) {
  std::string program = R"(
package test

fn loop_fn(i: bits[4], data: bits[16], x: bits[16]) -> bits[16] {
  ret literal.10: bits[16] = literal(value=0)
}

fn f(x: bits[16]) -> bits[16] {
  literal.2: bits[16] = literal(value=2)

  ret counted_for.100: bits[16] = counted_for(x, trip_count=2, body=loop_fn)
}
)";

  EXPECT_THAT(Parser::ParsePackage(program).status(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("counted_for body should have 2 parameters, "
                                 "got 3 instead")));
}

TEST(IrParserTest, CountedForBodyParamCountTooFew0) {
  std::string program = R"(
package test

fn loop_fn(i: bits[4], data: bits[16], x: bits[16], y: bits[16]) -> bits[16] {
  ret literal.10: bits[16] = literal(value=0)
}

fn f(x: bits[16]) -> bits[16] {
  literal.2: bits[16] = literal(value=2, id=2)
  literal.3: bits[16] = literal(value=3, id=3)
  literal.4: bits[16] = literal(value=4)

  ret counted_for.100: bits[16] = counted_for(x, trip_count=2, body=loop_fn,
      invariant_args=[literal.2, literal.3, literal.4])
}
)";

  EXPECT_THAT(Parser::ParsePackage(program).status(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("counted_for body should have 5 parameters, "
                                 "got 4 instead")));
}

TEST(IrParserTest, CountedForBodyParamCountTooFew1) {
  std::string program = R"(
package test

fn loop_fn(i: bits[4], data: bits[16]) -> bits[16] {
  ret literal.10: bits[16] = literal(value=0)
}

fn f(x: bits[16]) -> bits[16] {
  literal.2: bits[16] = literal(value=2)

  ret counted_for.100: bits[16] = counted_for(x, trip_count=2, body=loop_fn,
      invariant_args=[literal.2])
}
)";

  EXPECT_THAT(Parser::ParsePackage(program).status(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("counted_for body should have 3 parameters, "
                                 "got 2 instead")));
}

TEST(IrParserTest, CountedForBodyParamCountTooFew2) {
  std::string program = R"(
package test

fn loop_fn() -> bits[16] {
  ret literal.10: bits[16] = literal(value=0)
}

fn f(x: bits[16]) -> bits[16] {
  literal.2: bits[16] = literal(value=2)

  ret counted_for.100: bits[16] = counted_for(x, trip_count=2, body=loop_fn)
}
)";

  EXPECT_THAT(Parser::ParsePackage(program).status(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("counted_for body should have 2 parameters, "
                                 "got 0 instead")));
}

TEST(IrParserTest, CountedForBodyBitWidthSufficient0) {
  std::string program = R"(
package test

fn loop_fn(i: bits[4], data: bits[16]) -> bits[16] {
  ret literal.10: bits[16] = literal(value=0, id=10)
}

fn f(x: bits[16]) -> bits[16] {
  literal.2: bits[16] = literal(value=2, id=2)

  ret counted_for.100: bits[16] = counted_for(x, trip_count=16, stride=1,
                                              body=loop_fn, id=100)
}
)";

  ParsePackageAndCheckDump(program);
}

TEST(IrParserTest, CountedForBodyBitWidthZeroIteration) {
  std::string program = R"(
package test

fn loop_fn(i: bits[1], data: bits[16]) -> bits[16] {
  ret literal.10: bits[16] = literal(value=0, id=10)
}

fn f(x: bits[16]) -> bits[16] {
  literal.2: bits[16] = literal(value=2, id=2)

  ret counted_for.100: bits[16] = counted_for(x, trip_count=0, stride=1,
                                              body=loop_fn, id=100)
}
)";

  ParsePackageAndCheckDump(program);
}

TEST(IrParserTest, CountedForBodyBitWidthOneIteration) {
  std::string program = R"(
package test

fn loop_fn(i: bits[1], data: bits[16]) -> bits[16] {
  ret literal.10: bits[16] = literal(value=0, id=10)
}

fn f(x: bits[16]) -> bits[16] {
  literal.2: bits[16] = literal(value=2, id=2)

  ret counted_for.100: bits[16] = counted_for(x, trip_count=1, stride=1,
                                              body=loop_fn, id=100)
}
)";

  ParsePackageAndCheckDump(program);
}

TEST(IrParserTest, CountedForBodyBitWidthInsufficient) {
  std::string program = R"(
package test

fn loop_fn(i: bits[4], data: bits[16]) -> bits[16] {
  ret literal.10: bits[16] = literal(value=0)
}

fn f(x: bits[16]) -> bits[16] {
  literal.2: bits[16] = literal(value=2, id=2)

  ret counted_for.100: bits[16] = counted_for(x, trip_count=17, body=loop_fn, id=100)
}
)";

  EXPECT_THAT(
      Parser::ParsePackage(program).status(),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("counted_for body should have bits[N] type, where "
                         "N >= 5, got bits[4]")));
}

TEST(IrParserTest, CountedForBodyBitWidthInsufficientWithStride) {
  std::string program = R"(
package test

fn loop_fn(i: bits[4], data: bits[16]) -> bits[16] {
  ret literal.10: bits[16] = literal(value=0, id=10)
}

fn f(x: bits[16]) -> bits[16] {
  literal.2: bits[16] = literal(value=2, id=2)

  ret counted_for.100: bits[16] = counted_for(x, trip_count=16, stride=2,
                                              body=loop_fn)
}
)";

  EXPECT_THAT(
      Parser::ParsePackage(program).status(),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("counted_for body should have bits[N] type, where "
                         "N >= 5, got bits[4]")));
}

TEST(IrParserTest, CountedForBodyBitWidthTypeMismatch0) {
  std::string program = R"(
package test

fn loop_fn(i: bits[4][1], data: bits[16]) -> bits[16] {
  ret literal.10: bits[16] = literal(value=0, id=10)
}

fn f(x: bits[16]) -> bits[16] {
  literal.2: bits[16] = literal(value=2, id=2)

  ret counted_for.100: bits[16] = counted_for(x, trip_count=0, body=loop_fn, id=100)
}
)";

  EXPECT_THAT(
      Parser::ParsePackage(program).status(),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("counted_for body should have bits[N] type, where "
                         "N >= 1, got bits[4][1]")));
}

TEST(IrParserTest, CountedForBodyBitWidthTypeMismatch1) {
  std::string program = R"(
package test

fn loop_fn(i: (bits[4]), data: bits[16]) -> bits[16] {
  ret literal.10: bits[16] = literal(value=0, id=10)
}

fn f(x: bits[16]) -> bits[16] {
  literal.2: bits[16] = literal(value=2, id=2)

  ret counted_for.100: bits[16] = counted_for(x, trip_count=1, body=loop_fn, id=100)
}
)";

  EXPECT_THAT(
      Parser::ParsePackage(program).status(),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("counted_for body should have bits[N] type, where "
                         "N >= 1, got (bits[4])")));
}

TEST(IrParserTest, CountedForBodyDataTypeMismatch) {
  std::string program = R"(
package test

fn loop_fn(i: bits[4], data: bits[13]) -> bits[16] {
  ret literal.10: bits[16] = literal(value=0, id=10)
}

fn f(x: bits[16]) -> bits[16] {
  literal.2: bits[16] = literal(value=2, id=2)

  ret counted_for.100: bits[16] = counted_for(x, trip_count=1, body=loop_fn, id=100)
}
)";

  EXPECT_THAT(
      Parser::ParsePackage(program).status(),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("should have bits[16] type, got bits[13] instead")));
}

TEST(IrParserTest, CountedForReturnTypeMismatch) {
  std::string program = R"(
package test

fn loop_fn(i: bits[4], data: bits[16]) -> bits[15] {
  ret literal.10: bits[15] = literal(value=0, id=10)
}

fn f(x: bits[16]) -> bits[16] {
  literal.2: bits[16] = literal(value=2, id=2)

  ret counted_for.100: bits[16] = counted_for(x, trip_count=1, body=loop_fn, id=100)
}
)";

  EXPECT_THAT(
      Parser::ParsePackage(program).status(),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("should have bits[16] type, got bits[15] instead")));
}

TEST(IrParserTest, CountedForBodyInvariantArgTypeMismatch0) {
  std::string program = R"(
package test

fn loop_fn(i: bits[4], data: bits[16],
           x: bits[4], y: (bits[4], bits[4]), z: bits[4][1]) -> bits[16] {
  ret literal.10: bits[16] = literal(value=0, id=10)
}

fn f(x: bits[16]) -> bits[16] {
  literal.2: bits[4] = literal(value=2, id=2)
  literal.3: bits[4] = literal(value=3, id=3)
  literal.4: bits[4] = literal(value=4, id=4)

  literal.112: bits[4] = literal(value=1, id=112)
  tuple.113: (bits[4], bits[4]) = tuple(literal.2, literal.3, id=113)
  array.114: bits[4][1] = array(literal.4, id=114)

  ret counted_for.200: bits[16] = counted_for(x, trip_count=2, body=loop_fn,
      invariant_args=[literal.112, array.114, array.114])
}
)";

  EXPECT_THAT(
      Parser::ParsePackage(program).status(),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("Parameter 3 (y) of function loop_fn used as "
                         "counted_for body should have bits[4][1] type")));
}

TEST(IrParserTest, CountedForBodyInvariantArgTypeMismatch1) {
  std::string program = R"(
package test

fn loop_fn(i: bits[4], data: bits[16],
           x: bits[4], y: (bits[4], bits[4]), z: bits[4][1]) -> bits[16] {
  ret literal.10: bits[16] = literal(value=0)
}

fn f(x: bits[16]) -> bits[16] {
  literal.2: bits[4] = literal(value=2, id=2)
  literal.3: bits[4] = literal(value=3, id=3)
  literal.4: bits[4] = literal(value=4)

  literal.112: bits[4] = literal(value=1, id=112)
  tuple.113: (bits[4], bits[4]) = tuple(literal.2, literal.3, id=113)
  array.114: bits[4][1] = array(literal.4)

  ret counted_for.200: bits[16] = counted_for(x, trip_count=2, body=loop_fn,
      invariant_args=[literal.112, tuple.113, literal.112])
}
)";

  EXPECT_THAT(Parser::ParsePackage(program).status(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Parameter 4 (z) of function loop_fn used as "
                                 "counted_for body should have bits[4] type")));
}

TEST(IrParserTest, ParseBitSlice) {
  std::string input = R"(
fn bitslice(x: bits[32]) -> bits[14] {
  ret bit_slice.1: bits[14] = bit_slice(x, start=7, width=14, id=1)
}
)";
  ParseFunctionAndCheckDump(input);
}

TEST(IrParserTest, ParseDynamicBitSlice) {
  std::string input = R"(
fn dynamicbitslice(x: bits[32], y: bits[32]) -> bits[14] {
  ret dynamic_bit_slice.1: bits[14] = dynamic_bit_slice(x, y, width=14, id=1)
}
)";
  ParseFunctionAndCheckDump(input);
}

TEST(IrParserTest, ParseAfterAllEmpty) {
  std::string input = R"(
fn after_all_func() -> token {
  ret after_all.1: token = after_all(id=1)
}
)";
  ParseFunctionAndCheckDump(input);
}

TEST(IrParserTest, ParseAfterAllMany) {
  std::string input = R"(
fn after_all_func() -> token {
  after_all.1: token = after_all(id=1)
  after_all.2: token = after_all(id=2)
  ret after_all.3: token = after_all(after_all.1, after_all.2, id=3)
}
)";
  ParseFunctionAndCheckDump(input);
}

TEST(IrParserTest, ParseAfterAllNonToken) {
  Package p("my_package");
  std::string input = R"(
fn after_all_func() -> token {
  after_all.1: token = after_all(id=1)
  after_all.2: token = after_all(id=2)
  ret after_all.3: bits[2] = after_all(after_all.1, after_all.2, id=3)
}
)";
  EXPECT_THAT(Parser::ParseFunction(input, &p).status(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Expected token type @")));
}

TEST(IrParserTest, ParseMinDelay) {
  std::string input = R"(
fn after_all_func() -> token {
  after_all.1: token = after_all(id=1)
  min_delay.2: token = min_delay(after_all.1, delay=3, id=2)
  ret min_delay.3: token = min_delay(min_delay.2, delay=0, id=3)
}
)";
  ParseFunctionAndCheckDump(input);
}

TEST(IrParserTest, ParseMinDelayNonToken) {
  Package p("my_package");
  std::string input = R"(
fn after_all_func() -> token {
  after_all.1: token = after_all(id=1)
  ret min_delay.2: bits[2] = min_delay(after_all.1, delay=3, id=2)
}
)";
  EXPECT_THAT(Parser::ParseFunction(input, &p).status(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Expected token type @")));
}

TEST(IrParserTest, ParseMinDelayNegative) {
  Package p("my_package");
  std::string input = R"(
fn after_all_func() -> token {
  after_all.1: token = after_all(id=1)
  ret min_delay.2: token = min_delay(after_all.1, delay=-1, id=2)
}
)";
  EXPECT_THAT(Parser::ParseFunction(input, &p).status(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Delay cannot be negative")));
}

TEST(IrParserTest, ParseArray) {
  std::string input = R"(
fn array_and_array(x: bits[32], y: bits[32], z: bits[32]) -> bits[32][3] {
  ret array.1: bits[32][3] = array(x, y, z, id=1)
}
)";
  ParseFunctionAndCheckDump(input);
}

TEST(IrParserTest, ParseReverse) {
  std::string input = R"(
fn reverse(x: bits[32]) -> bits[32] {
  ret reverse.1: bits[32] = reverse(x, id=1)
}
)";
  ParseFunctionAndCheckDump(input);
}

TEST(IrParserTest, ParseArrayOfTuples) {
  std::string input = R"(
fn array_and_array(x: (bits[32], bits[1]), y: (bits[32], bits[1])) -> (bits[32], bits[1])[3] {
  ret array.1: (bits[32], bits[1])[3] = array(x, y, x, id=1)
}
)";
  ParseFunctionAndCheckDump(input);
}

TEST(IrParserTest, ParseNestedBitsArrayIndex) {
  std::string input = R"(
fn array_and_array(p: bits[2][5][4][42], q: bits[32], r: bits[2]) -> bits[2][5] {
  ret array_index.1: bits[2][5] = array_index(p, indices=[q, r], id=1)
}
)";
  ParseFunctionAndCheckDump(input);
}

TEST(IrParserTest, ParseNestedBitsArrayUpdate) {
  std::string input = R"(
fn array_and_array(p: bits[2][5][4][42], q: bits[32], v: bits[2][5][4]) -> bits[2][5][4][42] {
  ret array_update.1: bits[2][5][4][42] = array_update(p, v, indices=[q], id=1)
}
)";
  ParseFunctionAndCheckDump(input);
}

TEST(IrParserTest, DifferentWidthMultiplies) {
  std::string input = R"(
fn multiply(x: bits[32], y: bits[7]) -> bits[42] {
  ret umul.1: bits[42] = umul(x, y, id=1)
}
)";
  ParseFunctionAndCheckDump(input);
}

TEST(IrParserTest, EmptyBitsBounds) {
  Package p("my_package");
  std::string input = R"(fn f() -> bits[] {
  ret literal.1: bits[] = literal(value=0, id=1)
})";
  EXPECT_THAT(Parser::ParseFunction(input, &p).status(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Expected token of type \"literal\"")));
}

TEST(IrParserTest, ParseSingleEmptyPackage) {
  std::string input = R"(package EmptyPackage)";
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> package,
                           Parser::ParsePackage(input));
  EXPECT_EQ(package->name(), "EmptyPackage");
  EXPECT_EQ(0, package->functions().size());

  ParsePackageAndCheckDump(input);
}

TEST(IrParserTest, ParseSingleFunctionPackage) {
  std::string input = R"(package SingleFunctionPackage

fn two_plus_two() -> bits[32] {
  literal.1: bits[32] = literal(value=2, id=1)
  literal.2: bits[32] = literal(value=2, id=2)
  ret add.3: bits[32] = add(literal.1, literal.2, id=3)
}
)";
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> package,
                           Parser::ParsePackage(input));
  EXPECT_EQ(package->name(), "SingleFunctionPackage");
  EXPECT_EQ(1, package->functions().size());
  Function* func = package->functions().front().get();
  EXPECT_EQ(func->name(), "two_plus_two");

  ParsePackageAndCheckDump(input);
}

TEST(IrParserTest, ParseMultiFunctionPackage) {
  std::string input = R"(package MultiFunctionPackage

fn two_plus_two() -> bits[32] {
  literal.1: bits[32] = literal(value=2, id=1)
  literal.2: bits[32] = literal(value=2, id=2)
  ret add.3: bits[32] = add(literal.1, literal.2, id=3)
}

fn seven_and_five() -> bits[32] {
  literal.4: bits[32] = literal(value=7, id=4)
  literal.5: bits[32] = literal(value=5, id=5)
  ret and.6: bits[32] = and(literal.4, literal.5, id=6)
}
)";

  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> package,
                           Parser::ParsePackage(input));
  EXPECT_EQ(package->name(), "MultiFunctionPackage");
  EXPECT_EQ(2, package->functions().size());
  EXPECT_EQ(package->functions()[0]->name(), "two_plus_two");
  EXPECT_EQ(package->functions()[1]->name(), "seven_and_five");

  ParsePackageAndCheckDump(input);
}

TEST(IrParserTest, ParsePackageWithError) {
  std::string input = R"(package MultiFunctionPackage

Garbage
)";
  EXPECT_THAT(Parser::ParsePackage(input).status(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Expected attribute or declaration")));
}

TEST(IrParserTest, ParseEmptyStringAsPackage) {
  EXPECT_THAT(Parser::ParsePackage("").status(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Expected token, but found EOF")));
}

TEST(IrParserTest, ParsePackageWithMissingPackageLine) {
  std::string input = R"(fn two_plus_two() -> bits[32] {
  literal.1: bits[32] = literal(value=2, id=1)
  literal.2: bits[32] = literal(value=2, id=2)
  ret add.3: bits[32] = add(literal.1, literal.2, id=3)
}
)";
  absl::Status status = Parser::ParsePackage(input).status();
  EXPECT_THAT(Parser::ParsePackage(input).status(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Expected 'package' keyword")));
}

TEST(IrParserTest, ParseBinaryConcat) {
  std::string input = R"(package p
fn concat_wrapper(x: bits[31], y: bits[1]) -> bits[32] {
  ret concat.1: bits[32] = concat(x, y, id=1)
}
)";
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> p,
                           Parser::ParsePackage(input));
  ASSERT_EQ(1, p->functions().size());
  std::unique_ptr<Function>& f = p->functions()[0];
  EXPECT_EQ(f->return_value()->op(), Op::kConcat);
  EXPECT_FALSE(f->return_value()->Is<BinOp>());
  EXPECT_TRUE(f->return_value()->Is<Concat>());
  EXPECT_EQ(p->GetBitsType(32), f->return_value()->GetType());
}

TEST(IrParserTest, ParseNaryConcat) {
  std::string input = R"(package p
fn concat_wrapper(x: bits[31], y: bits[1]) -> bits[95] {
  ret concat.1: bits[95] = concat(x, y, x, x, y, id=1)
}
)";
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> p,
                           Parser::ParsePackage(input));
  ASSERT_EQ(1, p->functions().size());
  std::unique_ptr<Function>& f = p->functions()[0];
  EXPECT_EQ(f->return_value()->op(), Op::kConcat);
  EXPECT_TRUE(f->return_value()->Is<Concat>());
  EXPECT_EQ(p->GetBitsType(95), f->return_value()->GetType());
}

TEST(IrParserTest, ParseMap) {
  std::string input = R"(
package SimpleMap

fn to_apply(element: bits[42]) -> bits[1] {
  literal.2: bits[42] = literal(value=10, id=2)
  ret ult.3: bits[1] = ult(element, literal.2, id=3)
}

fn main(input: bits[42][123]) -> bits[1][123] {
  ret map.5: bits[1][123] = map(input, to_apply=to_apply, id=5)
}
)";
  ParsePackageAndCheckDump(input);
}

TEST(IrParserTest, ParseBinarySel) {
  const std::string input = R"(
package ParseSel

fn sel_wrapper(x: bits[1], y: bits[32], z: bits[32]) -> bits[32] {
  ret sel.1: bits[32] = sel(x, cases=[y, z], id=1)
}
  )";
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> pkg,
                           Parser::ParsePackage(input));
  ASSERT_EQ(pkg->functions().size(), 1);
  Function& f = *(pkg->functions()[0]);
  EXPECT_EQ(f.return_value()->op(), Op::kSel);
  EXPECT_FALSE(f.return_value()->Is<BinOp>());
  EXPECT_TRUE(f.return_value()->Is<Select>());
  EXPECT_EQ(f.return_value()->GetType(), pkg->GetBitsType(32));

  ParsePackageAndCheckDump(input);
}

TEST(IrParserTest, ParseTernarySelectWithDefault) {
  const std::string input = R"(
package ParseSel

fn sel_wrapper(p: bits[2], x: bits[32], y: bits[32], z: bits[32]) -> bits[32] {
  literal.1: bits[32] = literal(value=0, id=1)
  ret sel.2: bits[32] = sel(p, cases=[x, y, z], default=literal.1, id=2)
}
  )";
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> pkg,
                           Parser::ParsePackage(input));
  ASSERT_EQ(pkg->functions().size(), 1);
  Function& f = *(pkg->functions()[0]);
  EXPECT_EQ(f.return_value()->op(), Op::kSel);
  EXPECT_FALSE(f.return_value()->Is<BinOp>());
  EXPECT_TRUE(f.return_value()->Is<Select>());
  EXPECT_EQ(f.return_value()->GetType(), pkg->GetBitsType(32));

  ParsePackageAndCheckDump(input);
}

TEST(IrParserTest, ParseOneHotLsbPriority) {
  const std::string input = R"(
package ParseOneHot

fn sel_wrapper(x: bits[42]) -> bits[43] {
  ret one_hot.1: bits[43] = one_hot(x, lsb_prio=true, id=1)
}
  )";
  ParsePackageAndCheckDump(input);
}

TEST(IrParserTest, ParseOneHotMsbPriority) {
  const std::string input = R"(
package ParseOneHot

fn sel_wrapper(x: bits[42]) -> bits[43] {
  ret one_hot.1: bits[43] = one_hot(x, lsb_prio=false, id=1)
}
  )";
  ParsePackageAndCheckDump(input);
}

TEST(IrParserTest, ParseOneHotSelect) {
  const std::string input = R"(
package ParseOneHotSel

fn sel_wrapper(p: bits[3], x: bits[32], y: bits[32], z: bits[32]) -> bits[32] {
  ret one_hot_sel.1: bits[32] = one_hot_sel(p, cases=[x, y, z], id=1)
}
  )";
  ParsePackageAndCheckDump(input);
}

TEST(IrParserTest, ParsePrioritySelect) {
  const std::string input = R"(
package ParsePrioritySel

fn sel_wrapper(p: bits[3], x: bits[32], y: bits[32], z: bits[32]) -> bits[32] {
  ret priority_sel.1: bits[32] = priority_sel(p, cases=[x, y, z], id=1)
}
  )";
  ParsePackageAndCheckDump(input);
}

TEST(IrParserTest, ParseParamReturn) {
  std::string input = R"(
package ParseParamReturn

fn simple_neg(x: bits[2]) -> bits[2] {
  ret x: bits[2] = param(name=x)
}
)";
  ParsePackageAndCheckDump(input);
}

TEST(IrParserTest, ParseInvoke) {
  const std::string input = R"(package foobar

fn bar(x: bits[32], y: bits[32]) -> bits[32] {
  ret add.1: bits[32] = add(x, y, id=1)
}

fn foo(x: bits[32]) -> bits[32] {
  literal.2: bits[32] = literal(value=5, id=2)
  ret invoke.3: bits[32] = invoke(x, literal.2, to_apply=bar, id=3)
}
)";
  ParsePackageAndCheckDump(input);
}

TEST(IrParserTest, ParseAssert) {
  const std::string input = R"(package foobar

fn bar(tkn: token, cond: bits[1]) -> token {
  ret assert.1: token = assert(tkn, cond, message="The foo is bar", id=1)
}
)";
  ParsePackageAndCheckDump(input);
}

TEST(IrParserTest, ParseAssertWithLabel) {
  const std::string input = R"(package foobar

fn bar(tkn: token, cond: bits[1]) -> token {
  ret assert.1: token = assert(tkn, cond, message="The foo is bar", label="assert_label", id=1)
}
)";
  ParsePackageAndCheckDump(input);
}

TEST(IrParserTest, ParseTrace) {
  const std::string input = R"(package foobar

fn bar(tkn: token, cond: bits[1], x: bits[3]) -> token {
  ret trace.1: token = trace(tkn, cond, format="x is {}", data_operands=[x], id=1)
}
)";
  ParsePackageAndCheckDump(input);
}

TEST(IrParserTest, ParseTraceWithVerbosity) {
  const std::string input = R"(package foobar

fn bar(tkn: token, cond: bits[1], x: bits[3]) -> token {
  ret trace.1: token = trace(tkn, cond, format="x is {}", data_operands=[x], verbosity=1, id=1)
}
)";
  ParsePackageAndCheckDump(input);
}

TEST(IrParserTest, ParseTraceWrongOperands) {
  const std::string input = R"(package foobar

fn bar(tkn: token, cond: bits[1], x: bits[3], y: bits[7]) -> token {
  ret trace.1: token = trace(tkn, cond, format="x is {}", data_operands=[x,y], id=1)
}
)";
  EXPECT_THAT(
      Parser::ParsePackage(input).status(),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr(
                   "Trace node expects 1 data operands, but 2 were supplied")));
}

TEST(IrParserTest, ParseTraceNegativeVerbosity) {
  const std::string input = R"(package foobar

fn bar(tkn: token, cond: bits[1], x: bits[3], y: bits[7]) -> token {
  ret trace.1: token = trace(tkn, cond, format="x is {}", data_operands=[x], verbosity=-1, id=1)
}
)";
  EXPECT_THAT(Parser::ParsePackage(input).status(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Verbosity must be >= 0: got -1")));
}

TEST(IrParserTest, ParseCover) {
  const std::string input = R"(package foobar

fn bar(tkn: token, cond: bits[1]) -> token {
  ret cover.1: token = cover(tkn, cond, label="The foo is bar", id=1)
}
)";
  ParsePackageAndCheckDump(input);
}

TEST(IrParserTest, ParseBitSliceUpdate) {
  const std::string input = R"(package foobar

fn bar(to_update: bits[123], start: bits[8], value: bits[23]) -> bits[123] {
  ret bit_slice_update.1: bits[123] = bit_slice_update(to_update, start, value, id=1)
}
)";
  ParsePackageAndCheckDump(input);
}

TEST(IrParserTest, ParseStatelessProc) {
  const std::string input = R"(package test

chan ch_in(bits[32], id=0, kind=streaming, ops=receive_only, flow_control=none, strictness=proven_mutually_exclusive, metadata="""""")
chan ch_out(bits[32], id=1, kind=streaming, ops=send_only, flow_control=none, strictness=proven_mutually_exclusive, metadata="""""")

proc my_proc() {
  my_token: token = literal(value=token, id=1)
  receive.2: (token, bits[32]) = receive(my_token, channel=ch_in, id=2)
  tuple_index.3: token = tuple_index(receive.2, index=0, id=3)
  tuple_index.4: bits[32] = tuple_index(receive.2, index=1, id=4)
  send.5: token = send(tuple_index.3, tuple_index.4, channel=ch_out, id=5)
}
)";
  ParsePackageAndCheckDump(input);
}

TEST(IrParserTest, ParseSimpleProc) {
  const std::string input = R"(package test

chan ch(bits[32], id=0, kind=streaming, ops=send_receive, flow_control=none, strictness=proven_mutually_exclusive, metadata="""""")

proc my_proc(my_token: token, my_state: bits[32], init={token, 42}) {
  send.1: token = send(my_token, my_state, channel=ch, id=1)
  literal.2: bits[1] = literal(value=1, id=2)
  receive.3: (token, bits[32]) = receive(send.1, predicate=literal.2, channel=ch, id=3)
  tuple_index.4: token = tuple_index(receive.3, index=0, id=4)
  next (tuple_index.4, my_state)
}
)";
  ParsePackageAndCheckDump(input);
}

TEST(IrParserTest, ParseProcWithExplicitNext) {
  const std::string input = R"(package test

chan ch(bits[32], id=0, kind=streaming, ops=send_receive, flow_control=none, strictness=proven_mutually_exclusive, metadata="""""")

proc my_proc(my_token: token, my_state: bits[32], init={token, 42}) {
  send.1: token = send(my_token, my_state, channel=ch, id=1)
  literal.2: bits[1] = literal(value=1, id=2)
  receive.3: (token, bits[32]) = receive(send.1, predicate=literal.2, channel=ch, id=3)
  tuple_index.4: token = tuple_index(receive.3, index=0, id=4)
  next_value.5: () = next_value(param=my_token, value=tuple_index.4, id=5)
  next_value.6: () = next_value(param=my_state, value=my_state, id=6)
}
)";
  ParsePackageAndCheckDump(input);
}

TEST(IrParserTest, ParseProcWithMixedNextValueStyles) {
  const std::string input = R"(package test

chan ch(bits[32], id=0, kind=streaming, ops=send_receive, flow_control=none, strictness=proven_mutually_exclusive, metadata="""""")

proc my_proc(my_token: token, my_state_1: bits[32], my_state_2: bits[32], init={token, 42, 64}) {
  send.1: token = send(my_token, my_state_1, channel=ch, id=1)
  literal.2: bits[1] = literal(value=1, id=2)
  receive.3: (token, bits[32]) = receive(send.1, predicate=literal.2, channel=ch, id=3)
  tuple_index.4: token = tuple_index(receive.3, index=0, id=4)
  next_value.5: () = next_value(param=my_state_2, value=my_state_2, id=5)
  next (tuple_index.4, my_state_1, my_state_2)
}
)";
  EXPECT_THAT(Parser::ParsePackage(input).status(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Proc includes both next_value nodes (e.g., "
                                 "next_value.5) and next-state values")));
}

TEST(IrParserTest, ParseProcWithBadNextParam) {
  const std::string input = R"(package test

chan ch(bits[32], id=0, kind=streaming, ops=send_receive, flow_control=none, strictness=proven_mutually_exclusive, metadata="""""")

proc my_proc(my_token: token, my_state: bits[32], init={token, 42}) {
  send.1: token = send(my_token, my_state, channel=ch, id=1)
  literal.2: bits[1] = literal(value=1, id=2)
  receive.3: (token, bits[32]) = receive(send.1, predicate=literal.2, channel=ch, id=3)
  tuple_index.4: token = tuple_index(receive.3, index=0, id=4)
  next_value.5: () = next_value(param=my_token, value=tuple_index.4, id=5)
  next_value.6: () = next_value(param=not_my_state, value=my_state, id=6)
}
)";
  EXPECT_THAT(Parser::ParsePackage(input).status(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       AllOf(HasSubstr("Referred to a name"),
                             HasSubstr("not previously defined"),
                             HasSubstr("\"not_my_state\""))));
}

TEST(IrParserTest, ParseProcWithBadNextValueType) {
  const std::string input = R"(package test

chan ch(bits[32], id=0, kind=streaming, ops=send_receive, flow_control=none, strictness=proven_mutually_exclusive, metadata="""""")

proc my_proc(my_token: token, my_state: bits[32], init={token, 42}) {
  send.1: token = send(my_token, my_state, channel=ch, id=1)
  literal.2: bits[1] = literal(value=1, id=2)
  receive.3: (token, bits[32]) = receive(send.1, predicate=literal.2, channel=ch, id=3)
  tuple_index.4: token = tuple_index(receive.3, index=0, id=4)
  next_value.5: () = next_value(param=my_token, value=tuple_index.4, id=5)
  next_value.6: () = next_value(param=my_state, value=literal.2, id=6)
}
)";
  EXPECT_THAT(Parser::ParsePackage(input).status(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("next value for param 'my_state' must be of "
                                 "type bits[32]; is: bits[1]")));
}

TEST(IrParserTest, ParseNewStyleProc) {
  const std::string input = R"(package test

proc my_proc<in_ch: bits[32] single_value in, out_ch: bits[32] streaming out>(my_token: token, my_state: bits[32], init={token, 42}) {
  send.1: token = send(my_token, my_state, channel=out_ch, id=1)
  literal.2: bits[1] = literal(value=1, id=2)
  receive.3: (token, bits[32]) = receive(send.1, predicate=literal.2, channel=in_ch, id=3)
  tuple_index.4: token = tuple_index(receive.3, index=0, id=4)
  next (tuple_index.4, my_state)
}
)";
  ParsePackageAndCheckDump(input);
}

TEST(IrParserTest, ParseNewStyleProcNoInterfaceChannels) {
  const std::string input = R"(package test

proc my_proc<>(my_token: token, my_state: bits[32], init={token, 42}) {
  next (my_token, my_state)
}
)";
  ParsePackageAndCheckDump(input);
}

TEST(IrParserTest, ParseInstantiatedProc) {
  const std::string input = R"(package test

proc my_proc<in_ch: bits[32] single_value in, out_ch: bits[32] streaming out>(my_token: token, my_state: bits[32], init={token, 42}) {
  send.1: token = send(my_token, my_state, channel=out_ch, id=1)
  literal.2: bits[1] = literal(value=1, id=2)
  receive.3: (token, bits[32]) = receive(send.1, predicate=literal.2, channel=in_ch, id=3)
  tuple_index.4: token = tuple_index(receive.3, index=0, id=4)
  next (tuple_index.4, my_state)
}

proc other_proc<>(my_token: token, my_state: bits[32], init={token, 42}) {
  chan ch_a(bits[32], id=0, kind=single_value, ops=send_receive,  metadata="""""")
  chan ch_b(bits[32], id=1, kind=streaming, ops=send_receive,  flow_control=none, strictness=proven_mutually_exclusive, metadata="""""")
  proc_instantiation foo(ch_a, ch_b, proc=my_proc)
  next (my_token, my_state)
}
)";
  ParsePackageAndCheckDump(input);
}

TEST(IrParserTest, ParseInstantiatedProcWithZeroArgs) {
  const std::string input = R"(package test

proc my_proc<>(my_token: token, my_state: bits[32], init={token, 42}) {
  next (my_token, my_state)
}

proc other_proc<>(my_token: token, my_state: bits[32], init={token, 42}) {
  proc_instantiation foo(proc=my_proc)
  next (my_token, my_state)
}
)";
  ParsePackageAndCheckDump(input);
}

TEST(IrParserTest, ParseNewStyleProcWithChannelDefinitions) {
  const std::string input = R"(package test

proc my_proc<>(my_token: token, my_state: bits[32], init={token, 42}) {
  chan ch(bits[32], id=0, kind=streaming, ops=send_receive,  flow_control=none, strictness=proven_mutually_exclusive, metadata="""""")
  send.1: token = send(my_token, my_state, channel=ch, id=1)
  receive.2: (token, bits[32]) = receive(send.1, channel=ch, id=2)
  tuple_index.3: token = tuple_index(receive.2, index=0, id=3)
  next (tuple_index.3, my_state)
}
)";
  ParsePackageAndCheckDump(input);
}

TEST(IrParserTest, NewStyleProcSendOnInput) {
  Package p("my_package");
  const std::string input =
      R"(proc my_proc<in_ch: bits[32] streaming in>(my_token: token, my_state: bits[32], init={token, 42}) {
  send: token = send(my_token, my_state, channel=in_ch)
  next (send, my_state)
}
)";
  EXPECT_THAT(Parser::ParseProc(input, &p).status(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Cannot send on channel `in_ch`")));
}

TEST(IrParserTest, NewStyleProcReceiveOnOutput) {
  Package p("my_package");
  const std::string input =
      R"(proc my_proc<out_ch: bits[32] streaming out>(my_token: token, my_state: bits[32], init={token, 42}) {
  rcv: (token, bits[32]) = receive(my_token, channel=out_ch)
  rcv_token: token = tuple_index(rcv, index=0)
  next (rcv_token, my_state)
}
)";
  EXPECT_THAT(Parser::ParseProc(input, &p).status(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Cannot receive on channel `out_ch`")));
}

TEST(IrParserTest, InstantiateNonexistentProc) {
  const std::string input = R"(package test

proc the_proc<>(my_token: token, my_state: bits[32], init={token, 42}) {
  chan ch(bits[32], id=0, kind=streaming, ops=send_receive,  flow_control=none, strictness=proven_mutually_exclusive, metadata="""""")
  proc_instantiation foo(ch, ch, proc=not_a_proc)
  next (my_token, my_state)
}
)";
  EXPECT_THAT(Parser::ParsePackage(input),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("No such proc 'not_a_proc'")));
}

TEST(IrParserTest, InstantiateOldStyleProc) {
  const std::string input = R"(package test

proc og_proc(my_token: token, my_state: bits[32], init={token, 42}) {
  next (my_token, my_state)
}

proc the_proc<>(my_token: token, my_state: bits[32], init={token, 42}) {
  chan ch(bits[32], id=0, kind=streaming, ops=send_receive,  flow_control=none, strictness=proven_mutually_exclusive, metadata="""""")
  proc_instantiation foo(ch, ch, proc=og_proc)
  next (my_token, my_state)
}
)";
  EXPECT_THAT(Parser::ParsePackage(input),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Proc `og_proc` is not a new style proc")));
}

TEST(IrParserTest, ProcInstantiationWrongNumberOfArguments) {
  const std::string input = R"(package test

proc my_proc<in_ch: bits[32] streaming in, out_ch: bits[32] streaming out>(my_token: token, my_state: bits[32], init={token, 42}) {
  send.1: token = send(my_token, my_state, channel=out_ch, id=1)
  literal.2: bits[1] = literal(value=1, id=2)
  receive.3: (token, bits[32]) = receive(send.1, predicate=literal.2, channel=in_ch, id=3)
  tuple_index.4: token = tuple_index(receive.3, index=0, id=4)
  next (tuple_index.4, my_state)
}

proc other_proc<>(my_token: token, my_state: bits[32], init={token, 42}) {
  chan ch(bits[32], id=0, kind=streaming, ops=send_receive,  flow_control=none, strictness=proven_mutually_exclusive, metadata="""""")
  proc_instantiation foo(ch, proc=my_proc)
  next (my_token, my_state)
}
)";
  EXPECT_THAT(
      Parser::ParsePackage(input),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("Proc `my_proc` expects 2 channel arguments, got 1")));
}

TEST(IrParserTest, ProcInstantiationInOldStyleProc) {
  const std::string input = R"(package test

proc my_proc<>(my_token: token, my_state: bits[32], init={token, 42}) {
  next (my_token, my_state)
}

proc og_proc(my_token: token, my_state: bits[32], init={token, 42}) {
  proc_instantiation foo(proc=my_proc)
  next (my_token, my_state)
}
)";
  EXPECT_THAT(
      Parser::ParsePackage(input),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          HasSubstr(
              "Proc instantiations can only be declared in new-style procs")));
}

TEST(IrParserTest, DirectionMismatchInInstantiatedProc) {
  const std::string input = R"(package test

proc my_proc<in_ch: bits[32] streaming in, out_ch: bits[32] streaming out>(my_token: token, my_state: bits[32], init={token, 42}) {
  send.1: token = send(my_token, my_state, channel=out_ch, id=1)
  literal.2: bits[1] = literal(value=1, id=2)
  receive.3: (token, bits[32]) = receive(send.1, predicate=literal.2, channel=in_ch, id=3)
  tuple_index.4: token = tuple_index(receive.3, index=0, id=4)
  next (tuple_index.4, my_state)
}

proc other_proc<in_ch: bits[32] streaming in, out_ch: bits[32] streaming out>(my_token: token, my_state: bits[32], init={token, 42}) {
  proc_instantiation foo(out_ch, in_ch, proc=my_proc)
  next (my_token, my_state)
}
)";
  EXPECT_THAT(Parser::ParsePackage(input),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("No such receive channel `out_ch` for proc "
                                 "instantiation arg 0")));
}

TEST(IrParserTest, DeclareChannelInOldStyleProc) {
  Package p("my_package");
  const std::string input =
      R"(proc my_proc(my_token: token, my_state: bits[32], init={token, 42}) {
  chan ch(bits[32], id=0, kind=streaming, ops=send_receive,  flow_control=none, strictness=proven_mutually_exclusive, metadata="""""")
  next (rcv_token, my_state)
}
)";
  EXPECT_THAT(
      Parser::ParseProc(input, &p).status(),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("Channels can only be declared in new-style procs")));
}

TEST(IrParserTest, DeclareChannelInFunction) {
  Package p("my_package");
  const std::string input =
      R"(fn my_func()  -> bits[1] {
  chan ch(bits[32], id=0, kind=streaming, ops=send_receive,  flow_control=none, strictness=proven_mutually_exclusive, metadata="""""")
  ret result: bits[1] = literal(value=0, id=1)
}
)";
  EXPECT_THAT(Parser::ParseFunction(input, &p).status(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("chan keyword only supported in procs")));
}

TEST(IrParserTest, NewStyleProcUsingGlobalChannel) {
  const std::string input =
      R"(package test

chan ch(bits[32], id=0, kind=streaming, ops=receive_only, flow_control=none, strictness=proven_mutually_exclusive, metadata="""""")

proc my_proc<>(my_token: token, my_state: bits[32], init={token, 42}) {
  rcv: (token, bits[32]) = receive(my_token, channel=ch)
  rcv_token: token = tuple_index(rcv, index=0)
  next (rcv_token, my_state)
}
)";
  EXPECT_THAT(Parser::ParsePackage(input),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("No such channel `ch`")));
}

TEST(IrParserTest, NewStyleProcWithDuplicateChannelNames) {
  Package p("my_package");
  const std::string input =
      R"(proc my_proc<ch: bits[32] streaming in, ch: bits[32] streaming out>(my_token: token, my_state: bits[32], init={token, 42}) {
  send.1: token = send(my_token, my_state, channel=ch, id=1)
  literal.2: bits[1] = literal(value=1, id=2)
  receive.3: (token, bits[32]) = receive(send.1, predicate=literal.2, channel=ch, id=3)
  tuple_index.4: token = tuple_index(receive.3, index=0, id=4)
  next (tuple_index.4, my_state)
}
)";
  EXPECT_THAT(
      Parser::ParseProc(input, &p).status(),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("Cannot add channel `ch` to proc `my_proc`. "
                         "Already an input channel of same name on the proc")));
}

TEST(IrParserTest, InstantiatedProcWithUnknownChannel) {
  const std::string input = R"(package test

proc my_proc<in_ch: bits[32] streaming in, out_ch: bits[32] streaming out>(my_token: token, my_state: bits[32], init={token, 42}) {
  send.1: token = send(my_token, my_state, channel=out_ch, id=1)
  literal.2: bits[1] = literal(value=1, id=2)
  receive.3: (token, bits[32]) = receive(send.1, predicate=literal.2, channel=in_ch, id=3)
  tuple_index.4: token = tuple_index(receive.3, index=0, id=4)
  next (tuple_index.4, my_state)
}

proc other_proc<>(my_token: token, my_state: bits[32], init={token, 42}) {
  proc_instantiation foo(not_a_channel, also_not_a_channel, proc=my_proc)
  next (my_token, my_state)
}
)";
  EXPECT_THAT(Parser::ParsePackage(input),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("No such receive channel `not_a_channel`")));
}

TEST(IrParserTest, ParseNewStyleProcWithComplexChannelTypes) {
  const std::string input = R"(package test

proc my_proc<in_ch: () streaming in, out_ch: ((), bits[32][1]) streaming out>(my_token: token, my_state: ((), bits[32][1]), init={token, ((), [42])}) {
  send.1: token = send(my_token, my_state, channel=out_ch, id=1)
  literal.2: bits[1] = literal(value=1, id=2)
  receive.3: (token, ()) = receive(send.1, predicate=literal.2, channel=in_ch, id=3)
  tuple_index.4: token = tuple_index(receive.3, index=0, id=4)
  next (tuple_index.4, my_state)
}
)";
  ParsePackageAndCheckDump(input);
}

TEST(IrParserTest, ParseSimpleBlock) {
  const std::string input = R"(package test

block my_block(a: bits[32], b: bits[32], out: bits[32]) {
  a: bits[32] = input_port(name=a, id=2)
  b: bits[32] = input_port(name=b, id=3)
  add.4: bits[32] = add(a, b, id=4)
  out: () = output_port(add.4, name=out, id=5)
}
)";
  ParsePackageAndCheckDump(input);
}

TEST(IrParserTest, ParseBlockWithRegister) {
  const std::string input = R"(package test

block my_block(in: bits[32], clk: clock, out: bits[32]) {
  reg foo(bits[32])
  in: bits[32] = input_port(name=in, id=1)
  foo_d: () = register_write(in, register=foo, id=2)
  foo_q: bits[32] = register_read(register=foo, id=3)
  out: () = output_port(foo_q, name=out, id=4)
}
)";
  ParsePackageAndCheckDump(input);
}

TEST(IrParserTest, ParseBlockWithRegisterWithResetValue) {
  const std::string input = R"(package test

block my_block(clk: clock, rst: bits[1], in: bits[32], out: bits[32]) {
  reg foo(bits[32], reset_value=42, asynchronous=true, active_low=false)
  rst: bits[1] = input_port(name=rst, id=1)
  in: bits[32] = input_port(name=in, id=2)
  foo_d: () = register_write(in, register=foo, reset=rst, id=4)
  foo_q: bits[32] = register_read(register=foo, id=3)
  out: () = output_port(foo_q, name=out, id=5)
}
)";
  ParsePackageAndCheckDump(input);
}

TEST(IrParserTest, ParseBlockWithRegisterWithLoadEnable) {
  const std::string input = R"(package test

block my_block(clk: clock, in: bits[32], le: bits[1], out: bits[32]) {
  reg foo(bits[32])
  in: bits[32] = input_port(name=in, id=1)
  le: bits[1] = input_port(name=le, id=2)
  foo_d: () = register_write(in, register=foo, load_enable=le, id=4)
  foo_q: bits[32] = register_read(register=foo, id=3)
  out: () = output_port(foo_q, name=out, id=5)
}
)";
  ParsePackageAndCheckDump(input);
}

TEST(IrParserTest, ParseBlockWithBlockInstantiation) {
  const std::string input = R"(package test

block sub_block(in: bits[38], out: bits[32]) {
  in: bits[38] = input_port(name=in, id=1)
  zero: bits[32] = literal(value=0, id=2)
  out: () = output_port(zero, name=out, id=3)
}

block my_block(x: bits[8], y: bits[32]) {
  instantiation foo(block=sub_block, kind=block)
  x: bits[8] = input_port(name=x, id=4)
  foo_in: () = instantiation_input(x, instantiation=foo, port_name=in, id=5)
  foo_out: bits[32] = instantiation_output(instantiation=foo, port_name=out, id=6)
  y: () = output_port(foo_out, name=y, id=7)
}
)";
  ParsePackageAndCheckDump(input);
}

TEST(IrParserTest, ParseInstantiationOfDegenerateBlock) {
  const std::string input = R"(package test

block sub_block() {
}

block my_block(x: bits[8], y: bits[8]) {
  instantiation foo(block=sub_block, kind=block)
  x: bits[8] = input_port(name=x, id=1)
  y: () = output_port(x, name=y, id=2)
}
)";
  ParsePackageAndCheckDump(input);
}

TEST(IrParserTest, ParseInstantiationOfNoInputBlock) {
  const std::string input = R"(package test
block sub_block(out: bits[32]) {
  zero: bits[32] = literal(value=0, id=1)
  out: () = output_port(zero, name=out, id=2)
}

block my_block(y: bits[32]) {
  instantiation foo(block=sub_block, kind=block)
  out: bits[32] = instantiation_output(instantiation=foo, port_name=out, id=3)
  y: () = output_port(out, name=y, id=4)
}
)";
  ParsePackageAndCheckDump(input);
}

TEST(IrParserTest, ParseInstantiationOfNoOutputBlock) {
  const std::string input = R"(package test
block sub_block(in: bits[32]) {
  in: bits[32] = input_port(name=in, id=1)
}

block my_block(x: bits[32]) {
  instantiation foo(block=sub_block, kind=block)
  x: bits[32] = input_port(name=x, id=2)
  x_in: () = instantiation_input(x, instantiation=foo, port_name=in, id=3)
}
)";
  ParsePackageAndCheckDump(input);
}

TEST(IrParserTest, ParseInstantiationWithChannel) {
  constexpr std::string_view input = R"(package test
chan foo(bits[32], id=42, kind=streaming, ops=send_receive, flow_control=none, strictness=proven_mutually_exclusive, fifo_depth=0, bypass=true, register_push_outputs=false, register_pop_outputs=false, metadata="""""")

proc placeholder_channel_user(tok: token, init={token}) {
  recv_out: (token, bits[32]) = receive(tok, channel=foo, id=1)
  recv_tok: token = tuple_index(recv_out, index=0, id=2)
  recv_data: bits[32] = tuple_index(recv_out, index=1, id=3)
  send_out: token = send(recv_tok, recv_data, channel=foo, id=4)
  next (send_out)
}

block sub_block(in: bits[32], out: bits[32]) {
  in: bits[32] = input_port(name=in, id=5)
  zero: bits[32] = literal(value=0, id=6)
  out: () = output_port(zero, name=out, id=7)
}

block my_block(x: bits[32], y: bits[32]) {
  instantiation foo_inst(data_type=bits[32], depth=0, bypass=true, register_push_outputs=false, register_pop_outputs=false, channel=foo, kind=fifo)
  instantiation bar(block=sub_block, kind=block)
  x: bits[32] = input_port(name=x, id=8)
  x_in: () = instantiation_input(x, instantiation=bar, port_name=in, id=9)
  x_out: bits[32] = instantiation_output(instantiation=bar, port_name=out, id=10)
  y: () = output_port(x_out, name=y, id=11)
}
)";
  ParsePackageAndCheckDump(input);
}

TEST(IrParserTest, ParseInstantiationWithNoBypassChannel) {
  constexpr std::string_view input = R"(package test
chan foo(bits[32], id=42, kind=streaming, ops=send_receive, flow_control=none, strictness=proven_mutually_exclusive, fifo_depth=1, bypass=false, register_push_outputs=true, register_pop_outputs=true, metadata="""""")

proc placeholder_channel_user(tok: token, init={token}) {
  recv_out: (token, bits[32]) = receive(tok, channel=foo, id=1)
  recv_tok: token = tuple_index(recv_out, index=0, id=2)
  recv_data: bits[32] = tuple_index(recv_out, index=1, id=3)
  send_out: token = send(recv_tok, recv_data, channel=foo, id=4)
  next (send_out)
}

block sub_block(in: bits[32], out: bits[32]) {
  in: bits[32] = input_port(name=in, id=5)
  zero: bits[32] = literal(value=0, id=6)
  out: () = output_port(zero, name=out, id=7)
}

block my_block(x: bits[32], y: bits[32]) {
  instantiation foo_inst(data_type=bits[32], depth=1, bypass=false, register_push_outputs=true, register_pop_outputs=true, channel=foo, kind=fifo)
  instantiation bar(block=sub_block, kind=block)
  x: bits[32] = input_port(name=x, id=8)
  x_in: () = instantiation_input(x, instantiation=bar, port_name=in, id=9)
  x_out: bits[32] = instantiation_output(instantiation=bar, port_name=out, id=10)
  y: () = output_port(x_out, name=y, id=11)
}
)";
  ParsePackageAndCheckDump(input);
}

TEST(IrParserTest, ParseArrayIndex) {
  const std::string input = R"(
fn foo(x: bits[32][6]) -> bits[32] {
  literal.1: bits[32] = literal(value=5, id=1)
  ret array_index.2: bits[32] = array_index(x, indices=[literal.1], id=2)
}
)";
  ParseFunctionAndCheckDump(input);
}

TEST(IrParserTest, ParseArraySlice) {
  const std::string input = R"(
fn foo(arr: bits[32][6]) -> bits[32][2] {
  literal.1: bits[32] = literal(value=5, id=1)
  ret array_slice.2: bits[32][2] = array_slice(arr, literal.1, width=2, id=2)
}
)";
  ParseFunctionAndCheckDump(input);
}

TEST(IrParserTest, ParseArrayUpdate) {
  const std::string input = R"(
fn foo(array: bits[32][3], idx: bits[32], newval: bits[32]) -> bits[32][3] {
  ret array_update.4: bits[32][3] = array_update(array, newval, indices=[idx], id=4)
}
)";
  ParseFunctionAndCheckDump(input);
}

TEST(IrParserTest, ParseArrayUpdateNonArary) {
  const std::string input = R"(
fn foo(array: bits[32], idx: bits[32], newval: bits[32]) -> bits[32][3] {
  ret array_update.4: bits[32][3] = array_update(array, newval, indices=[idx],  id=4)
}
)";
  Package p("my_package");
  EXPECT_THAT(
      Parser::ParseFunction(input, &p).status(),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          HasSubstr(
              "Too many indices (1) to index into array of type bits[32]")));
}

TEST(IrParserTest, ParseArrayUpdateIncompatibleTypes) {
  const std::string input = R"(
fn foo(array: bits[32][3], idx: bits[32], newval: bits[64]) -> bits[32][3] {
  ret array_update.4: bits[32][3] = array_update(array, newval, indices=[idx], id=4)
}
)";
  Package p("my_package");
  EXPECT_THAT(Parser::ParseFunction(input, &p).status(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Expected update value to have type bits[32]; "
                                 "has type bits[64]")));
}

TEST(IrParserTest, ParseArrayConcat0) {
  const std::string input = R"(
fn foo(a0: bits[32][3], a1: bits[32][1]) -> bits[32][4] {
  ret array_concat.3: bits[32][4] = array_concat(a0, a1, id=3)
}
)";
  ParseFunctionAndCheckDump(input);
}

TEST(IrParserTest, ParseArrayConcat1) {
  const std::string input = R"(
fn foo(a0: bits[32][0], a1: bits[32][1]) -> bits[32][1] {
  ret array_concat.3: bits[32][1] = array_concat(a0, a1, id=3)
}
)";
  ParseFunctionAndCheckDump(input);
}

TEST(IrParserTest, ParseArrayConcatMixedOperands) {
  const std::string input = R"(
fn f(a0: bits[32][2], a1: bits[32][3], a2: bits[32][1]) -> bits[32][7] {
  array_concat.4: bits[32][1] = array_concat(a2, id=4)
  array_concat.5: bits[32][2] = array_concat(array_concat.4, array_concat.4, id=5)
  array_concat.6: bits[32][7] = array_concat(a0, array_concat.5, a1, id=6)
  ret array_concat.7: bits[32][7] = array_concat(array_concat.6, id=7)
}
)";
  ParseFunctionAndCheckDump(input);
}

TEST(IrParserTest, ParseArrayConcatNonArrayType) {
  const std::string input = R"(
fn foo(a0: bits[16], a1: bits[16][1]) -> bits[16][2] {
  ret array_concat.3: bits[16][2] = array_concat(a0, a1, id=3)
}
)";
  Package p("my_package");
  EXPECT_THAT(Parser::ParseFunction(input, &p).status(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Cannot array-concat node a0 because it has "
                                 "non-array type bits[16]")));
}

TEST(IrParserTest, ParseArrayIncompatibleElementType) {
  const std::string input = R"(
fn foo(a0: bits[16][1], a1: bits[32][1]) -> bits[16][2] {
  ret array_concat.3: bits[16][2] = array_concat(a0, a1, id=3)
}
)";
  Package p("my_package");
  EXPECT_THAT(
      Parser::ParseFunction(input, &p).status(),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("Cannot array-concat node a1 because it has element "
                         "type bits[32] but expected bits[16]")));
}

TEST(IrParserTest, ParseArrayIncompatibleReturnType) {
  const std::string input = R"(
fn foo(a0: bits[16][1], a1: bits[16][1]) -> bits[16][3] {
  ret array_concat.3: bits[16][3] = array_concat(a0, a1, id=3)
}
)";
  Package p("my_package");
  EXPECT_THAT(Parser::ParseFunction(input, &p).status(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Declared type bits[16][3] does not match "
                                 "expected type bits[16][2]")));
}

TEST(IrParserTest, ParseTupleIndex) {
  const std::string input = R"(
fn foo(x: bits[42]) -> bits[33] {
  literal.1: bits[32] = literal(value=5, id=1)
  literal.2: bits[33] = literal(value=123, id=2)
  tuple.3: (bits[42], bits[32], bits[33]) = tuple(x, literal.1, literal.2, id=3)
  ret tuple_index.4: bits[33] = tuple_index(tuple.3, index=2, id=4)
}
)";
  ParseFunctionAndCheckDump(input);
}

TEST(IrParserTest, ParseIdentity) {
  const std::string input = R"(
fn foo(x: bits[32]) -> bits[32] {
  ret identity.2: bits[32] = identity(x, id=2)
}
)";
  ParseFunctionAndCheckDump(input);
}

TEST(IrParserTest, ParseUnsignedInequalities) {
  std::string program = R"(
fn parse_inequalities() -> bits[1] {
  literal.1: bits[32] = literal(value=2, id=1)
  literal.2: bits[32] = literal(value=2, id=2)
  uge.3: bits[1] = uge(literal.1, literal.2, id=3)
  ugt.4: bits[1] = ugt(literal.1, literal.2, id=4)
  ule.5: bits[1] = ule(literal.1, literal.2, id=5)
  ult.6: bits[1] = ult(literal.1, literal.2, id=6)
  ret eq.7: bits[1] = eq(literal.1, literal.2, id=7)
}
)";
  ParseFunctionAndCheckDump(program);
}

TEST(IrParserTest, ParseSignedInequalities) {
  std::string program = R"(
fn parse_inequalities() -> bits[1] {
  literal.1: bits[32] = literal(value=2, id=1)
  literal.2: bits[32] = literal(value=2, id=2)
  sge.3: bits[1] = sge(literal.1, literal.2, id=3)
  sgt.4: bits[1] = sgt(literal.1, literal.2, id=4)
  sle.5: bits[1] = sle(literal.1, literal.2, id=5)
  slt.6: bits[1] = slt(literal.1, literal.2, id=6)
  ret eq.7: bits[1] = eq(literal.1, literal.2, id=7)
}
)";
  ParseFunctionAndCheckDump(program);
}

TEST(IrParserTest, StandAloneRet) {
  const std::string input = R"(package foobar

fn foo(x: bits[32]) -> bits[32] {
  identity.2: bits[32] = identity(x, id=2)
  ret identity.2
}
)";

  EXPECT_THAT(Parser::ParsePackage(input).status(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Expected token of type \":\"")));
}

TEST(IrParserTest, ParseEndOfLineComment) {
  const std::string input = R"(// top comment
package foobar
// a comment

fn foo(x: bits[32]) -> bits[32] {  // another comment
  ret identity.2: bits[32] = identity(x)  // yep, another one

// comment

}
)";
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> pkg,
                           Parser::ParsePackage(input));
  ASSERT_EQ(pkg->functions().size(), 1);
}

TEST(IrParserTest, ParseTupleType) {
  const std::string input = R"(
    package foobar

    fn foo(x: bits[32]) -> (bits[32], bits[32]) {
       ret tuple.1: (bits[32], bits[32]) = tuple(x, x, id=1)
    }
  )";
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> pkg,
                           Parser::ParsePackage(input));
  ASSERT_EQ(pkg->functions().size(), 1);
  Type* t = pkg->functions()[0]->return_value()->GetType();
  EXPECT_TRUE(t->IsTuple());
  EXPECT_EQ(t->AsTupleOrDie()->size(), 2);
  EXPECT_TRUE(t->AsTupleOrDie()->element_type(0)->IsBits());
  EXPECT_TRUE(t->AsTupleOrDie()->element_type(1)->IsBits());
}

TEST(IrParserTest, ParseEmptyTuple) {
  const std::string input = R"(
    package foobar

    fn foo(x: bits[32]) -> () {
       ret tuple.1: () = tuple(id=1)
    }
  )";
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> pkg,
                           Parser::ParsePackage(input));
  ASSERT_EQ(pkg->functions().size(), 1);
  Type* t = pkg->functions()[0]->return_value()->GetType();
  EXPECT_TRUE(t->IsTuple());
  EXPECT_EQ(t->AsTupleOrDie()->size(), 0);
}

TEST(IrParserTest, ParseNestedTuple) {
  const std::string input = R"(
    package foobar

    fn foo(x: bits[32]) -> ((bits[32], bits[32]), bits[32]) {
       tuple.1: (bits[32], bits[32]) = tuple(x, x, id=1)
       ret tuple.2: ((bits[32], bits[32]), bits[32]) = tuple(tuple.1, x, id=2)
    }
  )";
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> pkg,
                           Parser::ParsePackage(input));
  ASSERT_EQ(pkg->functions().size(), 1);
  Type* t = pkg->functions()[0]->return_value()->GetType();
  EXPECT_TRUE(t->IsTuple());
  EXPECT_EQ(t->AsTupleOrDie()->size(), 2);
  EXPECT_TRUE(t->AsTupleOrDie()->element_type(0)->IsTuple());
  EXPECT_EQ(t->AsTupleOrDie()->element_type(0)->AsTupleOrDie()->size(), 2);
  EXPECT_TRUE(t->AsTupleOrDie()->element_type(1)->IsBits());
}

TEST(IrParserTest, ParseArrayLiterals) {
  const std::string input = R"(
fn foo(x: bits[32]) -> bits[32] {
  literal.1: bits[32][2] = literal(value=[0, 1], id=1)
  literal.2: bits[3] = literal(value=1, id=2)
  ret array_index.3: bits[32] = array_index(literal.1, indices=[literal.2], id=3)
}
)";
  ParseFunctionAndCheckDump(input);
}

TEST(IrParserTest, ParseNestedArrayLiterals) {
  const std::string input = R"(
fn foo() -> bits[32][2][3][1] {
  ret literal.1: bits[32][2][3][1] = literal(value=[[[0, 1], [2, 3], [4, 5]]], id=1)
}
)";
  ParseFunctionAndCheckDump(input);
}

TEST(IrParserTest, ParseArrayLiteralWithInsufficientBits) {
  Package p("my_package");
  const std::string input = R"(
fn foo() -> bits[7][2] {
  ret literal.1: bits[7][2] = literal(value=[0, 12345], id=1)
}
)";
  EXPECT_THAT(
      Parser::ParseFunction(input, &p).status(),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("Value 12345 is not representable in 7 bits")));
}

TEST(IrParserTest, ReturnArrayLiteral) {
  const std::string input = R"(
package foobar

fn foo(x: bits[32]) -> bits[32][2] {
  ret literal.1: bits[32][2] = literal(value=[0, 1], id=1)
}
)";
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> pkg,
                           Parser::ParsePackage(input));
  ASSERT_EQ(pkg->functions().size(), 1);
  Type* t = pkg->functions()[0]->return_value()->GetType();
  EXPECT_TRUE(t->IsArray());
}

TEST(IrParserTest, ReturnArrayOfTuplesLiteral) {
  const std::string input = R"(
package foobar

fn foo() -> (bits[32], bits[3])[2] {
  ret literal.1: (bits[32], bits[3])[2] = literal(value=[(2, 2), (0, 1)], id=1)
}
)";
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> pkg,
                           Parser::ParsePackage(input));
  ASSERT_EQ(pkg->functions().size(), 1);
  Type* t = pkg->functions()[0]->return_value()->GetType();
  EXPECT_TRUE(t->IsArray());
}

TEST(IrParserTest, ArrayValueInBitsLiteral) {
  Package p("my_package");
  const std::string input = R"(
fn foo() -> bits[42] {
  ret literal.1: bits[42] = literal(value=[0, 123], id=1)
}
)";
  EXPECT_THAT(Parser::ParseFunction(input, &p).status(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Expected token of type \"literal\"")));
}

TEST(IrParserTest, BitsValueInArrayLiteral) {
  Package p("my_package");
  const std::string input = R"(
fn foo() -> bits[7][42] {
  ret literal.1: bits[7][42] = literal(value=123], id=1)
}
)";
  EXPECT_THAT(Parser::ParseFunction(input, &p).status(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Expected token of type \"[\"")));
}

TEST(IrParserTest, ParseTupleLiteral) {
  const std::string input = R"(
fn foo() -> (bits[32][2], bits[1]) {
  ret literal.1: (bits[32][2], bits[1]) = literal(value=([123, 456], 0), id=1)
}
)";
  ParseFunctionAndCheckDump(input);
}

TEST(IrParserTest, ParseNestedTupleLiteral) {
  const std::string input = R"(
fn foo() -> (bits[32][2], bits[1], (), (bits[44])) {
  ret literal.1: (bits[32][2], bits[1], (), (bits[44])) = literal(value=([123, 456], 0, (), (10)), id=1)
}
)";
  ParseFunctionAndCheckDump(input);
}

TEST(IrParserTest, ParseNaryXor) {
  const std::string input = R"(
fn foo(x: bits[8]) -> bits[8] {
  ret xor.2: bits[8] = xor(x, x, x, x, id=2)
}
)";
  ParseFunctionAndCheckDump(input);
}

TEST(IrParserTest, ParseExtendOps) {
  const std::string input = R"(
fn foo(x: bits[8]) -> bits[32] {
  zero_ext.1: bits[32] = zero_ext(x, new_bit_count=32, id=1)
  sign_ext.2: bits[32] = sign_ext(x, new_bit_count=32, id=2)
  ret xor.3: bits[32] = xor(zero_ext.1, sign_ext.2, id=3)
}
)";
  ParseFunctionAndCheckDump(input);
}

TEST(IrParserTest, ParseInconsistentExtendOp) {
  const std::string input = R"(
fn foo(x: bits[8]) -> bits[32] {
  ret zero_ext.1: bits[33] = zero_ext(x, new_bit_count=32, id=1)
}
)";
  Package p("my_package");
  EXPECT_THAT(
      Parser::ParseFunction(input, &p).status(),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("differs from its new_bit_count annotation 32")));
}

TEST(IrParserTest, ParseDecode) {
  const std::string input = R"(
fn foo(x: bits[8]) -> bits[256] {
  decode.1: bits[42] = decode(x, width=42, id=1)
  ret decode.2: bits[256] = decode(x, width=256, id=2)
}
)";
  ParseFunctionAndCheckDump(input);
}

TEST(IrParserTest, ParseEncode) {
  const std::string input = R"(
fn foo(x: bits[16]) -> bits[4] {
  ret encode.1: bits[4] = encode(x, id=1)
}
)";
  ParseFunctionAndCheckDump(input);
}

TEST(IrParserTest, Gate) {
  const std::string input = R"(
fn foo(cond: bits[1], x: bits[16]) -> bits[16] {
  ret gate.1: bits[16] = gate(cond, x, id=1)
}
)";
  ParseFunctionAndCheckDump(input);
}

TEST(IrParserTest, ArrayIndexOfTuple) {
  const std::string input = R"(
fn foo(x: (bits[8])) -> bits[32] {
  literal.1: bits[32] = literal(value=0, id=1)
  ret array_index.2: bits[8] = array_index(x, indices=[literal.1], id=2)
}
)";
  Package p("my_package");
  EXPECT_THAT(
      Parser::ParseFunction(input, &p).status(),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          HasSubstr(
              "Too many indices (1) to index into array of type (bits[8])")));
}

TEST(IrParserTest, TupleIndexOfArray) {
  const std::string input = R"(
fn foo(x: bits[8][5]) -> bits[8] {
  ret tuple_index.1: bits[8] = tuple_index(x, index=0, id=1)
}
)";
  Package p("my_package");
  EXPECT_THAT(Parser::ParseFunction(input, &p).status(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("tuple_index operand is not a tuple")));
}

TEST(IrParserTest, NicerErrorOnEmptyString) {
  const std::string input = "";  // NOLINT: emphasize empty string here.
  EXPECT_THAT(
      Parser::ParsePackage(input).status(),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          HasSubstr(
              "Expected keyword 'package': Expected token, but found EOF.")));
}

TEST(IrParserTest, ParsesComplexValue) {
  const std::string input = "(0xf00, [0xba5, 0xba7], [0])";
  Package p("test_package");
  auto* u32 = p.GetBitsType(32);
  auto* u12 = p.GetBitsType(12);
  auto* u1 = p.GetBitsType(1);
  auto* array_1xu1 = p.GetArrayType(1, u1);
  auto* array_2xu12 = p.GetArrayType(2, u12);
  auto* overall = p.GetTupleType({u32, array_2xu12, array_1xu1});
  XLS_ASSERT_OK_AND_ASSIGN(Value v, Parser::ParseValue(input, overall));
  Value expected = Value::Tuple({
      Value(UBits(0xf00, /*bit_count=*/32)),
      Value::ArrayOrDie({
          Value(UBits(0xba5, /*bit_count=*/12)),
          Value(UBits(0xba7, /*bit_count=*/12)),
      }),
      Value::ArrayOrDie({Value(UBits(0, /*bit_count=*/1))}),
  });
  EXPECT_EQ(expected, v);
}

TEST(IrParserTest, ParsesComplexValueWithEmbeddedTypes) {
  const std::string input =
      "(bits[32]:0xf00, [bits[12]:0xba5, bits[12]:0xba7], [bits[1]:0])";
  XLS_ASSERT_OK_AND_ASSIGN(Value v, Parser::ParseTypedValue(input));
  Value expected = Value::Tuple({
      Value(UBits(0xf00, /*bit_count=*/32)),
      Value::ArrayOrDie({
          Value(UBits(0xba5, /*bit_count=*/12)),
          Value(UBits(0xba7, /*bit_count=*/12)),
      }),
      Value::ArrayOrDie({Value(UBits(0, /*bit_count=*/1))}),
  });
  EXPECT_EQ(expected, v);
}

TEST(IrParserTest, ParsesTokenType) {
  const std::string input = "token";
  XLS_ASSERT_OK_AND_ASSIGN(Value v, Parser::ParseTypedValue(input));
  Value expected = Value::Token();
  EXPECT_EQ(expected, v);
}

TEST(IrParserTest, ParsesComplexValueWithEmbeddedTokens) {
  const std::string input =
      "(bits[32]:0xf00, [bits[12]:0xba5, bits[12]:0xba7], [token, token], "
      "[bits[1]:0], token)";
  XLS_ASSERT_OK_AND_ASSIGN(Value v, Parser::ParseTypedValue(input));
  Value expected = Value::Tuple({
      Value(UBits(0xf00, /*bit_count=*/32)),
      Value::ArrayOrDie({
          Value(UBits(0xba5, /*bit_count=*/12)),
          Value(UBits(0xba7, /*bit_count=*/12)),
      }),
      Value::ArrayOrDie({Value::Token(), Value::Token()}),
      Value::ArrayOrDie({Value(UBits(0, /*bit_count=*/1))}),
      Value::Token(),
  });
  EXPECT_EQ(expected, v);
}

// TODO(leary): 2019-08-01 Figure out if we want to reify the type into the
// empty array Value.
TEST(IrParserTest, DISABLED_ParsesEmptyArray) {
  const std::string input = "[]";
  Package p("test_package");
  auto* u1 = p.GetBitsType(1);
  auto* array_0xu1 = p.GetArrayType(0, u1);
  XLS_ASSERT_OK_AND_ASSIGN(Value v, Parser::ParseValue(input, array_0xu1));
  Value expected = Value::ArrayOrDie({});
  EXPECT_EQ(expected, v);
}

TEST(IrParserTest, BigOrdinalAnnotation) {
  std::string program = R"(
package test

fn main() -> bits[1] {
  ret literal.1000: bits[1] = literal(value=0, id=1000)
}
)";
  XLS_ASSERT_OK_AND_ASSIGN(auto package, Parser::ParsePackage(program));
  EXPECT_GT(package->next_node_id(), 1000);
}

TEST(IrParserTest, TrivialProc) {
  std::string program = R"(
package test

proc foo(my_token: token, my_state: bits[32], init={token, 42}) {
  next (my_token, my_state)
}
)";
  XLS_ASSERT_OK_AND_ASSIGN(auto package, Parser::ParsePackage(program));
  EXPECT_EQ(package->functions().size(), 0);
  EXPECT_EQ(package->procs().size(), 1);
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, package->GetProc("foo"));
  EXPECT_EQ(proc->node_count(), 2);
  EXPECT_EQ(proc->params().size(), 2);
  EXPECT_EQ(proc->GetInitValueElement(0).ToString(), "token");
  EXPECT_EQ(proc->GetStateElementType(0)->ToString(), "token");
  EXPECT_EQ(proc->GetInitValueElement(1).ToString(), "bits[32]:42");
  EXPECT_EQ(proc->GetStateElementType(1)->ToString(), "bits[32]");
  EXPECT_EQ(proc->GetStateParam(0)->GetName(), "my_token");
  EXPECT_EQ(proc->GetStateParam(1)->GetName(), "my_state");
}

TEST(IrParserTest, StatelessProcWithInitAndNext) {
  std::string program = R"(
package test

proc foo(init={}) {
  next ()
}
)";
  XLS_ASSERT_OK_AND_ASSIGN(auto package, Parser::ParsePackage(program));
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, package->GetProc("foo"));
  EXPECT_EQ(proc->node_count(), 0);
  EXPECT_THAT(proc->StateParams(), IsEmpty());
}

TEST(IrParserTest, StatelessProcWithNextButNotInit) {
  std::string program = R"(
package test

proc foo() {
  next ()
}
)";
  XLS_ASSERT_OK_AND_ASSIGN(auto package, Parser::ParsePackage(program));
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, package->GetProc("foo"));
  EXPECT_EQ(proc->node_count(), 0);
  EXPECT_THAT(proc->StateParams(), IsEmpty());
}

TEST(IrParserTest, FunctionAndProc) {
  std::string program = R"(
package test

fn my_function() -> bits[1] {
  ret literal.1: bits[1] = literal(value=0, id=1)
}

proc my_proc(my_token: token, my_state: bits[32], init={token, 42}) {
  next (my_token, my_state)
}
)";
  XLS_ASSERT_OK_AND_ASSIGN(auto package, Parser::ParsePackage(program));
  EXPECT_EQ(package->functions().size(), 1);
  EXPECT_EQ(package->procs().size(), 1);
  XLS_ASSERT_OK_AND_ASSIGN(Function * function,
                           package->GetFunction("my_function"));
  EXPECT_EQ(function->name(), "my_function");
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, package->GetProc("my_proc"));
  EXPECT_EQ(proc->name(), "my_proc");
}

TEST(IrParserTest, ProcWithMultipleStateElements) {
  std::string program = R"(
package test

proc foo( x: bits[32], y: (), z: bits[32], init={42, (), 123}) {
  sum: bits[32] = add(x, z)
  next (x, y, sum)
}
)";
  XLS_ASSERT_OK_AND_ASSIGN(auto package, Parser::ParsePackage(program));
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, package->GetProc("foo"));
  EXPECT_EQ(proc->GetStateElementCount(), 3);
}

TEST(IrParserTest, ProcWithTokenAfterStateElements) {
  std::string program = R"(
package test

proc foo(x: bits[32], y: (), z: bits[32], my_token: token, init={42, (), 123, token}) {
  sum: bits[32] = add(x, z)
  next (x, y, sum, my_token)
}
)";
  XLS_ASSERT_OK_AND_ASSIGN(auto package, Parser::ParsePackage(program));
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, package->GetProc("foo"));
  EXPECT_EQ(proc->GetStateElementCount(), 4);
}

TEST(IrParserTest, ProcWithTokenBetweenStateElements) {
  std::string program = R"(
package test

proc foo(x: bits[32], my_token: token, y: (), z: bits[32], init={42, token, (), 123}) {
  sum: bits[32] = add(x, z)
  next (x, my_token, y, sum)
}
)";
  XLS_ASSERT_OK_AND_ASSIGN(auto package, Parser::ParsePackage(program));
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, package->GetProc("foo"));
  EXPECT_EQ(proc->GetStateElementCount(), 4);
}

TEST(IrParserTest, ProcWithMultipleTokens) {
  std::string program = R"(
package test

proc foo(tok1: token, x: bits[32], tok2: token, y: (), z: bits[32], init={token, 42, token, (), 123}) {
  sum: bits[32] = add(x, z)
  next (tok2, x, tok1, y, sum)
}
)";
  XLS_ASSERT_OK_AND_ASSIGN(auto package, Parser::ParsePackage(program));
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, package->GetProc("foo"));
  EXPECT_EQ(proc->GetStateElementCount(), 5);
}

TEST(IrParserTest, ProcTooFewInitialValues) {
  std::string program = R"(
package test

proc foo(my_token: token, my_state: bits[32], total_garbage: bits[1], init={token, 42}) {
  next (my_token, my_state, total_garbage)
}
)";
  EXPECT_THAT(Parser::ParsePackage(program).status(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Too few initial values given")));
}

TEST(IrParserTest, ProcTooManyInitialValues) {
  std::string program = R"(
package test

proc foo(my_token: token, my_state: bits[32], init={token, 42, 1, 2, 3}) {
  next (my_token, my_state)
}
)";
  EXPECT_THAT(Parser::ParsePackage(program).status(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Too many initial values given")));
}

TEST(IrParserTest, ProcWithMissingInitValues) {
  std::string program = R"(
package test

proc foo(my_token: token) {
  next (my_token)
}
)";
  EXPECT_THAT(
      Parser::ParsePackage(program).status(),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("Mandatory keyword argument `init` not found")));
}

TEST(IrParserTest, ProcWithTooFewNextStateElements) {
  std::string program = R"(
package test

proc foo(my_token: token, x: bits[32], y: (), z: bits[32], init={token, 42, (), 123}) {
  next (my_token, x, y)
}
)";
  EXPECT_THAT(
      Parser::ParsePackage(program).status(),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          HasSubstr("Number of recurrent state elements given (3) does not "
                    "equal the number of state elements in the proc (4)")));
}

TEST(IrParserTest, ProcWithTooManyNextStateElements) {
  std::string program = R"(
package test

proc foo(my_token: token, x: bits[32], y: (), z: bits[32], init={token, 42, (), 123}) {
  next (my_token, x, y, z, z)
}
)";
  EXPECT_THAT(
      Parser::ParsePackage(program).status(),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          HasSubstr("Number of recurrent state elements given (5) does not "
                    "equal the number of state elements in the proc (4)")));
}

TEST(IrParserTest, ProcWrongInitValueType) {
  std::string program = R"(
package test

proc foo(my_token: token, my_state: bits[32], init={token, (1, 2, 3)}) {
  next (my_token, my_state)
}
)";
  EXPECT_THAT(Parser::ParsePackage(program).status(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Expected token of type \"literal\"")));
}

TEST(IrParserTest, ProcWrongInitValueTypeToken) {
  std::string program = R"(
package test

proc foo(my_token: token, my_state: bits[32], init={1, (1, 2, 3)}) {
  next (my_token, my_state)
}
)";
  EXPECT_THAT(Parser::ParsePackage(program).status(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Expected 'token' keyword")));
}

TEST(IrParserTest, ProcWrongReturnType) {
  std::string program = R"(
package test

proc foo(my_token: token, my_state: bits[32], init={token, 42}) {
  literal.1: bits[32] = literal(value=123, id=1)
  next (literal.1, my_state)
}
)";
  EXPECT_THAT(
      Parser::ParsePackage(program).status(),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("Recurrent state type token does not match proc "
                         "parameter state type bits[32] for element 0.")));
}

TEST(IrParserTest, ProcWithRet) {
  std::string program = R"(
package test

proc foo(my_token: token, my_state: bits[32], init={token, 42}) {
  ret literal.1: bits[32] = literal(value=123, id=1)
}
)";
  EXPECT_THAT(Parser::ParsePackage(program).status(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("ret keyword only supported in functions")));
}

TEST(IrParserTest, FunctionWithNext) {
  std::string program = R"(
package test

fn foo(x: bits[32]) -> bits[32] {
  next (x, x)
}
)";
  EXPECT_THAT(Parser::ParsePackage(program).status(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("next keyword only supported in procs")));
}

TEST(IrParserTest, ProcWithBogusNextToken) {
  std::string program = R"(
package test

proc foo(my_token: token, my_state: bits[32], init={token, 42}) {
  next (foobar, my_state)
}
)";
  EXPECT_THAT(Parser::ParsePackage(program).status(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Proc next state name @ 5:9  was not "
                                 "previously defined")));
}

TEST(IrParserTest, ProcWithBogusNextState) {
  std::string program = R"(
package test

proc foo(my_token: token, my_state: bits[32], init={token, 42}) {
  next (my_token, sfsdfsfd)
}
)";
  EXPECT_THAT(
      Parser::ParsePackage(program).status(),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr(
                   "Proc next state name @ 5:19  was not previously defined")));
}

TEST(IrParserTest, ParseSendReceiveChannel) {
  Package p("my_package");
  XLS_ASSERT_OK_AND_ASSIGN(Channel * ch,
                           Parser::ParseChannel(
                               R"(chan foo(bits[32], id=42, kind=single_value,
                      ops=send_receive,
                      metadata="module_port { flopped: true }"))",
                               &p));
  EXPECT_EQ(ch->name(), "foo");
  EXPECT_EQ(ch->id(), 42);
  EXPECT_EQ(ch->supported_ops(), ChannelOps::kSendReceive);
  EXPECT_EQ(ch->kind(), ChannelKind::kSingleValue);
  EXPECT_EQ(ch->type(), p.GetBitsType(32));
  EXPECT_TRUE(ch->initial_values().empty());
  EXPECT_EQ(ch->metadata().channel_oneof_case(),
            ChannelMetadataProto::kModulePort);
  EXPECT_TRUE(ch->metadata().module_port().flopped());
}

TEST(IrParserTest, ParseSendReceiveChannelWithInitialValues) {
  Package p("my_package");
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * ch,
      Parser::ParseChannel(
          R"(chan foo(bits[32], initial_values={2, 4, 5}, id=42, kind=streaming,
                         flow_control=none, ops=send_receive,
                         metadata="module_port { flopped: true }"))",
          &p));
  EXPECT_EQ(ch->name(), "foo");
  EXPECT_EQ(ch->id(), 42);
  EXPECT_EQ(ch->supported_ops(), ChannelOps::kSendReceive);
  EXPECT_EQ(ch->kind(), ChannelKind::kStreaming);
  EXPECT_EQ(ch->type(), p.GetBitsType(32));
  EXPECT_THAT(ch->initial_values(),
              ElementsAre(Value(UBits(2, 32)), Value(UBits(4, 32)),
                          Value(UBits(5, 32))));
  EXPECT_EQ(ch->metadata().channel_oneof_case(),
            ChannelMetadataProto::kModulePort);
  EXPECT_TRUE(ch->metadata().module_port().flopped());
}

TEST(IrParserTest, ParseSendReceiveChannelWithTupleType) {
  Package p("my_package");
  XLS_ASSERT_OK_AND_ASSIGN(Channel * ch, Parser::ParseChannel(
                                             R"(chan foo((bits[32], bits[1]),
                      initial_values={(123, 1), (42, 0)},
                      id=42, kind=streaming, flow_control=ready_valid,
                      ops=send_receive,
                      metadata="module_port { flopped: true }"))",
                                             &p));
  EXPECT_EQ(ch->name(), "foo");
  EXPECT_THAT(
      ch->initial_values(),
      ElementsAre(Value::Tuple({Value(UBits(123, 32)), Value(UBits(1, 1))}),
                  Value::Tuple({Value(UBits(42, 32)), Value(UBits(0, 1))})));
}

TEST(IrParserTest, ParseSendOnlyChannel) {
  Package p("my_package");
  XLS_ASSERT_OK_AND_ASSIGN(Channel * ch, Parser::ParseChannel(
                                             R"(chan bar((bits[32], bits[1]),
                         id=7, kind=single_value, ops=send_only,
                         metadata="module_port { flopped: false }"))",
                                             &p));
  EXPECT_EQ(ch->name(), "bar");
  EXPECT_EQ(ch->id(), 7);
  EXPECT_EQ(ch->supported_ops(), ChannelOps::kSendOnly);
  EXPECT_EQ(ch->type(), p.GetTupleType({p.GetBitsType(32), p.GetBitsType(1)}));
  EXPECT_EQ(ch->metadata().channel_oneof_case(),
            ChannelMetadataProto::kModulePort);
  EXPECT_FALSE(ch->metadata().module_port().flopped());
}

TEST(IrParserTest, ParseReceiveOnlyChannel) {
  Package p("my_package");
  XLS_ASSERT_OK_AND_ASSIGN(Channel * ch, Parser::ParseChannel(
                                             R"(chan meh(bits[32][4], id=0,
                         kind=single_value, ops=receive_only,
                         metadata="module_port { flopped: true }"))",
                                             &p));
  EXPECT_EQ(ch->name(), "meh");
  EXPECT_EQ(ch->id(), 0);
  EXPECT_EQ(ch->supported_ops(), ChannelOps::kReceiveOnly);
  EXPECT_EQ(ch->type(), p.GetArrayType(4, p.GetBitsType(32)));
  EXPECT_EQ(ch->metadata().channel_oneof_case(),
            ChannelMetadataProto::kModulePort);
  EXPECT_TRUE(ch->metadata().module_port().flopped());
}

TEST(IrParserTest, ParseStreamingChannelWithStrictness) {
  Package p("my_package");
  XLS_ASSERT_OK_AND_ASSIGN(Channel * ch,
                           Parser::ParseChannel(
                               R"(chan foo(bits[32], id=42, kind=streaming,
                         flow_control=none, ops=send_receive,
                         strictness=arbitrary_static_order, metadata=""""""))",
                               &p));
  EXPECT_EQ(ch->name(), "foo");
  EXPECT_EQ(ch->id(), 42);
  EXPECT_EQ(ch->supported_ops(), ChannelOps::kSendReceive);
  EXPECT_EQ(ch->kind(), ChannelKind::kStreaming);
  EXPECT_EQ(ch->type(), p.GetBitsType(32));
  EXPECT_EQ(down_cast<StreamingChannel*>(ch)->GetStrictness(),
            ChannelStrictness::kArbitraryStaticOrder);
}

TEST(IrParserTest, ParseStreamingChannelWithExtraFifoMetadata) {
  Package p("my_package");
  XLS_ASSERT_OK_AND_ASSIGN(Channel * ch,
                           Parser::ParseChannel(
                               R"(chan foo(bits[32], id=42, kind=streaming,
                         flow_control=none, ops=send_receive, fifo_depth=3,
                         bypass=false, metadata=""""""))",
                               &p));
  EXPECT_EQ(ch->name(), "foo");
  EXPECT_EQ(ch->id(), 42);
  EXPECT_EQ(ch->supported_ops(), ChannelOps::kSendReceive);
  ASSERT_EQ(ch->kind(), ChannelKind::kStreaming);
  EXPECT_EQ(ch->type(), p.GetBitsType(32));
  ASSERT_THAT(down_cast<StreamingChannel*>(ch)->fifo_config(),
              Not(Eq(std::nullopt)));
  EXPECT_EQ(down_cast<StreamingChannel*>(ch)->fifo_config()->depth(), 3);
  EXPECT_EQ(down_cast<StreamingChannel*>(ch)->fifo_config()->bypass(), false);
}

TEST(IrParserTest, ParseStreamingValueChannelWithBlockPortMapping) {
  // For testing round-trip parsing.
  std::string ch_ir_text;

  {
    Package p("my_package");
    XLS_ASSERT_OK_AND_ASSIGN(Channel * ch, Parser::ParseChannel(
                                               R"(chan meh(bits[32][4], id=0,
                         kind=streaming, flow_control=ready_valid,
                         ops=send_only,
                         metadata="""block_ports { data_port_name : "data",
                                                   block_name : "blk",
                                                   ready_port_name : "rdy",
                                                   valid_port_name: "vld"
                                                 }"""))",
                                               &p));
    EXPECT_EQ(ch->name(), "meh");
    EXPECT_EQ(ch->id(), 0);
    EXPECT_EQ(ch->supported_ops(), ChannelOps::kSendOnly);
    EXPECT_TRUE(ch->metadata().has_block_ports());

    EXPECT_TRUE(ch->metadata().block_ports().has_data_port_name());
    EXPECT_TRUE(ch->metadata().block_ports().has_ready_port_name());
    EXPECT_TRUE(ch->metadata().block_ports().has_valid_port_name());

    EXPECT_EQ(ch->metadata().block_ports().block_name(), "blk");
    EXPECT_EQ(ch->metadata().block_ports().data_port_name(), "data");
    EXPECT_EQ(ch->metadata().block_ports().ready_port_name(), "rdy");
    EXPECT_EQ(ch->metadata().block_ports().valid_port_name(), "vld");

    ch_ir_text = ch->ToString();
  }

  {
    Package p("my_package_2");
    XLS_ASSERT_OK_AND_ASSIGN(Channel * ch,
                             Parser::ParseChannel(ch_ir_text, &p));
    EXPECT_EQ(ch->name(), "meh");

    EXPECT_EQ(ch->id(), 0);

    EXPECT_EQ(ch->supported_ops(), ChannelOps::kSendOnly);
    EXPECT_TRUE(ch->metadata().has_block_ports());

    EXPECT_TRUE(ch->GetBlockName().has_value());
    EXPECT_TRUE(ch->GetDataPortName().has_value());
    EXPECT_TRUE(ch->GetValidPortName().has_value());
    EXPECT_TRUE(ch->GetReadyPortName().has_value());

    EXPECT_EQ(ch->GetBlockName().value(), "blk");
    EXPECT_EQ(ch->GetDataPortName().value(), "data");
    EXPECT_EQ(ch->GetValidPortName().value(), "vld");
    EXPECT_EQ(ch->GetReadyPortName().value(), "rdy");
  }
}

TEST(IrParserTest, ParseSingleValueChannelWithBlockPortMapping) {
  // For testing round-trip parsing.
  std::string ch_ir_text;

  {
    Package p("my_package");
    XLS_ASSERT_OK_AND_ASSIGN(Channel * ch, Parser::ParseChannel(
                                               R"(chan meh(bits[32][4], id=0,
                         kind=single_value, ops=receive_only,
                         metadata="""module_port { flopped: true },
                                     block_ports { data_port_name : "data",
                                                   block_name : "blk"}"""))",
                                               &p));
    EXPECT_EQ(ch->name(), "meh");
    EXPECT_EQ(ch->id(), 0);
    EXPECT_EQ(ch->supported_ops(), ChannelOps::kReceiveOnly);
    EXPECT_EQ(ch->metadata().channel_oneof_case(),
              ChannelMetadataProto::kModulePort);
    EXPECT_TRUE(ch->metadata().module_port().flopped());

    EXPECT_TRUE(ch->metadata().has_block_ports());
    EXPECT_EQ(ch->metadata().block_ports().block_name(), "blk");
    EXPECT_EQ(ch->metadata().block_ports().data_port_name(), "data");
    EXPECT_FALSE(ch->metadata().block_ports().has_ready_port_name());
    EXPECT_FALSE(ch->metadata().block_ports().has_valid_port_name());

    ch_ir_text = ch->ToString();
  }

  {
    Package p("my_package_2");
    XLS_ASSERT_OK_AND_ASSIGN(Channel * ch,
                             Parser::ParseChannel(ch_ir_text, &p));
    EXPECT_EQ(ch->name(), "meh");

    EXPECT_EQ(ch->id(), 0);
    EXPECT_EQ(ch->supported_ops(), ChannelOps::kReceiveOnly);
    EXPECT_TRUE(ch->metadata().has_block_ports());

    EXPECT_TRUE(ch->GetBlockName().has_value());
    EXPECT_TRUE(ch->GetDataPortName().has_value());
    EXPECT_FALSE(ch->GetValidPortName().has_value());
    EXPECT_FALSE(ch->GetReadyPortName().has_value());

    EXPECT_EQ(ch->GetBlockName().value(), "blk");
    EXPECT_EQ(ch->GetDataPortName().value(), "data");
  }
}

TEST(IrParserTest, ChannelParsingErrors) {
  Package p("my_package");
  EXPECT_THAT(Parser::ParseChannel(
                  R"(chan meh(bits[32][4], kind=single_value,
                         ops=receive_only,
                         metadata="module_port { flopped: true }"))",
                  &p)
                  .status(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Mandatory keyword argument `id` not found")));

  EXPECT_THAT(
      Parser::ParseChannel(
          R"(chan meh(bits[32][4], id=42, ops=receive_only,
                         metadata="module_port { flopped: true }"))",
          &p)
          .status(),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("Mandatory keyword argument `kind` not found")));

  EXPECT_THAT(Parser::ParseChannel(
                  R"(chan meh(bits[32][4], id=42, kind=bogus,
                         ops=receive_only,
                         metadata="module_port { flopped: true }"))",
                  &p)
                  .status(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Invalid channel kind \"bogus\"")));

  EXPECT_THAT(
      Parser::ParseChannel(
          R"(chan meh(bits[32][4], id=7, kind=streaming,
                         metadata="module_port { flopped: true }"))",
          &p)
          .status(),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("Mandatory keyword argument `ops` not found")));

  // Unrepresentable initial value.
  EXPECT_THAT(Parser::ParseChannel(
                  R"(chan meh(bits[4], initial_values={128}, kind=streaming,
                         ops=send_receive, id=7,
                         metadata="module_port { flopped: true }"))",
                  &p)
                  .status(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Value 128 is not representable in 4 bits")));

  // Wrong initial value type.
  EXPECT_THAT(Parser::ParseChannel(
                  R"(chan meh(bits[4], initial_values={(1, 2)}, kind=streaming,
                         ops=send_receive, id=7
                         metadata="module_port { flopped: true }"))",
                  &p)
                  .status(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Expected token of type \"literal\"")));

  EXPECT_THAT(
      Parser::ParseChannel(
          R"(chan meh(bits[32][4], id=7, kind=streaming,
                     ops=receive_only))",
          &p)
          .status(),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("Mandatory keyword argument `metadata` not found")));

  EXPECT_THAT(Parser::ParseChannel(
                  R"(chan meh(id=44, kind=streaming, ops=receive_only,
                         metadata="module_port { flopped: true }"))",
                  &p)
                  .status(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Expected 'bits' keyword")));

  EXPECT_THAT(Parser::ParseChannel(
                  R"(chan meh(bits[32], id=44, kind=streaming,
                         ops=receive_only, bogus="totally!",
                         metadata="module_port { flopped: true }"))",
                  &p)
                  .status(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Invalid keyword argument `bogus`")));

  // Bad channel name.
  EXPECT_THAT(Parser::ParseChannel(
                  R"(chan 444meh(foo: bits[32], id=7, kind=streaming,
                         ops=receive_only, metadata=""))",
                  &p)
                  .status(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Expected token of type \"ident\"")));

  // FIFO depth on single-value channel.
  EXPECT_THAT(
      Parser::ParseChannel(
          R"(chan meh(bits[32], id=44, kind=single_value,
                         ops=receive_only, fifo_depth=123, metadata=""))",
          &p)
          .status(),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("Only streaming channels can have fifo_depth")));

  // Strictness on single-value channel.
  EXPECT_THAT(
      Parser::ParseChannel(
          R"(chan meh(bits[32], id=44, kind=single_value, ops=receive_only,
                         strictness=proven_mutually_exclusive, metadata=""))",
          &p)
          .status(),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("Only streaming channels can have strictness")));

  // Bypass, register_push_outputs, or register_pop_outputs without fifo_depth.
  EXPECT_THAT(Parser::ParseChannel(
                  R"(chan meh(bits[32], id=44, kind=streaming, ops=receive_only,
                      bypass=true, metadata=""))",
                  &p)
                  .status(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("fifo_depth must be specified")));
}

TEST(IrParserTest, PackageWithSingleDataElementChannels) {
  std::string program = R"(
package test

chan hbo(bits[32], id=0, kind=streaming, flow_control=none, ops=receive_only,
            fifo_depth=42, metadata="module_port { flopped: true }")
chan mtv(bits[32], id=1, kind=streaming, flow_control=none, ops=send_only,
            metadata="module_port { flopped: true }")

proc my_proc(my_token: token, my_state: bits[32], init={token, 42}) {
  receive.1: (token, bits[32]) = receive(my_token, channel=hbo)
  tuple_index.2: token = tuple_index(receive.1, index=0, id=2)
  tuple_index.3: bits[32] = tuple_index(receive.1, index=1, id=3)
  add.4: bits[32] = add(my_state, tuple_index.3, id=4)
  send.5: token = send(tuple_index.2, add.4, channel=mtv)
  next (send.5, add.4)
}
)";
  XLS_ASSERT_OK_AND_ASSIGN(auto package, Parser::ParsePackage(program));
  EXPECT_EQ(package->procs().size(), 1);
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, package->GetProc("my_proc"));
  EXPECT_EQ(proc->name(), "my_proc");
  XLS_ASSERT_OK_AND_ASSIGN(Channel * hbo, package->GetChannel("hbo"));
  EXPECT_THAT(down_cast<StreamingChannel*>(hbo)->GetFifoDepth(), Optional(42));
  XLS_ASSERT_OK_AND_ASSIGN(Channel * mtv, package->GetChannel("mtv"));
  EXPECT_EQ(down_cast<StreamingChannel*>(mtv)->GetFifoDepth(), std::nullopt);
}

TEST(IrParserTest, ParseTupleIndexWithInvalidBValue) {
  const std::string input = R"(
fn f(x: bits[4], y: bits[4][1]) -> bits[4] {
  onehot.10: bits[16] = decode(y, width=16, id=10)
  ret ind.20: bits[4] = tuple_index(onehot.10, index=0, id=20)
}
)";

  Package p("my_package");
  EXPECT_THAT(Parser::ParseFunction(input, &p).status(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Decode argument must be of Bits type")));
}

TEST(IrParserTest, NodeNames) {
  std::string program = R"(package test

fn foo(x: bits[32], foobar: bits[32]) -> bits[32] {
  add.1: bits[32] = add(x, foobar, id=1)
  ret qux: bits[32] = not(add.1, id=123)
}
)";
  XLS_ASSERT_OK_AND_ASSIGN(auto package, Parser::ParsePackage(program));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, package->GetFunction("foo"));
  EXPECT_EQ(package->DumpIr(), program);

  Node* x = f->param(0);
  EXPECT_TRUE(x->HasAssignedName());
  EXPECT_EQ(x->GetName(), "x");
  EXPECT_EQ(x->id(), 1);

  Node* foobar = f->param(1);
  EXPECT_TRUE(foobar->HasAssignedName());
  EXPECT_EQ(foobar->GetName(), "foobar");
  EXPECT_EQ(foobar->id(), 2);

  Node* add = f->return_value()->operand(0);
  EXPECT_FALSE(add->HasAssignedName());
  EXPECT_EQ(add->GetName(), "add.1");
  EXPECT_EQ(add->id(), 1);

  Node* qux = f->return_value();
  EXPECT_TRUE(qux->HasAssignedName());
  EXPECT_EQ(qux->GetName(), "qux");
}

TEST(IrParserTest, InvalidName) {
  const std::string input = R"(
fn f(x: bits[4]) -> bits[4] {
  ret blahblah.30: bits[4] = add(x, x, id=30)
}
)";
  Package p("my_package");
  EXPECT_THAT(
      Parser::ParseFunction(input, &p).status(),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("The substring 'blahblah' in node name blahblah.30 "
                         "does not match the node op 'add")));
}

TEST(IrParserTest, IdAttributes) {
  const std::string input = R"(
fn f(x: bits[4]) -> bits[4] {
  foo: bits[4] = not(x)
  bar: bits[4] = not(foo, id=42)
  not.123: bits[4] = not(bar, id=123)
  ret not.333: bits[4] = not(not.123, id=333)
}
)";
  Package p("my_package");
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, Parser::ParseFunction(input, &p));
  EXPECT_EQ(f->return_value()->id(), 333);
  EXPECT_EQ(f->return_value()->operand(0)->id(), 123);
  EXPECT_EQ(f->return_value()->operand(0)->operand(0)->id(), 42);
  EXPECT_EQ(f->return_value()->operand(0)->operand(0)->operand(0)->id(), 2);
}

TEST(IrParserTest, MismatchedId) {
  const std::string input = R"(
fn f(x: bits[4]) -> bits[4] {
  ret add.30: bits[4] = add(x, x, id=42)
}
)";
  Package p("my_package");
  EXPECT_THAT(
      Parser::ParseFunction(input, &p).status(),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("The id '30' in node name add.30 does not match the "
                         "id '42' specified as an attribute")));
}

TEST(IrParserTest, FunctionWithPort) {
  const std::string input = R"(
fn foo(a: bits[32]) -> bits[32][3] {
  b: bits[32] = input_port(name=b, id=1)
  ret sum: bits[32] = add(a, b)
}
)";
  Package p("my_package");
  EXPECT_THAT(
      Parser::ParseFunction(input, &p).status(),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("input_port operations only supported in blocks")));
}

TEST(IrParserTest, BlockWithReturnValue) {
  const std::string input = R"(
block my_block(a: bits[32]) {
  ret a: bits[32] = input_port(name=a)
}
)";
  Package p("my_package");
  EXPECT_THAT(Parser::ParseBlock(input, &p).status(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("ret keyword only supported in functions")));
}

TEST(IrParserTest, WriteOfNonexistentRegister) {
  const std::string input = R"(
block my_block(clk: clock, in: bits[32], out: bits[32]) {
  reg foo(bits[32])
  in: bits[32] = input_port(name=in, id=1)
  foo_d: () = register_write(in, register=bar, id=2)
  foo_q: bits[32] = register_read(register=foo, id=3)
  out: () = output_port(foo_q, name=out, id=4)
}
)";
  Package p("my_package");
  EXPECT_THAT(Parser::ParseBlock(input, &p).status(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("No such register named bar")));
}

TEST(IrParserTest, ReadOfNonexistentRegister) {
  const std::string input = R"(
block my_block(clk: clock, in: bits[32], out: bits[32]) {
  reg foo(bits[32])
  in: bits[32] = input_port(name=in, id=1)
  foo_d: () = register_write(in, register=foo, id=2)
  foo_q: bits[32] = register_read(register=bar, id=3)
  out: () = output_port(foo_q, name=out, id=4)
}
)";
  Package p("my_package");
  EXPECT_THAT(Parser::ParseBlock(input, &p).status(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("No such register named bar")));
}

TEST(IrParserTest, ParseBlockWithRegisterWithWrongResetValueType) {
  const std::string input = R"(
block my_block(clk: clock, in: bits[32], out: bits[32]) {
  reg foo(bits[32], reset_value=(42, 43))
  in: bits[32] = input_port(name=in, id=1)
  foo_q: bits[32] = register_read(register=foo, id=3)
  foo_d: () = register_write(in, register=foo, id=2)
  out: () = output_port(foo_q, name=out, id=4)
}
)";
  Package p("my_package");
  EXPECT_THAT(Parser::ParseBlock(input, &p).status(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Expected token of type \"literal\"")));
}

TEST(IrParserTest, RegisterInFunction) {
  const std::string input = R"(package my_package

fn f(foo: bits[32]) -> bits[32] {
  reg bar(bits[32], reset_value=(42, 43))
  ret result: bits[32] not(foo)
}
)";
  EXPECT_THAT(Parser::ParsePackage(input).status(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("reg keyword only supported in blocks")));
}

TEST(IrParserTest, ParseBlockWithDuplicateRegisters) {
  const std::string input = R"(
block my_block(clk: clock, in: bits[32], out: bits[32]) {
  reg foo(bits[32])
  reg foo(bits[32])
  in: bits[32] = input_port(name=in, id=1)
  foo_q: bits[32] = register_read(register=foo, id=3)
  foo_d: () = register_write(in, register=foo, id=2)
  out: () = output_port(foo_q, name=out, id=4)
}
)";
  Package p("my_package");
  EXPECT_THAT(Parser::ParseBlock(input, &p).status(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Register already exists with name foo")));
}

TEST(IrParserTest, ParseBlockWithIncompleteResetDefinition) {
  const std::string input = R"(
block my_block(clk: clock, in: bits[32], out: bits[32]) {
  reg foo(bits[32], reset_value=42)
  in: bits[32] = input_port(name=in, id=1)
  foo_q: bits[32] = register_read(register=foo, id=3)
  foo_d: () = register_write(in, register=foo, id=2)
  out: () = output_port(foo_q, name=out, id=4)
}
)";
  Package p("my_package");
  EXPECT_THAT(Parser::ParseBlock(input, &p).status(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Register reset incompletely specified")));
}

TEST(IrParserTest, BlockWithRegistersButNoClock) {
  const std::string input = R"(
block my_block(in: bits[32], out: bits[32]) {
  reg foo(bits[32])
  in: bits[32] = input_port(name=in, id=1)
  foo_q: bits[32] = register_read(register=foo, id=3)
  foo_d: () = register_write(in, register=foo, id=2)
  out: () = output_port(foo_q, name=out, id=4)
}
)";
  Package p("my_package");
  EXPECT_THAT(Parser::ParseBlock(input, &p).status(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Block has registers but no clock port")));
}

TEST(IrParserTest, BlockWithTwoClocks) {
  const std::string input = R"(
block my_block(clk1: clock, clk2: clock, in: bits[32], out: bits[32]) {
  reg foo(bits[32])
  in: bits[32] = input_port(name=in, id=1)
  foo_q: bits[32] = register_read(register=foo, id=3)
  foo_d: () = register_write(in, register=foo, id=2)
  out: () = output_port(foo_q, name=out, id=4)
}
)";
  Package p("my_package");
  EXPECT_THAT(Parser::ParseBlock(input, &p).status(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Block has multiple clocks")));
}

TEST(IrParserTest, BlockWithIncompletePortList) {
  const std::string input = R"(
block my_block(clk: clock, in: bits[32]) {
  reg foo(bits[32])
  in: bits[32] = input_port(name=in, id=1) foo_q: bits[32] = register_read(register=foo, id=3)
  foo_d: () = register_write(in, register=foo, id=2)
  out: () = output_port(foo_q, name=out, id=4)
}
)";
  Package p("my_package");
  EXPECT_THAT(
      Parser::ParseBlock(input, &p).status(),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("Block signature does not contain port \"out\"")));
}

TEST(IrParserTest, BlockWithExtraPort) {
  const std::string input = R"(
block my_block(clk: clock, in: bits[32], out: bits[32], bogus_port: bits[32]) {
  reg foo(bits[32])
  in: bits[32] = input_port(name=in, id=1)
  foo_q: bits[32] = register_read(register=foo, id=3)
  foo_d: () = register_write(in, register=foo, id=2)
  out: () = output_port(foo_q, name=out, id=4)
}
)";
  Package p("my_package");
  EXPECT_THAT(Parser::ParseBlock(input, &p).status(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Block port bogus_port has no corresponding "
                                 "input_port or output_port node")));
}

TEST(IrParserTest, BlockWithDuplicatePort) {
  const std::string input = R"(
block my_block(clk: clock, in: bits[32], out: bits[32], out: bits[32]) {
  reg foo(bits[32])
  in: bits[32] = input_port(name=in, id=1)
  foo_q: bits[32] = register_read(register=foo, id=3)
  foo_d: () = register_write(in, register=foo, id=2)
  out: () = output_port(foo_q, name=out, id=4)
}
)";
  Package p("my_package");
  EXPECT_THAT(Parser::ParseBlock(input, &p).status(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Duplicate port name \"out\"")));
}

TEST(IrParserTest, BlockWithInvalidRegisterField) {
  const std::string input = R"(
block my_block(clk: clock, in: bits[32], out: bits[32]) {
  reg foo(bits[32], bogus_field=1, reset_value=42, asynchronous=true, active_low=false)
  in: bits[32] = input_port(name=in, id=1)
  foo_q: bits[32] = register_read(register=foo, id=3)
  foo_d: () = register_write(in, register=foo, id=2)
  out: () = output_port(foo_q, name=out, id=4)
}
)";
  Package p("my_package");
  EXPECT_THAT(Parser::ParseBlock(input, &p).status(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Invalid keyword argument `bogus_field`")));
}

TEST(IrParserTest, ParseBlockWithMissingInstantiatedBlock) {
  const std::string input = R"(package test

block my_block(x: bits[8], y: bits[32]) {
  instantiation foo(block=sub_block, kind=block)
  x: bits[8] = input_port(name=x, id=4)
  foo_out: bits[32] = instantiation_output(instantiation=foo, port_name=out, id=6)
  foo_in: () = instantiation_input(x, instantiation=foo, port_name=in, id=5)
  y: () = output_port(foo_out, name=y, id=7)
}
)";
  EXPECT_THAT(Parser::ParsePackage(input).status(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("No such block 'sub_block'")));
}

TEST(IrParserTest, ParseBlockWithUnknownInstantiation) {
  const std::string input = R"(package test

block my_block(x: bits[8], y: bits[32]) {
  x: bits[8] = input_port(name=x)
  foo_out: bits[32] = instantiation_output(instantiation=foo, port_name=out)
  foo_in: () = instantiation_input(x, instantiation=foo, port_name=in)
  y: () = output_port(foo_out, name=y)
}
)";
  EXPECT_THAT(Parser::ParsePackage(input).status(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("No instantiation named `foo`")));
}

TEST(IrParserTest, ParseBlockWithDuplicateInstantiationPort) {
  const std::string input = R"(package test

block sub_block(in: bits[38], out: bits[32]) {
  zero: bits[32] = literal(value=0)
  in: bits[38] = input_port(name=in)
  out: () = output_port(zero, name=out)
}

block my_block(x: bits[8], y: bits[32]) {
  instantiation foo(block=sub_block, kind=block)
  x: bits[8] = input_port(name=x)
  foo_out: bits[32] = instantiation_output(instantiation=foo, port_name=out)
  foo_out2: bits[32] = instantiation_output(instantiation=foo, port_name=out)
  foo_in: () = instantiation_input(x, instantiation=foo, port_name=in)
  y: () = output_port(foo_out, name=y)
}
)";
  EXPECT_THAT(
      Parser::ParsePackage(input).status(),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("Duplicate instantiation input/output nodes for port "
                         "`out` in instantiation `foo` of block `sub_block`")));
}

TEST(IrParserTest, ParseBlockWithMissingInstantiationPort) {
  const std::string input = R"(package test

block sub_block(in: bits[38], out: bits[32]) {
  zero: bits[32] = literal(value=0)
  in: bits[38] = input_port(name=in)
  out: () = output_port(zero, name=out)
}

block my_block(x: bits[8], y: bits[32]) {
  instantiation foo(block=sub_block, kind=block)
  x: bits[8] = input_port(name=x)
  foo_out: bits[32] = instantiation_output(instantiation=foo, port_name=out)
  y: () = output_port(foo_out, name=y)
}
)";
  EXPECT_THAT(
      Parser::ParsePackage(input).status(),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("Instantiation `foo` of block `sub_block` is missing "
                         "instantation input/output node for port `in`")));
}

TEST(IrParserTest, ParseBlockWithWronglyNamedInstantiationPort) {
  const std::string input = R"(package test

block sub_block(in: bits[38], out: bits[32]) {
  zero: bits[32] = literal(value=0)
  in: bits[38] = input_port(name=in)
  out: () = output_port(zero, name=out)
}

block my_block(x: bits[8], y: bits[32]) {
  instantiation foo(block=sub_block, kind=block)
  x: bits[8] = input_port(name=x)
  foo_out: bits[32] = instantiation_output(instantiation=foo, port_name=out)
  foo_in: () = instantiation_input(x, instantiation=foo, port_name=in)
  foo_bogus: () = instantiation_input(x, instantiation=foo, port_name=bogus)
  y: () = output_port(foo_out, name=y)
}
)";
  EXPECT_THAT(Parser::ParsePackage(input).status(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("No port `bogus` on instantiated block "
                                 "`sub_block` for instantiation `foo`")));
}

TEST(IrParserTest, ParseBlockWithWronglyTypedSignature) {
  const std::string input = R"(package test

block my_block(x: bits[8], y: bits[32]) {
  x: bits[8] = input_port(name=x)
  y: () = output_port(x, name=y)
}
)";
  EXPECT_THAT(Parser::ParsePackage(input).status(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Type of output port \"y\" "
                                 "in block signature bits[8] does not match "
                                 "type of output_port operation: bits[32]")));
}

TEST(IrParserTest, ParseTopFunction) {
  const std::string input = R"(package test

top fn my_function(x: bits[32], y: bits[32]) -> bits[32] {
  ret add: bits[32] = add(x, y)
}

)";
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> pkg,
                           Parser::ParsePackage(input));
  XLS_ASSERT_OK_AND_ASSIGN(FunctionBase * my_function,
                           pkg->GetFunction("my_function"));
  EXPECT_TRUE(pkg->GetTop().has_value());
  EXPECT_EQ(pkg->GetTop().value(), my_function);
}

TEST(IrParserTest, ParseTopProc) {
  const std::string input = R"(package test

top proc my_proc(tkn: token, st: bits[32], init={token, 42}) {
  literal: bits[32] = literal(value=1)
  add: bits[32] = add(literal, st)
  next (tkn, add)
}

)";
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> pkg,
                           Parser::ParsePackage(input));
  XLS_ASSERT_OK_AND_ASSIGN(FunctionBase * my_proc, pkg->GetProc("my_proc"));
  EXPECT_TRUE(pkg->GetTop().has_value());
  EXPECT_EQ(pkg->GetTop().value(), my_proc);
}

TEST(IrParserTest, ParseTopBlock) {
  const std::string input = R"(package test

top block my_block(a: bits[32], b: bits[32], out: bits[32]) {
  a: bits[32] = input_port(name=a)
  b: bits[32] = input_port(name=b)
  add: bits[32] = add(a, b)
  out: () = output_port(add, name=out)
}

)";
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> pkg,
                           Parser::ParsePackage(input));
  XLS_ASSERT_OK_AND_ASSIGN(FunctionBase * my_block, pkg->GetBlock("my_block"));
  EXPECT_TRUE(pkg->GetTop().has_value());
  EXPECT_EQ(pkg->GetTop().value(), my_block);
}

TEST(IrParserTest, ParseWithTwoTops) {
  const std::string input = R"(
package my_package

top fn my_function(x: bits[32], y: bits[32]) -> bits[32] {
  ret add: bits[32] = add(x, y)
}

top block my_block(a: bits[32], b: bits[32], out: bits[32]) {
  a: bits[32] = input_port(name=a)
  b: bits[32] = input_port(name=b)
  add: bits[32] = add(a, b)
  out: () = output_port(add, name=out)
}
)";
  EXPECT_THAT(
      Parser::ParsePackage(input),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          HasSubstr("Top declared more than once, previous declaration @")));
}

TEST(IrParserTest, ParseTopInvalidTopEntity) {
  const std::string input = R"(
package invalid_top_entity_package

top invalid_top_entity
)";
  EXPECT_THAT(
      Parser::ParsePackage(input),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("Expected fn, proc or block definition, got")));
}

TEST(IrParserTest, ParseTopInvalidTopEntityWithKeyword) {
  const std::string input = R"(
package invalid_top_entity_package

top reg
)";
  EXPECT_THAT(
      Parser::ParsePackage(input),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("Expected fn, proc or block definition, got")));
}

TEST(IrParserTest, ParseIIFunction) {
  std::string input = R"(package test

#[initiation_interval(12)]
top fn example() -> bits[32] {
  ret literal.1: bits[32] = literal(value=2, id=1)
}
)";
  ParsePackageAndCheckDump(input);
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> pkg,
                           Parser::ParsePackage(input));
  XLS_ASSERT_OK_AND_ASSIGN(FunctionBase * f, pkg->GetFunction("example"));
  EXPECT_EQ(f->GetInitiationInterval(), 12);
}

TEST(IrParserTest, ParseIIProc) {
  std::string input = R"(package test

#[initiation_interval(12)]
top proc example(tkn: token, init={token}) {
  next (tkn)
}
)";
  ParsePackageAndCheckDump(input);
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> pkg,
                           Parser::ParsePackage(input));
  XLS_ASSERT_OK_AND_ASSIGN(FunctionBase * f, pkg->GetProc("example"));
  EXPECT_EQ(f->GetInitiationInterval(), 12);
}

TEST(IrParserTest, ParseNonexistentAttributeFunction) {
  std::string input = R"(package test

#[foobar(12)]
top fn example() -> bits[32] {
  ret literal.1: bits[32] = literal(value=2, id=1)
}
)";
  EXPECT_THAT(Parser::ParsePackage(input),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Invalid attribute for function: foobar")));
}

TEST(IrParserTest, ParseNonexistentAttributeProc) {
  std::string input = R"(package test

#[foobar(12)]
top proc example(tkn: token, init={token}) {
  next (tkn)
}
)";
  EXPECT_THAT(Parser::ParsePackage(input),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Invalid attribute for proc: foobar")));
}

TEST(IrParserTest, ParseTrailingAttribute) {
  std::string input = R"(package test

#[foobar(12)]
)";
  EXPECT_THAT(Parser::ParsePackage(input),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Illegal attribute at end of file")));
}

TEST(IrParserTest, ParseBlockAttribute) {
  std::string input = R"(package test

#[foobar(12)]
block example(in: bits[32], out: bits[32]) {
  in: bits[32] = input_port(name=in, id=2)
  out: () = output_port(in, name=out, id=5)
}
)";
  EXPECT_THAT(
      Parser::ParsePackage(input),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("Attribute foobar is not supported on blocks.")));
}

TEST(IrParserTest, ParseBlockAttributeInitiationInterval) {
  std::string input = R"(package test

#[initiation_interval(12)]
block example(in: bits[32], out: bits[32]) {
  in: bits[32] = input_port(name=in, id=2)
  out: () = output_port(in, name=out, id=5)
}
)";
  XLS_EXPECT_OK(Parser::ParsePackage(input));
}

TEST(IrParserTest, ParseChannelAttribute) {
  std::string input = R"(package test

#[foobar(12)]
chan ch(bits[32], id=0, kind=streaming, ops=send_receive, flow_control=none, metadata="""""")
)";
  EXPECT_THAT(
      Parser::ParsePackage(input),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          HasSubstr("Attributes are not supported on channel declarations.")));
}

TEST(IrParserTest, ParseFileNumberAttribute) {
  std::string input = R"(package test

#[foobar(12)]
file_number 0 "fake_file.x"
)";
  EXPECT_THAT(
      Parser::ParsePackage(input),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          HasSubstr(
              "Attributes are not supported on file number declarations.")));
}

TEST(IrParserTest, ParseValidFifoInstantiation) {
  constexpr std::string_view ir_text = R"(package test

block my_block(in: bits[32], out: bits[32]) {
  in: bits[32] = input_port(name=in)
  instantiation my_inst(data_type=bits[32], depth=3, bypass=true, register_push_outputs=false, register_pop_outputs=false, kind=fifo)
  in_inst_input: () = instantiation_input(in, instantiation=my_inst, port_name=push_data)
  pop_data_inst_output: bits[32] = instantiation_output(instantiation=my_inst, port_name=pop_data)
  out_output_port: () = output_port(pop_data_inst_output, name=out)
}
)";
  XLS_EXPECT_OK(Parser::ParsePackage(ir_text));
}

TEST(IrParserTest, ParseInvalidFifoInstantiation) {
  for (std::string_view inst_args :
       {"data_type=bits[32], depth=3", "data_type=bits[32], bypass=true",
        "depth=3, bypass=true"}) {
    std::string ir_text = absl::StrFormat(R"(package test

block my_block(in: bits[32], out: bits[32]) {
  in: bits[32] = input_port(name=in)
  instantiation my_inst(%s, kind=fifo)
  in_inst_input: () = instantiation_input(in, instantiation=my_inst, port_name=push_data)
  pop_data_inst_output: bits[32] = instantiation_output(instantiation=my_inst, port_name=pop_data)
  out_output_port: () = output_port(pop_data_inst_output, name=out)
}
)",
                                          inst_args);
    EXPECT_THAT(Parser::ParsePackage(ir_text),
                StatusIs(absl::StatusCode::kInvalidArgument,
                         HasSubstr("Instantiated fifo must specify")));
  }
}

}  // namespace xls
