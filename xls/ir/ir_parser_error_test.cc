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

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "xls/common/casts.h"
#include "xls/common/status/matchers.h"
#include "xls/ir/bits.h"
#include "xls/ir/channel.h"
#include "xls/ir/channel.pb.h"
#include "xls/ir/channel_ops.h"
#include "xls/ir/ir_parser.h"
#include "xls/ir/nodes.h"
#include "xls/ir/op.h"
#include "xls/ir/package.h"
#include "xls/ir/type.h"
#include "xls/ir/value.h"

namespace xls {

using ::absl_testing::StatusIs;
using ::testing::ElementsAre;
using ::testing::Eq;
using ::testing::HasSubstr;
using ::testing::IsEmpty;
using ::testing::Not;
using ::testing::Optional;
using ::testing::Property;

TEST(IrParserErrorTest, DuplicateKeywordArgs) {
  Package p("my_package");
  const std::string input =
      R"(fn f() -> bits[37] {
  ret literal.1: bits[37] = literal(value=42, value=123)
})";
  EXPECT_THAT(Parser::ParseFunction(input, &p).status(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Duplicate keyword argument `value`")));
}

TEST(IrParserErrorTest, WrongDeclaredNodeType) {
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

TEST(IrParserErrorTest, WrongFunctionReturnType) {
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

TEST(IrParserErrorTest, MissingMandatoryKeyword) {
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

TEST(IrParserErrorTest, UndefinedOperand) {
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

TEST(IrParserErrorTest, InvalidOp) {
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

TEST(IrParserErrorTest, PositionalArgumentAfterKeywordArgument) {
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

TEST(IrParserErrorTest, ExtraOperands) {
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

TEST(IrParserErrorTest, TooFewOperands) {
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

TEST(IrParserErrorTest, DuplicateName) {
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

TEST(IrParserErrorTest, CountedForMissingBody) {
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

TEST(IrParserErrorTest, CountedForBodyParamCountTooMany0) {
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

TEST(IrParserErrorTest, CountedForBodyParamCountTooMany1) {
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

TEST(IrParserErrorTest, CountedForBodyParamCountTooFew0) {
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

TEST(IrParserErrorTest, CountedForBodyParamCountTooFew1) {
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

TEST(IrParserErrorTest, CountedForBodyParamCountTooFew2) {
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

TEST(IrParserErrorTest, CountedForBodyBitWidthInsufficient) {
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

TEST(IrParserErrorTest, CountedForBodyBitWidthInsufficientWithStride) {
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

TEST(IrParserErrorTest, CountedForBodyBitWidthTypeMismatch0) {
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

TEST(IrParserErrorTest, CountedForBodyBitWidthTypeMismatch1) {
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

TEST(IrParserErrorTest, CountedForBodyDataTypeMismatch) {
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

TEST(IrParserErrorTest, CountedForReturnTypeMismatch) {
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

TEST(IrParserErrorTest, CountedForBodyInvariantArgTypeMismatch0) {
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

TEST(IrParserErrorTest, CountedForBodyInvariantArgTypeMismatch1) {
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

TEST(IrParserErrorTest, ParseAfterAllNonToken) {
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

TEST(IrParserErrorTest, ParseMinDelayNonToken) {
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

TEST(IrParserErrorTest, ParseMinDelayNegative) {
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

TEST(IrParserErrorTest, EmptyBitsBounds) {
  Package p("my_package");
  std::string input = R"(fn f() -> bits[] {
  ret literal.1: bits[] = literal(value=0, id=1)
})";
  EXPECT_THAT(Parser::ParseFunction(input, &p).status(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Expected token of type \"literal\"")));
}

TEST(IrParserErrorTest, ParsePackageWithError) {
  std::string input = R"(package MultiFunctionPackage

Garbage
)";
  EXPECT_THAT(Parser::ParsePackage(input).status(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Expected attribute or declaration")));
}

TEST(IrParserErrorTest, ParseEmptyStringAsPackage) {
  EXPECT_THAT(Parser::ParsePackage("").status(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Expected token, but found EOF")));
}

TEST(IrParserErrorTest, ParsePackageWithMissingPackageLine) {
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

TEST(IrParserErrorTest, ParseTraceWrongOperands) {
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

TEST(IrParserErrorTest, ParseTraceNegativeVerbosity) {
  const std::string input = R"(package foobar

fn bar(tkn: token, cond: bits[1], x: bits[3], y: bits[7]) -> token {
  ret trace.1: token = trace(tkn, cond, format="x is {}", data_operands=[x], verbosity=-1, id=1)
}
)";
  EXPECT_THAT(Parser::ParsePackage(input).status(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Verbosity must be >= 0: got -1")));
}

TEST(IrParserErrorTest, ParseProcWithMixedNextValueStyles) {
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

TEST(IrParserErrorTest, ParseProcWithBadNextParam) {
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

TEST(IrParserErrorTest, ParseProcWithBadNextValueType) {
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
                       HasSubstr("next value for state element 'my_state' must "
                                 "be of type bits[32]; is: bits[1]")));
}

TEST(IrParserErrorTest, NewStyleProcSendOnInput) {
  Package p("my_package");
  const std::string input =
      R"(proc my_proc<in_ch: bits[32] in kind=streaming>(my_token: token, my_state: bits[32], init={token, 42}) {
  send: token = send(my_token, my_state, channel=in_ch)
  next (send, my_state)
}
)";
  EXPECT_THAT(Parser::ParseProc(input, &p).status(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Cannot send on channel `in_ch`")));
}

TEST(IrParserErrorTest, NewStyleProcReceiveOnOutput) {
  Package p("my_package");
  const std::string input =
      R"(proc my_proc<out_ch: bits[32] out kind=streaming>(my_token: token, my_state: bits[32], init={token, 42}) {
  rcv: (token, bits[32]) = receive(my_token, channel=out_ch)
  rcv_token: token = tuple_index(rcv, index=0)
  next (rcv_token, my_state)
}
)";
  EXPECT_THAT(Parser::ParseProc(input, &p).status(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Cannot receive on channel `out_ch`")));
}

TEST(IrParserErrorTest, InstantiateNonexistentProc) {
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

TEST(IrParserErrorTest, InstantiateOldStyleProc) {
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

TEST(IrParserErrorTest, ProcInstantiationWrongNumberOfArguments) {
  const std::string input = R"(package test

proc my_proc<in_ch: bits[32] in kind=streaming, out_ch: bits[32] out kind=streaming>(my_token: token, my_state: bits[32], init={token, 42}) {
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

TEST(IrParserErrorTest, ProcInstantiationInOldStyleProc) {
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

TEST(IrParserErrorTest, DirectionMismatchInInstantiatedProc) {
  const std::string input = R"(package test

proc my_proc<in_ch: bits[32] in kind=streaming, out_ch: bits[32] out kind=streaming>(my_token: token, my_state: bits[32], init={token, 42}) {
  send.1: token = send(my_token, my_state, channel=out_ch, id=1)
  literal.2: bits[1] = literal(value=1, id=2)
  receive.3: (token, bits[32]) = receive(send.1, predicate=literal.2, channel=in_ch, id=3)
  tuple_index.4: token = tuple_index(receive.3, index=0, id=4)
  next (tuple_index.4, my_state)
}

proc other_proc<in_ch: bits[32] in kind=streaming, out_ch: bits[32] out kind=streaming>(my_token: token, my_state: bits[32], init={token, 42}) {
  proc_instantiation foo(out_ch, in_ch, proc=my_proc)
  next (my_token, my_state)
}
)";
  EXPECT_THAT(Parser::ParsePackage(input),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("No such receive channel `out_ch` for proc "
                                 "instantiation arg 0")));
}

TEST(IrParserErrorTest, DeclareChannelInOldStyleProc) {
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

TEST(IrParserErrorTest, DeclareChannelInFunction) {
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

TEST(IrParserErrorTest, NewStyleProcUsingGlobalChannel) {
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

TEST(IrParserErrorTest, NewStyleProcWithDuplicateChannelNames) {
  Package p("my_package");
  const std::string input =
      R"(proc my_proc<ch: bits[32] in kind=streaming, ch: bits[32] out kind=streaming>(my_token: token, my_state: bits[32], init={token, 42}) {
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

TEST(IrParserErrorTest, InstantiatedProcWithUnknownChannel) {
  const std::string input = R"(package test

proc my_proc<in_ch: bits[32] in kind=streaming, out_ch: bits[32] out kind=streaming>(my_token: token, my_state: bits[32], init={token, 42}) {
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

TEST(IrParserErrorTest, ParseArrayUpdateNonArary) {
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

TEST(IrParserErrorTest, ParseArrayUpdateIncompatibleTypes) {
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

TEST(IrParserErrorTest, ParseArrayConcatNonArrayType) {
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

TEST(IrParserErrorTest, ParseArrayIncompatibleElementType) {
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

TEST(IrParserErrorTest, ParseArrayIncompatibleReturnType) {
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

TEST(IrParserErrorTest, StandAloneRet) {
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

TEST(IrParserErrorTest, ParseEndOfLineComment) {
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

TEST(IrParserErrorTest, ParseTupleType) {
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

TEST(IrParserErrorTest, ParseEmptyTuple) {
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

TEST(IrParserErrorTest, ParseNestedTuple) {
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

TEST(IrParserErrorTest, ParseArrayLiteralWithInsufficientBits) {
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

TEST(IrParserErrorTest, ReturnArrayLiteral) {
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

TEST(IrParserErrorTest, ReturnArrayOfTuplesLiteral) {
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

TEST(IrParserErrorTest, ArrayValueInBitsLiteral) {
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

TEST(IrParserErrorTest, BitsValueInArrayLiteral) {
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

TEST(IrParserErrorTest, ParseInconsistentExtendOp) {
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

TEST(IrParserErrorTest, ArrayIndexOfTuple) {
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

TEST(IrParserErrorTest, TupleIndexOfArray) {
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

TEST(IrParserErrorTest, NicerErrorOnEmptyString) {
  const std::string input = "";  // NOLINT: emphasize empty string here.
  EXPECT_THAT(
      Parser::ParsePackage(input).status(),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          HasSubstr(
              "Expected keyword 'package': Expected token, but found EOF.")));
}

TEST(IrParserErrorTest, ParsesComplexValue) {
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

TEST(IrParserErrorTest, ParsesComplexValueWithEmbeddedTypes) {
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

TEST(IrParserErrorTest, ParsesTokenType) {
  const std::string input = "token";
  XLS_ASSERT_OK_AND_ASSIGN(Value v, Parser::ParseTypedValue(input));
  Value expected = Value::Token();
  EXPECT_EQ(expected, v);
}

TEST(IrParserErrorTest, ParsesComplexValueWithEmbeddedTokens) {
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
TEST(IrParserErrorTest, DISABLED_ParsesEmptyArray) {
  const std::string input = "[]";
  Package p("test_package");
  auto* u1 = p.GetBitsType(1);
  auto* array_0xu1 = p.GetArrayType(0, u1);
  XLS_ASSERT_OK_AND_ASSIGN(Value v, Parser::ParseValue(input, array_0xu1));
  Value expected = Value::ArrayOrDie({});
  EXPECT_EQ(expected, v);
}

TEST(IrParserErrorTest, BigOrdinalAnnotation) {
  std::string program = R"(
package test

fn main() -> bits[1] {
  ret literal.1000: bits[1] = literal(value=0, id=1000)
}
)";
  XLS_ASSERT_OK_AND_ASSIGN(auto package, Parser::ParsePackage(program));
  EXPECT_GT(package->next_node_id(), 1000);
}

TEST(IrParserErrorTest, TrivialProc) {
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
  EXPECT_EQ(proc->node_count(), 4);
  EXPECT_EQ(proc->StateElements().size(), 2);
  EXPECT_EQ(proc->GetStateElement(int64_t{0})->initial_value().ToString(),
            "token");
  EXPECT_EQ(proc->GetStateElement(int64_t{0})->type()->ToString(), "token");
  EXPECT_EQ(proc->GetStateElement(1)->initial_value().ToString(),
            "bits[32]:42");
  EXPECT_EQ(proc->GetStateElement(1)->type()->ToString(), "bits[32]");
  EXPECT_EQ(proc->GetStateElement(int64_t{0})->name(), "my_token");
  EXPECT_EQ(proc->GetStateElement(1)->name(), "my_state");
}

TEST(IrParserErrorTest, StatelessProcWithInitAndNext) {
  std::string program = R"(
package test

proc foo(init={}) {
  next ()
}
)";
  XLS_ASSERT_OK_AND_ASSIGN(auto package, Parser::ParsePackage(program));
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, package->GetProc("foo"));
  EXPECT_EQ(proc->node_count(), 0);
  EXPECT_THAT(proc->StateElements(), IsEmpty());
}

TEST(IrParserErrorTest, StatelessProcWithNextButNotInit) {
  std::string program = R"(
package test

proc foo() {
  next ()
}
)";
  XLS_ASSERT_OK_AND_ASSIGN(auto package, Parser::ParsePackage(program));
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, package->GetProc("foo"));
  EXPECT_EQ(proc->node_count(), 0);
  EXPECT_THAT(proc->StateElements(), IsEmpty());
}

TEST(IrParserErrorTest, FunctionAndProc) {
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

TEST(IrParserErrorTest, ProcWithMultipleStateElements) {
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

TEST(IrParserErrorTest, ProcWithTokenAfterStateElements) {
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

TEST(IrParserErrorTest, ProcWithTokenBetweenStateElements) {
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

TEST(IrParserErrorTest, ProcWithMultipleTokens) {
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

TEST(IrParserErrorTest, ProcTooFewInitialValues) {
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

TEST(IrParserErrorTest, ProcTooManyInitialValues) {
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

TEST(IrParserErrorTest, ProcWithMissingInitValues) {
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

TEST(IrParserErrorTest, ProcWithTooFewNextStateElements) {
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

TEST(IrParserErrorTest, ProcWithTooManyNextStateElements) {
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

TEST(IrParserErrorTest, ProcWrongInitValueType) {
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

TEST(IrParserErrorTest, ProcWrongInitValueTypeToken) {
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

TEST(IrParserErrorTest, ProcWrongReturnType) {
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
               HasSubstr("Recurrent state type token does not match provided "
                         "state type bits[32] for element 0.")));
}

TEST(IrParserErrorTest, ProcWithRet) {
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

TEST(IrParserErrorTest, FunctionWithNext) {
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

TEST(IrParserErrorTest, ProcWithBogusNextToken) {
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

TEST(IrParserErrorTest, ProcWithBogusNextState) {
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

TEST(IrParserErrorTest, ParseSendReceiveChannel) {
  Package p("my_package");
  XLS_ASSERT_OK_AND_ASSIGN(Channel * ch,
                           Parser::ParseChannel(
                               R"(chan foo(bits[32], id=42, kind=single_value,
                      ops=send_receive,
                      metadata=""))",
                               &p));
  EXPECT_EQ(ch->name(), "foo");
  EXPECT_EQ(ch->id(), 42);
  EXPECT_EQ(ch->supported_ops(), ChannelOps::kSendReceive);
  EXPECT_EQ(ch->kind(), ChannelKind::kSingleValue);
  EXPECT_EQ(ch->type(), p.GetBitsType(32));
  EXPECT_TRUE(ch->initial_values().empty());
}

TEST(IrParserErrorTest, ParseSendReceiveChannelWithInitialValues) {
  Package p("my_package");
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * ch,
      Parser::ParseChannel(
          R"(chan foo(bits[32], initial_values={2, 4, 5}, id=42, kind=streaming,
                         flow_control=none, ops=send_receive,
                         metadata=""))",
          &p));
  EXPECT_EQ(ch->name(), "foo");
  EXPECT_EQ(ch->id(), 42);
  EXPECT_EQ(ch->supported_ops(), ChannelOps::kSendReceive);
  EXPECT_EQ(ch->kind(), ChannelKind::kStreaming);
  EXPECT_EQ(ch->type(), p.GetBitsType(32));
  EXPECT_THAT(ch->initial_values(),
              ElementsAre(Value(UBits(2, 32)), Value(UBits(4, 32)),
                          Value(UBits(5, 32))));
}

TEST(IrParserErrorTest, ParseSendReceiveChannelWithTupleType) {
  Package p("my_package");
  XLS_ASSERT_OK_AND_ASSIGN(Channel * ch, Parser::ParseChannel(
                                             R"(chan foo((bits[32], bits[1]),
                      initial_values={(123, 1), (42, 0)},
                      id=42, kind=streaming, flow_control=ready_valid,
                      ops=send_receive,
                      metadata=""))",
                                             &p));
  EXPECT_EQ(ch->name(), "foo");
  EXPECT_THAT(
      ch->initial_values(),
      ElementsAre(Value::Tuple({Value(UBits(123, 32)), Value(UBits(1, 1))}),
                  Value::Tuple({Value(UBits(42, 32)), Value(UBits(0, 1))})));
}

TEST(IrParserErrorTest, ParseSendOnlyChannel) {
  Package p("my_package");
  XLS_ASSERT_OK_AND_ASSIGN(Channel * ch, Parser::ParseChannel(
                                             R"(chan bar((bits[32], bits[1]),
                         id=7, kind=single_value, ops=send_only,
                         metadata=""))",
                                             &p));
  EXPECT_EQ(ch->name(), "bar");
  EXPECT_EQ(ch->id(), 7);
  EXPECT_EQ(ch->supported_ops(), ChannelOps::kSendOnly);
  EXPECT_EQ(ch->type(), p.GetTupleType({p.GetBitsType(32), p.GetBitsType(1)}));
}

TEST(IrParserErrorTest, ParseReceiveOnlyChannel) {
  Package p("my_package");
  XLS_ASSERT_OK_AND_ASSIGN(Channel * ch, Parser::ParseChannel(
                                             R"(chan meh(bits[32][4], id=0,
                         kind=single_value, ops=receive_only,
                         metadata=""))",
                                             &p));
  EXPECT_EQ(ch->name(), "meh");
  EXPECT_EQ(ch->id(), 0);
  EXPECT_EQ(ch->supported_ops(), ChannelOps::kReceiveOnly);
  EXPECT_EQ(ch->type(), p.GetArrayType(4, p.GetBitsType(32)));
}

TEST(IrParserErrorTest, ParseStreamingChannelWithStrictness) {
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

TEST(IrParserErrorTest, ParseStreamingChannelWithExtraFifoMetadata) {
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
  ASSERT_THAT(down_cast<StreamingChannel*>(ch)->channel_config().fifo_config(),
              Not(Eq(std::nullopt)));
  EXPECT_EQ(
      down_cast<StreamingChannel*>(ch)->channel_config().fifo_config()->depth(),
      3);
  EXPECT_EQ(down_cast<StreamingChannel*>(ch)
                ->channel_config()
                .fifo_config()
                ->bypass(),
            false);
}

TEST(IrParserErrorTest, ParseStreamingValueChannelWithBlockPortMapping) {
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

    EXPECT_THAT(ch->metadata().block_ports(),
                ElementsAre(AllOf(
                    Property(&BlockPortMappingProto::block_name, "blk"),
                    Property(&BlockPortMappingProto::data_port_name, "data"),
                    Property(&BlockPortMappingProto::valid_port_name, "vld"),
                    Property(&BlockPortMappingProto::ready_port_name, "rdy"))));

    ch_ir_text = ch->ToString();
  }

  {
    Package p("my_package_2");
    XLS_ASSERT_OK_AND_ASSIGN(Channel * ch,
                             Parser::ParseChannel(ch_ir_text, &p));
    EXPECT_EQ(ch->name(), "meh");

    EXPECT_EQ(ch->id(), 0);

    EXPECT_EQ(ch->supported_ops(), ChannelOps::kSendOnly);

    EXPECT_THAT(ch->metadata().block_ports(),
                ElementsAre(AllOf(
                    Property(&BlockPortMappingProto::block_name, "blk"),
                    Property(&BlockPortMappingProto::data_port_name, "data"),
                    Property(&BlockPortMappingProto::valid_port_name, "vld"),
                    Property(&BlockPortMappingProto::ready_port_name, "rdy"))));
  }
}

TEST(IrParserErrorTest, ParseSingleValueChannelWithBlockPortMapping) {
  // For testing round-trip parsing.
  std::string ch_ir_text;

  {
    Package p("my_package");
    XLS_ASSERT_OK_AND_ASSIGN(Channel * ch, Parser::ParseChannel(
                                               R"(chan meh(bits[32][4], id=0,
                         kind=single_value, ops=receive_only,
                         metadata="""block_ports { data_port_name : "data",
                                                   block_name : "blk"}"""))",
                                               &p));
    EXPECT_EQ(ch->name(), "meh");
    EXPECT_EQ(ch->id(), 0);
    EXPECT_EQ(ch->supported_ops(), ChannelOps::kReceiveOnly);

    EXPECT_THAT(
        ch->metadata().block_ports(),
        ElementsAre(AllOf(
            Property(&BlockPortMappingProto::block_name, "blk"),
            Property(&BlockPortMappingProto::data_port_name, "data"),
            Property(&BlockPortMappingProto::has_valid_port_name, false),
            Property(&BlockPortMappingProto::has_ready_port_name, false))));

    ch_ir_text = ch->ToString();
  }

  {
    Package p("my_package_2");
    XLS_ASSERT_OK_AND_ASSIGN(Channel * ch,
                             Parser::ParseChannel(ch_ir_text, &p));
    EXPECT_EQ(ch->name(), "meh");

    EXPECT_EQ(ch->id(), 0);
    EXPECT_EQ(ch->supported_ops(), ChannelOps::kReceiveOnly);

    EXPECT_THAT(
        ch->metadata().block_ports(),
        ElementsAre(AllOf(
            Property(&BlockPortMappingProto::block_name, "blk"),
            Property(&BlockPortMappingProto::data_port_name, "data"),
            Property(&BlockPortMappingProto::has_valid_port_name, false),
            Property(&BlockPortMappingProto::has_ready_port_name, false))));
  }
}

TEST(IrParserErrorTest, ChannelParsingErrors) {
  Package p("my_package");
  EXPECT_THAT(Parser::ParseChannel(
                  R"(chan meh(bits[32][4], kind=single_value,
                         ops=receive_only,
                         metadata=""))",
                  &p)
                  .status(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Mandatory keyword argument `id` not found")));

  EXPECT_THAT(
      Parser::ParseChannel(
          R"(chan meh(bits[32][4], id=42, ops=receive_only,
                         metadata=""))",
          &p)
          .status(),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("Mandatory keyword argument `kind` not found")));

  EXPECT_THAT(Parser::ParseChannel(
                  R"(chan meh(bits[32][4], id=42, kind=bogus,
                         ops=receive_only,
                         metadata=""))",
                  &p)
                  .status(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Invalid channel kind \"bogus\"")));

  EXPECT_THAT(
      Parser::ParseChannel(
          R"(chan meh(bits[32][4], id=7, kind=streaming,
                         metadata=""))",
          &p)
          .status(),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("Mandatory keyword argument `ops` not found")));

  // Unrepresentable initial value.
  EXPECT_THAT(Parser::ParseChannel(
                  R"(chan meh(bits[4], initial_values={128}, kind=streaming,
                         ops=send_receive, id=7,
                         metadata=""))",
                  &p)
                  .status(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Value 128 is not representable in 4 bits")));

  // Wrong initial value type.
  EXPECT_THAT(Parser::ParseChannel(
                  R"(chan meh(bits[4], initial_values={(1, 2)}, kind=streaming,
                         ops=send_receive, id=7
                         metadata=""))",
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
                         metadata=""))",
                  &p)
                  .status(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Expected 'bits' keyword")));

  EXPECT_THAT(Parser::ParseChannel(
                  R"(chan meh(bits[32], id=44, kind=streaming,
                         ops=receive_only, bogus="totally!",
                         metadata=""))",
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

TEST(IrParserErrorTest, PackageWithSingleDataElementChannels) {
  std::string program = R"(
package test

chan hbo(bits[32], id=0, kind=streaming, flow_control=none, ops=receive_only,
            fifo_depth=42, metadata="")
chan mtv(bits[32], id=1, kind=streaming, flow_control=none, ops=send_only,
            metadata="")

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

TEST(IrParserErrorTest, ParseTupleIndexWithInvalidBValue) {
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

TEST(IrParserErrorTest, NodeNames) {
  std::string program = R"(package test

fn foo(x: bits[32] id=42, foobar: bits[32] id=43) -> bits[32] {
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
  EXPECT_EQ(x->id(), 42);

  Node* foobar = f->param(1);
  EXPECT_TRUE(foobar->HasAssignedName());
  EXPECT_EQ(foobar->GetName(), "foobar");
  EXPECT_EQ(foobar->id(), 43);

  Node* add = f->return_value()->operand(0);
  EXPECT_FALSE(add->HasAssignedName());
  EXPECT_EQ(add->GetName(), "add.1");
  EXPECT_EQ(add->id(), 1);

  Node* qux = f->return_value();
  EXPECT_TRUE(qux->HasAssignedName());
  EXPECT_EQ(qux->GetName(), "qux");
}

TEST(IrParserErrorTest, InvalidName) {
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

TEST(IrParserErrorTest, IdAttributes) {
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
}

TEST(IrParserErrorTest, MismatchedId) {
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

TEST(IrParserErrorTest, FunctionWithPort) {
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

TEST(IrParserErrorTest, BlockWithReturnValue) {
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

TEST(IrParserErrorTest, WriteOfNonexistentRegister) {
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

TEST(IrParserErrorTest, ReadOfNonexistentRegister) {
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

TEST(IrParserErrorTest, ParseBlockWithRegisterWithWrongResetValueType) {
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

TEST(IrParserErrorTest, RegisterInFunction) {
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

TEST(IrParserErrorTest, ParseBlockWithDuplicateRegisters) {
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

TEST(IrParserErrorTest, ParseBlockWithIncompleteResetDefinition) {
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

TEST(IrParserErrorTest, BlockWithRegistersButNoClock) {
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

TEST(IrParserErrorTest, BlockWithTwoClocks) {
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

TEST(IrParserErrorTest, BlockWithIncompletePortList) {
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

TEST(IrParserErrorTest, BlockWithExtraPort) {
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

TEST(IrParserErrorTest, BlockWithDuplicatePort) {
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

TEST(IrParserErrorTest, BlockWithInvalidRegisterField) {
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

TEST(IrParserErrorTest, ParseBlockWithMissingInstantiatedBlock) {
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

TEST(IrParserErrorTest, ParseBlockWithUnknownInstantiation) {
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

TEST(IrParserErrorTest, ParseBlockWithDuplicateInstantiationPort) {
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

TEST(IrParserErrorTest, ParseBlockWithMissingInstantiationPort) {
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

TEST(IrParserErrorTest, ParseBlockWithWronglyNamedInstantiationPort) {
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

TEST(IrParserErrorTest, ParseBlockWithWronglyTypedSignature) {
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

TEST(IrParserErrorTest, ParseTopFunction) {
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

TEST(IrParserErrorTest, ParseTopProc) {
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

TEST(IrParserErrorTest, ParseTopBlock) {
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

TEST(IrParserErrorTest, ParseWithTwoTops) {
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

TEST(IrParserErrorTest, ParseTopInvalidTopEntity) {
  const std::string input = R"(
package invalid_top_entity_package

top invalid_top_entity
)";
  EXPECT_THAT(
      Parser::ParsePackage(input),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("Expected fn, proc or block definition, got")));
}

TEST(IrParserErrorTest, ParseTopInvalidTopEntityWithKeyword) {
  const std::string input = R"(
package invalid_top_entity_package

top reg
)";
  EXPECT_THAT(
      Parser::ParsePackage(input),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("Expected fn, proc or block definition, got")));
}

TEST(IrParserErrorTest, ParseNonexistentAttributeFunction) {
  std::string input = R"(package test

#[foobar(12)]
top fn example() -> bits[32] {
  ret literal.1: bits[32] = literal(value=2, id=1)
}
)";
  EXPECT_THAT(Parser::ParsePackage(input),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Unknown attribute: foobar")));
}

TEST(IrParserErrorTest, ParseNonexistentAttributeProc) {
  std::string input = R"(package test

#[foobar(12)]
top proc example(tkn: token, init={token}) {
  next (tkn)
}
)";
  EXPECT_THAT(Parser::ParsePackage(input),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Unknown attribute: foobar")));
}

TEST(IrParserErrorTest, ParseTrailingAttribute) {
  std::string input = R"(package test

#[initiation_interval(12)]
)";
  EXPECT_THAT(Parser::ParsePackage(input),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Illegal attribute at end of file")));
}

TEST(IrParserErrorTest, ParseBlockAttribute) {
  std::string input = R"(package test

#[foobar(12)]
block example(in: bits[32], out: bits[32]) {
  in: bits[32] = input_port(name=in, id=2)
  out: () = output_port(in, name=out, id=5)
}
)";
  EXPECT_THAT(Parser::ParsePackage(input),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Unknown attribute: foobar")));
}

TEST(IrParserErrorTest, ParseBlockAttributeInitiationInterval) {
  std::string input = R"(package test

#[initiation_interval(12)]
block example(in: bits[32], out: bits[32]) {
  in: bits[32] = input_port(name=in, id=2)
  out: () = output_port(in, name=out, id=5)
}
)";
  XLS_EXPECT_OK(Parser::ParsePackage(input));
}

TEST(IrParserErrorTest, ParseChannelAttribute) {
  std::string input = R"(package test

#[initiation_interval(12)]
chan ch(bits[32], id=0, kind=streaming, ops=send_receive, flow_control=none, metadata="""""")
)";
  EXPECT_THAT(
      Parser::ParsePackage(input),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          HasSubstr("Attributes are not supported on channel declarations.")));
}

TEST(IrParserErrorTest, ParseFileNumberAttribute) {
  std::string input = R"(package test

#[initiation_interval(12)]
file_number 0 "fake_file.x"
)";
  EXPECT_THAT(
      Parser::ParsePackage(input),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          HasSubstr(
              "Attributes are not supported on file number declarations.")));
}

TEST(IrParserErrorTest, ParseValidFifoInstantiation) {
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

TEST(IrParserErrorTest, ParseInvalidFifoInstantiation) {
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

TEST(IrParserErrorTest, InvalidKeywordArgumentInParameter) {
  Package p("my_package");
  const std::string input =
      R"(fn f(x: bits[32] foo=bar) -> bits[32] {
  ret literal.1: bits[32] = literal(value=42, value=123)
})";
  EXPECT_THAT(Parser::ParseFunction(input, &p).status(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Invalid parameter keyword argument `foo`")));
}

TEST(IrParserErrorTest, HalfBakedKeywordArgumentInParameter) {
  Package p("my_package");
  const std::string input =
      R"(fn f(x: bits[32] id) -> bits[32] {
  ret literal.1: bits[32] = literal(value=42, value=123)
})";
  EXPECT_THAT(Parser::ParseFunction(input, &p).status(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Expected token of type \"=\"")));
}

TEST(IrParserErrorTest, MissingValueKeywordArgumentInParameter) {
  Package p("my_package");
  const std::string input =
      R"(fn f(x: bits[32] id=) -> bits[32] {
  ret literal.1: bits[32] = literal(value=42, value=123)
})";
  EXPECT_THAT(Parser::ParseFunction(input, &p).status(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Expected token of type \"literal\"")));
}

TEST(IrParserErrorTest, InvalidParameterId) {
  Package p("my_package");
  const std::string input =
      R"(fn f(x: bits[32] id=0) -> bits[32] {
  ret literal.1: bits[32] = literal(value=42, value=123)
})";
  EXPECT_THAT(
      Parser::ParseFunction(input, &p).status(),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("Invalid node id 0, must be greater than zero")));
}

TEST(IrParserErrorTest, InvalidNodeId) {
  Package p("my_package");
  const std::string input =
      R"(fn f(x: bits[32] id=123) -> bits[32] {
  ret myliteral: bits[32] = literal(value=42, id=-1)
})";
  EXPECT_THAT(
      Parser::ParseFunction(input, &p).status(),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("Invalid node id -1, must be greater than zero")));
}

}  // namespace xls
