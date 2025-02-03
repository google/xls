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

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status.h"
#include "absl/strings/ascii.h"
#include "absl/strings/substitute.h"
#include "xls/common/casts.h"
#include "xls/common/source_location.h"
#include "xls/common/status/matchers.h"
#include "xls/ir/bits.h"
#include "xls/ir/bits_ops.h"
#include "xls/ir/block.h"
#include "xls/ir/channel.h"
#include "xls/ir/channel.pb.h"
#include "xls/ir/channel_ops.h"
#include "xls/ir/nodes.h"
#include "xls/ir/op.h"
#include "xls/ir/package.h"
#include "xls/ir/state_element.h"
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

TEST(IrParserTest, ParseSingleEmptyPackage) {
  std::string input = R"(package EmptyPackage)";
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> package,
                           Parser::ParsePackage(input));
  EXPECT_EQ(package->name(), "EmptyPackage");
  EXPECT_EQ(0, package->functions().size());
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
  EXPECT_EQ(proc->node_count(), 4);
  EXPECT_EQ(proc->StateElements().size(), 2);
  EXPECT_EQ(proc->GetStateElement(0)->initial_value().ToString(), "token");
  EXPECT_EQ(proc->GetStateElement(0)->type()->ToString(), "token");
  EXPECT_EQ(proc->GetStateElement(1)->initial_value().ToString(),
            "bits[32]:42");
  EXPECT_EQ(proc->GetStateElement(1)->type()->ToString(), "bits[32]");
  EXPECT_EQ(proc->GetStateElement(0)->name(), "my_token");
  EXPECT_EQ(proc->GetStateElement(1)->name(), "my_state");
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
  EXPECT_THAT(proc->StateElements(), IsEmpty());
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
  EXPECT_THAT(proc->StateElements(), IsEmpty());
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

TEST(IrParserTest, ProcWithExplicitStateRead) {
  std::string program = R"(
package test

proc foo( x: bits[32], y: (), z: bits[32], init={42, (), 123}) {
  x: bits[32] = state_read(state_element=x)
  sum: bits[32] = add(x, z)
  next (x, y, sum)
}
)";
  XLS_ASSERT_OK_AND_ASSIGN(auto package, Parser::ParsePackage(program));
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, package->GetProc("foo"));
  EXPECT_EQ(proc->GetStateElementCount(), 3);
  XLS_ASSERT_OK_AND_ASSIGN(StateElement * x, proc->GetStateElement("x"));
  EXPECT_THAT(proc->GetStateRead(x)->predicate(), std::nullopt);
}

TEST(IrParserTest, ProcWithPredicatedStateRead) {
  std::string program = R"(
package test

proc foo( x: bits[32], y: bits[1], z: bits[32], init={42, 1, 123}) {
  x: bits[32] = state_read(state_element=x, predicate=y)
  z: bits[32] = state_read(state_element=z)
  sum: bits[32] = add(x, z)
  next (x, y, sum)
}
)";
  XLS_ASSERT_OK_AND_ASSIGN(auto package, Parser::ParsePackage(program));
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, package->GetProc("foo"));
  EXPECT_EQ(proc->GetStateElementCount(), 3);

  XLS_ASSERT_OK_AND_ASSIGN(StateElement * x, proc->GetStateElement("x"));
  std::optional<Node*> x_predicate = proc->GetStateRead(x)->predicate();
  ASSERT_TRUE(x_predicate.has_value());
  ASSERT_EQ((*x_predicate)->op(), Op::kStateRead);
  EXPECT_EQ((*x_predicate)->As<StateRead>()->state_element()->name(), "y");

  XLS_ASSERT_OK_AND_ASSIGN(StateElement * y, proc->GetStateElement("y"));
  ASSERT_FALSE(proc->GetStateRead(y)->predicate().has_value());

  XLS_ASSERT_OK_AND_ASSIGN(StateElement * z, proc->GetStateElement("z"));
  ASSERT_FALSE(proc->GetStateRead(z)->predicate().has_value());
}

TEST(IrParserTest, ParseSendReceiveChannel) {
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

  EXPECT_FALSE(p.ChannelsAreProcScoped());
}

TEST(IrParserTest, ParseSendReceiveChannelWithInitialValues) {
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

TEST(IrParserTest, ParseSendReceiveChannelWithTupleType) {
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

TEST(IrParserTest, ParseSendOnlyChannel) {
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

TEST(IrParserTest, ParseReceiveOnlyChannel) {
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

TEST(IrParserTest, ParseStreamingChannelWithExtraFifoMetadataNoFlops) {
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

TEST(IrParserTest, ParseStreamingChannelWithExtraFifoMetadata) {
  Package p("my_package");
  XLS_ASSERT_OK_AND_ASSIGN(Channel * ch,
                           Parser::ParseChannel(
                               R"(chan foo(bits[32], id=42, kind=streaming,
                         flow_control=none, ops=send_receive, fifo_depth=3,
                         input_flop_kind=skid, output_flop_kind=zero_latency,
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
  EXPECT_EQ(
      down_cast<StreamingChannel*>(ch)->channel_config().input_flop_kind(),
      FlopKind::kSkid);
  EXPECT_EQ(
      down_cast<StreamingChannel*>(ch)->channel_config().output_flop_kind(),
      FlopKind::kZeroLatency);
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

TEST(IrParserTest, ParseSingleValueChannelWithBlockPortMapping) {
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

TEST(IrParserTest, PackageWithSingleDataElementChannels) {
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

TEST(IrParserTest, NodeNames) {
  std::string program = R"(package test

fn foo(x: bits[32] id=2, foobar: bits[32] id=3) -> bits[32] {
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
  EXPECT_EQ(x->id(), 2);

  Node* foobar = f->param(1);
  EXPECT_TRUE(foobar->HasAssignedName());
  EXPECT_EQ(foobar->GetName(), "foobar");
  EXPECT_EQ(foobar->id(), 3);

  Node* add = f->return_value()->operand(0);
  EXPECT_FALSE(add->HasAssignedName());
  EXPECT_EQ(add->GetName(), "add.1");
  EXPECT_EQ(add->id(), 1);

  Node* qux = f->return_value();
  EXPECT_TRUE(qux->HasAssignedName());
  EXPECT_EQ(qux->GetName(), "qux");
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

  EXPECT_FALSE(pkg->ChannelsAreProcScoped());
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

TEST(IrParserTest, ParseIIFunction) {
  std::string input = R"(package test

#[initiation_interval(12)]
top fn example() -> bits[32] {
  ret literal.1: bits[32] = literal(value=2, id=1)
}
)";
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
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> pkg,
                           Parser::ParsePackage(input));
  XLS_ASSERT_OK_AND_ASSIGN(FunctionBase * f, pkg->GetProc("example"));
  EXPECT_EQ(f->GetInitiationInterval(), 12);
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

TEST(IrParserTest, ParseWithUnspecifiedIds) {
  std::string input = R"(package test

 top fn example() -> bits[32] {
   node0: bits[32] = literal(value=2)
   node1: bits[32] = literal(value=2, id=1)
   node2: bits[32] = literal(value=2)
   ret node3: bits[32] = literal(value=2, id=2)
 }
)";
  XLS_ASSERT_OK(Parser::ParsePackage(input).status());
}

TEST(IrParserTest, TrivialNewStyleProc) {
  const std::string input = R"(package test

top proc my_proc<>() {
}

)";
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> pkg,
                           Parser::ParsePackage(input));
  EXPECT_TRUE(pkg->ChannelsAreProcScoped());
}

TEST(IrParserTest, FfiAttribute) {
  const std::string input = R"(
package some_package

#[ffi_proto("""code_template: "verilog_module {fn} (.a({x}), .b({y}), .out({return}));"
""")]
fn ffi_callee(x: bits[32], y: bits[32]) -> bits[32] {
  ret add.1: bits[32] = add(x, y)
}
)";
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> pkg,
                           Parser::ParsePackage(input));
  XLS_ASSERT_OK_AND_ASSIGN(FunctionBase * f, pkg->GetFunction("ffi_callee"));
  ASSERT_TRUE(f->ForeignFunctionData().has_value());
  EXPECT_EQ(f->ForeignFunctionData()->code_template(),
            "verilog_module {fn} (.a({x}), .b({y}), .out({return}));");
  EXPECT_FALSE(f->ForeignFunctionData()->has_delay_ps());
}

TEST(IrParserTest, ParseChannelPortMetadata) {
  constexpr std::string_view input = R"(package test

block my_block(in: bits[32], in_valid: bits[1], in_ready: bits[1],
               out: bits[32]) {
  #![channel_ports(name=foo, type=bits[32], direction=in, kind=streaming, flop=skid, data_port=in, ready_port=in_ready, valid_port=in_valid)]
  #![channel_ports(name=bar, type=bits[32], direction=out, kind=single_value, data_port=out)]
  in: bits[32] = input_port(name=in)
  in_valid: bits[1] = input_port(name=in_valid)
  data: bits[32] = literal(value=42)
  one: bits[1] = literal(value=1)
  out_port: () = output_port(data, name=out)
  in_ready_port: () = output_port(one, name=in_ready)
}
)";
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> pkg,
                           Parser::ParsePackage(input));
  XLS_ASSERT_OK_AND_ASSIGN(Block * b, pkg->GetBlock("my_block"));

  XLS_ASSERT_OK_AND_ASSIGN(ChannelPortMetadata foo_metadata,
                           b->GetChannelPortMetadata("foo"));
  EXPECT_EQ(foo_metadata.channel_name, "foo");
  EXPECT_EQ(foo_metadata.type, pkg->GetBitsType(32));
  EXPECT_EQ(foo_metadata.direction, PortDirection::kInput);
  EXPECT_EQ(foo_metadata.channel_kind, ChannelKind::kStreaming);
  EXPECT_EQ(foo_metadata.flop_kind, FlopKind::kSkid);
  EXPECT_THAT(foo_metadata.data_port, Optional(Eq("in")));
  EXPECT_THAT(foo_metadata.ready_port, Optional(Eq("in_ready")));
  EXPECT_THAT(foo_metadata.valid_port, Optional(Eq("in_valid")));

  XLS_ASSERT_OK_AND_ASSIGN(ChannelPortMetadata bar_metadata,
                           b->GetChannelPortMetadata("bar"));
  EXPECT_EQ(bar_metadata.channel_name, "bar");
  EXPECT_EQ(bar_metadata.type, pkg->GetBitsType(32));
  EXPECT_EQ(bar_metadata.direction, PortDirection::kOutput);
  EXPECT_EQ(bar_metadata.channel_kind, ChannelKind::kSingleValue);
  EXPECT_EQ(bar_metadata.flop_kind, FlopKind::kNone);
  EXPECT_THAT(bar_metadata.data_port, Optional(Eq("out")));
  EXPECT_EQ(bar_metadata.ready_port, std::nullopt);
  EXPECT_EQ(bar_metadata.valid_port, std::nullopt);
}

}  // namespace xls
