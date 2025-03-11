// Copyright 2021 The XLS Authors
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
#include <filesystem>  // NOLINT
#include <iostream>
#include <memory>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_replace.h"
#include "xls/common/status/matchers.h"
#include "xls/common/status/status_macros.h"
#include "xls/dslx/create_import_data.h"
#include "xls/dslx/error_printer.h"
#include "xls/dslx/error_test_utils.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/frontend/ast_node_visitor_with_default.h"
#include "xls/dslx/frontend/pos.h"
#include "xls/dslx/import_data.h"
#include "xls/dslx/parse_and_typecheck.h"
#include "xls/dslx/type_system/type_info.h"
#include "xls/dslx/type_system/typecheck_test_utils.h"
#include "xls/dslx/virtualizable_file_system.h"

namespace xls::dslx {
namespace {

using ::absl_testing::StatusIs;
using ::testing::AllOf;
using ::testing::HasSubstr;

// These matchers expect different errors in v1 and v2, allowing the errors to
// diverge, at least temporarily.

enum class TypeInferenceVersion : uint8_t { kVersion1 = 1, kVersion2 = 2 };

MATCHER_P2(HasSubstrInV1, param, substr, "") {
  return param == TypeInferenceVersion::kVersion2 ||
         ExplainMatchResult(HasSubstr(substr), arg, result_listener);
}

MATCHER_P2(HasSubstrInV2, param, substr, "") {
  return param == TypeInferenceVersion::kVersion1 ||
         ExplainMatchResult(HasSubstr(substr), arg, result_listener);
}

MATCHER_P3(HasSizeMismatchInV1, param, type1, type2, "") {
  return param == TypeInferenceVersion::kVersion2 ||
         ExplainMatchResult(HasSizeMismatch(type1, type2), arg,
                            result_listener);
}

MATCHER_P3(HasSizeMismatchInV2, param, type1, type2, "") {
  return param == TypeInferenceVersion::kVersion1 ||
         ExplainMatchResult(HasSizeMismatch(type1, type2), arg,
                            result_listener);
}

MATCHER_P3(HasTypeMismatchInV1, param, type1, type2, "") {
  return param == TypeInferenceVersion::kVersion2 ||
         ExplainMatchResult(HasTypeMismatch(type1, type2), arg,
                            result_listener);
}

MATCHER_P3(HasTypeMismatchInV2, param, type1, type2, "") {
  return param == TypeInferenceVersion::kVersion1 ||
         ExplainMatchResult(HasTypeMismatch(type1, type2), arg,
                            result_listener);
}

MATCHER_P3(HasSignednessMismatchInV1, param, type1, type2, "") {
  return param == TypeInferenceVersion::kVersion2 ||
         ExplainMatchResult(HasTypeMismatch(type1, type2), arg,
                            result_listener);
}

MATCHER_P3(HasSignednessMismatchInV2, param, type1, type2, "") {
  return param == TypeInferenceVersion::kVersion1 ||
         ExplainMatchResult(HasTypeMismatch(type1, type2), arg,
                            result_listener);
}

// Base class for general tests that run against both versions of type
// inference.
class TypecheckBothVersionsTest
    : public ::testing::TestWithParam<TypeInferenceVersion> {
 public:
  absl::StatusOr<TypecheckResult> Typecheck(std::string_view program) {
    return GetParam() == TypeInferenceVersion::kVersion1
               ? ::xls::dslx::Typecheck(program)
               : ::xls::dslx::TypecheckV2(program);
  }
};

TEST(TypecheckErrorTest, EnumItemSelfReference) {
  std::string_view text = "enum E:u2{ITEM=E::ITEM}";
  EXPECT_THAT(Typecheck(text),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Cannot find a definition for name: \"E\"")));
}

TEST(TypecheckErrorTest, TypeAliasSelfReference) {
  std::string_view text = "type T=uN[T::A as u2];";
  EXPECT_THAT(Typecheck(text),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Cannot find a definition for name: \"T\"")));
}

TEST(TypecheckErrorTest, SendInFunction) {
  std::string_view text = R"(
fn f(tok: token, output_r: chan<u8> in, expected: u8) -> token {
  let (tok, value) = recv(tok, output_r);
  tok
}
)";
  EXPECT_THAT(Typecheck(text),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Cannot recv() outside of a proc")));
}

TEST_P(TypecheckBothVersionsTest, ParametricWhoseTypeIsDeterminedByArgBinding) {
  std::string_view text = R"(
fn p<A: u32, B: u32, C: bits[B] = {bits[B]:0}>(x: bits[B]) -> bits[B] { x }
fn main() -> u2 { p<u32:1>(u2:0) }
)";
  XLS_EXPECT_OK(Typecheck(text));
}

// The type of the default expression is wrong for the parametric binding of X.
//
// It's /always/ wrong, but it's also not used, so this test is pointing out "do
// we care / flag that?"
//
// Right now:
// * we do not ignore that the default expression is improperly typed, but we
//   probably should, because we've been presented with an explicit parametric
//   argument that does work
// * we should flag as an error that the default expression is impossible to
//   reach because `X` is always bound via a parameter binding.
TEST_P(TypecheckBothVersionsTest,
       ParametricWithDefaultExpressionThatHasWrongType) {
  std::string_view text = R"(
fn p<X: u32 = {u1:0}>(x: bits[X]) -> bits[X] { x }
fn main() -> u2 { p<u32:1>(u1:0) }
)";
  EXPECT_THAT(
      Typecheck(text),
      StatusIs(absl::StatusCode::kInvalidArgument,
               AllOf(HasSizeMismatch("uN[32]", "uN[1]"),
                     HasSubstrInV1(GetParam(),
                                   "Annotated type of derived parametric "
                                   "value did not match inferred type"))));
}

TEST_P(TypecheckBothVersionsTest,
       ParametricThatWorksForTheOneBindingPresented) {
  std::string_view text = R"(
fn p<X: u32, Y: bits[X] = {u1:0}>(x: bits[X]) -> bits[X] { x }
fn main() -> u1 { p(u1:0) }
)";
  if (GetParam() == TypeInferenceVersion::kVersion1) {
    // The following is the explanation of why this case didn't work in v1:
    // Right now the bits[X] is a symbolic type and it is not allowed to accept
    // the `u1`. Once we drive all typechecking from instantiations this will
    // work with the concrete type presented (via the invocation in `main()`).
    EXPECT_THAT(
        Typecheck(text),
        StatusIs(
            absl::StatusCode::kInvalidArgument,
            HasSubstr("uN[X] vs uN[1]: Annotated type of derived parametric "
                      "value did not match inferred type")));
  } else {
    XLS_EXPECT_OK(Typecheck(text));
  }
}

TEST_P(TypecheckBothVersionsTest, ParametricWrongArgCount) {
  std::string_view text = R"(
fn id<N: u32>(x: bits[N]) -> bits[N] { x }
fn f() -> u32 { id(u8:3, u8:4) }
)";
  EXPECT_THAT(
      Typecheck(text),
      StatusIs(absl::StatusCode::kInvalidArgument,
               AllOf(HasSubstrInV1(GetParam(),
                                   "Expected 1 parameter(s) for function id, "
                                   "but got 2 argument(s)"),
                     HasSubstrInV2(GetParam(),
                                   "Expected 1 argument(s) but got 2"))));
}

TEST_P(TypecheckBothVersionsTest, ParametricTooManyExplicitSupplied) {
  std::string_view text = R"(
fn id<X: u32>(x: bits[X]) -> bits[X] { x }
fn main() -> u32 { id<u32:32, u32:64>(u32:5) }
)";
  EXPECT_THAT(
      Typecheck(text),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          HasSubstr("Too many parametric values supplied; limit: 1 given: 2")));
}

TEST_P(TypecheckBothVersionsTest, ReturnTypeMismatch) {
  EXPECT_THAT(
      Typecheck("fn f(x: bits[3], y: bits[4]) -> bits[5] { y }"),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          AllOf(HasSizeMismatchInV1(GetParam(), "uN[4]", "uN[5]"),
                HasSizeMismatchInV2(GetParam(), "bits[4]", "bits[5]"),
                HasSubstrInV1(GetParam(), "Return type of function body"))));
}

TEST_P(TypecheckBothVersionsTest, ReturnTypeMismatchWithImplicitUnitReturn) {
  EXPECT_THAT(
      Typecheck("fn f(x: bits[1]) -> bits[1] { x; }"),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          AllOf(HasTypeMismatchInV1(GetParam(), "()", "uN[1]"),
                HasTypeMismatchInV2(GetParam(), "()", "bits[1]"),
                HasSubstrInV1(
                    GetParam(),
                    "Return type of function body for 'f' did not match "
                    "the annotated return type"),
                HasSubstrInV1(GetParam(), "terminated with a semicolon"))));
}

TEST_P(TypecheckBothVersionsTest, ReturnTypeMismatchWithExplicitUnitReturn) {
  EXPECT_THAT(
      Typecheck("fn f(x: bits[1]) -> bits[1] { () }"),
      StatusIs(absl::StatusCode::kInvalidArgument,
               AllOf(HasTypeMismatchInV1(GetParam(), "()", "uN[1]"),
                     HasTypeMismatchInV2(GetParam(), "()", "bits[1]"),
                     HasSubstrInV1(
                         GetParam(),
                         "Return type of function body for 'f' did not match "
                         "the annotated return type"))));
}

TEST_P(TypecheckBothVersionsTest, Identity) {
  XLS_EXPECT_OK(Typecheck("fn f(x: u32) -> u32 { x }"));
  XLS_EXPECT_OK(Typecheck("fn f(x: bits[3], y: bits[4]) -> bits[3] { x }"));
  XLS_EXPECT_OK(Typecheck("fn f(x: bits[3], y: bits[4]) -> bits[4] { y }"));
}

TEST(TypecheckTest, TokenIdentity) {
  XLS_EXPECT_OK(Typecheck("fn f(x: token) -> token { x }"));
}

TEST_P(TypecheckBothVersionsTest, Unit) {
  XLS_EXPECT_OK(Typecheck("fn f(x: u32) -> () { () }"));
  XLS_EXPECT_OK(Typecheck("fn f(x: u32) { () }"));
}

TEST_P(TypecheckBothVersionsTest, Arithmetic) {
  // Simple add.
  XLS_EXPECT_OK(Typecheck("fn f(x: u32, y: u32) -> u32 { x + y }"));

  // Wrong annotated return type (implicitly unit).
  EXPECT_THAT(Typecheck("fn f(x: u32, y: u32) { x + y }"),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       AllOf(HasTypeMismatchInV1(GetParam(), "uN[32]", "()"),
                             HasTypeMismatchInV2(GetParam(), "u32", "()"))));

  // Wrong annotated return type (implicitly unit).
  EXPECT_THAT(
      Typecheck(R"(
      fn f<N: u32>(x: bits[N], y: bits[N]) { x + y }
      fn g() -> u64 { f(u64:5, u64:5) }
      )"),
      StatusIs(absl::StatusCode::kInvalidArgument,
               AllOf(HasTypeMismatchInV1(GetParam(), "()", "uN[64]"),
                     HasTypeMismatchInV2(GetParam(), "()", "u64"),
                     HasSubstrInV1(GetParam(),
                                   "function body for 'f' did not match "
                                   "the annotated return type"))));

  // Mixing widths not permitted.
  EXPECT_THAT(Typecheck("fn f(x: u32, y: bits[4]) { x + y }"),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       AllOf(HasSizeMismatchInV1(GetParam(), "uN[32]", "uN[4]"),
                             HasTypeMismatchInV2(GetParam(), "()", "u32"))));

  // Parametric same-width is ok!
  XLS_EXPECT_OK(
      Typecheck("fn f<N: u32>(x: bits[N], y: bits[N]) -> bits[N] { x + y }"));
}

TEST_P(TypecheckBothVersionsTest, Unary) {
  XLS_EXPECT_OK(Typecheck("fn f(x: u32) -> u32 { !x }"));
  XLS_EXPECT_OK(Typecheck("fn f(x: u32) -> u32 { -x }"));
}

TEST_P(TypecheckBothVersionsTest, Let) {
  XLS_EXPECT_OK(Typecheck("fn f() -> u32 { let x: u32 = u32:2; x }"));
  EXPECT_THAT(Typecheck(
                  R"(fn f() -> u32 {
        let x: u32 = u32:2;
        let y: bits[4] = bits[4]:3;
        y
      }
      )"),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       AllOf(HasSizeMismatchInV1(GetParam(), "uN[4]", "uN[32]"),
                             HasSizeMismatchInV2(GetParam(), "uN[4]", "u32"))));
  XLS_EXPECT_OK(Typecheck(
      "fn f() -> u32 { let (x, y): (u32, bits[4]) = (u32:2, bits[4]:3); x }"));
}

TEST_P(TypecheckBothVersionsTest, LetBadRhs) {
  EXPECT_THAT(
      Typecheck(
          R"(fn f() -> bits[2] {
          let (x, (y, (z,))): (u32, (bits[4], (bits[2],))) = (u32:2, bits[4]:3);
          z
        })"),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          AllOf(HasSubstrInV1(GetParam(),
                              "did not match inferred type of right hand side"),
                HasTypeMismatchInV2(GetParam(), "(bits[4], (bits[2],))",
                                    "bits[4]"))));
}

TEST_P(TypecheckBothVersionsTest, ParametricInvocation) {
  constexpr std::string_view kProgram = R"(
fn p<N: u32>(x: bits[N]) -> bits[N] { x + bits[N]:1 }
fn f() -> u32 { p(u32:3) }
)";
  XLS_EXPECT_OK(Typecheck(kProgram));
}

TEST_P(TypecheckBothVersionsTest, ParametricInvocationWithTuple) {
  constexpr std::string_view kProgram = R"(
fn p<N: u32>(x: bits[N]) -> (bits[N], bits[N]) { (x, x) }
fn f() -> (u32, u32) { p(u32:3) }
)";
  XLS_EXPECT_OK(Typecheck(kProgram));
}

TEST_P(TypecheckBothVersionsTest, DoubleParametricInvocation) {
  constexpr std::string_view kProgram = R"(
fn p<N: u32>(x: bits[N]) -> bits[N] { x + bits[N]:1 }
fn o<M: u32>(x: bits[M]) -> bits[M] { p(x) }
fn f() -> u32 { o(u32:3) }
)";
  XLS_EXPECT_OK(Typecheck(kProgram));
}

TEST_P(TypecheckBothVersionsTest, XbitsBinding) {
  constexpr std::string_view kProgram = R"(
fn p<S: bool, N: u32>(x: xN[S][N]) -> (bool, u32) { (S, N) }
fn f() -> (bool, u32)[2] { [p(u4:0), p(s8:0)] }
)";
  XLS_EXPECT_OK(Typecheck(kProgram));
}

TEST_P(TypecheckBothVersionsTest, XbitsCast) {
  constexpr std::string_view kProgram = R"(
fn f(x: u1) -> s1 { x as xN[true][1] }
fn g() -> s1 { u1:0 as xN[true][1] }
)";
  XLS_EXPECT_OK(Typecheck(kProgram));
}

// Previously this would flag an internal error.
TEST_P(TypecheckBothVersionsTest, CastToXbitsBasedBoolArray) {
  constexpr std::string_view kProgram = R"(
const ARRAY_SIZE = u32:44;
type MyXn = xN[bool:0x0][1];  // equivalent to a bool

fn main() -> bool[44] {
  let x: u44 = u44:0;
  // Equivalent to casting bits to corresponding bool array.
  x as MyXn[ARRAY_SIZE]
}
)";
  XLS_EXPECT_OK(Typecheck(kProgram));
}

// Various samples of actual-argument compatibility with an `xN` field within a
// struct via a struct instantiation expression.
TEST_P(TypecheckBothVersionsTest, StructInstantiateParametricXnField) {
  constexpr std::string_view kProgram = R"(
struct XnWrapper<S: bool, N: u32> {
  field: xN[S][N]
}
fn f() -> XnWrapper<false, u32:8> { XnWrapper<false, u32:8> { field: u8:0 } }
fn g() -> XnWrapper<true, u32:8> { XnWrapper<true, u32:8> { field: s8:1 } }
fn h() -> XnWrapper<false, u32:8> { XnWrapper<false, u32:8> { field: xN[false][8]:2 } }
fn i() -> XnWrapper<true, u32:8> { XnWrapper<true, u32:8> { field: xN[true][8]:3 } }
)";
  XLS_EXPECT_OK(Typecheck(kProgram));
}

TEST_P(TypecheckBothVersionsTest, ParametricPlusGlobal) {
  constexpr std::string_view kProgram = R"(
const GLOBAL = u32:4;
fn p<N: u32>() -> bits[N+GLOBAL] { bits[N+GLOBAL]:0 }
fn f() -> u32 { p<u32:28>() }
)";
  XLS_EXPECT_OK(Typecheck(kProgram));
}

TEST_P(TypecheckBothVersionsTest, TupleWithExplicitlyAnnotatedType) {
  constexpr std::string_view kProgram = R"(
const MY_TUPLE = (u32, u64):(u32:32, u64:64);
)";
  XLS_EXPECT_OK(Typecheck(kProgram));
}

TEST(TypecheckTest, ProcWithImplEmpty) {
  constexpr std::string_view kProgram = R"(
proc Foo {}

impl Foo {}
)";
  XLS_EXPECT_OK(Typecheck(kProgram));
}

TEST(TypecheckTest, ProcWithImplAndMembers) {
  constexpr std::string_view kProgram = R"(
proc Foo {
  foo: u32,
  bar: s8[7],
}

impl Foo {}
)";
  XLS_EXPECT_OK(Typecheck(kProgram));
}

TEST(TypecheckTest, ProcWithImplAndParametrics) {
  constexpr std::string_view kProgram = R"(
proc Proc<A: u32, B: u32 = {u32:32}, C:u32 = {B / u32:2}> {
  a: uN[A],
  b: uN[B],
  c: uN[C],
}

impl Proc {}
  )";
  XLS_EXPECT_OK(Typecheck(kProgram));
}

TEST(TypecheckTest, ProcWithImplEmptyInstantiation) {
  constexpr std::string_view kProgram = R"(
proc Foo {}

impl Foo {}

fn foo() -> Foo {
  Foo{}
}
  )";
  EXPECT_THAT(
      Typecheck(kProgram),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr(
                   "Instantiation of impl-style procs is not yet supported.")));
}

TEST(TypecheckTest, ProcWithImplInstantiation) {
  constexpr std::string_view kProgram = R"(
proc Foo<N: u32> {
  foo: u32,
  bar: bits[N],
}

impl Foo {}

fn foo() -> Foo<u32:8> {
  Foo<u32:8> { foo: u32:5, bar: u8:6 }
}
  )";
  EXPECT_THAT(
      Typecheck(kProgram),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr(
                   "Instantiation of impl-style procs is not yet supported.")));
}

TEST(TypecheckTest, FailsOnProcWithImplZero) {
  constexpr std::string_view kProgram = R"(
proc Foo {
  foo: u32,
  bar: bool,
}

impl Foo {}

fn foo() -> Foo {
  zero!<Foo>()
}
  )";
  EXPECT_THAT(Typecheck(kProgram),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Cannot make a zero-value of proc type.")));
}

TEST(TypecheckTest, FailsOnProcWithImplAsStructMember) {
  constexpr std::string_view kProgram = R"(
proc Foo {
  foo: u32,
  bar: bool,
}

impl Foo {}

struct Bar {
  the_proc: Foo
}

fn foo() -> Foo {
  zero!<Bar>()
}
  )";
  EXPECT_THAT(Typecheck(kProgram),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Structs cannot contain procs as members.")));
}

TEST(TypecheckTest, FailsOnProcWithImplAsStructMemberInArray) {
  constexpr std::string_view kProgram = R"(
proc Foo {
  foo: u32,
  bar: bool,
}

impl Foo {}

struct Bar {
  subprocs: Foo[2]
}

fn foo() -> Foo {
  zero!<Bar>()
}
  )";
  EXPECT_THAT(Typecheck(kProgram),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Structs cannot contain procs as members.")));
}

TEST(TypecheckTest, FailsOnProcWithImplAsStructMemberInTuple) {
  constexpr std::string_view kProgram = R"(
proc Foo {
  foo: u32,
  bar: bool,
}

impl Foo {}

struct Bar {
  subprocs: (Foo, Foo)
}

fn foo() -> Foo {
  zero!<Bar>()
}
  )";
  EXPECT_THAT(Typecheck(kProgram),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Structs cannot contain procs as members.")));
}

TEST(TypecheckTest, ProcWithImplAsProcMember) {
  constexpr std::string_view kProgram = R"(
proc Foo {
  foo: u32,
  bar: bool,
}

impl Foo {}

proc Bar {
  subproc: Foo
}
  )";
  XLS_EXPECT_OK(Typecheck(kProgram));
}

TEST(TypecheckTest, ProcWithImplAsProcMemberInArray) {
  constexpr std::string_view kProgram = R"(
proc Foo {
  foo: u32,
  bar: bool,
}

impl Foo {}

proc Bar {
  subprocs: Foo[2]
}
  )";

  XLS_EXPECT_OK(Typecheck(kProgram));
}

// Note: this previously caused a fatal error as the slice in this position is
// handled in the AstCloner.
TEST(TypecheckErrorTest, ProcWithSliceOfNumber) {
  constexpr std::string_view kProgram = R"(
proc o {
  h: chan<u2[0[:]]> in;
  config(h: chan<u3> out) {}
}
)";
  EXPECT_THAT(Typecheck(kProgram),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Procs must define `init`")));
}

TEST(TypecheckTest, ProcWithImplAsProcMemberInTuple) {
  constexpr std::string_view kProgram = R"(
proc Foo {
  foo: u32,
  bar: bool,
}

impl Foo {}

proc Bar {
  subprocs: (Foo, Foo)
}
  )";

  XLS_EXPECT_OK(Typecheck(kProgram));
}

TEST(TypecheckTest, ProcWithImplAsImportedProcMember) {
  constexpr std::string_view kImported = R"(
pub proc Foo {
  foo: u32,
  bar: bool,
}

impl Foo {}
)";
  constexpr std::string_view kProgram = R"(
import imported;

proc Bar {
  subproc: imported::Foo
})";
  auto import_data = CreateImportDataForTest();

  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule module,
      ParseAndTypecheck(kImported, "imported.x", "imported", &import_data));
  XLS_EXPECT_OK(
      ParseAndTypecheck(kProgram, "fake_main_path.x", "main", &import_data));
}

TEST(TypecheckTest, UseOfConstant) {
  constexpr std::string_view kImported = R"(
pub const MY_CONSTANT: u32 = u32:42;
)";
  constexpr std::string_view kProgram = R"(#![feature(use_syntax)]
use imported::MY_CONSTANT;

fn f() -> u32 {
  MY_CONSTANT
}
)";
  absl::flat_hash_map<std::filesystem::path, std::string> files = {
      {std::filesystem::path("/imported.x"), std::string(kImported)},
      {std::filesystem::path("/fake_main_path.x"), std::string(kProgram)},
  };
  auto vfs = std::make_unique<FakeFilesystem>(
      files, /*cwd=*/std::filesystem::path("/"));
  ImportData import_data = CreateImportDataForTest(std::move(vfs));
  absl::StatusOr<TypecheckedModule> main_module =
      ParseAndTypecheck(kProgram, "fake_main_path.x", "main", &import_data);
  XLS_EXPECT_OK(main_module.status()) << main_module.status();
}

TEST(TypecheckTest, UseOfStdlibModule) {
  constexpr std::string_view kProgram = R"(#![feature(use_syntax)]
use std;

fn main() -> u32 {
  std::popcount(u32:42)
}
)";
  XLS_EXPECT_OK(Typecheck(kProgram));
}

TEST(TypecheckTest, FailsOnProcWithImplAsImportedStructMember) {
  constexpr std::string_view kImported = R"(
pub proc Foo {
  foo: u32,
  bar: bool,
}

impl Foo {}
)";
  constexpr std::string_view kProgram = R"(
import imported;

struct Bar {
  subproc: imported::Foo
})";
  auto import_data = CreateImportDataForTest();

  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule module,
      ParseAndTypecheck(kImported, "imported.x", "imported", &import_data));
  EXPECT_THAT(
      ParseAndTypecheck(kProgram, "fake_main_path.x", "main", &import_data),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("Structs cannot contain procs as members.")));
}

TEST_P(TypecheckBothVersionsTest, ParametricStructInstantiatedByGlobal) {
  constexpr std::string_view kProgram = R"(
struct MyStruct<WIDTH: u32> {
  f: bits[WIDTH]
}
fn p<FIELD_WIDTH: u32>(s: MyStruct<FIELD_WIDTH>) -> u15 {
  s.f
}
const GLOBAL = u32:15;
fn f(s: MyStruct<GLOBAL>) -> u15 { p(s) }
)";
  XLS_EXPECT_OK(Typecheck(kProgram));
}

TEST_P(TypecheckBothVersionsTest, TopLevelConstTypeMismatch) {
  constexpr std::string_view kProgram = R"(
const GLOBAL: u64 = u32:4;
)";
  EXPECT_THAT(
      Typecheck(kProgram),
      StatusIs(absl::StatusCode::kInvalidArgument,
               AllOf(HasSizeMismatchInV1(GetParam(), "uN[64]", "uN[32]"),
                     HasSizeMismatchInV2(GetParam(), "u64", "u32"),
                     HasSubstrInV1(GetParam(),
                                   "Constant definition's annotated type did "
                                   "not match its expression's type"))));
}

TEST_P(TypecheckBothVersionsTest, TopLevelConstTypeMatch) {
  constexpr std::string_view kProgram = R"(
const GLOBAL: u32 = u32:4;
)";
  XLS_EXPECT_OK(Typecheck(kProgram));
}

TEST(TypecheckErrorTest, LetTypeAnnotationIsXn) {
  constexpr std::string_view kProgram = "fn f() { let x: xN = u32:42; }";
  EXPECT_THAT(Typecheck(kProgram),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Could not determine signedness to turn `xN` "
                                 "into a concrete bits type.")));
}

TEST_P(TypecheckBothVersionsTest, ParametricIdentifierLtValue) {
  constexpr std::string_view kProgram = R"(
fn p<N: u32>(x: bits[N]) -> bits[N] { x }

fn f() -> bool { p < u32:42 }
)";
  EXPECT_THAT(
      Typecheck(kProgram),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          AllOf(HasSubstrInV1(GetParam(),
                              "Name 'p' is a parametric function, but it is "
                              "not being invoked"),
                HasSubstrInV2(GetParam(),
                              "Expected type `uN[32]` but got `p`, which is a "
                              "parametric function not being invoked."))));
}

// X is not bound in this example but it's also not used anywhere. This is
// erroneously accepted in v1 but fixed in v2.
TEST_P(TypecheckBothVersionsTest,
       Gh1473_UnboundButAlsoUnusedParametricNoDefaultExpr) {
  constexpr std::string_view kProgram = R"(
fn p<X: u32, Y: u32>(y: uN[Y]) -> u32 { Y }
fn f() -> u32 { p(u7:0) }
)";
  if (GetParam() == TypeInferenceVersion::kVersion1) {
    XLS_EXPECT_OK(Typecheck(kProgram));
  } else {
    EXPECT_THAT(Typecheck(kProgram),
                StatusIs(absl::StatusCode::kInvalidArgument,
                         HasSubstr("Could not infer parametric(s): X")));
  }
}

TEST_P(TypecheckBothVersionsTest,
       Gh1473_UnboundButAlsoUnusedParametricWithDefaultExpr) {
  constexpr std::string_view kProgram = R"(
fn p<X: u32, Y: u32 = {X+X}>(y: uN[Y]) -> u32 { Y }
fn f() -> u32 { p(u7:0) }
)";
  EXPECT_THAT(
      Typecheck(kProgram),
      StatusIs(absl::StatusCode::kInvalidArgument,
               AllOf(HasSubstrInV1(
                         GetParam(),
                         "Parametric expression `X + X` referred to `X` which "
                         "is not present in the parametric environment"),
                     HasSubstrInV2(GetParam(),
                                   "Could not infer parametric(s): X"))));
}

// In this example we do not bind X via the arguments, but we try to use it in
// forming a return type.
TEST_P(TypecheckBothVersionsTest, Gh1473_UnboundAndUsedParametric) {
  constexpr std::string_view kProgram = R"(
fn p<X: u32, Y: u32>(y: uN[Y]) -> uN[X] { Y }
fn f() -> u32 { p(u7:0) }
)";
  EXPECT_THAT(
      Typecheck(kProgram),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          AllOf(
              HasSubstrInV1(GetParam(),
                            "uN[X] Instantiated return type did not have "
                            "the following parametrics resolved: X"),
              HasSubstrInV2(GetParam(), "Could not infer parametric(s): X"))));
}

// In this example we do not bind X via the arguments, but we try to use it in
// the body of the function. This is erroneously accepted in v1 but fixed in v1.
TEST_P(TypecheckBothVersionsTest, Gh1473_UnboundAndUsedParametricInBody) {
  constexpr std::string_view kProgram = R"(
fn p<X: u32, Y: u32>(y: uN[Y]) -> bool { uN[X]:0 == uN[X]:1 }
fn f() -> bool { p(u7:0) }
)";
  if (GetParam() == TypeInferenceVersion::kVersion1) {
    XLS_EXPECT_OK(Typecheck(kProgram));
  } else {
    EXPECT_THAT(Typecheck(kProgram),
                StatusIs(absl::StatusCode::kInvalidArgument,
                         HasSubstr("Could not infer parametric(s): X")));
  }
}

TEST(TypecheckTest, MapOfParametric) {
  constexpr std::string_view kProgram = R"(
fn p<N: u32>(x: bits[N]) -> bits[N] { x }

fn f() -> u32[3] {
  map(u32[3]:[1, 2, 3], p)
}
)";
  XLS_EXPECT_OK(Typecheck(kProgram));
}

TEST(TypecheckTest, MapOfParametricExplicit) {
  constexpr std::string_view kProgram =
      R"(
fn f<N:u32, K:u32>(x: u32) -> uN[N] { x as uN[N] + K as uN[N] }
fn main() -> u5[4] { map(u32[4]:[0, 1, 2, 3], f<u32:5, u32:17>) }
)";

  XLS_EXPECT_OK(Typecheck(kProgram));
}

TEST(TypecheckTest, MapOfParametricExplicitWithWrongNumberOfArgs) {
  constexpr std::string_view kProgram =
      R"(
fn f<N:u32, K:u32>(x: u32) -> uN[N] { x as uN[N] + K as uN[N] }
fn main() -> u5[4] { map(u32[4]:[0, 1, 2, 3], f<u32:5, u32:17, u32:18>) }
)";

  EXPECT_THAT(
      Typecheck(kProgram),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          HasSubstr("Too many parametric values supplied; limit: 2 given: 3")));
}

TEST(TypecheckTest, MapImportedNonPublicFunction) {
  constexpr std::string_view kImported = R"(
fn some_function(x: u32) -> u32 { x }
)";
  constexpr std::string_view kProgram = R"(
import imported;

fn main() -> u32[3] {
  map(u32[3]:[1, 2, 3], imported::some_function)
})";
  auto import_data = CreateImportDataForTest();
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule module,
      ParseAndTypecheck(kImported, "imported.x", "imported", &import_data));
  EXPECT_THAT(
      ParseAndTypecheck(kProgram, "fake_main_path.x", "main", &import_data),
      StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("not public")));
}

TEST(TypecheckTest, MapImportedNonPublicInferredParametricFunction) {
  constexpr std::string_view kImported = R"(
fn some_function<N: u32>(x: bits[N]) -> bits[N] { x }
)";
  constexpr std::string_view kProgram = R"(
import imported;

fn main() -> u32[3] {
  map(u32[3]:[1, 2, 3], imported::some_function)
})";
  auto import_data = CreateImportDataForTest();
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule module,
      ParseAndTypecheck(kImported, "imported.x", "imported", &import_data));
  EXPECT_THAT(
      ParseAndTypecheck(kProgram, "fake_main_path.x", "main", &import_data),
      StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("not public")));
}

TEST_P(TypecheckBothVersionsTest, ParametricInvocationConflictingArgs) {
  constexpr std::string_view kProgram = R"(
fn id<N: u32>(x: bits[N], y: bits[N]) -> bits[N] { x }
fn f() -> u32 { id(u8:3, u32:5) }
)";
  EXPECT_THAT(
      Typecheck(kProgram),
      StatusIs(absl::StatusCode::kInvalidArgument,
               AllOf(HasSubstrInV1(GetParam(), "saw: 8; then: 32"),
                     HasSizeMismatchInV2(GetParam(), "bits[8]", "u32"))));
}

TEST_P(TypecheckBothVersionsTest, ParametricWrongKind) {
  constexpr std::string_view kProgram = R"(
fn id<N: u32>(x: bits[N]) -> bits[N] { x }
fn f() -> u32 { id((u8:3,)) }
)";
  EXPECT_THAT(
      Typecheck(kProgram),
      StatusIs(absl::StatusCode::kInvalidArgument,
               AllOf(HasSubstrInV1(GetParam(), "different kinds"),
                     HasTypeMismatchInV2(GetParam(), "(u8,)", "bits[N]"))));
}

TEST_P(TypecheckBothVersionsTest, ParametricWrongNumberOfDims) {
  constexpr std::string_view kProgram = R"(
fn id<N: u32, M: u32>(x: bits[N][M]) -> bits[N][M] { x }
fn f() -> u32 { id(u32:42) }
)";
  EXPECT_THAT(
      Typecheck(kProgram),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          AllOf(HasSubstrInV1(GetParam(),
                              "types are different kinds (array vs ubits)"),
                HasTypeMismatchInV2(GetParam(), "u32", "bits[N][M]"))));
}

TEST(TypecheckErrorTest, RecursionCausesError) {
  constexpr std::string_view kProgram = "fn f(x: u32) -> u32 { f(x) }";
  EXPECT_THAT(Typecheck(kProgram),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Recursion of function `f` detected")));
}

TEST(TypecheckErrorTest, ParametricRecursionCausesError) {
  constexpr std::string_view kProgram = R"(
fn f<X: u32>(x: bits[X]) -> u32 { f(x) }
fn g() -> u32 { f(u32: 5) }
)";
  EXPECT_THAT(Typecheck(kProgram),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Recursion of function `f` detected")));
}

TEST(TypecheckErrorTest, HigherOrderRecursionCausesError) {
  constexpr std::string_view kProgram = R"(
fn h<Y: u32>(y: bits[Y]) -> bits[Y] { h(y) }
fn g() -> u32[3] {
    let x0 = u32[3]:[0, 1, 2];
    map(x0, h)
}
)";
  EXPECT_THAT(Typecheck(kProgram),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Recursion of function `h` detected")));
}

TEST_P(TypecheckBothVersionsTest, InvokeWrongArg) {
  constexpr std::string_view kProgram = R"(
fn id_u32(x: u32) -> u32 { x }
fn f(x: u8) -> u8 { id_u32(x) }
)";
  EXPECT_THAT(
      Typecheck(kProgram),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          AllOf(HasSubstrInV1(GetParam(),
                              "Mismatch between parameter and argument types"),
                HasSizeMismatchInV2(GetParam(), "u32", "u8"))));
}

TEST_P(TypecheckBothVersionsTest, InvokeNumberValue) {
  constexpr std::string_view kProgram = R"(
fn f(x: u8) -> u8 { 42(x) }
)";
  EXPECT_THAT(
      Typecheck(kProgram),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("invocation callee must be a function, with a "
                         "possible scope indicated using `::` or `.`")));
}

// Since the parametric is not instantiated we don't detect this type error.
TEST_P(TypecheckBothVersionsTest, InvokeNumberValueInUninstantiatedParametric) {
  constexpr std::string_view kProgram = R"(
fn f<N: u32>(x: u8) -> u8 { 42(x) }
)";
  XLS_EXPECT_OK(Typecheck(kProgram));
}

TEST(TypecheckErrorTest, BadTupleType) {
  constexpr std::string_view kProgram = R"(
fn f() -> u32 {
  let (a, b, c): (u32, u32) = (u32:1, u32:2, u32:3);
  a
}
)";
  EXPECT_THAT(
      Typecheck(kProgram),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("Annotated type did not match inferred type")));
}

// -- logical ops

class LogicalOpTypecheckTest
    : public testing::Test,
      public testing::WithParamInterface<
          std::tuple<TypeInferenceVersion, std::string_view>> {
 public:
  absl::Status TypecheckOp(std::string_view tmpl) {
    const std::string program = absl::StrReplaceAll(
        tmpl, {{"$OP", std::get<std::string_view>(GetParam())}});
    return std::get<TypeInferenceVersion>(GetParam()) ==
                   TypeInferenceVersion::kVersion1
               ? ::xls::dslx::Typecheck(program).status()
               : ::xls::dslx::TypecheckV2(program).status();
  }
};

TEST_P(LogicalOpTypecheckTest, LogicalAndOnVariousOkTypes) {
  XLS_EXPECT_OK(TypecheckOp("fn f(a: u1, b: u1) -> u1 { a $OP b }"));
  XLS_EXPECT_OK(TypecheckOp("fn f(a: bool, b: bool) -> bool { a $OP b }"));
  XLS_EXPECT_OK(TypecheckOp("fn f(a: uN[1], b: uN[1]) -> uN[1] { a $OP b }"));
  XLS_EXPECT_OK(
      TypecheckOp("fn f(a: bits[1], b: bits[1]) -> bits[1] { a $OP b }"));
  XLS_EXPECT_OK(TypecheckOp(
      "fn f(a: xN[false][1], b: xN[false][1]) -> xN[false][1] { a $OP b }"));
}

TEST_P(LogicalOpTypecheckTest, LogicalAndOnVariousInvalidTypes) {
  TypeInferenceVersion version = std::get<TypeInferenceVersion>(GetParam());
  EXPECT_THAT(TypecheckOp("fn f(a: u2, b: u2) -> bool { a $OP b }"),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       AllOf(HasSizeMismatchInV1(version, "uN[2]", "uN[1]"),
                             HasSizeMismatchInV2(version, "u2", "bool"))));
  EXPECT_THAT(TypecheckOp("fn f(a: u32, b: u32) -> bool { a $OP b }"),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       AllOf(HasSizeMismatchInV1(version, "uN[32]", "uN[1]"),
                             HasSizeMismatchInV2(version, "u32", "bool"))));
  EXPECT_THAT(
      TypecheckOp("fn f(a: s1, b: s1) -> bool { a $OP b }"),
      StatusIs(absl::StatusCode::kInvalidArgument,
               AllOf(HasSignednessMismatchInV1(version, "sN[1]", "uN[1]"),
                     HasSignednessMismatchInV2(version, "s1", "bool"))));
  EXPECT_THAT(TypecheckOp("fn f(a: s32, b: s32) -> bool { a $OP b }"),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       AllOf(HasTypeMismatchInV1(version, "sN[32]", "uN[1]"),
                             HasTypeMismatchInV2(version, "s32", "bool"))));
}

INSTANTIATE_TEST_SUITE_P(
    LogicalOpTypecheckTestInstance, LogicalOpTypecheckTest,
    ::testing::Combine(::testing::Values(TypeInferenceVersion::kVersion1,
                                         TypeInferenceVersion::kVersion2),
                       ::testing::Values("&&", "||")));

// --

TEST_P(TypecheckBothVersionsTest, LogicalAndOfComparisons) {
  XLS_EXPECT_OK(Typecheck("fn f(a: u8, b: u8) -> bool { a == b }"));
  XLS_EXPECT_OK(Typecheck(
      "fn f(a: u8, b: u8, c: u32, d: u32) -> bool { a == b && c == d }"));
}

TEST_P(TypecheckBothVersionsTest, Typedef) {
  XLS_EXPECT_OK(Typecheck(R"(
type MyTypeAlias = (u32, u8);
fn id(x: MyTypeAlias) -> MyTypeAlias { x }
fn f() -> MyTypeAlias { id((u32:42, u8:127)) }
)"));
}

TEST(TypecheckTest, For) {
  XLS_EXPECT_OK(Typecheck(R"(
fn f() -> u32 {
  for (i, accum): (u32, u32) in range(u32:0, u32:3) {
    let new_accum: u32 = accum + i;
    new_accum
  }(u32:0)
})"));
}

TEST(TypecheckTest, ForInParametricInvokedTwice) {
  XLS_EXPECT_OK(Typecheck(R"(
fn p<N: u32>(x: uN[N]) -> uN[N] {
    for (idx, accum) in range(u32:0, N) {
        accum
    }(uN[N]:0)
}

#[test]
fn two_invocation_test() {
    assert_eq(u4:0, p(u4:8));
    assert_eq(u4:0, p(u4:8));
}
)"));
}

TEST(TypecheckTest, ForNoAnnotation) {
  XLS_EXPECT_OK(Typecheck(R"(
fn f() -> u32 {
  for (i, accum) in range(u32:0, u32:3) {
    accum
  }(u32:0)
})"));
}

TEST(TypecheckTest, ForWildcardIvar) {
  XLS_EXPECT_OK(Typecheck(R"(
fn f() -> u32 {
  for (_, accum) in range(u32:0, u32:3) {
    accum
  }(u32:0)
})"));
}

TEST(TypecheckErrorTest, RangeBuiltinWithNonConstexprArg) {
  EXPECT_THAT(
      Typecheck(R"(fn f(limit: u8) -> u32 {
  for (_, accum) in range(u32:0, limit as u32) {
    accum
  }(u32:0)
}
)"),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr(
                   "Argument to built-in function `range` must be constexpr")));
}

TEST(TypecheckTest, RangeBuiltinWithConstexprArgViaModuleCall) {
  XLS_EXPECT_OK(Typecheck(R"(import std;
const MOL = u32:42;
fn f() -> u32 {
  for (_, accum) in range(u32:0, std::clog2(MOL)) {
    accum
  }(u32:0)
}
)"));
}

TEST(TypecheckTest, UseOfClog2InModuleScopedConstantDefinition) {
  XLS_EXPECT_OK(Typecheck(R"(#![feature(use_syntax)]
use std::clog2;

const MAX_BITS: u32 = clog2(u32:256);

fn main() -> u32 {
    MAX_BITS
}
)"));
}

TEST(TypecheckTest, UseOfClog2InParametricOutputType) {
  XLS_EXPECT_OK(Typecheck(R"(#![feature(use_syntax)]
use std::{clog2, is_pow2};

fn p<N: u32, OUT: u32 = {clog2(N)}>(x: uN[N]) -> uN[OUT] {
  const_assert!(is_pow2(N));
  uN[OUT]:0
}

fn main() -> u5 { p(u32:42) }
)"));
}

TEST_P(TypecheckBothVersionsTest, ConstAssertParametricOk) {
  XLS_EXPECT_OK(Typecheck(R"(
fn p<N: u32>() -> u32 {
  const_assert!(N == u32:42);
  N
}
fn main() -> u32 {
  p<u32:42>()
})"));
}

TEST_P(TypecheckBothVersionsTest, ConstAssertViaConstBindings) {
  XLS_EXPECT_OK(Typecheck(R"(
fn main() -> () {
  const M = u32:1;
  const N = u32:2;
  const O = M + N;
  const_assert!(O == u32:3);
  ()
})"));
}

TEST_P(TypecheckBothVersionsTest, ConstAssertCallFunction) {
  XLS_EXPECT_OK(Typecheck(R"(
fn is_mol(x: u32) -> bool {
  x == u32:42
}
fn p<N: u32>() -> () {
  const_assert!(is_mol(N));
  ()
}
fn main() -> () {
  p<u32:42>()
})"));
}

TEST_P(TypecheckBothVersionsTest, ConstAssertFalse) {
  EXPECT_THAT(Typecheck(R"(
fn main() -> () {
  const_assert!(false);
})"),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("const_assert! failure: `false`")));
}

TEST_P(TypecheckBothVersionsTest, ConstAssertFalseExpr) {
  EXPECT_THAT(Typecheck(R"(
fn main() -> () {
  const_assert!(u32:2 + u32:3 != u32:5);
})"),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("const_assert! failure")));
}

TEST_P(TypecheckBothVersionsTest, ConstAssertNonConstexpr) {
  EXPECT_THAT(Typecheck(R"(
fn main(p: u32) -> () {
  const_assert!(p == u32:42);
})"),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("const_assert! expression is not constexpr")));
}

TEST_P(TypecheckBothVersionsTest, FitsInTypeSN0) {
  EXPECT_THAT(Typecheck(R"(
fn main() -> sN[0] {
  sN[0]:0xffff_ffff_ffff_ffff_ffff
})"),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Value '0xffff_ffff_ffff_ffff_ffff' does not "
                                 "fit in the bitwidth of a sN[0]")));
}

TEST_P(TypecheckBothVersionsTest, ParametricBindArrayToTuple) {
  EXPECT_THAT(
      Typecheck(R"(
fn p<N: u32>(x: (uN[N], uN[N])) -> uN[N] { x.0 }

fn main() -> u32 {
  p(u32[2]:[0, 1])
})"),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          AllOf(
              HasSubstrInV1(GetParam(),
                            "Parameter 0 and argument types are different "
                            "kinds (tuple vs array)"),
              HasTypeMismatchInV2(GetParam(), "uN[32][2]", "(uN[N], uN[N])"))));
}

TEST_P(TypecheckBothVersionsTest, ParametricBindNested) {
  EXPECT_THAT(
      Typecheck(R"(
fn p<N: u32>(x: (u32, u64)[N]) -> u32 { x[0].0 }

fn main() -> u32 {
  p(u32[1][1]:[[u32:0]])
})"),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          AllOf(
              HasTypeMismatchInV1(GetParam(), "(uN[32], uN[64])", "uN[32][1]"),
              HasTypeMismatchInV2(GetParam(), "(u32, u64)", "uN[32][1]"),
              HasSubstrInV1(GetParam(),
                            "expected argument kind 'array' to match parameter "
                            "kind 'tuple'"))));
}

TEST(TypecheckTest, ForBuiltinInBody) {
  XLS_EXPECT_OK(Typecheck(R"(
fn f() -> u32 {
  for (i, accum): (u32, u32) in range(u32:0, u32:3) {
    trace!(accum)
  }(u32:0)
})"));
}

TEST(TypecheckTest, ForNestedBindings) {
  XLS_EXPECT_OK(Typecheck(R"(
fn f(x: u32) -> (u32, u8) {
  for (i, (x, y)): (u32, (u32, u8)) in range(u32:0, u32:3) {
    (x, y)
  }((x, u8:42))
}
)"));
}

TEST(TypecheckTest, ForWithBadTypeTree) {
  EXPECT_THAT(Typecheck(R"(
fn f(x: u32) -> (u32, u8) {
  for (i, (x, y)): (u32, u8) in range(u32:0, u32:3) {
    (x, y)
  }((x, u8:42))
})"),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       AllOf(HasSubstr("uN[8]\nvs (uN[32], uN[8])"),
                             HasSubstr("For-loop annotated accumulator type "
                                       "did not match inferred type."))));
}

TEST(TypecheckTest, ForWithBadTypeAnnotation) {
  EXPECT_THAT(
      Typecheck(R"(
fn f(x: u32) -> (u32, u8) {
  for (i, _): u32 in range(u32:0, u32:3) {
    i
  }(u32:0)
})"),
      StatusIs(absl::StatusCode::kInvalidArgument,
               AllOf(HasSubstr("For-loop annotated type should be a tuple "
                               "containing a type for the iterable and a type "
                               "for the accumulator."))));
}

TEST(TypecheckTest, ForWithWrongResultType) {
  EXPECT_THAT(Typecheck(R"(
fn f(x: u32) -> (u32, u8) {
  for (i, _): (u32, u32) in range(u32:0, u32:3) {
    i as u64
  }(u32:0)
})"),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       AllOf(HasSubstr("uN[32] vs uN[64]"),
                             HasSubstr("For-loop init value type did not match "
                                       "for-loop body's result type."))));
}

TEST(TypecheckTest, ForWithWrongNumberOfArguments) {
  EXPECT_THAT(
      Typecheck(R"(
fn f(x: u32) -> u32 {
  for (i, j, acc): (u32, u32, u32) in range(u32:0, u32:3) {
    i + j + acc
  }(u32:42)
})"),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          HasSubstr("For-loop annotated type should specify a type for the "
                    "iterable and a type for the accumulator; got 3 types.")));
}

TEST(IrConverterTest, ForWithIndexTypeTooSmallForRange) {
  EXPECT_THAT(Typecheck(R"(
fn test() -> u4 {
  for(i, acc): (u4, u32) in u32:0..u32:120 {
    i as u32 + acc
  }(u32:0)
})"),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       AllOf(HasSubstr("uN[4]\nvs uN[32]"),
                             HasSubstr("For-loop annotated index type did not "
                                       "match inferred type."))));
}

TEST(TypecheckTest, UnrollForSimple) {
  XLS_EXPECT_OK(Typecheck(R"(
fn test() -> u32 {
  unroll_for!(i, acc): (u32, u32) in u32:0..u32:4 {
    i + acc
  }(u32:0)
})"));
}

TEST(TypecheckTest, UnrollForNestedBindings) {
  XLS_EXPECT_OK(Typecheck(R"(
fn f(x: u32) -> (u32, u8) {
  unroll_for! (_, (x, y)): (u32, (u32, u8)) in range(u32:0, u32:3) {
    (x, y)
  }((x, u8:42))
}
)"));
}

TEST(TypecheckTest, UnrollForWithBadTypeTree) {
  EXPECT_THAT(Typecheck(R"(
fn f(x: u32) -> (u32, u8) {
  unroll_for! (i, (x, y)): (u32, u8) in range(u32:0, u32:3) {
    (x, y)
  }((x, u8:42))
})"),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       AllOf(HasSubstr("uN[8]\nvs (uN[32], uN[8])"),
                             HasSubstr("For-loop annotated accumulator type "
                                       "did not match inferred type."))));
}

TEST(TypecheckTest, UnrollForWithBadTypeAnnotation) {
  EXPECT_THAT(
      Typecheck(R"(
fn f(x: u32) -> (u32, u8) {
  unroll_for! (i, _): u32 in range(u32:0, u32:3) {
    i
  }(u32:0)
})"),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("For-loop annotated type should be a tuple "
                         "containing a type for the iterable and a type "
                         "for the accumulator.")));
}

TEST(TypecheckTest, UnrollForWithoutIndexAccTypeAnnotation) {
  XLS_EXPECT_OK(Typecheck(R"(
proc SomeProc {
  init { () }
  config() { }
  next(state: ()) {
    unroll_for! (i, a) in u32:0..u32:4 {
      a
    }(u32:0);
  }
})"));
}

TEST(TypecheckTest, UnrollForWithWrongResultType) {
  EXPECT_THAT(Typecheck(R"(
fn f(x: u32) -> (u32, u8) {
  unroll_for! (i, _): (u32, u32) in range(u32:0, u32:3) {
    i as u64
  }(u32:0)
})"),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       AllOf(HasSubstr("uN[32] vs uN[64]"),
                             HasSubstr("For-loop init value type did not match "
                                       "for-loop body's result type."))));
}

TEST(TypecheckTest, UnrollForWithWrongNumberOfArguments) {
  EXPECT_THAT(
      Typecheck(R"(
fn test() -> u32 {
  unroll_for!(i, j, acc): (u32, u32, u32) in u32:0..u32:4 {
    i + j + acc
  }(u32:0)
})"),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          AllOf(HasSubstr("(uN[32], uN[32], uN[32])\nvs (uN[32], uN[32])"),
                HasSubstr(
                    "For-loop annotated type should specify a type for the "
                    "iterable and a type for the accumulator; got 3 types."))));
}

TEST(TypecheckTest, UnrollForWithIndexTypeTooSmallForRange) {
  EXPECT_THAT(Typecheck(R"(
fn test() -> u4 {
  unroll_for!(i, acc): (u4, u4) in u32:0..u32:120 {
    i + acc
  }(u4:0)
})"),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       AllOf(HasSubstr("uN[4]\nvs uN[32]"),
                             HasSubstr("For-loop annotated index type did not "
                                       "match inferred type."))));
}

// https://github.com/google/xls/issues/1717
TEST(TypecheckTest, UnrollForWithInvocationInTypeAlias) {
  XLS_EXPECT_OK(Typecheck(R"(
import std;
type A = bits[std::clog2(u32:256)];

fn muladd(a: u8, b: u8, c: u8) -> u8 {
  let d = unroll_for! (_, j): (u32, u8) in u32:0..u32:8 {
    A:0 + j
  }(A:0);
    a * b + c + d
})"));
}

TEST_P(TypecheckBothVersionsTest, DerivedParametricStruct) {
  XLS_EXPECT_OK(Typecheck(R"(
struct StructFoo<A: u32, B: u32 = {u32:32}, C:u32 = {B / u32:2}> {
  a: uN[A],
  b: uN[B],
  c: uN[C],
}

fn Foo() {
  let a = zero!<StructFoo<u32:32>>();
  let b = StructFoo<u32:32>{a: u32:0, b: u32:1, c: u16:4};
}
)"));
}

TEST_P(TypecheckBothVersionsTest, ZeroMacroSpecified) {
  XLS_EXPECT_OK(Typecheck("const Y = zero!<u10>();"));
}

TEST_P(TypecheckBothVersionsTest, ZeroMacroFromParametric) {
  XLS_EXPECT_OK(Typecheck(R"(
fn f<N:u32>() -> uN[N] { zero!<uN[N]>() }
const Y = f<u32:10>();
)"));
}

TEST_P(TypecheckBothVersionsTest, ZeroMacroFromParametricError) {
  constexpr std::string_view kProgram = R"(
fn f<N:u32>() -> uN[N] { zero!<N>() }
const Y = f<u32:10>();
)";
  EXPECT_THAT(
      Typecheck(kProgram),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          AllOf(
              HasSubstrInV1(GetParam(), "Expected a type in zero! macro type"),
              HasSubstrInV2(
                  GetParam(),
                  "Expected a type argument in `zero!<N>()`; saw `N`"))));
}

TEST_P(TypecheckBothVersionsTest, ZeroMacroFromStructConstError) {
  constexpr std::string_view kProgram = R"(
struct S{}
impl S { const X = u32:10; }
const Y = zero!<S::X>();
)";
  EXPECT_THAT(
      Typecheck(kProgram),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          AllOf(
              HasSubstrInV1(GetParam(), "Expected a type in zero! macro type"),
              HasSubstrInV2(
                  GetParam(),
                  "Expected a type argument in `zero!<S::X>()`; saw `S::X`"))));
}

TEST(TypecheckTest, ZeroMacroImportedType) {
  constexpr std::string_view kImported = R"(
pub type foo_t = u32;
)";
  constexpr std::string_view kProgram = R"(
import imported;
const Y = zero!<imported::foo_t>();
)";
  auto import_data = CreateImportDataForTest();
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule module,
      ParseAndTypecheck(kImported, "imported.x", "imported", &import_data));
  XLS_ASSERT_OK(
      ParseAndTypecheck(kProgram, "fake_main_path.x.x", "main", &import_data));
}

TEST_P(TypecheckBothVersionsTest, DerivedParametricStructUsingNonDefault) {
  XLS_EXPECT_OK(Typecheck(R"(
struct StructFoo<A: u32, B:u32 = {A * u32:2}> {
  x: uN[B],
}

fn Foo() {
  let b = StructFoo<u32:8>{x: u16:0};
}
)"));
}

TEST_P(TypecheckBothVersionsTest, DerivedParametricStructValueMissing) {
  EXPECT_THAT(
      Typecheck(R"(
struct StructFoo<A: u32, B: u32, C:u32 = {B * u32:2}> {
  x: uN[C],
}

fn Foo() {
  let b = StructFoo<u32:8>{x: u16:0};
}
)"),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          AllOf(
              HasSubstrInV1(GetParam(), "No parametric value provided for 'B'"),
              HasSubstrInV2(
                  GetParam(),
                  "No parametric value provided for `B` in `StructFoo`"))));
}

TEST_P(TypecheckBothVersionsTest, DerivedParametricStructNoParametrics) {
  XLS_EXPECT_OK(Typecheck(R"(
struct StructFoo<A: u32> {
  x: uN[A],
}

fn extract_field() -> u16 {
  let foo = StructFoo{x: u16:0};
  foo.x
}
)"));
}

// See https://github.com/google/xls/issues/1615
TEST_P(TypecheckBothVersionsTest,
       ParametricStructWithWrongOrderParametricValues) {
  EXPECT_THAT(
      Typecheck(R"(
struct StructFoo<A: u32, B: u32> {
  x: uN[A],
  y: uN[B],
}

fn wrong_order<A: u32, B: u32>(x:uN[A], y:uN[B]) -> StructFoo<B, A> {
  StructFoo<B, A>{x, y}
}

fn test() -> StructFoo<u32:32, u32:33> {
  wrong_order<u32:32, u32:33>(u32:2, u33:3)
}

)"),
      StatusIs(absl::StatusCode::kInvalidArgument,
               AllOf(HasSubstrInV1(GetParam(),
                                   "uN[33] vs uN[32]: Mismatch between member "
                                   "and argument types."),
                     HasSubstrInV2(GetParam(),
                                   "Value mismatch for parametric `A` of "
                                   "struct `StructFoo`: u32:33 vs. u32:32"))));
}

TEST_P(TypecheckBothVersionsTest,
       ParametricStructWithCorrectReverseOrderParametricValues) {
  XLS_EXPECT_OK(Typecheck(R"(
struct StructFoo<A: u32, B: u32> {
  x: uN[A],
  y: uN[B],
}

fn wrong_order<A: u32, B: u32>(x:uN[A], y:uN[B]) -> StructFoo<B, A> {
  StructFoo<B, A>{x:y, y:x}
}

fn test() -> StructFoo<u32:33, u32:32> {
  wrong_order<u32:32, u32:33>(u32:2, u33:3)
}

)"));
}

TEST_P(TypecheckBothVersionsTest, DerivedExprTypeMismatch) {
  EXPECT_THAT(
      Typecheck(R"(
fn p<X: u32, Y: bits[4] = {X+X}>(x: bits[X]) -> bits[X] { x }
fn f() -> u32 { p(u32:3) }
)"),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          AllOf(HasSubstrInV1(
                    GetParam(),
                    "Annotated type of derived parametric value did not match"),
                HasSizeMismatchInV2(GetParam(), "uN[32]", "uN[4]"))));
}

TEST_P(TypecheckBothVersionsTest, ParametricExpressionInBitsReturnType) {
  XLS_EXPECT_OK(Typecheck(R"(
fn parametric<X: u32>() -> bits[X * u32:4] {
  type Ret = bits[X * u32:4];
  Ret:0
}
fn main() -> bits[4] { parametric<u32:1>() }
)"));
}

TEST_P(TypecheckBothVersionsTest, ParametricInstantiationVsArgOk) {
  XLS_EXPECT_OK(Typecheck(R"(
fn parametric<X: u32 = {u32:5}> (x: bits[X]) -> bits[X] { x }
fn main() -> bits[5] { parametric(u5:1) }
)"));
}

TEST_P(TypecheckBothVersionsTest, ParametricInstantiationVsArgError) {
  EXPECT_THAT(
      Typecheck(R"(
fn foo<X: u32 = {u32:5}>(x: bits[X]) -> bits[X] { x }
fn bar() -> bits[10] { foo(u5:1) + foo(u10: 1) }
)"),
      StatusIs(absl::StatusCode::kInvalidArgument,
               AllOf(HasSubstrInV1(
                         GetParam(),
                         "Inconsistent parametric instantiation of function, "
                         "first saw X = u32:10; then saw X = u32:5 = u32:5"),
                     HasTypeMismatchInV2(GetParam(), "bits[5]", "uN[10]"))));
}

TEST_P(TypecheckBothVersionsTest, ParametricInstantiationVsBodyOk) {
  XLS_EXPECT_OK(Typecheck(R"(
fn parametric<X: u32 = {u32:5}>() -> bits[5] { bits[X]:1 + bits[5]:1 }
fn main() -> bits[5] { parametric() }
)"));
}

TEST_P(TypecheckBothVersionsTest, ParametricInstantiationVsBodyError) {
  EXPECT_THAT(
      Typecheck(R"(
fn foo<X: u32 = {u32:5}>() -> bits[10] { bits[X]:1 + bits[10]:1 }
fn bar() -> bits[10] { foo() }
)"),
      StatusIs(absl::StatusCode::kInvalidArgument,
               AllOf(HasSubstrInV1(GetParam(),
                                   "uN[5] vs uN[10]: Could not deduce type for "
                                   "binary operation '+'"),
                     HasSizeMismatchInV2(GetParam(), "uN[5]", "uN[10]"))));
}

TEST_P(TypecheckBothVersionsTest, ParametricInstantiationVsReturnOk) {
  XLS_EXPECT_OK(Typecheck(R"(
fn parametric<X: u32 = {u32: 5}>() -> bits[5] { bits[X]: 1 }
fn main() -> bits[5] { parametric() }
)"));
}

TEST_P(TypecheckBothVersionsTest, ParametricInstantiationVsReturnError) {
  EXPECT_THAT(
      Typecheck(R"(
fn foo<X: u32 = {u32: 5}>() -> bits[10] { bits[X]: 1 }
fn bar() -> bits[10] { foo() }
)"),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          AllOf(HasSubstrInV1(
                    GetParam(),
                    "Return type of function body for 'foo' did not match"),
                HasSizeMismatchInV2(GetParam(), "uN[5]", "uN[10]"))));
}

TEST_P(TypecheckBothVersionsTest, ParametricIndirectInstantiationVsArgOk) {
  XLS_EXPECT_OK(Typecheck(R"(
fn foo<X: u32>(x1: bits[X], x2: bits[X]) -> bits[X] { x1 + x2 }
fn fazz<Y: u32>(y: bits[Y]) -> bits[Y] { foo(y, y + bits[Y]: 1) }
fn bar() -> bits[10] { fazz(u10: 1) }
)"));
}

TEST_P(TypecheckBothVersionsTest, ParametricInstantiationVsArgError2) {
  EXPECT_THAT(
      Typecheck(R"(
fn foo<X: u32>(x1: bits[X], x2: bits[X]) -> bits[X] { x1 + x2 }
fn fazz<Y: u32>(y: bits[Y]) -> bits[Y] { foo(y, y++y) }
fn bar() -> bits[10] { fazz(u10: 1) }
)"),
      StatusIs(absl::StatusCode::kInvalidArgument,
               AllOf(HasSubstrInV1(
                         GetParam(),
                         "Parametric value X was bound to different values"),
                     HasSizeMismatchInV2(GetParam(), "uN[10]", "uN[20]"))));
}

TEST_P(TypecheckBothVersionsTest, ParametricIndirectInstantiationVsBodyOk) {
  XLS_EXPECT_OK(Typecheck(R"(
fn foo<X: u32, R: u32 = {X + X}>(x: bits[X]) -> bits[R] {
  let a = bits[R]: 5;
  x++x + a
}
fn fazz<Y: u32, T: u32 = {Y + Y}>(y: bits[Y]) -> bits[T] { foo(y) }
fn bar() -> bits[10] { fazz(u5:1) }
)"));
}

TEST_P(TypecheckBothVersionsTest, ParametricIndirectInstantiationVsBodyError) {
  EXPECT_THAT(
      Typecheck(R"(
fn foo<X: u32, D: u32 = {X + X}>(x: bits[X]) -> bits[X] {
  let a = bits[D]:5;
  x + a
}
fn fazz<Y: u32>(y: bits[Y]) -> bits[Y] { foo(y) }
fn bar() -> bits[5] { fazz(u5:1) })"),
      StatusIs(absl::StatusCode::kInvalidArgument,
               AllOf(HasSubstrInV1(GetParam(),
                                   "uN[5] vs uN[10]: Could not deduce type for "
                                   "binary operation '+'"),
                     HasSizeMismatchInV2(GetParam(), "uN[10]", "uN[5]"))));
}

TEST_P(TypecheckBothVersionsTest, ParametricIndirectInstantiationVsReturnOk) {
  XLS_EXPECT_OK(Typecheck(R"(
fn foo<X: u32, R: u32 = {X + X}>(x: bits[X]) -> bits[R] { x++x }
fn fazz<Y: u32, T: u32 = {Y + Y}>(y: bits[Y]) -> bits[T] { foo(y) }
fn bar() -> bits[10] { fazz(u5:1) }
)"));
}

TEST_P(TypecheckBothVersionsTest,
       ParametricIndirectInstantiationVsReturnError) {
  EXPECT_THAT(
      Typecheck(R"(
fn foo<X: u32, R: u32 = {X + X}>(x: bits[X]) -> bits[R] { x * x }
fn fazz<Y: u32, T: u32 = {Y + Y}>(y: bits[Y]) -> bits[T] { foo(y) }
fn bar() -> bits[10] { fazz(u5:1) }
)"),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          AllOf(HasSubstrInV1(
                    GetParam(),
                    "Return type of function body for 'foo' did not match"),
                HasSizeMismatchInV2(GetParam(), "uN[5]", "uN[10]"))));
}

TEST_P(TypecheckBothVersionsTest, ParametricDerivedInstantiationVsArgOk) {
  XLS_EXPECT_OK(Typecheck(R"(
fn foo<X: u32, Y: u32 = {X + X}>(x: bits[X], y: bits[Y]) -> bits[X] { x }
fn bar() -> bits[5] { foo(u5:1, u10: 2) }
)"));
}

TEST_P(TypecheckBothVersionsTest, ParametricDerivedInstantiationVsArgError) {
  EXPECT_THAT(
      Typecheck(R"(
fn foo<X: u32, Y: u32 = {X + X}>(x: bits[X], y: bits[Y]) -> bits[X] { x }
fn bar() -> bits[5] { foo(u5:1, u11: 2) }
)"),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          AllOf(HasSubstrInV1(
                    GetParam(),
                    "Inconsistent parametric instantiation of function, first "
                    "saw Y = u32:11; then saw Y = X + X = u32:10"),
                HasSizeMismatchInV2(GetParam(), "bits[10]", "u11"))));
}

TEST_P(TypecheckBothVersionsTest, ParametricDerivedInstantiationVsBodyOk) {
  XLS_EXPECT_OK(Typecheck(R"(
fn foo<W: u32, Z: u32 = {W + W}>(w: bits[W]) -> bits[1] {
    let val: bits[Z] = w++w + bits[Z]: 5;
    and_reduce(val)
}
fn bar() -> bits[1] { foo(u5: 5) + foo(u10: 10) }
)"));
}

TEST_P(TypecheckBothVersionsTest, ParametricDerivedInstantiationVsBodyError) {
  EXPECT_THAT(Typecheck(R"(
fn foo<W: u32, Z: u32 = {W + W}>(w: bits[W]) -> bits[1] {
  let val: bits[Z] = w + w;
  and_reduce(val)
}
fn bar() -> bits[1] { foo(u5:5) }
)"),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSizeMismatch("uN[10]", "uN[5]")));
}

TEST_P(TypecheckBothVersionsTest, ParametricDerivedInstantiationVsReturnOk) {
  XLS_EXPECT_OK(Typecheck(R"(
fn double<X: u32, Y: u32 = {X + X}> (x: bits[X]) -> bits[Y] { x++x }
fn foo<W: u32, Z: u32 = {W + W}> (w: bits[W]) -> bits[Z] { double(w) }
fn bar() -> bits[10] { foo(u5:1) }
)"));
}

TEST_P(TypecheckBothVersionsTest, ParametricDerivedInstantiationVsReturnError) {
  EXPECT_THAT(
      Typecheck(R"(
fn double<X: u32, Y: u32 = {X + X}>(x: bits[X]) -> bits[Y] { x + x }
fn foo<W: u32, Z: u32 = {W + W}>(w: bits[W]) -> bits[Z] { double(w) }
fn bar() -> bits[10] { foo(u5:1) }
)"),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          AllOf(HasSubstrInV1(
                    GetParam(),
                    "Return type of function body for 'double' did not match"),
                HasSizeMismatchInV2(GetParam(), "uN[5]", "uN[10]"))));
}

TEST_P(TypecheckBothVersionsTest, ParametricDerivedInstantiationViaFnCall) {
  XLS_EXPECT_OK(Typecheck(R"(
fn double(n: u32) -> u32 { n * u32: 2 }
fn foo<W: u32, Z: u32 = {double(W)}>(w: bits[W]) -> bits[Z] { w++w }
fn bar() -> bits[10] { foo(u5:1) }
)"));
}

TEST_P(TypecheckBothVersionsTest, ParametricFnNotAlwaysPolymorphic) {
  EXPECT_THAT(
      Typecheck(R"(
fn foo<X: u32>(x: bits[X]) -> u1 {
  let non_polymorphic = x + u5: 1;
  u1:0
}
fn bar() -> bits[1] {
  foo(u5:5) ^ foo(u10:5)
}
)"),
      StatusIs(absl::StatusCode::kInvalidArgument,
               AllOf(HasSubstrInV1(GetParam(),
                                   "uN[10] vs uN[5]: Could not deduce type for "
                                   "binary operation '+'"),
                     HasTypeMismatchInV2(GetParam(), "uN[5]", "uN[10]"))));
}

TEST_P(TypecheckBothVersionsTest, ParametricWidthSliceStartError) {
  EXPECT_THAT(
      Typecheck(R"(
fn make_u1<N: u32>(x: bits[N]) -> u1 {
  x[4 +: bits[1]]
}
fn bar() -> bits[1] {
  make_u1(u10:5) ^ make_u1(u2:1)
}
)"),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          AllOf(
              HasSubstrInV1(GetParam(), "Cannot fit slice start 4 in 2 bits"),
              HasSubstrInV2(GetParam(),
                            "Slice range out of bounds for array of size 2"))));
}

TEST_P(TypecheckBothVersionsTest, NonParametricCallInParametricExpr) {
  XLS_EXPECT_OK(Typecheck(R"(
fn id(x: u32) -> u32 { x }
fn p<X: u32, Y: u32 = {id(X)}>() -> u32 { Y }
fn main() -> u32 { p<u32:42>() }
)"));
}

TEST(TypecheckTest, ParametricCallInParametricExpr) {
  XLS_EXPECT_OK(Typecheck(R"(
fn pid<N: u32>(x: bits[N]) -> bits[N] { x }
fn p<X: u32, Y: u32 = {pid(X)}>() -> u32 { Y }
fn main() -> u32 { p<u32:42>() }
)"));
}

TEST(TypecheckTest, ParametricCallWithDeducedValuesFromArgs) {
  XLS_EXPECT_OK(Typecheck(R"(
fn umax<N: u32>(x: uN[N], y: uN[N]) -> uN[N] { if x > y { x } else { y } }

fn uadd<N: u32, M: u32, R: u32 = {umax(N, M) + u32:1}>(x: uN[N], y: uN[M]) -> uN[R] {
    (x as uN[R]) + (y as uN[R])
}

#[test]
fn uadd_test() {
    assert_eq(u4:4, uadd(u3:2, u3:2));
}
)"));
}

TEST_P(TypecheckBothVersionsTest, BitSliceOnParametricWidth) {
  XLS_EXPECT_OK(Typecheck(R"(
fn get_middle_bits<N: u32, R: u32 = {N - u32:2}>(x: bits[N]) -> bits[R] {
  x[1:-1]
}

fn caller() {
  let x1: u2 = get_middle_bits(u4:15);
  let x2: u4 = get_middle_bits(u6:63);
  ()
}
)"));
}

TEST(TypecheckErrorTest, WidthSliceWithANonType) {
  EXPECT_THAT(
      Typecheck(R"(import float32;

fn f(x: u32) -> u2 {
  x[0 +: float32::F32_EXP_SZ]  // Note this is a constant not a type.
}
)"),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr(
                   "Expected type-reference to refer to a type definition, but "
                   "this did not resolve to a type; instead got: `uN[32]`")));
}

// This test adds a literal u5 to a parametric-typed number -- which only works
// when that parametric-type number is also coincidentally a u5.
TEST(TypecheckErrorTest, ParametricMapNonPolymorphic) {
  EXPECT_THAT(Typecheck(R"(
fn add_one<N: u32>(x: bits[N]) -> bits[N] { x + u5:1 }

fn main() {
  let arr = u5[3]:[1, 2, 3];
  let mapped_arr: u5[3] = map(arr, add_one);
  let type_error = add_one(u6:1);
}
)"),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("uN[6] vs uN[5]")));
}

TEST_P(TypecheckBothVersionsTest, LetBindingInferredDoesNotMatchAnnotation) {
  EXPECT_THAT(
      Typecheck(R"(
fn f() -> u32 {
  let x: u32 = bits[4]:7;
  x
}
)"),
      StatusIs(absl::StatusCode::kInvalidArgument,
               AllOf(HasSubstrInV1(GetParam(),
                                   "Annotated type did not match inferred type "
                                   "of right hand side"),
                     HasSizeMismatchInV2(GetParam(), "bits[4]", "u32"))));
}

TEST(TypecheckErrorTest, CoverBuiltinWrongArgc) {
  EXPECT_THAT(
      Typecheck(R"(
fn f() -> () {
  cover!()
}
)"),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("Invalid number of arguments passed to 'cover!'")));
}

TEST(TypecheckErrorTest, MapBuiltinWrongArgc0) {
  EXPECT_THAT(
      Typecheck(R"(
fn f() {
  map()
}
)"),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          HasSubstr(
              "Expected 2 arguments to `map` builtin but got 0 argument(s)")));
}

TEST(TypecheckErrorTest, MapBuiltinWrongArgc1) {
  EXPECT_THAT(
      Typecheck(R"(
fn f(x: u32[3]) -> u32[3] {
  map(x)
}
)"),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          HasSubstr(
              "Expected 2 arguments to `map` builtin but got 1 argument(s)")));
}

TEST(TypecheckTest, UpdateBuiltin) {
  XLS_EXPECT_OK(Typecheck(R"(
fn f() -> u32[3] {
  let x: u32[3] = u32[3]:[0, 1, 2];
  update(x, u32:1, u32:3)
}
)"));
}

TEST(TypecheckTest, UpdateBuiltin2D) {
  XLS_EXPECT_OK(Typecheck(R"(
fn f() -> u32[2][3] {
  let x: u32[2][3] = u32[2][3]:[[u32:0,u32:1], [u32:2,u32:3], [u32:3,u32:4]];
  update(x, (u1:0, u32:1), u32:3)
}
)"));
}

TEST(TypecheckTest, UpdateBuiltinNotAnIndex) {
  EXPECT_THAT(Typecheck(R"(
fn f() -> u32[2][3] {
  let x: u32[2][3] = u32[2][3]:[[u32:0,u32:1], [u32:2,u32:3], [u32:3,u32:4]];
  update(x, [u32:0, u32:1], u32:3)
}
)"),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Want index value at argno 1 to either be a "
                                 "`uN` or a tuple of `uN`s")));
}

TEST(TypecheckTest, UpdateBuiltinIndexTupleHasSigned) {
  EXPECT_THAT(Typecheck(R"(
fn f() -> u32[2][3] {
  let x: u32[2][3] = u32[2][3]:[[u32:0,u32:1], [u32:2,u32:3], [u32:3,u32:4]];
  update(x, (u32:0, s32:1), u32:3)
}
)"),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Want index value within tuple to be `uN`; "
                                 "member 1 was `sN[32]`")));
}

TEST(TypecheckTest, UpdateBuiltinOutOfDimensions) {
  EXPECT_THAT(
      Typecheck(R"(
fn f() -> u32[2][3] {
  let x: u32[2][3] = u32[2][3]:[[u32:0,u32:1], [u32:2,u32:3], [u32:3,u32:4]];
  update(x, (u1:0, u32:1, u32:2), u32:3)
}
)"),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          HasSubstr(
              "Want argument 0 type uN[32][2][3] dimensions: 2 to be larger")));
}

TEST(TypecheckTest, UpdateBuiltinTypeMismatch) {
  EXPECT_THAT(
      Typecheck(R"(
fn f() -> u32[2][3] {
  let x: u32[2][3] = u32[2][3]:[[u32:0,u32:1], [u32:2,u32:3], [u32:3,u32:4]];
  update(x, (u1:0, u32:1), u8:3)
}
)"),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("Want argument 0 element type uN[32] to match")));
}

TEST(TypecheckTest, UpdateBuiltinEmptyIndex) {
  XLS_EXPECT_OK(Typecheck(R"(
fn f() -> u32[2][3] {
  let x: u32[2][3] = u32[2][3]:[[u32:0,u32:1], [u32:2,u32:3], [u32:3,u32:4]];
  update(x, (), u32[2][3]:[[u32:0,u32:1], [u32:2,u32:3], [u32:3,u32:4]])
}
)"));
}

TEST(TypecheckTest, SliceBuiltin) {
  XLS_EXPECT_OK(Typecheck(R"(
fn f() -> u32[3] {
  let x: u32[2] = u32[2]:[0, 1];
  array_slice(x, u32:0, u32[3]:[0, 0, 0])
}
)"));
}

TEST(TypecheckTest, EnumerateBuiltin) {
  XLS_EXPECT_OK(Typecheck(R"(
type MyTup = (u32, u2);
fn f(x: u2[7]) -> MyTup[7] {
  enumerate(x)
}
)"));
}

TEST_P(TypecheckBothVersionsTest, TernaryEmptyBlocks) {
  XLS_EXPECT_OK(Typecheck(R"(
fn f(p: bool) -> () {
  if p { } else { }
}
)"));
}

TEST_P(TypecheckBothVersionsTest, TernaryNonBoolean) {
  EXPECT_THAT(
      Typecheck(R"(
fn f(x: u32) -> u32 {
  if x { u32:42 } else { u32:64 }
}
)"),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          AllOf(HasSubstrInV1(
                    GetParam(),
                    "Test type for conditional expression is not \"bool\""),
                HasSizeMismatchInV2(GetParam(), "u32", "bool"))));
}

TEST(TypecheckTest, SizeofImportedType) {
  constexpr std::string_view kImported = R"(
pub type foo_t = u32;
)";
  constexpr std::string_view kProgram = R"(
import std;
import imported;

fn main() -> u32 {
  std::sizeof(imported::foo_t)
})";
  auto import_data = CreateImportDataForTest();
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule module,
      ParseAndTypecheck(kImported, "imported.x", "imported", &import_data));
  EXPECT_THAT(
      ParseAndTypecheck(kProgram, "fake_main_path.x", "main", &import_data),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("Cannot pass a type as a function argument.")));
}

TEST(TypecheckTest, InvalidParametricTypeConstantArrayContents) {
  constexpr std::string_view kImported = R"(
pub struct MyStruct<A: u32, B: u32> {
  a: bits[A],
  b: bits[B],
}
)";
  constexpr std::string_view kProgram = R"(
import imported;

const MY_ARRAY = imported::MyStruct<u32:4, u32:8>[2]:[
    imported::MyStruct
];
)";
  auto import_data = CreateImportDataForTest();
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule module,
      ParseAndTypecheck(kImported, "imported.x", "imported", &import_data));
  absl::StatusOr<TypecheckedModule> result =
      ParseAndTypecheck(kProgram, "fake_main_path.x", "main", &import_data);
  EXPECT_THAT(result, StatusIs(absl::StatusCode::kInvalidArgument,
                               HasSubstr("Array element cannot be a metatype")))
      << result.status();
}

TEST(TypecheckErrorTest, ArraySizeOfBitsType) {
  EXPECT_THAT(
      Typecheck(R"(
fn f(x: u32) -> u32 { array_size(x) }
)"),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          HasSubstr(
              "Want argument 0 to 'array_size' to be an array; got uN[32]")));
}

TEST(TypecheckTest, ArraySizeOfStructs) {
  XLS_EXPECT_OK(Typecheck(R"(
struct MyStruct {}
fn f(x: MyStruct[5]) -> u32 { array_size(x) }
)"));
}

TEST(TypecheckTest, ArraySizeOfNil) {
  XLS_EXPECT_OK(Typecheck(R"(
fn f(x: ()[5]) -> u32 { array_size(x) }
)"));
}

TEST(TypecheckTest, ArraySizeOfTupleArray) {
  XLS_EXPECT_OK(Typecheck(R"(
fn f(x: (u32, u64)[5]) -> u32 { array_size(x) }
)"));
}

TEST(TypecheckTest, BitSliceUpdateBuiltIn) {
  XLS_EXPECT_OK(Typecheck(R"(
fn f(x: u32, y: u17, z: u15) -> u32 {
  bit_slice_update(x, y, z)
}
)"));
}

TEST(TypecheckErrorTest, UpdateIncompatibleValue) {
  EXPECT_THAT(Typecheck(R"(
fn f(x: u32[5]) -> u32[5] {
  update(x, u32:1, u8:0)
}
)"),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("uN[32] to match argument 2 type uN[8]")));
}

TEST_P(TypecheckBothVersionsTest, MissingAnnotation) {
  XLS_EXPECT_OK(Typecheck(R"(
fn f() -> u32 {
  let x = u32:2;
  x + x
}
)"));
}

TEST(TypecheckTest, Index) {
  // Indexing a 4-element bit-constructed array with a constant index.
  XLS_EXPECT_OK(Typecheck("fn f(x: uN[32][4]) -> u32 { x[u32:0] }"));

  // Indexing a 5-element array with a dynamic index.
  XLS_EXPECT_OK(Typecheck("fn f(x: u32[5], i: u8) -> u32 { x[i] }"));

  // Indexing a bit value is not allowed, only arrays.
  EXPECT_THAT(
      Typecheck("fn f(x: u32, i: u8) -> u32 { x[i] }"),
      StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("not an array")));

  // Indexing a parametric-signedness bit value is not allowed.
  EXPECT_THAT(
      Typecheck("fn f(x: xN[false][5], i: u8) -> u1 { x[i] }"),
      StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("not an array")));

  // Cannot use an array as an index.
  EXPECT_THAT(Typecheck("fn f(x: u32[5], i: u8[5]) -> u32 { x[i] }"),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Index is not bits typed")));

  // Cannot use a signed number as an index.
  EXPECT_THAT(Typecheck("fn f(x: u32[5], i: s8) -> u32 { x[i] }"),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Index is not unsigned-bits typed")));
}

TEST_P(TypecheckBothVersionsTest, OutOfRangeNumber) {
  XLS_EXPECT_OK(Typecheck("fn f() -> u8 { u8:255 }"));
  XLS_EXPECT_OK(Typecheck("fn f() -> s8 { s8:-1 }"));
  EXPECT_THAT(
      Typecheck("fn f() -> u8 { u8:256 }"),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          HasSubstr("Value '256' does not fit in the bitwidth of a uN[8]")));
  EXPECT_THAT(
      Typecheck("fn f() -> s8 { s8:256 }"),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          HasSubstr("Value '256' does not fit in the bitwidth of a sN[8]")));
}

TEST_P(TypecheckBothVersionsTest, OutOfRangeNumberInConstantArray) {
  EXPECT_THAT(
      Typecheck("fn f() -> u8[3] { u8[3]:[1, 2, 256] }"),
      StatusIs(absl::StatusCode::kInvalidArgument,
               AllOf(HasSubstrInV1(
                         GetParam(),
                         "Value '256' does not fit in the bitwidth of a uN[8]"),
                     HasSizeMismatchInV2(GetParam(), "uN[9]", "u8"))));
}

TEST(TypecheckErrorTest, BadTypeForConstantArrayOfNumbers) {
  EXPECT_THAT(
      Typecheck("const A = u8[3][4]:[1, 2, 3, 4];"),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          HasSubstr("Non-bits type (array) used to define a numeric literal")));
}

TEST_P(TypecheckBothVersionsTest, ConstantArrayEmptyMembersWrongCountVsDecl) {
  auto result = Typecheck("const MY_ARRAY = u32[1]:[];");
  EXPECT_THAT(
      result,
      StatusIs(absl::StatusCode::kInvalidArgument,
               AllOf(HasSubstrInV1(GetParam(),
                                   "uN[32][1] Array has zero elements "
                                   "but type annotation size is 1"),
                     HasTypeMismatchInV2(GetParam(), "u32[0]", "u32[1]"))))
      << result.status();
}

TEST(TypecheckTest, MatchNoArms) {
  EXPECT_THAT(Typecheck("fn f(x: u8) -> u8 { let _ = match x {}; x }"),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Match construct has no arms")));
}

TEST_P(TypecheckBothVersionsTest, MatchArmMismatch) {
  EXPECT_THAT(
      Typecheck("fn f(x: u8) -> u8 { match x { u8:0 => u8:3, _ => u3:3 } }"),
      StatusIs(absl::StatusCode::kInvalidArgument,
               AllOf(HasSubstrInV1(GetParam(),
                                   "match arm did not have the same type"),
                     HasSizeMismatchInV2(GetParam(), "u3", "u8"))));
}

TEST(TypecheckTest, MatchOnParametricFunction) {
  absl::StatusOr<TypecheckResult> result = Typecheck(R"(
fn p<N: u32>() -> u32 {
  match p {
    p => N
  }
}

const X: u32 = p<u32:42>();
)");
  EXPECT_THAT(
      result,
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("Match construct cannot match on this type.")));
}

TEST(TypecheckErrorTest, MatchWithFunctionInPattern) {
  absl::StatusOr<TypecheckResult> result = Typecheck(R"(
fn p<N: u32>(x: bool) -> u32 {
  match x {
    p => N,
    _ => u32:0,
  }
}

const X: u32 = p<u32:42>(false);
)");
  EXPECT_THAT(result, StatusIs(absl::StatusCode::kInvalidArgument,
                               HasSubstr("pattern expects uN[1]")));
}

TEST(TypecheckErrorTest, MatchNonExhaustive) {
  absl::StatusOr<TypecheckResult> result = Typecheck(R"(
fn f(x: u32) -> u32 {
    match x {
        u32:1 => u32:64,
        u32:2 => u32:42,
    }
}
)");
  EXPECT_THAT(result, StatusIs(absl::StatusCode::kInvalidArgument,
                               HasSubstr("Match patterns are not exhaustive")));
}

TEST(TypecheckErrorTest, MatchWithOneNonExhaustivePattern) {
  absl::StatusOr<TypecheckResult> result = Typecheck(R"(
fn f(x: u32) -> u32 {
    match x {
        u32:1 => u32:64,
    }
}
)");
  EXPECT_THAT(result, StatusIs(absl::StatusCode::kInvalidArgument,
                               HasSubstr("Match pattern is not exhaustive")));
}

TEST_P(TypecheckBothVersionsTest, ArrayInconsistency) {
  EXPECT_THAT(
      Typecheck(R"(
type Foo = (u8, u32);
fn f() -> Foo {
  let xs = Foo[2]:[(u8:0, u32:1), u32:2];
  xs[u32:1]
}
)"),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          AllOf(HasTypeMismatchInV1(GetParam(), "(uN[8], uN[32])", "uN[32]"),
                HasTypeMismatchInV2(GetParam(), "(u8, u32)", "u32"),
                HasSubstrInV1(GetParam(),
                              "Array member did not have same "
                              "type as other members."))));
}

TEST_P(TypecheckBothVersionsTest, MatchWithRange) {
  XLS_EXPECT_OK(Typecheck(R"(
fn f(x: u32) -> u32 {
  match x {
    u32:0..u32:2 => x,
    _ => u32:42,
  }
}

const X: u32 = f(u32:1);
const_assert!(X == u32:1);
)"));
}

TEST_P(TypecheckBothVersionsTest, ArrayOfConsts) {
  XLS_EXPECT_OK(Typecheck(R"(
fn f() -> u4 {
  let a: u4 = u4:1;
  let my_array = [a];
  a
}
)"));
}

TEST(TypecheckTest, EnumIdentity) {
  XLS_EXPECT_OK(Typecheck(R"(
enum MyEnum : u1 {
  A = false,
  B = true,
}
fn f(x: MyEnum) -> MyEnum { x }
)"));
}

TEST(TypecheckTest, ImplicitWidthEnum) {
  XLS_EXPECT_OK(Typecheck(R"(
enum MyEnum {
  A = false,
  B = true,
}
)"));
}

TEST(TypecheckTest, ImplicitWidthEnumFromConstexprs) {
  XLS_EXPECT_OK(Typecheck(R"(
const X = u8:42;
const Y = u8:64;
enum MyEnum {
  A = X,
  B = Y,
}
)"));
}

TEST(TypecheckTest, ImplicitWidthEnumWithConstexprAndBareLiteral) {
  XLS_EXPECT_OK(Typecheck(R"(
const X = u8:42;
enum MyEnum {
  A = 64,
  B = X,
}

const EXTRACTED_A = MyEnum::A as u8;
)"));
}

TEST(TypecheckTest, ImplicitWidthEnumFromConstexprsMismatch) {
  EXPECT_THAT(Typecheck(R"(
const X = u7:42;
const Y = u8:64;
enum MyEnum {
  A = X,
  B = Y,
}
)"),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("uN[7] vs uN[8]: Inconsistent member types in "
                                 "enum definition.")));
}

TEST(TypecheckTest, ImplicitWidthEnumMismatch) {
  EXPECT_THAT(
      Typecheck(R"(
enum MyEnum {
  A = u1:0,
  B = u2:1,
}
)"),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          HasSubstr(
              "uN[1] vs uN[2]: Inconsistent member types in enum definition")));
}

TEST(TypecheckTest, ExplicitWidthEnumMismatch) {
  EXPECT_THAT(Typecheck(R"(
enum MyEnum : u2 {
  A = u1:0,
  B = u1:1,
}
)"),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("uN[1] vs uN[2]: Enum-member type did not "
                                 "match the enum's underlying type")));
}

TEST_P(TypecheckBothVersionsTest, ArrayEllipsis) {
  XLS_EXPECT_OK(Typecheck("fn main() -> u8[2] { u8[2]:[0, ...] }"));
}

// See https://github.com/google/xls/issues/1587 #1
TEST_P(TypecheckBothVersionsTest, ArrayEllipsisTypeSmallerThanElements) {
  auto result =
      Typecheck("fn main() -> u32[2] { u32[2]:[u32:0, u32:1, u32:0, ...] }");
  EXPECT_THAT(
      result,
      StatusIs(absl::StatusCode::kInvalidArgument,
               AllOf(HasSubstrInV1(GetParam(),
                                   "Annotated array size 2 is too small "
                                   "for observed array member count 3"),
                     HasSubstrInV2(GetParam(),
                                   "Annotated array size is too small for "
                                   "explicit element count"))));
}

// See https://github.com/google/xls/issues/1587 #2
TEST_P(TypecheckBothVersionsTest, ArrayEllipsisTypeEqElementCount) {
  XLS_EXPECT_OK(
      Typecheck("fn main() -> u32[2] { u32[2]:[u32:0, u32:1, ...] }"));
}

TEST_P(TypecheckBothVersionsTest, ArrayEllipsisNoTrailingElement) {
  EXPECT_THAT(
      Typecheck("fn main() -> u8[2] { u8[2]:[...] }"),
      StatusIs(absl::StatusCode::kInvalidArgument,
               AllOf(HasSubstrInV1(
                         GetParam(),
                         "Array cannot have an ellipsis without an element to "
                         "repeat; please add at least one element"),
                     HasSubstrInV2(GetParam(),
                                   "Array cannot have an ellipsis (`...`) "
                                   "without an element to repeat."))));
}

TEST_P(TypecheckBothVersionsTest, ArrayEllipsisNoLeadingTypeAnnotation) {
  constexpr std::string_view kProgram = R"(fn main() -> u8[2] {
    let x: u8[2] = [u8:0, ...];
    x
})";

  if (GetParam() == TypeInferenceVersion::kVersion1) {
    EXPECT_THAT(
        Typecheck(kProgram),
        StatusIs(absl::StatusCode::kInvalidArgument,
                 HasSubstr("does not have a type annotation; please add a type "
                           "annotation to indicate how many elements to expand "
                           "to; for example: `uN[8][N]:[u8:0, ...]`")));
  } else {
    // Unification in v2 will let the type annotation on the LHS govern the size
    // of the RHS in the absence of an RHS annotation.
    XLS_EXPECT_OK(Typecheck(kProgram));
  }
}

TEST_P(TypecheckBothVersionsTest, BadArrayAddition) {
  EXPECT_THAT(Typecheck(R"(
fn f(a: bits[32][4], b: bits[32][4]) -> bits[32][4] {
  a + b
}
)"),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Binary operations can only be applied")));
}

TEST(TypecheckTest, OneHotBadPrioType) {
  EXPECT_THAT(
      Typecheck(R"(
fn f(x: u7, prio: u2) -> u8 {
  one_hot(x, prio)
}
)"),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          HasSubstr("Expected argument 1 to 'one_hot' to be a u1; got uN[2]")));
}

TEST(TypecheckTest, OneHotSelOfSignedValues) {
  XLS_EXPECT_OK(Typecheck(R"(
fn f() -> s4 {
  let a: s4 = s4:1;
  let b: s4 = s4:2;
  let s: u2 = u2:0b01;
  one_hot_sel(s, [a, b])
}
)"));
}

TEST(TypecheckTest, OverlargeEnumValue) {
  EXPECT_THAT(
      Typecheck(R"(
enum Foo : u1 {
  A = 0,
  B = 1,
  C = 2,
}
)"),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("Value '2' does not fit in the bitwidth of a uN[1]")));
}

TEST(TypecheckTest, CannotAddEnums) {
  EXPECT_THAT(
      Typecheck(R"(
enum Foo : u2 {
  A = 0,
  B = 1,
}
fn f() -> Foo {
  Foo::A + Foo::B
}
)"),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("Cannot use '+' on values with enum type Foo")));
}

TEST(TypecheckBothVersionsTest, SlicesWithMismatchedTypes) {
  EXPECT_THAT(Typecheck("fn f(x: u8) -> u8 { x[s4:0 : s5:1] }"),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Slice limit type (sN[5]) did not match")));
}

TEST(TypecheckTest, SliceWithOutOfRangeLimit) {
  EXPECT_THAT(Typecheck("fn f(x: uN[128]) -> uN[128] { x[s4:0 :] }"),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Slice limit does not fit in index type")));
  EXPECT_THAT(Typecheck("fn f(x: uN[8]) -> uN[8] { x[s3:0 :] }"),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Slice limit does not fit in index type")));
}

TEST_P(TypecheckBothVersionsTest, SliceWithNonS32LiteralBounds) {
  // overlarge value in start
  EXPECT_THAT(
      Typecheck("fn f(x: uN[128]) -> uN[128] { x[40000000000000000000:] }"),
      StatusIs(absl::StatusCode::kInvalidArgument,
               AllOf(HasSubstrInV1(
                         GetParam(),
                         "Value '40000000000000000000' does not fit in the "
                         "bitwidth of a sN[32]"),
                     HasSizeMismatchInV2(GetParam(), "s32", "sN[67]"))));
  // overlarge value in limit
  EXPECT_THAT(
      Typecheck("fn f(x: uN[128]) -> uN[128] { x[:40000000000000000000] }"),
      StatusIs(absl::StatusCode::kInvalidArgument,
               AllOf(HasSubstrInV1(
                         GetParam(),
                         "Value '40000000000000000000' does not fit in the "
                         "bitwidth of a sN[32]"),
                     HasSizeMismatchInV2(GetParam(), "s32", "sN[67]"))));
}

TEST_P(TypecheckBothVersionsTest, WidthSlices) {
  XLS_EXPECT_OK(Typecheck("fn f(x: u32) -> bits[0] { x[0+:bits[0]] }"));

  if (GetParam() == TypeInferenceVersion::kVersion1) {
    XLS_EXPECT_OK(Typecheck("fn f(x: u32) -> u2 { x[32+:u2] }"));
  } else {
    // V2 does not permit slicing past the end.
    EXPECT_THAT(
        Typecheck("fn f(x: u32) -> u2 { x[32+:u2] }"),
        StatusIs(absl::StatusCode::kInvalidArgument,
                 HasSubstr("Slice range out of bounds for array of size 32")));
  }

  XLS_EXPECT_OK(Typecheck("fn f(x: u32) -> u1 { x[31+:u1] }"));
}

TEST_P(TypecheckBothVersionsTest, WidthSliceNegativeStartNumberLiteral) {
  if (GetParam() == TypeInferenceVersion::kVersion1) {
    // Start literal cannot be negative in v1.
    EXPECT_THAT(
        Typecheck("fn f(x: u32) -> u1 { x[-1+:u1] }"),
        StatusIs(
            absl::StatusCode::kInvalidArgument,
            HasSubstr(
                "only unsigned values are permitted; got start value: -1")));
  } else {
    // In v2, width slicing accepts any start value that regular slicing
    // accepts.
    XLS_EXPECT_OK(Typecheck("fn f(x: u32) -> u1 { x[-1+:u1] }"));
  }

  EXPECT_THAT(
      Typecheck("fn f(x: u32) -> u2 { x[-1+:u2] }"),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          AllOf(HasSubstrInV1(
                    GetParam(),
                    "only unsigned values are permitted; got start value: -1"),
                HasSubstrInV2(
                    GetParam(),
                    "Slice range out of bounds for array of size 32"))));
  EXPECT_THAT(
      Typecheck("fn f(x: u32) -> u3 { x[-2+:u3] }"),
      AllOf(StatusIs(
          absl::StatusCode::kInvalidArgument,
          AllOf(HasSubstrInV1(
                    GetParam(),
                    "only unsigned values are permitted; got start value: -2"),
                HasSubstrInV2(
                    GetParam(),
                    "Slice range out of bounds for array of size 32")))));
}

TEST_P(TypecheckBothVersionsTest, WidthSliceEmptyStartNumber) {
  // Start literal is treated as unsigned.
  XLS_EXPECT_OK(Typecheck("fn f(x: u32) -> u31 { x[:-1] }"));
  XLS_EXPECT_OK(Typecheck("fn f(x: u32) -> u30 { x[:-2] }"));
  XLS_EXPECT_OK(Typecheck("fn f(x: u32) -> u29 { x[:-3] }"));
}

TEST(TypecheckTest, WidthSliceUnsignedStart) {
  // Unsigned start literals are ok.
  XLS_EXPECT_OK(Typecheck("fn f(start: u32, x: u32) -> u3 { x[start+:u3] }"));
}

TEST(TypecheckTest, WidthSliceSignedStart) {
  // We reject signed start literals.
  EXPECT_THAT(
      Typecheck("fn f(start: s32, x: u32) -> u3 { x[start+:u3] }"),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          HasSubstr("Start index for width-based slice must be unsigned")));
}

TEST_P(TypecheckBothVersionsTest, WidthSliceTupleStart) {
  EXPECT_THAT(
      Typecheck("fn f(start: (s32), x: u32) -> u3 { x[start+:u3] }"),
      StatusIs(absl::StatusCode::kInvalidArgument,
               AllOf(HasSubstrInV1(
                         GetParam(),
                         "Start expression for width slice must be bits typed"),
                     HasTypeMismatchInV2(GetParam(), "(s32,)", "s32"))));
}

TEST_P(TypecheckBothVersionsTest, WidthSliceTupleSubject) {
  EXPECT_THAT(
      Typecheck("fn f(start: s32, x: (u32)) -> u3 { x[start+:u3] }"),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          AllOf(
              HasSubstrInV1(GetParam(), "Value to slice is not of 'bits' type"),
              HasSubstrInV2(GetParam(),
                            "Tuples should not be indexed with array-style "
                            "syntax. Use `tuple.<number>` syntax instead."))));
}

TEST_P(TypecheckBothVersionsTest, OverlargeWidthSlice) {
  EXPECT_THAT(
      Typecheck("fn f(x: u32) -> u33 { x[0+:u33] }"),
      StatusIs(absl::StatusCode::kInvalidArgument,
               AllOf(HasSubstrInV1(GetParam(),
                                   "Slice type must have <= original number of "
                                   "bits; attempted slice from 32 to 33 bits."),
                     HasSubstrInV2(
                         GetParam(),
                         "Slice range out of bounds for array of size 32"))));
}

TEST_P(TypecheckBothVersionsTest, BadAttributeAccessOnTuple) {
  EXPECT_THAT(
      Typecheck(R"(
fn main() -> () {
  let x: (u32,) = (u32:42,);
  x.a
}
)"),
      StatusIs(absl::StatusCode::kInvalidArgument,
               AllOf(HasSubstrInV1(GetParam(),
                                   "Expected a struct for attribute access"),
                     HasSubstrInV2(GetParam(),
                                   "Invalid access of member `a` of non-struct "
                                   "type: `(uN[32],)`"))));
}

TEST_P(TypecheckBothVersionsTest, BadAttributeAccessOnBits) {
  EXPECT_THAT(
      Typecheck(R"(
fn main() -> () {
  let x = u32:42;
  x.a
}
)"),
      StatusIs(absl::StatusCode::kInvalidArgument,
               AllOf(HasSubstrInV1(GetParam(),
                                   "Expected a struct for attribute access"),
                     HasSubstrInV2(
                         GetParam(),
                         "Builtin type 'u32' does not have attribute 'a'"))));
}

TEST_P(TypecheckBothVersionsTest, BadArrayLiteralType) {
  EXPECT_THAT(
      Typecheck("const X = s32:[1, 2];"),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          AllOf(HasSubstrInV1(GetParam(),
                              "Array was not annotated with an array type"),
                HasTypeMismatchInV2(GetParam(), "s32", "uN[2][2]"))));
}

TEST_P(TypecheckBothVersionsTest, CharLiteralArray) {
  XLS_EXPECT_OK(Typecheck(R"(
fn main() -> u8[3] {
  u8[3]:['X', 'L', 'S']
}
)"));
}

TEST(TypecheckTest, BadEnumRef) {
  EXPECT_THAT(
      Typecheck(R"(
enum MyEnum : u1 { A = 0, B = 1 }
fn f() -> MyEnum { MyEnum::C }
)"),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("Name 'C' is not defined by the enum MyEnum")));
}

// Nominal typing not structural, e.g. OtherPoint cannot be passed where we want
// a Point, even though their members are the same.
TEST_P(TypecheckBothVersionsTest, NominalTyping) {
  EXPECT_THAT(
      Typecheck(R"(
struct Point { x: s8, y: u32 }
struct OtherPoint { x: s8, y: u32 }
fn f(x: Point) -> Point { x }
fn g() -> Point {
  let shp = OtherPoint { x: s8:-1, y: u32:1024 };
  f(shp)
}
)"),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          AllOf(HasSubstrInV1(GetParam(),
                              "Point { x: sN[8], y: uN[32] }\nvs OtherPoint "
                              "{ x: sN[8], y: uN[32] }"),
                HasTypeMismatchInV2(GetParam(), "OtherPoint", "Point"))));
}

TEST_P(TypecheckBothVersionsTest, ParametricWithConstantArrayEllipsis) {
  XLS_EXPECT_OK(Typecheck(R"(
fn p<N: u32>(_: bits[N]) -> u8[2] { u8[2]:[0, ...] }
fn main() -> u8[2] { p(false) }
)"));
}

// In this test case we:
// * make a call to `q`, where we give it an explicit parametric value,
// * by invoking `r` (which is also parametric),
// * doing that from within a parametric function `p`.
TEST(TypecheckBothVersionsTest, ExplicitParametricCallInParametricFn) {
  XLS_EXPECT_OK(Typecheck(R"(
fn r<R: u32>(x: bits[R]) -> bits[R] { x }
fn q<Q: u32>(x: bits[Q]) -> bits[Q] { x }
fn p<P: u32>(x: bits[P]) -> bits[P] { q<{r(P+u32:0)}>(x) }
fn main() -> u32 { p(u32:42) }
)"));
}

TEST(TypecheckErrorTest, BadQuickcheckFunctionRet) {
  EXPECT_THAT(Typecheck(R"(
#[quickcheck]
fn f() -> u5 { u5:1 }
)"),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("must return a bool")));
}

TEST(TypecheckErrorTest, BadQuickcheckFunctionParametrics) {
  EXPECT_THAT(
      Typecheck(R"(
#[quickcheck]
fn f<N: u32>() -> bool { true }
)"),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("Quickchecking parametric functions is unsupported")));
}

TEST_P(TypecheckBothVersionsTest, NumbersAreConstexpr) {
  // Visitor to check all nodes in the below program to determine if all numbers
  // are indeed constexpr.
  class IsConstVisitor : public AstNodeVisitorWithDefault {
   public:
    explicit IsConstVisitor(TypeInfo* type_info) : type_info_(type_info) {}

    absl::Status HandleFunction(const Function* node) override {
      XLS_RETURN_IF_ERROR(node->body()->Accept(this));
      return absl::OkStatus();
    }

    absl::Status HandleStatementBlock(const StatementBlock* node) override {
      for (auto child : node->GetChildren(/*want_types=*/false)) {
        XLS_RETURN_IF_ERROR(child->Accept(this));
      }
      return absl::OkStatus();
    }

    absl::Status HandleStatement(const Statement* node) override {
      for (auto child : node->GetChildren(/*want_types=*/false)) {
        XLS_RETURN_IF_ERROR(child->Accept(this));
      }
      return absl::OkStatus();
    }

    absl::Status HandleLet(const Let* node) override {
      XLS_RETURN_IF_ERROR(node->rhs()->Accept(this));
      return absl::OkStatus();
    }

    absl::Status HandleNumber(const Number* node) override {
      if (type_info_->GetConstExpr(node).ok()) {
        constexpr_numbers_seen_++;
      } else {
        nonconstexpr_numbers_seen_++;
      }
      return absl::OkStatus();
    }

    int constexpr_numbers_seen() { return constexpr_numbers_seen_; }
    int nonconstexpr_numbers_seen() { return nonconstexpr_numbers_seen_; }

   private:
    int constexpr_numbers_seen_ = 0;
    int nonconstexpr_numbers_seen_ = 0;
    TypeInfo* type_info_;
  };

  constexpr std::string_view kProgram = R"(
fn main() {
  let foo = u32:0;
  let foo = u64:0x666;
  ()
}
)";

  ImportData import_data(CreateImportDataForTest());
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule tm,
      ParseAndTypecheck(kProgram, "fake.x", "fake", &import_data));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f,
                           tm.module->GetMemberOrError<Function>("main"));
  IsConstVisitor visitor(tm.type_info);
  XLS_ASSERT_OK(f->Accept(&visitor));
  EXPECT_EQ(visitor.constexpr_numbers_seen(), 2);
  EXPECT_EQ(visitor.nonconstexpr_numbers_seen(), 0);
}

TEST_P(TypecheckBothVersionsTest, BasicTupleIndex) {
  XLS_EXPECT_OK(Typecheck(R"(
fn main() -> u18 {
  (u32:7, u24:6, u18:5, u12:4, u8:3).2
}
)"));
}

TEST_P(TypecheckBothVersionsTest, DuplicateRestOfTupleError) {
  EXPECT_THAT(Typecheck(R"(
fn main() {
  let (x, .., ..) = (u32:7, u24:6, u18:5, u12:4, u8:3);
}
)"),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("can only be used once")));
}

TEST_P(TypecheckBothVersionsTest, TupleCountMismatch) {
  EXPECT_THAT(Typecheck(R"(
fn main() {
  let (x, y) = (u32:7, u24:6, u18:5, u12:4, u8:3);
}
)"),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("a 5-element tuple to 2 values")));
}

TEST_P(TypecheckBothVersionsTest, RestOfTupleCountMismatch) {
  EXPECT_THAT(Typecheck(R"(
fn main() {
  let (x, .., y, z) = (u32:7, u8:3);
}
)"),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("a 2-element tuple to 3 values")));
}

TEST_P(TypecheckBothVersionsTest, RestOfTupleCountMismatchNested) {
  EXPECT_THAT(Typecheck(R"(
fn main() {
  let (x, .., (y, .., z)) = (u32:7, u8:3, (u12:4,));
}
)"),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("a 1-element tuple to 2 values")));
}

TEST_P(TypecheckBothVersionsTest, TupleAssignsTypes) {
  constexpr std::string_view kProgram = R"(
fn main() {
  let (x, y): (u32, s8) = (u32:7, s8:3);
}
)";
  XLS_EXPECT_OK(Typecheck(kProgram));
}

TEST_P(TypecheckBothVersionsTest, RestOfTupleSkipsMiddle) {
  constexpr std::string_view kProgram = R"(
fn main() {
  let (x, .., y) = (u32:7, u12:4, s8:3);
  let (xx, yy): (u32, s8) = (x, y);
}
)";
  XLS_EXPECT_OK(Typecheck(kProgram));
}

TEST_P(TypecheckBothVersionsTest, RestOfTupleSkipsNone) {
  constexpr std::string_view kProgram = R"(
fn main() {
  let (x, .., y) = (u32:7, s8:3);
  let (xx, yy): (u32, s8) = (x, y);
}
)";
  XLS_EXPECT_OK(Typecheck(kProgram));
}

TEST_P(TypecheckBothVersionsTest, RestOfTuplekSkipsNoneWithThree) {
  constexpr std::string_view kProgram = R"(
fn main() {
  let (x, y, .., z) = (u32:7, u12:4, s8:3);
  let (xx, yy, zz): (u32, u12, s8) = (x, y, z);
}
)";
  XLS_EXPECT_OK(Typecheck(kProgram));
}

TEST_P(TypecheckBothVersionsTest, RestOfTupleSkipsEnd) {
  constexpr std::string_view kProgram = R"(
fn main() {
  let (x, y, ..) = (u32:7, s8:3, u12:4);
  let (xx, yy): (u32, s8) = (x, y);
}
)";
  XLS_EXPECT_OK(Typecheck(kProgram));
}

TEST_P(TypecheckBothVersionsTest, RestOfTupleSkipsManyAtEnd) {
  constexpr std::string_view kProgram = R"(
fn main() {
  let (x, y, ..) = (u32:7, s8:3, u12:4, u32:0);
  let (xx, yy): (u32, s8) = (x, y);
}
)";
  XLS_EXPECT_OK(Typecheck(kProgram));
}

TEST_P(TypecheckBothVersionsTest, RestOfTupleSkipsManyInMiddle) {
  constexpr std::string_view kProgram = R"(
fn main() {
  let (x, .., y) = (u32:7, u8:3, u12:4, s8:3);
  let (xx, yy): (u32, s8) = (x, y);
}
)";
  XLS_EXPECT_OK(Typecheck(kProgram));
}

TEST_P(TypecheckBothVersionsTest, RestOfTupleSkipsBeginning) {
  constexpr std::string_view kProgram = R"(
fn main() {
  let (.., x, y) = (u12:7, u8:3, u32:4, s8:3);
  let (xx, yy): (u32, s8) = (x, y);
}
)";
  XLS_EXPECT_OK(Typecheck(kProgram));
}

TEST_P(TypecheckBothVersionsTest, RestOfTupleSkipsManyAtBeginning) {
  constexpr std::string_view kProgram = R"(
fn main() {
  let (.., x) = (u8:3, u12:4, u32:7);
  let xx: u32 = x;
}
)";
  XLS_EXPECT_OK(Typecheck(kProgram));
}

TEST_P(TypecheckBothVersionsTest, RestOfTupleNested) {
  constexpr std::string_view kProgram = R"(
fn main() {
  let (x, .., (.., y)) = (u32:7, u8:3, u18:5, (u12:4, u11:5, s8:3));
  let (xx, yy): (u32, s8) = (x, y);
}
)";
  XLS_EXPECT_OK(Typecheck(kProgram));
}

TEST_P(TypecheckBothVersionsTest, RestOfTupleNestedSingleton) {
  constexpr std::string_view kProgram = R"(
fn main() {
  let (x, .., (y,)) = (u32:7, u8:3, (s8:3,));
  let (xx, yy): (u32, s8) = (x, y);
}
)";
  XLS_EXPECT_OK(Typecheck(kProgram));
}

TEST_P(TypecheckBothVersionsTest, RestOfTupleIsLikeWildcard) {
  constexpr std::string_view kProgram = R"(
fn main() {
  let (x, .., (.., y)) = (u32:7, u18:5, (u12:4, s8:3));
  let (xx, yy): (u32, s8) = (x, y);
}
)";
  XLS_EXPECT_OK(Typecheck(kProgram));
}

TEST_P(TypecheckBothVersionsTest, RestOfTupleDeeplyNested) {
  constexpr std::string_view kProgram = R"(
fn main() {
  let (x, y, .., ((.., z), .., d)) = (u32:7, u8:1,
                            ((u32:3, u64:4, uN[128]:5), u12:4, s8:3));
  let (xx, yy, zz): (u32, u8, uN[128]) = (x, y, z);
  }
)";
  XLS_EXPECT_OK(Typecheck(kProgram));
}

TEST_P(TypecheckBothVersionsTest, RestOfTupleDeeplyNestedNonConstants) {
  constexpr std::string_view kProgram = R"(
fn main() {
  // Initial values
  let (xi, yi, zi): (u32, u8, uN[128]) = (u32:7, u8:1, uN[128]:5);
  let (x, y, .., ((.., z), .., d)) = (xi, yi,
                            ((u32:3, u64:4, zi), u12:4, s8:3));
  let (xx, yy, zz): (u32, u8, uN[128]) = (x, y, z);
  }
)";
  XLS_EXPECT_OK(Typecheck(kProgram));
}

TEST(TypecheckTest, BasicRange) {
  constexpr std::string_view kProgram = R"(#[test]
fn main() {
  let a = u32:0..u32:4;
  let b = u32[4]:[0, 1, 2, 3];
  assert_eq(a, b)
}
)";

  XLS_EXPECT_OK(Typecheck(kProgram));
}

// Helper for struct instance based tests.
static absl::Status TypecheckStructInstance(TypeInferenceVersion version,
                                            std::string program) {
  program = R"(
struct Point {
  x: s8,
  y: u32,
}
)" + program;
  return version == TypeInferenceVersion::kVersion1
             ? Typecheck(program).status()
             : TypecheckV2(program).status();
}

TEST_P(TypecheckBothVersionsTest, AccessMissingStructMember) {
  EXPECT_THAT(
      TypecheckStructInstance(GetParam(), "fn f(p: Point) -> () { p.z }"),
      StatusIs(absl::StatusCode::kInvalidArgument,
               AllOf(HasSubstrInV1(
                         GetParam(),
                         "Struct 'Point' does not have a member with name 'z'"),
                     HasSubstrInV2(GetParam(),
                                   "No member `z` in struct `Point`."))));
}

TEST_P(TypecheckBothVersionsTest, WrongTypeInStructInstanceMember) {
  EXPECT_THAT(
      TypecheckStructInstance(
          GetParam(), "fn f() -> Point { Point { y: u8:42, x: s8:-1 } }"),
      StatusIs(absl::StatusCode::kInvalidArgument,
               AllOf(HasSizeMismatchInV1(GetParam(), "uN[32]", "uN[8]"),
                     HasSizeMismatchInV2(GetParam(), "u32", "u8"))));
}

TEST_P(TypecheckBothVersionsTest, MissingFieldXInStructInstance) {
  EXPECT_THAT(
      TypecheckStructInstance(GetParam(),
                              "fn f() -> Point { Point { y: u32:42 } }"),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          AllOf(HasSubstrInV1(GetParam(),
                              "Struct instance is missing member(s): 'x'"),
                HasSubstrInV2(
                    GetParam(),
                    "Instance of struct `Point` is missing member(s): `x`"))));
}

TEST_P(TypecheckBothVersionsTest, MissingFieldYInStructInstance) {
  EXPECT_THAT(
      TypecheckStructInstance(GetParam(),
                              "fn f() -> Point { Point { x: s8: -1 } }"),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          AllOf(HasSubstrInV1(GetParam(),
                              "Struct instance is missing member(s): 'y'"),
                HasSubstrInV2(
                    GetParam(),
                    "Instance of struct `Point` is missing member(s): `y`"))));
}

TEST_P(TypecheckBothVersionsTest, OutOfOrderStructInstanceOk) {
  XLS_EXPECT_OK(TypecheckStructInstance(
      GetParam(), "fn f() -> Point { Point { y: u32:42, x: s8:-1 } }"));
}

TEST_P(TypecheckBothVersionsTest, ProvideExtraFieldZInStructInstance) {
  EXPECT_THAT(
      TypecheckStructInstance(
          GetParam(),
          "fn f() -> Point { Point { x: s8:-1, y: u32:42, z: u32:1024 } }"),
      StatusIs(absl::StatusCode::kInvalidArgument,
               AllOf(HasSubstrInV1(
                         GetParam(),
                         "Struct \'Point\' has no member \'z\', but it was "
                         "provided by this instance."),
                     HasSubstrInV2(GetParam(),
                                   "Struct `Point` has no member `z`, but it "
                                   "was provided by this instance."))));
}

TEST_P(TypecheckBothVersionsTest, DuplicateFieldYInStructInstance) {
  EXPECT_THAT(
      TypecheckStructInstance(
          GetParam(),
          "fn f() -> Point { Point { x: s8:-1, y: u32:42, y: u32:1024 } }"),
      StatusIs(absl::StatusCode::kInvalidArgument,
               AllOf(HasSubstrInV1(
                         GetParam(),
                         "Duplicate value seen for \'y\' in this \'Point\' "
                         "struct instance."),
                     HasSubstrInV2(GetParam(),
                                   "Duplicate value seen for `y` in this "
                                   "`Point` struct instance."))));
}

TEST_P(TypecheckBothVersionsTest, StructIncompatibleWithTupleEquivalent) {
  EXPECT_THAT(
      TypecheckStructInstance(GetParam(), R"(
fn f(x: (s8, u32)) -> (s8, u32) { x }
fn g() -> (s8, u32) {
  let p = Point { x: s8:-1, y: u32:1024 };
  f(p)
}
)"),
      StatusIs(absl::StatusCode::kInvalidArgument,
               AllOf(HasTypeMismatchInV1(GetParam(), "(sN[8], uN[32])",
                                         "Point { x: sN[8], y: uN[32] }"),
                     HasTypeMismatchInV2(GetParam(), "(s8, u32)", "Point"))));
}

TEST_P(TypecheckBothVersionsTest, SplatWithDuplicate) {
  EXPECT_THAT(
      TypecheckStructInstance(
          GetParam(),
          "fn f(p: Point) -> Point { Point { x: s8:42, x: s8:64, ..p } }"),
      StatusIs(absl::StatusCode::kInvalidArgument,
               AllOf(HasSubstrInV1(
                         GetParam(),
                         "Duplicate value seen for \'x\' in this \'Point\' "
                         "struct instance."),
                     HasSubstrInV2(GetParam(),
                                   "Duplicate value seen for `x` in this "
                                   "`Point` struct instance."))));
}

TEST_P(TypecheckBothVersionsTest, SplatWithExtraFieldQ) {
  EXPECT_THAT(
      TypecheckStructInstance(
          GetParam(), "fn f(p: Point) -> Point { Point { q: u32:42, ..p } }"),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          AllOf(HasSubstrInV1(GetParam(), "Struct 'Point' has no member 'q'"),
                HasSubstrInV2(GetParam(),
                              "Struct `Point` has no member `q`, but it was "
                              "provided by this instance."))));
}

TEST_P(TypecheckBothVersionsTest, MulExprInStructMember) {
  const std::string_view kProgram = R"(
struct Point<N: u32> {
  x: uN[N],
  y: uN[N * u32:2]
}

fn f(p: Point<3>) -> uN[6] {
  p.y
}
)";
  XLS_EXPECT_OK(Typecheck(kProgram));
}

TEST_P(TypecheckBothVersionsTest, NonU32Parametric) {
  const std::string_view kProgram = R"(
struct Point<N: u5, N_U32: u32 = {N as u32}> {
  x: uN[N_U32],
}

fn f(p: Point<u5:3>) -> uN[3] {
  p.x
}
)";
  if (GetParam() != TypeInferenceVersion::kVersion1) {
    XLS_EXPECT_OK(Typecheck(kProgram));
  }
}

// Helper for parametric struct instance based tests.
static absl::Status TypecheckParametricStructInstance(
    TypeInferenceVersion version, std::string program) {
  program = R"(
struct Point<N: u32, M: u32 = {N + N}> {
  x: bits[N],
  y: bits[M],
}
)" + program;
  return version == TypeInferenceVersion::kVersion1
             ? Typecheck(program).status()
             : TypecheckV2(program).status();
}

TEST(TypecheckParametricStructInstanceTest, WrongDerivedType) {
  EXPECT_THAT(
      TypecheckParametricStructInstance(
          TypeInferenceVersion::kVersion1,
          "fn f() -> Point<32, 63> { Point { x: u32:5, y: u63:255 } }"),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("first saw M = u32:63; then saw M = N + N = u32:64")));
}

TEST_P(TypecheckBothVersionsTest, TooManyParametricStructArgs) {
  EXPECT_THAT(
      TypecheckParametricStructInstance(
          GetParam(),
          "fn f() -> Point<u32:5, u32:10, u32:15> { Point { x: u5:5, y: "
          "u10:255 } }"),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          AllOf(
              HasSubstrInV1(
                  GetParam(),
                  "Expected 2 parametric arguments for 'Point'; got 3"),
              HasSubstrInV2(
                  GetParam(),
                  "Too many parametric values supplied; limit: 2 given: 3"))));
}

TEST_P(TypecheckBothVersionsTest, PhantomParametricStructReturnTypeMismatch) {
  // Erroneous code.
  EXPECT_THAT(
      Typecheck(
          R"(struct MyStruct<N: u32> {}
          fn main(x: MyStruct<u32:8>) -> MyStruct<u32:42> { x }
      )"),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          AllOf(HasSubstrInV1(
                    GetParam(),
                    "Parametric argument of the returned value does not match "
                    "the function return type. Expected 42; got 8."),
                HasSubstrInV2(GetParam(),
                              "Value mismatch for parametric `N` of struct "
                              "`MyStruct`: u32:8 vs. u32:42"))));

  // Fixed version.
  XLS_EXPECT_OK(Typecheck(
      R"(struct MyStruct<N: u32> {}
          fn main(x: MyStruct<u32:42>) -> MyStruct<u32:42> { x }
      )"));
}

TEST_P(TypecheckBothVersionsTest,
       PhantomParametricParameterizedStructReturnType) {
  // Erroneous code.
  EXPECT_THAT(
      Typecheck(
          R"(struct MyStruct<N: u32> {}
          fn foo<N: u32>(x: MyStruct<u32:8>) -> MyStruct<N> { x }
          fn bar(x: MyStruct<u32:8>) -> MyStruct<u32:8> { foo<u32:42>(x) }
      )"),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          AllOf(HasSubstrInV1(
                    GetParam(),
                    "Parametric argument of the returned value does not match "
                    "the function return type. Expected 8; got 42."),
                HasSubstrInV2(GetParam(),
                              "Value mismatch for parametric `N` of struct "
                              "`MyStruct`: u32:42 vs. u32:8"))));

  // Fixed version.
  XLS_EXPECT_OK(Typecheck(
      R"(struct MyStruct<N: u32> {}
          fn foo<N: u32>(x: MyStruct<u32:8>) -> MyStruct<N> { x }
          fn bar(x: MyStruct<u32:8>) -> MyStruct<u32:8> { foo<u32:8>(x) }
      )"));
}

TEST_P(TypecheckBothVersionsTest, PhantomParametricWithExpr) {
  // Erroneous code.
  EXPECT_THAT(
      Typecheck(
          R"(struct MyStruct<M: u32, N: u32 = {M + M}> {}
          fn main(x: MyStruct<u32:8, u32:16>) -> MyStruct<u32:42, u32:84> { x }
      )"),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          AllOf(HasSubstrInV1(
                    GetParam(),
                    "Parametric argument of the returned value does not match "
                    "the function return type. Expected 42; got 8."),
                HasSubstrInV2(GetParam(),
                              "Value mismatch for parametric `M` of struct "
                              "`MyStruct`: u32:8 vs. u32:42"))));

  // Fixed version.
  XLS_EXPECT_OK(Typecheck(
      R"(struct MyStruct<M: u32, N: u32 = {M + M}> {}
          fn main(x: MyStruct<u32:8, u32:16>) -> MyStruct<u32:8, u32:16> { x }
      )"));
}

TEST_P(TypecheckBothVersionsTest, OutOfOrderOk) {
  XLS_EXPECT_OK(TypecheckParametricStructInstance(
      GetParam(),
      "fn f() -> Point<32, 64> { Point { y: u64:42, x: u32:255 } }"));
}

TEST_P(TypecheckBothVersionsTest, OkInstantiationInParametricFunction) {
  XLS_EXPECT_OK(TypecheckParametricStructInstance(GetParam(), R"(
fn f<A: u32, B: u32>(x: bits[A], y: bits[B]) -> Point<A, B> { Point { x, y } }
fn main() {
  let _ = f(u5:1, u10:2);
  let _ = f(u14:1, u28:2);
  ()
}
)"));
}

TEST_P(TypecheckBothVersionsTest, BadParametricStructReturnType) {
  EXPECT_THAT(
      TypecheckParametricStructInstance(
          GetParam(),
          "fn f() -> Point<5, 10> { Point { x: u32:5, y: u64:255 } }"),
      StatusIs(absl::StatusCode::kInvalidArgument,
               AllOf(HasSubstrInV1(GetParam(),
                                   "Point { x: uN[32], y: uN[64] }\nvs Point { "
                                   "x: uN[5], y: uN[10] }"),
                     HasSizeMismatchInV2(GetParam(), "u32", "bits[5]"))));
}

// Bad struct type-parametric instantiation in parametric function.
TEST(TypecheckParametricStructInstanceTest, BadParametricInstantiation) {
  EXPECT_THAT(
      TypecheckParametricStructInstance(TypeInferenceVersion::kVersion1, R"(
fn f<A: u32, B: u32>(x: bits[A], y: bits[B]) -> Point<A, B> {
  Point { x, y }
}

fn main() {
  let _ = f(u5:1, u10:2);
  let _ = f(u14:1, u15:2);
  ()
}
)"),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          HasSubstr("Inconsistent parametric instantiation of struct, first "
                    "saw M = u32:15; then saw M = N + N = u32:28")));
}

TEST_P(TypecheckBothVersionsTest, BadParametricSplatInstantiation) {
  EXPECT_THAT(
      TypecheckParametricStructInstance(GetParam(), R"(
fn f<A: u32, B: u32>(x: bits[A], y: bits[B]) -> Point<A, B> {
  let p = Point { x, y };
  Point { x: (x++x), ..p }
}

fn main() {
  let _ = f(u5:1, u10:2);
  ()
}
)"),
      StatusIs(absl::StatusCode::kInvalidArgument,
               AllOf(HasSubstrInV1(
                         GetParam(),
                         "first saw M = u32:10; then saw M = N + N = u32:20"),
                     HasSizeMismatchInV2(GetParam(), "uN[10]", "uN[5]"))));
}

TEST_P(TypecheckBothVersionsTest, AttrViaColonRef) {
  XLS_EXPECT_OK(Typecheck("fn f() -> u8 { u8::ZERO }"));
  XLS_EXPECT_OK(Typecheck("fn f() -> u8 { u8::MAX }"));
  XLS_EXPECT_OK(Typecheck("fn f() -> u8 { u8::MIN }"));
}

TEST_P(TypecheckBothVersionsTest, ColonRefTypeAlias) {
  XLS_EXPECT_OK(Typecheck(R"(
type MyU8 = u8;
fn f() -> u8 { MyU8::MAX }
fn g() -> u8 { MyU8::ZERO }
fn h() -> u8 { MyU8::MIN }
)"));
}

TEST_P(TypecheckBothVersionsTest, MinAttrUsedInConstAsserts) {
  XLS_EXPECT_OK(Typecheck(R"(
const_assert!(u8::MIN == u8:0);
const_assert!(s4::MIN == s4:-8);
)"));
}

TEST_P(TypecheckBothVersionsTest, MaxAttrUsedToDefineAType) {
  XLS_EXPECT_OK(Typecheck(R"(
type MyU255 = uN[u8::MAX as u32];
fn f() -> MyU255 { uN[255]:42 }
)"));
}

TEST_P(TypecheckBothVersionsTest, ZeroAttrUsedToDefineAType) {
  XLS_EXPECT_OK(Typecheck(R"(
type MyU0 = uN[u8::ZERO as u32];
fn f() -> MyU0 { bits[0]:0 }
)"));
}

TEST_P(TypecheckBothVersionsTest, TypeAliasOfStructWithBoundParametrics) {
  XLS_EXPECT_OK(Typecheck(R"(
struct S<X: u32, Y: u32> {
  x: bits[X],
  y: bits[Y],
}
type MyS = S<3, 4>;
fn f() -> MyS { MyS{x: bits[3]:3, y: bits[4]:4 } }
)"));
}

TEST(TypecheckTest, SplatWithAllStructMembersSpecifiedGivesWarning) {
  const std::string program = R"(
struct S {
  x: u32,
  y: u32,
}
fn f(s: S) -> S { S{x: u32:4, y: u32:8, ..s} }
)";
  XLS_ASSERT_OK_AND_ASSIGN(TypecheckResult tr, Typecheck(program));
  TypecheckedModule& tm = tr.tm;
  Fileno fileno = tr.import_data->file_table().GetOrCreate("fake.x");

  ASSERT_THAT(tm.warnings.warnings().size(), 1);
  EXPECT_EQ(tm.warnings.warnings().at(0).span,
            Span(Pos(fileno, 5, 42), Pos(fileno, 5, 43)));
  EXPECT_EQ(tm.warnings.warnings().at(0).message,
            "'Splatted' struct instance has all members of struct defined, "
            "consider removing the `..s`");
  UniformContentFilesystem vfs(program);
  XLS_ASSERT_OK(PrintPositionalError(
      tm.warnings.warnings().at(0).span, tm.warnings.warnings().at(0).message,
      std::cerr, PositionalErrorColor::kWarningColor,
      tr.import_data->file_table(), vfs));
}

TEST(TypecheckTest, LetWithWildcardMatchGivesWarning) {
  const std::string program = R"(
fn f(x: u32) -> u32 {
  let _ = x + x;
  x
}
)";
  XLS_ASSERT_OK_AND_ASSIGN(TypecheckResult tr, Typecheck(program));
  TypecheckedModule& tm = tr.tm;
  FileTable& file_table = tr.import_data->file_table();
  Fileno fileno = file_table.GetOrCreate("fake.x");

  ASSERT_THAT(tm.warnings.warnings().size(), 1);
  EXPECT_EQ(tm.warnings.warnings().at(0).span,
            Span(Pos(fileno, 2, 6), Pos(fileno, 2, 7)));
  EXPECT_EQ(tm.warnings.warnings().at(0).message,
            "`let _ = expr;` statement can be simplified to `expr;` -- there "
            "is no need for a `let` binding here");
  UniformContentFilesystem vfs(program);
  XLS_ASSERT_OK(PrintPositionalError(
      tm.warnings.warnings().at(0).span, tm.warnings.warnings().at(0).message,
      std::cerr, PositionalErrorColor::kWarningColor, file_table, vfs));
}

TEST(TypecheckTest, UselessTrailingNilGivesWarning) {
  const std::string program = R"(
fn f() -> () {
  trace_fmt!("oh no");
  ()
}
)";
  XLS_ASSERT_OK_AND_ASSIGN(TypecheckResult tr, Typecheck(program));
  TypecheckedModule& tm = tr.tm;
  FileTable& file_table = tr.import_data->file_table();
  Fileno fileno = file_table.GetOrCreate("fake.x");

  ASSERT_THAT(tm.warnings.warnings().size(), 1);
  EXPECT_EQ(tm.warnings.warnings().at(0).span,
            Span(Pos(fileno, 3, 2), Pos(fileno, 3, 4)));
  EXPECT_EQ(tm.warnings.warnings().at(0).message,
            "Block has a trailing nil (empty) tuple after a semicolon -- this "
            "is implied, please remove it");
  UniformContentFilesystem vfs(program);
  XLS_ASSERT_OK(PrintPositionalError(
      tm.warnings.warnings().at(0).span, tm.warnings.warnings().at(0).message,
      std::cerr, PositionalErrorColor::kWarningColor, file_table, vfs));
}

TEST_P(TypecheckBothVersionsTest, NonstandardConstantNamingGivesWarning) {
  const constexpr std::string_view kProgram = R"(const mol = u32:42;)";
  XLS_ASSERT_OK_AND_ASSIGN(TypecheckResult tr, Typecheck(kProgram));
  TypecheckedModule& tm = tr.tm;
  ASSERT_THAT(tm.warnings.warnings().size(), 1);
  EXPECT_EQ(tm.warnings.warnings().at(0).message,
            "Standard style is SCREAMING_SNAKE_CASE for constant identifiers; "
            "got: `mol`");
}

TEST_P(TypecheckBothVersionsTest, NonstandardConstantNamingOkViaAllow) {
  const constexpr std::string_view kProgram =
      R"(#![allow(nonstandard_constant_naming)]
const mol = u32:42;)";
  XLS_ASSERT_OK_AND_ASSIGN(TypecheckResult tr, Typecheck(kProgram));
  TypecheckedModule& tm = tr.tm;
  ASSERT_TRUE(tm.warnings.warnings().empty());
}

TEST(TypecheckTest, BadTraceFmtWithUseOfChannel) {
  constexpr std::string_view kProgram =
      R"(
proc Counter {
  in_ch: chan<u32> in;
  out_ch: chan<u32> out;

  init {
  }

  config(in_ch: chan<u32> in, out_ch: chan<u32> out) {
    (in_ch, out_ch)
  }

  next(state: ()) {
    let (tok, in_data) = recv(join(), in_ch);
    trace_fmt!("{}", in_ch);
    send(tok, out_ch, in_data);
  }
}
)";

  EXPECT_THAT(
      Typecheck(kProgram),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("Cannot format an expression with channel type")));
}

TEST(TypecheckTest, BadTraceFmtWithUseOfFunction) {
  constexpr std::string_view kProgram =
      R"(
pub fn some_function() -> u32 { u32:0 }

pub fn other_function() -> u32 {
    trace_fmt!("{}", some_function);
}
)";

  EXPECT_THAT(
      Typecheck(kProgram),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("Cannot format an expression with function type")));
}

TEST(TypecheckTest, CatchesBadInvocationCallee) {
  constexpr std::string_view kImported = R"(
pub fn some_function() -> u32 { u32:0 }
)";
  constexpr std::string_view kProgram = R"(
import imported;

fn main() -> u32 {
  imported.some_function()
})";
  auto import_data = CreateImportDataForTest();
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule module,
      ParseAndTypecheck(kImported, "imported.x", "imported", &import_data));
  EXPECT_THAT(
      ParseAndTypecheck(kProgram, "fake_main_path.x", "main", &import_data),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("An invocation callee must be a function, with a "
                         "possible scope")));
}

// See https://github.com/google/xls/issues/1540#issuecomment-2297711953
TEST(TypecheckTest, ProcWithImportedEnumParametricGithubIssue1540) {
  constexpr std::string_view kImported = R"(
pub enum MyEnum : bits[1] {
  kA = 0,
  kB = 1,
}
)";
  constexpr std::string_view kProgram = R"(
import imported;

proc foo_proc<N: imported::MyEnum> {
    config() { () }
    init { () }
    next(state: ()) { () }
}

proc bar_proc {
    config() {
      spawn foo_proc<imported::MyEnum::kA>();
    }
    init { () }
    next(state: ()) { () }
})";
  auto import_data = CreateImportDataForTest();
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule imported,
      ParseAndTypecheck(kImported, "imported.x", "imported", &import_data));
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule main,
      ParseAndTypecheck(kProgram, "fake_main_path.x", "main", &import_data));
}

// See https://github.com/google/xls/issues/1540#issuecomment-2291819096
TEST(TypecheckTest, ImportedTypeAliasAttributeGithubIssue1540) {
  constexpr std::string_view kImported = R"(
pub const W = s32:16;
pub type T = bits[W as u32];
)";
  constexpr std::string_view kProgram = R"(
import imported;

fn f() -> imported::T {
  imported::T::MAX
}
)";
  auto import_data = CreateImportDataForTest();
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule imported,
      ParseAndTypecheck(kImported, "imported.x", "imported", &import_data));
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule main,
      ParseAndTypecheck(kProgram, "fake_main_path.x", "main", &import_data));
}

TEST(TypecheckTest, MissingWideningCastFromValueError) {
  constexpr std::string_view kProgram = R"(
fn main(x: u32) -> u64 {
  widening_cast<u64>()
}
)";

  EXPECT_THAT(Typecheck(kProgram),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Invalid number of arguments passed to")));
}

TEST(TypecheckTest, MissingCheckedCastFromValueError) {
  constexpr std::string_view kProgram = R"(
fn main(x: u32) -> u64 {
  checked_cast<u64>()
}
)";

  EXPECT_THAT(Typecheck(kProgram),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Invalid number of arguments passed to")));
}

TEST(TypecheckTest, MissingWideningCastToTypeError) {
  constexpr std::string_view kProgram = R"(
fn main(x: u32) -> u64 {
  widening_cast(x)
}
)";

  EXPECT_THAT(Typecheck(kProgram),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Invalid number of parametrics passed to")));
}

TEST(TypecheckTest, MissingCheckedCastToTypeError) {
  constexpr std::string_view kProgram = R"(
fn main(x: u32) -> u64 {
  checked_cast(x)
}
)";

  EXPECT_THAT(Typecheck(kProgram),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Invalid number of parametrics passed to")));
}

TEST(TypecheckTest, WideningCastToSmallerUnError) {
  constexpr std::string_view kProgram = R"(
fn main() {
  widening_cast<u33>(u32:0);
  widening_cast<u32>(u32:0);
  widening_cast<u31>(u32:0);
}
)";

  EXPECT_THAT(Typecheck(kProgram),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Can not cast from type uN[32] (32 bits) to "
                                 "uN[31] (31 bits) with widening_cast")));
}

TEST(TypecheckTest, WideningCastToSmallerSnError) {
  constexpr std::string_view kProgram = R"(
fn main() {
  widening_cast<s33>(s32:0);
  widening_cast<s32>(s32:0);
  widening_cast<s31>(s32:0);
}
)";

  EXPECT_THAT(Typecheck(kProgram),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Can not cast from type sN[32] (32 bits) to "
                                 "sN[31] (31 bits) with widening_cast")));
}

TEST(TypecheckTest, WideningCastToUnError) {
  constexpr std::string_view kProgram = R"(
fn main() {
  widening_cast<u4>(u3:0);
  widening_cast<u4>(u4:0);
  widening_cast<u4>(s1:0);
}
)";

  EXPECT_THAT(Typecheck(kProgram),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Can not cast from type sN[1] (1 bits) to "
                                 "uN[4] (4 bits) with widening_cast")));
}

TEST(TypecheckTest, WideningCastsUnError2) {
  constexpr std::string_view kProgram =
      R"(
fn main(x: u8) -> u32 {
  let x_32 = widening_cast<u32>(x);
  let x_4  = widening_cast<u4>(x_32);
  x_32 + widening_cast<u32>(x_4)
}
)";
  EXPECT_THAT(Typecheck(kProgram),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Can not cast from type uN[32] (32 bits) to "
                                 "uN[4] (4 bits) with widening_cast")));
}

TEST(TypecheckTest, WideningCastToSnError1) {
  constexpr std::string_view kProgram = R"(
fn main() {
  widening_cast<s4>(u3:0);
  widening_cast<s4>(s4:0);
  widening_cast<s4>(u4:0);
}
)";

  EXPECT_THAT(Typecheck(kProgram),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Can not cast from type uN[4] (4 bits) to "
                                 "sN[4] (4 bits) with widening_cast")));
}

TEST(TypecheckTest, WideningCastsSnError2) {
  constexpr std::string_view kProgram =
      R"(
fn main(x: s8) -> s32 {
  let x_32 = widening_cast<s32>(x);
  let x_4  = widening_cast<s4>(x_32);
  x_32 + widening_cast<s32>(x_4)
}
)";
  EXPECT_THAT(Typecheck(kProgram),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Can not cast from type sN[32] (32 bits) to "
                                 "sN[4] (4 bits) with widening_cast")));
}

TEST(TypecheckTest, WideningCastsUnToSnError) {
  constexpr std::string_view kProgram =
      R"(
fn main(x: u8) -> s32 {
  let x_9 = widening_cast<s9>(x);
  let x_8 = widening_cast<s8>(x);
  checked_cast<s32>(x_9) + checked_cast<s32>(x_8)
}
)";
  EXPECT_THAT(Typecheck(kProgram),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Can not cast from type uN[8] (8 bits) to "
                                 "sN[8] (8 bits) with widening_cast")));
}

TEST(TypecheckTest, WideningCastsSnToUnError) {
  constexpr std::string_view kProgram =
      R"(
fn main(x: s8) -> s32 {
  let x_9 = widening_cast<u9>(x);
  checked_cast<s32>(x_9)
}
)";
  EXPECT_THAT(Typecheck(kProgram),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Can not cast from type sN[8] (8 bits) to "
                                 "uN[9] (9 bits) with widening_cast")));
}

TEST(TypecheckTest, OverlargeValue80Bits) {
  constexpr std::string_view kProgram =
      R"(
fn f() {
  let x:sN[0] = sN[80]:0x800000000000000000000;
}
)";
  EXPECT_THAT(Typecheck(kProgram),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Value '0x800000000000000000000' does not fit "
                                 "in the bitwidth of a sN[80] (80)")));
}

TEST(TypecheckTest, NegateTuple) {
  constexpr std::string_view kProgram =
      R"(
fn f() -> (u32, u32) {
  -(u32:42, u32:64)
}
)";
  EXPECT_THAT(Typecheck(kProgram),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Unary operation `-` can only be applied to "
                                 "bits-typed operands")));
}

TEST_P(TypecheckBothVersionsTest, MatchOnBitsWithEmptyTuplePattern) {
  constexpr std::string_view kProgram =
      R"(
fn f(x: u32) -> u32 {
  match x {
    () => x,
  }
}
)";
  EXPECT_THAT(
      Typecheck(kProgram),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          AllOf(HasSubstrInV1(
                    GetParam(),
                    "uN[32] Pattern expected matched-on type to be a tuple"),
                HasTypeMismatchInV2(GetParam(), "()", "u32"))));
}

TEST_P(TypecheckBothVersionsTest, MatchOnBitsWithIrrefutableTuplePattern) {
  constexpr std::string_view kProgram =
      R"(
fn f(x: u32) -> u32 {
  match x {
    (y) => y,
  }
}
)";
  EXPECT_THAT(
      Typecheck(kProgram),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          AllOf(HasSubstrInV1(
                    GetParam(),
                    "uN[32] Pattern expected matched-on type to be a tuple."),
                HasTypeMismatchInV2(GetParam(), "(Any,)", "u32"))));
}

TEST_P(TypecheckBothVersionsTest, MatchOnTupleWithWrongSizedTuplePattern) {
  constexpr std::string_view kProgram =
      R"(
fn f(x: (u32)) -> u32 {
  match x {
    (y, z) => y,
  }
}
)";
  EXPECT_THAT(
      Typecheck(kProgram),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          AllOf(HasSubstrInV1(GetParam(),
                              "Cannot match a 1-element tuple to 2 values."),
                HasTypeMismatchInV2(GetParam(), "(Any, Any)", "(u32,)"))));
}

TEST_P(TypecheckBothVersionsTest, MatchOnTupleWithRestOfTupleSkipsEnd) {
  constexpr std::string_view kProgram =
      R"(
fn f(x: (u32, u33, u34)) -> u32 {
  match x {
    (y, ..) => y,
  }
}
)";
  XLS_ASSERT_OK(Typecheck(kProgram));
}

TEST_P(TypecheckBothVersionsTest, MatchOnTupleWithRestOfTupleSkipsBeginning) {
  constexpr std::string_view kProgram =
      R"(
fn f(x: (u30, u31, u32)) -> u32 {
  match x {
    (.., y) => y,
  }
}
)";
  XLS_ASSERT_OK(Typecheck(kProgram));
}

TEST_P(TypecheckBothVersionsTest,
       MatchOnTupleWithRestOfTupleSkipsBeginningThenMatches) {
  constexpr std::string_view kProgram =
      R"(
fn f(x: (u29, u30, u31, u32)) -> u31 {
  match x {
    (.., y, z) => y,
  }
}
)";
  XLS_ASSERT_OK(Typecheck(kProgram));
}

TEST_P(TypecheckBothVersionsTest, MatchOnTupleWithRestOfTupleSkipsMiddle) {
  constexpr std::string_view kProgram =
      R"(
fn f(x: (u30, u31, u32, u33)) -> u30 {
  match x {
    (y, .., z) => y,
  }
}
)";
  XLS_ASSERT_OK(Typecheck(kProgram));
}

TEST_P(TypecheckBothVersionsTest, MatchOnTupleWithRestOfTupleSkipsNone) {
  constexpr std::string_view kProgram =
      R"(
fn f(x: (u32, u33)) -> u32 {
  match x {
    (y, .., z) => y,
  }
}
)";
  XLS_ASSERT_OK(Typecheck(kProgram));
}

TEST(TypecheckTest, UnusedBindingInBodyGivesWarning) {
  const constexpr std::string_view kProgram = R"(
fn f(x: u32) -> u32 {
    let y = x + u32:42;
    x
}
)";
  XLS_ASSERT_OK_AND_ASSIGN(TypecheckResult tr, Typecheck(kProgram));
  TypecheckedModule& tm = tr.tm;
  ASSERT_THAT(tm.warnings.warnings().size(), 1);
  EXPECT_EQ(tm.warnings.warnings().at(0).message,
            "Definition of `y` (type `uN[32]`) is not used in function `f`");
}

TEST(TypecheckTest, FiveUnusedBindingsInLetBindingPattern) {
  const constexpr std::string_view kProgram = R"(
fn f(t: (u32, u32, u32, u32, u32)) -> u32 {
    let (a, b, c, d, e) = t;
    t.0
}
)";
  XLS_ASSERT_OK_AND_ASSIGN(TypecheckResult tr, Typecheck(kProgram));
  TypecheckedModule& tm = tr.tm;
  ASSERT_THAT(tm.warnings.warnings().size(), 5);
  EXPECT_EQ(tm.warnings.warnings().at(0).message,
            "Definition of `a` (type `uN[32]`) is not used in function `f`");
  EXPECT_EQ(tm.warnings.warnings().at(1).message,
            "Definition of `b` (type `uN[32]`) is not used in function `f`");
  EXPECT_EQ(tm.warnings.warnings().at(2).message,
            "Definition of `c` (type `uN[32]`) is not used in function `f`");
  EXPECT_EQ(tm.warnings.warnings().at(3).message,
            "Definition of `d` (type `uN[32]`) is not used in function `f`");
  EXPECT_EQ(tm.warnings.warnings().at(4).message,
            "Definition of `e` (type `uN[32]`) is not used in function `f`");
}

TEST(TypecheckTest, UnusedMatchBindingInBodyGivesWarning) {
  const constexpr std::string_view kProgram = R"(
fn f(x: u32) -> u32 {
  match x {
    y => x
  }
}
)";
  XLS_ASSERT_OK_AND_ASSIGN(TypecheckResult tr, Typecheck(kProgram));
  TypecheckedModule& tm = tr.tm;
  ASSERT_THAT(tm.warnings.warnings().size(), 1);
  EXPECT_EQ(tm.warnings.warnings().at(0).message,
            "Definition of `y` (type `uN[32]`) is not used in function `f`");
}

TEST_P(TypecheckBothVersionsTest, ConcatU1U1) {
  XLS_ASSERT_OK(Typecheck("fn f(x: u1, y: u1) -> u2 { x ++ y }"));
}

TEST_P(TypecheckBothVersionsTest, ConcatU1S1) {
  EXPECT_THAT(Typecheck("fn f(x: u1, y: s1) -> u2 { x ++ y }").status(),
              IsPosError("TypeInferenceError",
                         HasSubstr("Concatenation requires operand "
                                   "types to both be unsigned bits")));
}

TEST_P(TypecheckBothVersionsTest, ConcatS1S1) {
  EXPECT_THAT(Typecheck("fn f(x: s1, y: s1) -> u2 { x ++ y }").status(),
              IsPosError("TypeInferenceError",
                         HasSubstr("Concatenation requires operand "
                                   "types to both be unsigned bits")));
}

TEST_P(TypecheckBothVersionsTest, ConcatU2S1) {
  EXPECT_THAT(Typecheck("fn f(x: u2, y: s1) -> u3 { x ++ y }").status(),
              IsPosError("TypeInferenceError",
                         HasSubstr("Concatenation requires operand "
                                   "types to both be unsigned bits")));
}

TEST_P(TypecheckBothVersionsTest, ConcatU1Nil) {
  EXPECT_THAT(Typecheck("fn f(x: u1, y: ()) -> () { x ++ y }").status(),
              IsPosError("TypeInferenceError",
                         HasSubstr("Concatenation requires operand types "
                                   "to be either both-arrays or both-bits")));
}

TEST_P(TypecheckBothVersionsTest, ConcatS1Nil) {
  EXPECT_THAT(
      Typecheck("fn f(x: s1, y: ()) -> () { x ++ y }").status(),
      IsPosError("TypeInferenceError",
                 AllOf(HasSubstrInV1(GetParam(),
                                     "Concatenation requires operand types "
                                     "to be either both-arrays or both-bits"),
                       HasSubstrInV2(GetParam(),
                                     "Concatenation requires operand types to "
                                     "both be unsigned bits"))));
}

TEST_P(TypecheckBothVersionsTest, ConcatNilNil) {
  EXPECT_THAT(Typecheck("fn f(x: (), y: ()) -> () { x ++ y }").status(),
              IsPosError("TypeInferenceError",
                         HasSubstr("Concatenation requires operand types to "
                                   "be either both-arrays or both-bits")));
}

TEST(TypecheckTest, ConcatEnumU2) {
  EXPECT_THAT(Typecheck(R"(
enum MyEnum : u2 {
  A = 1,
  B = 2,
}
fn f(x: MyEnum, y: u2) -> () { x ++ y }
)")
                  .status(),
              IsPosError("TypeInferenceError",
                         HasSubstr("Enum values must be cast to unsigned bits "
                                   "before concatenation")));
}

TEST(TypecheckTest, ConcatU2Enum) {
  EXPECT_THAT(Typecheck(R"(
enum MyEnum : u2 {
  A = 1,
  B = 2,
}
fn f(x: u2, y: MyEnum) -> () { x ++ y }
)")
                  .status(),
              IsPosError("TypeInferenceError",
                         HasSubstr("Enum values must be cast to unsigned bits "
                                   "before concatenation")));
}

TEST(TypecheckTest, ConcatEnumEnum) {
  EXPECT_THAT(Typecheck(R"(
enum MyEnum : u2 {
  A = 1,
  B = 2,
}
fn f(x: MyEnum, y: MyEnum) -> () { x ++ y }
)")
                  .status(),
              IsPosError("TypeInferenceError",
                         HasSubstr("Enum values must be cast "
                                   "to unsigned bits before concatenation")));
}

TEST_P(TypecheckBothVersionsTest, ConcatStructStruct) {
  EXPECT_THAT(Typecheck(R"(
struct S {}
fn f(x: S, y: S) -> () { x ++ y }
)")
                  .status(),
              IsPosError("TypeInferenceError",
                         HasSubstr("Concatenation requires operand types to be "
                                   "either both-arrays or both-bits")));
}

TEST_P(TypecheckBothVersionsTest, ConcatUnWithXn) {
  XLS_ASSERT_OK(Typecheck(R"(
fn f(x: u32, y: xN[false][32]) -> xN[false][64] { x ++ y }
)"));
}

TEST_P(TypecheckBothVersionsTest, ConcatU1ArrayOfOneU8) {
  EXPECT_THAT(
      Typecheck("fn f(x: u1, y: u8[1]) -> u2 { x ++ y }").status(),
      IsPosError(
          "TypeInferenceError",
          HasSubstr(
              "Attempting to concatenate array/non-array values together")));
}

TEST_P(TypecheckBothVersionsTest, ConcatArrayOfThreeU8ArrayOfOneU8) {
  XLS_ASSERT_OK(Typecheck("fn f(x: u8[3], y: u8[1]) -> u8[4] { x ++ y }"));
}

TEST(TypecheckTest, ParametricWrapperAroundBuiltin) {
  XLS_ASSERT_OK(Typecheck(R"(fn f<N: u32>(x: uN[N]) -> uN[N] { rev(x) }

fn main(arg: u32) -> u32 {
  f(arg)
})"));
}

TEST(TypecheckTest, AssertBuiltinIsUnitType) {
  XLS_ASSERT_OK(Typecheck(R"(fn main() {
  assert!(true, "oh_no");
})"));

  XLS_ASSERT_OK(Typecheck(R"(fn main() {
  assert!(true, "oh_no");
})"));

  XLS_ASSERT_OK(Typecheck(R"(fn main() {
  let () = assert!(true, "oh_no");
})"));
}

TEST(TypecheckTest, ConcatNilArrayOfOneU8) {
  EXPECT_THAT(
      Typecheck("fn f(x: (), y: u8[1]) -> () { x ++ y }").status(),
      IsPosError(
          "TypeInferenceError",
          HasSubstr(
              "Attempting to concatenate array/non-array values together")));
}

TEST(TypecheckTest, ParametricStructWithoutAllParametricsBoundInReturnType) {
  EXPECT_THAT(
      Typecheck(R"(
struct Point1D<N: u32> { x: bits[N] }

fn f(x: Point1D) -> Point1D { x }
)")
          .status(),
      IsPosError("TypeInferenceError",
                 HasSubstr("Parametric type being returned from function")));
}

// See https://github.com/google/xls/issues/1030
TEST(TypecheckTest, InstantiateImportedParametricStruct) {
  constexpr std::string_view kImported = R"(
pub struct my_struct<N: u32> {
    my_field: uN[N],
}
)";
  constexpr std::string_view kProgram = R"(
import imported;

fn main() -> u5 {
  const local_struct = imported::my_struct<5> { my_field: u5:10 };
  local_struct.my_field
}
)";
  auto import_data = CreateImportDataForTest();
  XLS_EXPECT_OK(
      ParseAndTypecheck(kImported, "imported.x", "imported", &import_data));
  XLS_EXPECT_OK(
      ParseAndTypecheck(kProgram, "fake_main_path.x", "main", &import_data));
}

TEST(TypecheckTest, InstantiateImportedParametricStructNoParametrics) {
  constexpr std::string_view kImported = R"(
pub struct my_struct<N: u32> {
    my_field: uN[N],
}
)";
  constexpr std::string_view kProgram = R"(
import imported;

fn main() -> u5 {
  const local_struct = imported::my_struct { my_field: u5:10 };
  local_struct.my_field
}
)";
  auto import_data = CreateImportDataForTest();
  XLS_EXPECT_OK(
      ParseAndTypecheck(kImported, "imported.x", "imported", &import_data));
  XLS_EXPECT_OK(
      ParseAndTypecheck(kProgram, "fake_main_path.x", "main", &import_data));
}

TEST(TypecheckTest, InstantiateImportedParametricStructTypeAlias) {
  constexpr std::string_view kImported = R"(
pub struct my_struct<N: u32> {
    my_field: uN[N],
}
)";
  constexpr std::string_view kProgram = R"(
import imported;

type MyTypeAlias = imported::my_struct<5>;

fn extract_field(x: MyTypeAlias) -> u5 {
  x.my_field
}

fn main() -> u5 {
  const local_struct = imported::my_struct<5> { my_field: u5:10 };
  extract_field(local_struct)
}
)";
  auto import_data = CreateImportDataForTest();
  XLS_EXPECT_OK(
      ParseAndTypecheck(kImported, "imported.x", "imported", &import_data));
  XLS_EXPECT_OK(
      ParseAndTypecheck(kProgram, "fake_main_path.x", "main", &import_data));
}

TEST(TypecheckTest, InstantiateImportedParametricStructArray) {
  constexpr std::string_view kImported = R"(
pub struct my_struct<N: u32> {
    my_field: uN[N],
}
)";
  constexpr std::string_view kProgram = R"(
import imported;

const imported_structs = imported::my_struct<5>[2]:[
  imported::my_struct { my_field: u5:10 },
  imported::my_struct { my_field: u5:11 }
];

fn main() -> u5 {
  imported_structs[1].my_field
}
)";
  auto import_data = CreateImportDataForTest();
  XLS_EXPECT_OK(
      ParseAndTypecheck(kImported, "imported.x", "imported", &import_data));
  XLS_EXPECT_OK(
      ParseAndTypecheck(kProgram, "fake_main_path.x", "main", &import_data));
}

TEST(TypecheckTest, InstantiateParametricStructArray) {
  constexpr std::string_view kProgram = R"(
struct my_struct<N: u32> {
    my_field: uN[N],
}

const local_structs = my_struct<5>[1]:[
  //my_struct { my_field: u5:10 },
  my_struct { my_field: u5:11 }
];

fn main() -> u5 {
  local_structs[0].my_field
}
)";
  auto import_data = CreateImportDataForTest();
  XLS_EXPECT_OK(
      ParseAndTypecheck(kProgram, "fake_main_path.x", "main", &import_data));
}

TEST(TypecheckTest, CallImportedParametricFn) {
  constexpr std::string_view kImported = R"(
pub fn my_fn<N: u32>(x: uN[N]) -> uN[N] {
   x+uN[N]:1
}
)";
  constexpr std::string_view kProgram = R"(
import imported;

fn main() -> u5 {
  let x = imported::my_fn<u32:5>(u5:10);
  x
}
)";
  auto import_data = CreateImportDataForTest();
  XLS_EXPECT_OK(
      ParseAndTypecheck(kImported, "imported.x", "imported", &import_data));
  XLS_EXPECT_OK(
      ParseAndTypecheck(kProgram, "fake_main_path.x", "main", &import_data));
}

TEST(TypecheckErrorTest, PrioritySelectOnNonBitsType) {
  EXPECT_THAT(
      Typecheck(R"(
struct MyStruct { }

fn f() {
let default_value = MyStruct{};
priority_sel(u2:0b00, MyStruct[2]:[MyStruct{}, MyStruct{}], default_value) }
)")
          .status(),
      IsPosError(
          "TypeInferenceError",
          HasSubstr("Want argument 1 element type to be bits; got MyStruct")));
}

TEST(TypecheckErrorTest, PrioritySelectDefaultWrongSize) {
  EXPECT_THAT(Typecheck(R"(
fn f() { priority_sel(u3:0b000, u4[3]:[u4:1, u4:2, u4:4], u5:0); }
)")
                  .status(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Want argument 2 type uN[4] to match "
                                 "argument 1 element type")));
}

TEST(TypecheckErrorTest, OperatorOnParametricBuiltin) {
  EXPECT_THAT(Typecheck(R"(
fn f() { fail!%2 }
)")
                  .status(),
              IsPosError("TypeInferenceError",
                         HasSubstr("Name 'fail!' is a parametric function, but "
                                   "it is not being invoked")));
}

TEST(TypecheckErrorTest, InvokeATokenValueViaShadowing) {
  EXPECT_THAT(
      Typecheck(R"(
fn shadowed() {}

fn f(shadowed: token) {
  shadowed()
}
)")
          .status(),
      IsPosError("TypeInferenceError",
                 HasSubstr("Invocation callee `shadowed` is not a function")));
}

TEST(TypecheckErrorTest, InvokeATokenValueNoShadowing) {
  EXPECT_THAT(
      Typecheck(R"(
fn f(tok: token) {
  tok()
}
)")
          .status(),
      IsPosError("TypeInferenceError",
                 HasSubstr("Invocation callee `tok` is not a function")));
}

TEST(TypecheckErrorTest, MapOfNonFunctionInTestProc) {
  EXPECT_THAT(
      Typecheck(R"(
#[test_proc]
proc t {
    result_in: chan<u32> in;

    config(terminator: chan<bool> out) {
        let (result_out, result_in) = chan<u32>("result");
        (result_in,)
    }

    init {  }

    next(state: ()) {
        let (ok, result) = map(join(), result_in);
    }
}
)")
          .status(),
      IsPosError(
          "TypeInferenceError",
          HasSubstr("Cannot resolve callee `result_in` to a function; No "
                    "function in module `fake` with name `result_in`")));
}

TEST(TypecheckErrorTest, ReferenceToBuiltinFunctionInNext) {
  XLS_EXPECT_OK(Typecheck(R"(
proc t {
    config() { () }

    init { token() }

    next(state: token) { state }
}
)"));
}

TEST(TypecheckErrorTest, InstantiateALiteralNumber) {
  constexpr std::string_view kProgram = "fn f(x:u2) { 0<> }";
  EXPECT_THAT(Typecheck(kProgram).status(),
              IsPosError("TypeInferenceError",
                         HasSubstr("Could not infer a type for this number, "
                                   "please annotate a type.")));
}

TEST(TypecheckErrorTest, SignedValueToBuiltinExpectingUNViaParametric) {
  EXPECT_THAT(Typecheck(R"(
fn p<S: bool, N: u32>() -> u32 {
  clz(xN[S][N]:0xdeadbeef) as u32
}

fn main() -> u32 {
  p<true, u32:32>()
}
)")
                  .status(),
              IsPosError("TypeInferenceError",
                         HasSubstr("Want argument 0 to be unsigned; got "
                                   "xN[is_signed=1][32] (type is signed)")));
}

// Passes a signed value to a builtin function that expects a `uN[N]`.
TEST(TypecheckErrorTest, SignedValueToBuiltinExpectingUN) {
  EXPECT_THAT(Typecheck(R"(
fn main() {
  clz(s32:0xdeadbeef)
}
)")
                  .status(),
              IsPosError("TypeInferenceError",
                         HasSubstr("Want argument 0 to be unsigned; got "
                                   "sN[32] (type is signed)")));
}

TEST(TypecheckErrorTest, DuplicateParametricBinding) {
  EXPECT_THAT(
      Typecheck("fn p<N: u1, N: u2>() -> u32 { u32:42 }").status(),
      IsPosError("ParseError", HasSubstr("Duplicate parametric binding: `N`")));
}

TEST(TypecheckTest, MapOfXbitsArray) {
  constexpr std::string_view kProgram = R"(
type MyXN = xN[bool:0x0][1];  // effectively a bool

fn f(x15: MyXN) {}

fn main(u: u4) {
    let a: MyXN[4] = u as MyXN[4];
    map(a, f);
}
)";
  XLS_EXPECT_OK(Typecheck(kProgram));
}

// This test case was creating an error because it caused us to try to
// `deduce_ctx->Resolve()` when there were no entries on the function stack.
TEST(TypecheckTest, ResolveInTopLevelContext) {
  constexpr std::string_view kProgram = R"(
type x51 = u40;
fn x42<x46: xN[bool:0x0][38] = {xN[bool:0x0][38]:0x0}>(x43: u40) {}
fn x36<x41: u40 = {u40:0xff_ffff_ffff}>() { x42(x41); }
fn main() { x36() }
)";
  XLS_EXPECT_OK(Typecheck(kProgram));
}

// Tests that it is OK to ignore the return value of a function.
TEST(TypecheckTest, IgnoreReturnValue) {
  constexpr std::string_view kProgram = R"(
fn foo() -> u32 { u32:0 }

fn main() -> u32 {
  foo();
  u32:1
}
)";
  XLS_EXPECT_OK(Typecheck(kProgram));
}

TEST(TypecheckTest, UnassignedReturnValueTypeMismatchParametric) {
  constexpr std::string_view kProgram = R"(
fn ignored<N:u32>() -> uN[N] { zero!<uN[N]>() }

fn main(x: u32) -> u32 {
  ignored<u32:31>() + x;
  u32:1
}
)";
  EXPECT_THAT(
      Typecheck(kProgram).status(),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          HasSubstr(
              "uN[31] vs uN[32]: Could not deduce type for binary operation")));
}

// Previously this would cause us to RET_CHECK because we were assuming we
// wanted to grab the root type information instead of the parametric
// invocation's type information.
TEST_P(TypecheckBothVersionsTest, AttrViaParametricBinding) {
  constexpr std::string_view kProgram = R"(
fn f<N: u32>() -> uN[N]{
    type UN = uN[N];
    let max = UN::MAX;
    max
}

const_assert!(f<u32:8>() == u8:255);
)";
  XLS_EXPECT_OK(Typecheck(kProgram));
}

TEST(TypecheckTest, MatchPackageLevelConstant) {
  constexpr std::string_view kProgram = R"(
const FOO = u8:0xff;
fn f(x: u8) -> u2 {
  match x {
    FOO => u2:0,
    _ => u2:0,
  }
}
)";
  XLS_ASSERT_OK_AND_ASSIGN(TypecheckResult result, Typecheck(kProgram));
  const TypeInfo& type_info = *result.tm.type_info;
  // Get the pattern match for the first arm of the match expression inside of
  // function `f`.
  Function* f = result.tm.module->GetFunction("f").value();
  Statement* stmt = f->body()->statements()[0];
  Expr* expr = std::get<Expr*>(stmt->wrapped());
  Match* m = dynamic_cast<Match*>(expr);
  ASSERT_NE(m, nullptr);
  const MatchArm* arm = m->arms()[0];
  const NameDefTree* pattern = arm->patterns()[0];

  // Check that the pattern is just a leaf NameRef.
  NameRef* name_ref = std::get<NameRef*>(pattern->leaf());
  ASSERT_NE(name_ref, nullptr);
  EXPECT_EQ(name_ref->identifier(), "FOO");

  std::optional<InterpValue> const_expr =
      type_info.GetConstExprOption(name_ref);
  ASSERT_TRUE(const_expr.has_value());
  EXPECT_EQ(const_expr->ToString(), "u8:255");
}

TEST(TypecheckTest, MatchPackageLevelConstantIntoTypeAlias) {
  constexpr std::string_view kProgram = R"(
const C = u32:2;
fn f(x: u32) -> u2 {
  match x {
    C => {
      const D = C;
      type T = uN[D];
      T::MAX
    },
    _ => u2:0,
  }
}
)";
  XLS_EXPECT_OK(Typecheck(kProgram));
}

TEST(TypecheckTest, BitCount) {
  XLS_ASSERT_OK(Typecheck(R"(
struct S {
  a: u32
}

struct T<N: u32> {
  a: uN[N]
}

type A = S;
type B = T;

fn main() -> u32 {
  bit_count<u32>() +
  bit_count<s64>() +
  bit_count<u32[u32:4]>() +
  bit_count<bool>() +
  bit_count<S>() +
  bit_count<T<u32:4>>() +
  bit_count<(u32, bool)>() +
  bit_count<A>() +
  bit_count<B<u32:5>>()
})"));
}

TEST(TypecheckTest, ElementCount) {
  XLS_ASSERT_OK(Typecheck(R"(
struct S {
  a: u32,
  b: u32
}

struct T<N: u32> {
  a: uN[N],
  b: u32
}

type A = S;
type B = T;

fn main() -> u32 {
  element_count<u32>() +
  element_count<s64>() +
  element_count<u32[u32:4]>() +
  element_count<u32[u32:4][u32:5]>() +
  element_count<bool>() +
  element_count<S>() +
  element_count<T<u32:4>>() +
  element_count<(u32, bool)>() +
  element_count<A>() +
  element_count<B<u32:5>>()
})"));
}

TEST(TypecheckTest, BitCountAsConstExpr) {
  XLS_ASSERT_OK(Typecheck(R"(
fn main() -> u64 {
  uN[bit_count<u32[2]>()]:0
})"));
}

TEST(TypecheckTest, BitCountWithNoTypeArgument) {
  EXPECT_THAT(Typecheck(R"(
fn main() -> u32 {
  bit_count<>()
})"),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Invalid number of parametrics passed to "
                                 "'bit_count', expected 1, got 0")));
}

TEST(TypecheckTest, BitCountWithMultipleTypeArguments) {
  EXPECT_THAT(Typecheck(R"(
fn main() -> u32 {
  bit_count<u32, u16>()
})"),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Invalid number of parametrics passed to "
                                 "'bit_count', expected 1, got 2")));
}

TEST(TypecheckTest, BitCountWithExprArgument) {
  EXPECT_THAT(Typecheck(R"(
fn main() -> u32 {
  bit_count<u32:0>()
})"),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("The parametric argument for 'bit_count' "
                                 "should be a type and not a value.")));
}

// Note: we use two different type signatures here for the two functions being
// binop'd -- this can help show any place we assume we can "diff" the two
// function types when they're not type-compatible.
TEST(TypecheckTest, BinaryOpsOnFunctionType) {
  const std::string kProgramTemplate = R"(fn f() -> u32 { u32:42 }
fn g() -> u8 { u8:43 }

fn main() {
  f {op} g
}
)";
  for (BinopKind binop : kAllBinopKinds) {
    std::string program = absl::StrReplaceAll(
        kProgramTemplate, {{"{op}", BinopKindFormat(binop)}});
    EXPECT_THAT(Typecheck(program).status(),
                StatusIs(absl::StatusCode::kInvalidArgument,
                         AnyOf(HasSubstr("can only be applied to"),
                               HasSubstr("requires operand types to be"),
                               HasSubstr("Cannot use operator"))));
  }
}

TEST(TypecheckTest, UnaryOpsOnFunctionType) {
  const std::string kProgramTemplate = R"(fn f() -> u32 { u32:42 }
fn main() -> u32 {
  {op} f
}
)";
  for (UnopKind unop : kAllUnopKinds) {
    std::string program =
        absl::StrReplaceAll(kProgramTemplate, {{"{op}", UnopKindFormat(unop)}});
    EXPECT_THAT(Typecheck(program).status(),
                StatusIs(absl::StatusCode::kInvalidArgument,
                         AnyOf(HasSubstr("can only be applied to"),
                               HasSubstr("requires operand types to be"),
                               HasSubstr("Cannot use operator"))));
  }
}

// Table-oriented test that lets us validate that *types on parameters* are
// compatible with *particular values* that should be type-compatible.
TEST(PassValueToIdentityFnTest, ParameterVsValue) {
  constexpr std::string_view kTemplate =
      R"(fn id(x: $PARAM_TYPE) -> $PARAM_TYPE { x }

fn main() -> $VALUE_TYPE { id($VALUE_TYPE:$VALUE) }
)";
  constexpr std::string_view kBuiltinTemplate = R"(
fn builtins() -> (u1, u1, u1, u32, u32, u32, $VALUE_TYPE) {
  let v: $PARAM_TYPE = $VALUE_TYPE:$VALUE;
  let x = xor_reduce(v);
  let o = or_reduce(v);
  let a = and_reduce(v);
  let lz = clz(v) as u32;
  let tz = ctz(v) as u32;
  let e = encode(v) as u32;
  let r = rev(v);
  (x, o, a, lz, tz, e, r)
}
)";

  struct TestCase {
    std::string param_type;
    std::string value_tye;
    std::string value;
    // Whether we want unsigned-taking (unary) builtins to be run on the
    // value.
    bool want_unsigned_builtins;
  } kTestCases[] = {
      // xN[false][8] should be type compatible with u8
      TestCase{.param_type = "xN[false][8]",
               .value_tye = "u8",
               .value = "42",
               .want_unsigned_builtins = true},
      TestCase{.param_type = "u8",
               .value_tye = "xN[false][8]",
               .value = "42",
               .want_unsigned_builtins = true},

      // xN[true][8] should be type compatible with s8
      TestCase{.param_type = "xN[true][8]",
               .value_tye = "s8",
               .value = "42",
               .want_unsigned_builtins = false},
      TestCase{.param_type = "s8",
               .value_tye = "xN[true][8]",
               .value = "42",
               .want_unsigned_builtins = false},

      // uN[8] should be type compatible with u8
      TestCase{.param_type = "uN[8]",
               .value_tye = "u8",
               .value = "42",
               .want_unsigned_builtins = true},
      TestCase{.param_type = "u8",
               .value_tye = "uN[8]",
               .value = "42",
               .want_unsigned_builtins = true},

      // bits[8] should be type compatible with uN[8]
      TestCase{.param_type = "uN[8]",
               .value_tye = "bits[8]",
               .value = "42",
               .want_unsigned_builtins = true},
      TestCase{.param_type = "bits[8]",
               .value_tye = "uN[8]",
               .value = "42",
               .want_unsigned_builtins = true},
  };

  for (const auto& [param_type, value_type, value, want_unsigned_builtins] :
       kTestCases) {
    const std::string program =
        absl::StrReplaceAll(kTemplate, {
                                           {"$PARAM_TYPE", param_type},
                                           {"$VALUE_TYPE", value_type},
                                           {"$VALUE", value},
                                       });
    XLS_EXPECT_OK(Typecheck(program));

    if (want_unsigned_builtins) {
      std::string unsigned_builtins_program =
          absl::StrReplaceAll(kBuiltinTemplate, {
                                                    {"$PARAM_TYPE", param_type},
                                                    {"$VALUE_TYPE", value_type},
                                                    {"$VALUE", value},
                                                });
      XLS_EXPECT_OK(Typecheck(unsigned_builtins_program));
    }
  }
}

INSTANTIATE_TEST_SUITE_P(TypecheckBothVersionsTestSuite,
                         TypecheckBothVersionsTest,
                         testing::Values(TypeInferenceVersion::kVersion1,
                                         TypeInferenceVersion::kVersion2));

}  // namespace
}  // namespace xls::dslx
