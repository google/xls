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

#include <filesystem>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/container/flat_hash_map.h"
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
#include "xls/dslx/interp_value.h"
#include "xls/dslx/parse_and_typecheck.h"
#include "xls/dslx/type_system/type_info.h"
#include "xls/dslx/type_system/typecheck_test_utils.h"
#include "xls/dslx/virtualizable_file_system.h"
#include "xls/dslx/warning_kind.h"

namespace xls::dslx {
namespace {

using ::absl_testing::StatusIs;
using ::testing::AllOf;
using ::testing::HasSubstr;

// Base class for general tests that run against both versions of type
// inference.
class TypecheckV2Test : public ::testing::Test {
 public:
  absl::StatusOr<TypecheckResult> Typecheck(
      std::string_view program, std::string_view module_name = "fake",
      ImportData* import_data = nullptr) {
    if (import_data == nullptr) {
      import_data_ = CreateImportDataPtrForTest();
      import_data = import_data_.get();
    }
    return ::xls::dslx::TypecheckV2(program, module_name, import_data);
  }

  TypeInferenceVersion GetParam() { return TypeInferenceVersion::kVersion2; }

 private:
  std::unique_ptr<ImportData> import_data_;
};

TEST_F(TypecheckV2Test, TestFunctionMustHaveUnitSignature) {
  constexpr std::string_view kProgram = R"(
#[test]
fn o() -> u32 { u32:2 }
)";
  EXPECT_THAT(
      Typecheck(kProgram),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("Test functions must have function type `() -> ()`")));
}

TEST_F(TypecheckV2Test, EnumItemSelfReference) {
  std::string_view text = "enum E:u2{ITEM=E::ITEM}";
  EXPECT_THAT(Typecheck(text),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Cannot find a definition for name: \"E\"")));
}

TEST_F(TypecheckV2Test, TypeAliasSelfReference) {
  std::string_view text = "type T=uN[T::A as u2];";
  EXPECT_THAT(Typecheck(text),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Cannot find a definition for name: \"T\"")));
}

TEST_F(TypecheckV2Test, RecvInFunction) {
  std::string_view text = R"(
fn f(tok: token, output_r: chan<u8> in, expected: u8) -> token {
  let (tok, value) = recv(tok, output_r);
  tok
}
)";
  EXPECT_THAT(
      Typecheck(text),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          AllOf(HasSubstrInV1(GetParam(), "Cannot recv() outside of a proc"),
                HasSubstrInV2(GetParam(),
                              "Cannot call `recv` outside a `proc`"))));
}

TEST_F(TypecheckV2Test, ParametricWhoseTypeIsDeterminedByArgBinding) {
  std::string_view text = R"(
fn p<A: u32, B: u32, C: bits[B] = {bits[B]:0}>(x: bits[B]) -> bits[B] { x }
fn main() -> u2 { p<u32:1>(u2:0) }
)";
  XLS_EXPECT_OK(Typecheck(text));
}

TEST_F(TypecheckV2Test, IndexZeroSizedArray) {
  std::string_view text = R"(fn f(a: u8[0], b: u3) -> u8 { a[b] })";
  EXPECT_THAT(Typecheck(text),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Zero-sized arrays cannot be indexed")));
}

TEST_F(TypecheckV2Test, ZeroMacroFunctionRefIsNotValue) {
  // Bare parametric macro reference should have function type `() -> E` and
  // not be directly usable as a value of type `E`.
  std::string_view text = R"(enum E:u2{} fn a()->E{ zero!<E> })";
  constexpr const char* kV1Msg =
      "Return type of function body for 'a' did not match the annotated return "
      "type.\nType mismatch:\n   () -> E\nvs E";
  absl::Status st = Typecheck(text).status();
  EXPECT_THAT(st, HasTypeSystemError(GetParam(),
                                     ::testing::HasSubstr(std::string(kV1Msg)),
                                     HasTypeMismatch("() -> E", "E")));
}

TEST_F(TypecheckV2Test, AllOnesMacroFunctionRefIsNotValue) {
  std::string_view text = R"(enum E:u2{} fn a()->E{ all_ones!<E> })";
  constexpr const char* kV1Msg =
      "Return type of function body for 'a' did not match the annotated return "
      "type.\nType mismatch:\n   () -> E\nvs E";
  absl::Status st = Typecheck(text).status();
  EXPECT_THAT(st, HasTypeSystemError(GetParam(),
                                     ::testing::HasSubstr(std::string(kV1Msg)),
                                     HasTypeMismatch("() -> E", "E")));
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
TEST_F(TypecheckV2Test, ParametricWithDefaultExpressionThatHasWrongType) {
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

TEST_F(TypecheckV2Test, ParametricThatWorksForTheOneBindingPresented) {
  std::string_view text = R"(
fn p<X: u32, Y: bits[X] = {u1:0}>(x: bits[X]) -> bits[X] { x }
fn main() -> u1 { p(u1:0) }
)";
  XLS_EXPECT_OK(Typecheck(text));
}

TEST_F(TypecheckV2Test, ParametricWrongArgCount) {
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

TEST_F(TypecheckV2Test, ParametricTooManyExplicitSupplied) {
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

TEST_F(TypecheckV2Test, ReturnTypeMismatch) {
  EXPECT_THAT(
      Typecheck("fn f(x: bits[3], y: bits[4]) -> bits[5] { y }"),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          AllOf(HasSizeMismatchInV1(GetParam(), "uN[4]", "uN[5]"),
                HasSizeMismatchInV2(GetParam(), "bits[4]", "bits[5]"),
                HasSubstrInV1(GetParam(), "Return type of function body"))));
}

TEST_F(TypecheckV2Test, ReturnTypeMismatchWithImplicitUnitReturn) {
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

TEST_F(TypecheckV2Test, ReturnTypeMismatchWithExplicitUnitReturn) {
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

TEST_F(TypecheckV2Test, Identity) {
  XLS_EXPECT_OK(Typecheck("fn f(x: u32) -> u32 { x }"));
  XLS_EXPECT_OK(Typecheck("fn f(x: bits[3], y: bits[4]) -> bits[3] { x }"));
  XLS_EXPECT_OK(Typecheck("fn f(x: bits[3], y: bits[4]) -> bits[4] { y }"));
}

TEST_F(TypecheckV2Test, TokenIdentity) {
  XLS_EXPECT_OK(Typecheck("fn f(x: token) -> token { x }"));
}

TEST_F(TypecheckV2Test, Unit) {
  XLS_EXPECT_OK(Typecheck("fn f(x: u32) -> () { () }"));
  XLS_EXPECT_OK(Typecheck("fn f(x: u32) { () }"));
}

TEST_F(TypecheckV2Test, Arithmetic) {
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

TEST_F(TypecheckV2Test, Unary) {
  XLS_EXPECT_OK(Typecheck("fn f(x: u32) -> u32 { !x }"));
  XLS_EXPECT_OK(Typecheck("fn f(x: u32) -> u32 { -x }"));
}

TEST_F(TypecheckV2Test, Let) {
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
                             HasSizeMismatchInV2(GetParam(), "u4", "u32"))));
  XLS_EXPECT_OK(Typecheck(
      "fn f() -> u32 { let (x, y): (u32, bits[4]) = (u32:2, bits[4]:3); x }"));
}

TEST_F(TypecheckV2Test, LetBadRhs) {
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

TEST_F(TypecheckV2Test, ParametricInvocation) {
  constexpr std::string_view kProgram = R"(
fn p<N: u32>(x: bits[N]) -> bits[N] { x + bits[N]:1 }
fn f() -> u32 { p(u32:3) }
)";
  XLS_EXPECT_OK(Typecheck(kProgram));
}

TEST_F(TypecheckV2Test, ParametricInvocationWithTuple) {
  constexpr std::string_view kProgram = R"(
fn p<N: u32>(x: bits[N]) -> (bits[N], bits[N]) { (x, x) }
fn f() -> (u32, u32) { p(u32:3) }
)";
  XLS_EXPECT_OK(Typecheck(kProgram));
}

TEST_F(TypecheckV2Test, DoubleParametricInvocation) {
  constexpr std::string_view kProgram = R"(
fn p<N: u32>(x: bits[N]) -> bits[N] { x + bits[N]:1 }
fn o<M: u32>(x: bits[M]) -> bits[M] { p(x) }
fn f() -> u32 { o(u32:3) }
)";
  XLS_EXPECT_OK(Typecheck(kProgram));
}

TEST_F(TypecheckV2Test, XbitsBinding) {
  constexpr std::string_view kProgram = R"(
fn p<S: bool, N: u32>(x: xN[S][N]) -> (bool, u32) { (S, N) }
fn f() -> (bool, u32)[2] { [p(u4:0), p(s8:0)] }
)";
  XLS_EXPECT_OK(Typecheck(kProgram));
}

TEST_F(TypecheckV2Test, XbitsCast) {
  constexpr std::string_view kProgram = R"(
fn f(x: u1) -> s1 { x as xN[true][1] }
fn g() -> s1 { u1:0 as xN[true][1] }
)";
  XLS_EXPECT_OK(Typecheck(kProgram));
}

// Previously this would flag an internal error.
TEST_F(TypecheckV2Test, CastToXbitsBasedBoolArray) {
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
TEST_F(TypecheckV2Test, StructInstantiateParametricXnField) {
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

TEST_F(TypecheckV2Test, ParametricPlusGlobal) {
  constexpr std::string_view kProgram = R"(
const GLOBAL = u32:4;
fn p<N: u32>() -> bits[N+GLOBAL] { bits[N+GLOBAL]:0 }
fn f() -> u32 { p<u32:28>() }
)";
  XLS_EXPECT_OK(Typecheck(kProgram));
}

TEST_F(TypecheckV2Test, TupleWithExplicitlyAnnotatedType) {
  constexpr std::string_view kProgram = R"(
const MY_TUPLE = (u32, u64):(u32:32, u64:64);
)";
  XLS_EXPECT_OK(Typecheck(kProgram));
}

TEST_F(TypecheckV2Test, ProcWithImplEmpty) {
  constexpr std::string_view kProgram = R"(
proc Foo {}

impl Foo {}
)";
  XLS_EXPECT_OK(Typecheck(kProgram));
}

TEST_F(TypecheckV2Test, ProcWithImplAndMembers) {
  constexpr std::string_view kProgram = R"(
proc Foo {
  foo: u32,
  bar: s8[7],
}

impl Foo {}
)";
  XLS_EXPECT_OK(Typecheck(kProgram));
}

TEST_F(TypecheckV2Test, ProcWithImplAndParametrics) {
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

TEST_F(TypecheckV2Test, ProcWithImplEmptyInstantiation) {
  constexpr std::string_view kProgram = R"(
proc Foo {}

impl Foo {}

fn foo() -> Foo {
  Foo{}
}
  )";
  EXPECT_THAT(Typecheck(kProgram),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Impl-style procs are a work in progress and "
                                 "cannot yet be instantiated.")));
}

TEST_F(TypecheckV2Test, ProcWithImplInstantiation) {
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
  EXPECT_THAT(Typecheck(kProgram),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Impl-style procs are a work in progress and "
                                 "cannot yet be instantiated.")));
}

TEST_F(TypecheckV2Test, FailsOnProcWithImplZero) {
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
  EXPECT_THAT(
      Typecheck(kProgram),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          HasSubstr("Expected a type argument in `zero!<Foo>()`; saw `Foo`.")));
}

TEST_F(TypecheckV2Test, FailsOnProcWithImplAsStructMember) {
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

TEST_F(TypecheckV2Test, FailsOnProcWithImplAsStructMemberInArray) {
  constexpr std::string_view kProgram = R"(
proc Foo {
  foo: u32,
  bar: bool,
}

impl Foo {}

struct Bar {
  subprocs: Foo[2]
}

fn foo() -> Bar {
  zero!<Bar>()
}
  )";
  EXPECT_THAT(Typecheck(kProgram),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Cannot make a zero-value of proc type.")));
}

TEST_F(TypecheckV2Test, FailsOnProcWithImplAsStructMemberInTuple) {
  constexpr std::string_view kProgram = R"(
proc Foo {
  foo: u32,
  bar: bool,
}

impl Foo {}

struct Bar {
  subprocs: (Foo, Foo)
}

fn foo() -> Bar {
  zero!<Bar>()
}
  )";
  EXPECT_THAT(Typecheck(kProgram),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Cannot make a zero-value of proc type.")));
}

TEST_F(TypecheckV2Test, ProcWithImplAsProcMember) {
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

TEST_F(TypecheckV2Test, ProcWithImplAsProcMemberInArray) {
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
TEST_F(TypecheckV2Test, ProcWithSliceOfNumber) {
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

TEST_F(TypecheckV2Test, ProcWithImplAsProcMemberInTuple) {
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

TEST_F(TypecheckV2Test, ImportPublicFunction) {
  constexpr std::string_view kImported = R"(
pub fn foo(x: u32) -> u32 { x }
)";
  constexpr std::string_view kProgram = R"(
import imported;

fn main() -> u32 {
  imported::foo(u32:42)
})";
  ImportData import_data = CreateImportDataForTest();
  XLS_ASSERT_OK_AND_ASSIGN(TypecheckResult module,
                           Typecheck(kImported, "imported", &import_data));
  XLS_EXPECT_OK(Typecheck(kProgram, "main", &import_data));
}

TEST_F(TypecheckV2Test, ProcWithImplAsImportedProcMember) {
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
  ImportData import_data = CreateImportDataForTest();
  XLS_ASSERT_OK_AND_ASSIGN(TypecheckResult module,
                           Typecheck(kImported, "imported", &import_data));
  XLS_EXPECT_OK(Typecheck(kProgram, "main", &import_data));
}

TEST_F(TypecheckV2Test, UseOfConstant) {
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
  absl::StatusOr<TypecheckResult> result =
      Typecheck(kProgram, "fake_main_path", &import_data);
  XLS_EXPECT_OK(result.status()) << result.status();
  ;
}

TEST_F(TypecheckV2Test, DISABLED_UseOfStdlibModule) {
  constexpr std::string_view kProgram = R"(#![feature(use_syntax)]
use std;

fn main() -> u32 {
  std::popcount(u32:42)
}
)";
  XLS_EXPECT_OK(Typecheck(kProgram));
}

TEST_F(TypecheckV2Test, FailsOnProcWithImplAsImportedStructMember) {
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

TEST_F(TypecheckV2Test, ParametricStructInstantiatedByGlobal) {
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

TEST_F(TypecheckV2Test, TopLevelConstTypeMismatch) {
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

TEST_F(TypecheckV2Test, TopLevelConstTypeMatch) {
  constexpr std::string_view kProgram = R"(
const GLOBAL: u32 = u32:4;
)";
  XLS_EXPECT_OK(Typecheck(kProgram));
}

TEST_F(TypecheckV2Test, LetTypeAnnotationIsXn) {
  constexpr std::string_view kProgram = "fn f() { let x: xN = u32:42; }";
  EXPECT_THAT(
      Typecheck(kProgram),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          AllOf(HasSubstrInV1(GetParam(),
                              "Could not determine signedness to turn `xN` "
                              "into a concrete bits type."),
                HasSubstrInV2(GetParam(),
                              "`xN` requires a specified signedness"))));
}

TEST_F(TypecheckV2Test, ParametricIdentifierLtValue) {
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
TEST_F(TypecheckV2Test, Gh1473_UnboundButAlsoUnusedParametricNoDefaultExpr) {
  constexpr std::string_view kProgram = R"(
fn p<X: u32, Y: u32>(y: uN[Y]) -> u32 { Y }
fn f() -> u32 { p(u7:0) }
)";

  EXPECT_THAT(Typecheck(kProgram),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Could not infer parametric(s): X")));
}

TEST_F(TypecheckV2Test, Gh1473_UnboundButAlsoUnusedParametricWithDefaultExpr) {
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
TEST_F(TypecheckV2Test, Gh1473_UnboundAndUsedParametric) {
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
TEST_F(TypecheckV2Test, Gh1473_UnboundAndUsedParametricInBody) {
  constexpr std::string_view kProgram = R"(
fn p<X: u32, Y: u32>(y: uN[Y]) -> bool { uN[X]:0 == uN[X]:1 }
fn f() -> bool { p(u7:0) }
)";
  EXPECT_THAT(Typecheck(kProgram),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Could not infer parametric(s): X")));
}

TEST_F(TypecheckV2Test, MapOfParametric) {
  constexpr std::string_view kProgram = R"(
fn p<N: u32>(x: bits[N]) -> bits[N] { x }

fn f() -> u32[3] {
  map(u32[3]:[1, 2, 3], p)
}
)";
  XLS_EXPECT_OK(Typecheck(kProgram));
}

TEST_F(TypecheckV2Test, MapOfParametricConst) {
  constexpr std::string_view kProgram = R"(
fn one<M: u32>(input: uN[M]) -> uN[M] { uN[M]:1}
const Y = map([u30:5, u30:6], one<u32:30>);
)";
  XLS_EXPECT_OK(Typecheck(kProgram));
}

TEST_F(TypecheckV2Test, MapOfParametricExplicit) {
  constexpr std::string_view kProgram =
      R"(
fn f<N:u32, K:u32>(x: u32) -> uN[N] { x as uN[N] + K as uN[N] }
fn main() -> u5[4] { map(u32[4]:[0, 1, 2, 3], f<u32:5, u32:17>) }
)";

  XLS_EXPECT_OK(Typecheck(kProgram));
}

TEST_F(TypecheckV2Test, MapOfParametricExplicitWithWrongNumberOfArgs) {
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

TEST_F(TypecheckV2Test, MapImportedNonPublicFunction) {
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

TEST_F(TypecheckV2Test, MapImportedNonPublicInferredParametricFunction) {
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

TEST_F(TypecheckV2Test, ParametricInvocationConflictingArgs) {
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

TEST_F(TypecheckV2Test, ParametricWrongKind) {
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

TEST_F(TypecheckV2Test, ParametricWrongNumberOfDims) {
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

TEST_F(TypecheckV2Test, RecursionCausesError) {
  constexpr std::string_view kProgram = "fn f(x: u32) -> u32 { f(x) }";
  EXPECT_THAT(Typecheck(kProgram),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Recursion of function `f` detected")));
}

TEST_F(TypecheckV2Test, ParametricRecursionCausesError) {
  constexpr std::string_view kProgram = R"(
fn f<X: u32>(x: bits[X]) -> u32 { f(x) }
fn g() -> u32 { f(u32: 5) }
)";
  EXPECT_THAT(Typecheck(kProgram),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Recursion of function `f` detected")));
}

TEST_F(TypecheckV2Test, HigherOrderRecursionCausesError) {
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

TEST_F(TypecheckV2Test, InvokeWrongArg) {
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

TEST_F(TypecheckV2Test, InvokeNumberValue) {
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
TEST_F(TypecheckV2Test, InvokeNumberValueInUninstantiatedParametric) {
  constexpr std::string_view kProgram = R"(
fn f<N: u32>(x: u8) -> u8 { 42(x) }
)";
  XLS_EXPECT_OK(Typecheck(kProgram));
}

TEST_F(TypecheckV2Test, BadTupleType) {
  constexpr std::string_view kProgram = R"(
fn f() -> u32 {
  let (a, b, c): (u32, u32) = (u32:1, u32:2, u32:3);
  a
}
)";
  EXPECT_THAT(
      Typecheck(kProgram),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          AllOf(HasSubstrInV1(GetParam(),
                              "Annotated type did not match inferred type"),
                HasSubstrInV2(GetParam(),
                              "Out-of-bounds tuple index specified: 2"))));
}

// -- logical ops

class LogicalOpTypecheckTest
    : public testing::Test,
      public testing::WithParamInterface<std::tuple<std::string_view>> {
 public:
  absl::Status TypecheckOp(std::string_view tmpl) {
    const std::string program = absl::StrReplaceAll(
        tmpl, {{"$OP", std::get<std::string_view>(GetParam())}});
    return ::xls::dslx::TypecheckV2(program).status();
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
  EXPECT_THAT(TypecheckOp("fn f(a: u2, b: u2) -> bool { a $OP b }"),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSizeMismatch("u2", "bool")));
  EXPECT_THAT(TypecheckOp("fn f(a: u32, b: u32) -> bool { a $OP b }"),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSizeMismatch("u32", "bool")));
  EXPECT_THAT(TypecheckOp("fn f(a: s1, b: s1) -> bool { a $OP b }"),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSignednessMismatch("s1", "bool")));
  EXPECT_THAT(TypecheckOp("fn f(a: s32, b: s32) -> bool { a $OP b }"),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasTypeMismatch("s32", "bool")));
}

INSTANTIATE_TEST_SUITE_P(LogicalOpTypecheckTestInstance, LogicalOpTypecheckTest,
                         ::testing::Values("&&", "||"));

// --

TEST_F(TypecheckV2Test, LogicalAndOfComparisons) {
  XLS_EXPECT_OK(Typecheck("fn f(a: u8, b: u8) -> bool { a == b }"));
  XLS_EXPECT_OK(Typecheck(
      "fn f(a: u8, b: u8, c: u32, d: u32) -> bool { a == b && c == d }"));
}

TEST_F(TypecheckV2Test, Typedef) {
  XLS_EXPECT_OK(Typecheck(R"(
type MyTypeAlias = (u32, u8);
fn id(x: MyTypeAlias) -> MyTypeAlias { x }
fn f() -> MyTypeAlias { id((u32:42, u8:127)) }
)"));
}

TEST_F(TypecheckV2Test, For) {
  XLS_EXPECT_OK(Typecheck(R"(
fn f() -> u32 {
  for (i, accum): (u32, u32) in u32:0..u32:3 {
    let new_accum: u32 = accum + i;
    new_accum
  }(u32:0)
})"));
}

TEST_F(TypecheckV2Test, ForInParametricInvokedTwice) {
  XLS_EXPECT_OK(Typecheck(R"(
fn p<N: u32>(x: uN[N]) -> uN[N] {
    for (idx, accum) in u32:0..N {
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

TEST_F(TypecheckV2Test, ForNoAnnotation) {
  XLS_EXPECT_OK(Typecheck(R"(
fn f() -> u32 {
  for (i, accum) in u32:0..u32:3 {
    accum
  }(u32:0)
})"));
}

TEST_F(TypecheckV2Test, ForWildcardIvar) {
  XLS_EXPECT_OK(Typecheck(R"(
fn f() -> u32 {
  for (_, accum) in u32:0..u32:3 {
    accum
  }(u32:0)
})"));
}

// TODO: https://github.com/google/xls/issues/2876 - 'use' syntax is not yet
// supported in TIv2.
TEST_F(TypecheckV2Test, DISABLED_UseOfClog2InModuleScopedConstantDefinition) {
  XLS_EXPECT_OK(Typecheck(R"(#![feature(use_syntax)]
use std::clog2;

const MAX_BITS: u32 = clog2(u32:256);

fn main() -> u32 {
    MAX_BITS
}
)"));
}

// TODO: https://github.com/google/xls/issues/2876 - 'use' syntax is not yet
// supported in TIv2.
TEST_F(TypecheckV2Test, DISABLED_UseOfClog2InParametricOutputType) {
  XLS_EXPECT_OK(Typecheck(R"(#![feature(use_syntax)]
use std::{clog2, is_pow2};

fn p<N: u32, OUT: u32 = {clog2(N)}>(x: uN[N]) -> uN[OUT] {
  const_assert!(is_pow2(N));
  uN[OUT]:0
}

fn main() -> u5 { p(u32:42) }
)"));
}

TEST_F(TypecheckV2Test, ConstAssertParametricOk) {
  XLS_EXPECT_OK(Typecheck(R"(
fn p<N: u32>() -> u32 {
  const_assert!(N == u32:42);
  N
}
fn main() -> u32 {
  p<u32:42>()
})"));
}

TEST_F(TypecheckV2Test, ConstAssertViaConstBindings) {
  XLS_EXPECT_OK(Typecheck(R"(
fn main() -> () {
  const M = u32:1;
  const N = u32:2;
  const O = M + N;
  const_assert!(O == u32:3);
  ()
})"));
}

TEST_F(TypecheckV2Test, ConstAssertCallFunction) {
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

TEST_F(TypecheckV2Test, ConstAssertFalse) {
  EXPECT_THAT(Typecheck(R"(
fn main() -> () {
  const_assert!(false);
})"),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("const_assert! failure: `false`")));
}

TEST_F(TypecheckV2Test, ConstAssertFalseExpr) {
  EXPECT_THAT(Typecheck(R"(
fn main() -> () {
  const_assert!(u32:2 + u32:3 != u32:5);
})"),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("const_assert! failure")));
}

TEST_F(TypecheckV2Test, ConstAssertNonConstexpr) {
  EXPECT_THAT(Typecheck(R"(
fn main(p: u32) -> () {
  const_assert!(p == u32:42);
})"),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("`p == u32:42` is not constexpr")));
}

TEST_F(TypecheckV2Test, FitsInTypeSN0) {
  EXPECT_THAT(Typecheck(R"(
fn main() -> sN[0] {
  sN[0]:0xffff_ffff_ffff_ffff_ffff
})"),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Value '0xffff_ffff_ffff_ffff_ffff' does not "
                                 "fit in the bitwidth of a sN[0]")));
}

TEST_F(TypecheckV2Test, ParametricBindArrayToTuple) {
  EXPECT_THAT(
      Typecheck(R"(
fn p<N: u32>(x: (uN[N], uN[N])) -> uN[N] { x.0 }

fn main() -> u32 {
  p(u32[2]:[0, 1])
})"),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          AllOf(HasSubstrInV1(GetParam(),
                              "Parameter 0 and argument types are different "
                              "kinds (tuple vs array)"),
                HasTypeMismatchInV2(GetParam(), "u32[2]", "(uN[N], uN[N])"))));
}

TEST_F(TypecheckV2Test, ParametricBindNested) {
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
              HasTypeMismatchInV2(GetParam(), "(u32, u64)", "u32[1]"),
              HasSubstrInV1(GetParam(),
                            "expected argument kind 'array' to match parameter "
                            "kind 'tuple'"))));
}

TEST_F(TypecheckV2Test, ForBuiltinInBody) {
  XLS_EXPECT_OK(Typecheck(R"(
fn f() -> u32 {
  for (i, accum): (u32, u32) in u32:0..u32:3 {
    trace!(accum)
  }(u32:0)
})"));
}

TEST_F(TypecheckV2Test, ForNestedBindings) {
  XLS_EXPECT_OK(Typecheck(R"(
fn f(x: u32) -> (u32, u8) {
  for (i, (x, y)): (u32, (u32, u8)) in u32:0..u32:3 {
    (x, y)
  }((x, u8:42))
}
)"));
}

TEST_F(TypecheckV2Test, ForWithBadTypeTree) {
  EXPECT_THAT(
      Typecheck(R"(
fn f(x: u32) -> (u32, u8) {
  for (i, (x, y)): (u32, u8) in u32:0..u32:3 {
    (x, y)
  }((x, u8:42))
})"),
      StatusIs(absl::StatusCode::kInvalidArgument,
               AllOf(HasSubstrInV1(GetParam(), "uN[8]\nvs (uN[32], uN[8])"),
                     HasSubstrInV1(GetParam(),
                                   "For-loop annotated accumulator type did "
                                   "not match inferred type."),
                     HasTypeMismatchInV2(GetParam(), "(u32, u8)", "u8"))));
}

TEST_F(TypecheckV2Test, ForWithBadTypeAnnotation) {
  EXPECT_THAT(
      Typecheck(R"(
fn f(x: u32) -> (u32, u8) {
  for (i, _): u32 in u32:0..u32:3 {
    i
  }(u32:0)
})"),
      StatusIs(absl::StatusCode::kInvalidArgument,
               AllOf(HasSubstr("For-loop annotated type should be a tuple "
                               "containing a type for the iterable and a type "
                               "for the accumulator."))));
}

TEST_F(TypecheckV2Test, ForWithWrongResultType) {
  EXPECT_THAT(
      Typecheck(R"(
fn f(x: u32) -> (u32, u8) {
  for (i, _): (u32, u32) in u32:0..u32:3 {
    i as u64
  }(u32:0)
})"),
      StatusIs(absl::StatusCode::kInvalidArgument,
               AllOf(HasSubstrInV1(GetParam(), "uN[32] vs uN[64]"),
                     HasSubstrInV1(GetParam(),
                                   "For-loop init value type did not match "
                                   "for-loop body's result type."),
                     HasTypeMismatchInV2(GetParam(), "(u32, u8)", "u32"))));
}

TEST_F(TypecheckV2Test, ForWithWrongNumberOfArguments) {
  EXPECT_THAT(
      Typecheck(R"(
fn f(x: u32) -> u32 {
  for (i, j, acc): (u32, u32, u32) in u32:0..u32:3 {
    i + j + acc
  }(u32:42)
})"),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          AllOf(HasSubstrInV1(
                    GetParam(),
                    "For-loop annotated type should specify a type for the "
                    "iterable and a type for the accumulator; got 3 types."),
                HasSubstrInV2(
                    GetParam(),
                    "For-loop annotated type should be a tuple containing a "
                    "type for the iterable and a type for the accumulator."))));
}

TEST_F(TypecheckV2Test, ForWithIndexTypeTooSmallForRange) {
  EXPECT_THAT(
      Typecheck(R"(
fn test() -> u4 {
  for(i, acc): (u4, u32) in u32:0..u32:120 {
    i as u32 + acc
  }(u32:0)
})"),
      StatusIs(absl::StatusCode::kInvalidArgument,
               AllOf(HasSubstrInV1(GetParam(), "uN[4]\nvs uN[32]"),
                     HasSubstrInV1(GetParam(),
                                   "For-loop annotated index type did not "
                                   "match inferred type."),
                     HasSizeMismatchInV2(GetParam(), "u4", "u32"))));
}

TEST_F(TypecheckV2Test, UnrollForSimple) {
  XLS_EXPECT_OK(Typecheck(R"(
fn test() -> u32 {
  unroll_for!(i, acc): (u32, u32) in u32:0..u32:4 {
    i + acc
  }(u32:0)
})"));
}

TEST_F(TypecheckV2Test, UnrollForNestedBindings) {
  XLS_EXPECT_OK(Typecheck(R"(
fn f(x: u32) -> (u32, u8) {
  unroll_for! (_, (x, y)): (u32, (u32, u8)) in u32:0..u32:3 {
    (x, y)
  }((x, u8:42))
}
)"));
}

TEST_F(TypecheckV2Test, UnrollForWithBadTypeTree) {
  EXPECT_THAT(
      Typecheck(R"(
fn f(x: u32) -> (u32, u8) {
  unroll_for! (i, (x, y)): (u32, u8) in u32:0..u32:3 {
    (x, y)
  }((x, u8:42))
})"),
      StatusIs(absl::StatusCode::kInvalidArgument,
               AllOf(HasSubstrInV1(GetParam(), "uN[8]\nvs (uN[32], uN[8])"),
                     HasSubstrInV1(GetParam(),
                                   "For-loop annotated accumulator type "
                                   "did not match inferred type."),
                     HasTypeMismatchInV2(GetParam(), "(u32, u8)", "u8"))));
}

TEST_F(TypecheckV2Test, UnrollForWithBadTypeAnnotation) {
  EXPECT_THAT(
      Typecheck(R"(
fn f(x: u32) -> (u32, u8) {
  unroll_for! (i, _): u32 in u32:0..u32:3 {
    i
  }(u32:0)
})"),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("For-loop annotated type should be a tuple "
                         "containing a type for the iterable and a type "
                         "for the accumulator.")));
}

TEST_F(TypecheckV2Test, UnrollForWithoutIndexAccTypeAnnotation) {
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

TEST_F(TypecheckV2Test, UnrollForWithWrongResultType) {
  EXPECT_THAT(
      Typecheck(R"(
fn f(x: u32) -> (u32, u8) {
  unroll_for! (i, _): (u32, u32) in u32:0..u32:3 {
    i as u64
  }(u32:0)
})"),
      StatusIs(absl::StatusCode::kInvalidArgument,
               AllOf(HasSubstrInV1(GetParam(), "uN[32] vs uN[64]"),
                     HasSubstrInV1(GetParam(),
                                   "For-loop init value type did not match "
                                   "for-loop body's result type."),
                     HasTypeMismatchInV2(GetParam(), "(u32, u8)", "u32"))));
}

TEST_F(TypecheckV2Test, UnrollForWithWrongNumberOfArguments) {
  EXPECT_THAT(
      Typecheck(R"(
fn test() -> u32 {
  unroll_for!(i, j, acc): (u32, u32, u32) in u32:0..u32:4 {
    i + j + acc
  }(u32:0)
})"),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          AllOf(HasTypeMismatchInV1(GetParam(), "(uN[32], uN[32], uN[32])",
                                    "(uN[32], uN[32])"),
                HasSubstrInV1(
                    GetParam(),
                    "For-loop annotated type should specify a type for the "
                    "iterable and a type for the accumulator; got 3 types."),
                HasSubstrInV2(
                    GetParam(),
                    "For-loop annotated type should be a tuple containing a "
                    "type for the iterable and a type for the accumulator."))));
}

TEST_F(TypecheckV2Test, UnrollForWithIndexTypeTooSmallForRange) {
  EXPECT_THAT(
      Typecheck(R"(
fn test() -> u4 {
  unroll_for!(i, acc): (u4, u4) in u32:0..u32:120 {
    i + acc
  }(u4:0)
})"),
      StatusIs(absl::StatusCode::kInvalidArgument,
               AllOf(HasSizeMismatchInV1(GetParam(), "uN[4]", "uN[32]"),
                     HasSubstrInV1(GetParam(),
                                   "For-loop annotated index type did not "
                                   "match inferred type."),
                     HasSizeMismatchInV2(GetParam(), "u4", "u32"))));
}

// https://github.com/google/xls/issues/1717
TEST_F(TypecheckV2Test, UnrollForWithInvocationInTypeAlias) {
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

TEST_F(TypecheckV2Test, DerivedParametricStruct) {
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

TEST_F(TypecheckV2Test, ZeroMacroSpecified) {
  XLS_EXPECT_OK(Typecheck("const Y = zero!<u10>();"));
}

TEST_F(TypecheckV2Test, ZeroMacroFromParametric) {
  XLS_EXPECT_OK(Typecheck(R"(
fn f<N:u32>() -> uN[N] { zero!<uN[N]>() }
const Y = f<u32:10>();
)"));
}

TEST_F(TypecheckV2Test, ZeroMacroFromParametricError) {
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

TEST_F(TypecheckV2Test, ZeroMacroFromStructConstError) {
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

TEST_F(TypecheckV2Test, ZeroMacroImportedType) {
  constexpr std::string_view kImported = R"(
pub type foo_t = u32;
)";
  constexpr std::string_view kProgram = R"(
import imported;
const Y = zero!<imported::foo_t>();
)";
  auto import_data = CreateImportDataForTest();
  XLS_EXPECT_OK(Typecheck(kImported, "imported", &import_data));
  XLS_ASSERT_OK(Typecheck(kProgram, "main", &import_data));
}

TEST_F(TypecheckV2Test, DerivedParametricStructUsingNonDefault) {
  XLS_EXPECT_OK(Typecheck(R"(
struct StructFoo<A: u32, B:u32 = {A * u32:2}> {
  x: uN[B],
}

fn Foo() {
  let b = StructFoo<u32:8>{x: u16:0};
}
)"));
}

TEST_F(TypecheckV2Test, DerivedParametricStructValueMissing) {
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

TEST_F(TypecheckV2Test, DerivedParametricStructNoParametrics) {
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
TEST_F(TypecheckV2Test, ParametricStructWithWrongOrderParametricValues) {
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

TEST_F(TypecheckV2Test,
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

TEST_F(TypecheckV2Test, DerivedExprTypeMismatch) {
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

TEST_F(TypecheckV2Test, ParametricExpressionInBitsReturnType) {
  XLS_EXPECT_OK(Typecheck(R"(
fn parametric<X: u32>() -> bits[X * u32:4] {
  type Ret = bits[X * u32:4];
  Ret:0
}
fn main() -> bits[4] { parametric<u32:1>() }
)"));
}

TEST_F(TypecheckV2Test, ParametricInstantiationVsArgOk) {
  XLS_EXPECT_OK(Typecheck(R"(
fn parametric<X: u32 = {u32:5}> (x: bits[X]) -> bits[X] { x }
fn main() -> bits[5] { parametric(u5:1) }
)"));
}

TEST_F(TypecheckV2Test, ParametricInstantiationVsArgError) {
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
                     HasTypeMismatchInV2(GetParam(), "bits[5]", "bits[10]"))));
}

TEST_F(TypecheckV2Test, ParametricInstantiationVsBodyOk) {
  XLS_EXPECT_OK(Typecheck(R"(
fn parametric<X: u32 = {u32:5}>() -> bits[5] { bits[X]:1 + bits[5]:1 }
fn main() -> bits[5] { parametric() }
)"));
}

TEST_F(TypecheckV2Test, ParametricInstantiationVsBodyError) {
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

TEST_F(TypecheckV2Test, ParametricInstantiationVsReturnOk) {
  XLS_EXPECT_OK(Typecheck(R"(
fn parametric<X: u32 = {u32: 5}>() -> bits[5] { bits[X]: 1 }
fn main() -> bits[5] { parametric() }
)"));
}

TEST_F(TypecheckV2Test, ParametricInstantiationVsReturnError) {
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

TEST_F(TypecheckV2Test, ParametricIndirectInstantiationVsArgOk) {
  XLS_EXPECT_OK(Typecheck(R"(
fn foo<X: u32>(x1: bits[X], x2: bits[X]) -> bits[X] { x1 + x2 }
fn fazz<Y: u32>(y: bits[Y]) -> bits[Y] { foo(y, y + bits[Y]: 1) }
fn bar() -> bits[10] { fazz(u10: 1) }
)"));
}

TEST_F(TypecheckV2Test, ParametricInstantiationVsArgError2) {
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

TEST_F(TypecheckV2Test, ParametricIndirectInstantiationVsBodyOk) {
  XLS_EXPECT_OK(Typecheck(R"(
fn foo<X: u32, R: u32 = {X + X}>(x: bits[X]) -> bits[R] {
  let a = bits[R]: 5;
  x++x + a
}
fn fazz<Y: u32, T: u32 = {Y + Y}>(y: bits[Y]) -> bits[T] { foo(y) }
fn bar() -> bits[10] { fazz(u5:1) }
)"));
}

TEST_F(TypecheckV2Test, ParametricIndirectInstantiationVsBodyError) {
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

TEST_F(TypecheckV2Test, ParametricIndirectInstantiationVsReturnOk) {
  XLS_EXPECT_OK(Typecheck(R"(
fn foo<X: u32, R: u32 = {X + X}>(x: bits[X]) -> bits[R] { x++x }
fn fazz<Y: u32, T: u32 = {Y + Y}>(y: bits[Y]) -> bits[T] { foo(y) }
fn bar() -> bits[10] { fazz(u5:1) }
)"));
}

TEST_F(TypecheckV2Test, ParametricIndirectInstantiationVsReturnError) {
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

TEST_F(TypecheckV2Test, ParametricDerivedInstantiationVsArgOk) {
  XLS_EXPECT_OK(Typecheck(R"(
fn foo<X: u32, Y: u32 = {X + X}>(x: bits[X], y: bits[Y]) -> bits[X] { x }
fn bar() -> bits[5] { foo(u5:1, u10: 2) }
)"));
}

TEST_F(TypecheckV2Test, ParametricDerivedInstantiationVsArgError) {
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

TEST_F(TypecheckV2Test, ParametricDerivedInstantiationVsBodyOk) {
  XLS_EXPECT_OK(Typecheck(R"(
fn foo<W: u32, Z: u32 = {W + W}>(w: bits[W]) -> bits[1] {
    let val: bits[Z] = w++w + bits[Z]: 5;
    and_reduce(val)
}
fn bar() -> bits[1] { foo(u5: 5) + foo(u10: 10) }
)"));
}

TEST_F(TypecheckV2Test, ParametricDerivedInstantiationVsBodyError) {
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

TEST_F(TypecheckV2Test, ParametricDerivedInstantiationVsReturnOk) {
  XLS_EXPECT_OK(Typecheck(R"(
fn double<X: u32, Y: u32 = {X + X}> (x: bits[X]) -> bits[Y] { x++x }
fn foo<W: u32, Z: u32 = {W + W}> (w: bits[W]) -> bits[Z] { double(w) }
fn bar() -> bits[10] { foo(u5:1) }
)"));
}

TEST_F(TypecheckV2Test, ParametricDerivedInstantiationVsReturnError) {
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

TEST_F(TypecheckV2Test, ParametricDerivedInstantiationViaFnCall) {
  XLS_EXPECT_OK(Typecheck(R"(
fn double(n: u32) -> u32 { n * u32: 2 }
fn foo<W: u32, Z: u32 = {double(W)}>(w: bits[W]) -> bits[Z] { w++w }
fn bar() -> bits[10] { foo(u5:1) }
)"));
}

TEST_F(TypecheckV2Test, ParametricFnNotAlwaysPolymorphic) {
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

TEST_F(TypecheckV2Test, ParametricWidthSliceStartError) {
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
          AllOf(HasSubstrInV1(GetParam(), "Cannot fit slice start 4 in 2 bits"),
                HasSubstrInV2(GetParam(),
                              "Inferred type of slice bound (3 bits) is too "
                              "large for slicing an array of size 2"))));
}

TEST_F(TypecheckV2Test, NonParametricCallInParametricExpr) {
  XLS_EXPECT_OK(Typecheck(R"(
fn id(x: u32) -> u32 { x }
fn p<X: u32, Y: u32 = {id(X)}>() -> u32 { Y }
fn main() -> u32 { p<u32:42>() }
)"));
}

TEST_F(TypecheckV2Test, ParametricCallInParametricExpr) {
  XLS_EXPECT_OK(Typecheck(R"(
fn pid<N: u32>(x: bits[N]) -> bits[N] { x }
fn p<X: u32, Y: u32 = {pid(X)}>() -> u32 { Y }
fn main() -> u32 { p<u32:42>() }
)"));
}

TEST_F(TypecheckV2Test, ParametricCallWithDeducedValuesFromArgs) {
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

TEST_F(TypecheckV2Test, BitSliceOnParametricWidth) {
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

TEST_F(TypecheckV2Test, WidthSliceWithANonType) {
  EXPECT_THAT(
      Typecheck(R"(import float32;

fn f(x: u32) -> u2 {
  x[0 +: float32::F32_EXP_SZ]  // Note this is a constant not a type.
}
)"),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("Expected a type, got `float32::F32_EXP_SZ`.")));
}

// This test adds a literal u5 to a parametric-typed number -- which only works
// when that parametric-type number is also coincidentally a u5.
TEST_F(TypecheckV2Test, ParametricMapNonPolymorphic) {
  EXPECT_THAT(Typecheck(R"(
fn add_one<N: u32>(x: bits[N]) -> bits[N] { x + u5:1 }

fn main() {
  let arr = u5[3]:[1, 2, 3];
  let mapped_arr: u5[3] = map(arr, add_one);
  let type_error = add_one(u6:1);
}
)"),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasTypeMismatch("uN[6]", "uN[5]")));
}

TEST_F(TypecheckV2Test, LetBindingInferredDoesNotMatchAnnotation) {
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

TEST_F(TypecheckV2Test, CoverBuiltinWrongArgc) {
  EXPECT_THAT(
      Typecheck(R"(
fn f() -> () {
  cover!()
}
)"),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          AllOf(
              HasSubstrInV1(GetParam(),
                            "Invalid number of arguments passed to 'cover!'"),
              HasSubstrInV2(GetParam(), "Expected 2 argument(s) but got 0."))));
}

TEST_F(TypecheckV2Test, MapBuiltinWrongArgc0) {
  EXPECT_THAT(
      Typecheck(R"(
fn f() {
  map()
}
)"),
      StatusIs(absl::StatusCode::kInvalidArgument,
               AllOf(HasSubstrInV1(GetParam(),
                                   "Expected 2 arguments to `map` builtin but "
                                   "got 0 argument(s)"),
                     HasSubstrInV2(GetParam(),
                                   "Expected 2 argument(s) but got 0"))));
}

TEST_F(TypecheckV2Test, MapBuiltinWrongArgc1) {
  EXPECT_THAT(
      Typecheck(R"(
fn f(x: u32[3]) -> u32[3] {
  map(x)
}
)"),
      StatusIs(absl::StatusCode::kInvalidArgument,
               AllOf(HasSubstrInV1(GetParam(),
                                   "Expected 2 arguments to `map` builtin but "
                                   "got 1 argument(s)"),
                     HasSubstrInV2(GetParam(),
                                   "Expected 2 argument(s) but got 1"))));
}

TEST_F(TypecheckV2Test, UpdateBuiltin) {
  XLS_EXPECT_OK(Typecheck(R"(
fn f() -> u32[3] {
  let x: u32[3] = u32[3]:[0, 1, 2];
  update(x, u32:1, u32:3)
}
)"));
}

TEST_F(TypecheckV2Test, UpdateBuiltin2D) {
  XLS_EXPECT_OK(Typecheck(R"(
fn f() -> u32[2][3] {
  let x: u32[2][3] = u32[2][3]:[[u32:0,u32:1], [u32:2,u32:3], [u32:3,u32:4]];
  update(x, (u1:0, u32:1), u32:3)
}
)"));
}

TEST_F(TypecheckV2Test, UpdateBuiltinNotAnIndex) {
  EXPECT_THAT(
      Typecheck(R"(
fn f() -> u32[2][3] {
  let x: u32[2][3] = u32[2][3]:[[u32:0,u32:1], [u32:2,u32:3], [u32:3,u32:4]];
  update(x, [u32:0, u32:1], u32:3)
}
)"),
      StatusIs(absl::StatusCode::kInvalidArgument,
               AllOf(HasSubstrInV1(GetParam(),
                                   "Want index value at argno 1 to either be a "
                                   "`uN` or a tuple of `uN`s"),
                     HasSubstrInV2(GetParam(),
                                   "`update` index type must be a bits type; "
                                   "got `uN[32][2]`"))));
}

TEST_F(TypecheckV2Test, UpdateBuiltinIndexTupleHasSigned) {
  EXPECT_THAT(
      Typecheck(R"(
fn f() -> u32[2][3] {
  let x: u32[2][3] = u32[2][3]:[[u32:0,u32:1], [u32:2,u32:3], [u32:3,u32:4]];
  update(x, (u32:0, s32:1), u32:3)
}
)"),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          AllOf(HasSubstrInV1(GetParam(),
                              "Want index value within tuple to be `uN`; "
                              "member 1 was `sN[32]`"),
                HasSubstrInV2(
                    GetParam(),
                    "`update` index type must be unsigned; got `sN[32]`"))));
}

TEST_F(TypecheckV2Test, UpdateBuiltinOutOfDimensions) {
  EXPECT_THAT(
      Typecheck(R"(
fn f() -> u32[2][3] {
  let x: u32[2][3] = u32[2][3]:[[u32:0,u32:1], [u32:2,u32:3], [u32:3,u32:4]];
  update(x, (u1:0, u32:1, u32:2), u32:3)
}
)"),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          AllOf(
              HasSubstrInV1(GetParam(),
                            "Want argument 0 type uN[32][2][3] dimensions: 2 "
                            "to be larger"),
              HasSubstrInV2(GetParam(),
                            "Array dimension in `update` expected to be larger "
                            "than the number of indices (3); got 2"))));
}

TEST_F(TypecheckV2Test, UpdateBuiltinTypeMismatch) {
  EXPECT_THAT(
      Typecheck(R"(
fn f() -> u32[2][3] {
  let x: u32[2][3] = u32[2][3]:[[u32:0,u32:1], [u32:2,u32:3], [u32:3,u32:4]];
  update(x, (u1:0, u32:1), u8:3)
}
)"),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          AllOf(HasSubstrInV1(GetParam(),
                              "Want argument 0 element type uN[32] to match"),
                HasSizeMismatchInV2(GetParam(), "uN[32]", "uN[8]"))));
}

TEST_F(TypecheckV2Test, UpdateBuiltinEmptyIndex) {
  XLS_EXPECT_OK(Typecheck(R"(
fn f() -> u32[2][3] {
  let x: u32[2][3] = u32[2][3]:[[u32:0,u32:1], [u32:2,u32:3], [u32:3,u32:4]];
  update(x, (), u32[2][3]:[[u32:0,u32:1], [u32:2,u32:3], [u32:3,u32:4]])
}
)"));
}

TEST_F(TypecheckV2Test, SliceBuiltin) {
  XLS_EXPECT_OK(Typecheck(R"(
fn f() -> u32[3] {
  let x: u32[2] = u32[2]:[0, 1];
  array_slice(x, u32:0, u32[3]:[0, 0, 0])
}
)"));
}

TEST_F(TypecheckV2Test, EnumerateBuiltin) {
  XLS_EXPECT_OK(Typecheck(R"(
type MyTup = (u32, u2);
fn f(x: u2[7]) -> MyTup[7] {
  enumerate(x)
}
)"));
}

TEST_F(TypecheckV2Test, TernaryEmptyBlocks) {
  XLS_EXPECT_OK(Typecheck(R"(
fn f(p: bool) -> () {
  if p { } else { }
}
)"));
}

TEST_F(TypecheckV2Test, TernaryNonBoolean) {
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

TEST_F(TypecheckV2Test, SizeofImportedType) {
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

TEST_F(TypecheckV2Test, InvalidParametricTypeConstantArrayContents) {
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
  EXPECT_THAT(result,
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Array element cannot be a type reference.")))
      << result.status();
}

TEST_F(TypecheckV2Test, ArraySizeOfBitsType) {
  EXPECT_THAT(
      Typecheck(R"(
fn f(x: u32) -> u32 { array_size(x) }
)"),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          AllOf(
              HasSubstrInV1(
                  GetParam(),
                  "Want argument 0 to 'array_size' to be an array; got uN[32]"),
              HasSizeMismatchInV2(GetParam(), "u32", "Any[N]"))));
}

// TODO: this test fails in v2: TypeInferenceError:
// type mismatch: MyStruct[5] at fake.x:5:9-5:20 vs. Any[N] at
// fake.x:19:35-19:39
TEST_F(TypecheckV2Test, ArraySizeOfStructs) {
  XLS_EXPECT_OK(Typecheck(R"(
struct MyStruct {}
fn f(x: MyStruct[5]) -> u32 { array_size(x) }
)"));
}

TEST_F(TypecheckV2Test, ArraySizeOfNil) {
  XLS_EXPECT_OK(Typecheck(R"(
fn f(x: ()[5]) -> u32 { array_size(x) }
)"));
}

TEST_F(TypecheckV2Test, ArraySizeOfTupleArray) {
  XLS_EXPECT_OK(Typecheck(R"(
fn f(x: (u32, u64)[5]) -> u32 { array_size(x) }
)"));
}

TEST_F(TypecheckV2Test, BitSliceUpdateBuiltIn) {
  XLS_EXPECT_OK(Typecheck(R"(
fn f(x: u32, y: u17, z: u15) -> u32 {
  bit_slice_update(x, y, z)
}
)"));
}

TEST_F(TypecheckV2Test, UpdateIncompatibleValue) {
  EXPECT_THAT(
      Typecheck(R"(
fn f(x: u32[5]) -> u32[5] {
  update(x, u32:1, u8:0)
}
)"),
      StatusIs(absl::StatusCode::kInvalidArgument,
               AllOf(HasSubstrInV1(GetParam(),
                                   "uN[32] to match argument 2 type uN[8]"),
                     HasSizeMismatchInV2(GetParam(), "uN[32]", "uN[8]"))));
}

TEST_F(TypecheckV2Test, MissingAnnotation) {
  XLS_EXPECT_OK(Typecheck(R"(
fn f() -> u32 {
  let x = u32:2;
  x + x
}
)"));
}

TEST_F(TypecheckV2Test, Index) {
  // Indexing a 4-element bit-constructed array with a constant index.
  XLS_EXPECT_OK(Typecheck("fn f(x: uN[32][4]) -> u32 { x[u32:0] }"));

  // Indexing a 5-element array with a dynamic index.
  XLS_EXPECT_OK(Typecheck("fn f(x: u32[5], i: u8) -> u32 { x[i] }"));

  // Indexing a bit value is not allowed, only arrays.
  EXPECT_THAT(Typecheck("fn f(x: u32, i: u8) -> u32 { x[i] }"),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Bits-like value cannot be indexed, value to "
                                 "index is not an array.")));

  // Indexing a parametric-signedness bit value is not allowed.
  // TODO: Fix this bad error message.
  EXPECT_THAT(Typecheck("fn f(x: xN[false][5], i: u8) -> u1 { x[i] }"),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("`xN` requires a specified bit count")));

  // Cannot use an array as an index.
  EXPECT_THAT(Typecheck("fn f(x: u32[5], i: u8[5]) -> u32 { x[i] }"),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasTypeMismatch("u8[5]", "u32")));

  // Cannot use a signed number as an index.
  EXPECT_THAT(Typecheck("fn f(x: u32[5], i: s8) -> u32 { x[i] }"),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSignednessMismatch("s8", "u32")));
}

TEST_F(TypecheckV2Test, OutOfRangeNumber) {
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

TEST_F(TypecheckV2Test, OutOfRangeNumberInConstantArray) {
  EXPECT_THAT(
      Typecheck("fn f() -> u8[3] { u8[3]:[1, 2, 256] }"),
      StatusIs(absl::StatusCode::kInvalidArgument,
               AllOf(HasSubstrInV1(
                         GetParam(),
                         "Value '256' does not fit in the bitwidth of a uN[8]"),
                     HasSizeMismatchInV2(GetParam(), "u9", "u8"))));
}

TEST_F(TypecheckV2Test, BadTypeForConstantArrayOfNumbers) {
  EXPECT_THAT(
      Typecheck("const A = u8[3][4]:[1, 2, 3, 4];"),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          AllOf(HasSubstrInV1(
                    GetParam(),
                    "Non-bits type (array) used to define a numeric literal"),
                HasTypeMismatchInV2(GetParam(), "u8[3]", "uN[1]"))));
}

TEST_F(TypecheckV2Test, ConstantArrayEmptyMembersWrongCountVsDecl) {
  auto result = Typecheck("const MY_ARRAY = u32[1]:[];");
  EXPECT_THAT(
      result,
      StatusIs(absl::StatusCode::kInvalidArgument,
               AllOf(HasSubstrInV1(GetParam(),
                                   "uN[32][1] Array has zero elements "
                                   "but type annotation size is 1"),
                     HasTypeMismatchInV2(GetParam(), "u32[0]", "u32[1]"))));
}

TEST_F(TypecheckV2Test, MatchNoArms) {
  EXPECT_THAT(
      Typecheck("fn f(x: u8) -> u8 { let _ = match x {}; x }"),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          AllOf(HasSubstrInV1(GetParam(), "Match construct has no arms"),
                HasSubstrInV2(GetParam(), "`match` expression has no arms"))));
}

TEST_F(TypecheckV2Test, MatchArmMismatch) {
  EXPECT_THAT(
      Typecheck("fn f(x: u8) -> u8 { match x { u8:0 => u8:3, _ => u3:3 } }"),
      StatusIs(absl::StatusCode::kInvalidArgument,
               AllOf(HasSubstrInV1(GetParam(),
                                   "match arm did not have the same type"),
                     HasSizeMismatchInV2(GetParam(), "u3", "u8"))));
}

TEST_F(TypecheckV2Test, MatchOnParametricFunction) {
  EXPECT_THAT(
      Typecheck(R"(
fn p<N: u32>() -> u32 {
  match p {
    p => N
  }
}

const X: u32 = p<u32:42>();
)"),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          AllOf(HasSubstrInV1(GetParam(),
                              "Match construct cannot match on this type."),
                // TODO: improve this error message in v2; similar for
                // other tests with this error message.
                HasSubstrInV2(GetParam(),
                              "Attempting to concretize `Any` type"))));
}

TEST_F(TypecheckV2Test, MatchWithFunctionInPattern) {
  EXPECT_THAT(Typecheck(R"(
fn p<N: u32>(x: bool) -> u32 {
  match x {
    p => N,
    _ => u32:0,
  }
}

const X: u32 = p<u32:42>(false);
)"),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       AllOf(HasSubstrInV1(GetParam(), "pattern expects uN[1]"),
                             HasSubstrInV2(
                                 GetParam(),
                                 "Expected type `uN[1]` but got `p`, which is "
                                 "a parametric function not being invoked"))));
}

TEST_F(TypecheckV2Test, MatchNonExhaustive) {
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

TEST_F(TypecheckV2Test, MatchWithOneNonExhaustivePattern) {
  absl::StatusOr<TypecheckResult> result = Typecheck(R"(
fn f(x: u32) -> u32 {
    match x {
        u32:1 => u32:64,
    }
}
)");
  EXPECT_THAT(result, StatusIs(absl::StatusCode::kInvalidArgument,
                               HasSubstr("Match patterns are not exhaustive")));
}

TEST_F(TypecheckV2Test, ArrayInconsistency) {
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

TEST_F(TypecheckV2Test, MatchWithRange) {
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

TEST_F(TypecheckV2Test, MatchWithRangeInclusive) {
  XLS_EXPECT_OK(Typecheck(R"(
fn f(x: u32) -> u32 {
  match x {
    u32:0..=u32:2 => x,
    _ => u32:42,
  }
}

const X: u32 = f(u32:2);
const_assert!(X == u32:2);
)"));
}

TEST_F(TypecheckV2Test, MatchArmDuplicated) {
  EXPECT_THAT(
      Typecheck(R"(
const X = u32:1;
const Y = u32:2;
const Z = match X {
  u32:1 => X,
  u32:0 => X,
  u32:1 => Y,
  _ => Y
};
)"),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("Exact-duplicate pattern match detected `u32:1`")));
}

TEST_F(TypecheckV2Test, ArrayOfConsts) {
  XLS_EXPECT_OK(Typecheck(R"(
fn f() -> u4 {
  let a: u4 = u4:1;
  let my_array = [a];
  a
}
)"));
}

TEST_F(TypecheckV2Test, EnumIdentity) {
  XLS_EXPECT_OK(Typecheck(R"(
enum MyEnum : u1 {
  A = false,
  B = true,
}
fn f(x: MyEnum) -> MyEnum { x }
)"));
}

TEST_F(TypecheckV2Test, ImplicitWidthEnum) {
  XLS_EXPECT_OK(Typecheck(R"(
enum MyEnum {
  A = false,
  B = true,
}
)"));
}

TEST_F(TypecheckV2Test, ImplicitWidthEnumFromConstexprs) {
  XLS_EXPECT_OK(Typecheck(R"(
const X = u8:42;
const Y = u8:64;
enum MyEnum {
  A = X,
  B = Y,
}
)"));
}

TEST_F(TypecheckV2Test, ImplicitWidthEnumWithConstexprAndBareLiteral) {
  XLS_EXPECT_OK(Typecheck(R"(
const X = u8:42;
enum MyEnum {
  A = 64,
  B = X,
}

const EXTRACTED_A = MyEnum::A as u8;
)"));
}

TEST_F(TypecheckV2Test, ImplicitWidthEnumFromConstexprsMismatch) {
  EXPECT_THAT(Typecheck(R"(
const X = u7:42;
const Y = u8:64;
enum MyEnum {
  A = X,
  B = Y,
}
)"),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       AllOf(HasSubstrInV1(
                                 GetParam(),
                                 "uN[7] vs uN[8]: Inconsistent member types in "
                                 "enum definition."),
                             HasTypeMismatchInV2(GetParam(), "u7", "u8"))));
}

TEST_F(TypecheckV2Test, ImplicitWidthEnumMismatch) {
  EXPECT_THAT(
      Typecheck(R"(
enum MyEnum {
  A = u1:0,
  B = u2:1,
}
)"),
      StatusIs(absl::StatusCode::kInvalidArgument,
               AllOf(HasSubstrInV1(GetParam(),
                                   "uN[1] vs uN[2]: Inconsistent member "
                                   "types in enum definition"),
                     HasTypeMismatchInV2(GetParam(), "u1", "u2"))));
}

TEST_F(TypecheckV2Test, ExplicitWidthEnumMismatch) {
  EXPECT_THAT(
      Typecheck(R"(
enum MyEnum : u2 {
  A = u1:0,
  B = u1:1,
}
)"),
      StatusIs(absl::StatusCode::kInvalidArgument,
               AllOf(HasSubstrInV1(GetParam(),
                                   "uN[1] vs uN[2]: Enum-member type did not "
                                   "match the enum's underlying type"),
                     HasSizeMismatchInV2(GetParam(), "u1", "u2"))));
}

TEST_F(TypecheckV2Test, ArrayEllipsis) {
  XLS_EXPECT_OK(Typecheck("fn main() -> u8[2] { u8[2]:[0, ...] }"));
}

// See https://github.com/google/xls/issues/1587 #1
TEST_F(TypecheckV2Test, ArrayEllipsisTypeSmallerThanElements) {
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
TEST_F(TypecheckV2Test, ArrayEllipsisTypeEqElementCount) {
  XLS_EXPECT_OK(
      Typecheck("fn main() -> u32[2] { u32[2]:[u32:0, u32:1, ...] }"));
}

TEST_F(TypecheckV2Test, ArrayEllipsisNoTrailingElement) {
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

TEST_F(TypecheckV2Test, ArrayEllipsisNoLeadingTypeAnnotation) {
  constexpr std::string_view kProgram = R"(fn main() -> u8[2] {
    let x: u8[2] = [u8:0, ...];
    x
})";

  XLS_EXPECT_OK(Typecheck(kProgram));
}

TEST_F(TypecheckV2Test, BadArrayAddition) {
  EXPECT_THAT(Typecheck(R"(
fn f(a: bits[32][4], b: bits[32][4]) -> bits[32][4] {
  a + b
}
)"),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Binary operations can only be applied")));
}

TEST_F(TypecheckV2Test, OneHotBadPrioType) {
  EXPECT_THAT(
      Typecheck(R"(
fn f(x: u7, prio: u2) -> u8 {
  one_hot(x, prio)
}
)"),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          AllOf(HasSubstrInV1(
                    GetParam(),
                    "Expected argument 1 to 'one_hot' to be a u1; got uN[2]"),
                HasTypeMismatchInV2(GetParam(), "u1", "u2"))));
}

TEST_F(TypecheckV2Test, OneHotSelOfSignedValues) {
  XLS_EXPECT_OK(Typecheck(R"(
fn f() -> s4 {
  let a: s4 = s4:1;
  let b: s4 = s4:2;
  let s: u2 = u2:0b01;
  one_hot_sel(s, [a, b])
}
)"));
}

TEST_F(TypecheckV2Test, OverlargeEnumValue) {
  EXPECT_THAT(
      Typecheck(R"(
enum Foo : u1 {
  A = 0,
  B = 1,
  C = 2,
}
)"),
      StatusIs(absl::StatusCode::kInvalidArgument,
               AllOf(HasSubstrInV1(
                         GetParam(),
                         "Value '2' does not fit in the bitwidth of a uN[1]"),
                     HasSizeMismatchInV2(GetParam(), "u1", "u2"))));
}

TEST_F(TypecheckV2Test, CannotAddEnums) {
  EXPECT_THAT(Typecheck(R"(
enum Foo : u2 {
  A = 0,
  B = 1,
}
fn f() -> Foo {
  Foo::A + Foo::B
}
)"),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       AllOf(HasSubstrInV1(
                                 GetParam(),
                                 "Cannot use '+' on values with enum type Foo"),
                             HasSubstrInV2(GetParam(),
                                           "Binary operations can only be "
                                           "applied to bits-typed operands"))));
}

TEST_F(TypecheckV2Test, SlicesWithMismatchedTypes) {
  EXPECT_THAT(
      Typecheck("fn f(x: u8) -> u8 { x[s4:0 : s5:1] }"),
      StatusIs(absl::StatusCode::kInvalidArgument,
               AllOf(HasSubstrInV1(GetParam(),
                                   "Slice limit type (sN[5]) did not match"),
                     HasSizeMismatchInV2(GetParam(), "s4", "s5"))));
}

TEST_F(TypecheckV2Test, SliceWithOutOfRangeLimit) {
  // Note: v1 did not like this because it took it to imply a limit of s4:128.
  // v2 does not consider the start type annotation applicable to the limit
  // value.
  XLS_EXPECT_OK(Typecheck("fn f(x: uN[128]) -> uN[128] { x[s4:0 :] }"));
  XLS_EXPECT_OK(Typecheck("fn f(x: uN[8]) -> uN[8] { x[s3:0 :] }"));
}

TEST_F(TypecheckV2Test, SliceWithNonS32LiteralBounds) {
  // overlarge value in start
  EXPECT_THAT(
      Typecheck("fn f(x: uN[128]) -> uN[128] { x[40000000000000000000:] }"),
      StatusIs(absl::StatusCode::kInvalidArgument,
               AllOf(HasSubstrInV1(
                         GetParam(),
                         "Value '40000000000000000000' does not fit in the "
                         "bitwidth of a sN[32]"),
                     HasSubstrInV2(GetParam(),
                                   "Value is too large (67 bits); at most 32 "
                                   "bits can be used here."))));
  // overlarge value in limit
  EXPECT_THAT(
      Typecheck("fn f(x: uN[128]) -> uN[128] { x[:40000000000000000000] }"),
      StatusIs(absl::StatusCode::kInvalidArgument,
               AllOf(HasSubstrInV1(
                         GetParam(),
                         "Value '40000000000000000000' does not fit in the "
                         "bitwidth of a sN[32]"),
                     HasSubstrInV2(GetParam(),
                                   "Value is too large (67 bits); at most 32 "
                                   "bits can be used here."))));
}

TEST_F(TypecheckV2Test, WidthSlices) {
  XLS_EXPECT_OK(Typecheck("fn f(x: u32) -> bits[0] { x[0+:bits[0]] }"));
  XLS_EXPECT_OK(Typecheck("fn f(x: u32) -> u1 { x[31+:u1] }"));
}

TEST_F(TypecheckV2Test, WidthSliceTypeTooLarge) {
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

TEST_F(TypecheckV2Test, WidthSliceOutOfRangeConsideringStartIndex) {
  XLS_ASSERT_OK_AND_ASSIGN(TypecheckResult result,
                           Typecheck("fn f(x: u32) -> u2 { x[32+:u2] }"));

  // This is not an error since `u2` is not larger than the type of `x`.
  ASSERT_THAT(result.tm.warnings.warnings().size(), 1);
  EXPECT_EQ(result.tm.warnings.warnings().at(0).kind,
            WarningKind::kWidthSliceOutOfRange);
}

TEST_F(TypecheckV2Test, WidthSliceNegativeStartNumberLiteral) {
  EXPECT_THAT(
      Typecheck("fn f(x: u32) -> u1 { x[-1+:u1] }"),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          AllOf(HasSubstrInV1(
                    GetParam(),
                    "only unsigned values are permitted; got start value: -1"),
                HasSignednessMismatchInV2(GetParam(), "s1", "u32"))));

  EXPECT_THAT(
      Typecheck("fn f(x: u32) -> u2 { x[-1+:u2] }"),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          AllOf(HasSubstrInV1(
                    GetParam(),
                    "only unsigned values are permitted; got start value: -1"),
                HasSignednessMismatchInV2(GetParam(), "s1", "u32"))));
  EXPECT_THAT(
      Typecheck("fn f(x: u32) -> u3 { x[-2+:u3] }"),
      AllOf(StatusIs(
          absl::StatusCode::kInvalidArgument,
          AllOf(HasSubstrInV1(
                    GetParam(),
                    "only unsigned values are permitted; got start value: -2"),
                HasSignednessMismatchInV2(GetParam(), "s2", "u32")))));
}

TEST_F(TypecheckV2Test, WidthSliceEmptyStartNumber) {
  // Start literal is treated as unsigned.
  XLS_EXPECT_OK(Typecheck("fn f(x: u32) -> u31 { x[:-1] }"));
  XLS_EXPECT_OK(Typecheck("fn f(x: u32) -> u30 { x[:-2] }"));
  XLS_EXPECT_OK(Typecheck("fn f(x: u32) -> u29 { x[:-3] }"));
}

TEST_F(TypecheckV2Test, WidthSliceUnsignedStart) {
  // Unsigned start literals are ok.
  XLS_EXPECT_OK(Typecheck("fn f(start: u32, x: u32) -> u3 { x[start+:u3] }"));
}

TEST_F(TypecheckV2Test, WidthSliceSignedStart) {
  // We reject signed start literals.
  EXPECT_THAT(
      Typecheck("fn f(start: s32, x: u32) -> u3 { x[start+:u3] }"),
      StatusIs(absl::StatusCode::kInvalidArgument,
               AllOf(HasSubstrInV1(
                         GetParam(),
                         "Start index for width-based slice must be unsigned"),
                     HasSignednessMismatchInV2(GetParam(), "s32", "u32"))));
}

TEST_F(TypecheckV2Test, WidthSliceTupleStart) {
  EXPECT_THAT(
      Typecheck("fn f(start: (s32), x: u32) -> u3 { x[start+:u3] }"),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          AllOf(HasSubstrInV1(
                    GetParam(),
                    "Start expression for width slice must be bits typed"),
                HasSubstrInV2(
                    GetParam(),
                    "Expected slice bound to be bits-typed; got `(s32,)`"))));
}

TEST_F(TypecheckV2Test, WidthSliceTupleSubject) {
  EXPECT_THAT(
      Typecheck("fn f(start: s32, x: (u32)) -> u3 { x[start+:u3] }"),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          AllOf(
              HasSubstrInV1(GetParam(), "Value to slice is not of 'bits' type"),
              HasSubstrInV2(GetParam(),
                            "Expected a bits-like type; got: `(u32,)`"))));
}

TEST_F(TypecheckV2Test, TokenOrderSimple) {
  std::string kProgram = R"(
proc TokenOrder {
  req_r: chan<u32> in;
  resp_s: chan<u32> out;

  config(
    req_r: chan<u32> in, resp_s: chan<u32> out,
  ) {
    (req_r, resp_s)
  }

  init {  }

  next (state: ()) {
    let tok0 = join();
    let (tok1, data) = recv(tok0, req_r);
    send(tok1, resp_s, data);
  }
}
)";
  XLS_ASSERT_OK_AND_ASSIGN(TypecheckResult result,
                           Typecheck(kProgram));
  EXPECT_EQ(result.tm.warnings.warnings().size(), 0);
}

TEST_F(TypecheckV2Test, TokenOrderSendIfMismatch) {
  std::string kProgram = R"(
proc TokenOrder {
  req_r: chan<u32> in;
  resp_s: chan<u32> out;

  config(
    req_r: chan<u32> in, resp_s: chan<u32> out,
  ) {
    (req_r, resp_s)
  }

  init {  }

  next (state: ()) {
    let tok0 = join();
    let (tok1, data) = recv(tok0, req_r);
    send_if(tok0, resp_s, data > 15, data);
  }
}
)";
  XLS_ASSERT_OK_AND_ASSIGN(TypecheckResult result,
                           Typecheck(kProgram));

  ASSERT_THAT(result.tm.warnings.warnings().size(), 1);
  EXPECT_EQ(result.tm.warnings.warnings().at(0).kind,
            WarningKind::kIOOrderingMismatch);
}

TEST_F(TypecheckV2Test, TokenOrderLiteral) {
  std::string kProgram = R"(
proc TokenOrder {
  req_r: chan<u32> in;
  resp_s: chan<u32> out;

  config(
    req_r: chan<u32> in, resp_s: chan<u32> out,
  ) {
    (req_r, resp_s)
  }

  init {  }

  next (state: ()) {
    let tok0 = join();
    let (tok1, _) = recv(tok0, req_r);
    send(tok0, resp_s, u32:5);
  }
}
)";
  XLS_ASSERT_OK_AND_ASSIGN(TypecheckResult result,
                           Typecheck(kProgram));
  EXPECT_EQ(result.tm.warnings.warnings().size(), 0);
}

TEST_F(TypecheckV2Test, TokenOrderSimpleMismatch) {
  std::string kProgram = R"(
proc TokenOrder {
  req_r: chan<u32> in;
  resp_s: chan<u32> out;

  config(
    req_r: chan<u32> in, resp_s: chan<u32> out,
  ) {
    (req_r, resp_s)
  }

  init {  }

  next (state: ()) {
    let tok0 = join();
    let (tok1, data) = recv(tok0, req_r);
    send(tok0, resp_s, data);
  }
}
)";
  XLS_ASSERT_OK_AND_ASSIGN(TypecheckResult result,
                           Typecheck(kProgram));

  ASSERT_THAT(result.tm.warnings.warnings().size(), 1);
  EXPECT_EQ(result.tm.warnings.warnings().at(0).kind,
            WarningKind::kIOOrderingMismatch);
}

TEST_F(TypecheckV2Test, TokenOrderState) {
  std::string kProgram = R"(
proc TokenOrder {
  req_r: chan<u32> in;
  resp_s: chan<u32> out;

  config(
    req_r: chan<u32> in, resp_s: chan<u32> out,
  ) {
    (req_r, resp_s)
  }

  init { u32:0 }

  next (state: u32) {
    let tok0 = send(join(), resp_s, state);
    let (tok1, data) = recv(tok0, req_r);
    data + state
  }
}
)";
  XLS_ASSERT_OK_AND_ASSIGN(TypecheckResult result,
                           Typecheck(kProgram));

  ASSERT_THAT(result.tm.warnings.warnings().size(), 0);
}

TEST_F(TypecheckV2Test, TokenOrderDecomposedTuple) {
  std::string kProgram = R"(
proc TokenOrder {
  req_r: chan<(u32, u32)> in;
  resp_s: chan<u32> out;

  config(
    req_r: chan<(u32, u32)> in, resp_s: chan<u32> out,
  ) {
    (req_r, resp_s)
  }

  init {  }

  next (state: ()) {
    let (tok0, (first, second)) = recv(join(), req_r);
    send(tok0, resp_s, first);
  }
}
)";
  XLS_ASSERT_OK_AND_ASSIGN(TypecheckResult result,
                           Typecheck(kProgram));

  ASSERT_THAT(result.tm.warnings.warnings().size(), 0);
}

TEST_F(TypecheckV2Test, TokenOrderDecomposedTupleMismatch) {
  std::string kProgram = R"(
proc TokenOrder {
  req_r: chan<(u32, u32)> in;
  resp_s: chan<u32> out;

  config(
    req_r: chan<(u32, u32)> in, resp_s: chan<u32> out,
  ) {
    (req_r, resp_s)
  }

  init {  }

  next (state: ()) {
    let (tok0, (first, second)) = recv(join(), req_r);
    send(join(), resp_s, first);
  }
}
)";
  XLS_ASSERT_OK_AND_ASSIGN(TypecheckResult result,
                           Typecheck(kProgram));

  ASSERT_THAT(result.tm.warnings.warnings().size(), 1);
  EXPECT_EQ(result.tm.warnings.warnings().at(0).kind,
            WarningKind::kIOOrderingMismatch);
}

TEST_F(TypecheckV2Test, TokenOrderDataAlias) {
  std::string kProgram = R"(
proc TokenOrder {
  req_r: chan<u32> in;
  resp_s: chan<u32> out;

  config(
    req_r: chan<u32> in, resp_s: chan<u32> out,
  ) {
    (req_r, resp_s)
  }

  init {  }

  next (state: ()) {
    let (tok0, data0) = recv(join(), req_r);
    let data1 = data0;
    send(tok0, resp_s, data1);
  }
}
)";
  XLS_ASSERT_OK_AND_ASSIGN(TypecheckResult result,
                           Typecheck(kProgram));

  ASSERT_THAT(result.tm.warnings.warnings().size(), 0);
}

TEST_F(TypecheckV2Test, TokenOrderDataAliasMismatch) {
  std::string kProgram = R"(
proc TokenOrder {
  req_r: chan<u32> in;
  resp_s: chan<u32> out;

  config(
    req_r: chan<u32> in, resp_s: chan<u32> out,
  ) {
    (req_r, resp_s)
  }

  init {  }

  next (state: ()) {
    let (tok0, data0) = recv(join(), req_r);
    let data1 = data0;
    send(join(), resp_s, data1);
  }
}
)";
  XLS_ASSERT_OK_AND_ASSIGN(TypecheckResult result,
                           Typecheck(kProgram));

  ASSERT_THAT(result.tm.warnings.warnings().size(), 1);
  EXPECT_EQ(result.tm.warnings.warnings().at(0).kind,
            WarningKind::kIOOrderingMismatch);
}

TEST_F(TypecheckV2Test, TokenOrderCombinedVariables) {
  std::string kProgram = R"(
proc TokenOrder {
  req0_r: chan<u32> in;
  req1_r: chan<u32> in;
  resp_s: chan<(u32, u32)> out;

  config(
    req0_r: chan<u32> in,
    req1_r: chan<u32> in,
    resp_s: chan<(u32, u32)> out
  ) {
    (req0_r, req1_r, resp_s)
  }

  init {  }

  next (state: ()) {
    let tok0 = join();
    let (tok1, data0) = recv(join(), req0_r);
    let (tok2, data1) = recv(tok1, req1_r);
    send(tok2, resp_s, (data0, data1));
  }
}
)";
  XLS_ASSERT_OK_AND_ASSIGN(TypecheckResult result,
                           Typecheck(kProgram));

  EXPECT_EQ(result.tm.warnings.warnings().size(), 0);
}

TEST_F(TypecheckV2Test, TokenOrderCombinedVariablesMismatch) {
  std::string kProgram = R"(
proc TokenOrder {
  req0_r: chan<u32> in;
  req1_r: chan<u32> in;
  resp_s: chan<(u32, u32)> out;

  config(
    req0_r: chan<u32> in,
    req1_r: chan<u32> in,
    resp_s: chan<(u32, u32)> out
  ) {
    (req0_r, req1_r, resp_s)
  }

  init {  }

  next (state: ()) {
    let tok0 = join();
    let (tok1, data0) = recv(join(), req0_r);
    let (tok2, data1) = recv(tok1, req1_r);
    send(tok1, resp_s, (data0, data1));
  }
}
)";
  XLS_ASSERT_OK_AND_ASSIGN(TypecheckResult result,
                           Typecheck(kProgram));

  ASSERT_THAT(result.tm.warnings.warnings().size(), 1);
  EXPECT_EQ(result.tm.warnings.warnings().at(0).kind,
            WarningKind::kIOOrderingMismatch);
}

TEST_F(TypecheckV2Test, TokenOrderCombinedVariablesAlias) {
  std::string kProgram = R"(
proc TokenOrder {
  req0_r: chan<u32> in;
  req1_r: chan<u32> in;
  resp_s: chan<(u32, u32)> out;

  config(
    req0_r: chan<u32> in,
    req1_r: chan<u32> in,
    resp_s: chan<(u32, u32)> out
  ) {
    (req0_r, req1_r, resp_s)
  }

  init {  }

  next (state: ()) {
    let tok0 = join();
    let (tok1, data0) = recv(join(), req0_r);
    let (tok2, data1) = recv(tok1, req1_r);
    let combined_data = (data0, data1);
    send(tok2, resp_s, combined_data);
  }
}
)";
  XLS_ASSERT_OK_AND_ASSIGN(TypecheckResult result,
                           Typecheck(kProgram));

  EXPECT_EQ(result.tm.warnings.warnings().size(), 0);
}

TEST_F(TypecheckV2Test, TokenOrderCombinedVariablesAliasMismatch) {
  std::string kProgram = R"(
proc TokenOrder {
  req0_r: chan<u32> in;
  req1_r: chan<u32> in;
  resp_s: chan<(u32, u32)> out;

  config(
    req0_r: chan<u32> in,
    req1_r: chan<u32> in,
    resp_s: chan<(u32, u32)> out
  ) {
    (req0_r, req1_r, resp_s)
  }

  init {  }

  next (state: ()) {
    let tok0 = join();
    let (tok1, data0) = recv(join(), req0_r);
    let (tok2, data1) = recv(tok1, req1_r);
    let combined_data = (data0, data1);
    send(tok1, resp_s, combined_data);
  }
}
)";
  XLS_ASSERT_OK_AND_ASSIGN(TypecheckResult result,
                           Typecheck(kProgram));

  ASSERT_THAT(result.tm.warnings.warnings().size(), 1);
  EXPECT_EQ(result.tm.warnings.warnings().at(0).kind,
            WarningKind::kIOOrderingMismatch);
}

TEST_F(TypecheckV2Test, TokenOrderBitSliceUpdate) {
  std::string kProgram = R"(
proc TokenOrder {
  req0_r: chan<u16> in;
  req1_r: chan<u16> in;
  resp_s: chan<u32> out;

  config(
    req0_r: chan<u16> in,
    req1_r: chan<u16> in,
    resp_s: chan<u32> out
  ) {
    (req0_r, req1_r, resp_s)
  }

  init {  }

  next (state: ()) {
    let tok0 = join();
    let (tok1, data0) = recv(join(), req0_r);
    let (tok2, data1) = recv(tok1, req1_r);
    let data = data0 as u32;
    let data = bit_slice_update(data, u32:16, data1);
    send(tok2, resp_s, data);
  }
}
)";
  XLS_ASSERT_OK_AND_ASSIGN(TypecheckResult result,
                           Typecheck(kProgram));

  ASSERT_THAT(result.tm.warnings.warnings().size(), 0);
}

TEST_F(TypecheckV2Test, TokenOrderBitSliceUpdateMismatch) {
  std::string kProgram = R"(
proc TokenOrder {
  req0_r: chan<u16> in;
  req1_r: chan<u16> in;
  resp_s: chan<u32> out;

  config(
    req0_r: chan<u16> in,
    req1_r: chan<u16> in,
    resp_s: chan<u32> out
  ) {
    (req0_r, req1_r, resp_s)
  }

  init {  }

  next (state: ()) {
    let tok0 = join();
    let (tok1, data0) = recv(join(), req0_r);
    let (tok2, data1) = recv(tok1, req1_r);
    let data = data0 as u32;
    let data = bit_slice_update(data, u32:16, data1);
    send(tok1, resp_s, data);
  }
}
)";
  XLS_ASSERT_OK_AND_ASSIGN(TypecheckResult result,
                           Typecheck(kProgram));

  ASSERT_THAT(result.tm.warnings.warnings().size(), 1);
  EXPECT_EQ(result.tm.warnings.warnings().at(0).kind,
            WarningKind::kIOOrderingMismatch);
}

TEST_F(TypecheckV2Test, TokenOrderNestedJoin) {
  std::string kProgram = R"(
proc TokenOrder {
  req_r: chan<u32> in;
  resp_s: chan<u32> out;

  config(
    req_r: chan<u32> in, resp_s: chan<u32> out,
  ) {
    (req_r, resp_s)
  }

  init {  }

  next (state: ()) {
    let tok0 = join();
    let tok1 = join();
    let tok2 = join();
    let (tok3, data) = recv(join(tok0, join(tok1, tok2)), req_r);
    send(tok3, resp_s, data);
  }
}
)";
  XLS_ASSERT_OK_AND_ASSIGN(TypecheckResult result,
                           Typecheck(kProgram));
  EXPECT_EQ(result.tm.warnings.warnings().size(), 0);
}

TEST_F(TypecheckV2Test, TokenOrderAlias) {
  std::string kProgram = R"(
proc TokenOrder {
  req_r: chan<u32> in;
  resp_s: chan<u32> out;

  config(
    req_r: chan<u32> in, resp_s: chan<u32> out,
  ) {
    (req_r, resp_s)
  }

  init {  }

  next (state: ()) {
    let tok0 = join();
    let tok1 = tok0;
    let (tok2, data) = recv(join(tok0, tok1), req_r);
    send(tok2, resp_s, data);
  }
}
)";
  XLS_ASSERT_OK_AND_ASSIGN(TypecheckResult result,
                           Typecheck(kProgram));
  EXPECT_EQ(result.tm.warnings.warnings().size(), 0);
}

TEST_F(TypecheckV2Test, TokenOrderAliasMismatch) {
  std::string kProgram = R"(
proc TokenOrder {
  req_r: chan<u32> in;
  resp_s: chan<u32> out;

  config(
    req_r: chan<u32> in, resp_s: chan<u32> out,
  ) {
    (req_r, resp_s)
  }

  init {  }

  next (state: ()) {
    let tok0 = join();
    let tok2 = tok0;
    // ...
    let (tok1, data) = recv(tok0, req_r);
    send(tok2, resp_s, data);
  }
}
)";
  XLS_ASSERT_OK_AND_ASSIGN(TypecheckResult result,
                           Typecheck(kProgram));

  ASSERT_THAT(result.tm.warnings.warnings().size(), 1);
  EXPECT_EQ(result.tm.warnings.warnings().at(0).kind,
            WarningKind::kIOOrderingMismatch);
}

TEST_F(TypecheckV2Test, TokenOrderJoinMixedLevels) {
  std::string kProgram = R"(
proc TokenOrder {
  req1_r: chan<u32> in;
  resp1_s: chan<u32> out;
  req2_r: chan<u32> in;
  resp2_s: chan<u32> out;

  config(
    req1_r: chan<u32> in, resp1_s: chan<u32> out,
    req2_r: chan<u32> in, resp2_s: chan<u32> out
  ) {
    (req1_r, resp1_s, req2_r, resp2_s)
  }

  init {  }

  next (state: ()) {
    let tok0 = join();
    let (tok1, data0) = recv(tok0, req1_r);
    let tok2 = send(tok1, resp1_s, data0);
    let (tok3, data1) = recv(tok2, req2_r);
    send(join(tok0, tok3), resp2_s, data1);
  }
}
)";
  XLS_ASSERT_OK_AND_ASSIGN(TypecheckResult result,
                           Typecheck(kProgram));

  ASSERT_THAT(result.tm.warnings.warnings().size(), 0);
}

TEST_F(TypecheckV2Test, TokenOrderJoinMismatch) {
  std::string kProgram = R"(
proc TokenOrder {
  req1_r: chan<u32> in;
  resp1_s: chan<u32> out;
  req2_r: chan<u32> in;
  resp2_s: chan<u32> out;

  config(
    req1_r: chan<u32> in, resp1_s: chan<u32> out,
    req2_r: chan<u32> in, resp2_s: chan<u32> out
  ) {
    (req1_r, resp1_s, req2_r, resp2_s)
  }

  init {  }

  next (state: ()) {
    let tok0 = join();
    let (tok1, data0) = recv(tok0, req1_r);
    let tok2 = send(tok1, resp1_s, data0);
    let (tok3, data1) = recv(tok2, req2_r);
    send(join(tok0, tok2), resp2_s, data1);
  }
}
)";
  XLS_ASSERT_OK_AND_ASSIGN(TypecheckResult result,
                           Typecheck(kProgram));

  ASSERT_THAT(result.tm.warnings.warnings().size(), 1);
  EXPECT_EQ(result.tm.warnings.warnings().at(0).kind,
            WarningKind::kIOOrderingMismatch);
}

TEST_F(TypecheckV2Test, TokenOrderTupleIndex) {
  std::string kProgram = R"(
proc TokenOrder {
  req_r: chan<u32> in;
  resp_s: chan<u32> out;

  config(
    req_r: chan<u32> in, resp_s: chan<u32> out,
  ) {
    (req_r, resp_s)
  }

  init {  }

  next (state: ()) {
    let tok0 = join();
    let resp = recv(tok0, req_r);
    send(resp.0, resp_s, resp.1);
  }
}
)";
  XLS_ASSERT_OK_AND_ASSIGN(TypecheckResult result,
                           Typecheck(kProgram));

  ASSERT_THAT(result.tm.warnings.warnings().size(), 0);
}

TEST_F(TypecheckV2Test, TokenOrderTupleIndexMismatch) {
  std::string kProgram = R"(
proc TokenOrder {
  req_r: chan<u32> in;
  resp_s: chan<u32> out;

  config(
    req_r: chan<u32> in, resp_s: chan<u32> out,
  ) {
    (req_r, resp_s)
  }

  init {  }

  next (state: ()) {
    let tok0 = join();
    let resp = recv(tok0, req_r);
    send(tok0, resp_s, resp.1);
  }
}
)";
  XLS_ASSERT_OK_AND_ASSIGN(TypecheckResult result,
                           Typecheck(kProgram));

  ASSERT_THAT(result.tm.warnings.warnings().size(), 1);
  EXPECT_EQ(result.tm.warnings.warnings().at(0).kind,
            WarningKind::kIOOrderingMismatch);
}

TEST_F(TypecheckV2Test, BadAttributeAccessOnTuple) {
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
                                   "type: `(u32,)`"))));
}

TEST_F(TypecheckV2Test, BadAttributeAccessOnBits) {
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

TEST_F(TypecheckV2Test, BadArrayLiteralType) {
  EXPECT_THAT(
      Typecheck("const X = s32:[1, 2];"),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          AllOf(HasSubstrInV1(GetParam(),
                              "Array was not annotated with an array type"),
                HasTypeMismatchInV2(GetParam(), "s32", "s32[2]"))));
}

TEST_F(TypecheckV2Test, CharLiteralArray) {
  XLS_EXPECT_OK(Typecheck(R"(
fn main() -> u8[3] {
  u8[3]:['X', 'L', 'S']
}
)"));
}

TEST_F(TypecheckV2Test, BadEnumRef) {
  EXPECT_THAT(Typecheck(R"(
enum MyEnum : u1 { A = 0, B = 1 }
fn f() -> MyEnum { MyEnum::C }
)"),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("name `C` in `MyEnum::C` is undefined")));
}

// Nominal typing not structural, e.g. OtherPoint cannot be passed where we want
// a Point, even though their members are the same.
TEST_F(TypecheckV2Test, NominalTyping) {
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

TEST_F(TypecheckV2Test, ParametricWithConstantArrayEllipsis) {
  XLS_EXPECT_OK(Typecheck(R"(
fn p<N: u32>(_: bits[N]) -> u8[2] { u8[2]:[0, ...] }
fn main() -> u8[2] { p(false) }
)"));
}

// In this test case we:
// * make a call to `q`, where we give it an explicit parametric value,
// * by invoking `r` (which is also parametric),
// * doing that from within a parametric function `p`.
TEST_F(TypecheckV2Test, ExplicitParametricCallInParametricFn) {
  XLS_EXPECT_OK(Typecheck(R"(
fn r<R: u32>(x: bits[R]) -> bits[R] { x }
fn q<Q: u32>(x: bits[Q]) -> bits[Q] { x }
fn p<P: u32>(x: bits[P]) -> bits[P] { q<{r(P+u32:0)}>(x) }
fn main() -> u32 { p(u32:42) }
)"));
}

TEST_F(TypecheckV2Test, BadQuickcheckFunctionRet) {
  EXPECT_THAT(
      Typecheck(R"(
#[quickcheck]
fn f() -> u5 { u5:1 }
)"),
      StatusIs(absl::StatusCode::kInvalidArgument,
               AllOf(HasSubstrInV1(GetParam(), "must return a bool"),
                     HasTypeMismatchInV2(GetParam(), "uN[1]", "uN[5]"))));
}

// TODO: in v2 this fails with a CHECK error.
TEST_F(TypecheckV2Test, BadQuickcheckFunctionParametrics) {
  EXPECT_THAT(
      Typecheck(R"(
#[quickcheck]
fn f<N: u32>() -> bool { true }
)"),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("Quickchecking parametric functions is unsupported")));
}

TEST_F(TypecheckV2Test, NumbersAreConstexpr) {
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

TEST_F(TypecheckV2Test, BasicTupleIndex) {
  XLS_EXPECT_OK(Typecheck(R"(
fn main() -> u18 {
  (u32:7, u24:6, u18:5, u12:4, u8:3).2
}
)"));
}

TEST_F(TypecheckV2Test, DuplicateRestOfTupleError) {
  EXPECT_THAT(Typecheck(R"(
fn main() {
  let (x, .., ..) = (u32:7, u24:6, u18:5, u12:4, u8:3);
}
)"),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("can only be used once")));
}

TEST_F(TypecheckV2Test, TupleCountMismatch) {
  EXPECT_THAT(Typecheck(R"(
fn main() {
  let (x, y) = (u32:7, u24:6, u18:5, u12:4, u8:3);
}
)"),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("a 5-element tuple to 2 values")));
}

TEST_F(TypecheckV2Test, RestOfTupleCountMismatch) {
  EXPECT_THAT(Typecheck(R"(
fn main() {
  let (x, .., y, z) = (u32:7, u8:3);
}
)"),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("a 2-element tuple to 3 values")));
}

TEST_F(TypecheckV2Test, RestOfTupleCountMismatchNested) {
  EXPECT_THAT(Typecheck(R"(
fn main() {
  let (x, .., (y, .., z)) = (u32:7, u8:3, (u12:4,));
}
)"),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("a 1-element tuple to 2 values")));
}

TEST_F(TypecheckV2Test, TupleAssignsTypes) {
  constexpr std::string_view kProgram = R"(
fn main() {
  let (x, y): (u32, s8) = (u32:7, s8:3);
}
)";
  XLS_EXPECT_OK(Typecheck(kProgram));
}

TEST_F(TypecheckV2Test, RestOfTupleSkipsMiddle) {
  constexpr std::string_view kProgram = R"(
fn main() {
  let (x, .., y) = (u32:7, u12:4, s8:3);
  let (xx, yy): (u32, s8) = (x, y);
}
)";
  XLS_EXPECT_OK(Typecheck(kProgram));
}

TEST_F(TypecheckV2Test, RestOfTupleSkipsNone) {
  constexpr std::string_view kProgram = R"(
fn main() {
  let (x, .., y) = (u32:7, s8:3);
  let (xx, yy): (u32, s8) = (x, y);
}
)";
  XLS_EXPECT_OK(Typecheck(kProgram));
}

TEST_F(TypecheckV2Test, RestOfTuplekSkipsNoneWithThree) {
  constexpr std::string_view kProgram = R"(
fn main() {
  let (x, y, .., z) = (u32:7, u12:4, s8:3);
  let (xx, yy, zz): (u32, u12, s8) = (x, y, z);
}
)";
  XLS_EXPECT_OK(Typecheck(kProgram));
}

TEST_F(TypecheckV2Test, RestOfTupleSkipsEnd) {
  constexpr std::string_view kProgram = R"(
fn main() {
  let (x, y, ..) = (u32:7, s8:3, u12:4);
  let (xx, yy): (u32, s8) = (x, y);
}
)";
  XLS_EXPECT_OK(Typecheck(kProgram));
}

TEST_F(TypecheckV2Test, RestOfTupleSkipsManyAtEnd) {
  constexpr std::string_view kProgram = R"(
fn main() {
  let (x, y, ..) = (u32:7, s8:3, u12:4, u32:0);
  let (xx, yy): (u32, s8) = (x, y);
}
)";
  XLS_EXPECT_OK(Typecheck(kProgram));
}

TEST_F(TypecheckV2Test, RestOfTupleSkipsManyInMiddle) {
  constexpr std::string_view kProgram = R"(
fn main() {
  let (x, .., y) = (u32:7, u8:3, u12:4, s8:3);
  let (xx, yy): (u32, s8) = (x, y);
}
)";
  XLS_EXPECT_OK(Typecheck(kProgram));
}

TEST_F(TypecheckV2Test, RestOfTupleSkipsBeginning) {
  constexpr std::string_view kProgram = R"(
fn main() {
  let (.., x, y) = (u12:7, u8:3, u32:4, s8:3);
  let (xx, yy): (u32, s8) = (x, y);
}
)";
  XLS_EXPECT_OK(Typecheck(kProgram));
}

TEST_F(TypecheckV2Test, RestOfTupleSkipsManyAtBeginning) {
  constexpr std::string_view kProgram = R"(
fn main() {
  let (.., x) = (u8:3, u12:4, u32:7);
  let xx: u32 = x;
}
)";
  XLS_EXPECT_OK(Typecheck(kProgram));
}

TEST_F(TypecheckV2Test, RestOfTupleNested) {
  constexpr std::string_view kProgram = R"(
fn main() {
  let (x, .., (.., y)) = (u32:7, u8:3, u18:5, (u12:4, u11:5, s8:3));
  let (xx, yy): (u32, s8) = (x, y);
}
)";
  XLS_EXPECT_OK(Typecheck(kProgram));
}

TEST_F(TypecheckV2Test, RestOfTupleNestedSingleton) {
  constexpr std::string_view kProgram = R"(
fn main() {
  let (x, .., (y,)) = (u32:7, u8:3, (s8:3,));
  let (xx, yy): (u32, s8) = (x, y);
}
)";
  XLS_EXPECT_OK(Typecheck(kProgram));
}

TEST_F(TypecheckV2Test, RestOfTupleIsLikeWildcard) {
  constexpr std::string_view kProgram = R"(
fn main() {
  let (x, .., (.., y)) = (u32:7, u18:5, (u12:4, s8:3));
  let (xx, yy): (u32, s8) = (x, y);
}
)";
  XLS_EXPECT_OK(Typecheck(kProgram));
}

TEST_F(TypecheckV2Test, RestOfTupleDeeplyNested) {
  constexpr std::string_view kProgram = R"(
fn main() {
  let (x, y, .., ((.., z), .., d)) = (u32:7, u8:1,
                            ((u32:3, u64:4, uN[128]:5), u12:4, s8:3));
  let (xx, yy, zz): (u32, u8, uN[128]) = (x, y, z);
  }
)";
  XLS_EXPECT_OK(Typecheck(kProgram));
}

TEST_F(TypecheckV2Test, RestOfTupleDeeplyNestedNonConstants) {
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

TEST_F(TypecheckV2Test, BasicRange) {
  constexpr std::string_view kProgram = R"(#[test]
fn main() {
  let a = u32:0..u32:4;
  let b = u32[4]:[0, 1, 2, 3];
  assert_eq(a, b)
}
)";

  XLS_EXPECT_OK(Typecheck(kProgram));
}

TEST_F(TypecheckV2Test, RangeInclusive) {
  constexpr std::string_view kProgram = R"(
const a:u32[5] = u32:0..=u32:4;
const_assert!(a == u32[5]:[0, 1, 2, 3, 4]);
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
  return TypecheckV2(program).status();
}

TEST_F(TypecheckV2Test, AccessMissingStructMember) {
  EXPECT_THAT(
      TypecheckStructInstance(GetParam(), "fn f(p: Point) -> () { p.z }"),
      StatusIs(absl::StatusCode::kInvalidArgument,
               AllOf(HasSubstrInV1(
                         GetParam(),
                         "Struct 'Point' does not have a member with name 'z'"),
                     HasSubstrInV2(GetParam(),
                                   "No member `z` in struct `Point`."))));
}

TEST_F(TypecheckV2Test, WrongTypeInStructInstanceMember) {
  EXPECT_THAT(
      TypecheckStructInstance(
          GetParam(), "fn f() -> Point { Point { y: u8:42, x: s8:-1 } }"),
      StatusIs(absl::StatusCode::kInvalidArgument,
               AllOf(HasSizeMismatchInV1(GetParam(), "uN[32]", "uN[8]"),
                     HasSizeMismatchInV2(GetParam(), "u32", "u8"))));
}

TEST_F(TypecheckV2Test, MissingFieldXInStructInstance) {
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

TEST_F(TypecheckV2Test, MissingFieldYInStructInstance) {
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

TEST_F(TypecheckV2Test, OutOfOrderStructInstanceOk) {
  XLS_EXPECT_OK(TypecheckStructInstance(
      GetParam(), "fn f() -> Point { Point { y: u32:42, x: s8:-1 } }"));
}

TEST_F(TypecheckV2Test, ProvideExtraFieldZInStructInstance) {
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

TEST_F(TypecheckV2Test, DuplicateFieldYInStructInstance) {
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

TEST_F(TypecheckV2Test, StructIncompatibleWithTupleEquivalent) {
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

TEST_F(TypecheckV2Test, SplatWithDuplicate) {
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

TEST_F(TypecheckV2Test, SplatWithExtraFieldQ) {
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

TEST_F(TypecheckV2Test, MulExprInStructMember) {
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

TEST_F(TypecheckV2Test, NonU32Parametric) {
  const std::string_view kProgram = R"(
struct Point<N: u5, N_U32: u32 = {N as u32}> {
  x: uN[N_U32],
}

fn f(p: Point<u5:3>) -> uN[3] {
  p.x
}
)";

  XLS_EXPECT_OK(Typecheck(kProgram));
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
  return TypecheckV2(program).status();
}

TEST_F(TypecheckV2Test, TooManyParametricStructArgs) {
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

TEST_F(TypecheckV2Test, PhantomParametricStructReturnTypeMismatch) {
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

TEST_F(TypecheckV2Test, PhantomParametricParameterizedStructReturnType) {
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

TEST_F(TypecheckV2Test, PhantomParametricWithExpr) {
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

TEST_F(TypecheckV2Test, OutOfOrderOk) {
  XLS_EXPECT_OK(TypecheckParametricStructInstance(
      GetParam(),
      "fn f() -> Point<32, 64> { Point { y: u64:42, x: u32:255 } }"));
}

TEST_F(TypecheckV2Test, OkInstantiationInParametricFunction) {
  XLS_EXPECT_OK(TypecheckParametricStructInstance(GetParam(), R"(
fn f<A: u32, B: u32>(x: bits[A], y: bits[B]) -> Point<A, B> { Point { x, y } }
fn main() {
  let _ = f(u5:1, u10:2);
  let _ = f(u14:1, u28:2);
  ()
}
)"));
}

TEST_F(TypecheckV2Test, BadParametricStructReturnType) {
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

TEST_F(TypecheckV2Test, BadParametricSplatInstantiation) {
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

TEST_F(TypecheckV2Test, AttrViaColonRef) {
  XLS_EXPECT_OK(Typecheck("fn f() -> u8 { u8::ZERO }"));
  XLS_EXPECT_OK(Typecheck("fn f() -> u8 { u8::MAX }"));
  XLS_EXPECT_OK(Typecheck("fn f() -> u8 { u8::MIN }"));
}

TEST_F(TypecheckV2Test, ColonRefTypeAlias) {
  XLS_EXPECT_OK(Typecheck(R"(
type MyU8 = u8;
fn f() -> u8 { MyU8::MAX }
fn g() -> u8 { MyU8::ZERO }
fn h() -> u8 { MyU8::MIN }
)"));
}

TEST_F(TypecheckV2Test, MinAttrUsedInConstAsserts) {
  XLS_EXPECT_OK(Typecheck(R"(
const_assert!(u8::MIN == u8:0);
const_assert!(s4::MIN == s4:-8);
)"));
}

TEST_F(TypecheckV2Test, MaxAttrUsedToDefineAType) {
  XLS_EXPECT_OK(Typecheck(R"(
type MyU255 = uN[u8::MAX as u32];
fn f() -> MyU255 { uN[255]:42 }
)"));
}

TEST_F(TypecheckV2Test, ZeroAttrUsedToDefineAType) {
  XLS_EXPECT_OK(Typecheck(R"(
type MyU0 = uN[u8::ZERO as u32];
fn f() -> MyU0 { bits[0]:0 }
)"));
}

TEST_F(TypecheckV2Test, TypeAliasOfStructWithBoundParametrics) {
  XLS_EXPECT_OK(Typecheck(R"(
struct S<X: u32, Y: u32> {
  x: bits[X],
  y: bits[Y],
}
type MyS = S<3, 4>;
fn f() -> MyS { MyS{x: bits[3]:3, y: bits[4]:4 } }
)"));
}

TEST_F(TypecheckV2Test, SplatWithAllStructMembersSpecifiedGivesWarning) {
  const std::string program = R"(
struct S {
  x: u32,
  y: u32,
}
fn f(s: S) -> S { S{x: u32:4, y: u32:8, ..s} }
)";
  ImportData import_data = CreateImportDataForTest();
  XLS_ASSERT_OK_AND_ASSIGN(TypecheckResult tr,
                           Typecheck(program, "fake", &import_data));
  TypecheckedModule& tm = tr.tm;
  Fileno fileno = import_data.file_table().GetOrCreate("fake.x");

  ASSERT_THAT(tm.warnings.warnings().size(), 1);
  EXPECT_EQ(tm.warnings.warnings().at(0).span,
            Span(Pos(fileno, 7, 42), Pos(fileno, 7, 43)));
  EXPECT_EQ(tm.warnings.warnings().at(0).message,
            "'Splatted' struct instance has all members of struct defined, "
            "consider removing the `..s`");
  UniformContentFilesystem vfs(program);
  XLS_ASSERT_OK(PrintPositionalError(
      {tm.warnings.warnings().at(0).span}, tm.warnings.warnings().at(0).message,
      std::cerr, PositionalErrorColor::kWarningColor, import_data.file_table(),
      vfs));
}

TEST_F(TypecheckV2Test, LetWithWildcardMatchGivesWarning) {
  const std::string program = R"(
fn f(x: u32) -> u32 {
  let _ = x + x;
  x
}
)";
  ImportData import_data = CreateImportDataForTest();
  XLS_ASSERT_OK_AND_ASSIGN(TypecheckResult tr,
                           Typecheck(program, "fake", &import_data));
  TypecheckedModule& tm = tr.tm;
  FileTable& file_table = import_data.file_table();
  Fileno fileno = file_table.GetOrCreate("fake.x");

  ASSERT_THAT(tm.warnings.warnings().size(), 1);
  EXPECT_EQ(tm.warnings.warnings().at(0).span,
            Span(Pos(fileno, 4, 6), Pos(fileno, 4, 7)));
  EXPECT_EQ(tm.warnings.warnings().at(0).message,
            "`let _ = expr;` statement can be simplified to `expr;` -- there "
            "is no need for a `let` binding here");
  UniformContentFilesystem vfs(program);
  XLS_ASSERT_OK(PrintPositionalError(
      {tm.warnings.warnings().at(0).span}, tm.warnings.warnings().at(0).message,
      std::cerr, PositionalErrorColor::kWarningColor, file_table, vfs,
      /*error_context_line_count=*/5));
}

TEST_F(TypecheckV2Test, UselessTrailingNilGivesWarning) {
  const std::string program = R"(
fn f() -> () {
  trace_fmt!("oh no");
  ()
}
)";
  ImportData import_data = CreateImportDataForTest();
  XLS_ASSERT_OK_AND_ASSIGN(TypecheckResult tr,
                           Typecheck(program, "fake", &import_data));
  TypecheckedModule& tm = tr.tm;
  FileTable& file_table = import_data.file_table();
  Fileno fileno = file_table.GetOrCreate("fake.x");

  ASSERT_THAT(tm.warnings.warnings().size(), 1);
  EXPECT_EQ(tm.warnings.warnings().at(0).span,
            Span(Pos(fileno, 5, 2), Pos(fileno, 5, 4)));
  EXPECT_EQ(tm.warnings.warnings().at(0).message,
            "Block has a trailing nil (empty) tuple after a semicolon -- this "
            "is implied, please remove it");
  UniformContentFilesystem vfs(program);
  XLS_ASSERT_OK(PrintPositionalError(
      {tm.warnings.warnings().at(0).span}, tm.warnings.warnings().at(0).message,
      std::cerr, PositionalErrorColor::kWarningColor, file_table, vfs,
      /*error_context_line_count=*/5));
}

TEST_F(TypecheckV2Test, NonstandardConstantNamingGivesWarning) {
  const constexpr std::string_view kProgram = R"(const mol = u32:42;)";
  XLS_ASSERT_OK_AND_ASSIGN(TypecheckResult tr, Typecheck(kProgram));
  TypecheckedModule& tm = tr.tm;
  ASSERT_THAT(tm.warnings.warnings().size(), 1);
  EXPECT_EQ(tm.warnings.warnings().at(0).message,
            "Standard style is SCREAMING_SNAKE_CASE for constant identifiers; "
            "got: `mol`");
}

TEST_F(TypecheckV2Test, NonstandardConstantNamingOkViaAllow) {
  const constexpr std::string_view kProgram =
      R"(#![allow(nonstandard_constant_naming)]
const mol = u32:42;)";
  XLS_ASSERT_OK_AND_ASSIGN(TypecheckResult tr, Typecheck(kProgram));
  TypecheckedModule& tm = tr.tm;
  ASSERT_TRUE(tm.warnings.warnings().empty());
}

TEST_F(TypecheckV2Test, BadTraceFmtWithUseOfChannel) {
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

TEST_F(TypecheckV2Test, BadTraceFmtWithUseOfFunction) {
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

TEST_F(TypecheckV2Test, CatchesBadInvocationCallee) {
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
TEST_F(TypecheckV2Test, ProcWithImportedEnumParametricGithubIssue1540) {
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
TEST_F(TypecheckV2Test, ImportedTypeAliasAttributeGithubIssue1540) {
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

TEST_F(TypecheckV2Test, MissingWideningCastFromValueError) {
  constexpr std::string_view kProgram = R"(
fn main(x: u32) -> u64 {
  widening_cast<u64>()
}
)";

  EXPECT_THAT(
      Typecheck(kProgram),
      StatusIs(absl::StatusCode::kInvalidArgument,
               AllOf(HasSubstrInV1(GetParam(),
                                   "Invalid number of arguments passed to"),
                     HasSubstrInV2(GetParam(),
                                   "Expected 1 argument(s) but got 0"))));
}

TEST_F(TypecheckV2Test, MissingCheckedCastFromValueError) {
  constexpr std::string_view kProgram = R"(
fn main(x: u32) -> u64 {
  checked_cast<u64>()
}
)";

  EXPECT_THAT(
      Typecheck(kProgram),
      StatusIs(absl::StatusCode::kInvalidArgument,
               AllOf(HasSubstrInV1(GetParam(),
                                   "Invalid number of arguments passed to"),
                     HasSubstrInV2(GetParam(),
                                   "Expected 1 argument(s) but got 0"))));
}

TEST_F(TypecheckV2Test, MissingWideningCastToTypeError) {
  constexpr std::string_view kProgram = R"(
fn main(x: u32) -> u64 {
  widening_cast(x)
}
)";

  EXPECT_THAT(
      Typecheck(kProgram),
      StatusIs(absl::StatusCode::kInvalidArgument,
               AllOf(HasSubstrInV1(GetParam(),
                                   "Invalid number of parametrics passed to"),
                     HasSubstrInV2(GetParam(),
                                   "Could not infer parametric(s): DEST"))));
}

TEST_F(TypecheckV2Test, MissingCheckedCastToTypeError) {
  constexpr std::string_view kProgram = R"(
fn main(x: u32) -> u64 {
  checked_cast(x)
}
)";

  EXPECT_THAT(
      Typecheck(kProgram),
      StatusIs(absl::StatusCode::kInvalidArgument,
               AllOf(HasSubstrInV1(GetParam(),
                                   "Invalid number of parametrics passed to"),
                     HasSubstrInV2(GetParam(),
                                   "Could not infer parametric(s): DEST"))));
}

TEST_F(TypecheckV2Test, WideningCastToSmallerUnError) {
  constexpr std::string_view kProgram = R"(
fn main() {
  widening_cast<u33>(u32:0);
  widening_cast<u32>(u32:0);
  widening_cast<u31>(u32:0);
}
)";

  EXPECT_THAT(
      Typecheck(kProgram),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          AllOf(HasSubstrInV1(GetParam(),
                              "Can not cast from type uN[32] (32 bits) to "
                              "uN[31] (31 bits) with widening_cast"),
                HasSubstrInV2(GetParam(),
                              "Cannot cast from type `uN[32]` (32 bits) to "
                              "`uN[31]` (31 bits) with widening_cast"))));
}

TEST_F(TypecheckV2Test, WideningCastToSmallerSnError) {
  constexpr std::string_view kProgram = R"(
fn main() {
  widening_cast<s33>(s32:0);
  widening_cast<s32>(s32:0);
  widening_cast<s31>(s32:0);
}
)";

  EXPECT_THAT(
      Typecheck(kProgram),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          AllOf(HasSubstrInV1(GetParam(),
                              "Can not cast from type sN[32] (32 bits) to "
                              "sN[31] (31 bits) with widening_cast"),
                HasSubstrInV2(GetParam(),
                              "Cannot cast from type `sN[32]` (32 bits) to "
                              "`sN[31]` (31 bits) with widening_cast"))));
}

TEST_F(TypecheckV2Test, WideningCastToUnError) {
  constexpr std::string_view kProgram = R"(
fn main() {
  widening_cast<u4>(u3:0);
  widening_cast<u4>(u4:0);
  widening_cast<u4>(s1:0);
}
)";

  EXPECT_THAT(
      Typecheck(kProgram),
      StatusIs(absl::StatusCode::kInvalidArgument,
               AllOf(HasSubstrInV1(GetParam(),
                                   "Can not cast from type sN[1] (1 bits) to "
                                   "uN[4] (4 bits) with widening_cast"),
                     HasSubstrInV2(GetParam(),
                                   "Cannot cast from type `sN[1]` (1 bits) to "
                                   "`uN[4]` (4 bits) with widening_cast"))));
}

TEST_F(TypecheckV2Test, WideningCastsUnError2) {
  constexpr std::string_view kProgram =
      R"(
fn main(x: u8) -> u32 {
  let x_32 = widening_cast<u32>(x);
  let x_4  = widening_cast<u4>(x_32);
  x_32 + widening_cast<u32>(x_4)
}
)";
  EXPECT_THAT(
      Typecheck(kProgram),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          AllOf(HasSubstrInV1(GetParam(),
                              "Can not cast from type uN[32] (32 bits) to "
                              "uN[4] (4 bits) with widening_cast"),
                HasSubstrInV2(GetParam(),
                              "Cannot cast from type `uN[32]` (32 bits) to "
                              "`uN[4]` (4 bits) with widening_cast"))));
}

TEST_F(TypecheckV2Test, WideningCastToSnError1) {
  constexpr std::string_view kProgram = R"(
fn main() {
  widening_cast<s4>(u3:0);
  widening_cast<s4>(s4:0);
  widening_cast<s4>(u4:0);
}
)";

  EXPECT_THAT(
      Typecheck(kProgram),
      StatusIs(absl::StatusCode::kInvalidArgument,
               AllOf(HasSubstrInV1(GetParam(),
                                   "Can not cast from type uN[4] (4 bits) to "
                                   "sN[4] (4 bits) with widening_cast"),
                     HasSubstrInV2(GetParam(),
                                   "Cannot cast from type `uN[4]` (4 bits) to "
                                   "`sN[4]` (4 bits) with widening_cast"))));
}

TEST_F(TypecheckV2Test, WideningCastsSnError2) {
  constexpr std::string_view kProgram =
      R"(
fn main(x: s8) -> s32 {
  let x_32 = widening_cast<s32>(x);
  let x_4  = widening_cast<s4>(x_32);
  x_32 + widening_cast<s32>(x_4)
}
)";
  EXPECT_THAT(
      Typecheck(kProgram),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          AllOf(HasSubstrInV1(GetParam(),
                              "Can not cast from type sN[32] (32 bits) to "
                              "sN[4] (4 bits) with widening_cast"),
                HasSubstrInV2(GetParam(),
                              "Cannot cast from type `sN[32]` (32 bits) to "
                              "`sN[4]` (4 bits) with widening_cast"))));
}

TEST_F(TypecheckV2Test, WideningCastsUnToSnError) {
  constexpr std::string_view kProgram =
      R"(
fn main(x: u8) -> s32 {
  let x_9 = widening_cast<s9>(x);
  let x_8 = widening_cast<s8>(x);
  checked_cast<s32>(x_9) + checked_cast<s32>(x_8)
}
)";
  EXPECT_THAT(
      Typecheck(kProgram),
      StatusIs(absl::StatusCode::kInvalidArgument,
               AllOf(HasSubstrInV1(GetParam(),
                                   "Can not cast from type uN[8] (8 bits) to "
                                   "sN[8] (8 bits) with widening_cast"),
                     HasSubstrInV2(GetParam(),
                                   "Cannot cast from type `uN[8]` (8 bits) to "
                                   "`sN[8]` (8 bits) with widening_cast"))));
}

TEST_F(TypecheckV2Test, WideningCastsSnToUnError) {
  constexpr std::string_view kProgram =
      R"(
fn main(x: s8) -> s32 {
  let x_9 = widening_cast<u9>(x);
  checked_cast<s32>(x_9)
}
)";
  EXPECT_THAT(
      Typecheck(kProgram),
      StatusIs(absl::StatusCode::kInvalidArgument,
               AllOf(HasSubstrInV1(GetParam(),
                                   "Can not cast from type sN[8] (8 bits) to "
                                   "uN[9] (9 bits) with widening_cast"),
                     HasSubstrInV2(GetParam(),
                                   "Cannot cast from type `sN[8]` (8 bits) to "
                                   "`uN[9]` (9 bits) with widening_cast"))));
}

TEST_F(TypecheckV2Test, OverlargeValue80Bits) {
  constexpr std::string_view kProgram =
      R"(
fn f() {
  let x:sN[0] = sN[80]:0x800000000000000000000;
}
)";
  EXPECT_THAT(
      Typecheck(kProgram),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          AllOf(HasSubstrInV1(GetParam(),
                              "Value '0x800000000000000000000' does not fit "
                              "in the bitwidth of a sN[80] (80)"),
                HasSizeMismatchInV2(GetParam(), "sN[0]", "sN[80]"))));
}

TEST_F(TypecheckV2Test, NegateTuple) {
  constexpr std::string_view kProgram =
      R"(
fn f() -> (u32, u32) {
  -(u32:42, u32:64)
}
)";
  EXPECT_THAT(
      Typecheck(kProgram),
      StatusIs(absl::StatusCode::kInvalidArgument,
               AllOf(HasSubstrInV1(GetParam(),
                                   "Unary operation `-` can only be applied to "
                                   "bits-typed operands"),
                     HasSubstrInV2(GetParam(),
                                   "Unary operations can only be applied to "
                                   "bits-typed operands"))));
}

TEST_F(TypecheckV2Test, MatchOnBitsWithEmptyTuplePattern) {
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

TEST_F(TypecheckV2Test, MatchOnBitsWithIrrefutableTuplePattern) {
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

TEST_F(TypecheckV2Test, MatchOnTupleWithWrongSizedTuplePattern) {
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
                HasSubstrInV2(GetParam(),
                              "Cannot match a 2-element tuple to 1 values."))));
}

TEST_F(TypecheckV2Test, MatchOnTupleWithRestOfTupleSkipsEnd) {
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

TEST_F(TypecheckV2Test, MatchOnTupleWithRestOfTupleSkipsBeginning) {
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

TEST_F(TypecheckV2Test, MatchOnTupleWithRestOfTupleSkipsBeginningThenMatches) {
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

TEST_F(TypecheckV2Test, MatchOnTupleWithRestOfTupleSkipsMiddle) {
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

TEST_F(TypecheckV2Test, MatchOnTupleWithRestOfTupleSkipsNone) {
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

TEST_F(TypecheckV2Test, UnusedBindingInBodyGivesWarning) {
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

TEST_F(TypecheckV2Test, FiveUnusedBindingsInLetBindingPattern) {
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

TEST_F(TypecheckV2Test, UnusedMatchBindingInBodyGivesWarning) {
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

TEST_F(TypecheckV2Test, ConcatU1U1) {
  XLS_ASSERT_OK(Typecheck("fn f(x: u1, y: u1) -> u2 { x ++ y }"));
}

TEST_F(TypecheckV2Test, ConcatU1S1) {
  EXPECT_THAT(Typecheck("fn f(x: u1, y: s1) -> u2 { x ++ y }").status(),
              IsPosError("TypeInferenceError",
                         HasSubstr("Concatenation requires operand "
                                   "types to both be unsigned bits")));
}

TEST_F(TypecheckV2Test, ConcatS1S1) {
  EXPECT_THAT(Typecheck("fn f(x: s1, y: s1) -> u2 { x ++ y }").status(),
              IsPosError("TypeInferenceError",
                         HasSubstr("Concatenation requires operand "
                                   "types to both be unsigned bits")));
}

TEST_F(TypecheckV2Test, ConcatU2S1) {
  EXPECT_THAT(Typecheck("fn f(x: u2, y: s1) -> u3 { x ++ y }").status(),
              IsPosError("TypeInferenceError",
                         HasSubstr("Concatenation requires operand "
                                   "types to both be unsigned bits")));
}

TEST_F(TypecheckV2Test, ConcatU1Nil) {
  EXPECT_THAT(Typecheck("fn f(x: u1, y: ()) -> () { x ++ y }").status(),
              IsPosError("TypeInferenceError",
                         HasSubstr("Concatenation requires operand types "
                                   "to be either both-arrays or both-bits")));
}

TEST_F(TypecheckV2Test, ConcatS1Nil) {
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

TEST_F(TypecheckV2Test, ConcatNilNil) {
  EXPECT_THAT(Typecheck("fn f(x: (), y: ()) -> () { x ++ y }").status(),
              IsPosError("TypeInferenceError",
                         HasSubstr("Concatenation requires operand types to "
                                   "be either both-arrays or both-bits")));
}

TEST_F(TypecheckV2Test, ConcatEnumU2) {
  EXPECT_THAT(
      Typecheck(R"(
enum MyEnum : u2 {
  A = 1,
  B = 2,
}
fn f(x: MyEnum, y: u2) -> () { x ++ y }
)")
          .status(),
      IsPosError("TypeInferenceError",
                 HasSubstr("MyEnum Concatenation requires operand types to be "
                           "either both-arrays or both-bits; got: MyEnum")));
}

TEST_F(TypecheckV2Test, ConcatU2Enum) {
  EXPECT_THAT(
      Typecheck(R"(
enum MyEnum : u2 {
  A = 1,
  B = 2,
}
fn f(x: u2, y: MyEnum) -> () { x ++ y }
)")
          .status(),
      IsPosError("TypeInferenceError",
                 HasSubstr("Concatenation requires operand types to be either "
                           "both-arrays or both-bits; got: MyEnum")));
}

TEST_F(TypecheckV2Test, ConcatEnumEnum) {
  EXPECT_THAT(
      Typecheck(R"(
enum MyEnum : u2 {
  A = 1,
  B = 2,
}
fn f(x: MyEnum, y: MyEnum) -> () { x ++ y }
)")
          .status(),
      IsPosError("TypeInferenceError",
                 HasSubstr("MyEnum Concatenation requires operand types to be "
                           "either both-arrays or both-bits; got: MyEnum")));
}

TEST_F(TypecheckV2Test, ConcatStructStruct) {
  EXPECT_THAT(Typecheck(R"(
struct S {}
fn f(x: S, y: S) -> () { x ++ y }
)")
                  .status(),
              IsPosError("TypeInferenceError",
                         HasSubstr("Concatenation requires operand types to be "
                                   "either both-arrays or both-bits")));
}

TEST_F(TypecheckV2Test, ConcatUnWithXn) {
  XLS_ASSERT_OK(Typecheck(R"(
fn f(x: u32, y: xN[false][32]) -> xN[false][64] { x ++ y }
)"));
}

TEST_F(TypecheckV2Test, ConcatU1ArrayOfOneU8) {
  EXPECT_THAT(
      Typecheck("fn f(x: u1, y: u8[1]) -> u2 { x ++ y }").status(),
      IsPosError(
          "TypeInferenceError",
          HasSubstr(
              "Attempting to concatenate array/non-array values together")));
}

TEST_F(TypecheckV2Test, ConcatArrayOfThreeU8ArrayOfOneU8) {
  XLS_ASSERT_OK(Typecheck("fn f(x: u8[3], y: u8[1]) -> u8[4] { x ++ y }"));
}

TEST_F(TypecheckV2Test, ParametricWrapperAroundBuiltin) {
  XLS_ASSERT_OK(Typecheck(R"(fn f<N: u32>(x: uN[N]) -> uN[N] { rev(x) }

fn main(arg: u32) -> u32 {
  f(arg)
})"));
}

TEST_F(TypecheckV2Test, AssertBuiltinIsUnitType) {
  XLS_ASSERT_OK(Typecheck(R"(fn main() {
  assert!(true, "oh_no");
})"));

  XLS_ASSERT_OK(Typecheck(R"(fn main() {
  let () = assert!(true, "oh_no");
})"));
}

TEST_F(TypecheckV2Test, ConcatNilArrayOfOneU8) {
  EXPECT_THAT(
      Typecheck("fn f(x: (), y: u8[1]) -> () { x ++ y }").status(),
      IsPosError("TypeInferenceError",
                 HasSubstr("Concatenation requires operand types to be either "
                           "both-arrays or both-bits; got: ()")));
}

TEST_F(TypecheckV2Test,
       ParametricStructWithoutAllParametricsBoundInReturnType) {
  EXPECT_THAT(Typecheck(R"(
struct Point1D<N: u32> { x: bits[N] }

fn f(x: Point1D) -> Point1D { x }
)")
                  .status(),
              IsPosError("TypeInferenceError",
                         HasSubstr("Could not infer parametric(s) for instance "
                                   "of struct Point1D: N")));
}

// See https://github.com/google/xls/issues/1030
TEST_F(TypecheckV2Test, InstantiateImportedParametricStruct) {
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

TEST_F(TypecheckV2Test, InstantiateImportedParametricStructNoParametrics) {
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

TEST_F(TypecheckV2Test, InstantiateImportedParametricStructTypeAlias) {
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

TEST_F(TypecheckV2Test, InstantiateImportedParametricStructArray) {
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

TEST_F(TypecheckV2Test, InstantiateParametricStructArray) {
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

TEST_F(TypecheckV2Test, CallImportedParametricFn) {
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

TEST_F(TypecheckV2Test, PrioritySelectOnNonBitsType) {
  EXPECT_THAT(Typecheck(R"(
struct MyStruct { }

fn f() {
let default_value = MyStruct{};
priority_sel(u2:0b00, MyStruct[2]:[MyStruct{}, MyStruct{}], default_value) }
)")
                  .status(),
              IsPosError("TypeInferenceError",
                         HasTypeMismatch("MyStruct[2]", "xN[S][M][N]")));
}

TEST_F(TypecheckV2Test, PrioritySelectDefaultWrongSize) {
  EXPECT_THAT(Typecheck(R"(
fn f() { priority_sel(u3:0b000, u4[3]:[u4:1, u4:2, u4:4], u5:0); }
)")
                  .status(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasTypeMismatch("xN[0][4]", "u5")));
}

TEST_F(TypecheckV2Test, OperatorOnParametricBuiltin) {
  EXPECT_THAT(
      Typecheck(R"(
fn f() { fail! % 2; }
)")
          .status(),
      IsPosError(
          "TypeInferenceError",
          AllOf(HasSubstrInV1(GetParam(),
                              "Name 'fail!' is a parametric function, but it "
                              "is not being invoked"),
                HasSubstrInV2(GetParam(),
                              "Expected type `uN[2]` but got `fail!`, which is "
                              "a parametric function not being invoked"))));
}

TEST_F(TypecheckV2Test, InvokeATokenValueViaShadowing) {
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

TEST_F(TypecheckV2Test, InvokeATokenValueNoShadowing) {
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

TEST_F(TypecheckV2Test, MapOfNonFunctionInTestProc) {
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
        let (ok, result) = map([u32:0], result_in);
    }
}
)"),
      StatusIs(absl::StatusCode::kInvalidArgument,
               AllOf(HasSubstrInV1(
                         GetParam(),
                         "Cannot resolve callee `result_in` to a function; No "
                         "function in module `fake` with name `result_in`"),
                     HasSubstrInV2(
                         GetParam(),
                         "Invocation callee `result_in` is not a function"))));
}

TEST_F(TypecheckV2Test, ReferenceToBuiltinFunctionInNext) {
  XLS_EXPECT_OK(Typecheck(R"(
proc t {
    config() { () }

    init { token() }

    next(state: token) { state }
}
)"));
}

TEST_F(TypecheckV2Test, SignedValueToBuiltinExpectingUNViaParametric) {
  EXPECT_THAT(
      Typecheck(R"(
fn p<S: bool, N: u32>() -> u32 {
  clz(xN[S][N]:0xdeadbeef) as u32
}

fn main() -> u32 {
  p<true, u32:32>()
}
)"),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          AllOf(HasSubstrInV1(GetParam(),
                              "Want argument 0 to be unsigned; got "
                              "xN[is_signed=1][32] (type is signed)"),
                HasSignednessMismatchInV2(GetParam(), "uN[32]", "sN[32]"))));
}

// Passes a signed value to a builtin function that expects a `uN[N]`.
TEST_F(TypecheckV2Test, SignedValueToBuiltinExpectingUN) {
  EXPECT_THAT(
      Typecheck(R"(
fn main() -> u32 {
  clz(s32:0xdeadbeef)
}
)"),
      StatusIs(absl::StatusCode::kInvalidArgument,
               AllOf(HasSubstrInV1(GetParam(),
                                   "Want argument 0 to be unsigned; got sN[32] "
                                   "(type is signed)"),
                     HasSignednessMismatchInV2(GetParam(), "uN[32]", "s32"))));
}

TEST_F(TypecheckV2Test, DuplicateParametricBinding) {
  EXPECT_THAT(
      Typecheck("fn p<N: u1, N: u2>() -> u32 { u32:42 }").status(),
      IsPosError("ParseError", HasSubstr("Duplicate parametric binding: `N`")));
}

TEST_F(TypecheckV2Test, MapOfXbitsArray) {
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
TEST_F(TypecheckV2Test, ResolveInTopLevelContext) {
  constexpr std::string_view kProgram = R"(
type x51 = u40;
fn x42<x46: xN[bool:0x0][38] = {xN[bool:0x0][38]:0x0}>(x43: u40) {}
fn x36<x41: u40 = {u40:0xff_ffff_ffff}>() { x42(x41); }
fn main() { x36() }
)";
  XLS_EXPECT_OK(Typecheck(kProgram));
}

// Tests that it is OK to ignore the return value of a function.
TEST_F(TypecheckV2Test, IgnoreReturnValue) {
  constexpr std::string_view kProgram = R"(
fn foo() -> u32 { u32:0 }

fn main() -> u32 {
  foo();
  u32:1
}
)";
  XLS_EXPECT_OK(Typecheck(kProgram));
}

TEST_F(TypecheckV2Test, UnassignedReturnValueTypeMismatchParametric) {
  constexpr std::string_view kProgram = R"(
fn ignored<N:u32>() -> uN[N] { zero!<uN[N]>() }

fn main(x: u32) -> u32 {
  ignored<u32:31>() + x;
  u32:1
}
)";
  EXPECT_THAT(
      Typecheck(kProgram).status(),
      StatusIs(absl::StatusCode::kInvalidArgument,
               AllOf(HasSubstrInV1(GetParam(),
                                   "uN[31] vs uN[32]: Could not deduce type "
                                   "for binary operation"),
                     HasSizeMismatchInV2(GetParam(), "uN[31]", "u32"))));
}

// Previously this would cause us to RET_CHECK because we were assuming we
// wanted to grab the root type information instead of the parametric
// invocation's type information.
TEST_F(TypecheckV2Test, AttrViaParametricBinding) {
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

TEST_F(TypecheckV2Test, MatchPackageLevelConstant) {
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

TEST_F(TypecheckV2Test, MatchPackageLevelConstantIntoTypeAlias) {
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

TEST_F(TypecheckV2Test, BitCount) {
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

TEST_F(TypecheckV2Test, ElementCount) {
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

TEST_F(TypecheckV2Test, ConfiguredValueOr) {
  XLS_ASSERT_OK(Typecheck(R"(
enum MyEnum : u2 {
  A = 0,
  B = 1,
  C = 2,
}

fn main() -> (bool, u32, s32, MyEnum, bool, u32, s32, MyEnum) {
  let b_default = configured_value_or<bool>("b_default", false);
  let u_default = configured_value_or<u32>("u32_default", u32:42);
  let s_default = configured_value_or<s32>("s32_default", s32:-100);
  let e_default = configured_value_or<MyEnum>("enum_default", MyEnum::C);
  let b_override = configured_value_or<bool>("b_override", false);
  let u_override = configured_value_or<u32>("u32_override", u32:42);
  let s_override = configured_value_or<s32>("s32_override", s32:-100);
  let e_override = configured_value_or<MyEnum>("enum_override", MyEnum::C);
  (b_default, u_default, s_default, e_default, b_override, u_override, s_override, e_override)
})"));
}

TEST_F(TypecheckV2Test, BitCountAsConstExpr) {
  XLS_ASSERT_OK(Typecheck(R"(
fn main() -> u64 {
  uN[bit_count<u32[2]>()]:0
})"));
}

TEST_F(TypecheckV2Test, BitCountWithNoTypeArgument) {
  EXPECT_THAT(
      Typecheck(R"(
fn main() -> u32 {
  bit_count<>()
})"),
      StatusIs(absl::StatusCode::kInvalidArgument,
               AllOf(HasSubstrInV1(GetParam(),
                                   "Invalid number of parametrics passed to "
                                   "'bit_count', expected 1, got 0"),
                     HasSubstrInV2(GetParam(),
                                   "Could not infer parametric(s): T"))));
}

TEST_F(TypecheckV2Test, BitCountWithMultipleTypeArguments) {
  EXPECT_THAT(
      Typecheck(R"(
fn main() -> u32 {
  bit_count<u32, u16>()
})"),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          AllOf(
              HasSubstrInV1(GetParam(),
                            "Invalid number of parametrics passed to "
                            "'bit_count', expected 1, got 2"),
              HasSubstrInV2(
                  GetParam(),
                  "Too many parametric values supplied; limit: 1 given: 2"))));
}

TEST_F(TypecheckV2Test, BitCountWithExprArgument) {
  EXPECT_THAT(
      Typecheck(R"(
fn main() -> u32 {
  bit_count<u32:0>()
})"),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          AllOf(HasSubstrInV1(GetParam(),
                              "The parametric argument for 'bit_count' should "
                              "be a type and not a value."),
                HasSubstrInV2(GetParam(),
                              "Expected parametric type, saw `u32:0`"))));
}

// Note: we use two different type signatures here for the two functions being
// binop'd -- this can help show any place we assume we can "diff" the two
// function types when they're not type-compatible.
TEST_F(TypecheckV2Test, BinaryOpsOnFunctionType) {
  const std::string kProgramTemplate = R"(fn f() -> u32 { u32:42 }
fn g() -> u8 { u8:43 }

fn main() {
  f {op} g
}
)";
  for (BinopKind binop : kAllBinopKinds) {
    std::string program = absl::StrReplaceAll(
        kProgramTemplate, {{"{op}", BinopKindFormat(binop)}});
    EXPECT_THAT(
        Typecheck(program).status(),
        StatusIs(
            absl::StatusCode::kInvalidArgument,
            testing::AnyOf(
                HasSizeMismatch("u8", "u32"),
                HasTypeMismatch("()", "() -> u32"),
                HasSubstr("Concatenation requires operand types to be either "
                          "both-arrays or both-bits; got: () -> uN[32]"))));
  }
}

TEST_F(TypecheckV2Test, UnaryOpsOnFunctionType) {
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
                         HasTypeMismatch("() -> u32", "u32")));
  }
}

// Table-oriented test that lets us validate that *types on parameters* are
// compatible with *particular values* that should be type-compatible.
TEST_F(TypecheckV2Test, ParameterVsValue) {
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

TEST_F(TypecheckV2Test, AssertFmtOk) {
  XLS_EXPECT_OK(Typecheck(R"(fn main() {
  assert_fmt!(true, "it's true");
})"));
}

TEST_F(TypecheckV2Test, AssertFmtOkWithConstexprArg) {
  XLS_EXPECT_OK(Typecheck(R"(
  const FIVE = u32:5;
  fn main() {
    assert_fmt!(true, "it's {}", FIVE);
})"));
}

TEST_F(TypecheckV2Test, AssertFmtNonBoolCondition) {
  EXPECT_THAT(
      Typecheck(R"(fn main() {
  assert_fmt!(u32:5, "this is not bool");
})"),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          AllOf(HasSubstrInV1(GetParam(),
                              "assert_fmt! condition must be a boolean value"),
                HasSizeMismatchInV2(GetParam(), "u32", "bool"))));
}

TEST_F(TypecheckV2Test, AssertFmtNonConstexprArg) {
  constexpr std::string_view kProgram = R"(fn main(x: u32) {
  assert_fmt!(true, "x is {}", x);
})";
  EXPECT_THAT(Typecheck(kProgram),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       AllOf(HasSubstr("NotConstantError:"),
                             HasSubstr("expr `x` is not constexpr"))));
}

TEST_F(TypecheckV2Test, AssertFmtArgCountMismatch) {
  constexpr std::string_view kProgram = R"(fn main() {
  assert_fmt!(true, "{} {}", u32:5);
})";
  EXPECT_THAT(
      Typecheck(kProgram),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("assert_fmt! macro expects 2 argument(s) from format "
                         "but has 1 argument(s)")));
}

}  // namespace
}  // namespace xls::dslx
