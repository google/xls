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

#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <variant>

#include "absl/base/casts.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/strings/substitute.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "xls/common/status/matchers.h"
#include "xls/dslx/create_import_data.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/frontend/ast_utils.h"
#include "xls/dslx/frontend/module.h"
#include "xls/dslx/import_data.h"
#include "xls/dslx/interp_value.h"
#include "xls/dslx/ir_convert/convert_options.h"
#include "xls/dslx/parse_and_typecheck.h"
#include "xls/dslx/type_system/type.h"
#include "xls/dslx/type_system/type_info.h"
#include "xls/dslx/type_system/typecheck_test_utils.h"
#include "xls/dslx/type_system_v2/matchers.h"
#include "xls/dslx/type_system_v2/type_system_test_utils.h"
#include "xls/dslx/virtualizable_file_system.h"
#include "xls/dslx/warning_collector.h"
#include "xls/dslx/warning_kind.h"

namespace xls::dslx {
namespace {

using ::absl_testing::IsOkAndHolds;
using ::testing::AllOf;
using ::testing::Contains;
using ::testing::Field;
using ::testing::HasSubstr;

TEST(TypecheckV2Test, SemanticSumCanonicalizationPreservesConfiguredValues) {
  constexpr std::string_view kProgram = R"(
#![feature(type_inference_v2)]
#![feature(generics)]
enum E { Unit, Payload(u32) }
$0
const VALUE = configured_value_or<u32>("K", u32:1);
)";
  for (std::string_view constructor : {"", "const X = E::Unit;"}) {
    SCOPED_TRACE(constructor);
    ImportData import_data = CreateImportDataForTest();
    ConvertOptions options;
    options.configured_values = {"K:7"};
    XLS_ASSERT_OK_AND_ASSIGN(
        TypecheckedModule result,
        ParseAndTypecheck(absl::Substitute(kProgram, constructor), "config.x",
                          "config", &import_data, /*comments=*/nullptr,
                          options));
    XLS_ASSERT_OK_AND_ASSIGN(ConstantDef * value,
                             result.module->GetConstantDef("VALUE"));
    EXPECT_THAT(result.type_info->GetConstExpr(value),
                IsOkAndHolds(InterpValue::MakeU32(7)));
  }
}

TEST(TypecheckV2Test, SemanticSumTupleConstructor) {
  EXPECT_THAT(
      R"(
enum MaybeU32 {
  None,
  Some(u32),
}
const X = MaybeU32::Some(u32:7);
)",
      TypecheckSucceeds(
          HasNodeWithType("X", "MaybeU32 { None | Some(uN[32]) }")));
}

TEST(TypecheckV2Test, SemanticSumTupleConstructorRejectsTooFewArguments) {
  EXPECT_THAT(
      R"(
enum MaybeU32 {
  None,
  Some(u32),
}
const X = MaybeU32::Some();
)",
      TypecheckFails(HasSubstr("Expected 1 argument(s) but got 0.")));
}

TEST(TypecheckV2Test, SemanticSumTupleConstructorRejectsTooManyArguments) {
  EXPECT_THAT(
      R"(
enum MaybeU32 {
  None,
  Some(u32),
}
const X = MaybeU32::Some(u32:7, u32:8);
)",
      TypecheckFails(HasSubstr("Expected 1 argument(s) but got 2.")));
}

TEST(TypecheckV2Test, SemanticSumStructConstructor) {
  EXPECT_THAT(
      R"(
enum MaybePoint {
  None,
  Point { x: u32, y: u32 },
}
const X = MaybePoint::Point { x: u32:1, y: u32:2 };
)",
      TypecheckSucceeds(HasNodeWithType(
          "X", "MaybePoint { None | Point { x: uN[32], y: uN[32] } }")));
}

TEST(TypecheckV2Test, SemanticSumStructConstructorRejectsSplat) {
  EXPECT_THAT(
      R"(
enum E { V { x: u32 } }
fn f(x: E) -> E { E::V { ..x } }
)",
      TypecheckFails(HasSubstr(
          "Struct-style sum constructors do not support splat syntax.")));
}

TEST(TypecheckV2Test, SemanticSumConstructorsCanonicalizeToSumInstances) {
  XLS_ASSERT_OK_AND_ASSIGN(TypecheckResult result, TypecheckV2(R"(
enum Option {
  None,
  Some(u32),
  Pair { lhs: u32, rhs: u32 },
}

const UNIT = Option::None;
const TUPLE = Option::Some(u32:7);
const STRUCT = Option::Pair { lhs: u32:3, rhs: u32:4 };
)"));
  XLS_ASSERT_OK_AND_ASSIGN(ConstantDef * unit,
                           result.tm.module->GetConstantDef("UNIT"));
  XLS_ASSERT_OK_AND_ASSIGN(ConstantDef * tuple,
                           result.tm.module->GetConstantDef("TUPLE"));
  XLS_ASSERT_OK_AND_ASSIGN(ConstantDef * named,
                           result.tm.module->GetConstantDef("STRUCT"));

  const auto* unit_instance =
      absl::down_cast<const SumInstance*>(unit->value());
  const auto* tuple_instance =
      absl::down_cast<const SumInstance*>(tuple->value());
  const auto* struct_instance =
      absl::down_cast<const SumInstance*>(named->value());
  EXPECT_TRUE(unit_instance->is_unit());
  EXPECT_TRUE(tuple_instance->is_tuple());
  EXPECT_TRUE(struct_instance->is_struct());
}

TEST(TypecheckV2Test, GenericSemanticSumUnitsHaveConcreteTypeAndCanonicalAst) {
  XLS_ASSERT_OK_AND_ASSIGN(TypecheckResult result, TypecheckV2(R"(
#![feature(generics)]
enum E<N: u32 = {u32:8}> { None, Some(uN[N]) }
const D = E::None;
fn f() -> E<u32:16> { E::None }
)"));
  EXPECT_THAT(TypeInfoToString(result.tm),
              IsOkAndHolds(AllOf(
                  HasNodeWithType("D", "E { None | Some(uN[8]) }"),
                  HasNodeWithType("f", "() -> E { None | Some(uN[16]) }"))));

  XLS_ASSERT_OK_AND_ASSIGN(ConstantDef * d,
                           result.tm.module->GetConstantDef("D"));
  const auto* defaulted = dynamic_cast<const SumInstance*>(d->value());
  ASSERT_NE(defaulted, nullptr);
  EXPECT_TRUE(defaulted->is_unit());
  EXPECT_EQ(defaulted->constructor_ref()->attr(), "None");

  std::optional<Function*> f = result.tm.module->GetFunction("f");
  ASSERT_TRUE(f.has_value());
  ASSERT_FALSE((*f)->body()->empty());
  const auto* contextual = dynamic_cast<const SumInstance*>(
      ToAstNode((*f)->body()->statements().back()->wrapped()));
  ASSERT_NE(contextual, nullptr);
  EXPECT_TRUE(contextual->is_unit());
  EXPECT_EQ(contextual->constructor_ref()->attr(), "None");
}

TEST(TypecheckV2Test, SemanticSumProcChannelCanonicalizesInProcContext) {
  EXPECT_THAT(
      R"(
enum Option {
  None,
  Some(u32),
}

proc Passthrough {
  in_ch: chan<Option> in;
  out_ch: chan<Option> out;

  init { () }

  config(in_ch: chan<Option> in, out_ch: chan<Option> out) {
    (in_ch, out_ch)
  }

  next(_: ()) {
    let (tok, value) = recv(join(), in_ch);
    send(tok, out_ch, value);
  }
}

proc Main {
  init { () }

  config() {
    let (input_p, input_c) = chan<Option>("input");
    let (output_p, output_c) = chan<Option>("output");
    spawn Passthrough(input_c, output_p);
    ()
  }

  next(_: ()) {
    let value = Option::Some(u32:42);
    ()
  }
}
)",
      TypecheckSucceeds(HasNodeWithType("Option::Some(u32:42)",
                                        "Option { None | Some(uN[32]) }")));
}

TEST(TypecheckV2Test, ImportedSemanticSumConstructorCanonicalizes) {
  constexpr std::string_view kImported = R"(
pub enum Option {
  None,
  Some(u32),
}
)";
  constexpr std::string_view kProgram = R"(
import imported;

type T = imported::Option;
const X: T = imported::Option::Some(u32:7);
)";

  ImportData import_data = CreateImportDataForTest();
  XLS_ASSERT_OK(TypecheckV2(kImported, "imported", &import_data));
  XLS_ASSERT_OK_AND_ASSIGN(TypecheckResult result,
                           TypecheckV2(kProgram, "main", &import_data));
  XLS_ASSERT_OK_AND_ASSIGN(ConstantDef * x,
                           result.tm.module->GetConstantDef("X"));
  EXPECT_EQ(x->value()->kind(), AstNodeKind::kSumInstance);
  // The alias borrows a ColonRef through its TypeRef, while X's value becomes
  // a SumInstance. Canonicalization must preserve that structural type edge.
  XLS_ASSERT_OK_AND_ASSIGN(TypeAlias * alias,
                           result.tm.module->GetMemberOrError<TypeAlias>("T"));
  const auto* annotation =
      absl::down_cast<const TypeRefTypeAnnotation*>(&alias->type_annotation());
  const TypeDefinition& definition = annotation->type_ref()->type_definition();
  ASSERT_TRUE(std::holds_alternative<ColonRef*>(definition));
  EXPECT_EQ(std::get<ColonRef*>(definition)->ToString(), "imported::Option");
}

TEST(TypecheckV2Test, ImportedGenericStructKeepsFinalSumDeclaration) {
  constexpr std::string_view kImported = R"(#![feature(type_inference_v2)]
#![feature(generics)]
pub struct Box<T: type> { value: T }
)";
  constexpr std::string_view kProgram = R"(#![feature(generics)]
import imported;
enum E { Unit, Payload(u32) }
$0
fn read(x: imported::Box<E>) -> E { x.value }
)";

  // The constructor used to trigger a second typecheck against the imported
  // Box's cached binding to the first E. The control changes only that line.
  for (std::string_view constructor : {"", "const X = E::Unit;"}) {
    for (bool preload : {false, true}) {
      SCOPED_TRACE(::testing::Message() << "constructor: " << constructor
                                        << ", preload: " << preload);
      absl::flat_hash_map<std::filesystem::path, std::string> files = {
          {"/imported.x", std::string(kImported)},
      };
      ImportData import_data = CreateImportDataForTest(
          std::make_unique<FakeFilesystem>(std::move(files), "/"));
      if (preload) {
        XLS_ASSERT_OK(Typecheck(kImported, "imported", &import_data,
                                /*add_version_attribute=*/false));
      }
      XLS_ASSERT_OK_AND_ASSIGN(
          TypecheckResult result,
          TypecheckV2(absl::Substitute(kProgram, constructor), "main",
                      &import_data));
      XLS_ASSERT_OK_AND_ASSIGN(SumDef * final_sum,
                               result.tm.module->GetMemberOrError<SumDef>("E"));
      XLS_ASSERT_OK_AND_ASSIGN(
          Function * read,
          result.tm.module->GetMemberOrError<Function>("read"));
      std::optional<Type*> member = result.tm.type_info->GetItem(
          ToAstNode(read->body()->statements().back()->wrapped()));
      ASSERT_TRUE(member.has_value());
      const auto* member_type = dynamic_cast<const SumType*>(*member);
      ASSERT_NE(member_type, nullptr);
      XLS_ASSERT_OK_AND_ASSIGN(
          FunctionType * signature,
          result.tm.type_info->GetItemAs<FunctionType>(read));
      const auto* return_type =
          dynamic_cast<const SumType*>(&signature->return_type());
      ASSERT_NE(return_type, nullptr);
      EXPECT_EQ(&member_type->nominal_type(), final_sum);
      EXPECT_EQ(&return_type->nominal_type(), final_sum);
    }
  }
}

TEST(TypecheckV2Test, ImportedGenericStructKeepsFinalStructDeclaration) {
  constexpr std::string_view kImported = R"(#![feature(type_inference_v2)]
#![feature(generics)]
pub struct Box<T: type> { value: T }
)";
  constexpr std::string_view kProgram = R"(#![feature(generics)]
import imported;
struct S { value: u32 }
enum E { Unit, Payload(u32) }
const X = E::Unit;
fn read(x: imported::Box<S>) -> S { x.value }
)";
  absl::flat_hash_map<std::filesystem::path, std::string> files = {
      {"/imported.x", std::string(kImported)},
  };
  ImportData import_data = CreateImportDataForTest(
      std::make_unique<FakeFilesystem>(std::move(files), "/"));
  XLS_ASSERT_OK_AND_ASSIGN(TypecheckResult result,
                           TypecheckV2(kProgram, "main", &import_data));
  XLS_ASSERT_OK_AND_ASSIGN(StructDef * final_struct,
                           result.tm.module->GetMemberOrError<StructDef>("S"));
  XLS_ASSERT_OK_AND_ASSIGN(
      Function * read, result.tm.module->GetMemberOrError<Function>("read"));
  XLS_ASSERT_OK_AND_ASSIGN(StructType * member_type,
                           result.tm.type_info->GetItemAs<StructType>(ToAstNode(
                               read->body()->statements().back()->wrapped())));
  XLS_ASSERT_OK_AND_ASSIGN(FunctionType * signature,
                           result.tm.type_info->GetItemAs<FunctionType>(read));
  const auto* return_type =
      dynamic_cast<const StructType*>(&signature->return_type());
  ASSERT_NE(return_type, nullptr);
  EXPECT_EQ(&member_type->nominal_type(), final_struct);
  EXPECT_EQ(&return_type->nominal_type(), final_struct);
}

TEST(TypecheckV2Test, SemanticSumNormalizationPreservesUseBindings) {
  constexpr std::string_view kImported = R"(#![feature(type_inference_v2)]
pub const VALUE = u32:7;
)";
  constexpr std::string_view kProgram = R"(#![feature(use_syntax)]
use imported::VALUE;
enum E { V(u32) }
const X = E::V(VALUE);
)";
  absl::flat_hash_map<std::filesystem::path, std::string> files = {
      {"/imported.x", std::string(kImported)},
  };
  ImportData import_data = CreateImportDataForTest(
      std::make_unique<FakeFilesystem>(std::move(files), "/"));
  EXPECT_THAT(
      TypecheckV2(kProgram, "main", &import_data),
      IsOkAndHolds(HasTypeInfo(HasNodeWithType("X", "E { V(uN[32]) }"))));
}

TEST(TypecheckV2Test, SemanticSumConstructorFromUseImport) {
  constexpr std::string_view kImported = R"(#![feature(type_inference_v2)]
pub enum E { V(u8) }
)";
  constexpr std::string_view kProgram = R"(#![feature(use_syntax)]
use imported::E;
const X = E::V(u8:1);
)";
  absl::flat_hash_map<std::filesystem::path, std::string> files = {
      {"/imported.x", std::string(kImported)},
  };
  ImportData import_data = CreateImportDataForTest(
      std::make_unique<FakeFilesystem>(std::move(files), "/"));
  EXPECT_THAT(
      TypecheckV2(kProgram, "main", &import_data),
      IsOkAndHolds(HasTypeInfo(HasNodeWithType("X", "E { V(uN[8]) }"))));
}

TEST(TypecheckV2Test, SemanticSumNormalizationPreservesGeneratedDomain) {
  XLS_ASSERT_OK_AND_ASSIGN(TypecheckResult result, TypecheckV2(R"(
#[fuzz_domain("S_Domain")]
struct S { x: u32 }
enum E { Unit, Payload(u32) }
const X = E::Unit;
fn domain_field(d: S_Domain) -> () { d.x }
)"));
  XLS_ASSERT_OK_AND_ASSIGN(StructDef * original,
                           result.tm.module->GetMemberOrError<StructDef>("S"));
  XLS_ASSERT_OK_AND_ASSIGN(
      StructDef * domain,
      result.tm.module->GetMemberOrError<StructDef>("S_Domain"));
  EXPECT_EQ(domain->name_def()->definer(), original);
  ASSERT_EQ(domain->members().size(), 1);
  EXPECT_EQ(domain->members()[0]->name(), "x");
  EXPECT_EQ(domain->members()[0]->type()->ToString(), "()");
}

TEST(TypecheckV2Test, SemanticSumNormalizationPreparesImportedDomain) {
  constexpr std::string_view kImported = R"(#![feature(type_inference_v2)]
#[fuzz_domain("S_Domain")]
pub struct S { x: u32 }
enum E { Unit, Payload(u32) }
const X = E::Unit;
)";
  constexpr std::string_view kProgram = R"(
import imported;
fn domain_field(d: imported::S_Domain) -> () { d.x }
)";
  absl::flat_hash_map<std::filesystem::path, std::string> files = {
      {"/imported.x", std::string(kImported)},
  };
  ImportData import_data = CreateImportDataForTest(
      std::make_unique<FakeFilesystem>(std::move(files), "/"));
  // Imported modules suppress semantic warnings, but still need preparation.
  EXPECT_THAT(TypecheckV2(kProgram, "main", &import_data),
              IsOkAndHolds(HasTypeInfo(HasNodeWithType("d.x", "()"))));
}

TEST(TypecheckV2Test, SemanticSumNormalizationPreservesCapturedLambda) {
  EXPECT_THAT(R"(
enum E { V(u32) }
fn f(capture: u32) -> E[1] {
  map(u32[1]:[0], |x| -> E { E::V(capture + x) })
}
)",
              TypecheckSucceeds(
                  HasNodeWithType("f", "(uN[32]) -> E { V(uN[32]) }[1]")));
}

TEST(TypecheckV2Test, ImportedTypesCannotBeSemanticSumPayloadValues) {
  constexpr std::string_view kImported = R"(
pub enum Tag: u8 { A = 0 }
)";
  constexpr std::string_view kPrograms[] = {
      R"(
import imported;
enum E { V(imported::Tag) }
fn f() -> E { E::V(imported::Tag) }
)",
      R"(
import imported;
enum E { V { x: imported::Tag } }
fn f() -> E { E::V { x: imported::Tag } }
)",
  };
  for (std::string_view program : kPrograms) {
    SCOPED_TRACE(program);
    ImportData import_data = CreateImportDataForTest();
    XLS_ASSERT_OK(TypecheckV2(kImported, "imported", &import_data));
    EXPECT_THAT(
        TypecheckV2(program, "main", &import_data),
        StatusIs(absl::StatusCode::kInvalidArgument,
                 HasSubstr("Cannot pass a type as a sum constructor payload")));
  }
}

TEST(TypecheckV2Test, ImportedEnumValuesAreSemanticSumPayloadValues) {
  constexpr std::string_view kImported = R"(
pub enum Tag: u8 { A = 0 }
)";
  constexpr std::string_view kProgram = R"(
import imported;
enum E { Tuple(imported::Tag), Named { x: imported::Tag } }
fn tuple() -> E { E::Tuple(imported::Tag::A) }
fn named() -> E { E::Named { x: imported::Tag::A } }
)";
  ImportData import_data = CreateImportDataForTest();
  XLS_ASSERT_OK(TypecheckV2(kImported, "imported", &import_data));
  XLS_EXPECT_OK(TypecheckV2(kProgram, "main", &import_data));
}

TEST(TypecheckV2Test, ImportedCanonicalizedSemanticSumCanBeUsedAsType) {
  constexpr std::string_view kImported = R"(
pub enum Option {
  None,
  Some(u32),
}
pub const SOME: Option = Option::Some(u32:7);
)";
  constexpr std::string_view kProgram = R"(
import imported;

fn identity(x: imported::Option) -> imported::Option {
  x
}
)";

  ImportData import_data = CreateImportDataForTest();
  XLS_ASSERT_OK(TypecheckV2(kImported, "imported", &import_data));
  EXPECT_THAT(TypecheckV2(kProgram, "main", &import_data),
              IsOkAndHolds(HasTypeInfo(
                  HasNodeWithType("x", "Option { None | Some(uN[32]) }"))));
}

TEST(TypecheckV2Test, SemanticSumExplicitDiscriminantsMustBeDistinct) {
  EXPECT_THAT(
      R"(
enum Message : u3 {
  Idle() = 0,
  Request(u8) = 3,
  Retry(u8) = 3,
}
)",
      TypecheckFails(AllOf(HasSubstr("Semantic sum `Message`"),
                           HasSubstr("duplicate discriminant"))));
}

TEST(TypecheckV2Test, LocalSemanticSumConstructorExplicitParametricsRejected) {
  EXPECT_THAT(
      R"(#![feature(generics)]
enum Option<N: u32> {
  None,
  Some(uN[N]),
}

fn make(x: u8) -> Option<u32:8> {
  Option::Some<u32:8>(x)
}
)",
      TypecheckFails(HasSubstr("Explicit parametrics belong on the sum type, "
                               "not the constructor")));
}

TEST(TypecheckV2Test,
     ImportedSemanticSumConstructorExplicitParametricsRejected) {
  constexpr std::string_view kImported = R"(#![feature(generics)]
pub enum Option<N: u32> {
  None,
  Some(uN[N]),
}
)";
  constexpr std::string_view kProgram = R"(#![feature(generics)]
import imported;

fn make(x: u8) -> imported::Option<u32:8> {
  imported::Option::Some<u32:8>(x)
}
)";

  ImportData import_data = CreateImportDataForTest();
  XLS_ASSERT_OK(TypecheckV2(kImported, "imported", &import_data));
  EXPECT_THAT(
      TypecheckV2(kProgram, "main", &import_data),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("Explicit parametrics belong on the sum type, not "
                         "the constructor")));
}

TEST(TypecheckV2Test,
     SemanticSumConstructorExplicitParametricsRejectedInGenericBody) {
  EXPECT_THAT(
      R"(#![feature(generics)]
enum E<N: u32> { V(uN[N]) }
fn make<N: u32>(x: uN[N]) -> E<N> { E::V<N>(x) }
)",
      TypecheckFails(HasSubstr("Explicit parametrics belong on the sum type, "
                               "not the constructor")));
}

TEST(TypecheckV2Test, NamedSemanticSumRejectsVariantSuffixParametrics) {
  constexpr std::string_view kImported = R"(#![feature(generics)]
pub enum E<N: u32 = {u32:8}> { V { x: uN[N] } }
)";
  ImportData import_data = CreateImportDataForTest();
  XLS_ASSERT_OK(TypecheckV2(kImported, "imported", &import_data));
  EXPECT_THAT(TypecheckV2(R"(#![feature(generics)]
import imported;
const X = imported::E::V<u32:16> { x: u8:0 };
)",
                          "main", &import_data),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Explicit parametrics belong on the sum type, "
                                 "not the constructor")));
  EXPECT_THAT(R"(#![feature(generics)]
enum E<N: u32 = {u32:8}> { V { x: uN[N] } }
const X = E<u32:16>::V { x: u8:0 };
)",
              TypecheckFails(HasSubstr("size mismatch")));
  EXPECT_THAT(R"(#![feature(generics)]
enum E<N: u32 = {u32:8}> { V { x: uN[N] } }
const DEFAULT = E::V { x: u8:0 };
const EXPLICIT = E<u32:16>::V { x: u16:0 };
)",
              TypecheckSucceeds(
                  AllOf(HasNodeWithType("DEFAULT", "E { V { x: uN[8] } }"),
                        HasNodeWithType("EXPLICIT", "E { V { x: uN[16] } }"))));
}

TEST(TypecheckV2Test, MissingSemanticSumConstructorReturnsUserError) {
  EXPECT_THAT(
      R"(
enum Option {
  None,
  Some(u32),
}

const X = Option::Missing(u32:7);
)",
      TypecheckFails(HasSubstr("Sum 'Option' has no constructor 'Missing'.")));
}

TEST(TypecheckV2Test, SemanticSumStructConstructorMissingMemberRejected) {
  EXPECT_THAT(
      R"(
enum MaybePoint {
  None,
  Point { x: u32, y: u32 },
}

const X = MaybePoint::Point { x: u32:1 };
)",
      TypecheckFails(
          HasSubstr("Instance of constructor `Point` is missing member(s): "
                    "`y`")));
}

TEST(TypecheckV2Test, SemanticSumStructConstructorExtraMemberRejected) {
  EXPECT_THAT(
      R"(
enum MaybePoint {
  None,
  Point { x: u32, y: u32 },
}

const X = MaybePoint::Point { x: u32:1, y: u32:2, z: u32:3 };
)",
      TypecheckFails(HasSubstr("Constructor `Point` has no member `z`")));
}

TEST(TypecheckV2Test, SemanticSumTuplePayloadAggregateRejectedInPhase1) {
  EXPECT_THAT(
      R"(
struct Point {
  x: u32,
  y: u32,
}

enum MaybePoint {
  None,
  Some(Point),
}

const X = MaybePoint::None;
)",
      TypecheckFails(AllOf(
          HasSubstr("Semantic sum payload members must be bits-like, enum "
                    "typed, or empty semantic sums"),
          HasSubstr("constructor `Some`"), HasSubstr("Point"))));
}

TEST(TypecheckV2Test, SemanticSumStructPayloadAggregateRejectedInPhase1) {
  EXPECT_THAT(
      R"(
enum PairBox {
  Pair { xy: (u32, u32) },
}

const X = PairBox::Pair { xy: (u32:1, u32:2) };
)",
      TypecheckFails(AllOf(
          HasSubstr("Semantic sum payload members must be bits-like, enum "
                    "typed, or empty semantic sums"),
          HasSubstr("constructor `Pair`"), HasSubstr("(uN[32], uN[32])"))));
}

TEST(TypecheckV2Test, ImplicitSemanticSumRejectsTagTypeAnnotationInPhase1) {
  EXPECT_THAT(
      R"(
enum MaybeU32 : u3 {
  None,
  Some(u32),
}
)",
      TypecheckFails(HasSubstr(
          "Semantic sum `MaybeU32` with a tag type annotation requires "
          "explicit discriminants on every variant.")));
}

TEST(TypecheckV2Test,
     ParametricSemanticTupleConstructorUsesContextualPayloadType) {
  EXPECT_THAT(
      R"(#![feature(generics)]
enum OptionN<N: u32> {
  None,
  Some(uN[N]),
}

fn make(x: u16) -> OptionN<u32:8> {
  OptionN::Some(x)
}
)",
      TypecheckFails(AllOf(HasSubstr("size mismatch"), HasSubstr("u16"),
                           HasSubstr("uN[8]"))));
}

TEST(TypecheckV2Test,
     ParametricSemanticStructConstructorUsesContextualPayloadType) {
  EXPECT_THAT(
      R"(#![feature(generics)]
enum BoxN<N: u32> {
  None,
  Pair { value: uN[N] },
}

fn make(x: u16) -> BoxN<u32:8> {
  BoxN::Pair { value: x }
}
)",
      TypecheckFails(AllOf(HasSubstr("size mismatch"), HasSubstr("u16"),
                           HasSubstr("uN[8]"))));
}

TEST(TypecheckV2Test, ExplicitSemanticSumParametricsMustAgreeWithContext) {
  EXPECT_THAT(
      R"(#![feature(generics)]
enum E<N: u32> { V(uN[N]) }
fn f(x: u16) -> E<u32:16> { E<u32:8>::V(x) }
)",
      TypecheckFails(HasSubstr("Value mismatch for parametric `N`")));
  EXPECT_THAT(
      R"(#![feature(generics)]
enum E<N: u32> { V(uN[N]) }
fn f(x: u16) -> E<u32:16> { E<u32:16>::V(x) }
)",
      TypecheckSucceeds(::testing::_));
}

TEST(TypecheckV2Test, ParametricSemanticTupleConstructorInfersValueParametric) {
  EXPECT_THAT(
      R"(#![feature(generics)]
enum OptionN<N: u32> {
  None,
  Some(uN[N]),
}

const X = OptionN::Some(u7:7);
)",
      TypecheckSucceeds(::testing::_));
}

TEST(TypecheckV2Test, ParametricSemanticSumAliasInfersValueParametric) {
  EXPECT_THAT(
      R"(#![feature(generics)]
enum OptionN<N: u32> {
  None,
  Some(uN[N]),
}
type AbstractOption = OptionN;

const X = AbstractOption::Some(u7:7);
)",
      TypecheckSucceeds(::testing::_));
}

TEST(TypecheckV2Test, ParametricSemanticStructConstructorInfersParametric) {
  EXPECT_THAT(
      R"(#![feature(generics)]
enum Box<N: u32> {
  Empty,
  Value { item: uN[N] },
}

const X = Box::Value { item: u16:7 };
)",
      TypecheckSucceeds(::testing::_));
}

TEST(TypecheckV2Test, ImportedParametricSemanticConstructorInfersParametric) {
  constexpr std::string_view kImported = R"(#![feature(generics)]
pub enum Option<N: u32> {
  None,
  Some(uN[N]),
}
)";
  constexpr std::string_view kProgram = R"(#![feature(generics)]
import imported;

const X = imported::Option::Some(u8:7);
)";

  ImportData import_data = CreateImportDataForTest();
  XLS_ASSERT_OK(TypecheckV2(kImported, "imported", &import_data));
  EXPECT_THAT(TypecheckV2(kProgram, "main", &import_data),
              IsOkAndHolds(::testing::_));
}

TEST(TypecheckV2Test, NonUnitSemanticSumConstructorCannotBeUsedAsValue) {
  EXPECT_THAT(
      R"(
enum MaybeU32 {
  None,
  Some(u32),
}

fn f() -> () {
  let make = MaybeU32::Some;
  ()
}
)",
      TypecheckFails(AllOf(HasSubstr("MaybeU32::Some"),
                           HasSubstr("cannot be used as a value"))));
}

TEST(TypecheckV2Test, SemanticSumMatchRejectedBeforePatternLayer) {
  EXPECT_THAT(
      R"(
enum Option {
  None,
  Some(u32),
}

fn f(x: Option) -> u32 {
  match x {
    _ => u32:0,
  }
}
)",
      TypecheckFails(
          HasSubstr("Match expressions over semantic sums are not supported")));
}

TEST(TypecheckV2Test, ZeroMacroImplicitSemanticSumUsesFirstVariant) {
  EXPECT_THAT(
      R"(
enum MaybeU32 {
  None,
  Some(u32),
}
const Y = zero!<MaybeU32>();
)",
      TypecheckSucceeds(
          HasNodeWithType("Y", "MaybeU32 { None | Some(uN[32]) }")));
}

TEST(TypecheckV2Test, ZeroMacroExplicitSemanticSumUsesZeroDiscriminant) {
  XLS_ASSERT_OK_AND_ASSIGN(TypecheckResult result, TypecheckV2(R"(
enum Message : u3 {
  Request(u8) = 3,
  Idle() = 0,
}
const Y = zero!<Message>();
)"));
  EXPECT_THAT(result, HasTypeInfo(HasNodeWithType(
                          "Y", "Message { Request(uN[8]) | Idle() }")));
  XLS_ASSERT_OK_AND_ASSIGN(ConstantDef * constant,
                           result.tm.module->GetConstantDef("Y"));
  XLS_ASSERT_OK_AND_ASSIGN(InterpValue value,
                           result.tm.type_info->GetConstExpr(constant));
  EXPECT_EQ(value, InterpValue::MakeTuple(
                       {InterpValue::MakeUBits(1, 1),
                        InterpValue::MakeTuple({InterpValue::MakeU8(0)})}));
}

TEST(TypecheckV2Test, AllOnesMacroSemanticSumReportsUnsupportedType) {
  EXPECT_THAT(
      R"(
enum Message { Idle, Data(u8) }
const Y = all_ones!<Message>();
)",
      TypecheckFails(HasSubstr(
          "Cannot use `all_ones!<Message>()` with sum type `Message`")));
}

TEST(TypecheckV2Test, ZeroMacroTupleContainingSemanticSumHasConstexprValue) {
  XLS_ASSERT_OK_AND_ASSIGN(TypecheckResult result, TypecheckV2(R"(
enum Option {
  None,
  Some(u32),
}
const Y = zero!<(Option,)>();
)"));
  XLS_ASSERT_OK_AND_ASSIGN(ConstantDef * constant,
                           result.tm.module->GetConstantDef("Y"));
  XLS_ASSERT_OK_AND_ASSIGN(InterpValue value,
                           result.tm.type_info->GetConstExpr(constant));
  const InterpValue none = InterpValue::MakeTuple(
      {InterpValue::MakeUBits(1, 0),
       InterpValue::MakeTuple({InterpValue::MakeU32(0)})});
  EXPECT_EQ(value, InterpValue::MakeTuple({none}));
}

TEST(TypecheckV2Test, ZeroMacroArrayContainingSemanticSumHasConstexprValue) {
  XLS_ASSERT_OK_AND_ASSIGN(TypecheckResult result, TypecheckV2(R"(
enum Option {
  None,
  Some(u32),
}
const Y = zero!<Option[1]>();
)"));
  XLS_ASSERT_OK_AND_ASSIGN(ConstantDef * constant,
                           result.tm.module->GetConstantDef("Y"));
  XLS_ASSERT_OK_AND_ASSIGN(InterpValue value,
                           result.tm.type_info->GetConstExpr(constant));
  const InterpValue none = InterpValue::MakeTuple(
      {InterpValue::MakeUBits(1, 0),
       InterpValue::MakeTuple({InterpValue::MakeU32(0)})});
  XLS_ASSERT_OK_AND_ASSIGN(InterpValue expected,
                           InterpValue::MakeArray({none}));
  EXPECT_EQ(value, expected);
}

TEST(TypecheckV2Test, ZeroMacroStructContainingSemanticSumHasConstexprValue) {
  XLS_ASSERT_OK_AND_ASSIGN(TypecheckResult result, TypecheckV2(R"(
enum Option {
  None,
  Some(u32),
}
struct Wrapper {
  value: Option,
}
const Y = zero!<Wrapper>();
)"));
  XLS_ASSERT_OK_AND_ASSIGN(ConstantDef * constant,
                           result.tm.module->GetConstantDef("Y"));
  XLS_ASSERT_OK_AND_ASSIGN(InterpValue value,
                           result.tm.type_info->GetConstExpr(constant));
  const InterpValue none = InterpValue::MakeTuple(
      {InterpValue::MakeUBits(1, 0),
       InterpValue::MakeTuple({InterpValue::MakeU32(0)})});
  EXPECT_EQ(value, InterpValue::MakeTuple({none}));
}

TEST(TypecheckV2Test, ZeroMacroExplicitSemanticSumWithoutZeroFails) {
  EXPECT_THAT(
      R"(
enum Message : u3 {
  Request(u8) = 3,
  Response(u8) = 7,
}
const Y = zero!<Message>();
)",
      TypecheckFails(
          HasSubstr("Sum type 'Message' does not have a known zero value.")));
}

TEST(TypecheckV2Test, ZeroMacroGenericSemanticSumKeepsEachInstancesVariant) {
  constexpr std::string_view kPrograms[] = {
      R"(#![feature(generics)]
enum E<N: u32>: u32 { A(u8) = N, B(u8) = N ^ u32:1 }
const Y = zero!<(E<u32:0>, E<u32:1>)>();
)",
      R"(#![feature(generics)]
enum E<N: u32>: u32 { A(u8) = N, B(u8) = N ^ u32:1 }
const Y = zero!<(E<u32:1>, E<u32:0>)>();
)",
  };
  for (int first_tag = 0; first_tag < 2; ++first_tag) {
    SCOPED_TRACE(kPrograms[first_tag]);
    XLS_ASSERT_OK_AND_ASSIGN(TypecheckResult result,
                             TypecheckV2(kPrograms[first_tag]));
    XLS_ASSERT_OK_AND_ASSIGN(ConstantDef * constant,
                             result.tm.module->GetConstantDef("Y"));
    XLS_ASSERT_OK_AND_ASSIGN(InterpValue value,
                             result.tm.type_info->GetConstExpr(constant));
    // The concrete payload shapes are identical. Only the instantiated
    // discriminants determine which variant has the zero value.
    const InterpValue payloads = InterpValue::MakeTuple(
        {InterpValue::MakeU8(0), InterpValue::MakeU8(0)});
    EXPECT_EQ(value,
              InterpValue::MakeTuple(
                  {InterpValue::MakeTuple(
                       {InterpValue::MakeUBits(1, first_tag), payloads}),
                   InterpValue::MakeTuple(
                       {InterpValue::MakeUBits(1, 1 - first_tag), payloads})}));
  }
}

TEST(TypecheckV2Test, ZeroMacroGenericSemanticSumInConstexprFunction) {
  XLS_ASSERT_OK_AND_ASSIGN(TypecheckResult result, TypecheckV2(R"(
#![feature(generics)]
enum E<N: u32>: u32 { A(u8) = N, B(u8) = N ^ u32:1 }
fn make() -> (E<u32:0>, E<u32:1>) {
  zero!<(E<u32:0>, E<u32:1>)>()
}
const Y = make();
)"));
  XLS_ASSERT_OK_AND_ASSIGN(ConstantDef * constant,
                           result.tm.module->GetConstantDef("Y"));
  XLS_ASSERT_OK_AND_ASSIGN(InterpValue value,
                           result.tm.type_info->GetConstExpr(constant));
  const InterpValue payloads =
      InterpValue::MakeTuple({InterpValue::MakeU8(0), InterpValue::MakeU8(0)});
  EXPECT_EQ(
      value,
      InterpValue::MakeTuple(
          {InterpValue::MakeTuple({InterpValue::MakeUBits(1, 0), payloads}),
           InterpValue::MakeTuple({InterpValue::MakeUBits(1, 1), payloads})}));
}

TEST(TypecheckV2Test, ZeroMacroImportedGenericSumInStructAndArray) {
  constexpr std::string_view kImported = R"(#![feature(generics)]
pub enum E<N: u32>: u32 { A(u8) = N, B(u8) = N ^ u32:1 }
)";
  constexpr std::string_view kProgram = R"(#![feature(generics)]
import imported;
type Alias = imported::E<u32:1>;
struct Wrapper { first: imported::E<u32:0>, rest: Alias[2] }
const Y = zero!<Wrapper>();
)";
  ImportData import_data = CreateImportDataForTest();
  XLS_ASSERT_OK(TypecheckV2(kImported, "imported", &import_data));
  XLS_ASSERT_OK_AND_ASSIGN(TypecheckResult result,
                           TypecheckV2(kProgram, "main", &import_data));
  XLS_ASSERT_OK_AND_ASSIGN(ConstantDef * constant,
                           result.tm.module->GetConstantDef("Y"));
  XLS_ASSERT_OK_AND_ASSIGN(InterpValue value,
                           result.tm.type_info->GetConstExpr(constant));
  const InterpValue payloads =
      InterpValue::MakeTuple({InterpValue::MakeU8(0), InterpValue::MakeU8(0)});
  const InterpValue a =
      InterpValue::MakeTuple({InterpValue::MakeUBits(1, 0), payloads});
  const InterpValue b =
      InterpValue::MakeTuple({InterpValue::MakeUBits(1, 1), payloads});
  XLS_ASSERT_OK_AND_ASSIGN(InterpValue array, InterpValue::MakeArray({b, b}));
  EXPECT_EQ(value, InterpValue::MakeTuple({a, array}));
}

TEST(TypecheckV2Test, ZeroMacroGenericSemanticSumWithoutZeroFails) {
  EXPECT_THAT(
      R"(#![feature(generics)]
enum E<N: u32>: u32 { A(u8) = N, B(u8) = N ^ u32:1 }
const Y = zero!<(E<u32:2>,)>();
)",
      TypecheckFails(
          HasSubstr("Sum type 'E' does not have a known zero value.")));
}

TEST(TypecheckV2Test,
     ZeroMacroGenericSemanticSumRequiresZeroConstructiblePayload) {
  EXPECT_THAT(
      R"(#![feature(generics)]
enum Never {}
enum E<N: u32>: u32 { A(Never) = N }
const Y = zero!<(E<u32:0>,)>();
)",
      TypecheckFails(
          HasSubstr("Sum type 'Never' does not have a known zero value.")));
}

TEST(TypecheckV2Test, ZeroMacroEmptySemanticSumFails) {
  EXPECT_THAT(
      R"(
enum Never {}
const Y = zero!<Never>();
)",
      TypecheckFails(
          HasSubstr("Sum type 'Never' does not have a known zero value.")));
}

TEST(TypecheckV2Test, ZeroMacroAnnotatedEmptySemanticSumFails) {
  EXPECT_THAT(
      R"(
enum Never : u3 {}
const Y = zero!<Never>();
)",
      TypecheckFails(
          HasSubstr("Enum type 'Never' does not have a known zero value.")));
}

TEST(TypecheckV2Test, ZeroMacroImportedSemanticSumUsesFirstVariant) {
  constexpr std::string_view kImported = R"(
pub enum ImportedMaybe {
  None,
  Some(u32),
}
)";
  constexpr std::string_view kProgram = R"(
import imported;
const Y = zero!<imported::ImportedMaybe>();
)";
  ImportData import_data = CreateImportDataForTest();
  XLS_EXPECT_OK(TypecheckV2(kImported, "imported", &import_data));
  EXPECT_THAT(TypecheckV2(kProgram, "main", &import_data),
              IsOkAndHolds(HasTypeInfo(HasNodeWithType(
                  "Y", "ImportedMaybe { None | Some(uN[32]) }"))));
}

TEST(TypecheckV2Test,
     SemanticSumConstructorsRejectTypeParametricsWithoutBindings) {
  constexpr std::string_view kPrograms[] = {
      R"(#![feature(generics)]
enum E { V(u8) }
const X = E<u32:1>::V(u8:0);
)",
      R"(#![feature(generics)]
enum E { Unit, V(u8) }
const X = E<u32:1>::Unit;
)",
      R"(#![feature(generics)]
enum E { V { x: u8 } }
const X = E<u32:1>::V { x: u8:0 };
)",
  };
  for (std::string_view program : kPrograms) {
    SCOPED_TRACE(program);
    EXPECT_THAT(program, TypecheckFails(HasSubstr(
                             "Too many parametric values supplied; limit: 0 "
                             "given: 1")));
  }
}

TEST(TypecheckV2Test,
     ParametricSemanticTupleConstructorRejectsExtraTypeParametric) {
  EXPECT_THAT(
      R"(#![feature(generics)]
enum E<N: u32> { V(uN[N]) }
const X = E<u32:8, u32:16>::V(u8:0);
)",
      TypecheckFails(
          HasSubstr("Too many parametric values supplied; limit: 1 given: 2")));
}

TEST(TypecheckV2Test, SemanticTupleConstructorRejectsWrongValueParametricType) {
  EXPECT_THAT(
      R"(#![feature(generics)]
enum E<N: u32> { V(u8) }
const X = E<u8:8>::V(u8:0);
)",
      TypecheckFails(HasSubstr("size mismatch")));
}

TEST(TypecheckV2Test, SemanticSumStructConstructorDuplicateMemberRejected) {
  EXPECT_THAT(
      R"(
enum E { V { x: u8 } }
const X = E::V { x: u8:0, x: u8:1 };
)",
      TypecheckFails(
          HasSubstr("Duplicate value seen for `x` in constructor `V`.")));
}

TEST(TypecheckV2Test, SemanticSumStructConstructorBindsReorderedMembersByName) {
  EXPECT_THAT(
      R"(
enum E { V { x: u8, y: u16 } }
const X = E::V { y: u16:2, x: u8:1 };
)",
      TypecheckSucceeds(
          HasNodeWithType("X", "E { V { x: uN[8], y: uN[16] } }")));
}

TEST(TypecheckV2Test, GenericSemanticSumDeclarationNeedsNoInstantiation) {
  EXPECT_THAT(
      R"(#![feature(generics)]
enum Box<T: type> { Value(T) }
)",
      TypecheckSucceeds(::testing::_));
}

TEST(TypecheckV2Test, GenericSemanticSumExplicitAndInferredTypes) {
  EXPECT_THAT(
      R"(#![feature(generics)]
enum Box<T: type> { Value(T) }
const EXPLICIT = Box<u8>::Value(u8:1);
const INFERRED = Box::Value(u8:2);
)",
      TypecheckSucceeds(
          AllOf(HasNodeWithType("EXPLICIT", "Box { Value(uN[8]) }"),
                HasNodeWithType("INFERRED", "Box { Value(uN[8]) }"))));
}

TEST(TypecheckV2Test, GenericSemanticSumRejectsMismatchedPayload) {
  EXPECT_THAT(
      R"(#![feature(generics)]
enum Box<T: type> { Value(T) }
fn f(x: u16) -> Box<u8> { Box::Value(x) }
)",
      TypecheckFails(HasSubstr("size mismatch")));
}

TEST(TypecheckV2Test, GenericSemanticSumValidatesInstantiatedPayloadType) {
  EXPECT_THAT(
      R"(#![feature(generics)]
enum Box<T: type> { Value(T) }
fn f(x: Box<u8[1]>) -> Box<u8[1]> { x }
)",
      TypecheckFails(HasSubstr("Semantic sum payload members must be")));
}

TEST(TypecheckV2Test, SemanticSumReferenceResolvesValueDefault) {
  EXPECT_THAT(
      R"(#![feature(generics)]
enum E<N: u32 = {u32:8}> { V(uN[N]) }
fn f(x: E) -> E { x }
const X = E::V(u8:1);
)",
      TypecheckSucceeds(HasNodeWithType("X", "E { V(uN[8]) }")));
}

TEST(TypecheckV2Test, SemanticSumReferenceResolvesTypeDefaultAndOverride) {
  EXPECT_THAT(
      R"(#![feature(generics)]
enum Box<T: type = u8> { Value(T) }
fn f(x: Box) -> Box { x }
const DEFAULT = Box::Value(u8:1);
const OVERRIDE = Box<u16>::Value(u16:2);
)",
      TypecheckSucceeds(
          AllOf(HasNodeWithType("DEFAULT", "Box { Value(uN[8]) }"),
                HasNodeWithType("OVERRIDE", "Box { Value(uN[16]) }"))));
}

TEST(TypecheckV2Test, SemanticSumValueDefaultSubstitutesLiteralType) {
  EXPECT_THAT(
      R"(#![feature(generics)]
enum E<T: type = u8, V: T = {T:0}> { A(u8) }
fn f(x: E) -> E { x }
fn g(x: E<u16>) -> E<u16> { x }
)",
      TypecheckSucceeds(::testing::_));
}

TEST(TypecheckV2Test, SemanticSumValueDefaultSubstitutesLiteralWidth) {
  EXPECT_THAT(
      R"(#![feature(generics)]
enum E<N: u32 = {u32:8}, V: uN[N] = {uN[N]:0}> { A(u8) }
fn f(x: E) -> E { x }
fn g(x: E<u32:16>) -> E<u32:16> { x }
)",
      TypecheckSucceeds(::testing::_));
}

TEST(TypecheckV2Test, SemanticSumUntypedDefaultsKeepBindingWidth) {
  EXPECT_THAT(
      R"(#![feature(generics)]
enum E<N: u32 = {8}, V: uN[N] = {0}> { A(uN[N]) }
const NARROW = E::A(u8:0);
const WIDE = E<u32:16>::A(u16:0);
)",
      TypecheckSucceeds(AllOf(HasNodeWithType("NARROW", "E { A(uN[8]) }"),
                              HasNodeWithType("WIDE", "E { A(uN[16]) }"))));
}

TEST(TypecheckV2Test, SemanticSumDefaultPreservesConstexprRolloverWarning) {
  XLS_ASSERT_OK_AND_ASSIGN(TypecheckResult result, TypecheckV2(R"(
#![feature(generics)]
enum E<N: u32 = {u32:0xffff_ffff + u32:1}> { A(u8) }
fn f(x: E) -> E { x }
)"));
  EXPECT_THAT(result.tm.warnings.warnings(),
              Contains(AllOf(
                  Field(&WarningCollector::Entry::kind,
                        WarningKind::kConstexprEvalRollover),
                  Field(&WarningCollector::Entry::message,
                        HasSubstr("constexpr evaluation detected rollover")))));
}

TEST(TypecheckV2Test, SemanticSumTypeDefaultRejectsMismatchedPayload) {
  EXPECT_THAT(
      R"(#![feature(generics)]
enum Box<T: type = u8> { Value(T) }
const X = Box::Value(u16:1);
)",
      TypecheckFails(HasSubstr("size mismatch")));
}

TEST(TypecheckV2Test, SemanticSumInfersBeforeDependentValueDefault) {
  EXPECT_THAT(
      R"(#![feature(generics)]
enum E<N: u32, M: u32 = {N + u32:1}> { V(uN[N], uN[M]) }
const X = E::V(u8:1, u9:2);
)",
      TypecheckSucceeds(HasNodeWithType("X", "E { V(uN[8], uN[9]) }")));
}

TEST(TypecheckV2Test, SemanticSumInfersBeforeValueDependentTypeDefault) {
  EXPECT_THAT(
      R"(#![feature(generics)]
enum E<N: u32, T: type = uN[N]> { V(uN[N], T) }
const X = E::V(u8:1, u8:2);
)",
      TypecheckSucceeds(HasNodeWithType("X", "E { V(uN[8], uN[8]) }")));
}

TEST(TypecheckV2Test, SemanticSumInfersBeforeTypeDependentTypeDefault) {
  EXPECT_THAT(
      R"(#![feature(generics)]
enum E<T: type, U: type = T> { V(T, U) }
const X = E::V(u8:1, u8:2);
)",
      TypecheckSucceeds(HasNodeWithType("X", "E { V(uN[8], uN[8]) }")));
}

TEST(TypecheckV2Test, SemanticSumSubstitutesNestedTypeArgumentsInDefault) {
  EXPECT_THAT(
      R"(#![feature(generics)]
struct W<T: type> { x: T }
enum E<T: type, U: type = W<T>> { V(T) }
const X = E::V(u8:1);
)",
      TypecheckSucceeds(HasNodeWithType("X", "E { V(uN[8]) }")));
}

TEST(TypecheckV2Test, SemanticSumKeepsConcreteValueBindingTypeInExpression) {
  EXPECT_THAT(
      R"(#![feature(generics)]
enum E<T: type, V: T> { Value(uN[V + u32:1]) }
const X = E<u32, u32:7>::Value(u8:1);
)",
      TypecheckSucceeds(HasNodeWithType("X", "E { Value(uN[8]) }")));
}

TEST(TypecheckV2Test, SemanticSumPartiallyExplicitTupleParametrics) {
  EXPECT_THAT(
      R"(#![feature(generics)]
enum TuplePair<N: u32, M: u32> { V(uN[N], uN[M]) }
const TUPLE = TuplePair<u32:8>::V(u8:1, u16:2);
)",
      TypecheckSucceeds(
          HasNodeWithType("TUPLE", "TuplePair { V(uN[8], uN[16]) }")));
}

TEST(TypecheckV2Test, SemanticSumPartiallyExplicitNamedParametrics) {
  EXPECT_THAT(
      R"(#![feature(generics)]
enum NamedPair<N: u32, M: u32> { V { x: uN[N], y: uN[M] } }
const NAMED = NamedPair<u32:8>::V { y: u16:2, x: u8:1 };
)",
      TypecheckSucceeds(
          HasNodeWithType("NAMED", "NamedPair { V { x: uN[8], y: uN[16] } }")));
}

TEST(TypecheckV2Test, SemanticSumAliasArgumentKeepsPayloadInference) {
  EXPECT_THAT(
      R"(#![feature(generics)]
type U = u32;
enum E<A: u32, B: u32> { V(uN[A], uN[B]) }
const X = E<U:8>::V(u8:1, u16:2);
)",
      TypecheckSucceeds(HasNodeWithType("X", "E { V(uN[8], uN[16]) }")));
}

TEST(TypecheckV2Test, ImportedSemanticSumPartiallyExplicitNamedParametrics) {
  constexpr std::string_view kImported = R"(#![feature(generics)]
pub enum E<N: u32, M: u32> { V { x: uN[N], y: uN[M] } }
)";
  constexpr std::string_view kProgram = R"(#![feature(generics)]
import imported;
const X = imported::E<u32:8>::V { x: u8:1, y: u16:2 };
)";
  ImportData import_data = CreateImportDataForTest();
  XLS_ASSERT_OK(TypecheckV2(kImported, "imported", &import_data));
  EXPECT_THAT(TypecheckV2(kProgram, "main", &import_data),
              IsOkAndHolds(HasTypeInfo(
                  HasNodeWithType("X", "E { V { x: uN[8], y: uN[16] } }"))));
}

TEST(TypecheckV2Test, SemanticSumConstructorInGenericFunction) {
  EXPECT_THAT(
      R"(#![feature(generics)]
enum E<N: u32> { V(uN[N]) }
fn wrap<N: u32>(x: uN[N]) -> E<N> { E::V(x) }
fn f() -> (E<u32:8>, E<u32:16>) {
  let a = wrap(u8:1);
  let b = wrap(u16:2);
  (a, b)
}
)",
      TypecheckSucceeds(AllOf(HasNodeWithType("a", "E { V(uN[8]) }"),
                              HasNodeWithType("b", "E { V(uN[16]) }"))));
}

TEST(TypecheckV2Test, SemanticSumConstructorWithImportedTypeArgument) {
  constexpr std::string_view kImported = R"(
pub enum Kind: u8 { A = 0 }
)";
  constexpr std::string_view kPrograms[] = {
      R"(#![feature(generics)]
import imported;
enum E<T: type> { A(T) }
const X = E<imported::Kind>::A(imported::Kind::A);
)",
      R"(#![feature(generics)]
import imported;
type K = imported::Kind;
enum E<T: type> { A(T) }
const X = E<K>::A(imported::Kind::A);
)",
      R"(#![feature(generics)]
import imported;
enum E<T: type, N: u32> { A(T, uN[N]) }
const X = E<imported::Kind>::A(imported::Kind::A, u8:0);
)",
  };
  for (std::string_view program : kPrograms) {
    SCOPED_TRACE(program);
    ImportData import_data = CreateImportDataForTest();
    XLS_ASSERT_OK(TypecheckV2(kImported, "imported", &import_data));
    XLS_ASSERT_OK_AND_ASSIGN(TypecheckResult result,
                             TypecheckV2(program, "main", &import_data));
    XLS_ASSERT_OK_AND_ASSIGN(ConstantDef * constant,
                             result.tm.module->GetConstantDef("X"));
    EXPECT_EQ(constant->value()->kind(), AstNodeKind::kSumInstance);
  }
}

TEST(TypecheckV2Test, SemanticSumValueArgumentUsesEarlierExplicitBinding) {
  EXPECT_THAT(
      R"(#![feature(generics)]
enum E<N: u32, V: uN[N]> { A(u8) }
const X = E<u32:8, u8:1>::A(u8:0);
)",
      TypecheckSucceeds(HasNodeWithType("X", "E { A(uN[8]) }")));
}

TEST(TypecheckV2Test,
     PartialSemanticSumValueArgumentUsesEarlierExplicitBinding) {
  EXPECT_THAT(
      R"(#![feature(generics)]
enum E<N: u32, V: uN[N], M: u32> { A(uN[M]) }
const X = E<u32:8, u8:1>::A(u16:0);
)",
      TypecheckSucceeds(HasNodeWithType("X", "E { A(uN[16]) }")));
}

TEST(TypecheckV2Test, SemanticSumValueArgumentRejectsWrongWidth) {
  constexpr std::string_view kPrograms[] = {
      R"(#![feature(generics)]
enum E<N: u32, V: uN[N]> { A(u8) }
const X = E<u32:8, u16:1>::A(u8:0);
)",
      R"(#![feature(generics)]
enum E<N: u32, V: uN[N], M: u32> { A(uN[M]) }
const X = E<u32:8, u16:1>::A(u16:0);
)",
  };
  for (std::string_view program : kPrograms) {
    SCOPED_TRACE(program);
    EXPECT_THAT(program, TypecheckFails(HasSubstr("size mismatch")));
  }
}

TEST(TypecheckV2Test, GenericSemanticSumReportsMissingPayloadArgument) {
  EXPECT_THAT(
      R"(#![feature(generics)]
enum E<N: u32> { V(uN[N]) }
const X = E::V();
)",
      TypecheckFails(HasSubstr("Expected 1 argument(s) but got 0.")));
}

TEST(TypecheckV2Test, GenericSemanticSumChecksDeclaredTagWidth) {
  EXPECT_THAT(
      R"(#![feature(generics)]
enum E<N: u32>: u1 { V(uN[N]) = 2 }
const X = E<u32:8>::V(u8:0);
)",
      TypecheckFails(HasSubstr("size mismatch: u2 vs. u1")));
}

TEST(TypecheckV2Test, GenericSemanticSumChecksTagWidthPerInstantiation) {
  EXPECT_THAT(
      R"(#![feature(generics)]
enum E<N: u32>: uN[N] { A(u8) = 0, B(u8) = 2 }
const WIDE = E<u32:2>::A(u8:0);
const NARROW = E<u32:1>::A(u8:0);
)",
      TypecheckFails(HasSubstr("size mismatch: u2 vs. uN[1]")));
}

TEST(TypecheckV2Test, GenericSemanticSumChecksDuplicateTagsPerInstantiation) {
  EXPECT_THAT(
      R"(#![feature(generics)]
enum E<N: u32>: u32 { A(u8) = 0, B(u8) = N }
const DISTINCT = E<u32:1>::A(u8:0);
const DUPLICATE = E<u32:0>::A(u8:0);
)",
      TypecheckFails(HasSubstr("duplicate discriminant")));
}

TEST(TypecheckV2Test, GenericSemanticSumSubstitutesPrimitiveTypeMembers) {
  EXPECT_THAT(
      R"(#![feature(generics)]
enum E<T: type>: T { A(u8) = T::ZERO, B(u8) = T::MAX }
fn f(x: E<u8>) -> E<u8> { x }
fn g(x: E<uN[16]>) -> E<uN[16]> { x }
)",
      TypecheckSucceeds(::testing::_));
}

TEST(TypecheckV2Test, GenericSemanticSumPreservesNominalTypeMembers) {
  EXPECT_THAT(
      R"(#![feature(generics)]
enum Tag: u8 { FIRST = 0, LAST = 1 }
enum E<T: type>: u8 { A(u8) = T::FIRST as u8, B(u8) = T::LAST as u8 }
fn f(x: E<Tag>) -> E<Tag> { x }
)",
      TypecheckSucceeds(::testing::_));
}

TEST(TypecheckV2Test, ImportedSemanticSumSubstitutesTypesInDiscriminants) {
  constexpr std::string_view kImported = R"(#![feature(generics)]
pub enum E<T: type>: T { A(u8) = T:0, B(u8) = T:1 }
)";
  constexpr std::string_view kProgram = R"(#![feature(generics)]
import imported;
const NARROW = imported::E<u8>::A(u8:0);
const WIDE = imported::E<u16>::B(u8:1);
)";
  ImportData import_data = CreateImportDataForTest();
  XLS_ASSERT_OK(TypecheckV2(kImported, "imported", &import_data));
  EXPECT_THAT(TypecheckV2(kProgram, "main", &import_data),
              IsOkAndHolds(HasTypeInfo(AllOf(
                  HasNodeWithType("NARROW", "E { A(uN[8]) | B(uN[8]) }"),
                  HasNodeWithType("WIDE", "E { A(uN[8]) | B(uN[8]) }")))));
}

TEST(TypecheckV2Test, SemanticSumReferenceRequiresUnboundParametrics) {
  EXPECT_THAT(
      R"(#![feature(generics)]
enum Box<T: type> { Value(T) }
fn f(x: Box) -> Box { x }
)",
      TypecheckFails(HasSubstr("must have all parametrics specified")));
}

TEST(TypecheckV2Test, SemanticSumUnifiesEquivalentTypeArgumentSpellings) {
  EXPECT_THAT(
      R"(#![feature(generics)]
enum Box<T: type> { Value(T) }
fn f(x: Box<bits[8]>) -> Box<u8> { x }
)",
      TypecheckSucceeds(::testing::_));
}

TEST(TypecheckV2Test,
     SemanticSumSharedPayloadInferenceIgnoresConstructorOrder) {
  constexpr std::string_view kType = "E { None | Some(uN[8]) }";
  EXPECT_THAT(
      R"(#![feature(generics)]
enum E<N: u32> { None, Some(uN[N]) }
fn f(b: bool) {
  let first = if b { E::Some(u8:0) } else { E::None };
  let last = if b { E::None } else { E::Some(u8:0) };
  let array_first = [E::Some(u8:0), E::None];
  let array_last = [E::None, E::Some(u8:0)];
}
)",
      TypecheckSucceeds(
          AllOf(HasNodeWithType("first", kType), HasNodeWithType("last", kType),
                HasNodeWithType("array_first", "E { None | Some(uN[8]) }[2]"),
                HasNodeWithType("array_last", "E { None | Some(uN[8]) }[2]"))));
}

TEST(TypecheckV2Test, SemanticSumSharedPayloadsJointlyInferParameters) {
  constexpr std::string_view kType =
      "E { Tuple(uN[8]) | Named { value: uN[16] } }";
  EXPECT_THAT(R"(#![feature(generics)]
enum E<N: u32, M: u32> { Tuple(uN[N]), Named { value: uN[M] } }
fn f(b: bool) {
  let tuple_first = if b { E::Tuple(u8:0) } else { E::Named { value: u16:0 } };
  let named_first = if b { E::Named { value: u16:0 } } else { E::Tuple(u8:0) };
}
)",
              TypecheckSucceeds(AllOf(HasNodeWithType("tuple_first", kType),
                                      HasNodeWithType("named_first", kType))));
}

TEST(TypecheckV2Test, SemanticSumSharedPayloadConstraintsStillRejectConflicts) {
  for (std::string_view expression : {
           "if b { E::Some(u8:0) } else { E::Some(u16:0) }",
           "if b { E::Some(u16:0) } else { E::Some(u8:0) }",
           "if b { E::Some(u8:0) } else { if b { E::None } else { "
           "E::Some(u16:0) } }",
           "if b { if b { E::None } else { E::Some(u16:0) } } else { "
           "E::Some(u8:0) }",
       }) {
    SCOPED_TRACE(expression);
    EXPECT_THAT(absl::Substitute(R"(#![feature(generics)]
enum E<N: u32> { None, Some(uN[N]) }
fn f(b: bool) { let _ = $0; }
)",
                                 expression),
                TypecheckFails(::testing::AnyOf(
                    HasSubstr("size mismatch"),
                    HasSubstr("Value mismatch for parametric"))));
  }
  EXPECT_THAT(R"(#![feature(generics)]
enum E<N: u32> { None, Some(uN[N]) }
fn f(b: bool) { let _ = if b { E::None } else { E::None }; }
)",
              TypecheckFails(HasSubstr("must have all parametrics specified")));
}

TEST(TypecheckV2Test, SemanticSumPayloadsRequireValuesThroughExpressions) {
  constexpr std::string_view kImported = R"(#![feature(generics)]
pub type Word = u32;
pub type Flag = bool;
pub const VALUE = u32:1;
pub const FLAG = true;
pub struct S { x: u32 }
)";
  // Each control changes only the type operand to a real value. These
  // expressions must not erase the distinction before payload validation.
  constexpr std::pair<std::string_view, std::string_view> kExpressions[] = {
      {"{ imported::Word }", "{ imported::VALUE }"},
      {"{ let x = imported::Word; x }", "{ let x = imported::VALUE; x }"},
      {"{ const X = imported::Word; X }", "{ const X = imported::VALUE; X }"},
      {"match true { _ => imported::Word }",
       "match true { _ => imported::VALUE }"},
      {"(imported::Word, u32:0).0", "(imported::VALUE, u32:0).0"},
      {"!imported::Word", "!imported::VALUE"},
      {"imported::Word + u32:1", "imported::VALUE + u32:1"},
      {"u32:1 + imported::Word", "u32:1 + imported::VALUE"},
      {"if imported::Flag { u32:1 } else { u32:0 }",
       "if imported::FLAG { u32:1 } else { u32:0 }"},
      {"match imported::Word { _ => u32:1 }",
       "match imported::VALUE { _ => u32:1 }"},
      {"for (_, x): (u32, u32) in u32:0..u32:1 { x }(imported::Word)",
       "for (_, x): (u32, u32) in u32:0..u32:1 { x }(imported::VALUE)"},
      {"imported::S.x", "(imported::S { x: u32:1 }).x"},
      {"(imported::S { x: imported::Word }).x",
       "(imported::S { x: imported::VALUE }).x"},
      {"(imported::S { x: imported::Word, ..imported::S { x: u32:0 } }).x",
       "(imported::S { x: imported::VALUE, ..imported::S { x: u32:0 } }).x"},
      {"(imported::S { ..imported::S }).x",
       "(imported::S { ..imported::S { x: u32:1 } }).x"},
  };
  for (const auto& [type_expression, value_expression] : kExpressions) {
    for (bool use_value : {false, true}) {
      std::string_view expression =
          use_value ? value_expression : type_expression;
      SCOPED_TRACE(expression);
      ImportData import_data = CreateImportDataForTest();
      XLS_ASSERT_OK(TypecheckV2(kImported, "imported", &import_data));
      const std::string program = absl::Substitute(R"(#![feature(generics)]
import imported;
enum E { V(u32) }
fn f() -> E { E::V($0) }
)",
                                                   expression);
      if (use_value) {
        EXPECT_THAT(TypecheckV2(program, "main", &import_data),
                    IsOkAndHolds(HasTypeInfo(
                        HasNodeWithType("f", "() -> E { V(uN[32]) }"))));
      } else {
        EXPECT_THAT(TypecheckV2(program, "main", &import_data),
                    StatusIs(absl::StatusCode::kInvalidArgument,
                             HasSubstr("Cannot use a type as a value.")));
      }
    }
  }
}

TEST(TypecheckV2Test, SemanticSumWrappedEnumPayloadsKeepTypeAndValueDistinct) {
  constexpr std::string_view kImported = "pub enum Tag: u8 { A = 0 }";
  for (std::string_view expression :
       {"E::Tuple({ { imported::Tag } })",
        "E::Named { value: { imported::Tag } }"}) {
    SCOPED_TRACE(expression);
    ImportData import_data = CreateImportDataForTest();
    XLS_ASSERT_OK(TypecheckV2(kImported, "imported", &import_data));
    EXPECT_THAT(TypecheckV2(absl::Substitute(R"(
import imported;
enum E { Tuple(imported::Tag), Named { value: imported::Tag } }
fn f() -> E { $0 }
)",
                                             expression),
                            "main", &import_data),
                StatusIs(absl::StatusCode::kInvalidArgument,
                         HasSubstr("Cannot use a type as a value.")));
  }
  ImportData import_data = CreateImportDataForTest();
  XLS_ASSERT_OK(TypecheckV2(kImported, "imported", &import_data));
  EXPECT_THAT(TypecheckV2(R"(#![feature(generics)]
import imported;
enum E { Tuple(imported::Tag), Named { value: imported::Tag } }
fn identity<T: type>(x: T) -> T { x }
fn tuple() -> E { E::Tuple({ { imported::Tag::A } }) }
fn named() -> E { E::Named { value: { imported::Tag::A } } }
fn argument() -> E { E::Tuple(identity<imported::Tag>(imported::Tag::A)) }
)",
                          "main", &import_data),
              IsOkAndHolds(::testing::_));
}

TEST(TypecheckV2Test, SemanticSumRejectsDifferentSameNamedTypeArguments) {
  EXPECT_THAT(
      R"(#![feature(generics)]
enum E<A: type> { Unit, Value(u8) }
fn f(condition: bool) {
  let _ = if condition {
    type T = u8;
    E<T>::Unit
  } else {
    type T = u16;
    E<T>::Unit
  };
}
)",
      TypecheckFails(HasSubstr("Value mismatch for parametric")));
}

TEST(TypecheckV2Test, ImportedSemanticSumResolvesDependentDefaults) {
  constexpr std::string_view kImported = R"(#![feature(generics)]
pub enum Box<N: u32 = {u32:8}, T: type = uN[N], V: T = {T:0}> { Value(T) }
)";
  constexpr std::string_view kProgram = R"(#![feature(generics)]
import imported;
fn f(x: imported::Box) -> imported::Box { x }
const DEFAULT = imported::Box::Value(u8:1);
const OVERRIDE = imported::Box<u32:16>::Value(u16:2);
)";
  ImportData import_data = CreateImportDataForTest();
  XLS_ASSERT_OK(TypecheckV2(kImported, "imported", &import_data));
  EXPECT_THAT(TypecheckV2(kProgram, "main", &import_data),
              IsOkAndHolds(HasTypeInfo(AllOf(
                  HasNodeWithType("DEFAULT", "Box { Value(uN[8]) }"),
                  HasNodeWithType("OVERRIDE", "Box { Value(uN[16]) }")))));
}

TEST(TypecheckV2Test, SemanticSumConstructorPreservesShadowedTypeAlias) {
  EXPECT_THAT(
      R"(
type T = u8;
enum E { V(u16) }
fn f() -> E {
  type T = u16;
  E::V(T:0)
}
)",
      TypecheckSucceeds(HasNodeWithType("E::V(T:0)", "E { V(uN[16]) }")));
}

TEST(TypecheckV2Test, SemanticSumNestedPayloadConstructorsCanonicalize) {
  XLS_ASSERT_OK_AND_ASSIGN(TypecheckResult result, TypecheckV2(R"(
enum E { V(u32) }
fn f() -> E {
  E::V({ let inner = E::V(u32:0); u32:1 })
}
)"));
  int instances = 0;
  for (const AstNode* node : FlattenToSet(result.tm.module)) {
    if (node->kind() == AstNodeKind::kSumInstance) {
      ++instances;
      const auto* instance = absl::down_cast<const SumInstance*>(node);
      EXPECT_EQ(instance->constructor_ref()->parent(), instance);
      EXPECT_EQ(instance->tuple_payload_args().front()->parent(), instance);
    } else {
      EXPECT_NE(node->kind(), AstNodeKind::kInvocation);
    }
  }
  EXPECT_EQ(instances, 2);
}

TEST(TypecheckV2Test, ImportedSemanticSumShapesPreserveParens) {
  constexpr std::string_view kImported = R"(
pub enum E {
  Unit,
  Tuple(u32),
  Struct { value: u32 },
  EmptyTuple(),
  EmptyStruct {},
}
)";
  constexpr std::string_view kProgram = R"(
import imported;
const UNIT = (imported::E::Unit);
const TUPLE = (imported::E::Tuple(u32:1));
const STRUCT = (imported::E::Struct { value: u32:2 });
const EMPTY_TUPLE = (imported::E::EmptyTuple());
const EMPTY_STRUCT = (imported::E::EmptyStruct {});
)";
  ImportData import_data = CreateImportDataForTest();
  XLS_ASSERT_OK(TypecheckV2(kImported, "imported", &import_data));
  XLS_ASSERT_OK_AND_ASSIGN(TypecheckResult result,
                           TypecheckV2(kProgram, "main", &import_data));
  struct Expected {
    std::string_view name;
    SumInstance::PayloadShape shape;
    std::string_view text;
  };
  const Expected expected[] = {
      {"UNIT", SumInstance::PayloadShape::kUnit, "(imported::E::Unit)"},
      {"TUPLE", SumInstance::PayloadShape::kTuple,
       "(imported::E::Tuple(u32:1))"},
      {"STRUCT", SumInstance::PayloadShape::kStruct,
       "(imported::E::Struct { value: u32:2 })"},
      {"EMPTY_TUPLE", SumInstance::PayloadShape::kTuple,
       "(imported::E::EmptyTuple())"},
      {"EMPTY_STRUCT", SumInstance::PayloadShape::kStruct,
       "(imported::E::EmptyStruct {})"},
  };
  for (const auto& [name, shape, text] : expected) {
    XLS_ASSERT_OK_AND_ASSIGN(ConstantDef * constant,
                             result.tm.module->GetConstantDef(name));
    ASSERT_EQ(constant->value()->kind(), AstNodeKind::kSumInstance);
    const auto* instance =
        absl::down_cast<const SumInstance*>(constant->value());
    EXPECT_EQ(instance->payload_shape(), shape);
    EXPECT_TRUE(instance->in_parens());
    EXPECT_EQ(instance->ToString(), text);
    EXPECT_EQ(instance->owner(), result.tm.module);
    EXPECT_EQ(instance->constructor_ref()->owner(), result.tm.module);
    EXPECT_EQ(instance->constructor_ref()->parent(), instance);
    ASSERT_TRUE(std::holds_alternative<ColonRef*>(
        instance->constructor_ref()->subject()));
    EXPECT_EQ(
        std::get<ColonRef*>(instance->constructor_ref()->subject())->ToString(),
        "imported::E");
  }
}

TEST(TypecheckV2Test, SemanticSumRejectsWrongZeroPayloadShape) {
  EXPECT_THAT(R"(
enum E { Unit, EmptyTuple(), EmptyStruct {} }
fn f() -> E { E::Unit() }
)",
              TypecheckFails(HasSubstr("is not callable here")));
  EXPECT_THAT(R"(
enum E { Unit, EmptyTuple(), EmptyStruct {} }
fn f() -> E { E::EmptyTuple {} }
)",
              TypecheckFails(HasSubstr("Attempted to instantiate non-struct")));
}

TEST(TypecheckV2Test, SemanticSumPrimitiveMemberPayloadTypeAndControl) {
  // This declaration-only case used to abort during annotation substitution;
  // replacing just its dependent width with u8 is the accepting control.
  for (std::string_view payload : {"uN[T::ZERO + u32:8]", "u8"}) {
    SCOPED_TRACE(payload);
    EXPECT_THAT(absl::Substitute(R"(#![feature(generics)]
enum E<T: type> { V($0) }
fn f(x: E<u32>) -> E<u32> { x }
)",
                                 payload),
                TypecheckSucceeds(HasNodeWithType(
                    "f", "(E { V(uN[8]) }) -> E { V(uN[8]) }")));
  }
}

TEST(TypecheckV2Test, SemanticSumPrimitiveMemberConstructorPayloads) {
  constexpr std::string_view k8 =
      "E { Tuple(uN[8], uN[8]) | Named { width: uN[8], value: uN[8] } }";
  constexpr std::string_view k16 =
      "E { Tuple(uN[16], uN[16]) | Named { width: uN[16], value: uN[16] } }";
  // The first payload supplies N in the partial cases. T is always explicit
  // or supplied by context, rather than inferred from its member's value.
  EXPECT_THAT(
      R"(#![feature(generics)]
enum E<T: type, N: u32> {
  Tuple(uN[N], uN[T::ZERO + N]),
  Named { width: uN[N], value: uN[T::ZERO + N] },
}
const EXPLICIT_TUPLE = E<u32, u32:8>::Tuple(u8:0, u8:1);
const PARTIAL_TUPLE = E<u32>::Tuple(u8:0, u8:2);
const CONTEXTUAL_TUPLE: E<u32, u32:16> = E::Tuple(u16:0, u16:3);
const EXPLICIT_NAMED = E<u32, u32:8>::Named { width: u8:0, value: u8:1 };
const PARTIAL_NAMED = E<u32>::Named { width: u8:0, value: u8:2 };
const CONTEXTUAL_NAMED: E<u32, u32:16> = E::Named { width: u16:0, value: u16:3 };
)",
      TypecheckSucceeds(AllOf(HasNodeWithType("EXPLICIT_TUPLE", k8),
                              HasNodeWithType("PARTIAL_TUPLE", k8),
                              HasNodeWithType("CONTEXTUAL_TUPLE", k16),
                              HasNodeWithType("EXPLICIT_NAMED", k8),
                              HasNodeWithType("PARTIAL_NAMED", k8),
                              HasNodeWithType("CONTEXTUAL_NAMED", k16))));
}

TEST(TypecheckV2Test, ImportedSemanticSumPrimitiveMemberPayloadAliases) {
  constexpr std::string_view kImported = R"(#![feature(type_inference_v2)]
#![feature(generics)]
pub type Word = u32;
pub enum E<T: type> {
  Tuple(uN[T::ZERO + u32:8]),
  Named { value: uN[T::ZERO + u32:8] },
}
)";
  constexpr std::string_view kProgram = R"(#![feature(generics)]
import imported;
type Alias = imported::E<imported::Word>;
type OtherWord = uN[32];
type OtherAlias = imported::E<OtherWord>;
const TUPLE: Alias = Alias::Tuple(u8:1);
const NAMED: Alias = Alias::Named { value: u8:2 };
const EQUIVALENT: OtherAlias = Alias::Tuple(u8:3);
)";
  absl::flat_hash_map<std::filesystem::path, std::string> files = {
      {"/imported.x", std::string(kImported)},
  };
  ImportData import_data = CreateImportDataForTest(
      std::make_unique<FakeFilesystem>(std::move(files), "/"));
  constexpr std::string_view kType =
      "E { Tuple(uN[8]) | Named { value: uN[8] } }";
  EXPECT_THAT(
      TypecheckV2(kProgram, "main", &import_data),
      IsOkAndHolds(HasTypeInfo(AllOf(HasNodeWithType("TUPLE", kType),
                                     HasNodeWithType("NAMED", kType),
                                     HasNodeWithType("EQUIVALENT", kType)))));
}

TEST(TypecheckV2Test, SemanticSumDimensionedPrimitiveMemberPayloads) {
  // MAX depends on the actual primitive width: uN[3] yields 7 and uN[4]
  // yields 15. Cast before addition so the arithmetic itself remains u32.
  EXPECT_THAT(R"(#![feature(generics)]
enum E<T: type> { V(uN[(T::MAX as u32) + u32:1]) }
fn f(x: E<uN[3]>) -> E<uN[3]> { x }
const EXPLICIT = E<uN[3]>::V(u8:0);
const CONTEXTUAL: E<uN[4]> = E::V(u16:0);
)",
              TypecheckSucceeds(AllOf(
                  HasNodeWithType("f", "(E { V(uN[8]) }) -> E { V(uN[8]) }"),
                  HasNodeWithType("EXPLICIT", "E { V(uN[8]) }"),
                  HasNodeWithType("CONTEXTUAL", "E { V(uN[16]) }"))));
}

TEST(TypecheckV2Test, SemanticSumPrimitiveMemberKeepsCallerWidth) {
  EXPECT_THAT(R"(#![feature(generics)]
enum E<T: type> { V(uN[(T::MAX as u32) + u32:1]) }
fn make<N: u32>(x: uN[u32:1 << N]) -> E<uN[N]> { E<uN[N]>::V(x) }
fn narrow(x: u8) -> E<uN[3]> { make<u32:3>(x) }
fn wide(x: u16) -> E<uN[4]> { make<u32:4>(x) }
)",
              TypecheckSucceeds(AllOf(
                  HasNodeWithType("narrow", "(uN[8]) -> E { V(uN[8]) }"),
                  HasNodeWithType("wide", "(uN[16]) -> E { V(uN[16]) }"))));
}

TEST(TypecheckV2Test, SemanticSumPrimitiveMemberRejectsUnknownMember) {
  EXPECT_THAT(R"(#![feature(generics)]
enum E<T: type> { V(uN[T::MISSING + u32:8]) }
fn f(x: E<u32>) -> E<u32> { x }
)",
              TypecheckFails(HasSubstr("does not have attribute 'MISSING'")));
}

TEST(TypecheckV2Test, SemanticSumPrimitiveMemberRejectsNonBitsType) {
  EXPECT_THAT(R"(#![feature(generics)]
enum E<T: type> { V(uN[(T::ZERO as u32) + u32:8]) }
fn f(x: E<(u8, u8)>) -> E<(u8, u8)> { x }
)",
              TypecheckFails(HasSubstr("has no member `ZERO`")));
}

}  // namespace
}  // namespace xls::dslx
