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
#include "absl/strings/str_cat.h"
#include "absl/strings/substitute.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "xls/common/status/matchers.h"
#include "xls/dslx/bytecode/bytecode.h"
#include "xls/dslx/bytecode/bytecode_emitter.h"
#include "xls/dslx/constexpr_evaluator.h"
#include "xls/dslx/create_import_data.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/frontend/ast_utils.h"
#include "xls/dslx/frontend/module.h"
#include "xls/dslx/frontend/semantics_analysis.h"
#include "xls/dslx/import_data.h"
#include "xls/dslx/interp_value.h"
#include "xls/dslx/ir_convert/convert_options.h"
#include "xls/dslx/parse_and_typecheck.h"
#include "xls/dslx/type_system/type.h"
#include "xls/dslx/type_system/type_info.h"
#include "xls/dslx/type_system/typecheck_test_utils.h"
#include "xls/dslx/type_system_v2/import_utils.h"
#include "xls/dslx/type_system_v2/inference_table.h"
#include "xls/dslx/type_system_v2/matchers.h"
#include "xls/dslx/type_system_v2/populate_table_visitor.h"
#include "xls/dslx/type_system_v2/type_system_test_utils.h"
#include "xls/dslx/type_system_v2/typecheck_module_v2.h"
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
using ::testing::Not;

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
      TypecheckFails(HasSubstr(
          "Match expressions over semantic sums are not supported yet")));
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

TEST(TypecheckV2Test, ImportedSemanticSumZeroConstantRemainsBytecodeLiteral) {
  constexpr std::string_view kImported = R"(
pub enum E { Unit, Payload(u32) }
pub const ZERO = zero!<E>();
)";
  constexpr std::string_view kProgram = R"(
import imported;
const X = imported::ZERO;
)";
  ImportData import_data = CreateImportDataForTest();
  XLS_ASSERT_OK_AND_ASSIGN(TypecheckResult imported,
                           TypecheckV2(kImported, "imported", &import_data));
  XLS_ASSERT_OK_AND_ASSIGN(ConstantDef * zero,
                           imported.tm.module->GetConstantDef("ZERO"));
  XLS_ASSERT_OK_AND_ASSIGN(InterpValue value,
                           imported.tm.type_info->GetConstExpr(zero));
  XLS_ASSERT_OK_AND_ASSIGN(TypecheckResult result,
                           TypecheckV2(kProgram, "main", &import_data));
  XLS_ASSERT_OK_AND_ASSIGN(ConstantDef * constant,
                           result.tm.module->GetConstantDef("X"));
  ASSERT_EQ(constant->value()->kind(), AstNodeKind::kColonRef);
  EXPECT_THAT(
      ConstexprEvaluator::EvaluateToValue(&import_data, result.tm.type_info,
                                          /*warning_collector=*/nullptr,
                                          /*bindings=*/{}, constant->value()),
      IsOkAndHolds(value));
  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<BytecodeFunction> bytecode,
      BytecodeEmitter::EmitExpression(&import_data, result.tm.type_info,
                                      constant->value(), /*env=*/{},
                                      /*caller_bindings=*/std::nullopt));
  ASSERT_EQ(bytecode->bytecodes().size(), 1);
  EXPECT_EQ(bytecode->bytecodes().front().op(), Bytecode::Op::kLiteral);
  EXPECT_THAT(bytecode->bytecodes().front().value_data(), IsOkAndHolds(value));
}

TEST(TypecheckV2Test, GenericSemanticSumDeclarationNeedsNoInstantiation) {
  EXPECT_THAT(
      R"(#![feature(generics)]
enum Box<T: type> { Value(T) }
)",
      TypecheckSucceeds(::testing::_));
}

TEST(TypecheckV2Test, GenericSemanticSumChecksDeclaredTagWidth) {
  EXPECT_THAT(
      R"(#![feature(generics)]
enum E<N: u32>: u1 { V(uN[N]) = 2 }
const X = zero!<E<u32:8>>();
)",
      TypecheckFails(HasSubstr("size mismatch: u2 vs. u1")));
}

TEST(TypecheckV2Test, GenericSemanticSumChecksTagWidthPerInstantiation) {
  EXPECT_THAT(
      R"(#![feature(generics)]
enum E<N: u32>: uN[N] { A(u8) = 0, B(u8) = 2 }
const WIDE = zero!<E<u32:2>>();
const NARROW = zero!<E<u32:1>>();
)",
      TypecheckFails(HasSubstr("size mismatch: u2 vs. uN[1]")));
}

TEST(TypecheckV2Test, GenericSemanticSumChecksDuplicateTagsPerInstantiation) {
  EXPECT_THAT(
      R"(#![feature(generics)]
enum E<N: u32>: u32 { A(u8) = 0, B(u8) = N }
const DISTINCT = zero!<E<u32:1>>();
const DUPLICATE = zero!<E<u32:0>>();
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
const NARROW = zero!<imported::E<u8>>();
const WIDE = zero!<imported::E<u16>>();
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

TEST(TypecheckV2Test, SemanticSumRejectsDifferentSameNamedTypeArguments) {
  EXPECT_THAT(
      R"(#![feature(generics)]
enum E<A: type> { Unit, Value(u8) }
fn f(condition: bool) {
  let _ = if condition {
    type T = u8;
    zero!<E<T>>()
  } else {
    type T = u16;
    zero!<E<T>>()
  };
}
)",
      TypecheckFails(HasSubstr("Value mismatch for parametric")));
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
