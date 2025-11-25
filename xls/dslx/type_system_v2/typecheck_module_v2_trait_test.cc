// Copyright 2025 The XLS Authors
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
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xls/common/status/matchers.h"
#include "xls/common/status/status_macros.h"
#include "xls/dslx/create_import_data.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/frontend/pos.h"
#include "xls/dslx/import_data.h"
#include "xls/dslx/type_system/type.h"
#include "xls/dslx/type_system/typecheck_test_utils.h"
#include "xls/dslx/type_system_v2/matchers.h"
#include "xls/dslx/type_system_v2/trait_deriver_dispatcher.h"
#include "xls/dslx/type_system_v2/type_annotation_utils.h"

namespace xls::dslx {
namespace {

class HasNonzeroFooFieldDeriver : public TraitDeriver {
 public:
  absl::StatusOr<StatementBlock*> DeriveFunctionBody(
      Module& module, const Trait& trait, const StructDef& struct_def,
      const StructType&, const Function& function) override {
    Param* self_param = function.params()[0];
    Expr* self = module.Make<NameRef>(Span::None(), self_param->identifier(),
                                      self_param->name_def());
    Expr* attr = module.Make<Attr>(Span::None(), self, "foo");
    Expr* zero = module.Make<Number>(Span::None(), "0", NumberKind::kOther,
                                     nullptr, false);
    Expr* ne = module.Make<Binop>(Span::None(), BinopKind::kNe, attr, zero,
                                  Span::None(), false);
    Statement* statement = module.Make<Statement>(ne);
    return module.Make<StatementBlock>(Span::None(),
                                       std::vector<Statement*>{statement},
                                       /*trailing_semi=*/false);
  }
};

class ContainsDeriver : public TraitDeriver {
 public:
  absl::StatusOr<StatementBlock*> DeriveFunctionBody(
      Module& module, const Trait& trait, const StructDef& struct_def,
      const StructType&, const Function& function) override {
    std::optional<Expr*> expr;
    if (struct_def.members().empty()) {
      expr = module.Make<Number>(Span::None(), "false", NumberKind::kBool,
                                 CreateBoolAnnotation(module, Span::None()),
                                 false);
    } else {
      Param* self_param = function.params()[0];
      Param* value_param = function.params()[1];
      Expr* self = module.Make<NameRef>(Span::None(), self_param->identifier(),
                                        self_param->name_def());
      Expr* arg = module.Make<NameRef>(Span::None(), value_param->identifier(),
                                       value_param->name_def());
      for (const std::string& next : struct_def.GetMemberNames()) {
        Expr* attr = module.Make<Attr>(Span::None(), self, next);
        Expr* next_clause = module.Make<Binop>(Span::None(), BinopKind::kEq,
                                               attr, arg, Span::None(), false);
        if (expr.has_value()) {
          expr = module.Make<Binop>(Span::None(), BinopKind::kOr, *expr,
                                    next_clause, Span::None(), false);
        } else {
          expr = next_clause;
        }
      }
    }
    Statement* statement = module.Make<Statement>(*expr);
    return module.Make<StatementBlock>(Span::None(),
                                       std::vector<Statement*>{statement},
                                       /*trailing_semi=*/false);
  }
};

class HasArrayOfSize3Deriver : public TraitDeriver {
 public:
  absl::StatusOr<StatementBlock*> DeriveFunctionBody(
      Module& module, const Trait& trait, const StructDef& struct_def,
      const StructType& struct_type, const Function& function) override {
    Expr* result = nullptr;
    for (const std::unique_ptr<Type>& member : struct_type.members()) {
      if (member->IsArray()) {
        XLS_ASSIGN_OR_RETURN(int64_t size,
                             member->AsArray().size().GetAsInt64());
        if (size == 3) {
          result = module.Make<Number>(Span::None(), "true", NumberKind::kBool,
                                       nullptr, false);
        }
      }
    }

    if (result == nullptr) {
      result = module.Make<Number>(Span::None(), "false", NumberKind::kBool,
                                   nullptr, false);
    }

    Statement* statement = module.Make<Statement>(result);
    return module.Make<StatementBlock>(Span::None(),
                                       std::vector<Statement*>{statement},
                                       /*trailing_semi=*/false);
  }
};

std::unique_ptr<TraitDeriver> CreateTestDeriver() {
  auto dispatcher = std::make_unique<TraitDeriverDispatcher>();
  dispatcher->SetHandler("HasNonzeroFooField", "has_nonzero_foo",
                         std::make_unique<HasNonzeroFooFieldDeriver>());
  dispatcher->SetHandler("Contains", "contains",
                         std::make_unique<ContainsDeriver>());
  dispatcher->SetHandler("HasArrayOfSize3", "has_array_of_size_3",
                         std::make_unique<HasArrayOfSize3Deriver>());
  return dispatcher;
}

TEST(TypecheckV2TraitTest, DeriveContains) {
  XLS_ASSERT_OK(TypecheckV2(R"(
trait Contains {
  fn contains(self, val: u32) -> bool;
}

#[derive(Contains)]
struct Foo {
  a: u32,
  foo: u32,
  c: u32
}

const_assert!(Foo {a: 5, foo: 6, c: 0}.contains(5));
const_assert!(Foo {a: 5, foo: 6, c: 0}.contains(6));
const_assert!(Foo {a: 0, foo: 5, c: 0}.contains(5));
const_assert!(!zero!<Foo>().contains(5));
)",
                            CreateTestDeriver()));
}

TEST(TypecheckV2TraitTest, DeriveMultiTraits) {
  XLS_ASSERT_OK(TypecheckV2(R"(
trait Contains {
  fn contains(self, val: u32) -> bool;
}

trait HasNonzeroFooField {
  fn has_nonzero_foo(self) -> bool;
}

#[derive(Contains, HasNonzeroFooField)]
struct Foo {
  a: u32,
  foo: u32,
  c: u32
}

const_assert!(Foo {a: 5, foo: 6, c: 0}.contains(5));
const_assert!(Foo {a: 1, foo: 5, c: 0}.has_nonzero_foo());
const_assert!(!zero!<Foo>().has_nonzero_foo());
)",
                            CreateTestDeriver()));
}

TEST(TypecheckV2TraitTest, DeriveUnknownTrait) {
  EXPECT_THAT(TypecheckV2(R"(
#[derive(Foobar)]
struct Foo {
  a: u32,
}

impl Foo {}

const C = Foo { a: 5 }.foobar();
)",
                          CreateTestDeriver()),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Unknown trait: `Foobar`")));
}

TEST(TypecheckV2TraitTest, DeriveClashingFunction) {
  EXPECT_THAT(
      TypecheckV2(R"(
trait Contains {
  fn contains(self, val: u32) -> bool;
}

#[derive(Contains)]
struct Foo {
  a: u32,
}

impl Foo {
    fn contains(self, val: u32) -> bool { true }
}

const_assert!(Foo { a: 3 }.contains(3));
)",
                  CreateTestDeriver()),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("Attempting to derive conflicting function `contains` "
                         "from trait `Contains`")));
}

TEST(TypecheckV2TraitTest, DeriveCoexistingWithHandwrittenImpl) {
  XLS_ASSERT_OK(TypecheckV2(R"(
trait Contains {
  fn contains(self, val: u32) -> bool;
}

#[derive(Contains)]
struct Foo {
  a: u32,
}

impl Foo {
    fn get_a_verbosely(self) -> u32 { self.a }
    fn check_contains_verbosely(self, val: u32) -> bool { self.contains(val) }
}

const F = Foo { a: 3 };
const_assert!(F.contains(3));
const_assert!(F.get_a_verbosely() == 3);
const_assert!(F.check_contains_verbosely(3));
)",
                            CreateTestDeriver()));
}

TEST(TypecheckV2TraitTest, DeriveHasArrayOfSize3) {
  XLS_ASSERT_OK(TypecheckV2(R"(
trait HasArrayOfSize3 {
  // Based on static knowledge of the concrete struct type.
  fn has_array_of_size_3(self) -> bool;
}

#[derive(HasArrayOfSize3)]
struct Foo<N: u32> {
  a: u32,
  b: u32[N],
  c: u32
}

#[derive(HasArrayOfSize3)]
struct Bar {
  field: s64[3]
}

const F1 = Foo {a: 5, b: [1], c: 0};
const F2 = Foo {a: 5, b: [1, 2, 3], c: 0};
const F3 = Bar { field: [1, 2, 3] };

const_assert!(!F1.has_array_of_size_3());
const_assert!(F2.has_array_of_size_3());
const_assert!(F3.has_array_of_size_3());
)",
                            CreateTestDeriver()));
}

TEST(TypecheckV2TraitTest, ToBitsSimple) {
  XLS_ASSERT_OK(TypecheckV2(R"(
#[derive(ToBits)]
struct Foo {
  a: u32,
}

#[derive(ToBits)]
struct Bar {
  a: s16,
  b: u8
}

const F1 = Foo {a: 5};
const F2 = Bar {a: 10, b: 1};

const_assert!(F1.to_bits() == u32:5);
const_assert!(F2.to_bits() == (u16:10 ++ u8:1));
)"));
}

TEST(TypecheckV2TraitTest, ToBitsWithTuple) {
  XLS_ASSERT_OK(TypecheckV2(R"(
#[derive(ToBits)]
struct Foo {
  a: u32,
  b: (s8, u64)
}

const F1 = Foo {a: 5, b: (-1, 0xfffffff)};
const F2 = Foo {a: 5, b: (60, 12345)};

const_assert!(F1.to_bits() == (u32:5 ++ u8:0xff ++ u64:0xfffffff));
const_assert!(F2.to_bits() == (u32:5 ++ u8:60 ++ u64:12345));
)"));
}

TEST(TypecheckV2TraitTest, ToBitsWithEnum) {
  XLS_ASSERT_OK(TypecheckV2(R"(
enum E : u8 {
  A = 10
}

#[derive(ToBits)]
struct Foo {
  a: u32,
  b: E
}

const F = Foo {a: 5, b: E::A};

const_assert!(F.to_bits() == (u32:5 ++ u8:10));
)"));
}

TEST(TypecheckV2TraitTest, ToBitsWithArray) {
  XLS_ASSERT_OK(TypecheckV2(R"(
#[derive(ToBits)]
struct Foo {
  a: u32,
  b: u8[3]
}

const F = Foo {a: 5, b: [5, 6, 7]};

const_assert!(F.to_bits() == (u32:5 ++ u8:5 ++ u8:6 ++ u8:7));
)"));
}

TEST(TypecheckV2TraitTest, ToBitsWithSubStruct) {
  XLS_ASSERT_OK(TypecheckV2(R"(
#[derive(ToBits)]
struct Foo {
  a: u32,
  b: u8[3]
}

#[derive(ToBits)]
struct Bar {
  f: Foo,
  g: s64
}

const B = Bar {f: Foo { a: 5, b: [5, 6, 7] }, g: 0xfffffffff};

const_assert!(B.to_bits() ==
    (u32:5 ++ u8:5 ++ u8:6 ++ u8:7 ++ u64:0xfffffffff));
)"));
}

TEST(TypecheckV2TraitTest, ToBitsOnImportedStruct) {
  constexpr std::string_view kImported = R"(
#[derive(ToBits)]
pub struct Foo {
  a: u32,
  b: u8[3]
}

#[derive(ToBits)]
pub struct Bar {
  f: Foo,
  g: s64
}
)";
  constexpr std::string_view kProgram = R"(
import imported;

const B1 = imported::Bar {
    f: imported::Foo { a: 5, b: [5, 6, 7] }, g: 0xfffffffff};
const B2 = imported::Bar {
    f: imported::Foo { a: 5, b: [5, 6, 7] }, g: 0xffffffff1};

const_assert!(B1.to_bits() ==
    (u32:5 ++ u8:5 ++ u8:6 ++ u8:7 ++ u64:0xfffffffff));
const_assert!(B2.to_bits() ==
    (u32:5 ++ u8:5 ++ u8:6 ++ u8:7 ++ u64:0xffffffff1));
)";
  ImportData import_data = CreateImportDataForTest();
  XLS_EXPECT_OK(TypecheckV2(kImported, "imported", &import_data).status());
  XLS_EXPECT_OK(TypecheckV2(kProgram, "main", &import_data));
}

TEST(TypecheckV2TraitTest, ToBitsWithUnsupportedSubStruct) {
  EXPECT_THAT(R"(
struct Foo {
  a: u32,
  b: u8[3]
}

#[derive(ToBits)]
struct Bar {
  f: Foo,
  g: s64
}

const B = Bar {f: Foo { a: 5, b: [5, 6, 7] }, g: 0xfffffffff};

const_assert!(B.to_bits() ==
    (u32:5 ++ u8:5 ++ u8:6 ++ u8:7 ++ u64:0xfffffffff));
)",
              TypecheckFails(
                  HasSubstr("No function `to_bits` on object of type: `Foo`")));
}

TEST(TypecheckV2TraitTest, ToBitsWithManuallySupportedSubStruct) {
  XLS_EXPECT_OK(TypecheckV2(R"(
struct Foo {
  a: u32,
  b: u8[3]
}

impl Foo {
  fn to_bits(self) -> uN[56] {
    self.b[2] ++ self.a ++ self.b[0] ++ self.b[1]
  }
}

#[derive(ToBits)]
struct Bar {
  f: Foo,
  g: s64
}

const B = Bar {f: Foo { a: 5, b: [5, 6, 7] }, g: 0xfffffffff};

const_assert!(B.to_bits() ==
    (u8:7 ++ u32:5 ++ u8:5 ++ u8:6 ++ u64:0xfffffffff));
)"));
}

TEST(TypecheckV2TraitTest, ToBitsOnParametricStruct) {
  XLS_ASSERT_OK(TypecheckV2(R"(
#[derive(ToBits)]
struct Foo<N: u32> {
  a: uN[N],
  b: u32[N]
}

const F1 = Foo {a: u4:2, b: [1, 2, 3, 4]};
const F2 = Foo {a: u2:1, b: [1000, 2000]};

const_assert!(F1.to_bits() == (u4:2 ++ u32:1 ++ u32:2 ++ u32:3 ++ u32:4));
const_assert!(F2.to_bits() == (u2:1 ++ u32:1000 ++ u32:2000));
)"));
}

TEST(TypecheckV2TraitTest, ToBitsOnEmptyStruct) {
  XLS_ASSERT_OK(TypecheckV2(R"(
#[derive(ToBits)]
struct Foo {}

const_assert!(Foo{}.to_bits() == bits[0]:0);
)"));
}

}  // namespace
}  // namespace xls::dslx
