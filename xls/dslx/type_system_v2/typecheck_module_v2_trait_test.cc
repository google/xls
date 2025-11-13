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
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xls/common/status/matchers.h"
#include "xls/common/status/status_macros.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/frontend/pos.h"
#include "xls/dslx/type_system/type.h"
#include "xls/dslx/type_system/typecheck_test_utils.h"
#include "xls/dslx/type_system_v2/matchers.h"
#include "xls/dslx/type_system_v2/type_annotation_utils.h"

namespace xls::dslx {
namespace {

class TestDeriver : public TraitDeriver {
 public:
  absl::StatusOr<StatementBlock*> DeriveFunctionBody(
      Module& module, const Trait& trait, const StructDef& struct_def,
      const StructType& struct_type, const Function& function) override {
    if (function.identifier() == "contains") {
      return DeriveContains(module, trait, struct_def, function);
    }
    if (function.identifier() == "has_nonzero_foo") {
      return DeriveHasNonzeroFooField(module, trait, struct_def, function);
    }
    if (function.identifier() == "has_array_of_size_3") {
      return DeriveHasArrayOfSize3(module, trait, struct_def, struct_type,
                                   function);
    }
    return absl::UnimplementedError("not implemented");
  }

  absl::StatusOr<StatementBlock*> DeriveHasNonzeroFooField(
      Module& module, const Trait& trait, const StructDef& struct_def,
      const Function& function) {
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

  absl::StatusOr<StatementBlock*> DeriveContains(Module& module,
                                                 const Trait& trait,
                                                 const StructDef& struct_def,
                                                 const Function& function) {
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

  absl::StatusOr<StatementBlock*> DeriveHasArrayOfSize3(
      Module& module, const Trait& trait, const StructDef& struct_def,
      const StructType& struct_type, const Function& function) {
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
                            std::make_unique<TestDeriver>()));
}

TEST(TypecheckV2TraitTest, DeriveMultiTraits) {
  XLS_ASSERT_OK(TypecheckV2(R"(
trait Contains {
  fn contains(self, val: u32) -> bool;
}

trait HasNonZeroFooField {
  fn has_nonzero_foo(self) -> bool;
}

#[derive(Contains, HasNonZeroFooField)]
struct Foo {
  a: u32,
  foo: u32,
  c: u32
}

const_assert!(Foo {a: 5, foo: 6, c: 0}.contains(5));
const_assert!(Foo {a: 1, foo: 5, c: 0}.has_nonzero_foo());
const_assert!(!zero!<Foo>().has_nonzero_foo());
)",
                            std::make_unique<TestDeriver>()));
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
                          std::make_unique<TestDeriver>()),
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
                  std::make_unique<TestDeriver>()),
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
                            std::make_unique<TestDeriver>()));
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
                            std::make_unique<TestDeriver>()));
}

}  // namespace
}  // namespace xls::dslx
