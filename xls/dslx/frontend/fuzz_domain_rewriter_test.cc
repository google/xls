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

#include <string_view>

#include "gtest/gtest.h"
#include "xls/common/status/matchers.h"
#include "xls/dslx/create_import_data.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/parse_and_typecheck.h"

namespace xls::dslx {
namespace {

TEST(FuzzDomainRewriterTest, RewriteBasicStructDomain) {
  constexpr std::string_view kProgram = R"(
#[fuzz_domain("MyStructDomain")]
struct MyStruct {
    x: u32,
}
)";
  auto import_data = CreateImportDataForTest();
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule tm,
      ParseAndTypecheck(kProgram, "test.x", "test", &import_data));

  XLS_ASSERT_OK_AND_ASSIGN(
      StructDef * domain_struct,
      tm.module->GetMemberOrError<StructDef>("MyStructDomain"));

  // Verify the domain struct has member 'x' of type tuple (empty tuple)
  ASSERT_EQ(domain_struct->members().size(), 1);
  EXPECT_EQ(domain_struct->members()[0]->name(), "x");
  EXPECT_TRUE(
      domain_struct->members()[0]->type()->IsAnnotation<TupleTypeAnnotation>());
  TupleTypeAnnotation* tuple =
      domain_struct->members()[0]->type()->AsAnnotation<TupleTypeAnnotation>();
  EXPECT_TRUE(tuple->empty());
}

TEST(FuzzDomainRewriterTest, RewriteNestedStructDomain) {
  constexpr std::string_view kProgram = R"(
#[fuzz_domain("InnerDomain")]
struct Inner {
    y: u32,
}

#[fuzz_domain("OuterDomain")]
struct Outer {
    x: Inner,
}
)";
  auto import_data = CreateImportDataForTest();
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule tm,
      ParseAndTypecheck(kProgram, "test.x", "test", &import_data));

  XLS_ASSERT_OK_AND_ASSIGN(
      StructDef * outer_domain,
      tm.module->GetMemberOrError<StructDef>("OuterDomain"));

  ASSERT_EQ(outer_domain->members().size(), 1);
  EXPECT_EQ(outer_domain->members()[0]->name(), "x");
  // The type of 'x' in OuterDomain should be 'InnerDomain'
  TypeAnnotation* x_type = outer_domain->members()[0]->type();
  ASSERT_TRUE(x_type->IsAnnotation<TypeRefTypeAnnotation>());
  auto* type_ref_type = x_type->AsAnnotation<TypeRefTypeAnnotation>();
  EXPECT_EQ(type_ref_type->type_ref()->ToString(), "InnerDomain");
}

}  // namespace
}  // namespace xls::dslx
