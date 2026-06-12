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

#include <optional>
#include <string_view>
#include <variant>

#include "gtest/gtest.h"
#include "absl/base/casts.h"
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
  EXPECT_EQ(domain_struct->ToString(), R"(struct MyStructDomain {
    x: (),
})");
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
  EXPECT_EQ(outer_domain->ToString(), R"(struct OuterDomain {
    x: InnerDomain,
})");
}

TEST(FuzzDomainRewriterTest, RewriteStructInstanceDomain) {
  constexpr std::string_view kProgram = R"(
#[fuzz_domain("MyStructDomain")]
struct MyStruct {
    x: u32,
    y: u8,
}

fn create_f_domain() -> MyStructDomain {
   MyStructDomain {
     x: u32:0..10,
   }
}

#[fuzz_test(domains=`create_f_domain()`)]
fn f(s: MyStruct) {}
)";
  auto import_data = CreateImportDataForTest();
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule tm,
      ParseAndTypecheck(kProgram, "test.x", "test", &import_data));

  std::optional<Function*> fn_opt = tm.module->GetFunction("create_f_domain");
  ASSERT_TRUE(fn_opt.has_value());
  Function* fn = *fn_opt;
  const Expr* body = fn->body();
  ASSERT_EQ(body->kind(), AstNodeKind::kStatementBlock);
  const auto* block = absl::down_cast<const StatementBlock*>(body);
  const Statement* last_stmt = block->statements().back();
  ASSERT_TRUE(std::holds_alternative<Expr*>(last_stmt->wrapped()));
  Expr* last_expr = std::get<Expr*>(last_stmt->wrapped());
  ASSERT_EQ(last_expr->kind(), AstNodeKind::kStructInstance);
  auto* struct_instance = absl::down_cast<StructInstance*>(last_expr);

  EXPECT_EQ(struct_instance->members().size(), 2);
  // The 'y' field should be an empty tuple, since it wasn't specified.
  XLS_ASSERT_OK_AND_ASSIGN(Expr * y_expr, struct_instance->GetExpr("y"));
  EXPECT_EQ(y_expr->kind(), AstNodeKind::kXlsTuple);
  auto* y_tuple = absl::down_cast<XlsTuple*>(y_expr);
  EXPECT_TRUE(y_tuple->members().empty());
}

}  // namespace
}  // namespace xls::dslx
