// Copyright 2024 The XLS Authors
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

#include "xls/dslx/frontend/zip_ast.h"

#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/container/btree_set.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "xls/common/status/matchers.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/frontend/ast_node_visitor_with_default.h"
#include "xls/dslx/frontend/module.h"
#include "xls/dslx/frontend/parser.h"
#include "xls/dslx/frontend/pos.h"
#include "xls/dslx/frontend/scanner.h"

namespace xls::dslx {
namespace {

using absl_testing::StatusIs;
using ::testing::Contains;
using ::testing::ElementsAre;
using ::testing::Not;

class ZipAstTest : public ::testing::Test {
 public:
  absl::StatusOr<std::unique_ptr<Module>> Parse(std::string_view program) {
    scanner_.emplace(file_table_,
                     file_table_.GetOrCreate(absl::StrCat("test", file_no_++)),
                     std::string(program));
    parser_.emplace("test", &*scanner_);
    return parser_->ParseModule();
  }
  FileTable file_table_;
  std::optional<Scanner> scanner_;
  std::optional<Parser> parser_;
  int file_no_ = 0;
};

class Collector : public AstNodeVisitorWithDefault {
 public:
  absl::Status DefaultHandler(const AstNode* node) override {
    nodes_.push_back(node);
    return absl::OkStatus();
  }

  const std::vector<const AstNode*>& nodes() const { return nodes_; }

  absl::btree_set<std::string> GetNodeStrings() const {
    absl::btree_set<std::string> result;
    for (const AstNode* node : nodes_) {
      result.insert(node->ToString());
    }
    return result;
  }

 private:
  std::vector<const AstNode*> nodes_;
};

TEST_F(ZipAstTest, ZipEmptyModules) {
  XLS_ASSERT_OK_AND_ASSIGN(auto module1, Parse(""));
  XLS_ASSERT_OK_AND_ASSIGN(auto module2, Parse(""));
  Collector collector;
  XLS_EXPECT_OK(
      ZipAst(module1.get(), module2.get(), &collector, &collector,
             [](const AstNode*, const AstNode*) {
               return absl::FailedPreconditionError(
                   "Should not invoke accept mismatch callback here.");
             }));
  EXPECT_THAT(collector.nodes(), ElementsAre(module1.get(), module2.get()));
}

TEST_F(ZipAstTest, Zip2IdenticalFunctions) {
  constexpr std::string_view kProgram = R"(
fn muladd<S: bool, N: u32>(a: xN[S][N], b: xN[S][N], c: xN[S][N]) -> xN[S][N] {
  a * b + c
}
  )";
  XLS_ASSERT_OK_AND_ASSIGN(auto module1, Parse(kProgram));
  XLS_ASSERT_OK_AND_ASSIGN(auto module2, Parse(kProgram));
  Collector collector1;
  Collector collector2;
  XLS_EXPECT_OK(
      ZipAst(module1.get(), module2.get(), &collector1, &collector2,
             [](const AstNode* a, const AstNode* b) {
               return absl::FailedPreconditionError(
                   "Should not invoke accept mismatch callback here.");
             }));
  EXPECT_EQ(collector1.GetNodeStrings(), collector2.GetNodeStrings());
}

TEST_F(ZipAstTest, ZipWithMismatchAccepted) {
  XLS_ASSERT_OK_AND_ASSIGN(auto module1, Parse(R"(
fn muladd<S: bool, N: u32>(a: xN[S][N], b: xN[S][N], c: xN[S][N]) -> xN[S][N] {
  (a + u32:1) * b + (c + u32:1)
}
  )"));
  XLS_ASSERT_OK_AND_ASSIGN(auto module2, Parse(R"(
fn muladd<S: bool, N: u32>(a: xN[S][N], b: xN[S][N], c: xN[S][N]) -> xN[S][N] {
  (a + u32:1) * u32:42 + (c + u32:1)
}
  )"));
  Collector collector1;
  Collector collector2;
  XLS_EXPECT_OK(ZipAst(module1.get(), module2.get(), &collector1, &collector2,
                       [](const AstNode* a, const AstNode* b) {
                         EXPECT_EQ(a->ToString(), "b");
                         EXPECT_EQ(b->ToString(), "u32:42");
                         return absl::OkStatus();
                       }));
  EXPECT_EQ(collector1.nodes().size(), collector2.nodes().size());
  EXPECT_THAT(collector1.GetNodeStrings(), Contains("(a + u32:1)"));
  EXPECT_THAT(collector2.GetNodeStrings(), Contains("(a + u32:1)"));
  EXPECT_THAT(collector1.GetNodeStrings(), Contains("(c + u32:1)"));
  EXPECT_THAT(collector2.GetNodeStrings(), Contains("(c + u32:1)"));
  EXPECT_THAT(collector1.GetNodeStrings(), Not(Contains("u32:42")));
  EXPECT_THAT(collector2.GetNodeStrings(), Not(Contains("u32:42")));
}

TEST_F(ZipAstTest, ZipWithMismatchRejected) {
  XLS_ASSERT_OK_AND_ASSIGN(auto module1, Parse(R"(
fn muladd<S: bool, N: u32>(a: xN[S][N], b: xN[S][N], c: xN[S][N]) -> xN[S][N] {
  (a + u32:1) * b + (c + u32:1)
}
  )"));
  XLS_ASSERT_OK_AND_ASSIGN(auto module2, Parse(R"(
fn muladd<S: bool, N: u32>(a: xN[S][N], b: xN[S][N], c: xN[S][N]) -> xN[S][N] {
  (a + u32:1) * u32:42 + (c + u32:1)
}
  )"));
  Collector collector1;
  Collector collector2;
  EXPECT_THAT(ZipAst(module1.get(), module2.get(), &collector1, &collector2,
                     [](const AstNode* a, const AstNode* b) {
                       EXPECT_EQ(a->ToString(), "b");
                       EXPECT_EQ(b->ToString(), "u32:42");
                       return absl::InvalidArgumentError("rejected");
                     }),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_EQ(collector1.nodes().size(), collector2.nodes().size());
  EXPECT_THAT(collector1.GetNodeStrings(), Contains("(a + u32:1)"));
  EXPECT_THAT(collector2.GetNodeStrings(), Contains("(a + u32:1)"));
  EXPECT_THAT(collector1.GetNodeStrings(), Not(Contains("(c + u32:1)")));
  EXPECT_THAT(collector2.GetNodeStrings(), Not(Contains("(c + u32:1)")));
}

}  // namespace
}  // namespace xls::dslx
