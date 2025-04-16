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

#include "xls/contrib/xlscc/expr_clone.h"

#include <string>

#include "gtest/gtest.h"
#include "xls/common/status/matchers.h"
#include "xls/contrib/xlscc/cc_parser.h"
#include "xls/contrib/xlscc/unit_tests/unit_test.h"

namespace {
class ExprCloneTest : public XlsccTestBase {
 public:
};

// From cc_parser_test
template <typename ClangT>
const ClangT* GetStmtInCompoundStmt(const clang::CompoundStmt* stmt) {
  for (const clang::Stmt* body_st : stmt->children()) {
    if (const clang::LabelStmt* label =
            clang::dyn_cast<clang::LabelStmt>(body_st);
        label != nullptr) {
      body_st = label->getSubStmt();
    }

    const ClangT* ret;
    if (const clang::CompoundStmt* cmpnd_stmt =
            clang::dyn_cast<clang::CompoundStmt>(body_st);
        cmpnd_stmt != nullptr) {
      ret = GetStmtInCompoundStmt<ClangT>(cmpnd_stmt);
    } else {
      ret = clang::dyn_cast<const ClangT>(body_st);
    }
    if (ret != nullptr) {
      return ret;
    }
  }

  return nullptr;
}

// From cc_parser_test
template <typename ClangT>
const ClangT* GetStmtInFunction(const clang::FunctionDecl* func) {
  const clang::Stmt* body = func->getBody();
  if (body == nullptr) {
    return nullptr;
  }

  for (const clang::Stmt* body_st : body->children()) {
    if (const clang::LabelStmt* label =
            clang::dyn_cast<clang::LabelStmt>(body_st);
        label != nullptr) {
      body_st = label->getSubStmt();
    }

    const ClangT* ret;
    if (const clang::CompoundStmt* cmpnd_stmt =
            clang::dyn_cast<clang::CompoundStmt>(body_st);
        cmpnd_stmt != nullptr) {
      ret = GetStmtInCompoundStmt<ClangT>(cmpnd_stmt);
    } else {
      ret = clang::dyn_cast<const ClangT>(body_st);
    }
    if (ret != nullptr) {
      return ret;
    }
  }

  return nullptr;
}

TEST_F(ExprCloneTest, CloneFunCall) {
  xlscc::CCParser parser;

  const std::string cpp_src = R"(
    int bar() {
      return 0;
    }
    int foo() {
      return bar();
    }
  )";

  XLS_ASSERT_OK(
      ScanTempFileWithContent(cpp_src, {}, &parser, /*top_name=*/"foo"));
  XLS_ASSERT_OK_AND_ASSIGN(const auto* top_fn, parser.GetTopFunction());
  ASSERT_NE(top_fn, nullptr);
  auto* ret_stmt = GetStmtInFunction<clang::ReturnStmt>(top_fn);
  ASSERT_NE(ret_stmt, nullptr);
  auto* ret_val = ret_stmt->getRetValue();
  ASSERT_NE(ret_val, nullptr);
  auto* ret_val_clone = xlscc::Clone(top_fn->getASTContext(), ret_val);
  ASSERT_NE(ret_val_clone, nullptr);
  // TODO: Check that the refs are the same.
  // TODO: Verify the cloned AST somehow?
}

}  // namespace
