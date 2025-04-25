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

#include "clang/AST/Expr.h"
#include "clang/AST/ExprCXX.h"
#include "gtest/gtest.h"
#include "xls/common/status/matchers.h"
#include "xls/contrib/xlscc/cc_parser.h"
#include "xls/contrib/xlscc/unit_tests/clang_util.h"
#include "xls/contrib/xlscc/unit_tests/unit_test.h"

void CheckStmtsEq(const clang::Stmt* first, const clang::Stmt* second) {
  ASSERT_NE(first, second);
  ASSERT_EQ(first->getStmtClass(), second->getStmtClass());

  auto* first_expr = clang::dyn_cast<clang::Expr>(first);
  auto* second_expr = clang::dyn_cast<clang::Expr>(second);
  if (first_expr && second_expr) {
    ASSERT_EQ(first_expr->getType(), second_expr->getType());
    ASSERT_EQ(first_expr->getValueKind(), second_expr->getValueKind());
    ASSERT_EQ(first_expr->getObjectKind(), second_expr->getObjectKind());
  }

#define ASSERT_FIELDS_EQ(class, field)                     \
  do {                                                     \
    auto* first_ = clang::dyn_cast<clang::class>(first);   \
    ASSERT_NE(first_, nullptr);                            \
    auto* second_ = clang::dyn_cast<clang::class>(second); \
    ASSERT_NE(second_, nullptr);                           \
    ASSERT_EQ(first_->field(), second_->field());          \
  } while (false)  // force trailing semicolon

  switch (first->getStmtClass()) {
    case clang::Stmt::IntegerLiteralClass:
      ASSERT_FIELDS_EQ(IntegerLiteral, getValue);
      break;
    case clang::Stmt::FloatingLiteralClass:
      ASSERT_FIELDS_EQ(FloatingLiteral, getValue);
      ASSERT_FIELDS_EQ(FloatingLiteral, isExact);
      break;
    case clang::Stmt::CharacterLiteralClass:
      ASSERT_FIELDS_EQ(CharacterLiteral, getValue);
      break;
    case clang::Stmt::StringLiteralClass:
      ASSERT_FIELDS_EQ(StringLiteral, getString);
      ASSERT_FIELDS_EQ(StringLiteral, isPascal);
      break;
    case clang::Stmt::UserDefinedLiteralClass:
      ASSERT_FIELDS_EQ(UserDefinedLiteral, getUDSuffix);
      break;
    case clang::Stmt::DeclRefExprClass:
      ASSERT_FIELDS_EQ(DeclRefExpr, getDecl);
      ASSERT_FIELDS_EQ(DeclRefExpr, getFoundDecl);
      ASSERT_FIELDS_EQ(DeclRefExpr, refersToEnclosingVariableOrCapture);
      ASSERT_FIELDS_EQ(DeclRefExpr, isNonOdrUse);
      break;
    case clang::Stmt::ImplicitCastExprClass:
      ASSERT_FIELDS_EQ(ImplicitCastExpr, getCastKind);
      break;
    case clang::Stmt::CStyleCastExprClass:
      ASSERT_FIELDS_EQ(CStyleCastExpr, getCastKind);
      break;
    case clang::Stmt::CXXFunctionalCastExprClass:
      ASSERT_FIELDS_EQ(CXXFunctionalCastExpr, getCastKind);
      break;
    case clang::Stmt::CXXStaticCastExprClass:
      ASSERT_FIELDS_EQ(CXXStaticCastExpr, getCastKind);
      break;
    case clang::Stmt::CXXDynamicCastExprClass:
      ASSERT_FIELDS_EQ(CXXDynamicCastExpr, getCastKind);
      break;
    case clang::Stmt::CXXReinterpretCastExprClass:
      ASSERT_FIELDS_EQ(CXXReinterpretCastExpr, getCastKind);
      break;
    case clang::Stmt::CXXConstCastExprClass:
      ASSERT_FIELDS_EQ(CXXConstCastExpr, getCastKind);
      break;
    case clang::Stmt::MemberExprClass:
      ASSERT_FIELDS_EQ(MemberExpr, isArrow);
      ASSERT_FIELDS_EQ(MemberExpr, isNonOdrUse);
      ASSERT_FIELDS_EQ(MemberExpr, getMemberDecl);
      ASSERT_FIELDS_EQ(MemberExpr, getFoundDecl);
      break;
    case clang::Stmt::UnaryOperatorClass:
      ASSERT_FIELDS_EQ(UnaryOperator, getOpcode);
      ASSERT_FIELDS_EQ(UnaryOperator, canOverflow);
      break;
    case clang::Stmt::BinaryOperatorClass:
      ASSERT_FIELDS_EQ(BinaryOperator, getOpcode);
      break;
    case clang::Stmt::CXXTemporaryObjectExprClass:
      ASSERT_FIELDS_EQ(CXXTemporaryObjectExpr, isElidable);
      ASSERT_FIELDS_EQ(CXXTemporaryObjectExpr, hadMultipleCandidates);
      ASSERT_FIELDS_EQ(CXXTemporaryObjectExpr, isListInitialization);
      ASSERT_FIELDS_EQ(CXXTemporaryObjectExpr, isStdInitListInitialization);
      ASSERT_FIELDS_EQ(CXXTemporaryObjectExpr, requiresZeroInitialization);
      ASSERT_FIELDS_EQ(CXXTemporaryObjectExpr, getConstructor);
      ASSERT_FIELDS_EQ(CXXTemporaryObjectExpr, getConstructionKind);
      break;
    case clang::Stmt::CompoundLiteralExprClass:
    case clang::Stmt::CallExprClass:
    case clang::Stmt::ConditionalOperatorClass:
    case clang::Stmt::InitListExprClass:
    case clang::Stmt::ParenExprClass:
    case clang::Stmt::ArraySubscriptExprClass:
    case clang::Stmt::CXXOperatorCallExprClass:
    case clang::Stmt::CXXMemberCallExprClass:
      break;
    default:
      FAIL() << "Unsupported Stmt";
      break;
  }

  ASSERT_EQ(llvm::range_size(first->children()),
            llvm::range_size(second->children()));
  for (auto [first_child, second_child] :
       llvm::zip(first->children(), second->children())) {
    CheckStmtsEq(first_child, second_child);
  }
}

namespace {
class ExprCloneTest : public XlsccTestBase {
 public:
  void CloneCheckReturnExpr(const char* top_name, const char* cpp_src) {
    xlscc::CCParser parser;
    XLS_ASSERT_OK(ScanTempFileWithContent(cpp_src, {}, &parser, top_name));
    XLS_ASSERT_OK_AND_ASSIGN(const auto* top_fn, parser.GetTopFunction());
    ASSERT_NE(top_fn, nullptr);
    auto* ret_stmt = GetStmtInFunction<clang::ReturnStmt>(top_fn);
    ASSERT_NE(ret_stmt, nullptr);
    auto* ret_val = ret_stmt->getRetValue();
    ASSERT_NE(ret_val, nullptr);
    llvm::errs() << "=> Original:\n";
    ret_val->dump(llvm::errs(), top_fn->getASTContext());

    auto* ret_val_clone = xlscc::Clone(top_fn->getASTContext(), ret_val);
    ASSERT_NE(ret_val_clone, nullptr);
    llvm::errs() << "=> Cloned:\n";
    ret_val_clone->dump(llvm::errs(), top_fn->getASTContext());

    CheckStmtsEq(ret_val, ret_val_clone);
  }
};

TEST_F(ExprCloneTest, CloneIntegerLiteral) {
  CloneCheckReturnExpr("foo", R"(int foo() {
    return 42;
  })");
}

TEST_F(ExprCloneTest, CloneFloatingLiteral) {
  CloneCheckReturnExpr("foo", R"(float foo() {
    return 3.14f;
  })");
}

TEST_F(ExprCloneTest, CloneCharacterLiteral) {
  CloneCheckReturnExpr("foo", R"(char foo() {
    return 'x';
  })");
}

TEST_F(ExprCloneTest, CloneStringLiteral) {
  CloneCheckReturnExpr("foo", R"(const char* foo() {
    return "hello world";
  })");
}

TEST_F(ExprCloneTest, CloneUserDefinedLiteral) {
  CloneCheckReturnExpr("foo", R"(
    int operator ""_qux(unsigned long long);
    int foo() {
      return 16_qux;
    }
  )");
}

TEST_F(ExprCloneTest, CloneCompoundLiteralExpr) {
  CloneCheckReturnExpr("foo", R"(int foo() {
    return (int){42};
  })");
}

TEST_F(ExprCloneTest, CloneDeclRefExpr) {
  CloneCheckReturnExpr("foo", R"(
    int foo() {
      int x = 0;
      return x;
    }
  )");
}

TEST_F(ExprCloneTest, CloneCallExpr) {
  CloneCheckReturnExpr("foo", R"(
    int bar();
    int foo() {
      return bar();
    }
  )");
}

TEST_F(ExprCloneTest, CloneCStyleCastExpr) {
  CloneCheckReturnExpr("foo", R"(float foo() {
    return (float) 42;
  })");
}

TEST_F(ExprCloneTest, CloneCXXFunctionalCastExpr) {
  CloneCheckReturnExpr("foo", R"(float foo() {
    return float(42);
  })");
}

TEST_F(ExprCloneTest, CloneCXXStaticCastExpr) {
  CloneCheckReturnExpr("foo", R"(float foo() {
    return static_cast<float>(42);
  })");
}

TEST_F(ExprCloneTest, CloneCXXDynamicCastExpr) {
  CloneCheckReturnExpr("foo", R"(
    struct bar {
      virtual ~bar() = default;
    };
    struct baz : bar {};
    baz* foo(bar* b) {
      return dynamic_cast<baz*>(b);
    }
  )");
}

TEST_F(ExprCloneTest, CloneCXXReinterpretCastExpr) {
  CloneCheckReturnExpr("foo", R"(float* foo(int* x) {
    return reinterpret_cast<float*>(x);
  })");
}

TEST_F(ExprCloneTest, CloneCXXConstCastExpr) {
  CloneCheckReturnExpr("foo", R"(int& foo(const int& x) {
    return const_cast<int&>(x);
  })");
}

TEST_F(ExprCloneTest, CloneMemberExpr) {
  CloneCheckReturnExpr("foo", R"(
    struct bar {
      int x;
    };
    int foo() {
      bar b;
      return b.x;
    }
  )");
}

TEST_F(ExprCloneTest, CloneUnaryOperator) {
  CloneCheckReturnExpr("foo", R"(int foo() {
    int a = 0;
    return -a;
  })");
}

TEST_F(ExprCloneTest, CloneBinaryOperator) {
  CloneCheckReturnExpr("foo", R"(int foo(int a) {
    int b = 0;
    return a + b;
  })");
}

TEST_F(ExprCloneTest, CloneConditionalOperator) {
  CloneCheckReturnExpr("foo", R"(int foo(bool b) {
    return b ? 1 : 0;
  })");
}

TEST_F(ExprCloneTest, CloneArraySubscriptExpr) {
  CloneCheckReturnExpr("foo", R"(int foo(int* a, int i) {
    return a[i];
  })");
}

TEST_F(ExprCloneTest, CloneInitListExpr) {
  CloneCheckReturnExpr("foo", R"(
    struct bar { int a, b, c; };
    bar foo() {
      return {1, 2, 3};
    }
  )");
}

TEST_F(ExprCloneTest, CloneCXXOperatorCallExpr) {
  CloneCheckReturnExpr("foo", R"(
    struct bar {};
    bar operator*(bar, bar);
    bar foo() {
      return bar{} * bar{};
    }
  )");
}

TEST_F(ExprCloneTest, CloneCXXMemberCallExpr) {
  CloneCheckReturnExpr("foo", R"(
    struct bar {
      float qux();
    };
    float foo() {
      bar b;
      return b.qux();
    }
  )");
}

TEST_F(ExprCloneTest, CloneCXXTemporaryObjectExpr) {
  CloneCheckReturnExpr("foo", R"(
    struct bar {};
    bar foo() {
      return bar();
    }
  )");
}

TEST_F(ExprCloneTest, CloneComplexExpr) {
  CloneCheckReturnExpr("foo", R"(
    int bar(int);
    float foo(int* a, int i) {
      int b = 0;
      return a[i] ? (float) a[i] * (bar(b) << 1) : 3.14;
    }
  )");
}

}  // namespace
