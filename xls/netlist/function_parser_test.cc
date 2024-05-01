// Copyright 2020 The XLS Authors
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

#include "xls/netlist/function_parser.h"

#include <string>

#include "gtest/gtest.h"
#include "xls/common/status/matchers.h"

namespace xls {
namespace netlist {
namespace function {
namespace {

// Extremely basic test that we're able to scan extremely simple Functions.
TEST(ScannerTest, SimpleScan) {
  std::string function = "A+B";
  Scanner scanner(function);
  XLS_ASSERT_OK_AND_ASSIGN(Token token, scanner.Pop());
  EXPECT_EQ(token.kind(), Token::Kind::kIdentifier);
  EXPECT_EQ(token.pos(), 0);
  EXPECT_EQ(token.payload(), "A");

  XLS_ASSERT_OK_AND_ASSIGN(token, scanner.Pop());
  EXPECT_EQ(token.kind(), Token::Kind::kOr);
  EXPECT_EQ(token.pos(), 1);

  XLS_ASSERT_OK_AND_ASSIGN(token, scanner.Pop());
  EXPECT_EQ(token.kind(), Token::Kind::kIdentifier);
  EXPECT_EQ(token.pos(), 2);
  EXPECT_EQ(token.payload(), "B");
}

// Verifies that we can correctly parse an identifer with a leading digit.
TEST(ScannerTest, ScanLeadingDigitIdentifier) {
  std::string function = R"("1A1"+"2B2")";
  Scanner scanner(function);

  XLS_ASSERT_OK_AND_ASSIGN(Token token, scanner.Pop());
  EXPECT_EQ(token.kind(), Token::Kind::kIdentifier);
  EXPECT_EQ(token.pos(), 0);
  EXPECT_EQ(token.payload(), "1A1");

  XLS_ASSERT_OK_AND_ASSIGN(token, scanner.Pop());
  EXPECT_EQ(token.kind(), Token::Kind::kOr);
  EXPECT_EQ(token.pos(), 5);

  XLS_ASSERT_OK_AND_ASSIGN(token, scanner.Pop());
  EXPECT_EQ(token.kind(), Token::Kind::kIdentifier);
  EXPECT_EQ(token.pos(), 6);
  EXPECT_EQ(token.payload(), "2B2");
}

// Just tests recognition of more tokens.
TEST(ScannerTest, ScanMoreChecks) {
  std::string function = "1A()!* &+|^01";
  Scanner scanner(function);
  XLS_ASSERT_OK_AND_ASSIGN(Token token, scanner.Pop());
  EXPECT_EQ(token.kind(), Token::Kind::kLogicOne);

  XLS_ASSERT_OK_AND_ASSIGN(token, scanner.Pop());
  EXPECT_EQ(token.kind(), Token::Kind::kIdentifier);

  XLS_ASSERT_OK_AND_ASSIGN(token, scanner.Pop());
  EXPECT_EQ(token.kind(), Token::Kind::kOpenParen);

  XLS_ASSERT_OK_AND_ASSIGN(token, scanner.Pop());
  EXPECT_EQ(token.kind(), Token::Kind::kCloseParen);

  XLS_ASSERT_OK_AND_ASSIGN(token, scanner.Pop());
  EXPECT_EQ(token.kind(), Token::Kind::kInvertFollowing);

  // One AND is "missing", because spaces following an operator are dropped
  // (this is not per the Liberty spec, but matches actual cell library
  // definitions).
  XLS_ASSERT_OK_AND_ASSIGN(token, scanner.Pop());
  EXPECT_EQ(token.kind(), Token::Kind::kAnd);

  XLS_ASSERT_OK_AND_ASSIGN(token, scanner.Pop());
  EXPECT_EQ(token.kind(), Token::Kind::kAnd);

  XLS_ASSERT_OK_AND_ASSIGN(token, scanner.Pop());
  EXPECT_EQ(token.kind(), Token::Kind::kOr);

  XLS_ASSERT_OK_AND_ASSIGN(token, scanner.Pop());
  EXPECT_EQ(token.kind(), Token::Kind::kOr);

  XLS_ASSERT_OK_AND_ASSIGN(token, scanner.Pop());
  EXPECT_EQ(token.kind(), Token::Kind::kXor);

  XLS_ASSERT_OK_AND_ASSIGN(token, scanner.Pop());
  EXPECT_EQ(token.kind(), Token::Kind::kLogicZero);

  XLS_ASSERT_OK_AND_ASSIGN(token, scanner.Pop());
  EXPECT_EQ(token.kind(), Token::Kind::kLogicOne);

  EXPECT_FALSE(scanner.Pop().ok());
}

TEST(FunctionParserTest, SimpleSmoke) {
  std::string function = R"(A+B)";
  XLS_ASSERT_OK_AND_ASSIGN(auto ast, Parser::ParseFunction(function));

  ASSERT_EQ(ast.kind(), Ast::Kind::kOr);
  ASSERT_EQ(ast.children()[0].kind(), Ast::Kind::kIdentifier);
  ASSERT_EQ(ast.children()[0].name(), "A");
  ASSERT_EQ(ast.children()[1].kind(), Ast::Kind::kIdentifier);
  ASSERT_EQ(ast.children()[1].name(), "B");
}

TEST(FunctionParserTest, SimpleCompound) {
  std::string function = R"(A+B*C)";
  XLS_ASSERT_OK_AND_ASSIGN(auto ast, Parser::ParseFunction(function));

  ASSERT_EQ(ast.kind(), Ast::Kind::kOr);
  ASSERT_EQ(ast.children()[0].kind(), Ast::Kind::kIdentifier);
  ASSERT_EQ(ast.children()[0].name(), "A");
  ASSERT_EQ(ast.children()[1].kind(), Ast::Kind::kAnd);
  ASSERT_EQ(ast.children()[1].children()[0].kind(), Ast::Kind::kIdentifier);
  ASSERT_EQ(ast.children()[1].children()[0].name(), "B");
  ASSERT_EQ(ast.children()[1].children()[1].kind(), Ast::Kind::kIdentifier);
  ASSERT_EQ(ast.children()[1].children()[1].name(), "C");
}

TEST(FunctionParserTest, Inversions) {
  // The AST should look something like (where all negations are '!'):
  //    +
  //   ! !
  //  !   *
  // A   ! C
  //    B
  std::string function = R"(!A'+(!B*C)')";
  XLS_ASSERT_OK_AND_ASSIGN(auto ast, Parser::ParseFunction(function));
  ASSERT_EQ(ast.kind(), Ast::Kind::kOr);

  // Left-hand side:
  const Ast* lhs = &ast.children()[0];
  ASSERT_EQ(lhs->kind(), Ast::Kind::kNot);
  lhs = &lhs->children()[0];
  ASSERT_EQ(lhs->kind(), Ast::Kind::kNot);
  lhs = &lhs->children()[0];
  ASSERT_EQ(lhs->kind(), Ast::Kind::kIdentifier);
  ASSERT_EQ(lhs->name(), "A");

  // Right-hand side:
  const Ast* rhs = &ast.children()[1];
  ASSERT_EQ(rhs->kind(), Ast::Kind::kNot);
  rhs = &rhs->children()[0];
  ASSERT_EQ(rhs->kind(), Ast::Kind::kAnd);
  ASSERT_EQ(rhs->children()[1].kind(), Ast::Kind::kIdentifier);
  ASSERT_EQ(rhs->children()[1].name(), "C");

  // Continue to the left children.
  rhs = &rhs->children()[0];
  ASSERT_EQ(rhs->kind(), Ast::Kind::kNot);
  rhs = &rhs->children()[0];
  ASSERT_EQ(rhs->kind(), Ast::Kind::kIdentifier);
  ASSERT_EQ(rhs->name(), "B");
}

TEST(FunctionParserTest, SpaceAsAnd) {
  // I really don't care for space-as-AND, but thus sayeth the standard.
  // Expected AST:
  //
  //              ||
  //       ||           &
  //   &       &      !   F
  // A   B   C   !    E
  //             D
  //
  std::string function = R"(A B+C !D+(E' F))";
  XLS_ASSERT_OK_AND_ASSIGN(auto node, Parser::ParseFunction(function));
  ASSERT_EQ(node.kind(), Ast::Kind::kOr);
  ASSERT_EQ(node.children()[0].kind(), Ast::Kind::kOr);
  ASSERT_EQ(node.children()[1].kind(), Ast::Kind::kAnd);

  // Consider the RHS first (just for simpler bookkeeping):
  const Ast* lhs = &node.children()[1].children()[0];
  const Ast* rhs = &node.children()[1].children()[1];
  ASSERT_EQ(lhs->kind(), Ast::Kind::kNot);
  ASSERT_EQ(lhs->children()[0].kind(), Ast::Kind::kIdentifier);
  ASSERT_EQ(lhs->children()[0].name(), "E");

  ASSERT_EQ(rhs->kind(), Ast::Kind::kIdentifier);
  ASSERT_EQ(rhs->name(), "F");

  // Now back to the left half of the tree:
  lhs = &node.children()[0].children()[0];
  ASSERT_EQ(lhs->kind(), Ast::Kind::kAnd);
  ASSERT_EQ(lhs->children()[0].kind(), Ast::Kind::kIdentifier);
  ASSERT_EQ(lhs->children()[0].name(), "A");
  ASSERT_EQ(lhs->children()[1].kind(), Ast::Kind::kIdentifier);
  ASSERT_EQ(lhs->children()[1].name(), "B");

  rhs = &node.children()[0].children()[1];
  ASSERT_EQ(rhs->kind(), Ast::Kind::kAnd);
  ASSERT_EQ(rhs->children()[0].kind(), Ast::Kind::kIdentifier);
  ASSERT_EQ(rhs->children()[0].name(), "C");
  ASSERT_EQ(rhs->children()[1].kind(), Ast::Kind::kNot);
  ASSERT_EQ(rhs->children()[1].children()[0].kind(), Ast::Kind::kIdentifier);
  ASSERT_EQ(rhs->children()[1].children()[0].name(), "D");
}

TEST(FunctionParserTest, LeadingAndTrailingWhitespace) {
  std::string function = " A B+C ";
  XLS_ASSERT_OK_AND_ASSIGN(auto node, Parser::ParseFunction(function));
  ASSERT_EQ(node.kind(), Ast::Kind::kOr);
  ASSERT_EQ(node.children()[0].kind(), Ast::Kind::kAnd);
  ASSERT_EQ(node.children()[1].kind(), Ast::Kind::kIdentifier);
}

}  // namespace
}  // namespace function
}  // namespace netlist
}  // namespace xls
