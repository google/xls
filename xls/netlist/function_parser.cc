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

#include <cctype>
#include <string>
#include <string_view>
#include <utility>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/ascii.h"
#include "absl/strings/str_format.h"
#include "xls/common/logging/logging.h"
#include "xls/common/status/ret_check.h"

namespace xls {
namespace netlist {
namespace function {
namespace {

// Returns true if the specified character represents a binary op.
bool IsBinOp(char c) {
  switch (c) {
    case '&':
    case '*':
    case '+':
    case '|':
    case '^':
      return true;
    default:
      return false;
  }
}

}  // namespace

Token Token::Simple(Kind kind, int64_t pos) { return Token(kind, pos); }

Token Token::Identifier(const std::string& s, int64_t pos) {
  return Token(Kind::kIdentifier, pos, s);
}

Scanner::Scanner(std::string function) : function_(function), current_pos_(0) {
  std::string_view stripped = absl::StripAsciiWhitespace(function_);
  if (stripped != function_) {
    LOG(WARNING)
        << "Function '" << function_ << "' has leading or trailing spaces. "
        << "Per the Liberty spec, spaces are AND, which are invalid here. "
        << "These spaces will be dropped.";
    function_ = std::string(stripped);
  }
}

absl::Status Scanner::DropSpaces() {
  XLS_ASSIGN_OR_RETURN(char next_char, PeekChar());
  while (next_char == ' ') {
    XLS_RETURN_IF_ERROR(DropChar());
    XLS_ASSIGN_OR_RETURN(next_char, PeekChar());
  }
  return absl::OkStatus();
}

absl::StatusOr<Token> Scanner::Pop() {
  if (peeked_) {
    Token ret = peeked_.value();
    peeked_.reset();
    return ret;
  }

  if (Eof()) {
    return absl::OutOfRangeError("At end of input.");
  }

  XLS_ASSIGN_OR_RETURN(char next_char, PeekChar());
  if (IsIdentifierStart(next_char)) {
    return ScanIdentifier();
  }

  XLS_RETURN_IF_ERROR(DropChar());
  int64_t last_pos = current_pos_ - 1;
  switch (next_char) {
    case '\'':
      return Token::Simple(Token::Kind::kInvertPrevious, last_pos);
    case '!':
      return Token::Simple(Token::Kind::kInvertFollowing, last_pos);
    case '(':
      return Token::Simple(Token::Kind::kOpenParen, last_pos);
    case ')':
      return Token::Simple(Token::Kind::kCloseParen, last_pos);
    case ' ': {
      // Unfortunately, the Liberty spec isn't consistently followed -
      // spaces are sometimes inserted into functions when not meant as ANDs.
      // So - we need to condense all whitespace into a single token, then check
      // to see if the next token is an expression or an operator.
      // If the former, then interpret it as an AND, otherwise, drop it.
      XLS_RETURN_IF_ERROR(DropSpaces());
      XLS_ASSIGN_OR_RETURN(next_char, PeekChar());
      if (IsBinOp(next_char)) {
        return Pop();
      }
      return Token::Simple(Token::Kind::kAnd, last_pos);
    }
    case '&':
    case '*': {
      XLS_RETURN_IF_ERROR(DropSpaces());
      return Token::Simple(Token::Kind::kAnd, last_pos);
    }
    case '+':
    case '|': {
      XLS_RETURN_IF_ERROR(DropSpaces());
      return Token::Simple(Token::Kind::kOr, last_pos);
    }
    case '^': {
      XLS_RETURN_IF_ERROR(DropSpaces());
      return Token::Simple(Token::Kind::kXor, last_pos);
    }
    case '0':
      return Token::Simple(Token::Kind::kLogicZero, last_pos);
    case '1':
      return Token::Simple(Token::Kind::kLogicOne, last_pos);
    default:
      return absl::InvalidArgumentError(absl::StrFormat(
          "Unhandled character for scanning @ %d: '%c'", last_pos, next_char));
  }
}

absl::StatusOr<Token> Scanner::Peek() {
  if (peeked_) {
    return peeked_.value();
  }

  XLS_ASSIGN_OR_RETURN(Token token, Pop());
  peeked_ = token;
  return token;
}

bool Scanner::IsIdentifierStart(char next_char) {
  // Pin names are [a-z][A-Z][0-9]+
  return std::isalpha(next_char) != 0|| next_char == '"' || next_char == '_';
}

absl::StatusOr<Token> Scanner::ScanIdentifier() {
  int64_t start_pos = current_pos_;
  XLS_ASSIGN_OR_RETURN(char next_char, PeekChar());
  bool need_closing_quote = next_char == '"';
  if (need_closing_quote) {
    XLS_RETURN_IF_ERROR(DropChar());
    XLS_ASSIGN_OR_RETURN(next_char, PeekChar());
  }

  std::string identifier;
  // Identifiers as defined in section 5.6 of the SystemVerilog standard.
  while (!Eof() && (std::isalnum(next_char) != 0 || next_char == '_' ||
                    next_char == '$')) {
    identifier.push_back(next_char);
    XLS_RETURN_IF_ERROR(DropChar());
    if (!Eof()) {
      XLS_ASSIGN_OR_RETURN(next_char, PeekChar());
    }
  }

  if (need_closing_quote) {
    if (Eof() || next_char != '"') {
      return absl::InvalidArgumentError(
          "Quoted identifier must end with closing quote!");
    }
    XLS_RETURN_IF_ERROR(DropChar());
  }

  return Token::Identifier(identifier, start_pos);
}

absl::StatusOr<char> Scanner::PeekChar() {
  XLS_RET_CHECK_LT(current_pos_, function_.size())
      << "Function: " << function_ << ": current: " << current_pos_
      << ", max: " << function_.size() - 1;
  return function_.at(current_pos_);
}

absl::Status Scanner::DropChar() {
  XLS_RET_CHECK_LT(current_pos_, function_.size());
  current_pos_++;
  return absl::OkStatus();
}

absl::StatusOr<Ast> Parser::ParseFunction(std::string function) {
  Parser parser(std::move(function));
  return parser.ParseOr();
}

Parser::Parser(std::string function) : scanner_(std::move(function)) {}

absl::StatusOr<Ast> Parser::ParseOr() {
  XLS_ASSIGN_OR_RETURN(auto lhs, ParseAnd());
  if (scanner_.Eof()) {
    return lhs;
  }

  // Assume all ops of equal precedence are left-associative, so keep stacking
  // them on.
  XLS_ASSIGN_OR_RETURN(Token token, scanner_.Peek());
  while (token.kind() == Token::Kind::kOr) {
    XLS_RETURN_IF_ERROR(scanner_.Pop().status());
    XLS_ASSIGN_OR_RETURN(auto rhs, ParseAnd());
    lhs =
        Ast::BinOp(Ast::Kind::kOr, std::move(lhs), std::move(rhs), token.pos());

    if (scanner_.Eof()) {
      return lhs;
    }
    XLS_ASSIGN_OR_RETURN(token, scanner_.Peek());
  }

  return lhs;
}

absl::StatusOr<Ast> Parser::ParseAnd() {
  XLS_ASSIGN_OR_RETURN(auto lhs, ParseXor());
  if (scanner_.Eof()) {
    return lhs;
  }

  // Assume all ops of equal precedence are left-associative, so keep stacking
  // them on.
  XLS_ASSIGN_OR_RETURN(Token token, scanner_.Peek());
  while (token.kind() == Token::Kind::kAnd) {
    XLS_RETURN_IF_ERROR(scanner_.Pop().status());
    XLS_ASSIGN_OR_RETURN(auto rhs, ParseXor());
    lhs = Ast::BinOp(Ast::Kind::kAnd, std::move(lhs), std::move(rhs),
                     token.pos());

    if (scanner_.Eof()) {
      return lhs;
    }
    XLS_ASSIGN_OR_RETURN(token, scanner_.Peek());
  }

  return lhs;
}

absl::StatusOr<Ast> Parser::ParseXor() {
  XLS_ASSIGN_OR_RETURN(auto lhs, ParseInvertNext());
  if (scanner_.Eof()) {
    return lhs;
  }

  // Assume all ops of equal precedence are left-associative, so keep stacking
  // them on.
  XLS_ASSIGN_OR_RETURN(Token token, scanner_.Peek());
  while (token.kind() == Token::Kind::kXor) {
    XLS_RETURN_IF_ERROR(scanner_.Pop().status());
    XLS_ASSIGN_OR_RETURN(auto rhs, ParseInvertNext());
    lhs = Ast::BinOp(Ast::Kind::kXor, std::move(lhs), std::move(rhs),
                     token.pos());

    if (scanner_.Eof()) {
      return lhs;
    }
    XLS_ASSIGN_OR_RETURN(token, scanner_.Peek());
  }

  return lhs;
}

absl::StatusOr<Ast> Parser::ParseInvertNext() {
  // Rather than have to track every potentially stacked negation, we'll just
  // see if we have an odd or even count and apply after processing the
  // following term.
  XLS_ASSIGN_OR_RETURN(Token token, scanner_.Peek());
  bool do_negate = 0;
  while (token.kind() == Token::Kind::kInvertFollowing) {
    XLS_RETURN_IF_ERROR(scanner_.Pop().status());
    do_negate = !do_negate;

    // Don't need to check for EOF here, b/c we must have a following term.
    XLS_ASSIGN_OR_RETURN(token, scanner_.Peek());
  }

  XLS_ASSIGN_OR_RETURN(auto term, ParseInvertPrev());
  if (do_negate) {
    term = Ast::Not(std::move(term), token.pos());
  }

  return term;
}

absl::StatusOr<Ast> Parser::ParseInvertPrev() {
  XLS_ASSIGN_OR_RETURN(auto term, ParseTerm());
  if (scanner_.Eof()) {
    return term;
  }

  XLS_ASSIGN_OR_RETURN(Token token, scanner_.Peek());
  while (token.kind() == Token::Kind::kInvertPrevious) {
    XLS_RETURN_IF_ERROR(scanner_.Pop().status());
    term = Ast::Not(std::move(term), token.pos());

    if (scanner_.Eof()) {
      return term;
    }
    XLS_ASSIGN_OR_RETURN(token, scanner_.Peek());
  }

  return term;
}

absl::StatusOr<Ast> Parser::ParseTerm() {
  XLS_ASSIGN_OR_RETURN(Token token, scanner_.Pop());
  switch (token.kind()) {
    case Token::Kind::kOpenParen: {
      XLS_ASSIGN_OR_RETURN(auto expr, ParseOr());
      XLS_ASSIGN_OR_RETURN(Token closeparen, scanner_.Pop());
      if (closeparen.kind() != Token::Kind::kCloseParen) {
        return absl::InvalidArgumentError(
            absl::StrFormat("Expected close parenthesis, got: %d @ %d",
                            static_cast<int>(token.kind()), token.pos()));
      }
      return expr;
    }
    case Token::Kind::kIdentifier:
      return Ast::Identifier(token.payload(), token.pos());
    case Token::Kind::kLogicZero:
      return Ast::LiteralZero(token.pos());
    case Token::Kind::kLogicOne:
      return Ast::LiteralOne(token.pos());
    default:
      return absl::InvalidArgumentError(
          absl::StrFormat("Invalid token kind: %d @ %d",
                          static_cast<int>(token.kind()), token.pos()));
  }
}

}  // namespace function
}  // namespace netlist
}  // namespace xls
