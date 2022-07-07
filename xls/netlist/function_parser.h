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

// Defines routines for parsing the "function" component of cell library
// "cell" groups.
// Specifications taken from the liberty reference:
// https://people.eecs.berkeley.edu/~alanmi/publications/other/liberty07_03.pdf
#ifndef XLS_NETLIST_FUNCTION_PARSER_H_
#define XLS_NETLIST_FUNCTION_PARSER_H_

#include <cstdint>
#include <string>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/types/optional.h"

namespace xls {
namespace netlist {
namespace function {

// Represents a single token in a cell group "function" attribute.
class Token {
 public:
  // The "kind" of the token; the lexographic element it represents.
  enum class Kind {
    kIdentifier,
    kOpenParen,
    kCloseParen,
    kAnd,
    kOr,
    kXor,
    kInvertFollowing,
    kInvertPrevious,
    kLogicOne,
    kLogicZero,
  };

  // Builder for identifier tokens.
  static Token Identifier(const std::string& s, int64_t pos);

  // Builder for all other tokens.
  static Token Simple(Kind kind, int64_t pos);

  Token(Kind kind, int64_t pos,
        std::optional<std::string> payload = absl::nullopt)
      : kind_(kind), pos_(pos), payload_(payload) {}

  Kind kind() { return kind_; }
  int64_t pos() { return pos_; }
  std::string payload() { return payload_.value(); }

 private:
  Kind kind_;
  int64_t pos_;
  std::optional<std::string> payload_;
};

// Scans a function attribute and returns the component tokens.
// Only needs a single element of lookahead, so repeated calls to Peek() without
// an intervening Pop() will always return the same token.
class Scanner {
 public:
  Scanner(std::string function);
  absl::StatusOr<Token> Pop();
  absl::StatusOr<Token> Peek();

  bool Eof() { return current_pos_ == function_.size(); }

 private:
  // Returns true if this character starts an identifier.
  bool IsIdentifierStart(char next_char);

  // Processes a single identifier.
  absl::StatusOr<Token> ScanIdentifier();

  // Looks at and optionally consumes the next character of input.
  absl::StatusOr<char> PeekChar();
  absl::Status DropChar();

  // Drops all spaces until the next token.
  absl::Status DropSpaces();

  std::string function_;
  // Set if we've peeked @ a token. Calling Pop() will clear it.
  std::optional<Token> peeked_;
  int64_t current_pos_;
};

// Represents the logical computation represented by a function attribute in a
// pin's "function" attribute (or subtree thereof).
class Ast {
 public:
  // Since our grammer is so simple, we'll squish all AST node types into one
  // enum.
  enum class Kind {
    kIdentifier,
    kLiteralZero,
    kLiteralOne,
    kAnd,
    kOr,
    kXor,
    kNot
  };

  static Ast Identifier(const std::string& name, int64_t pos) {
    return Ast(Kind::kIdentifier, pos, name);
  }

  static Ast LiteralOne(int64_t pos) { return Ast(Kind::kLiteralOne, pos); }

  static Ast LiteralZero(int64_t pos) { return Ast(Kind::kLiteralZero, pos); }

  static Ast Not(Ast expr, int64_t pos) {
    Ast ast(Kind::kNot, pos);
    ast.children_.push_back(expr);
    return ast;
  }

  static Ast BinOp(Kind kind, Ast lhs, Ast rhs, int64_t pos) {
    Ast ast(kind, pos);
    ast.children_.push_back(lhs);
    ast.children_.push_back(rhs);
    return ast;
  }

  Ast(Kind kind, int64_t pos, std::string name = "")
      : kind_(kind), pos_(pos), name_(name) {}

  Kind kind() const { return kind_; }
  int64_t pos() const { return pos_; }
  std::string name() const { return name_; }
  const std::vector<Ast>& children() const { return children_; }

 private:
  Kind kind_;
  int64_t pos_;
  std::string name_;
  std::vector<Ast> children_;
};

// Creates a function AST from an input cell/pin function description.
class Parser {
 public:
  static absl::StatusOr<Ast> ParseFunction(std::string function);

 private:
  Parser(std::string function);

  // Parsers for each syntactic element in a function, here ordered from highest
  // to lowest precedence.
  absl::StatusOr<Ast> ParseTerm();
  absl::StatusOr<Ast> ParseInvertPrev();
  absl::StatusOr<Ast> ParseInvertNext();
  absl::StatusOr<Ast> ParseXor();
  absl::StatusOr<Ast> ParseAnd();
  absl::StatusOr<Ast> ParseOr();

  Scanner scanner_;
};

}  // namespace function
}  // namespace netlist
}  // namespace xls

#endif  // XLS_NETLIST_FUNCTION_PARSER_H_
