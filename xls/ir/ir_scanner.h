// Copyright 2020 Google LLC
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

#ifndef THIRD_PARTY_XLS_IR_IR_SCANNER_H_
#define THIRD_PARTY_XLS_IR_IR_SCANNER_H_

#include <string>

#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "xls/common/integral_types.h"
#include "xls/common/logging/logging.h"
#include "xls/common/status/statusor.h"
#include "xls/ir/bits.h"

namespace xls {

enum class TokenType {
  kIdent,
  kKeyword,
  kLiteral,
  kMinus,
  kAdd,
  kColon,
  kGt,
  kLt,
  kDot,
  kComma,
  kCurlOpen,
  kCurlClose,
  kBracketOpen,
  kBracketClose,
  kParenOpen,
  kParenClose,
  kRightArrow,
  kEquals,
};

std::string TokenTypeToString(TokenType token_type);

struct TokenPos {
  int64 lineno;
  int64 colno;

  // Humans think of line 0 column 0 as "1:1" in text editors, typically.
  std::string ToHumanString() const;
};

class Token {
 public:
  // Returns the (singleton) set of keyword strings.
  static const absl::flat_hash_set<std::string>& GetKeywords() {
    static auto* keywords =
        new absl::flat_hash_set<std::string>{"fn", "bits", "ret", "package"};
    return *keywords;
  }

  // Helper factory, returns a token of kKeyword type if "value" is a keyword
  // string, and a token of kIdent type otherwise.
  static Token MakeIdentOrKeyword(absl::string_view value, int64 lineno,
                                  int64 colno) {
    TokenType type =
        GetKeywords().contains(value) ? TokenType::kKeyword : TokenType::kIdent;
    if (value == "true" || value == "false") {
      type = TokenType::kLiteral;
    }
    return Token(type, value, lineno, colno);
  }

  Token(TokenType type, int64 lineno, int64 colno)
      : type_(type), pos_({lineno, colno}) {}

  Token(TokenType type, absl::string_view value, int64 lineno, int64 colno)
      : type_(type), value_(value), pos_({lineno, colno}) {}

  TokenType type() const { return type_; }
  const std::string& value() const { return value_; }
  const TokenPos& pos() const { return pos_; }

  // Returns the token as a (u)int64 value. Token must be a literal. The
  // expected string representation is the same as with
  // ParseNumberAsBits. Returns an error if the number does not fit in a
  // (u)int64.
  xabsl::StatusOr<int64> GetValueInt64() const;

  // Returns the token as a bool value. Token must be a literal. Returns an
  // error if the number does not fit in a bool.
  xabsl::StatusOr<bool> GetValueBool() const;

  // Returns the token as a Bits value. Token must be a literal. The
  // expected string representation is the same as with ParseNumberAsBits.
  xabsl::StatusOr<Bits> GetValueBits() const;

  // Returns whether the token is a negative value. Token must be a literal.
  xabsl::StatusOr<bool> IsNegative() const;

  std::string ToString() const;

 private:
  TokenType type_;
  std::string value_;
  TokenPos pos_;
};

inline std::ostream& operator<<(std::ostream& os, const Token& token) {
  os << token.ToString();
  return os;
}

// Tokenizes the given string and returns the tokens. It maintains precise
// source location information.  Right now this is a eager implementation - it
// tokenizes the whole input. This can be easily changed later to a more demand
// driven tokenization.
xabsl::StatusOr<std::vector<Token>> TokenizeString(absl::string_view str);

class Scanner {
 public:
  static xabsl::StatusOr<Scanner> Create(absl::string_view text);

  // Peeks at the next token in the token stream, or returns an error if we're
  // at EOF and no more tokens are available.
  xabsl::StatusOr<Token> PeekToken() const;

  // Return the current token.
  const Token& PeekTokenOrDie() const {
    XLS_CHECK(!AtEof());
    return tokens_[token_idx_];
  }

  // Helper that makes sure we don't peek past EOF.
  bool PeekTokenIs(TokenType target) const {
    return !AtEof() && PeekTokenOrDie().type() == target;
  }

  // Pop the current token, advance token pointer to next token.
  Token PopToken() {
    XLS_VLOG(3) << "Popping token: " << tokens_[token_idx_];
    return tokens_.at(token_idx_++);
  }

  // Same as PopToken() but returns a status error if we are at EOF (in which
  // case a token cannot be popped).
  xabsl::StatusOr<Token> PopTokenOrError(absl::string_view context = "");

  // As above, but the caller must ensure we are not possibly at EOF: if we are,
  // then the program will CHECK-fail.
  void DropTokenOrDie() { XLS_CHECK_OK(PopTokenOrError().status()); }

  // Attempts to drop a token with type "target" from the token stream, and
  // returns true if it is possible to do so; otherwise, returns false.
  //
  // Note: This function is "EOF safe": trying to drop a token at EOF is ok.
  bool TryDropToken(TokenType target);

  bool TryDropKeyword(absl::string_view which) {
    if (PeekTokenIs(TokenType::kKeyword) && PeekTokenOrDie().value() == which) {
      DropTokenOrDie();
      return true;
    }
    return false;
  }

  // As with PopTokenOrError, but also supplies an error if the token is not of
  // type "target".
  xabsl::StatusOr<Token> PopTokenOrError(TokenType target,
                                         absl::string_view context = "");

  // Wrapper around PopTokenOrError(target) above that can be used with
  // XLS_RETURN_IF_ERROR.
  absl::Status DropTokenOrError(TokenType target,
                                absl::string_view context = "");

  // Pop a keyword token with keyword (payload) "keyword".
  //
  // Returns an absl::Status error if we cannot.
  absl::Status DropKeywordOrError(absl::string_view keyword);

  // Check if more tokens are available.
  bool AtEof() const { return token_idx_ >= tokens_.size(); }

 private:
  explicit Scanner(std::vector<Token> tokens) : tokens_(tokens) {}

  int64 token_idx_ = 0;
  std::vector<Token> tokens_;
};

}  // namespace xls

#endif  // THIRD_PARTY_XLS_IR_IR_SCANNER_H_
