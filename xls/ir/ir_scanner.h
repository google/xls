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

#ifndef XLS_IR_IR_SCANNER_H_
#define XLS_IR_IR_SCANNER_H_

#include <cstdint>
#include <ostream>
#include <string>
#include <string_view>
#include <vector>

#include "absl/base/no_destructor.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xls/common/logging/logging.h"
#include "xls/ir/bits.h"

namespace xls {

enum class LexicalTokenType {
  kAdd,
  kBracketClose,
  kBracketOpen,
  kColon,
  kComma,
  kCurlClose,
  kCurlOpen,
  kDot,
  kEquals,
  kGt,
  kIdent,
  kKeyword,
  kLiteral,
  kLt,
  kMinus,
  kParenClose,
  kParenOpen,
  kQuotedString,
  kRightArrow,
  kHash,
};

std::string LexicalTokenTypeToString(LexicalTokenType token_type);

struct TokenPos {
  int64_t lineno;
  int64_t colno;

  // Humans think of line 0 column 0 as "1:1" in text editors, typically.
  std::string ToHumanString() const;
};

class Token {
 public:
  // Returns the (singleton) set of keyword strings.
  static const absl::flat_hash_set<std::string>& GetKeywords() {
    // TODO(google/xls#1010) 2023-06-05 Verify these never used if kIdent needed
    static const absl::NoDestructor<absl::flat_hash_set<std::string>> keywords(
        {"fn", "bits", "token", "ret", "package", "proc", "chan", "reg", "next",
         "block", "clock", "instantiation", "top", "file_number",
         "proc_instantiation"});
    return *keywords;
  }

  // Helper factory, returns a token of kKeyword type if "value" is a keyword
  // string, and a token of kIdent type otherwise.
  static Token MakeIdentOrKeyword(std::string_view value, int64_t lineno,
                                  int64_t colno) {
    LexicalTokenType type = GetKeywords().contains(value)
                                ? LexicalTokenType::kKeyword
                                : LexicalTokenType::kIdent;
    if (value == "true" || value == "false") {
      type = LexicalTokenType::kLiteral;
    }
    return Token(type, value, lineno, colno);
  }

  Token(LexicalTokenType type, int64_t lineno, int64_t colno)
      : type_(type), pos_({lineno, colno}) {}

  Token(LexicalTokenType type, std::string_view value, int64_t lineno,
        int64_t colno)
      : type_(type), value_(value), pos_({lineno, colno}) {}

  LexicalTokenType type() const { return type_; }
  const std::string& value() const { return value_; }
  const TokenPos& pos() const { return pos_; }

  // Returns the token as a (u)int64_t value. Token must be a literal. The
  // expected string representation is the same as with
  // ParseNumberAsBits. Returns an error if the number does not fit in a
  // (u)int64_t.
  absl::StatusOr<int64_t> GetValueInt64() const;

  // Returns the token as a bool value. Token must be a literal. Returns an
  // error if the number does not fit in a bool.
  absl::StatusOr<bool> GetValueBool() const;

  // Returns the token as a Bits value. Token must be a literal. The
  // expected string representation is the same as with ParseNumberAsBits.
  absl::StatusOr<Bits> GetValueBits() const;

  // Returns whether the token is a negative value. Token must be a literal.
  absl::StatusOr<bool> IsNegative() const;

  std::string ToString() const;

 private:
  LexicalTokenType type_;
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
absl::StatusOr<std::vector<Token>> TokenizeString(std::string_view str);

class Scanner {
 public:
  static absl::StatusOr<Scanner> Create(std::string_view text);

  // Peeks at the next token in the token stream, or returns an error if we're
  // at EOF and no more tokens are available.
  absl::StatusOr<Token> PeekToken() const;

  // Return the current token.
  const Token& PeekTokenOrDie() const {
    CHECK(!AtEof());
    return tokens_[token_idx_];
  }

  // Returns true if the next token is the given type.
  bool PeekTokenIs(LexicalTokenType target) const {
    return !AtEof() && PeekTokenOrDie().type() == target;
  }

  // Returns true if the nth next token is the given type. If `n` is zero this
  // peeks at the immediate next token.
  bool PeekNthTokenIs(int64_t n, LexicalTokenType target) const {
    return (token_idx_ + n < tokens_.size()) &&
           tokens_[token_idx_ + n].type() == target;
  }

  // Pop the current token, advance token pointer to next token.
  Token PopToken() {
    VLOG(6) << "Popping token: " << tokens_[token_idx_];
    return tokens_.at(token_idx_++);
  }

  // Same as PopToken() but returns a status error if we are at EOF (in which
  // case a token cannot be popped).
  absl::StatusOr<Token> PopTokenOrError(std::string_view context = "");

  // As above, but the caller must ensure we are not possibly at EOF: if we are,
  // then the program will CHECK-fail.
  void DropTokenOrDie() { CHECK_OK(PopTokenOrError().status()); }

  // Attempts to drop a token with type "target" from the token stream, and
  // returns true if it is possible to do so; otherwise, returns false.
  //
  // Note: This function is "EOF safe": trying to drop a token at EOF is ok.
  bool TryDropToken(LexicalTokenType target);

  bool TryDropKeyword(std::string_view which) {
    if (PeekTokenIs(LexicalTokenType::kKeyword) &&
        PeekTokenOrDie().value() == which) {
      DropTokenOrDie();
      return true;
    }
    return false;
  }

  // As with PopTokenOrError, but also supplies an error if the token is not of
  // type "target".
  absl::StatusOr<Token> PopTokenOrError(LexicalTokenType target,
                                        std::string_view context = "");

  // Pops the token if it is an identifier or keyword, or returns an error
  // otherwise. Useful for tokens which are allowed to be arbitrary
  // identifier-like strings including keywords.
  absl::StatusOr<Token> PopKeywordOrIdentToken(std::string_view context = "");

  // Pops the token if it is a literal or string otherwise.
  // TODO(hzeller) 2023-06-05 Make a generic PopOneOfToken() for this and the
  // method above.
  absl::StatusOr<Token> PopLiteralOrStringToken(std::string_view context);

  // Wrapper around PopTokenOrError(target) above that can be used with
  // XLS_RETURN_IF_ERROR.
  absl::Status DropTokenOrError(LexicalTokenType target,
                                std::string_view context = "");

  // Pop a keyword token with keyword (payload) "keyword".
  //
  // Returns an absl::Status error if we cannot.
  absl::Status DropKeywordOrError(std::string_view keyword);

  // Check if more tokens are available.
  bool AtEof() const { return token_idx_ >= tokens_.size(); }

 private:
  explicit Scanner(std::vector<Token> tokens) : tokens_(tokens) {}

  int64_t token_idx_ = 0;
  std::vector<Token> tokens_;
};

}  // namespace xls

#endif  // XLS_IR_IR_SCANNER_H_
