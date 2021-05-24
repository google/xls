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

#ifndef XLS_DSLX_CPP_SCANNER_H_
#define XLS_DSLX_CPP_SCANNER_H_

#include <string>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/types/variant.h"
#include "xls/common/logging/logging.h"
#include "xls/common/status/status_macros.h"
#include "xls/dslx/pos.h"
#include "xls/dslx/scanner_keywords.inc"

namespace xls::dslx {

#define XLS_DSLX_TOKEN_KINDS(X)                                        \
  /* enum, pyname, str */                                              \
  X(kDot, DOT, ".")                                                    \
  X(kEof, EOF, "EOF")                                                  \
  X(kKeyword, KEYWORD, "keyword")                                      \
  X(kIdentifier, IDENTIFIER, "identifier")                             \
  X(kNumber, NUMBER, "number")                                         \
  X(kCharacter, CHARACTER, "character")                                \
  X(kOParen, OPAREN, "(")                                              \
  X(kCParen, CPAREN, ")")                                              \
  X(kOBrace, OBRACE, "{")                                              \
  X(kCBrace, CBRACE, "}")                                              \
  X(kPlus, PLUS, "+")                                                  \
  X(kMinus, MINUS, "-")                                                \
  X(kPlusColon, PLUS_COLON, "+:")                                      \
  X(kDoubleCAngle, DOUBLE_CANGLE, ">>")                                \
  X(kDoubleOAngle, DOUBLE_OANGLE, "<<")                                \
  X(kEquals, EQUALS, "=")                                              \
  X(kDoubleColon, DOUBLE_COLON, "::")                                  \
  X(kDoublePlus, DOUBLE_PLUS, "++")                                    \
  X(kDoubleEquals, DOUBLE_EQUALS, "==")                                \
  X(kCAngleEquals, CANGLE_EQUALS, ">=")                                \
  X(kOAngleEquals, OANGLE_EQUALS, "<=")                                \
  X(kBangEquals, BANG_EQUALS, "!=")                                    \
  X(kCAngle, CANGLE, ">")                                              \
  X(kOAngle, OANGLE, "<")                                              \
  X(kBang, BANG, "!")                                                  \
  X(kOBrack, OBRACK, "[")                                              \
  X(kCBrack, CBRACK, "]")                                              \
  X(kColon, COLON, ":")                                                \
  X(kComma, COMMA, ",")                                                \
  X(kDoubleQuote, DOUBLE_QUOTE, "\"")                                  \
  X(kStar, STAR, "*")                                                  \
  X(kSlash, SLASH, "/")                                                \
  X(kPercent, PERCENT, "%")                                            \
  X(kArrow, ARROW, "->")                                               \
  X(kSemi, SEMI, ";")                                                  \
  X(kAmpersand, AMPERSAND, "&")                                        \
  X(kDoubleAmpersand, DOUBLE_AMPERSAND, "&&")                          \
  X(kBar, BAR, "|")                                                    \
  X(kDoubleBar, DOUBLE_BAR, "||")                                      \
  X(kHat, HAT, "^")                                                    \
  X(kFatArrow, FAT_ARROW, "=>")                                        \
  X(kDoubleDot, DOUBLE_DOT, "..")                                      \
  X(kEllipsis, ELLIPSIS, "...")                                        \
  X(kHash, HASH, "#")                                                  \
  /* When in whitespace/comment mode; e.g. for syntax highlighting. */ \
  X(kWhitespace, WHITESPACE, "whitespace")                             \
  X(kComment, COMMENT, "comment")

#define XLS_FIRST_COMMA(A, ...) A,

enum class TokenKind { XLS_DSLX_TOKEN_KINDS(XLS_FIRST_COMMA) };

std::string TokenKindToString(TokenKind kind);

absl::StatusOr<TokenKind> TokenKindFromString(absl::string_view s);

inline std::ostream& operator<<(std::ostream& os, TokenKind kind) {
  os << TokenKindToString(kind);
  return os;
}

enum class Keyword { XLS_DSLX_KEYWORDS(XLS_FIRST_COMMA) };

std::string KeywordToString(Keyword keyword);

absl::optional<Keyword> KeywordFromString(absl::string_view s);

// Returns a singleton set of type keywords.
const absl::flat_hash_set<Keyword>& GetTypeKeywords();

// Token yielded by the Scanner below.
class Token {
 public:
  Token(TokenKind kind, Span span,
        absl::optional<std::string> value = absl::nullopt)
      : kind_(kind), span_(span), payload_(value) {}
  Token(Span span, Keyword keyword)
      : kind_(TokenKind::kKeyword), span_(span), payload_(keyword) {}

  TokenKind kind() const { return kind_; }
  const Span& span() const { return span_; }

  absl::optional<std::string> GetValue() const {
    if (absl::holds_alternative<Keyword>(payload_)) {
      return KeywordToString(GetKeyword());
    }
    return absl::get<absl::optional<std::string>>(payload_);
  }

  // Note: assumes that the payload is not a keyword.
  const std::string& GetStringValue() const {
    return *absl::get<absl::optional<std::string>>(payload_);
  }

  absl::StatusOr<int64_t> GetValueAsInt64() const;

  const absl::variant<absl::optional<std::string>, Keyword>& GetPayload()
      const {
    return payload_;
  }

  bool IsKeywordIn(const std::unordered_set<Keyword>& targets) const {
    return kind_ == TokenKind::kKeyword &&
           (targets.find(GetKeyword()) != targets.end());
  }

  bool IsTypeKeyword() const {
    return kind_ == TokenKind::kKeyword &&
           GetTypeKeywords().contains(GetKeyword());
  }

  Keyword GetKeyword() const { return absl::get<Keyword>(payload_); }

  bool IsKeyword(Keyword target) const {
    return kind_ == TokenKind::kKeyword && GetKeyword() == target;
  }
  bool IsIdentifier(absl::string_view target) const {
    return kind_ == TokenKind::kIdentifier && *GetValue() == target;
  }
  bool IsNumber(absl::string_view target) const {
    return kind_ == TokenKind::kNumber && *GetValue() == target;
  }

  bool IsKindIn(
      absl::Span<absl::variant<TokenKind, Keyword> const> targets) const {
    for (auto target : targets) {
      if (absl::holds_alternative<TokenKind>(target)) {
        if (kind() == absl::get<TokenKind>(target)) {
          return true;
        }
      } else {
        if (IsKeyword(absl::get<Keyword>(target))) {
          return true;
        }
      }
    }
    return false;
  }

  // Returns a string that represents this token suitable for use in displaying
  // this token for user error reporting; e.g. "keyword:true".
  std::string ToErrorString() const;

  std::string ToString() const;

  std::string ToRepr() const;

 private:
  TokenKind kind_;
  Span span_;
  absl::variant<absl::optional<std::string>, Keyword> payload_;
};

// Converts the conceptual character stream in a string of text into a stream of
// tokens according to the DSLX syntax.
class Scanner {
 public:
  Scanner(std::string filename, std::string text,
          bool include_whitespace_and_comments = false)
      : filename_(std::move(filename)),
        text_(std::move(text)),
        include_whitespace_and_comments_(include_whitespace_and_comments) {}

  // Gets the current position in the token stream. Note that the position in
  // the token stream can change on a Pop(), because whitespace and comments
  // may be discarded.
  //
  // TODO(leary): 2020-09-08 Attempt to privatize this, ideally consumers would
  // only care about the positions of tokens, not of the scanner itself.
  Pos GetPos() const {
    return Pos(filename_, lineno_, colno_);
  }

  // Pops a token from the current position in the character stream, or returns
  // a status error if no token can be scanned out.
  //
  // Note that if the current position in the character stream is whitespace
  // before the EOF, an EOF-kind token will be returned, and subsequently
  // AtEof() will be true.
  absl::StatusOr<Token> Pop();

  // Pops all tokens from the token stream until it is extinguished (as
  // determined by `AtEof()`) and returns them as a sequence.
  absl::StatusOr<std::vector<Token>> PopAll() {
    std::vector<Token> tokens;
    while (!AtEof()) {
      XLS_ASSIGN_OR_RETURN(Token tok, Pop());
      tokens.push_back(tok);
    }
    return tokens;
  }

  // Returns whether the scanner reached end of file: no more characters!
  //
  // Note: if there is trailing whitespace in the file, AtEof() will return
  // false until you try to Pop(), at which point you'll see an EOF-kind token.
  bool AtEof() const {
    return AtCharEof();
  }

  // Proceeds through the stream until an unescaped double quote is encountered.
  absl::StatusOr<std::string> ScanUntilDoubleQuote();

 private:
  // Helper routine that creates a canonically-formatted scan error (which uses
  // the status code for an InvalidArgumentError, on the assumption the invalid
  // argument is the input text character stream). Since `absl::Status` cannot
  // easily encode a payload, this canonical formatting is used to convey
  // metadata; e.g. the position at which the scan error occurred, so it may be
  // raised into Python land as a more structured form of exception.
  absl::Status ScanError(const Span& span, std::string message) {
    return absl::InvalidArgumentError(
        absl::StrFormat("ScanError: %s %s", span.ToString(), message));
  }

  // Determines whether string "s" matches a keyword -- if so, returns the
  // keyword enum that it corresponds to. Otherwise, typically the caller will
  // assume s is an identifier.
  static absl::optional<Keyword> GetKeyword(absl::string_view s);

  // Scans a number token out of the character stream. Note the number may have
  // a base determined by a radix-noting prefix; e.g. "0x" or "0b".
  //
  // Precondition: The character stream must be positioned over either a digit
  // or a minus sign.
  absl::StatusOr<Token> ScanNumber(char startc, const Pos& start_pos);

  // Scans a character literal from the character stream as a character token.
  //
  // Precondition: The character stream must be positioned at an open quote.
  absl::StatusOr<Token> ScanChar(const Pos& start_pos);

  // Scans from the current position until ftake returns false or EOF is
  // reached.
  std::string ScanWhile(std::string s, const std::function<bool(char)>& ftake) {
    while (!AtCharEof()) {
      char peek = PeekChar();
      if (!ftake(peek)) {
        break;
      }
      s.append(1, PopChar());
    }
    return s;
  }
  std::string ScanWhile(char c, const std::function<bool(char)>& ftake) {
    return ScanWhile(std::string(1, c), ftake);
  }

  // Scans the identifier-looping entity beginning with startc.
  //
  // Args:
  //  startc: first (already popped) character of the identifier/keyword token.
  //  start_pos: start position for the identifier/keyword token.
  //
  // Returns:
  //  Either a keyword (if the scanned identifier turns out to be in the set of
  //  keywords) or an identifier token.
  absl::StatusOr<Token> ScanIdentifierOrKeyword(char startc,
                                                const Pos& start_pos);

  // Drops comments/whitespace from the current scan position in the character
  // stream.
  void DropCommentsAndLeadingWhitespace();

  // Returns whether the current character stream cursor is positioned at a
  // whitespace character.
  bool AtWhitespace() const;

  // Returns whether the input character stream has been exhausted.
  bool AtCharEof() const {
    XLS_CHECK_LE(index_, text_.size());
    return index_ == text_.size();
  }

  // Peeks at the character at the head of the character stream.
  //
  // Precondition: have not hit the end of the character stream (i.e.
  // `!AtCharEof()`).
  char PeekChar() const {
    XLS_CHECK_LT(index_, text_.size());
    return text_[index_];
  }

  // Peeks at the character *after* the head of the character stream.
  //
  // If there is no such character in the character stream, returns '\0'.
  //
  // Note: This is a workable API because we generally "peek at 2nd character
  // and compare to something".
  char PeekChar2OrNull() const {
    if (index_ + 1 >= text_.size()) {
      return '\0';
    }
    return text_[index_ + 1];
  }

  // Pops a character from the head of the character stream and returns it.
  //
  // Precondition: the character stream is not extinguished (in which case this
  // routine will check-fail).
  ABSL_MUST_USE_RESULT char PopChar();

  // Drops "count" characters from the head of the character stream.
  //
  // Note: As with PopChar() if the character stream is extinguished when a
  // character is dropped this will check-fail.
  void DropChar(int64_t count = 1);

  // Attempts to drop a character equal to "target": if it is present at the
  // head of the character stream, drops it and return true; otherwise
  // (including if we are at end of file) returns false.
  bool TryDropChar(char target);

  // Pops all the characters from the current character cursor to the end of
  // line (or end of file) and returns that. (This is useful presuming a leading
  // EOL-comment-delimiter was observed.)
  absl::StatusOr<Token> PopComment(const Pos& start_pos);

  // Pops all the whitespace characters and returns them as a token. This is
  // useful e.g. in syntax-highlighting mode where we want whitespace and
  // comments to be preserved in the token stream.
  //
  // Precondition: the character stream cursor must be positioned over
  // whitespace.
  absl::StatusOr<Token> PopWhitespace(const Pos& start_pos);

  // Attempts to pop either whitespace (as a token) or a comment (as a token) at
  // the current character stream position. If the character stream is
  // extinguished, returns an EOF token.
  //
  // If none of whitespace, comment, or EOF is observed, returns nullopt.
  absl::StatusOr<absl::optional<Token>> TryPopWhitespaceOrComment();

  // Reads in the next "character" (or escape sequence) in the stream. A
  // string is returned instead of a char, since multi-byte Unicode characters
  // are valid constituents of a string.
  absl::StatusOr<std::string> ProcessNextStringChar();

  std::string filename_;
  std::string text_;
  bool include_whitespace_and_comments_;
  int64_t index_ = 0;
  int64_t lineno_ = 0;
  int64_t colno_ = 0;
};

}  // namespace xls::dslx

#endif  // XLS_DSLX_CPP_SCANNER_H_
