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

#ifndef XLS_DSLX_FRONTEND_SCANNER_H_
#define XLS_DSLX_FRONTEND_SCANNER_H_

#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/base/attributes.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xls/common/status/status_macros.h"
#include "xls/common/test_macros.h"
#include "xls/dslx/frontend/comment_data.h"
#include "xls/dslx/frontend/pos.h"
#include "xls/dslx/frontend/token.h"

namespace xls::dslx {

// Helper routine that creates a canonically-formatted scan error (which uses
// the status code for an InvalidArgumentError, on the assumption the invalid
// argument is the input text character stream). Since the former XLS `Status`
// could not easily encode a payload, this canonical formatting is used to
// convey metadata; e.g. the position at which the scan error occurred.
//
// TODO(leary): 2023-08-21 We can now use the payloads present in absl::Status
// (we no longer have our own project-specific Status) -- we should not need to
// encode conditional data in the string and parse it out.
absl::Status ScanErrorStatus(const Span& span, std::string_view message);

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
  Pos GetPos() const { return Pos(filename_, lineno_, colno_); }

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
  bool AtEof() const { return AtCharEof(); }

  void EnableDoubleCAngle() { double_c_angle_enabled_ = true; }
  void DisableDoubleCAngle() { double_c_angle_enabled_ = false; }

  std::string_view filename() const { return filename_; }

  absl::Span<const CommentData> comments() const { return comments_; }

  // Destructively pops out comment data from this object.
  std::vector<CommentData> PopComments() {
    std::vector<CommentData> result = std::move(comments_);
    comments_.clear();
    return result;
  }

 private:
  // These tests generally use the private ScanUntilDoubleQuote() helper.
  XLS_FRIEND_TEST(ScannerTest, RecognizesEscapes);
  XLS_FRIEND_TEST(ScannerTest, StringCharUnicodeBadStartChar);
  XLS_FRIEND_TEST(ScannerTest, StringCharUnicodeBadTerminator);
  XLS_FRIEND_TEST(ScannerTest, StringCharUnicodeEscapeEmpty);
  XLS_FRIEND_TEST(ScannerTest, StringCharUnicodeEscapeNonHexDigit);
  XLS_FRIEND_TEST(ScannerTest, StringCharUnicodeInvalidSequence);
  XLS_FRIEND_TEST(ScannerTest, StringCharUnicodeMoreThanSixDigits);

  // Determines whether string "s" matches a keyword -- if so, returns the
  // keyword enum that it corresponds to. Otherwise, typically the caller will
  // assume s is an identifier.
  static std::optional<Keyword> GetKeyword(std::string_view s);

  // Proceeds through the stream until an unescaped double quote is encountered.
  //
  // Note: since there are special character escapes in strings, they are
  // specially handled by this routine when an open quote is encountered.
  absl::StatusOr<std::string> ScanUntilDoubleQuote();

  // Scans a number token out of the character stream. Note the number may have
  // a base determined by a radix-noting prefix; e.g. "0x" or "0b".
  //
  // Precondition: The character stream must be positioned over either a digit
  // or a minus sign.
  absl::StatusOr<Token> ScanNumber(char startc, const Pos& start_pos);

  // Scans a string token out of the character stream -- character cursor should
  // be over the opening quote character.
  absl::StatusOr<Token> ScanString(const Pos& start_pos);

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

  // Drops leading whitespace from the current scan position in the character
  // stream.
  void DropLeadingWhitespace();

  // Returns whether the current character stream cursor is positioned at a
  // whitespace character.
  bool AtWhitespace() const;

  // Returns whether the input character stream has been exhausted.
  bool AtCharEof() const {
    CHECK_LE(index_, text_.size());
    return index_ == text_.size();
  }

  // Peeks at the character at the head of the character stream.
  //
  // Precondition: have not hit the end of the character stream (i.e.
  // `!AtCharEof()`).
  char PeekChar() const {
    CHECK_LT(index_, text_.size());
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
  Token PopComment(const Pos& start_pos);

  // Pops all the whitespace characters and returns them as a token. This is
  // useful e.g. in syntax-highlighting mode where we want whitespace and
  // comments to be preserved in the token stream.
  //
  // Precondition: the character stream cursor must be positioned over
  // whitespace.
  absl::StatusOr<Token> PopWhitespace(const Pos& start_pos);

  // Attempts to pop a comment and, if successful, returns the comment data.
  std::optional<CommentData> TryPopComment();

  // Attempts to pop either whitespace (as a token) or a comment (as a token) at
  // the current character stream position. If the character stream is
  // extinguished, returns an EOF token.
  //
  // If none of whitespace, comment, or EOF is observed, returns nullopt.
  absl::StatusOr<std::optional<Token>> TryPopWhitespaceOrComment();

  // Returns the next character literal.
  absl::StatusOr<char> ScanCharLiteral();

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
  bool double_c_angle_enabled_ = true;
  std::vector<CommentData> comments_;
};

}  // namespace xls::dslx

#endif  // XLS_DSLX_FRONTEND_SCANNER_H_
