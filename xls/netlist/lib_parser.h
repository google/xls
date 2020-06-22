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

// Infrastructure for parsing ".lib" files (cell libraries).
//
// Note that these files can be quite large (on the order of gigabytes) so we
// performance optimize this a bit.

#ifndef XLS_NETLIST_LIB_PARSER_H_
#define XLS_NETLIST_LIB_PARSER_H_

#include <fstream>
#include <string>

#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/types/optional.h"
#include "xls/common/logging/logging.h"
#include "xls/common/status/status_macros.h"
#include "xls/common/status/statusor.h"

namespace xls {
namespace netlist {
namespace cell_lib {

// Represents a position in the cell library text file.
struct Pos {
  int64 lineno;
  int64 colno;

  std::string ToHumanString() const {
    return absl::StrFormat("%d:%d", lineno + 1, colno + 1);
  }
};

// Wraps a file as a character stream with a 1- or 2-character lookahead
// interface.
class CharStream {
 public:
  static xabsl::StatusOr<CharStream> FromPath(absl::string_view path);
  static xabsl::StatusOr<CharStream> FromText(std::string text);

  ~CharStream() {
    if (if_.has_value()) {
      if_->close();
    }
  }

  CharStream(CharStream&& other) = default;

  Pos GetPos() const { return pos_; }
  bool AtEof() const {
    if (if_.has_value()) {
      return if_->eof();
    }
    return cursor_ >= text_.size();
  }
  char PeekCharOrDie() {
    if (if_.has_value()) {
      XLS_DCHECK(!if_->eof());
      return if_->peek();
    }
    XLS_DCHECK_LT(cursor_, text_.size());
    return text_[cursor_];
  }
  char PopCharOrDie() {
    char c = PeekCharOrDie();
    BumpPos(c);
    return c;
  }
  void DropCharOrDie() { (void)PopCharOrDie(); }

  // Attempts to drop character "c" from the character stream and returns true
  // if it can.
  bool TryDropChar(char c) {
    if (!AtEof() && PeekCharOrDie() == c) {
      DropCharOrDie();
      return true;
    }
    return false;
  }

  // Attempts to pop c0 followed by c1 in an atomic fashion.
  bool TryDropChars(char c0, char c1) {
    if (AtEof()) {
      return false;
    }
    if (TryDropChar(c0)) {
      if (AtEof() || PeekCharOrDie() != c1) {
        Unget(c0);
        return false;
      }
      DropCharOrDie();
      return true;
    }
    return false;
  }

 private:
  CharStream(std::ifstream file_stream) : if_(std::move(file_stream)) {}
  CharStream(std::string text) : text_(std::move(text)) {}

  void Unget(char c) {
    cursor_--;
    if (c == '\n') {
      pos_.lineno--;
      pos_.colno = last_colno_;
    } else {
      pos_.colno--;
    }
    if (if_.has_value()) {
      if_->unget();
    }
  }

  void BumpPos(char c) {
    cursor_++;
    last_colno_ = pos_.colno;
    if (c == '\n') {
      pos_.lineno++;
      pos_.colno = 0;
    } else {
      pos_.colno++;
    }
  }

  Pos pos_ = {0, 0};

  // We have both ifstream mode and text mode data, we store both inline.

  // ifstream mode
  absl::optional<std::ifstream> if_;

  // text mode
  std::string text_;
  int64 cursor_ = 0;
  int64 last_colno_ = 0;
};

enum class TokenKind {
  kIdentifier,
  kOpenParen,
  kCloseParen,
  kOpenCurl,
  kCloseCurl,
  kSemi,
  kColon,
  kQuotedString,
  kNumber,
  kComma,
};

std::string TokenKindToString(TokenKind kind);

// Represents a token in the file's token stream.
class Token {
 public:
  static Token Identifier(Pos pos, std::string s) {
    return Token(TokenKind::kIdentifier, pos, s);
  }
  static Token QuotedString(Pos pos, std::string s) {
    return Token(TokenKind::kQuotedString, pos, s);
  }
  static Token Number(Pos pos, std::string s) {
    return Token(TokenKind::kNumber, pos, s);
  }
  static Token Simple(Pos pos, TokenKind kind) { return Token(kind, pos); }

  Token(TokenKind kind, Pos pos,
        absl::optional<std::string> payload = absl::nullopt)
      : kind_(kind), pos_(pos), payload_(payload) {}

  std::string ToString() const {
    return absl::StrFormat("Token(%s, %s, \"%s\")", pos_.ToHumanString(),
                           TokenKindToString(kind_),
                           payload_.has_value() ? payload_.value() : "null");
  }

  TokenKind kind() const { return kind_; }
  const Pos& pos() const { return pos_; }
  absl::string_view payload() const { return payload_.value(); }
  std::string PopPayload() { return std::move(payload_.value()); }

 private:
  TokenKind kind_;
  Pos pos_;
  absl::optional<std::string> payload_;
};

inline std::ostream& operator<<(std::ostream& os, const Token& token) {
  os << token.ToString();
  return os;
}

// Converts a stream of characters to a stream of tokens.
class Scanner {
 public:
  explicit Scanner(CharStream* cs) : cs_(cs) { DropWhitespaceAndComments(); }

  // Pops a token off of the token stream.
  xabsl::StatusOr<Token> Pop() {
    XLS_RETURN_IF_ERROR(Peek().status());
    Token result = std::move(lookahead_.value());
    lookahead_ = absl::nullopt;
    return result;
  }

  xabsl::StatusOr<const Token*> Peek() {
    if (lookahead_.has_value()) {
      return &lookahead_.value();
    }
    XLS_RETURN_IF_ERROR(PeekInternal());
    return Peek();
  }

  bool AtEof() const { return !lookahead_.has_value() && cs_->AtEof(); }

  Pos GetPos() {
    if (lookahead_.has_value()) {
      return lookahead_.value().pos();
    }
    return cs_->GetPos();
  }

 private:
  static bool IsIdentifierStart(char c) { return std::isalpha(c); }
  static bool IsIdentifierRest(char c) {
    return std::isalpha(c) || std::isdigit(c) || c == '_';
  }
  static bool IsWhitespace(char c) { return c == ' ' || c == '\n'; }
  static bool IsNumberRest(char c) {
    return std::isdigit(c) || c == '.' || c == 'e' || c == '-';
  }

  // Scans an identifier token.
  xabsl::StatusOr<Token> ScanIdentifier();

  // Scans a number token.
  xabsl::StatusOr<Token> ScanNumber();

  // Scans a quoted string token. Character stream cursor should be over the
  // starting quote character.
  xabsl::StatusOr<Token> ScanQuotedString();

  // Drops whitespace, comments, and line continuation characters.
  void DropWhitespaceAndComments() {
  restart:
    while (!cs_->AtEof() && IsWhitespace(cs_->PeekCharOrDie())) {
      cs_->DropCharOrDie();
    }
    if (cs_->TryDropChars('/', '*')) {
      while (!cs_->AtEof() && !cs_->TryDropChars('*', '/')) {
        cs_->DropCharOrDie();
      }
      goto restart;
    }
    if (cs_->TryDropChars('/', '/')) {
      while (!cs_->AtEof() && !cs_->TryDropChar('\n')) {
        cs_->DropCharOrDie();
      }
      goto restart;
    }
    if (cs_->TryDropChars('\\', '\n')) {
      goto restart;
    }
  }

  // Peeks at a token and populates it as lookahead_.
  absl::Status PeekInternal();

  absl::optional<Token> lookahead_;
  CharStream* cs_;
};

// Grammar looks like:
//
//  top ::= "library" "(" identifier ")" "{"
//        entry *
//      "}"
//
// entry ::=
//    | key_value
//    | block
//
// key_value ::= identifier ":" value [";"]
//
// block ::= identifier "(" params ")" ";"
//         | identifier "(" params ")" "{"
//              entry *
//           "}"
//
// value ::= string | number | identifier
// params ::= value ["," value]*
//
// Note that the semicolons appears to be optional empirically, so we support
// newlines implicitly delimiting the end of a key/value entry.
//
// This was determined empirically but see also the liberty reference manual:
// https://people.eecs.berkeley.edu/~alanmi/publications/other/liberty07_03.pdf

// A key/value entry that can be contained inside of a block.
struct KVEntry {
  std::string key;
  std::string value;
};

struct Block;

// Inside of a block there are either key/value items or sub-blocks.
using BlockEntry = absl::variant<KVEntry, std::unique_ptr<Block>>;

// Represents a hierarchical entity in the cell library description, as shown in
// the grammar above.
struct Block {
  // The "kind" of this block that is given as a leading prefix; e.g. "library",
  // "cell", "pin".
  std::string kind;

  // Values in the parenthesized set; e.g. {"o"} in `pin (o) { ... }`.
  absl::InlinedVector<std::string, 4> args;

  // Data contained within the block; KV pairs and sub-blocks.
  std::vector<BlockEntry> entries;

  // Retrieves sub-blocks contained within this block.
  //
  // If target_kind is provided it is used as a filter (the only blocks returned
  // have subblock->kind == target_kind).
  std::vector<const Block*> GetSubBlocks(
      absl::optional<absl::string_view> target_kind = absl::nullopt) const;

  // Retrieves the first key/value pair in this block that corresponds to
  // target_key.
  const std::string& GetKVOrDie(absl::string_view target_key) const;

  // Counts the number of entries with either the key of "target" for a
  // key/value entry or a kind of "target" for a block entry.
  int64 CountEntries(absl::string_view target) const;

  // Helper used for converting entries contained within the block into strings.
  static std::string EntryToString(const BlockEntry& entry);

  // Converts this block to an AST-like string representation.
  std::string ToString() const;
};

class Parser {
 public:
  // See comment on kind_whitelist_ member below for details.
  explicit Parser(Scanner* scanner,
                  absl::optional<absl::flat_hash_set<std::string>>
                      kind_whitelist = absl::nullopt)
      : scanner_(scanner), kind_whitelist_(std::move(kind_whitelist)) {}

  xabsl::StatusOr<std::unique_ptr<Block>> ParseLibrary() {
    XLS_RETURN_IF_ERROR(DropIdentifierOrError("library"));
    return ParseBlock("library");
  }

 private:
  xabsl::StatusOr<bool> TryDropToken(TokenKind target, Pos* pos = nullptr);
  absl::Status DropTokenOrError(TokenKind kind);
  absl::Status DropIdentifierOrError(absl::string_view target);

  // Pops an identifier token and returns its payload, or errors.
  xabsl::StatusOr<std::string> PopIdentifierOrError();

  // Pops a value token and returns its payload, or errors.
  //
  // If last_pos is provided it is populated with the position of the last value
  // token. (This is useful for checking for newline termination in lieu of
  // semicolons.)
  xabsl::StatusOr<std::string> PopValueOrError(Pos* last_pos = nullptr);

  // Parses all of the entries contained within a block -- includes key/value
  // entries as well as sub-blocks.
  xabsl::StatusOr<std::vector<BlockEntry>> ParseEntries();

  // Parses a comma-delimited sequence of values and returns their payloads.
  xabsl::StatusOr<absl::InlinedVector<std::string, 4>> ParseValues(
      Pos* end_pos = nullptr);

  // Parses a block per the grammar above.
  //
  // If the identifier is provided by the caller it is not scanned out of the
  // token stream.
  xabsl::StatusOr<std::unique_ptr<Block>> ParseBlock(std::string identifier);

  Scanner* scanner_;

  // Optional whitelist of keys (including block kinds) that we're interested in
  // keeping in the result data structure. Non-whitelist block kinds are still
  // parsed properly, but then may have incomplete data (or not be present at
  // all) in the resulting data structure.
  //
  // This is very useful for minimizing memory usage when we're interested in
  // just a subset of particular fields, e.g. as part of a query.
  absl::optional<absl::flat_hash_set<std::string>> kind_whitelist_;
};

}  // namespace cell_lib
}  // namespace netlist
}  // namespace xls

#endif  // XLS_NETLIST_LIB_PARSER_H_
