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

#ifndef XLS_NETLIST_NETLIST_PARSER_H_
#define XLS_NETLIST_NETLIST_PARSER_H_

#include "absl/status/status.h"
#include "xls/common/status/statusor.h"
#include "xls/netlist/netlist.h"

namespace xls {
namespace netlist {
namespace rtl {

// Kinds of tokens the scanner emits.
enum class TokenKind {
  kOpenParen,     // (
  kCloseParen,    // )
  kOpenBracket,   // [
  kCloseBracket,  // ]
  kDot,
  kComma,
  kColon,
  kSemicolon,
  kName,
  kNumber,
};

// Returns a string representation of "kind" suitable for debugging.
std::string TokenKindToString(TokenKind kind);

// Represents a position in input text.
struct Pos {
  int64 lineno;
  int64 colno;

  std::string ToHumanString() const;
};

// Represents a scanned token (that comes from scanning a character stream).
struct Token {
  TokenKind kind;
  Pos pos;
  std::string value;

  std::string ToString() const;
};

// Token scanner for netlist files.
class Scanner {
 public:
  explicit Scanner(absl::string_view text) : text_(text) {}

  xabsl::StatusOr<Token> Peek();

  xabsl::StatusOr<Token> Pop();

  bool AtEof() {
    DropCommentsAndWhitespace();
    return index_ >= text_.size();
  }

 private:
  xabsl::StatusOr<Token> ScanName(char startc, Pos pos, bool is_escaped);
  xabsl::StatusOr<Token> ScanNumber(char startc, Pos pos);
  xabsl::StatusOr<Token> PeekInternal();

  void DropCommentsAndWhitespace();

  char PeekCharOrDie() const;
  char PeekChar2OrDie() const;
  char PopCharOrDie();
  void DropCharOrDie() { (void)PopCharOrDie(); }
  Pos GetPos() const { return Pos{lineno_, colno_}; }

  // Internal version of EOF checking that doesn't attempt to discard the
  // comments/whitespace as the public AtEof() does above -- this simply checks
  // whether the character stream index has reached the end of the text.
  bool AtEofInternal() const { return index_ >= text_.size(); }

  absl::string_view text_;
  int64 index_ = 0;
  int64 lineno_ = 0;
  int64 colno_ = 0;
  absl::optional<Token> lookahead_;
};

class Parser {
 public:
  // Parses a netlist with the given cell library and token scanner.
  // Returns a status on parse error.
  static xabsl::StatusOr<std::unique_ptr<Netlist>> ParseNetlist(
      CellLibrary* cell_library, Scanner* scanner);

 private:
  explicit Parser(CellLibrary* cell_library, Scanner* scanner)
      : cell_library_(cell_library), scanner_(scanner) {}

  // Parses a cell instantiation (e.g. in module scope).
  absl::Status ParseInstance(Module* module, Netlist& netlist);

  // Parses a cell module name out of the token stream and returns the
  // corresponding CellLibraryEntry for that module name.
  xabsl::StatusOr<const CellLibraryEntry*> ParseCellModule(Netlist& netlist);

  // Parses a wire declaration at the module scope.
  absl::Status ParseNetDecl(Module* module, NetDeclKind kind);

  // Parses a module-level statement (e.g. wire decl or cell instantiation).
  absl::Status ParseModuleStatement(Module* module, Netlist& netlist);

  // Parses a module definition (e.g. at the top of the file).
  xabsl::StatusOr<std::unique_ptr<Module>> ParseModule(Netlist& netlist);

  // Parses a reference to an already- declared net.
  xabsl::StatusOr<NetRef> ParseNetRef(Module* module);

  // Pops a name token and returns its contents or gives an error status if a
  // name token is not immediately present in the stream.
  xabsl::StatusOr<std::string> PopNameOrError();

  // Pops a name token and returns its value or gives an error status if a
  // number token is not immediately present in the stream.
  xabsl::StatusOr<int64> PopNumberOrError();

  // Pops either a name or number token or returns an error.
  xabsl::StatusOr<absl::variant<std::string, int64>> PopNameOrNumberOrError();

  // Drops a token of kind target from the head of the stream or gives an error
  // status.
  absl::Status DropTokenOrError(TokenKind target);

  // Drops a keyword token from the head of the stream or gives an error status.
  absl::Status DropKeywordOrError(absl::string_view target);

  // Attempts to drop a token of the target kind, or returns false if that
  // target token kind is not at the head of the token stream.
  bool TryDropToken(TokenKind target);

  // Attempts to drop a keyword token with the value "target" from the head of
  // the token stream, or returns false if it cannot.
  bool TryDropKeyword(absl::string_view target);

  // Pops a parenthesized name list from the token stream and returns it as a
  // vector of those names.
  xabsl::StatusOr<std::vector<std::string>> PopParenNameList();

  // Cell library definitions are resolved against.
  CellLibrary* cell_library_;

  // Set of (already-parsed) Modules that may be present in the Module currently
  // being processed as Cell-type references.
  absl::flat_hash_map<std::string, Module> modules_;

  // Scanner used for scanning out tokens (in a stream sequence).
  Scanner* scanner_;
};

}  // namespace rtl
}  // namespace netlist
}  // namespace xls

#endif  // XLS_NETLIST_NETLIST_PARSER_H_
