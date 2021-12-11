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

#ifndef XLS_NETLIST_NETLIST_PARSER_H_
#define XLS_NETLIST_NETLIST_PARSER_H_

#include <sys/types.h>

#include <string>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/ascii.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/types/variant.h"
#include "xls/common/logging/logging.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/common/string_to_int.h"
#include "xls/ir/bits.h"
#include "xls/netlist/netlist.h"
#include "re2/re2.h"

namespace xls {
namespace netlist {
namespace rtl {

// Kinds of tokens the scanner emits.
enum class TokenKind {
  kStartParams,   // #(
  kOpenParen,     // (
  kCloseParen,    // )
  kOpenBracket,   // [
  kCloseBracket,  // ]
  kOpenBrace,     // {
  kCloseBrace,    // }
  kDot,
  kComma,
  kColon,
  kSemicolon,
  kEquals,
  kQuote,
  kName,
  kNumber,
};

// Returns a string representation of "kind" suitable for debugging.
std::string TokenKindToString(TokenKind kind);

// Represents a position in input text.
struct Pos {
  int64_t lineno;
  int64_t colno;

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

  absl::StatusOr<Token> Peek();

  absl::StatusOr<Token> Pop();

  bool AtEof() {
    DropIgnoredChars();
    return index_ >= text_.size();
  }

 private:
  absl::StatusOr<Token> ScanName(char startc, Pos pos, bool is_escaped);
  absl::StatusOr<Token> ScanNumber(char startc, Pos pos);
  absl::StatusOr<Token> PeekInternal();

  // Drops any characters that should not be converted to Tokens, including
  // whitespace, comments, and attributes.
  // Note that we may eventually want to expose attributes to the
  // AbstractParser, but until then it's much simpler to treat attributes like
  // block comments and ignore everything inside of them. This also means that
  // the Scanner will accept attributes that are in invalid positions.
  void DropIgnoredChars();

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
  int64_t index_ = 0;
  int64_t lineno_ = 0;
  int64_t colno_ = 0;
  absl::optional<Token> lookahead_;
};

template <typename EvalT = bool>
class AbstractParser {
 public:
  // Parses a netlist with the given cell library and token scanner.
  // Returns a status on parse error.
  static absl::StatusOr<std::unique_ptr<AbstractNetlist<EvalT>>> ParseNetlist(
      AbstractCellLibrary<EvalT>* cell_library, Scanner* scanner, EvalT zero,
      EvalT one);
  template <typename = std::is_constructible<EvalT, bool>>
  static absl::StatusOr<std::unique_ptr<AbstractNetlist<EvalT>>> ParseNetlist(
      AbstractCellLibrary<EvalT>* cell_library, Scanner* scanner) {
    return ParseNetlist(cell_library, scanner, EvalT{false}, EvalT{true});
  }

 private:
  explicit AbstractParser(AbstractCellLibrary<EvalT>* cell_library,
                          Scanner* scanner, EvalT zero, EvalT one)
      : cell_library_(cell_library),
        scanner_(scanner),
        zero_(zero),
        one_(one) {}

  // Parses a cell instantiation (e.g. in module scope).
  absl::Status ParseInstance(AbstractModule<EvalT>* module,
                             AbstractNetlist<EvalT>& netlist);

  // Parses a cell module name out of the token stream and returns the
  // corresponding AbstractCellLibraryEntry for that module name.
  absl::StatusOr<const AbstractCellLibraryEntry<EvalT>*> ParseCellModule(
      AbstractNetlist<EvalT>& netlist);

  // Parses a wire declaration at the module scope.
  absl::Status ParseNetDecl(AbstractModule<EvalT>* module, NetDeclKind kind);

  struct Range {
    int64_t high;
    int64_t low;
  };

  // Parses an assign declaration at the module scope.
  absl::Status ParseAssignDecl(AbstractModule<EvalT>* module);
  // Parse a single assignment.  Called by ParseAssignDecl()
  absl::Status ParseOneAssignment(AbstractModule<EvalT>* module,
                                  absl::string_view lhs_name,
                                  absl::optional<Range> lhs_range);

  // Attempts to parse a range of the kind [high:low].  It also handles
  // indexing by setting parameter strict to false, by representing the range as
  // [high:high].  For example:
  //   "a" --> no range
  //   "a[1] --> [1:1] (strict == false)
  //   "a[1:0] --> [1:0]
  absl::StatusOr<absl::optional<Range>> ParseOptionalRange(bool strict = true);

  // Parses a module-level statement (e.g. wire decl or cell instantiation).
  absl::Status ParseModuleStatement(AbstractModule<EvalT>* module,
                                    AbstractNetlist<EvalT>& netlist);

  // Parses a module definition (e.g. at the top of the file).
  absl::StatusOr<std::unique_ptr<AbstractModule<EvalT>>> ParseModule(
      AbstractNetlist<EvalT>& netlist);

  // Parses a reference to an already- declared net.
  absl::StatusOr<AbstractNetRef<EvalT>> ParseNetRef(
      AbstractModule<EvalT>* module);

  // Pops a name token and returns its contents or gives an error status if a
  // name token is not immediately present in the stream.
  absl::StatusOr<std::string> PopNameOrError();

  // Pops a name token and returns its value or gives an error status if a
  // number token is not immediately present in the stream.
  absl::StatusOr<int64_t> PopNumberOrError();

  // Pops either a name or number token or returns an error.
  absl::StatusOr<absl::variant<std::string, int64_t>> PopNameOrNumberOrError();

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
  absl::StatusOr<std::vector<std::string>> PopParenNameList();

  // Cell library definitions are resolved against.
  AbstractCellLibrary<EvalT>* cell_library_;

  // Set of (already-parsed) Modules that may be present in the AbstractModule
  // currently being processed as AbstractCell-type references.
  absl::flat_hash_map<std::string, AbstractModule<EvalT>> modules_;

  // Scanner used for scanning out tokens (in a stream sequence).
  Scanner* scanner_;

  // Values representing zero/false and one/true in the EvalT type.
  EvalT zero_;
  EvalT one_;
};

using Parser = AbstractParser<>;

template <typename EvalT>
absl::StatusOr<std::string> AbstractParser<EvalT>::PopNameOrError() {
  XLS_ASSIGN_OR_RETURN(Token token, scanner_->Pop());
  if (token.kind == TokenKind::kName) {
    return token.value;
  }
  return absl::InvalidArgumentError("Expected name token; got: " +
                                    token.ToString());
}

template <typename EvalT>
absl::StatusOr<int64_t> AbstractParser<EvalT>::PopNumberOrError() {
  // We're assuming we won't see > 64b values. Fine for now, at least.
  XLS_ASSIGN_OR_RETURN(Token token, scanner_->Pop());
  if (token.kind == TokenKind::kNumber) {
    // Check for the big version first.
    std::string width_string, signed_string, base_string, value_string;
    // Precompute the regex matcher for Verilog number literals.
    static LazyRE2 number_re_ = {R"(([0-9]+)'([Ss]?)([bodhBODH])([0-9a-f]+))"};
    if (RE2::FullMatch(token.value, *number_re_, &width_string, &signed_string,
                       &base_string, &value_string)) {
      int64_t width;
      XLS_RET_CHECK(
          absl::SimpleAtoi(width_string, reinterpret_cast<int64_t*>(&width)))
          << "Unable to parse number width: " << width_string;
      int base;
      if (base_string == "b" || base_string == "B") {
        base = 2;
      } else if (base_string == "o" || base_string == "O") {
        base = 8;
      } else if (base_string == "d" || base_string == "D") {
        base = 10;
      } else if (base_string == "h" || base_string == "H") {
        base = 16;
      } else {
        return absl::InvalidArgumentError(
            absl::StrCat("Invalid numeric base: ", base_string));
      }

      XLS_ASSIGN_OR_RETURN(uint64_t temp, StrTo64Base(value_string, base));
      if (signed_string.empty()) {
        return static_cast<int64_t>(temp);
      }

      // If the number is actually signed, then throw it into a Bits for sign
      // conversion.
      XLS_ASSIGN_OR_RETURN(Bits bits, UBitsWithStatus(temp, width));
      return bits.ToInt64();
    }

    int64_t result;
    if (!absl::SimpleAtoi(token.value, &result)) {
      return absl::InternalError(
          "Number token's value cannot be parsed as an int64_t: " +
          token.value);
    }
    return result;
  }
  return absl::InvalidArgumentError("Expected number token; got: " +
                                    token.ToString());
}

template <typename EvalT>
absl::StatusOr<absl::variant<std::string, int64_t>>
AbstractParser<EvalT>::PopNameOrNumberOrError() {
  TokenKind kind = scanner_->Peek()->kind;
  if (kind == TokenKind::kName) {
    XLS_ASSIGN_OR_RETURN(Token token, scanner_->Pop());
    return token.value;
  } else if (kind == TokenKind::kNumber) {
    return PopNumberOrError();
  }
  return absl::InvalidArgumentError(absl::StrCat(
      "Expected name or number token; got: ", static_cast<int>(kind)));
}

template <typename EvalT>
absl::Status AbstractParser<EvalT>::DropTokenOrError(TokenKind target) {
  XLS_ASSIGN_OR_RETURN(Token token, scanner_->Pop());
  if (token.kind == target) {
    return absl::OkStatus();
  }
  return absl::UnimplementedError(absl::StrFormat(
      "Want token %s; got %s.", TokenKindToString(target), token.ToString()));
}

template <typename EvalT>
absl::StatusOr<std::vector<std::string>>
AbstractParser<EvalT>::PopParenNameList() {
  XLS_RETURN_IF_ERROR(DropTokenOrError(TokenKind::kOpenParen));
  std::vector<std::string> results;
  bool must_end = false;
  while (true) {
    if (TryDropToken(TokenKind::kCloseParen)) {
      break;
    }
    if (must_end) {
      XLS_RETURN_IF_ERROR(DropTokenOrError(TokenKind::kCloseParen));
      break;
    }
    XLS_ASSIGN_OR_RETURN(std::string name, PopNameOrError());
    results.push_back(name);
    must_end = !TryDropToken(TokenKind::kComma);
  }
  return results;
}

template <typename EvalT>
absl::Status AbstractParser<EvalT>::DropKeywordOrError(
    absl::string_view target) {
  XLS_ASSIGN_OR_RETURN(Token token, scanner_->Pop());
  if (token.kind == TokenKind::kName && token.value == target) {
    return absl::OkStatus();
  }
  return absl::InvalidArgumentError(
      absl::StrFormat("Want keyword '%s', got: %s", target, token.ToString()));
}

template <typename EvalT>
absl::StatusOr<const AbstractCellLibraryEntry<EvalT>*>
AbstractParser<EvalT>::ParseCellModule(AbstractNetlist<EvalT>& netlist) {
  XLS_ASSIGN_OR_RETURN(std::string name, PopNameOrError());
  auto status_or_module = netlist.GetModule(name);
  if (status_or_module.ok()) {
    return status_or_module.value()->AsCellLibraryEntry();
  }
  if (name == "SB_LUT4") {
    XLS_RETURN_IF_ERROR(DropTokenOrError(TokenKind::kStartParams));
    XLS_RETURN_IF_ERROR(DropTokenOrError(TokenKind::kDot));
    XLS_ASSIGN_OR_RETURN(std::string param_name, PopNameOrError());
    if (param_name != "LUT_INIT") {
      return absl::InvalidArgumentError(
          "Expected a single .LUT_INIT named parameter, got: " + param_name);
    }
    XLS_RETURN_IF_ERROR(DropTokenOrError(TokenKind::kOpenParen));
    XLS_ASSIGN_OR_RETURN(int64_t lut_mask, PopNumberOrError());
    XLS_RETURN_IF_ERROR(DropTokenOrError(TokenKind::kCloseParen));
    XLS_RETURN_IF_ERROR(DropTokenOrError(TokenKind::kCloseParen));
    return netlist.GetOrCreateLut4CellEntry(lut_mask, zero_, one_);
  }
  return cell_library_->GetEntry(name);
}

template <typename EvalT>
absl::StatusOr<AbstractNetRef<EvalT>> AbstractParser<EvalT>::ParseNetRef(
    AbstractModule<EvalT>* module) {
  using TokenT = absl::variant<std::string, int64_t>;
  XLS_ASSIGN_OR_RETURN(TokenT token, PopNameOrNumberOrError());
  if (absl::holds_alternative<int64_t>(token)) {
    int64_t value = absl::get<int64_t>(token);
    return module->AddOrResolveNumber(value);
  }

  std::string name = absl::get<std::string>(token);
  if (TryDropToken(TokenKind::kOpenBracket)) {
    XLS_ASSIGN_OR_RETURN(int64_t index, PopNumberOrError());
    XLS_RETURN_IF_ERROR(DropTokenOrError(TokenKind::kCloseBracket));
    absl::StrAppend(&name, "[", index, "]");
  }
  return module->ResolveNet(name);
}

template <typename EvalT>
absl::Status AbstractParser<EvalT>::ParseInstance(
    AbstractModule<EvalT>* module, AbstractNetlist<EvalT>& netlist) {
  XLS_ASSIGN_OR_RETURN(Token peek, scanner_->Peek());
  const Pos pos = peek.pos;

  XLS_ASSIGN_OR_RETURN(const AbstractCellLibraryEntry<EvalT>* cle,
                       ParseCellModule(netlist));
  XLS_ASSIGN_OR_RETURN(std::string name, PopNameOrError());
  XLS_RETURN_IF_ERROR(DropTokenOrError(TokenKind::kOpenParen));
  // LRM 23.3.2 Calls these "named parameter assignments".
  absl::flat_hash_map<std::string, AbstractNetRef<EvalT>>
      named_parameter_assignments;
  while (true) {
    XLS_RETURN_IF_ERROR(DropTokenOrError(TokenKind::kDot));
    XLS_ASSIGN_OR_RETURN(std::string pin_name, PopNameOrError());
    XLS_RETURN_IF_ERROR(DropTokenOrError(TokenKind::kOpenParen));
    XLS_ASSIGN_OR_RETURN(AbstractNetRef<EvalT> net, ParseNetRef(module));
    XLS_RETURN_IF_ERROR(DropTokenOrError(TokenKind::kCloseParen));
    XLS_VLOG(3) << "Adding named parameter assignment: " << pin_name;
    bool is_new = named_parameter_assignments.insert({pin_name, net}).second;
    if (!is_new) {
      return absl::InvalidArgumentError("Duplicate port seen: " + pin_name);
    }
    if (!TryDropToken(TokenKind::kComma)) {
      break;
    }
  }
  absl::optional<AbstractNetRef<EvalT>> clock;
  if (cle->clock_name().has_value()) {
    auto it = named_parameter_assignments.find(cle->clock_name().value());
    if (it == named_parameter_assignments.end()) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "Cell %s named %s requires a clock connection %s but none was found.",
          cle->name(), name, cle->clock_name().value()));
    }
    clock = it->second;
    named_parameter_assignments.erase(it);
  }
  XLS_ASSIGN_OR_RETURN(
      AbstractCell<EvalT> cell,
      AbstractCell<EvalT>::Create(cle, name, named_parameter_assignments, clock,
                                  module->GetDummyRef()),
      _ << " @ " << pos.ToHumanString());
  XLS_ASSIGN_OR_RETURN(AbstractCell<EvalT> * cell_ptr,
                       module->AddCell(std::move(cell)));
  absl::flat_hash_set<AbstractNetRef<EvalT>> connected_wires;
  for (auto& item : named_parameter_assignments) {
    if (connected_wires.contains(item.second)) {
      continue;
    }
    item.second->NoteConnectedCell(cell_ptr);
    connected_wires.insert(item.second);
  }
  XLS_RETURN_IF_ERROR(DropTokenOrError(TokenKind::kCloseParen));
  XLS_RETURN_IF_ERROR(DropTokenOrError(TokenKind::kSemicolon));
  return absl::OkStatus();
}

template <typename EvalT>
bool AbstractParser<EvalT>::TryDropToken(TokenKind target) {
  if (scanner_->AtEof()) {
    return false;
  }
  if (scanner_->Peek().value().kind == target) {
    XLS_CHECK_OK(scanner_->Pop().status());
    return true;
  }
  return false;
}

template <typename EvalT>
bool AbstractParser<EvalT>::TryDropKeyword(absl::string_view target) {
  if (scanner_->AtEof()) {
    return false;
  }
  Token peek = scanner_->Peek().value();
  if (peek.kind == TokenKind::kName && peek.value == target) {
    XLS_CHECK_OK(scanner_->Pop().status());
    return true;
  }
  return false;
}

template <typename EvalT>
absl::StatusOr<absl::optional<typename AbstractParser<EvalT>::Range>>
AbstractParser<EvalT>::ParseOptionalRange(bool strict) {
  absl::optional<Range> range;
  if (TryDropToken(TokenKind::kOpenBracket)) {
    XLS_ASSIGN_OR_RETURN(int64_t high, PopNumberOrError());
    int64_t low = high;
    if (TryDropToken(TokenKind::kColon)) {
      XLS_ASSIGN_OR_RETURN(low, PopNumberOrError());
      if (high < low) {
        return absl::InvalidArgumentError(
            absl::StrFormat("Expected net range to be [high:low] with low <= "
                            "high, got low: %d; high: %d",
                            low, high));
      }
    } else if (strict) {
      return absl::InvalidArgumentError(
          absl::StrFormat("Expecting net range, got a subscript instead"));
    }
    XLS_RETURN_IF_ERROR(DropTokenOrError(TokenKind::kCloseBracket));
    range = {high, low};
  }
  return range;
}

template <typename EvalT>
absl::Status AbstractParser<EvalT>::ParseNetDecl(AbstractModule<EvalT>* module,
                                                 NetDeclKind kind) {
  XLS_ASSIGN_OR_RETURN(auto range, ParseOptionalRange());
  std::vector<std::string> names;
  do {
    XLS_ASSIGN_OR_RETURN(std::string name, PopNameOrError());
    names.push_back(name);
  } while (TryDropToken(TokenKind::kComma));

  XLS_RETURN_IF_ERROR(DropTokenOrError(TokenKind::kSemicolon));

  if (names.size() > 1 && range.has_value()) {
    // Note: we could support this but not sure if the netlist ever contains
    // such a construct.
    return absl::UnimplementedError(
        "Multiple declarations for a ranged net is not yet supported.");
  }

  for (const std::string& name : names) {
    if (kind == NetDeclKind::kInput || kind == NetDeclKind::kOutput) {
      int64_t width = 1;
      if (range.has_value()) {
        width = range->high - range->low + 1;
      }
      XLS_RETURN_IF_ERROR(
          module->DeclarePort(name, width, kind == NetDeclKind::kOutput));
    }
    if (range.has_value()) {
      for (int64_t i = range->low; i <= range->high; ++i) {
        XLS_RETURN_IF_ERROR(
            module->AddNetDecl(kind, absl::StrFormat("%s[%d]", name, i)));
      }
    } else {
      XLS_RETURN_IF_ERROR(module->AddNetDecl(kind, name));
    }
  }
  return absl::OkStatus();
}

template <typename EvalT>
absl::Status AbstractParser<EvalT>::ParseOneAssignment(
    AbstractModule<EvalT>* module, absl::string_view lhs_name,
    absl::optional<Range> lhs_range) {
  // Extract the range from the lhs wire.  The high and low ends are identical
  // because the optional range might be an index dereference.
  int64_t lhs_high = 0, lhs_low = 0;
  if (lhs_range.has_value()) {
    lhs_high = lhs_range.value().high;
    lhs_low = lhs_range.value().low;
  }
  XLS_RET_CHECK(lhs_high >= lhs_low);

  using TokenT = absl::variant<std::string, int64_t>;
  XLS_ASSIGN_OR_RETURN(TokenT token, PopNameOrNumberOrError());
  if (absl::holds_alternative<int64_t>(token)) {
    int64_t rhs_value = absl::get<int64_t>(token);
    // We'll be right-shifting below, make sure that sign extensions do not
    // trip us up.
    if (rhs_value < 0) {
      return absl::UnimplementedError(
          "Negative number literals are not supported in assign statements.");
    }

    // Start converting the value to the input wires zero_ or one_, and
    // assign each input bit to the corresponding NetDecl, starting with the
    // low end of the range.  If we run out of wires while converting the
    // number, error out.

    while (lhs_low <= lhs_high) {
      bool bit = rhs_value & 1;
      if (lhs_range.has_value()) {
        XLS_RETURN_IF_ERROR(module->AddAssignDecl(
            absl::StrFormat("%s[%d]", lhs_name, lhs_low), bit));
      } else {
        // The loop will execute only once.
        XLS_RETURN_IF_ERROR(module->AddAssignDecl(lhs_name, bit));
      }
      lhs_low++;
      rhs_value >>= 1;
    }
    if (rhs_value != 0) {
      return absl::InvalidArgumentError(
          absl::StrFormat("Number literal is too wide for %s.", lhs_name));
    }

  } else {
    std::string rhs_name = absl::get<std::string>(token);
    XLS_ASSIGN_OR_RETURN(auto rhs_range, ParseOptionalRange(false));

    // Extract the range from the rhs wire.
    int64_t rhs_high = 0, rhs_low = 0;
    if (rhs_range.has_value()) {
      rhs_high = rhs_range.value().high;
      rhs_low = rhs_range.value().low;
    }
    XLS_CHECK(rhs_high >= rhs_low);

    // The two ranges must be the same width.
    if (rhs_high - rhs_low != lhs_high - lhs_low) {
      return absl::InvalidArgumentError(
          absl::StrFormat("Mismatched bit widths: left-hand side is %lld, "
                          "right-hand side is %lld.",
                          lhs_high - lhs_low + 1, rhs_high - rhs_low + 1));
    }

    // Start mapping the rhs wires to the lhs ones.
    while (lhs_low <= lhs_high) {
      std::string lhs_wire_name;
      if (lhs_range.has_value()) {
        lhs_wire_name = absl::StrFormat("%s[%d]", lhs_name, lhs_low);
      } else {
        lhs_wire_name = lhs_name;
      }
      std::string rhs_wire_name;
      if (rhs_range.has_value()) {
        rhs_wire_name = absl::StrFormat("%s[%d]", rhs_name, rhs_low);
      } else {
        rhs_wire_name = rhs_name;
      }
      XLS_RETURN_IF_ERROR(module->AddAssignDecl(lhs_wire_name, rhs_wire_name));
      lhs_low++;
      rhs_low++;
    }
    XLS_CHECK(rhs_low >= rhs_high);
  }

  return absl::OkStatus();
}

template <typename EvalT>
absl::Status AbstractParser<EvalT>::ParseAssignDecl(
    AbstractModule<EvalT>* module) {
  // Parse assign statements of the following format:
  //
  // assign idA = idB;
  // assign { idA0, idA1, ... } = { idB0, idB1, ... }
  //
  // Each identifier can be a literal, a single wire, or a wire with a
  // subscript, or a wire with a subscript range, e.g. "8'h00", "a", or "a[0]",
  // or "a[7:0]".
  //
  // The identifiers on the LHS and the RHS must be the same width, e.g.
  //
  // assign a = 1'b0
  // assign a[7:0] = 8'hff
  // assign a = b
  // assign { a, b[1], c[7:0] }  = { d, e[5], f[15:8] }
  //
  // Note: we do not handle all possible kinds of assign syntax.  For example,
  // the line "assign {a,b} = 2'h0;" is legal.  We error out in this case rather
  // than doing the wrong thing.  Support can be added in the future, if needed.

  if (TryDropToken(TokenKind::kOpenBrace)) {
    std::vector<std::pair<std::string, absl::optional<Range>>> lhs;
    // Parse the left-hand side.
    do {
      XLS_ASSIGN_OR_RETURN(std::string name, PopNameOrError());
      XLS_ASSIGN_OR_RETURN(auto range, ParseOptionalRange(false));
      lhs.push_back({name, range});
    } while (TryDropToken(TokenKind::kComma));
    XLS_RETURN_IF_ERROR(DropTokenOrError(TokenKind::kCloseBrace));
    XLS_RETURN_IF_ERROR(DropTokenOrError(TokenKind::kEquals));

    // Parse the right-hand side.  While parsing, iterate over the lhs elements
    // we collected, verify that widths match, then break them up into
    // individual NetDecl instances, and save the associationss.  The right-hand
    // side could map an integer to a wire range, in which case we break up the
    // integer bitwise and assign the values to the lhs wires.

    XLS_RETURN_IF_ERROR(DropTokenOrError(TokenKind::kOpenBrace));
    auto left = lhs.begin();
    do {
      absl::string_view lhs_name = left->first;
      absl::optional<Range> lhs_range = left->second;
      XLS_RETURN_IF_ERROR(ParseOneAssignment(module, lhs_name, lhs_range));
      left++;
    } while (TryDropToken(TokenKind::kComma));
    XLS_RETURN_IF_ERROR(DropTokenOrError(TokenKind::kCloseBrace));
  } else {
    // Parse the left-hand side.
    XLS_ASSIGN_OR_RETURN(std::string lhs_name, PopNameOrError());
    XLS_ASSIGN_OR_RETURN(auto lhs_range, ParseOptionalRange(false));
    // Parse the right-hand side.
    XLS_RETURN_IF_ERROR(DropTokenOrError(TokenKind::kEquals));
    XLS_RETURN_IF_ERROR(ParseOneAssignment(module, lhs_name, lhs_range));
  }
  XLS_RETURN_IF_ERROR(DropTokenOrError(TokenKind::kSemicolon));

  return absl::OkStatus();
}

template <typename EvalT>
absl::Status AbstractParser<EvalT>::ParseModuleStatement(
    AbstractModule<EvalT>* module, AbstractNetlist<EvalT>& netlist) {
  if (TryDropKeyword("input")) {
    return ParseNetDecl(module, NetDeclKind::kInput);
  }
  if (TryDropKeyword("output")) {
    return ParseNetDecl(module, NetDeclKind::kOutput);
  }
  if (TryDropKeyword("wire")) {
    return ParseNetDecl(module, NetDeclKind::kWire);
  }
  if (TryDropKeyword("assign")) {
    return ParseAssignDecl(module);
  }
  return ParseInstance(module, netlist);
}

template <typename EvalT>
absl::StatusOr<std::unique_ptr<AbstractModule<EvalT>>>
AbstractParser<EvalT>::ParseModule(AbstractNetlist<EvalT>& netlist) {
  XLS_RETURN_IF_ERROR(DropKeywordOrError("module"));
  XLS_ASSIGN_OR_RETURN(std::string module_name, PopNameOrError());
  XLS_ASSIGN_OR_RETURN(std::vector<std::string> module_ports,
                       PopParenNameList());
  XLS_RETURN_IF_ERROR(DropTokenOrError(TokenKind::kSemicolon));

  auto module = std::make_unique<AbstractModule<EvalT>>(module_name);
  module->DeclarePortsOrder(module_ports);

  while (true) {
    if (TryDropKeyword("endmodule")) {
      break;
    }
    XLS_RETURN_IF_ERROR(ParseModuleStatement(module.get(), netlist));
  }
  return module;
}

template <typename EvalT>
absl::StatusOr<std::unique_ptr<AbstractNetlist<EvalT>>>
AbstractParser<EvalT>::ParseNetlist(AbstractCellLibrary<EvalT>* cell_library,
                                    Scanner* scanner, EvalT zero, EvalT one) {
  auto netlist = std::make_unique<AbstractNetlist<EvalT>>();
  AbstractParser<EvalT> p(cell_library, scanner, zero, one);
  while (!scanner->AtEof()) {
    XLS_ASSIGN_OR_RETURN(std::unique_ptr<AbstractModule<EvalT>> module,
                         p.ParseModule(*netlist));
    netlist->AddModule(std::move(module));
  }
  return std::move(netlist);
}

}  // namespace rtl
}  // namespace netlist
}  // namespace xls

#endif  // XLS_NETLIST_NETLIST_PARSER_H_
