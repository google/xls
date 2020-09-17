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

#ifndef XLS_DSLX_TOKEN_PARSER_H_
#define XLS_DSLX_TOKEN_PARSER_H_

#include "xls/common/logging/logging.h"
#include "xls/dslx/cpp_bindings.h"
#include "xls/dslx/cpp_scanner.h"

namespace xls::dslx {

class TokenParser {
 public:
  explicit TokenParser(Scanner* scanner) : scanner_(XLS_DIE_IF_NULL(scanner)) {}

 protected:
  // Returns the current position of the parser in the text, via its current
  // position the token stream.
  Pos GetPos() const {
    if (lookahead_.has_value()) {
      return lookahead_->span().start();
    }
    return scanner_->GetPos();
  }

  bool AtEof() { return scanner_->AtEof(); }

  absl::StatusOr<absl::optional<Token>> TryPopIdentifierToken(
      absl::string_view target) {
    XLS_ASSIGN_OR_RETURN(const Token* tok, PeekToken());
    if (tok->IsIdentifier(target)) {
      return PopTokenOrDie();
    }
    return absl::nullopt;
  }

  absl::StatusOr<absl::optional<Token>> TryPopToken(TokenKind target) {
    XLS_ASSIGN_OR_RETURN(const Token* peek, PeekToken());
    if (peek->kind() == target) {
      return PopTokenOrDie();
    }
    return absl::nullopt;
  }

  absl::StatusOr<absl::optional<Token>> TryPopKeyword(Keyword target) {
    XLS_ASSIGN_OR_RETURN(const Token* peek, PeekToken());
    if (peek->IsKeyword(target)) {
      return PopTokenOrDie();
    }
    return absl::nullopt;
  }

  absl::StatusOr<bool> TryDropToken(TokenKind target) {
    XLS_ASSIGN_OR_RETURN(const Token* peek, PeekToken());
    if (peek->kind() == target) {
      DropTokenOrDie();
      return true;
    }
    return false;
  }

  absl::StatusOr<bool> TryDropIdentifierToken(absl::string_view target) {
    XLS_ASSIGN_OR_RETURN(absl::optional<Token> maybe_tok,
                         TryPopIdentifierToken(target));
    return maybe_tok != absl::nullopt;
  }

  absl::StatusOr<bool> TryDropKeyword(Keyword target) {
    XLS_ASSIGN_OR_RETURN(const Token* peek, PeekToken());
    if (peek->IsKeyword(target)) {
      DropTokenOrDie();
      return true;
    }
    return false;
  }

  absl::StatusOr<const Token*> PeekToken() {
    if (lookahead_.has_value()) {
      return &*lookahead_;
    }

    XLS_ASSIGN_OR_RETURN(lookahead_, scanner_->Pop());
    XLS_CHECK(lookahead_.has_value());
    return &*lookahead_;
  }

  // Returns a token that has been popped destructively from the token stream.
  absl::StatusOr<Token> PopToken() {
    XLS_ASSIGN_OR_RETURN(const Token* unused, PeekToken());
    (void)unused;
    XLS_CHECK(lookahead_.has_value());
    Token tok = std::move(lookahead_).value();
    lookahead_ = absl::nullopt;
    XLS_CHECK(!lookahead_.has_value());
    return tok;
  }

  absl::StatusOr<std::string> PopIdentifierOrError() {
    XLS_ASSIGN_OR_RETURN(Token tok, PopTokenOrError(TokenKind::kIdentifier));
    return *tok.GetValue();
  }

  // For use only when the caller knows there is lookahead present (in which
  // case we don't need to check for errors in potentially scanning the next
  // token).
  Token PopTokenOrDie() {
    XLS_CHECK(lookahead_.has_value());
    return PopToken().value();
  }

  // Wraps PopToken() to signify popping a token without needing the value.
  absl::Status DropToken() { return PopToken().status(); }

  void DropTokenOrDie() { XLS_CHECK_OK(DropToken()); }

  absl::StatusOr<bool> PeekTokenIs(TokenKind target) {
    XLS_ASSIGN_OR_RETURN(const Token* tok, PeekToken());
    return tok->kind() == target;
  }
  absl::StatusOr<bool> PeekTokenIs(Keyword target) {
    XLS_ASSIGN_OR_RETURN(const Token* tok, PeekToken());
    return tok->IsKeyword(target);
  }
  absl::StatusOr<bool> PeekTokenIn(absl::Span<TokenKind const> targets) {
    XLS_ASSIGN_OR_RETURN(const Token* tok, PeekToken());
    for (TokenKind target : targets) {
      if (target == tok->kind()) {
        return true;
      }
    }
    return false;
  }
  absl::StatusOr<bool> PeekKeywordIn(absl::Span<Keyword const> targets) {
    XLS_ASSIGN_OR_RETURN(const Token* tok, PeekToken());
    if (tok->kind() != TokenKind::kKeyword) {
      return false;
    }
    for (Keyword target : targets) {
      if (target == tok->GetKeyword()) {
        return true;
      }
    }
    return false;
  }

  absl::StatusOr<Token> PopTokenOrError(TokenKind target,
                                        const Token* start = nullptr,
                                        absl::string_view context = "") {
    XLS_ASSIGN_OR_RETURN(const Token* tok, PeekToken());
    if (tok->kind() == target) {
      return PopToken();
    }
    std::string msg;
    if (start == nullptr) {
      msg = absl::StrFormat("Expected '%s', got '%s'",
                            TokenKindToString(target), tok->ToErrorString());
    } else {
      msg = absl::StrFormat(
          "Expected '%s' for construct starting with '%s' @ %s, got '%s'",
          TokenKindToString(target), start->ToErrorString(),
          start->span().ToString(), tok->ToErrorString());
    }
    if (!context.empty()) {
      msg = absl::StrCat(msg, ": ", context);
    }
    return ParseError(tok->span(), msg);
  }

  // Wrapper around PopTokenOrError that does not return the token. Helps
  // signify that the intent was to drop the token in the caller code vs
  // 'forgetting' to do something with the popped token.
  absl::Status DropTokenOrError(TokenKind target, const Token* start = nullptr,
                                absl::string_view context = "") {
    XLS_ASSIGN_OR_RETURN(Token token, PopTokenOrError(target, start, context));
    return absl::OkStatus();
  }

  absl::StatusOr<Token> PopKeywordOrError(Keyword keyword,
                                          absl::string_view context = "") {
    XLS_ASSIGN_OR_RETURN(Token tok, PopToken());
    if (tok.IsKeyword(keyword)) {
      return std::move(tok);
    }
    std::string msg =
        absl::StrFormat("Expected keyword '%s', got %s'",
                        KeywordToString(keyword), tok.ToErrorString());
    if (!context.empty()) {
      msg = absl::StrCat(msg, ": ", context);
    }
    return ParseError(tok.span(), msg);
  }

  absl::Status DropKeywordOrError(Keyword target) {
    XLS_ASSIGN_OR_RETURN(Token token, PopKeywordOrError(target));
    return absl::OkStatus();
  }

 private:
  Scanner* scanner_;
  absl::optional<Token> lookahead_;
};

}  // namespace xls::dslx

#endif  // XLS_DSLX_TOKEN_PARSER_H_
