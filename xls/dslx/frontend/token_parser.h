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

#ifndef XLS_DSLX_FRONTEND_TOKEN_PARSER_H_
#define XLS_DSLX_FRONTEND_TOKEN_PARSER_H_

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/log/die_if_null.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xls/common/status/status_macros.h"
#include "xls/dslx/frontend/pos.h"
#include "xls/dslx/frontend/scanner.h"
#include "xls/dslx/frontend/token.h"

namespace xls::dslx {

class TokenParser {
 public:
  explicit TokenParser(Scanner* scanner)
      : scanner_(ABSL_DIE_IF_NULL(scanner)), index_(0) {}

 protected:
  // Currently just a plain integer, representing the index into the token
  // stream, but this could be swapped out for a different representation as
  // designs get larger.
  using ScannerCheckpoint = int64_t;

  // Returns the current location in the token stream. Used when "backtracking"
  // from a bad production in the parser.
  ScannerCheckpoint SaveScannerCheckpoint() const { return index_; }

  // Resets the current location in the token stream to that indicated.
  void RestoreScannerCheckpoint(ScannerCheckpoint checkpoint) {
    index_ = checkpoint;
  }

  // Returns the current position of the parser in the text, via its current
  // position the token stream.
  Pos GetPos() const {
    if (index_ < tokens_.size()) {
      return tokens_[index_]->span().start();
    }
    return scanner_->GetPos();
  }

  bool AtEof() { return scanner_->AtEof(); }

  // Attempts to pop an identifier-kind token from the head of the token stream.
  //
  // If the token at the head of the stream is not an identifier, returns
  // nullopt.
  absl::StatusOr<std::optional<Token>> TryPopIdentifierToken(
      std::string_view target) {
    XLS_ASSIGN_OR_RETURN(const Token* tok, PeekToken());
    if (tok->IsIdentifier(target)) {
      return PopTokenOrDie();
    }
    return std::nullopt;
  }

  absl::StatusOr<std::optional<Token>> TryPopToken(TokenKind target) {
    XLS_ASSIGN_OR_RETURN(const Token* peek, PeekToken());
    if (peek->kind() == target) {
      return PopTokenOrDie();
    }
    return std::nullopt;
  }

  absl::StatusOr<std::optional<Token>> TryPopKeyword(Keyword target,
                                                     Pos* pos = nullptr) {
    XLS_ASSIGN_OR_RETURN(const Token* peek, PeekToken());
    if (peek->IsKeyword(target)) {
      if (pos != nullptr) {
        *pos = peek->span().limit();
      }
      return PopTokenOrDie();
    }
    return std::nullopt;
  }

  absl::StatusOr<bool> TryDropToken(TokenKind target, Pos* pos = nullptr) {
    XLS_ASSIGN_OR_RETURN(const Token* peek, PeekToken());
    if (peek->kind() == target) {
      if (pos != nullptr) {
        *pos = peek->span().limit();
      }
      DropTokenOrDie();
      return true;
    }
    return false;
  }

  absl::StatusOr<bool> TryDropIdentifierToken(std::string_view target) {
    XLS_ASSIGN_OR_RETURN(std::optional<Token> maybe_tok,
                         TryPopIdentifierToken(target));
    return maybe_tok != std::nullopt;
  }

  absl::StatusOr<bool> TryDropKeyword(Keyword target, Pos* pos = nullptr) {
    XLS_ASSIGN_OR_RETURN(const Token* peek, PeekToken());
    if (peek->IsKeyword(target)) {
      if (pos != nullptr) {
        *pos = peek->span().limit();
      }
      DropTokenOrDie();
      return true;
    }
    return false;
  }

  // Returns a pointer to the token at the head of the token stream.
  //
  // Note: if there are no tokens in the token steam, a pointer to an EOF-kind
  // token is returned.
  //
  // Returns an error status in the case of things like scan errors.
  absl::StatusOr<const Token*> PeekToken() {
    if (index_ >= tokens_.size()) {
      XLS_ASSIGN_OR_RETURN(Token token, scanner_->Pop());
      tokens_.push_back(std::make_unique<Token>(std::move(token)));
    }
    return tokens_[index_].get();
  }

  // Returns a token that has been popped destructively from the token stream.
  absl::StatusOr<Token> PopToken() {
    if (index_ >= tokens_.size()) {
      XLS_RETURN_IF_ERROR(PeekToken().status());
    }
    Token token = *tokens_[index_];
    index_ += 1;
    return token;
  }

  absl::StatusOr<std::string> PopIdentifierOrError(Span* span_out = nullptr) {
    XLS_ASSIGN_OR_RETURN(Token tok, PopTokenOrError(TokenKind::kIdentifier));
    if (span_out != nullptr) {
      *span_out = tok.span();
    }
    return *tok.GetValue();
  }

  // For use only when the caller knows there is lookahead present (in which
  // case we don't need to check for errors in potentially scanning the next
  // token).
  Token PopTokenOrDie() { return PopToken().value(); }

  // Wraps PopToken() to signify popping a token without needing the value.
  absl::Status DropToken() { return PopToken().status(); }

  void DropTokenOrDie() { CHECK_OK(DropToken()); }

  absl::StatusOr<bool> PeekTokenIs(TokenKind target);
  absl::StatusOr<bool> PeekTokenIs(Keyword target);
  absl::StatusOr<bool> PeekTokenIn(absl::Span<TokenKind const> targets);
  absl::StatusOr<bool> PeekKeywordIn(absl::Span<Keyword const> targets);

  // Args:
  //  target: The kind of token we need to see and pop (otherwise we give back
  //    an error).
  //  start: The start position to report in the error message, if non-null.
  //  context: Used as the suffix of the error message if the target token is
  //    not found.
  //  limit_pos: Populated with the limit position of the token, if non-null.
  absl::StatusOr<Token> PopTokenOrError(TokenKind target,
                                        const Token* start = nullptr,
                                        std::string_view context = "",
                                        Pos* limit_pos = nullptr);

  // Wrapper around PopTokenOrError that does not return the token. Helps
  // signify that the intent was to drop the token in the caller code vs
  // 'forgetting' to do something with the popped token.
  absl::Status DropTokenOrError(TokenKind target, const Token* start = nullptr,
                                std::string_view context = "",
                                Pos* limit_pos = nullptr);

  absl::StatusOr<Token> PopKeywordOrError(Keyword keyword,
                                          std::string_view context = "",
                                          Pos* limit_pos = nullptr);

  absl::Status DropKeywordOrError(Keyword target, Pos* limit_pos = nullptr);

  void DisableDoubleCAngle() { scanner_->DisableDoubleCAngle(); }
  void EnableDoubleCAngle() { scanner_->EnableDoubleCAngle(); }

 private:
  Scanner* scanner_;
  int64_t index_;
  // A vector of std::unique_ptrs is used for pointer stability as pointers are
  // handed out by PeekToken
  std::vector<std::unique_ptr<Token>> tokens_;
};

}  // namespace xls::dslx

#endif  // XLS_DSLX_FRONTEND_TOKEN_PARSER_H_
