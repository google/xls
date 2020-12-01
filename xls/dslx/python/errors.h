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

#ifndef XLS_DSLX_PYTHON_ERRORS_H_
#define XLS_DSLX_PYTHON_ERRORS_H_

#include "absl/strings/match.h"
#include "absl/strings/str_split.h"
#include "pybind11/pybind11.h"
#include "xls/dslx/cpp_bindings.h"

namespace xls::dslx {

// Raised when expression evaluation fails (as in 'panic' style failure).
//
// This is used e.g. in tests, but may be reusable for things like fatal errors.
class FailureError : public std::exception {
 public:
  explicit FailureError(std::string message, Span span)
      : message_(std::move(message)), span_(std::move(span)) {}

  const char* what() const noexcept override { return message_.c_str(); }

  const Span& span() const { return span_; }

 private:
  std::string message_;
  Span span_;
};

// Raised when there is an error in token scanning.
class ScanError : public std::exception {
 public:
  ScanError(Pos pos, std::string message)
      : pos_(std::move(pos)), message_(std::move(message)) {}

  const char* what() const noexcept override { return message_.c_str(); }

  const Pos& pos() const { return pos_; }

 private:
  Pos pos_;
  std::string message_;
};

// Raised when there is an error in parsing.
//
// TODO(leary): 2020-11-13 Rename to ParseError now that parser is all C++.
class CppParseError : public std::exception {
 public:
  CppParseError(Span span, std::string message)
      : span_(std::move(span)), message_(std::move(message)) {}

  const char* what() const noexcept override { return message_.c_str(); }

  const Span& span() const { return span_; }
  const std::string& message() const { return message_; }

 private:
  Span span_;
  std::string message_;
};

// Raised when a type is missing from a TypeInfo mapping.
class TypeMissingError : public std::exception {
 public:
  explicit TypeMissingError(AstNode* node)
      : module_(node->owner()->shared_from_this()), node_(node) {
    message_ = absl::StrFormat("AST node is missing a corresponding type: %s",
                               node->ToString());
  }

  std::shared_ptr<Module> module() const { return module_; }

  AstNode* node() const { return node_; }
  void set_node(AstNode* node) { node_ = node; }

  AstNode* user() const { return user_; }
  void set_user(AstNode* user) { user_ = user; }
  void set_span(const Span& span) { span_ = span; }
  const absl::optional<Span>& span() const { return span_; }

  const char* what() const noexcept override { return message_.c_str(); }

 private:
  // Module reference is held so we ensure the AST node is not deallocated.
  std::shared_ptr<Module> module_;

  // AST node that was missing from the mapping.
  AstNode* node_;

  AstNode* user_ = nullptr;
  absl::optional<Span> span_;

  // Message to display when raised to Python.
  std::string message_;
};

// Sees if the status contains a stylized FailureError -- if so, throws it as a
// Python exception.
inline void TryThrowFailureError(const absl::Status& status) {
  if (status.code() == absl::StatusCode::kInternal &&
      absl::StartsWith(status.message(), "FailureError")) {
    std::pair<Span, std::string> data =
        ParseErrorGetData(status, "FailureError: ").value();
    throw FailureError(data.second, data.first);
  }
}

// As above, but ScanErrors have positions (single points) in lieu of spans
// (position ranges).
inline void TryThrowScanError(const absl::Status& status) {
  absl::string_view s = status.message();
  if (absl::ConsumePrefix(&s, "ScanError: ")) {
    std::vector<absl::string_view> pieces =
        absl::StrSplit(s, absl::MaxSplits(" ", 1));
    if (pieces.size() < 2) {
      return;
    }
    absl::StatusOr<Pos> pos = Pos::FromString(pieces[0]);
    throw ScanError(std::move(pos.value()), std::string(pieces[1]));
  }
}

// Since the parser can encounter ScanErrors in the course of parsing, this also
// checks for ScanErrors via TryThrowScanError.
inline void TryThrowCppParseError(const absl::Status& status) {
  TryThrowScanError(status);
  auto data_status = ParseErrorGetData(status);
  if (!data_status.ok()) {
    return;
  }
  auto [span, unused] = data_status.value();
  throw CppParseError(std::move(span), std::string(status.message()));
}

// If the status is "not found" throws a key error with the given status
// message.
inline void TryThrowKeyError(const absl::Status& status) {
  if (status.code() == absl::StatusCode::kNotFound) {
    throw pybind11::key_error(std::string(status.message()));
  }
}

}  // namespace xls::dslx

#endif  // XLS_DSLX_PYTHON_ERRORS_H_
