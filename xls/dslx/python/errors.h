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
#include "xls/common/string_to_int.h"
#include "xls/dslx/concrete_type.h"
#include "xls/dslx/cpp_bindings.h"
#include "xls/dslx/deduce_ctx.h"

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
  explicit TypeMissingError(AstNode* node, AstNode* user)
      : module_(node->owner()->shared_from_this()), node_(node), user_(user) {
    ResetMessage();
  }

  std::shared_ptr<Module> module() const { return module_; }

  AstNode* node() const { return node_; }
  void set_node(AstNode* node) { node_ = node; }

  AstNode* user() const { return user_; }
  void set_user(AstNode* user) {
    user_ = user;
    ResetMessage();
  }
  void set_span(const Span& span) { span_ = span; }
  const absl::optional<Span>& span() const { return span_; }

  const char* what() const noexcept override { return message_.c_str(); }

 private:
  void ResetMessage() {
    message_ =
        absl::StrFormat("%p %p AST node is missing a corresponding type: %s",
                        node_, user_, node_->ToString());
  }

  // Module reference is held so we ensure the AST node is not deallocated.
  std::shared_ptr<Module> module_;

  // AST node that was missing from the mapping.
  AstNode* node_;

  AstNode* user_ = nullptr;
  absl::optional<Span> span_;

  // Message to display when raised to Python.
  std::string message_;
};

// Raised when there is a type checking error in DSLX code (e.g. two types
// should match up by the typechecking rules, but they did not).
class XlsTypeError : public std::exception {
 public:
  XlsTypeError(Span span, std::unique_ptr<ConcreteType> lhs,
               std::unique_ptr<ConcreteType> rhs, absl::string_view suffix)
      : span_(std::move(span)), lhs_(std::move(lhs)), rhs_(std::move(rhs)) {
    std::string suffix_str;
    if (!suffix.empty()) {
      suffix_str = absl::StrCat(": ", suffix);
    }
    std::string lhs_str = lhs_ == nullptr ? "<none>" : lhs_->ToString();
    std::string rhs_str = rhs_ == nullptr ? "<none>" : rhs_->ToString();
    message_ = absl::StrFormat("%s Types are not compatible: %s vs %s%s",
                               span_.ToString(), lhs_str, rhs_str, suffix_str);
  }

  const Span& span() const { return span_; }
  const std::unique_ptr<ConcreteType>& lhs() const { return lhs_; }
  const std::unique_ptr<ConcreteType>& rhs() const { return rhs_; }
  const std::string& message() const { return message_; }

  const char* what() const noexcept override { return message_.c_str(); }

 private:
  Span span_;
  std::unique_ptr<ConcreteType> lhs_;
  std::unique_ptr<ConcreteType> rhs_;
  std::string message_;
};

static const char* kNoTypeIndicator = "<>";

// Error raised when an error occurs during deductive type inference.
//
// Attributes:
//   span: The span at which the type deduction error occurred.
//   type: The (AST) type that failed to deduce. May be null.
class TypeInferenceError : public std::exception {
 public:
  // Args:
  //  suffix: Message suffix to use when displaying the error.
  TypeInferenceError(Span span, std::unique_ptr<ConcreteType> type,
                     absl::string_view suffix)
      : span_(std::move(span)), type_(std::move(type)) {
    std::string type_str = kNoTypeIndicator;
    if (type != nullptr) {
      type_str = type->ToString();
    }
    message_ = absl::StrFormat("%s %s Could not infer type", span_.ToString(),
                               type_str);
    if (!suffix.empty()) {
      message_ += absl::StrCat(": ", suffix);
    }
  }

  const char* what() const noexcept override { return message_.c_str(); }

  const Span& span() const { return span_; }
  const ConcreteType* type() const { return type_.get(); }
  const std::string& message() const { return message_; }

 private:
  Span span_;
  std::unique_ptr<ConcreteType> type_;
  std::string message_;
};

class ArgCountMismatchError : public std::exception {
 public:
  ArgCountMismatchError(Span span, std::string message)
      : span_(std::move(span)), message_(std::move(message)) {}

  const char* what() const noexcept override { return message_.c_str(); }

  const Span& span() const { return span_; }
  const std::string& message() const { return message_; }

 private:
  Span span_;
  std::string message_;
};

// Sees if the status contains a stylized FailureError -- if so, throws it as a
// Python exception.
inline void TryThrowFailureError(const absl::Status& status) {
  if (status.code() == absl::StatusCode::kInternal &&
      absl::StartsWith(status.message(), "FailureError")) {
    std::pair<Span, std::string> data = ParseErrorGetData(status).value();
    throw FailureError(data.second, data.first);
  }
}

inline void TryThrowArgCountMismatchError(const absl::Status& status) {
  if (!status.ok() &&
      absl::StartsWith(status.message(), "ArgCountMismatchError")) {
    auto [span, message] = ParseErrorGetData(status).value();
    throw ArgCountMismatchError(span, message);
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

inline void TryThrowTypeMissingError(const absl::Status& status) {
  absl::string_view s = status.message();
  if (status.code() == absl::StatusCode::kInternal &&
      absl::ConsumePrefix(&s, "TypeMissingError: ")) {
    auto [node, user] = ParseTypeMissingErrorMessage(s);
    throw TypeMissingError(absl::bit_cast<AstNode*>(node),
                           absl::bit_cast<AstNode*>(user));
  }
}

inline void TryThrowTypeInferenceError(const absl::Status& status) {
  absl::string_view s = status.message();
  if (absl::ConsumePrefix(&s, "TypeInferenceError: ")) {
    std::vector<absl::string_view> pieces =
        absl::StrSplit(s, absl::MaxSplits(" ", 1));
    XLS_CHECK_EQ(pieces.size(), 2);
    absl::StatusOr<Span> span = Span::FromString(pieces[0]);
    absl::string_view rest = pieces[1];

    absl::StatusOr<std::unique_ptr<ConcreteType>> type;
    if (absl::ConsumePrefix(&rest, kNoTypeIndicator)) {
      type = nullptr;
    } else {
      type = ConcreteTypeFromString(&rest);
    }
    rest = absl::StripAsciiWhitespace(rest);
    XLS_CHECK(span.ok() && type.ok())
        << "Could not parse type inference error string: \"" << status.message()
        << "\" span: " << span.status() << " type: " << type.status();
    throw TypeInferenceError(std::move(span.value()), std::move(type).value(),
                             rest);
  }
}

inline void TryThrowXlsTypeError(const absl::Status& status) {
  absl::string_view s = status.message();
  if (absl::ConsumePrefix(&s, "XlsTypeError: ")) {
    std::vector<absl::string_view> pieces =
        absl::StrSplit(s, absl::MaxSplits(" ", 1));
    XLS_CHECK_EQ(pieces.size(), 2);
    absl::StatusOr<Span> span = Span::FromString(pieces[0]);
    absl::string_view rest = pieces[1];
    rest = absl::StripAsciiWhitespace(rest);

    absl::StatusOr<std::unique_ptr<ConcreteType>> lhs;
    if (absl::ConsumePrefix(&rest, "<none>")) {
      lhs = nullptr;
    } else {
      lhs = ConcreteTypeFromString(&rest);
    }

    rest = absl::StripAsciiWhitespace(rest);

    absl::StatusOr<std::unique_ptr<ConcreteType>> rhs;
    if (absl::ConsumePrefix(&rest, "<none>")) {
      rhs = nullptr;
    } else {
      rhs = ConcreteTypeFromString(&rest);
    }

    rest = absl::StripAsciiWhitespace(rest);

    XLS_CHECK(span.ok() && lhs.ok() && rhs.ok())
        << "Could not parse type inference error string: \"" << status.message()
        << "\" span: " << span.status() << " lhs: " << lhs.status()
        << " rhs: " << rhs.status();
    throw XlsTypeError(std::move(span.value()), std::move(lhs).value(),
                       std::move(rhs).value(), rest);
  }
}

}  // namespace xls::dslx

#endif  // XLS_DSLX_PYTHON_ERRORS_H_
