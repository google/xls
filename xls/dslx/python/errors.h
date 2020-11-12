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

}  // namespace xls::dslx

#endif  // XLS_DSLX_PYTHON_ERRORS_H_
