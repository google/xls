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
#include "xls/dslx/bindings.h"
#include "xls/dslx/concrete_type.h"
#include "xls/dslx/deduce_ctx.h"

namespace xls::dslx {

// Raised when there is an error in token scanning.
class ScanError : public std::exception {
 public:
  ScanError(Span span, std::string message)
      : span_(std::move(span)), message_(std::move(message)) {}

  const char* what() const noexcept override { return message_.c_str(); }

  const Span& span() const { return span_; }

 private:
  Span span_;
  std::string message_;
};

// As above, but ScanErrors have positions (single points) in lieu of spans
// (position ranges).
inline void TryThrowScanError(const absl::Status& status) {
  std::string_view s = status.message();
  if (absl::ConsumePrefix(&s, "ScanError: ")) {
    std::vector<std::string_view> pieces =
        absl::StrSplit(s, absl::MaxSplits(" ", 1));
    if (pieces.size() < 2) {
      return;
    }
    absl::StatusOr<Span> span = Span::FromString(pieces[0]);
    throw ScanError(std::move(span.value()), std::string(pieces[1]));
  }
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
