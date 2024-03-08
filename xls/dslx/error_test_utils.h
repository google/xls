// Copyright 2023 The XLS Authors
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

#ifndef XLS_DSLX_ERROR_TEST_UTILS_H_
#define XLS_DSLX_ERROR_TEST_UTILS_H_

#include <string>

#include "gmock/gmock.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xls/dslx/frontend/bindings.h"

namespace xls::dslx {

// Example:
//
//  EXPECT_THAT(
//    my_status,
//    IsPosError("ParseError",
//               HasSubstr("parse-imonious")))
MATCHER_P2(
    IsPosError, kind, matcher,
    absl::StrCat("Positional error with kind ", kind, " and message that ",
                 testing::DescribeMatcher<std::string>(matcher, negation))) {
  const std::string want_kind = kind;
  const absl::Status& status = arg;
  if (status.code() != absl::StatusCode::kInvalidArgument) {
    *result_listener << "where status code is " << status.code();
    return false;
  }
  absl::StatusOr<PositionalErrorData> data =
      GetPositionalErrorData(status, want_kind);
  if (!data.ok()) {
    *result_listener << "where positional error status is " << data.status();
    return false;
  }
  return ExplainMatchResult(matcher, data->message, result_listener);
}

}  // namespace xls::dslx

#endif  // XLS_DSLX_ERROR_TEST_UTILS_H_
