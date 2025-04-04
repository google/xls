// Copyright 2025 The XLS Authors
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

#ifndef XLS_DSLX_TYPE_SYSTEM_V2_MATCHERS_H_
#define XLS_DSLX_TYPE_SYSTEM_V2_MATCHERS_H_

#include <cstddef>
#include <string>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/substitute.h"
#include "xls/dslx/parse_and_typecheck.h"
#include "xls/dslx/type_system/typecheck_test_utils.h"
#include "xls/dslx/type_system_v2/type_system_test_utils.h"

namespace xls::dslx {

using ::absl_testing::StatusIs;
using ::testing::HasSubstr;

class HasRepeatedSubstrMatcher
    : public ::testing::internal::HasSubstrMatcher<std::string> {
 public:
  HasRepeatedSubstrMatcher(const std::string& substring, size_t n)
      : ::testing::internal::HasSubstrMatcher<std::string>(substring),
        substring_(substring),
        n_(n) {}

  bool MatchAndExplain(const std::string& s,
                       ::testing::MatchResultListener* /* listener */) const {
    size_t pos = 0;
    for (size_t i = 0; i < n_; i++) {
      pos = s.find(substring_, pos);
      if (pos == std::string::npos) {
        return false;
      }
      pos += substring_.size();
    }
    return true;
  }

 private:
  // substring_ is private in HasSubstrMatcher so we have to duplicate it here
  // to use.
  const std::string substring_;
  const size_t n_;
};

// Verifies that a type info string contains the given node string and type
// string combo.
MATCHER_P2(HasNodeWithType, node, type, "") {
  return ExplainMatchResult(
      HasSubstr(absl::Substitute("node: `$0`, type: $1", node, type)), arg,
      result_listener);
}

MATCHER_P3(HasRepeatedNodeWithType, node, type, n, "") {
  return ExplainMatchResult(
      ::testing::MakePolymorphicMatcher(HasRepeatedSubstrMatcher(
          absl::Substitute("node: `$0`, type: $1", node, type), n)),
      arg, result_listener);
}

// Verifies the type produced by `TypecheckV2`, for the topmost node only, in a
// simple AST (typically a one-liner). The `arg` is the DSLX code and `expected`
// is the type string.
MATCHER_P(TopNodeHasType, expected, "") {
  absl::StatusOr<TypecheckResult> result = TypecheckV2(arg);
  if (!result.ok()) {
    *result_listener << "Failed to typecheck: `" << arg
                     << "`; status: " << result.status();
    return false;
  }
  absl::StatusOr<std::string> type_info_string = TypeInfoToString(result->tm);
  if (!type_info_string.ok()) {
    *result_listener << "Failed to convert type info to string; status: "
                     << type_info_string.status();
    return false;
  }
  bool matched = ExplainMatchResult(HasNodeWithType(arg, expected),
                                    *type_info_string, result_listener);
  if (!matched) {
    *result_listener << "Type info: " << *type_info_string;
  }
  return matched;
}

// Verifies that the `TypecheckV2` output contains a one-line statement block
// with the given type.
MATCHER_P2(HasOneLineBlockWithType, expected_line, expected_type, "") {
  bool matched = ExplainMatchResult(
      HasSubstr(absl::Substitute("node: `{\n    $0\n}`, type: $1",
                                 expected_line, expected_type)),
      arg, result_listener);
  if (!matched) {
    *result_listener << "Type info: " << arg;
  }
  return matched;
}

// Verifies that `TypecheckV2` fails for the given DSLX code, using `matcher`
// for the error string. The `arg` is the DSLX code.
MATCHER_P(TypecheckFails, matcher, "") {
  return ExplainMatchResult(
      StatusIs(absl::StatusCode::kInvalidArgument, matcher), TypecheckV2(arg),
      result_listener);
}

template <typename M>
bool CheckTypeInfo(TypecheckedModule module,
                   ::testing::MatchResultListener* result_listener, M matcher) {
  absl::StatusOr<std::string> type_info_string = TypeInfoToString(module);
  if (!type_info_string.ok()) {
    *result_listener << "Failed to convert type info to string; status: "
                     << type_info_string.status();
    return false;
  }
  bool matched =
      ExplainMatchResult(matcher, *type_info_string, result_listener);
  if (!matched) {
    *result_listener << "Type info: " << *type_info_string;
  }
  return matched;
}

// Verifies that `TypecheckV2` succeeds for the given DSLX code and the
// resulting type info string satisfies the given `matcher`.
MATCHER_P(TypecheckSucceeds, matcher, "") {
  absl::StatusOr<TypecheckResult> result = TypecheckV2(arg);
  if (!result.ok()) {
    *result_listener << "Failed to typecheck: `" << arg
                     << "`; status: " << result.status();
    return false;
  }
  return CheckTypeInfo(result->tm, result_listener, matcher);
}

MATCHER_P(HasTypeInfo, matcher, "") {
  return CheckTypeInfo(arg.tm, result_listener, matcher);
}

}  // namespace xls::dslx

#endif  // XLS_DSLX_TYPE_SYSTEM_V2_MATCHERS_H_
