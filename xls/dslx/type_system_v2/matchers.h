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

#include <string>

#include "gmock/gmock.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/substitute.h"
#include "xls/dslx/type_system/typecheck_test_utils.h"
#include "xls/dslx/type_system_v2/type_system_test_utils.h"

namespace xls::dslx {

using ::absl_testing::StatusIs;
using ::testing::HasSubstr;

// Verifies that a type info string contains the given node string and type
// string combo.
MATCHER_P2(HasNodeWithType, node, type, "") {
  return ExplainMatchResult(
      HasSubstr(absl::Substitute("node: `$0`, type: $1", node, type)), arg,
      result_listener);
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

// Verifies that `TypecheckV2` succeeds for the given DSLX code and the
// resulting type info string satisfies the given `matcher`.
MATCHER_P(TypecheckSucceeds, matcher, "") {
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
  bool matched =
      ExplainMatchResult(matcher, *type_info_string, result_listener);
  if (!matched) {
    *result_listener << "Type info: " << *type_info_string;
  }
  return matched;
}

}  // namespace xls::dslx

#endif  // XLS_DSLX_TYPE_SYSTEM_V2_MATCHERS_H_
