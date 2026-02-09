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

// Testing utilities for working with absl::Status, absl::StatusOr.
//
// Defines the following utilities:
//
//   =================
//   XLS_EXPECT_OK(s)
//
//   XLS_ASSERT_OK(s)
//   =================
//   Convenience macros for `EXPECT_THAT(s, IsOk())`, where `s` is either
//   a `Status` or a `StatusOr<T>`.
//
//   There are no XLS_EXPECT_NOT_OK/XLS_ASSERT_NOT_OK macros since they would
//   not provide much value (when they fail, they would just print the OK status
//   which conveys no more information than EXPECT_FALSE(s.ok());
//   If you want to check for particular errors, better alternatives are:
//   EXPECT_THAT(s, StatusIs(expected_error));
//   EXPECT_THAT(s, StatusIs(_, _, HasSubstr("expected error")));
//

#ifndef XLS_COMMON_STATUS_MATCHERS_H_
#define XLS_COMMON_STATUS_MATCHERS_H_

#include <string_view>

#include "gmock/gmock.h"
#include "absl/status/status_matchers.h"  // IWYU pragma: keep
#include "xls/common/status/status_builder.h"
#include "xls/common/status/status_macros.h"

namespace xls {
namespace status_testing {
namespace internal_status {

void AddFatalFailure(std::string_view expression,
                     const xabsl::StatusBuilder& builder);

}  // namespace internal_status

// Macros for testing the results of functions that return absl::Status or
// absl::StatusOr<T> (for any type T).
#define XLS_EXPECT_OK(expression) \
  EXPECT_THAT(expression, ::absl_testing::IsOk())
#define XLS_ASSERT_OK(expression) \
  ASSERT_THAT(expression, ::absl_testing::IsOk())

// Executes an expression that returns a absl::StatusOr, and assigns the
// contained variable to lhs if the error code is OK.
// If the Status is non-OK, generates a test failure and returns from the
// current function, which must have a void return type.
//
// Example: Declaring and initializing a new value
//   XLS_ASSERT_OK_AND_ASSIGN(const ValueType& value, MaybeGetValue(arg));
//
// Example: Assigning to an existing value
//   ValueType value;
//   XLS_ASSERT_OK_AND_ASSIGN(value, MaybeGetValue(arg));
//
// The value assignment example would expand into something like:
//   auto status_or_value = MaybeGetValue(arg);
//   XLS_ASSERT_OK(status_or_value.status());
//   value = std::move(status_or_value).value();
//
// WARNING: Like XLS_ASSIGN_OR_RETURN, XLS_ASSERT_OK_AND_ASSIGN expands into
// multiple
//   statements; it cannot be used in a single statement (e.g. as the body of
//   an if statement without {})!
#define XLS_ASSERT_OK_AND_ASSIGN(lhs, rexpr)                                  \
  XLS_ASSIGN_OR_RETURN(/* NOLINT(clang-diagnostic-shadow) */                  \
                       lhs, rexpr,                                            \
                       xls::status_testing::internal_status::AddFatalFailure( \
                           #rexpr, _))

// Executes an expression that returns a absl::StatusOr, and compares the
// contained variable to rexpr if the error code is OK.
// If the Status is non-OK it generates a nonfatal test failure
#define XLS_EXPECT_OK_AND_EQ(lhs, rexpr) \
  EXPECT_THAT(lhs, ::absl_testing::IsOkAndHolds(rexpr));

}  // namespace status_testing
}  // namespace xls

#endif  // XLS_COMMON_STATUS_MATCHERS_H_
