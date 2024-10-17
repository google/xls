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
//   =========================================
//   CanonicalStatusIs(canonical_code_matcher,
//                     error_message_matcher)
//   =========================================
//
//   This gMock matcher matches a Status or StatusOr<T> value if
//   both of the following are true:
//
//     - the status' error_code() matches canonical_code_matcher and
//     - the status' error_message() matches error_message_matcher.
//
//   This differs from the StatusIs() matchers in that it will match a status
//   with any error_space(), as long as that error space maps the error_code()
//   to the expected canonical code.
//
//   =========================================
//   CanonicalStatusIs(canonical_code_matcher)
//   =========================================
//
//   This is a shorthand for
//     CanonicalStatusIs(canonical_code_matcher,
//                       testing::_)
//   In other words, it's like the 2-argument CanonicalStatusIs() except that
//   it ignores the error message.

#ifndef XLS_COMMON_STATUS_MATCHERS_H_
#define XLS_COMMON_STATUS_MATCHERS_H_

#include <ostream>  // NOLINT
#include <string>
#include <string_view>
#include <utility>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"  // IWYU pragma: keep
#include "absl/status/statusor.h"
#include "xls/common/status/status_builder.h"
#include "xls/common/status/status_macros.h"

namespace xls {
namespace status_testing {
namespace internal_status {

inline const absl::Status& GetStatus(const absl::Status& status) {
  return status;
}

template <typename T>
inline const absl::Status& GetStatus(const absl::StatusOr<T>& status) {
  return status.status();
}

// `StatusCode` is implicitly convertible from `int`, `absl::StatusCode`, and
// any enum that is associated with an error space, and explicitly convertible
// to these types as well.
//
// We need this class because absl::StatusCode (as a scoped enum) is not
// implicitly convertible to int. In order to handle use cases like
//   StatusIs(Anyof(absl::StatusCode::kUnknown, absl::StatusCode::kCancelled))
// which uses polymorphic matchers, we need to unify the interfaces into
// Matcher<StatusCode>.
class StatusCode {
 public:
  StatusCode(int code) : code_(code) {}  // NOLINT
  StatusCode(absl::StatusCode code)      // NOLINT
      : code_(static_cast<int>(code)) {}
  explicit operator int() const { return code_; }
  explicit operator absl::StatusCode() const {
    return static_cast<absl::StatusCode>(code_);
  }

  friend inline void PrintTo(const StatusCode& code, std::ostream* os) {
    *os << code.code_;
  }

 private:
  int code_;
};

// Relational operators to handle matchers like Eq, Lt, etc..
inline bool operator==(const StatusCode& lhs, const StatusCode& rhs) {
  return static_cast<int>(lhs) == static_cast<int>(rhs);
}
inline bool operator!=(const StatusCode& lhs, const StatusCode& rhs) {
  return static_cast<int>(lhs) != static_cast<int>(rhs);
}
inline bool operator<(const StatusCode& lhs, const StatusCode& rhs) {
  return static_cast<int>(lhs) < static_cast<int>(rhs);
}
inline bool operator<=(const StatusCode& lhs, const StatusCode& rhs) {
  return static_cast<int>(lhs) <= static_cast<int>(rhs);
}
inline bool operator>(const StatusCode& lhs, const StatusCode& rhs) {
  return static_cast<int>(lhs) > static_cast<int>(rhs);
}
inline bool operator>=(const StatusCode& lhs, const StatusCode& rhs) {
  return static_cast<int>(lhs) >= static_cast<int>(rhs);
}

// CanonicalStatusIs() is a polymorphic matcher.  This class is the common
// implementation of it shared by all types T where CanonicalStatusIs() can be
// used as a Matcher<T>.
class CanonicalStatusIsMatcherCommonImpl {
 public:
  CanonicalStatusIsMatcherCommonImpl(
      ::testing::Matcher<StatusCode> code_matcher,
      ::testing::Matcher<const std::string&> message_matcher)
      : code_matcher_(std::move(code_matcher)),
        message_matcher_(std::move(message_matcher)) {}

  void DescribeTo(std::ostream* os) const;

  void DescribeNegationTo(std::ostream* os) const;

  bool MatchAndExplain(const absl::Status& status,
                       ::testing::MatchResultListener* result_listener) const;

 private:
  const ::testing::Matcher<StatusCode> code_matcher_;
  const ::testing::Matcher<const std::string&> message_matcher_;
};

// Monomorphic implementation of matcher CanonicalStatusIs() for a given type
// T.  T can be Status, StatusOr<>, or a reference to either of them.
template <typename T>
class MonoCanonicalStatusIsMatcherImpl : public ::testing::MatcherInterface<T> {
 public:
  explicit MonoCanonicalStatusIsMatcherImpl(
      CanonicalStatusIsMatcherCommonImpl common_impl)
      : common_impl_(std::move(common_impl)) {}

  void DescribeTo(std::ostream* os) const override {
    common_impl_.DescribeTo(os);
  }

  void DescribeNegationTo(std::ostream* os) const override {
    common_impl_.DescribeNegationTo(os);
  }

  bool MatchAndExplain(
      T actual_value,
      ::testing::MatchResultListener* result_listener) const override {
    return common_impl_.MatchAndExplain(
        xls::status_testing::internal_status::GetStatus(actual_value),
        result_listener);
  }

 private:
  CanonicalStatusIsMatcherCommonImpl common_impl_;
};

// Implements CanonicalStatusIs() as a polymorphic matcher.
class CanonicalStatusIsMatcher {
 public:
  template <typename StatusCodeMatcher, typename StatusMessageMatcher>
  CanonicalStatusIsMatcher(StatusCodeMatcher&& code_matcher,
                           StatusMessageMatcher&& message_matcher)
      : common_impl_(::testing::MatcherCast<StatusCode>(
                         std::forward<StatusCodeMatcher>(code_matcher)),
                     ::testing::MatcherCast<const std::string&>(
                         std::forward<StatusMessageMatcher>(message_matcher))) {
  }

  // Converts this polymorphic matcher to a monomorphic matcher of the given
  // type.  T can be StatusOr<>, Status, or a reference to either of them.
  template <typename T>
  operator ::testing::Matcher<T>() const {  // NOLINT
    return ::testing::Matcher<T>(
        new MonoCanonicalStatusIsMatcherImpl<T>(common_impl_));
  }

 private:
  const CanonicalStatusIsMatcherCommonImpl common_impl_;
};

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
