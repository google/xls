// Copyright 2024 The XLS Authors
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

#ifndef XLS_SOLVERS_Z3_IR_TRANSLATOR_MATCHERS_H_
#define XLS_SOLVERS_Z3_IR_TRANSLATOR_MATCHERS_H_

#include <ostream>
#include <string>
#include <variant>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/strings/str_format.h"
#include "xls/solvers/z3_ir_translator.h"

namespace xls::solvers::z3 {

MATCHER(IsProvenTrue,
        absl::StrFormat("Value is %sProvenTrue", negation ? "not " : "")) {
  const ProverResult& result = arg;
  return testing::ExplainMatchResult(testing::IsTrue(),
                                     std::holds_alternative<ProvenTrue>(result),
                                     result_listener);
}

template <typename T>
class IsProvenFalseMatcher : public testing::MatcherInterface<ProverResult> {
 public:
  using is_gtest_matcher = void;
  explicit IsProvenFalseMatcher(T message) : message_(message) {}

  bool MatchAndExplain(ProverResult result,
                       testing::MatchResultListener* listener) const override {
    if (!std::holds_alternative<ProvenFalse>(result)) {
      *listener << "the result was proven true";
      return false;
    }
    const ProvenFalse& v = std::get<ProvenFalse>(result);
    return testing::ExplainMatchResult(message_, v.message, listener);
  }

  // Describes the property of a value matching this matcher.
  void DescribeTo(std::ostream* os) const override {
    *os << "is ProvenFalse with message "
        << testing::DescribeMatcher<std::string>(message_);
  }

  // Describes the property of a value NOT matching this matcher.
  void DescribeNegationTo(std::ostream* os) const override {
    *os << "is not ProvenFalse with message "
        << testing::DescribeMatcher<std::string>(message_);
  }

 private:
  T message_;
};

inline auto IsProvenFalse() {
  return IsProvenFalseMatcher<decltype(testing::A<std::string>())>(
      testing::A<std::string>());
}

template <typename T>
IsProvenFalseMatcher<T> IsProvenFalse(T t) {
  return IsProvenFalseMatcher<T>(t);
}

}  // namespace xls::solvers::z3

#endif  // XLS_SOLVERS_Z3_IR_TRANSLATOR_MATCHERS_H_
