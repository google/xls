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

#ifndef XLS_SOLVERS_Z3_ASSERT_TESTUTILS_H_
#define XLS_SOLVERS_Z3_ASSERT_TESTUTILS_H_

#include <cstdint>
#include <optional>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "xls/ir/function.h"
#include "xls/ir/function_base.h"
#include "xls/solvers/z3_ir_translator.h"
namespace xls::solvers::z3 {

namespace internal {
bool DoIsAssertClean(FunctionBase* fb, std::optional<int64_t> activations,
                     testing::MatchResultListener* result_listener);
absl::StatusOr<ProverResult> TryProveAssertClean(Function* func);

}  // namespace internal

MATCHER(IsAssertClean,
        absl::StrFormat("%s be proven to not throw asserts on any input",
                        negation ? "can" : "cannot")) {
  return internal::DoIsAssertClean(arg, std::nullopt, result_listener);
}

MATCHER_P(
    IsAssertClean, activations,
    absl::StrFormat(
        "%s be proven to not throw asserts on any input in %d activations",
        negation ? "can" : "cannot", activations)) {
  int64_t act = activations;
  if (arg->IsFunction()) {
    *result_listener << "Activations cannot be passed for functions like: "
                     << testing::PrintToString(arg);
    return false;
  }
  return internal::DoIsAssertClean(arg, act, result_listener);
}

}  // namespace xls::solvers::z3

#endif  // XLS_SOLVERS_Z3_ASSERT_TESTUTILS_H_
