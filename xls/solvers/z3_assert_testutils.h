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

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "xls/ir/function.h"
#include "xls/solvers/z3_ir_translator.h"
namespace xls::solvers::z3 {

namespace internal {
bool DoIsAssertClean(Function* arg,
                     testing::MatchResultListener* result_listener);
absl::StatusOr<ProverResult> TryProveAssertClean(Function* func);
}  // namespace internal

MATCHER(IsAssertClean,
        absl::StrFormat("%s be proven to not throw asserts on any input",
                        negation ? "can" : "cannot")) {
  return internal::DoIsAssertClean(arg, result_listener);
}

}  // namespace xls::solvers::z3

#endif  // XLS_SOLVERS_Z3_ASSERT_TESTUTILS_H_
