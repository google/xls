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

#ifndef XLS_SOLVERS_Z3_IR_EQUIVALENCE_H_
#define XLS_SOLVERS_Z3_IR_EQUIVALENCE_H_

#include <functional>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/time/time.h"
#include "xls/ir/function.h"
#include "xls/ir/package.h"
#include "xls/solvers/z3_ir_translator.h"

namespace xls::solvers::z3 {

// Verifies that the original function's behavior stays the same after running
// 'run_pass' on it. The callback *must* use the provided package and function
// and not the one passed to this call. The pass may not alter the type
// signature of the function. The pass must modify the provided function
// in-place.
//
// This call does not alter the input function at all.
//
// Returns 'true' if the pass does not cause the result to change.
absl::StatusOr<ProverResult> TryProveEquivalence(
    Function* original,
    const std::function<absl::Status(Package*, Function*)>& run_pass,
    absl::Duration timeout = absl::InfiniteDuration());

// Verifies that both functions have the same behaviors. Both functions must
// have exactly the same signatures, or an invalid argument error is returned.
//
// This call does not alter either function.
//
// Returns 'true' if the pass does not cause the result to change.
absl::StatusOr<ProverResult> TryProveEquivalence(
    Function* a, Function* b,
    absl::Duration timeout = absl::InfiniteDuration());

}  // namespace xls::solvers::z3

#endif  // XLS_SOLVERS_Z3_IR_EQUIVALENCE_H_
