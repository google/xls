// Copyright 2026 The XLS Authors
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

#ifndef XLS_TOOLS_DELAY_MODEL_FLAGS_H_
#define XLS_TOOLS_DELAY_MODEL_FLAGS_H_

#include <string>
#include <string_view>

#include "absl/base/nullability.h"
#include "absl/status/statusor.h"
#include "xls/estimators/delay_model/delay_estimator.h"

namespace xls {

// Returns whether `--delay_model` was specified on the command line.
// Note: programmatic mutations via `absl::SetFlag(&FLAGS_delay_model, ...)` do
// not mark the flag as specified on the command line.
bool IsDelayModelSpecifiedOnCommandLine();

// Returns the effective delay model name from `--delay_model`.
//
// Resolution order:
//  1. If `--delay_model` was specified on the command line, returns its value
//     (allowing `--delay_model=""` to explicitly clear a non-empty
//     `default_model`).
//  2. If the flag holds a non-empty value (e.g., set programmatically in tests
//     via `absl::SetFlag(&FLAGS_delay_model, ...)`), returns that value. Note:
//     setting the flag programmatically to `""` cannot be distinguished from an
//     untouched flag and falls through to `default_model`.
//  3. Otherwise returns `default_model`.
std::string GetDelayModelNameFromFlags(std::string_view default_model = "");

// Looks up the delay estimator selected by `--delay_model`, returning `nullptr`
// when the resolved model name is empty (either unspecified with an empty
// `default_model`, or explicitly cleared via `--delay_model=""`).
absl::StatusOr<DelayEstimator* absl_nullable> GetDelayEstimatorFromFlags(
    std::string_view default_model = "");

// Like `GetDelayEstimatorFromFlags`, but returns `InvalidArgumentError` when
// the resolved delay model name is empty.
absl::StatusOr<DelayEstimator* absl_nonnull> GetRequiredDelayEstimatorFromFlags(
    std::string_view default_model = "");

}  // namespace xls

#endif  // XLS_TOOLS_DELAY_MODEL_FLAGS_H_
