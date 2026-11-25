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

#ifndef XLS_TOOLS_AREA_MODEL_FLAGS_H_
#define XLS_TOOLS_AREA_MODEL_FLAGS_H_

#include <string>
#include <string_view>

#include "absl/base/nullability.h"
#include "absl/status/statusor.h"
#include "xls/estimators/area_model/area_estimator.h"

namespace xls {

// Returns whether `--area_model` was specified on the command line.
// Note: programmatic mutations via `absl::SetFlag(&FLAGS_area_model, ...)` do
// not mark the flag as specified on the command line.
bool IsAreaModelSpecifiedOnCommandLine();

// Returns the effective area model name from `--area_model`.
//
// Resolution order:
//  1. If `--area_model` was specified on the command line, returns its value
//     (allowing `--area_model=""` to explicitly clear a non-empty
//     `default_model`).
//  2. If the flag holds a non-empty value (e.g., set programmatically in tests
//     via `absl::SetFlag(&FLAGS_area_model, ...)`), returns that value. Note:
//     setting the flag programmatically to `""` cannot be distinguished from an
//     untouched flag and falls through to `default_model`.
//  3. Otherwise returns `default_model`.
std::string GetAreaModelNameFromFlags(std::string_view default_model = "");

// Looks up the area estimator selected by `--area_model`, returning `nullptr`
// when the resolved model name is empty (either unspecified with an empty
// `default_model`, or explicitly cleared via `--area_model=""`).
absl::StatusOr<AreaEstimator* absl_nullable> GetAreaEstimatorFromFlags(
    std::string_view default_model = "");

// Like `GetAreaEstimatorFromFlags`, but returns `InvalidArgumentError` when the
// resolved area model name is empty.
absl::StatusOr<AreaEstimator* absl_nonnull> GetRequiredAreaEstimatorFromFlags(
    std::string_view default_model = "");

}  // namespace xls

#endif  // XLS_TOOLS_AREA_MODEL_FLAGS_H_
