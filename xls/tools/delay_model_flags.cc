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

#include "xls/tools/delay_model_flags.h"

#include <string>
#include <string_view>

#include "absl/base/nullability.h"
#include "absl/flags/flag.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/estimators/delay_model/delay_estimator.h"
#include "xls/estimators/delay_model/delay_estimators.h"

// LINT.IfChange
ABSL_FLAG(std::string, delay_model, "",
          "Delay model name to use from the registry. If unspecified, uses the "
          "tool default; pass empty (--delay_model=) to disable optional delay "
          "estimation.");
// LINT.ThenChange(
//   //xls/build_rules/xls_ir_rules.bzl,
//   //xls/build_rules/xls_providers.bzl,
//   //docs_src/codegen_options.md
// )

namespace xls {

bool IsDelayModelSpecifiedOnCommandLine() {
  return FLAGS_delay_model.IsSpecifiedOnCommandLine();
}

std::string GetDelayModelNameFromFlags(std::string_view default_model) {
  std::string flag_value = absl::GetFlag(FLAGS_delay_model);
  if (FLAGS_delay_model.IsSpecifiedOnCommandLine() || !flag_value.empty()) {
    return flag_value;
  }
  return std::string(default_model);
}

absl::StatusOr<DelayEstimator* absl_nullable> GetDelayEstimatorFromFlags(
    std::string_view default_model) {
  std::string model_name = GetDelayModelNameFromFlags(default_model);
  if (model_name.empty()) {
    return nullptr;
  }
  return GetDelayEstimator(model_name);
}

absl::StatusOr<DelayEstimator* absl_nonnull> GetRequiredDelayEstimatorFromFlags(
    std::string_view default_model) {
  std::string model_name = GetDelayModelNameFromFlags(default_model);
  if (model_name.empty()) {
    return absl::InvalidArgumentError("--delay_model must be specified.");
  }
  XLS_ASSIGN_OR_RETURN(DelayEstimator * estimator,
                       GetDelayEstimator(model_name));
  XLS_RET_CHECK(estimator != nullptr);
  return estimator;
}

}  // namespace xls
