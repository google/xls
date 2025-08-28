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

#include "xls/passes/critical_path_delay_analysis.h"

#include <algorithm>
#include <cstdint>
#include <memory>

#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xls/common/status/status_macros.h"
#include "xls/estimators/delay_model/delay_estimator.h"
#include "xls/estimators/delay_model/delay_estimators.h"
#include "xls/ir/node.h"
#include "xls/passes/optimization_pass.h"

namespace xls {

CriticalPathDelayAnalysis::CriticalPathDelayAnalysis(
    const DelayEstimator* estimator)
    : delay_estimator_(estimator) {
  CHECK(delay_estimator_ != nullptr);
}

absl::StatusOr<std::shared_ptr<CriticalPathDelayAnalysis>>
CriticalPathDelayAnalysis::Create(const AnalysisOptions& options) {
  if (options.delay_model_name.has_value()) {
    XLS_ASSIGN_OR_RETURN(DelayEstimator * delay_estimator,
                         GetDelayEstimator(*options.delay_model_name));
    return std::make_shared<CriticalPathDelayAnalysis>(delay_estimator);
  } else {
    return std::make_shared<CriticalPathDelayAnalysis>(
        &GetStandardDelayEstimator());
  }
}

int64_t CriticalPathDelayAnalysis::ComputeInfo(
    Node* node, absl::Span<const int64_t* const> operand_infos) const {
  int64_t max_operand_arrival_time = 0;
  for (const int64_t* op_info : operand_infos) {
    if (op_info == nullptr) {
      continue;
    }
    max_operand_arrival_time = std::max(max_operand_arrival_time, *op_info);
  }
  absl::StatusOr<int64_t> delay = delay_estimator_->GetOperationDelayInPs(node);

  // If estimator returns error or negative delay, treat delay as 0.
  int64_t node_delay = 0;
  if (delay.ok() && *delay > 0) {
    node_delay = *delay;
  }

  int64_t node_arrival_time = node_delay + max_operand_arrival_time;
  return node_arrival_time;
}

absl::Status CriticalPathDelayAnalysis::MergeWithGiven(
    int64_t& info, const int64_t& given) const {
  // Arrival time is lower-bounded by givens.
  info = std::max(info, given);
  return absl::OkStatus();
}

}  // namespace xls
