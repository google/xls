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

#ifndef XLS_ESTIMATORS_DELAY_MODEL_FFI_DELAY_ESTIMATOR_H_
#define XLS_ESTIMATORS_DELAY_MODEL_FFI_DELAY_ESTIMATOR_H_

#include <cstdint>
#include <optional>

#include "absl/status/statusor.h"
#include "xls/estimators/delay_model/delay_estimator.h"
#include "xls/ir/node.h"

namespace xls {

// Delay estimator for foreign function calls.
// This delay estimator _only_ handles Op::kInvoke calls.
//
// TODO(hzeller) 2023-07-11 This simplistic implementation right now only
// returns a constant value until we have a classification that we can read
// at runtime from some e.g. protobuffer
class FfiDelayEstimator : public DelayEstimator {
 public:
  explicit FfiDelayEstimator(std::optional<int64_t> fallback_delay_estimate)
      : DelayEstimator("ffi_delay_estimator"),
        fallback_delay_estimate_(fallback_delay_estimate) {}

  // Returns the estimated delay for an Op::kInvoke node, all other nodes
  // are ignored to be handled by a different DelayEstimator.
  absl::StatusOr<int64_t> GetOperationDelayInPs(Node* node) const final;

 private:
  std::optional<int64_t> fallback_delay_estimate_;
};
}  // namespace xls
#endif  // XLS_ESTIMATORS_DELAY_MODEL_FFI_DELAY_ESTIMATOR_H_
