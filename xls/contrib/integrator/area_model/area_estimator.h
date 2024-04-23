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

#ifndef XLS_CONTRIB_INTEGRATOR_AREA_MODEL_AREA_ESTIMATOR_H_
#define XLS_CONTRIB_INTEGRATOR_AREA_MODEL_AREA_ESTIMATOR_H_

#include <cstdint>
#include <memory>
#include <string_view>

#include "absl/status/statusor.h"
#include "xls/delay_model/delay_estimator.h"
#include "xls/ir/node.h"

namespace xls {

// Abstraction describing an area model for XLS operations.
class AreaEstimator {
 public:
  explicit AreaEstimator(const DelayEstimator* delay_estimator)
      : delay_estimator_(delay_estimator) {}

  AreaEstimator(const AreaEstimator& other) = delete;
  void operator=(const AreaEstimator& other) = delete;

  // Returns the estimated area of the given node. Units of area
  // depend on the model / data used.
  absl::StatusOr<int64_t> GetOperationArea(Node* node) const {
    return delay_estimator_->GetOperationDelayInPs(node);
  }

 private:
  const DelayEstimator* delay_estimator_;
};

// Returns an AreaEstimator for the estimator with the given 'name'.
absl::StatusOr<std::unique_ptr<AreaEstimator>> GetAreaEstimatorByName(
    std::string_view name);

}  // namespace xls

#endif  // XLS_CONTRIB_INTEGRATOR_AREA_MODEL_AREA_ESTIMATOR_H_
