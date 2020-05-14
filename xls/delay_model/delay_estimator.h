// Copyright 2020 Google LLC
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

#ifndef THIRD_PARTY_XLS_DELAY_MODEL_DELAY_ESTIMATOR_H_
#define THIRD_PARTY_XLS_DELAY_MODEL_DELAY_ESTIMATOR_H_

#include "absl/container/flat_hash_map.h"
#include "absl/types/span.h"
#include "xls/common/integral_types.h"
#include "xls/common/status/status_macros.h"
#include "xls/common/status/statusor.h"
#include "xls/ir/node.h"

namespace xls {

// Abstraction describing a timing model for XLS operations.
class DelayEstimator {
 public:
  virtual ~DelayEstimator() = default;

  // Returns the estimated delay of the given node in picoseconds.
  virtual xabsl::StatusOr<int64> GetOperationDelayInPs(Node* node) const = 0;

  // Compute the delay of the given node using logical effort estimation. Only
  // relatively simple operations (kAnd, kOr, etc) are supported using this
  // method.
  static xabsl::StatusOr<int64> GetLogicalEffortDelayInPs(Node* node,
                                                          int64 tau_in_ps);
};

// An abstraction which holds multiple DelayEstimator objects organized by name.
class DelayEstimatorManager {
 public:
  // Returns the delay estimator with the given name, or returns an error if no
  // such estimator exists.
  xabsl::StatusOr<DelayEstimator*> GetDelayEstimator(
      absl::string_view name) const;

  // Adds a DelayEstimator to the manager and associates it with the given name.
  absl::Status RegisterDelayEstimator(
      absl::string_view name, std::unique_ptr<DelayEstimator> delay_estimator);

  // Returns a list of the names of available models in this manager.
  absl::Span<const std::string> estimator_names() const {
    return estimator_names_;
  }

 private:
  absl::flat_hash_map<std::string, std::unique_ptr<DelayEstimator>> estimators_;
  std::vector<std::string> estimator_names_;
};

// Returns the singleton manager which holds the
DelayEstimatorManager& GetDelayEstimatorManagerSingleton();

}  // namespace xls

#endif  // THIRD_PARTY_XLS_DELAY_MODEL_DELAY_ESTIMATOR_H_
