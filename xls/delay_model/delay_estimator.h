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

#ifndef XLS_DELAY_MODEL_DELAY_ESTIMATOR_H_
#define XLS_DELAY_MODEL_DELAY_ESTIMATOR_H_

#include <cstdint>

#include "absl/container/flat_hash_map.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/node.h"

namespace xls {

// Abstraction describing a timing model for XLS operations.
class DelayEstimator {
 public:
  DelayEstimator(std::string_view name) : name_(name) {}
  virtual ~DelayEstimator() = default;

  const std::string& name() const { return name_; }

  // Returns the estimated delay of the given node in picoseconds.
  virtual absl::StatusOr<int64_t> GetOperationDelayInPs(Node* node) const = 0;

  // Compute the delay of the given node using logical effort estimation. Only
  // relatively simple operations (kAnd, kOr, etc) are supported using this
  // method.
  static absl::StatusOr<int64_t> GetLogicalEffortDelayInPs(Node* node,
                                                           int64_t tau_in_ps);

 private:
  std::string name_;
};

// Decorates an underlying delay estimator with an overriding modifier function.
class DecoratingDelayEstimator : public DelayEstimator {
 public:
  DecoratingDelayEstimator(std::string_view name,
                           const DelayEstimator& decorated,
                           std::function<int64_t(Node*, int64_t)> modifier);

  ~DecoratingDelayEstimator() override = default;

  absl::StatusOr<int64_t> GetOperationDelayInPs(Node* node) const override;

 private:
  const DelayEstimator& decorated_;
  std::function<int64_t(Node*, int64_t)> modifier_;
};

enum class DelayEstimatorPrecedence {
  kLow = 1,
  kMedium = 2,
  kHigh = 3,
};

// An abstraction which holds multiple DelayEstimator objects organized by name.
class DelayEstimatorManager {
 public:
  // Returns the delay estimator with the given name, or returns an error if no
  // such estimator exists.
  absl::StatusOr<DelayEstimator*> GetDelayEstimator(
      std::string_view name) const;

  absl::StatusOr<DelayEstimator*> GetDefaultDelayEstimator() const;

  // Adds a DelayEstimator to the manager and associates it with the given name.
  absl::Status RegisterDelayEstimator(
      std::unique_ptr<DelayEstimator> delay_estimator,
      DelayEstimatorPrecedence precedence);

  // Returns a list of the names of available models in this manager.
  absl::Span<const std::string> estimator_names() const {
    return estimator_names_;
  }

 private:
  absl::flat_hash_map<std::string, std::pair<DelayEstimatorPrecedence,
                                             std::unique_ptr<DelayEstimator>>>
      estimators_;
  std::vector<std::string> estimator_names_;
};

// Returns the singleton manager which holds the
DelayEstimatorManager& GetDelayEstimatorManagerSingleton();

}  // namespace xls

#endif  // XLS_DELAY_MODEL_DELAY_ESTIMATOR_H_
