// Copyright 2024 The XLS Authors
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

#ifndef XLS_ESTIMATORS_AREA_MODEL_AREA_ESTIMATOR_H_
#define XLS_ESTIMATORS_AREA_MODEL_AREA_ESTIMATOR_H_

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xls/ir/node.h"

namespace xls {

// Abstraction describing an area model for XLS operations.
class AreaEstimator {
 private:
  // This function returns the estimation of area of 1-bit register.
  // This estimation comes from the sequential area of kIdentity, which has
  // two k-bit registers, where k is the width of the output.
  virtual absl::StatusOr<double> GetOneBitRegisterAreaInSquareMicrons()
      const = 0;

 public:
  explicit AreaEstimator(std::string_view name) : name_(name) {}
  virtual ~AreaEstimator() = default;

  const std::string& name() const { return name_; }

  // Returns the estimated area of the given node in square micrometers.
  virtual absl::StatusOr<double> GetOperationAreaInSquareMicrons(
      Node* node) const = 0;

  // Returns the estimated area of n-bit register
  absl::StatusOr<double> GetRegisterAreaInSquareMicrons(
      const uint64_t& register_width) const;

 private:
  std::string name_;
};

// A manager holding multiple Area Estimator singletons
class AreaEstimatorManager {
 public:
  // Returns the area estimator of the given name, or returns an error if no
  // such estimator exists.
  absl::StatusOr<AreaEstimator*> GetAreaEstimator(std::string_view name) const;

  // Adds an AreaEstimator object.
  absl::Status AddAreaEstimator(std::unique_ptr<AreaEstimator> estimator);

  // Returns a list of the names of available models in this manager.
  absl::Span<const std::string> estimator_names() const {
    return estimator_names_;
  }

 private:
  absl::flat_hash_map<std::string, std::unique_ptr<AreaEstimator>> estimators_;
  std::vector<std::string> estimator_names_;
};

// Returns the singleton manager which holds the AreaEstimator objects
AreaEstimatorManager& GetAreaEstimatorManagerSingleton();

}  // namespace xls

#endif  // XLS_ESTIMATORS_AREA_MODEL_AREA_ESTIMATOR_H_
