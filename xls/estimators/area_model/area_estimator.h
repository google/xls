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
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xls/estimators/estimator_model.pb.h"
#include "xls/ir/node.h"
#include "xls/ir/op.h"

namespace xls {

// Abstraction describing an area model for XLS operations.
class AreaEstimator {
 public:
  explicit AreaEstimator(std::string_view name) : name_(name) {}
  virtual ~AreaEstimator() = default;

  const std::string& name() const { return name_; }

  // Returns the estimated area of the given node in square micrometers.
  virtual absl::StatusOr<double> GetOperationAreaInSquareMicrons(
      Node* node) const = 0;

  // Returns the estimated area of the given function base in square
  // micrometers.
  virtual absl::StatusOr<double> GetFunctionBaseAreaInSquareMicrons(
      FunctionBase* fb) const;

  // Returns the estimated area of n-bit register
  absl::StatusOr<double> GetRegisterAreaInSquareMicrons(
      const uint64_t& register_width) const;

  // This function returns the estimation of area of 1-bit register.
  // This estimation comes from the sequential area of kIdentity, which has
  // two k-bit registers, where k is the width of the output.
  virtual absl::StatusOr<double> GetOneBitRegisterAreaInSquareMicrons()
      const = 0;

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

// AreaEstimator that evaluates a pre-computed CompiledEstimatorModel proto.
class ProtoAreaEstimator : public AreaEstimator {
 public:
  ProtoAreaEstimator(
      std::string_view name,
      absl::flat_hash_map<Op, estimator_model::CompiledOpModel> op_models,
      std::optional<double> one_bit_register_area)
      : AreaEstimator(name),
        op_models_(std::move(op_models)),
        one_bit_register_area_(one_bit_register_area) {}

  static absl::StatusOr<std::unique_ptr<ProtoAreaEstimator>> Create(
      std::string_view name,
      const estimator_model::CompiledEstimatorModel& model);

  static absl::StatusOr<std::unique_ptr<ProtoAreaEstimator>>
  CreateFromProtoData(std::string_view name, absl::Span<const uint8_t> data);

  absl::StatusOr<double> GetOperationAreaInSquareMicrons(
      Node* node) const override;

  absl::StatusOr<double> GetOneBitRegisterAreaInSquareMicrons() const override;

 private:
  absl::StatusOr<double> GetOpAreaInSquareMicrons(Op op, Node* node) const;

  absl::flat_hash_map<Op, estimator_model::CompiledOpModel> op_models_;
  std::optional<double> one_bit_register_area_;
};

// Registers a ProtoAreaEstimator parsed from embedded binary proto data.
absl::Status RegisterProtobufAreaEstimator(std::string_view name,
                                           absl::Span<const uint8_t> data);

}  // namespace xls

#endif  // XLS_ESTIMATORS_AREA_MODEL_AREA_ESTIMATOR_H_
