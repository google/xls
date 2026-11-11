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

#ifndef XLS_ESTIMATORS_PROTO_ESTIMATOR_UTILS_H_
#define XLS_ESTIMATORS_PROTO_ESTIMATOR_UTILS_H_

#include <string_view>

#include "absl/status/statusor.h"
#include "xls/estimators/estimator_model.pb.h"
#include "xls/ir/node.h"
#include "xls/ir/op.h"

namespace xls {

// Converts an Op enum identifier string (e.g., "kAdd") into an xls::Op.
absl::StatusOr<Op> EnumNameToOp(std::string_view enum_name);

// Selects the appropriate CompiledEstimator for the given node from op_model
// (checking specializations in order before falling back to the default).
absl::StatusOr<const estimator_model::CompiledEstimator*> SelectEstimator(
    const estimator_model::CompiledOpModel& op_model, Node* node);

// Evaluates a pre-fitted Regression estimator on the given IR node.
absl::StatusOr<double> EvalRegressionEstimator(
    const estimator_model::CompiledEstimator::Regression& reg, Node* node);

// Evaluates a pre-fitted AreaRegression estimator on the given IR node.
absl::StatusOr<double> EvalAreaRegressionEstimator(
    const estimator_model::CompiledEstimator::AreaRegression& reg, Node* node);

// Evaluates a BoundingBox estimator on the given IR node.
absl::StatusOr<double> EvalBoundingBoxEstimator(
    const estimator_model::CompiledEstimator::BoundingBox& bbox, Node* node,
    std::string_view metric_name);

}  // namespace xls

#endif  // XLS_ESTIMATORS_PROTO_ESTIMATOR_UTILS_H_
