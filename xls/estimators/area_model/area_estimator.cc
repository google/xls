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

#include "xls/estimators/area_model/area_estimator.h"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>

#include "absl/base/no_destructor.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "xls/common/status/status_macros.h"
#include "xls/estimators/estimator_model.pb.h"
#include "xls/estimators/proto_estimator_utils.h"
#include "xls/ir/function.h"
#include "xls/ir/function_base.h"
#include "xls/ir/node.h"
#include "xls/ir/nodes.h"
#include "xls/ir/op.h"

namespace xls {

absl::StatusOr<double> AreaEstimator::GetFunctionBaseAreaInSquareMicrons(
    FunctionBase* fb) const {
  double total_area_um2 = 0.0;

  for (Node* node : fb->nodes()) {
    XLS_ASSIGN_OR_RETURN(double area, GetOperationAreaInSquareMicrons(node));
    total_area_um2 += area;
  }
  return total_area_um2;
}

absl::StatusOr<double> AreaEstimator::GetRegisterAreaInSquareMicrons(
    const uint64_t& register_width) const {
  XLS_ASSIGN_OR_RETURN(const double one_bit_register_area,
                       GetOneBitRegisterAreaInSquareMicrons());
  return one_bit_register_area * static_cast<double>(register_width);
}

AreaEstimatorManager& GetAreaEstimatorManagerSingleton() {
  static absl::NoDestructor<AreaEstimatorManager> manager;
  return *manager;
}

absl::StatusOr<AreaEstimator*> AreaEstimatorManager::GetAreaEstimator(
    std::string_view name) const {
  if (!estimators_.contains(name)) {
    if (estimator_names_.empty()) {
      return absl::NotFoundError(absl::StrFormat(
          R"TXT(No area estimator found named "%s".

No estimators are registered.

This can be caused by a few different issues.

1) Was InitXls called?

   This needs be called early during binary startup or before running other xls
   code.

2) Were estimators linked into the binary?

   Estimators need to be linked in order to be available. The standard
   estimators are linked from '//xls/estimators/area_model/models'
   and are also exported as '//xls/estimators' and
   '//xls/public:passes_and_estimators'. At least one of these must
   be in the 'deps' tree to ensure that estimators are available.)TXT",
          name));
    }
    return absl::NotFoundError(absl::StrFormat(
        R"TXT(No area estimator found named \"%s\".

Available estimators: %s

It is possible you didn't link the estimator into the binary.

Do you have '//xls/estimators/area_model/models' in your dep tree?

Is the estimator library a dependency of '//.../models'?)TXT",
        name, absl::StrJoin(estimator_names_, ", ")));
  }
  return estimators_.at(name).get();
}

absl::Status AreaEstimatorManager::AddAreaEstimator(
    std::unique_ptr<AreaEstimator> estimator) {
  std::string name = estimator->name();
  if (estimators_.contains(name)) {
    return absl::InternalError(
        absl::StrFormat("Area estimator named %s already exists", name));
  }
  estimators_[name] = std::move(estimator);
  estimator_names_.push_back(name);
  std::sort(estimator_names_.begin(), estimator_names_.end());
  return absl::OkStatus();
}

/* static */ absl::StatusOr<std::unique_ptr<ProtoAreaEstimator>>
ProtoAreaEstimator::Create(
    std::string_view name,
    const estimator_model::CompiledEstimatorModel& model) {
  if (model.metric() != estimator_model::Metric::AREA_METRIC) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "CompiledEstimatorModel for area estimator '%s' does not have "
        "AREA_METRIC.",
        name));
  }
  absl::flat_hash_map<Op, estimator_model::CompiledOpModel> op_models;
  op_models.reserve(model.op_models_size());
  for (const auto& op_model : model.op_models()) {
    XLS_ASSIGN_OR_RETURN(Op op, EnumNameToOp(op_model.op()));
    op_models.emplace(op, op_model);
  }
  std::optional<double> one_bit_register_area;
  if (model.has_one_bit_register_area()) {
    one_bit_register_area = model.one_bit_register_area();
  }
  return std::make_unique<ProtoAreaEstimator>(name, std::move(op_models),
                                              one_bit_register_area);
}

/* static */ absl::StatusOr<std::unique_ptr<ProtoAreaEstimator>>
ProtoAreaEstimator::CreateFromProtoData(std::string_view name,
                                        absl::Span<const uint8_t> data) {
  estimator_model::CompiledEstimatorModel model;
  if (!model.ParseFromArray(data.data(), data.size())) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Failed to parse CompiledEstimatorModel proto for area model '%s'.",
        name));
  }
  return Create(name, model);
}

absl::StatusOr<double> ProtoAreaEstimator::GetOpAreaInSquareMicrons(
    Op op, Node* node) const {
  auto it = op_models_.find(op);
  if (it == op_models_.end()) {
    if (op == Op::kInvoke) {
      Function* to_apply = node->As<Invoke>()->to_apply();
      return GetFunctionBaseAreaInSquareMicrons(to_apply);
    }
    if (op == Op::kCountedFor) {
      CountedFor* counted_for = node->As<CountedFor>();
      Function* body = counted_for->body();
      XLS_ASSIGN_OR_RETURN(double body_area,
                           GetFunctionBaseAreaInSquareMicrons(body));
      return body_area * counted_for->trip_count();
    }
    return absl::UnimplementedError(
        absl::StrCat("Unhandled node for area estimation in area model '",
                     name(), "': ", node->ToStringWithOperandTypes()));
  }
  XLS_ASSIGN_OR_RETURN(const estimator_model::CompiledEstimator* estimator,
                       SelectEstimator(it->second, node));
  switch (estimator->estimator_case()) {
    case estimator_model::CompiledEstimator::kFixed:
      return static_cast<double>(estimator->fixed());
    case estimator_model::CompiledEstimator::kAliasOp: {
      XLS_ASSIGN_OR_RETURN(Op aliased_op, EnumNameToOp(estimator->alias_op()));
      return GetOpAreaInSquareMicrons(aliased_op, node);
    }
    case estimator_model::CompiledEstimator::kRegression:
      return EvalRegressionEstimator(estimator->regression(), node);
    case estimator_model::CompiledEstimator::kAreaRegression:
      return EvalAreaRegressionEstimator(estimator->area_regression(), node);
    case estimator_model::CompiledEstimator::kBoundingBox:
      return EvalBoundingBoxEstimator(estimator->bounding_box(), node,
                                      /*metric_name=*/"area");
    default:
      return absl::InvalidArgumentError(
          absl::StrCat("Unsupported CompiledEstimator for area model '", name(),
                       "': ", estimator->estimator_case()));
  }
}

absl::StatusOr<double> ProtoAreaEstimator::GetOperationAreaInSquareMicrons(
    Node* node) const {
  return GetOpAreaInSquareMicrons(node->op(), node);
}

absl::StatusOr<double>
ProtoAreaEstimator::GetOneBitRegisterAreaInSquareMicrons() const {
  if (one_bit_register_area_.has_value()) {
    return *one_bit_register_area_;
  }
  return absl::UnimplementedError(absl::StrFormat(
      "Area model \"%s\" does not have register area defined.", name()));
}

absl::Status RegisterProtobufAreaEstimator(std::string_view name,
                                           absl::Span<const uint8_t> data) {
  XLS_ASSIGN_OR_RETURN(std::unique_ptr<ProtoAreaEstimator> estimator,
                       ProtoAreaEstimator::CreateFromProtoData(name, data));
  return GetAreaEstimatorManagerSingleton().AddAreaEstimator(
      std::move(estimator));
}

}  // namespace xls
