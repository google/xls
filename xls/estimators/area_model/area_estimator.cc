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
#include <string>
#include <string_view>
#include <utility>

#include "absl/base/no_destructor.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "xls/common/status/status_macros.h"

namespace xls {

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
      return absl::NotFoundError(
          absl::StrFormat("No area estimator found named \"%s\". No "
                          "estimators are registered. Was InitXls called?",
                          name));
    }
    return absl::NotFoundError(absl::StrFormat(
        "No area estimator found named \"%s\". Available estimators: %s", name,
        absl::StrJoin(estimator_names_, ", ")));
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

}  // namespace xls
