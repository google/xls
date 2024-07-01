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

#include "xls/contrib/integrator/area_model/area_estimator.h"

#include <memory>
#include <string_view>

#include "absl/status/statusor.h"
#include "xls/common/status/status_macros.h"
#include "xls/delay_model/delay_estimator.h"
#include "xls/delay_model/delay_estimator.h"
#include "xls/delay_model/delay_estimator.h"

namespace xls {

absl::StatusOr<std::unique_ptr<AreaEstimator>> GetAreaEstimatorByName(
    std::string_view name) {
  DelayEstimatorManager& singleton = GetDelayEstimatorManagerSingleton();
  XLS_ASSIGN_OR_RETURN(DelayEstimator * delay_estimator,
                       singleton.GetDelayEstimator(name));
  return std::make_unique<AreaEstimator>(delay_estimator);
}

}  // namespace xls
