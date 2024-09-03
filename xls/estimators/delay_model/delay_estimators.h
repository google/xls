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

#ifndef XLS_ESTIMATORS_DELAY_MODEL_DELAY_ESTIMATORS_H_
#define XLS_ESTIMATORS_DELAY_MODEL_DELAY_ESTIMATORS_H_

#include <string_view>

#include "absl/status/statusor.h"
#include "xls/estimators/delay_model/delay_estimator.h"

namespace xls {

// Returns the registered delay estimator with the given name.
absl::StatusOr<DelayEstimator*> GetDelayEstimator(std::string_view name);

// Returns a reference to a singleton object which uses the "standard" delay
// estimation model.
// TODO(meheff): Remove this function and require users to specify the estimator
// explicitly.
const DelayEstimator& GetStandardDelayEstimator();

}  // namespace xls

#endif  // XLS_ESTIMATORS_DELAY_MODEL_DELAY_ESTIMATORS_H_
