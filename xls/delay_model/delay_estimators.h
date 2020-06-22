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

#ifndef XLS_DELAY_MODEL_DELAY_ESTIMATORS_H_
#define XLS_DELAY_MODEL_DELAY_ESTIMATORS_H_

#include <memory>
#include <string>
#include <vector>

#include "absl/strings/string_view.h"
#include "xls/common/status/statusor.h"
#include "xls/delay_model/delay_estimator.h"

namespace xls {

// Returns the registered delay estimator with the given name.
xabsl::StatusOr<DelayEstimator*> GetDelayEstimator(absl::string_view name);

// Returns a reference to a singleton object which uses the "standard" delay
// estimation model.
// TODO(meheff): Remove this function and require users to specify the estimator
// explicitly.
const DelayEstimator& GetStandardDelayEstimator();

}  // namespace xls

#endif  // XLS_DELAY_MODEL_DELAY_ESTIMATORS_H_
