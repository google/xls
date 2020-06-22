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

#include "xls/delay_model/delay_estimators.h"

#include "absl/flags/flag.h"

namespace xls {

xabsl::StatusOr<DelayEstimator*> GetDelayEstimator(absl::string_view name) {
  return GetDelayEstimatorManagerSingleton().GetDelayEstimator(name);
}

const DelayEstimator& GetStandardDelayEstimator() {
  return *GetDelayEstimatorManagerSingleton()
              .GetDefaultDelayEstimator()
              .value();
}

}  // namespace xls
