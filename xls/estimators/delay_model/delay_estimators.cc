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

#include "xls/estimators/delay_model/delay_estimators.h"

#include <cstdint>
#include <string_view>

#include "absl/status/statusor.h"
#include "xls/estimators/delay_model/delay_estimator.h"
#include "xls/ir/function.h"
#include "xls/ir/node.h"
#include "xls/ir/nodes.h"

namespace xls {

absl::StatusOr<DelayEstimator*> GetDelayEstimator(std::string_view name) {
  return GetDelayEstimatorManagerSingleton().GetDelayEstimator(name);
}

const DelayEstimator& GetStandardDelayEstimator() {
  return *GetDelayEstimatorManagerSingleton()
              .GetDefaultDelayEstimator()
              .value();
}

namespace delay_adapters {
absl::StatusOr<int64_t> FilterNonSynth::GetOperationDelayInPs(
    Node* node) const {
  if (node->Is<Invoke>() && node->As<Invoke>()->non_synth()) {
    return 0;
  }
  if (node->function_base()->IsFunction() &&
      node->function_base()->AsFunctionOrDie()->non_synth()) {
    return 0;
  }
  return decorated_.GetOperationDelayInPs(node);
}
}  // namespace delay_adapters
}  // namespace xls
