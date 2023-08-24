// Copyright 2022 The XLS Authors
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

// Dead Code Elimination.
//
#ifndef XLS_TOOLS_SCHEDULING_OPTIONS_FLAGS_H_
#define XLS_TOOLS_SCHEDULING_OPTIONS_FLAGS_H_

#include "absl/status/statusor.h"
#include "xls/delay_model/delay_estimator.h"
#include "xls/fdo/synthesizer.h"
#include "xls/ir/node.h"
#include "xls/scheduling/scheduling_options.h"

namespace xls {

// If you don't have a `Package` at hand, you can pass in `nullptr` and it will
// skip some checks.
absl::StatusOr<SchedulingOptions> SetUpSchedulingOptions(Package* p);
absl::StatusOr<DelayEstimator*> SetUpDelayEstimator();
absl::StatusOr<bool> IsDelayModelSpecifiedViaFlag();
absl::StatusOr<synthesis::Synthesizer*> SetUpSynthesizer();

}  // namespace xls

#endif  // XLS_TOOLS_SCHEDULING_OPTIONS_FLAGS_H_
