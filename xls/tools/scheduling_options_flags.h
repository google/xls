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

#include <optional>
#include <string>

#include "absl/flags/declare.h"
#include "absl/status/statusor.h"
#include "xls/estimators/delay_model/delay_estimator.h"
#include "xls/fdo/synthesizer.h"
#include "xls/ir/node.h"
#include "xls/scheduling/scheduling_options.h"
#include "xls/tools/scheduling_options_flags.pb.h"

ABSL_DECLARE_FLAG(std::optional<std::string>,
                  scheduling_options_used_textproto_file);

namespace xls {

absl::StatusOr<SchedulingOptionsFlagsProto> GetSchedulingOptionsFlagsProto();

absl::StatusOr<SchedulingOptions> SetUpSchedulingOptions(
    const SchedulingOptionsFlagsProto& flags, Package* p);

absl::StatusOr<DelayEstimator*> SetUpDelayEstimator(
    const SchedulingOptionsFlagsProto& flags);
absl::StatusOr<bool> IsDelayModelSpecifiedViaFlag(
    const SchedulingOptionsFlagsProto& flags);
absl::StatusOr<synthesis::Synthesizer*> SetUpSynthesizer(
    const SchedulingOptions& flags);

}  // namespace xls

#endif  // XLS_TOOLS_SCHEDULING_OPTIONS_FLAGS_H_
