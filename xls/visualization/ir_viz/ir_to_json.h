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

#ifndef XLS_VISUALIZATION_IR_VIZ_IR_TO_JSON_H_
#define XLS_VISUALIZATION_IR_VIZ_IR_TO_JSON_H_

#include <optional>
#include <string>
#include <string_view>

#include "absl/status/statusor.h"
#include "xls/scheduling/pipeline_schedule.h"

namespace xls {

// Returns a JSON representation of the given package for use by the
// visualizer. The JSON is based on the xls::viz::Package proto (see
// ir_to_json_test.cc for examples)
absl::StatusOr<std::string> IrToJson(
    Package* package, const DelayEstimator& delay_estimator,
    const PipelineSchedule* schedule = nullptr,
    std::optional<std::string_view> entry_name = std::nullopt);

}  // namespace xls

#endif  // XLS_VISUALIZATION_IR_VIZ_IR_TO_JSON_H_
