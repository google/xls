// Copyright 2023 The XLS Authors
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

#ifndef XLS_VISUALIZATION_IR_VIZ_IR_TO_PROTO_H_
#define XLS_VISUALIZATION_IR_VIZ_IR_TO_PROTO_H_

#include <optional>
#include <string_view>

#include "absl/status/statusor.h"
#include "xls/estimators/area_model/area_estimator.h"
#include "xls/estimators/delay_model/delay_estimator.h"
#include "xls/ir/package.h"
#include "xls/scheduling/pipeline_schedule.h"
#include "xls/visualization/ir_viz/visualization.pb.h"

namespace xls {

// Returns a xls::viz::Package proto representation of the given package for use
// by a visualizer.
absl::StatusOr<xls::viz::Package> IrToProto(
    Package* package, const DelayEstimator& delay_estimator,
    const AreaEstimator& area_estimator,
    const PipelineSchedule* schedule = nullptr,
    std::optional<std::string_view> entry_name = std::nullopt,
    bool token_dag = false);

// Returns a proto without any area information.
absl::StatusOr<xls::viz::Package> IrToProto(
    Package* package, const DelayEstimator& delay_estimator,
    const PipelineSchedule* schedule = nullptr,
    std::optional<std::string_view> entry_name = std::nullopt,
    bool token_dag = false);
}  // namespace xls

#endif  // XLS_VISUALIZATION_IR_VIZ_IR_TO_PROTO_H_
