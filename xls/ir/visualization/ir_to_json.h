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

#ifndef THIRD_PARTY_XLS_IR_VISUALIZATION_IR_TO_JSON_H_
#define THIRD_PARTY_XLS_IR_VISUALIZATION_IR_TO_JSON_H_

#include <string>

#include "absl/strings/string_view.h"
#include "xls/common/status/statusor.h"
#include "xls/scheduling/pipeline_schedule.h"

namespace xls {

// Parses the given IR text of a package and returns a JSON representation of
// the graph representation is generic (see ir_to_json_test.cc for examples) and
// the client is responsible for constructing the appropriate representation for
// the web view.
xabsl::StatusOr<std::string> IrToJson(
    Function* function, const DelayEstimator& delay_estimator,
    const PipelineSchedule* schedule = nullptr);

}  // namespace xls

#endif  // THIRD_PARTY_XLS_IR_VISUALIZATION_IR_TO_JSON_H_
