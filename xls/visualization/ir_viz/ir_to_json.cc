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

#include "xls/visualization/ir_viz/ir_to_json.h"

#include <optional>
#include <string>
#include <string_view>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "google/protobuf/util/json_util.h"
#include "xls/common/status/status_macros.h"
#include "xls/estimators/delay_model/delay_estimator.h"
#include "xls/ir/package.h"
#include "xls/scheduling/pipeline_schedule.h"
#include "xls/visualization/ir_viz/ir_to_proto.h"
#include "xls/visualization/ir_viz/visualization.pb.h"

namespace xls {

absl::StatusOr<std::string> IrToJson(
    Package* package, const DelayEstimator& delay_estimator,
    const PipelineSchedule* schedule,
    std::optional<std::string_view> entry_name) {
  XLS_ASSIGN_OR_RETURN(viz::Package proto, IrToProto(package, delay_estimator,
                                                     schedule, entry_name));

  std::string serialized_json;
  google::protobuf::util::JsonPrintOptions print_options;
  print_options.add_whitespace = true;
  print_options.preserve_proto_field_names = true;

  auto status =
      google::protobuf::util::MessageToJsonString(proto, &serialized_json, print_options);
  if (!status.ok()) {
    return absl::InternalError(std::string{status.message()});
  }
  return serialized_json;
}

}  // namespace xls
