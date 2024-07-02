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

#ifndef XLS_SCHEDULING_EXTRACT_STAGE_H_
#define XLS_SCHEDULING_EXTRACT_STAGE_H_

#include "absl/status/statusor.h"
#include "xls/ir/function.h"
#include "xls/ir/function_base.h"
#include "xls/scheduling/pipeline_schedule.h"

namespace xls {

// Creates a new function containing only the nodes within the specified
// pipeline stage and new params/output nodes.
absl::StatusOr<Function*> ExtractStage(FunctionBase* src,
                                       const PipelineSchedule& schedule,
                                       int stage);

}  // namespace xls

#endif  // XLS_SCHEDULING_EXTRACT_STAGE_H_
