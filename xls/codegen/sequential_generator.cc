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

#include "xls/codegen/sequential_generator.h"

#include <memory>
#include <utility>

#include "absl/status/status.h"
#include "absl/types/optional.h"
#include "xls/codegen/module_signature.h"
#include "xls/codegen/module_signature.pb.h"
#include "xls/codegen/pipeline_generator.h"
#include "xls/common/integral_types.h"
#include "xls/common/status/status_macros.h"
#include "xls/common/status/statusor.h"
#include "xls/delay_model/delay_estimator.h"
#include "xls/delay_model/delay_estimators.h"
#include "xls/ir/function.h"
#include "xls/scheduling/pipeline_schedule.h"

namespace xls {
namespace verilog {

xabsl::StatusOr<std::unique_ptr<ModuleGeneratorResult>>
GenerateLoopBodyPipeline(CountedFor* loop, bool use_system_verilog,
                         SchedulingOptions& scheduling_options,
                         const DelayEstimator& delay_estimator) {
  PipelineOptions pipeline_options;
  pipeline_options.flop_inputs(false).flop_outputs(false).use_system_verilog(
      use_system_verilog);

  // Get schedule.
  std::unique_ptr<ModuleGeneratorResult> result =
      std::make_unique<ModuleGeneratorResult>();
  Function* loop_body_function = loop->body();
  XLS_ASSIGN_OR_RETURN(
      PipelineSchedule schedule,
      PipelineSchedule::Run(loop_body_function, delay_estimator,
                            scheduling_options));
  XLS_RETURN_IF_ERROR(schedule.Verify());

  // Get pipeline module.
  XLS_ASSIGN_OR_RETURN(
      *result,
      ToPipelineModuleText(schedule, loop_body_function, pipeline_options));
  return std::move(result);
}

xabsl::StatusOr<ModuleGeneratorResult> ToSequentialModuleText(Function* func) {
  return absl::UnimplementedError("Sequential generator not supported yet.");
}

}  // namespace verilog
}  // namespace xls
