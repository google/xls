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

// Generate the signature for the top-level module.
xabsl::StatusOr<ModuleSignature> GenerateModuleSignature(
    const CountedFor* loop, const SequentialOptions& sequential_options) {
  std::string module_name = sequential_options.module_name().has_value()
                                ? sequential_options.module_name().value()
                                : loop->GetName() + "_sequential_module";
  ModuleSignatureBuilder sig_builder(module_name);

  // Default Inputs.
  sig_builder.WithClock("clk");
  for (const Node* op_in : loop->operands()) {
    sig_builder.AddDataInput(op_in->GetName() + "_in",
                             op_in->GetType()->GetFlatBitCount());
  }

  // Default Outputs.
  sig_builder.AddDataOutput(loop->GetName() + "_out",
                            loop->GetType()->GetFlatBitCount());

  // Reset.
  if (sequential_options.reset().has_value()) {
    sig_builder.WithReset(sequential_options.reset()->name(),
                          sequential_options.reset()->asynchronous(),
                          sequential_options.reset()->active_low());
  }

  // TODO(jbaileyhandle): Add options for other interfaces.
  std::string ready_in_name = "ready_in";
  std::string valid_in_name = "valid_in";
  std::string ready_out_name = "ready_out";
  std::string valid_out_name = "valid_out";
  sig_builder.WithReadyValidInterface(ready_in_name, valid_in_name,
                                      ready_out_name, valid_out_name);

  return sig_builder.Build();
}

// Generate a pipeline module that implements the loop's body.
xabsl::StatusOr<std::unique_ptr<ModuleGeneratorResult>>
GenerateLoopBodyPipeline(const CountedFor* loop,
                         const SequentialOptions& sequential_options,
                         const SchedulingOptions& scheduling_options,
                         const DelayEstimator& delay_estimator) {
  // Set pipeline options.
  PipelineOptions pipeline_options;
  pipeline_options.flop_inputs(false).flop_outputs(false).use_system_verilog(
      sequential_options.use_system_verilog());
  if (sequential_options.reset().has_value()) {
    pipeline_options.reset(sequential_options.reset().value());
  }

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
