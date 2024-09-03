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

#ifndef XLS_CODEGEN_PIPELINE_GENERATOR_H_
#define XLS_CODEGEN_PIPELINE_GENERATOR_H_


#include "absl/status/statusor.h"
#include "xls/codegen/codegen_options.h"
#include "xls/codegen/module_signature.h"
#include "xls/estimators/delay_model/delay_estimator.h"
#include "xls/ir/function.h"
#include "xls/ir/function_base.h"
#include "xls/ir/package.h"
#include "xls/scheduling/pipeline_schedule.h"

namespace xls {
namespace verilog {

// Class setting up default options passed to the pipeline generator.
inline CodegenOptions BuildPipelineOptions() {
  return CodegenOptions()
      .flop_inputs(true)
      .flop_outputs(true)
      .clock_name("clk")
      .emit_as_pipeline(true);
}

// Emits the given function as a verilog module which follows the given
// schedule. The module is pipelined with a latency and initiation interval
// given in the signature.
// If a delay estimator is provided, the signature also includes delay
// information about the pipeline stages.
absl::StatusOr<ModuleGeneratorResult> ToPipelineModuleText(
    const PipelineSchedule& schedule, Function* func,
    const CodegenOptions& options = BuildPipelineOptions(),
    const DelayEstimator* delay_estimator = nullptr);

// Emits the given function or proc as a verilog module which follows the given
// schedule. The module is pipelined with a latency and initiation interval
// given in the signature.
// If a delay estimator is provided, the signature also includes delay
// information about the pipeline stages.
absl::StatusOr<ModuleGeneratorResult> ToPipelineModuleText(
    const PipelineSchedule& schedule, FunctionBase* module,
    const CodegenOptions& options = BuildPipelineOptions(),
    const DelayEstimator* delay_estimator = nullptr);

// Emits the given package as a verilog module which follows the given
// schedules. Modules are pipelined with a latency and initiation interval
// given in the signature. If a delay estimator is provided, the signature also
// includes delay information about the pipeline stages.
absl::StatusOr<ModuleGeneratorResult> ToPipelineModuleText(
    const PackagePipelineSchedules& schedules, Package* package,
    const CodegenOptions& options = BuildPipelineOptions(),
    const DelayEstimator* delay_estimator = nullptr);

}  // namespace verilog
}  // namespace xls

#endif  // XLS_CODEGEN_PIPELINE_GENERATOR_H_
