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

#include "xls/codegen/pipeline_generator.h"

#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "pybind11/pybind11.h"
#include "pybind11_abseil/absl_casters.h"
#include "pybind11_abseil/statusor_caster.h"
#include "xls/codegen/module_signature.h"
#include "xls/common/status/import_status_module.h"
#include "xls/delay_model/delay_estimators.h"
#include "xls/ir/function.h"
#include "xls/ir/package.h"
#include "xls/ir/python/wrapper_types.h"
#include "xls/scheduling/pipeline_schedule.h"

namespace py = pybind11;

namespace xls {
namespace verilog {
namespace {

// Generates a pipeline with the given number of stages.
absl::StatusOr<ModuleGeneratorResult> GeneratePipelinedModuleWithNStages(
    Package* package, int64_t stages, std::string_view module_name) {
  XLS_ASSIGN_OR_RETURN(Function * f, package->GetTopAsFunction());
  XLS_ASSIGN_OR_RETURN(
      PipelineSchedule schedule,
      PipelineSchedule::Run(f, GetStandardDelayEstimator(),
                            SchedulingOptions().pipeline_stages(stages)));
  CodegenOptions options = BuildPipelineOptions();
  options.module_name(module_name);
  options.use_system_verilog(false);
  return ToPipelineModuleText(schedule, f, options);
}

// Generates a pipeline with the given clock period.
absl::StatusOr<ModuleGeneratorResult> GeneratePipelinedModuleWithClockPeriod(
    Package* package, int64_t clock_period_ps, std::string_view module_name) {
  XLS_ASSIGN_OR_RETURN(Function * f, package->GetTopAsFunction());
  XLS_ASSIGN_OR_RETURN(
      PipelineSchedule schedule,
      PipelineSchedule::Run(
          f, GetStandardDelayEstimator(),
          SchedulingOptions().clock_period_ps(clock_period_ps)));
  CodegenOptions options = BuildPipelineOptions();
  options.module_name(module_name);
  options.use_system_verilog(false);
  return ToPipelineModuleText(schedule, f, options);
}

}  // namespace

PYBIND11_MODULE(pipeline_generator, m) {
  ImportStatusModule();

  py::module::import("xls.codegen.python.module_signature");
  py::module::import("xls.ir.python.package");

  m.def("generate_pipelined_module_with_n_stages",
        PyWrap(&GeneratePipelinedModuleWithNStages), py::arg("package"),
        py::arg("stages"), py::arg("module_name"));

  m.def("generate_pipelined_module_with_clock_period",
        PyWrap(&GeneratePipelinedModuleWithClockPeriod), py::arg("package"),
        py::arg("clock_period_ps"), py::arg("module_name"));
}

}  // namespace verilog
}  // namespace xls
