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

#include "xls/codegen/pipeline_generator.h"

#include "pybind11/pybind11.h"
#include "xls/codegen/module_signature.h"
#include "xls/common/status/statusor.h"
#include "xls/common/status/statusor_pybind_caster.h"
#include "xls/ir/package.h"
#include "xls/ir/python/wrapper_types.h"

namespace py = pybind11;

namespace xls {
namespace verilog {
namespace {

xabsl::StatusOr<ModuleGeneratorResult>
ScheduleAndGeneratePipelinedModuleNoOptionalParams(Package* package,
                                                   int64 clock_period_ps) {
  // Don't expose optional parameters to pybind11.
  return ScheduleAndGeneratePipelinedModule(package, clock_period_ps);
}

}  // namespace

PYBIND11_MODULE(pipeline_generator, m) {
  py::module::import("xls.codegen.python.module_signature");
  py::module::import("xls.ir.python.package");

  m.def("schedule_and_generate_pipelined_module",
        PyWrap(&ScheduleAndGeneratePipelinedModuleNoOptionalParams),
        py::arg("package"), py::arg("clock_period_ps"));
}

}  // namespace verilog
}  // namespace xls
