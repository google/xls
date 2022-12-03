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

#include "xls/passes/standard_pipeline.h"

#include "pybind11/pybind11.h"
#include "pybind11_abseil/statusor_caster.h"
#include "xls/common/status/import_status_module.h"
#include "xls/ir/python/wrapper_types.h"

namespace py = pybind11;

namespace xls {

absl::StatusOr<bool> RunPipelineAtMaxOptLevel(Package *p) {
  return RunStandardPassPipeline(p, kMaxOptLevel);
}

PYBIND11_MODULE(standard_pipeline, m) {
  ImportStatusModule();

  py::module::import("xls.ir.python.package");

  m.def("run_standard_pass_pipeline", PyWrap(&RunPipelineAtMaxOptLevel),
        py::arg("package"));
}

}  // namespace xls
