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

#include "xls/ir/llvm_ir_jit.h"

#include "pybind11/pybind11.h"
#include "xls/common/python/absl_casters.h"
#include "xls/common/status/statusor_pybind_caster.h"
#include "xls/ir/python/wrapper_types.h"

namespace py = pybind11;

namespace xls {

PYBIND11_MODULE(llvm_ir_jit, m) {
  py::module::import("xls.ir.python.function");
  py::module::import("xls.ir.python.value");

  m.def("llvm_ir_jit_run", PyWrap(&CreateAndRun),
        py::arg("f"), py::arg("args"));
  m.def("quickcheck_jit", PyWrap(&CreateAndQuickCheck), py::arg("f"),
        py::arg("seed"), py::arg("num_tests"));
}

}  // namespace xls
