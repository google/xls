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

#include "xls/interpreter/ir_interpreter.h"

#include "pybind11/pybind11.h"
#include "xls/common/python/absl_casters.h"
#include "xls/common/status/statusor_pybind_caster.h"
#include "xls/ir/python/wrapper_types.h"

namespace py = pybind11;

namespace xls {

PYBIND11_MODULE(ir_interpreter, m) {
  ImportStatusModule();

  // clang-format off
  py::module::import("xls.interpreter."
                     "python.ir_interpreter_stats");
  py::module::import("xls.ir.python.function");
  py::module::import("xls.ir.python.value");
  // clang-format on

  m.def("run_function_kwargs", PyWrap(&IrInterpreter::RunKwargs), py::arg("f"),
        py::arg("args"), py::arg("stats") = nullptr);
  m.def("run_function", PyWrap(&IrInterpreter::Run), py::arg("f"),
        py::arg("args"), py::arg("stats") = nullptr);
}

}  // namespace xls
