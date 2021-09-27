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

#include "xls/ir/function.h"

#include "pybind11/pybind11.h"
#include "xls/ir/function_base.h"
#include "xls/ir/python/wrapper_types.h"

namespace py = pybind11;

namespace xls {

// Wrapper of method defined on FunctionBase for use by Function* objects.
std::string GetFunctionName(Function* f) { return f->name(); }

// Functions for accessing parameter type information.
int64_t GetParamCount(Function* f) { return f->params().size(); }
int64_t GetParamBitCount(Function* f, int64_t i) {
  return f->param(i)->GetType()->GetFlatBitCount();
}

PYBIND11_MODULE(function, m) {
  py::module::import("xls.ir.python.type");

  py::class_<FunctionBaseHolder>(m, "FunctionBase")
      .def("dump_ir", PyWrap(&FunctionBase::DumpIr))
      .def_property_readonly("name", PyWrap(&Function::name));

  py::class_<FunctionHolder>(m, "Function")
      .def("dump_ir", PyWrap(&Function::DumpIr))
      // TODO(meheff): Figure out how to add a get_type method on Function.
      // Attempts to do so fail with "unable to convert to python type" errors.
      .def("get_param_count", PyWrap(&GetParamCount))
      .def("get_param_bit_count", PyWrap(&GetParamBitCount),
           py::arg("param_no"))
      .def_property_readonly("name", PyWrap(&GetFunctionName));
}

}  // namespace xls
