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

#include "xls/ir/value.h"

#include "pybind11/pybind11.h"
#include "pybind11_abseil/absl_casters.h"
#include "pybind11_abseil/statusor_caster.h"
#include "xls/common/status/import_status_module.h"

namespace py = pybind11;

namespace xls {

PYBIND11_MODULE(value, m) {
  ImportStatusModule();
  py::module::import("xls.ir.python.bits");

  py::class_<Value>(m, "Value")
      .def(py::init<Bits>(), py::arg("bits"))
      .def("__eq__", &Value::operator==)
      .def("__ne__", &Value::operator!=)

      .def("is_bits", &Value::IsBits)
      .def("is_array", &Value::IsArray)
      .def("is_tuple", &Value::IsTuple)
      .def("get_bits", &Value::GetBitsWithStatus)

      .def("get_elements", &Value::GetElements)

      .def("__str__", [](const Value& v) { return v.ToHumanString(); })
      .def("to_str", [](const Value& v) { return v.ToString(); })

      .def_static("make_array", &Value::Array, py::arg("elements"))
      .def_static("make_tuple", &Value::Tuple, py::arg("elements"));
}

}  // namespace xls
