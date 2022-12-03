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

#include "xls/ir/type.h"

#include "google/protobuf/text_format.h"
#include "pybind11/pybind11.h"
#include "xls/ir/python/wrapper_types.h"

namespace py = pybind11;

namespace xls {

PYBIND11_MODULE(type, m) {
  auto type = py::class_<TypeHolder>(m, "Type");
  type.def("__str__", PyWrap(&Type::ToString));

  py::class_<BitsTypeHolder>(m, "BitsType", type)
      .def("get_bit_count", PyWrap(&BitsType::bit_count));

  py::class_<ArrayTypeHolder>(m, "ArrayType", type)
      .def("get_size", PyWrap(&ArrayType::size))
      .def("get_element_type", PyWrap(&ArrayType::element_type));

  py::class_<TupleTypeHolder>(m, "TupleType", type);  // NOLINT

  py::class_<FunctionTypeHolder>(m, "FunctionType")
      .def("__str__", PyWrap(&FunctionType::ToString))
      .def("return_type", PyWrap(&FunctionType::return_type))
      .def("get_parameter_count", PyWrap(&FunctionType::parameter_count))
      .def("get_parameter_type", PyWrap(&FunctionType::parameter_type),
           py::arg("i"))
      .def("to_textproto", [](const FunctionTypeHolder& type) {
        std::string output;
        google::protobuf::TextFormat::PrintToString(type.deref().ToProto(), &output);
        return output;
      });
}

}  // namespace xls
