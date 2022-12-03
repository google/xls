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

#include "xls/ir/package.h"

#include "pybind11/pybind11.h"
#include "pybind11_abseil/absl_casters.h"
#include "pybind11_abseil/statusor_caster.h"
#include "xls/common/status/import_status_module.h"
#include "xls/ir/function.h"
#include "xls/ir/python/wrapper_types.h"

namespace py = pybind11;

namespace xls {

PYBIND11_MODULE(package, m) {
  ImportStatusModule();
  py::module::import("xls.ir.python.type");
  py::module::import("xls.ir.python.fileno");
  py::module::import("xls.ir.python.function");

  py::class_<PackageHolder>(m, "Package")
      .def(py::init([](std::string_view name) {
             auto package = std::make_shared<Package>(name);
             return PackageHolder(package.get(), package);
           }),
           py::arg("name"))
      .def("dump_ir", PyWrap(&Package::DumpIr))
      .def("set_top_by_name", PyWrap(&Package::SetTopByName),
           py::arg("top_name"))
      .def("get_bits_type", PyWrap(&Package::GetBitsType), py::arg("bit_count"))
      .def("get_array_type", PyWrap(&Package::GetArrayType), py::arg("size"),
           py::arg("element"))
      .def(
          "get_tuple_type",
          [](PackageHolder* package,
             absl::Span<const TypeHolder> element_types) {
            // PyWrap doesn't support absl::Spans of pointers to wrapped
            // objects, need to wrap manually.
            std::vector<Type*> unwrapped_element_types;
            unwrapped_element_types.reserve(element_types.size());
            for (const auto& type_holder : element_types) {
              unwrapped_element_types.push_back(&type_holder.deref());
            }
            return TupleTypeHolder(
                package->deref().GetTupleType(unwrapped_element_types),
                package->package());
          },
          py::arg("element_types"))
      .def("get_or_create_fileno", PyWrap(&Package::GetOrCreateFileno),
           py::arg("filename"))
      .def("get_function", PyWrap(&Package::GetFunction), py::arg("func_name"))
      .def("get_function_names", PyWrap(&Package::GetFunctionNames));
}

}  // namespace xls
