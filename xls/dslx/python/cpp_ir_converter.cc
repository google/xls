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

#include "absl/base/casts.h"
#include "absl/status/statusor.h"
#include "absl/strings/ascii.h"
#include "pybind11/functional.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "xls/common/status/status_macros.h"
#include "xls/common/status/statusor_pybind_caster.h"
#include "xls/dslx/concrete_type.h"
#include "xls/dslx/cpp_extract_conversion_order.h"
#include "xls/dslx/deduce.h"
#include "xls/dslx/ir_converter.h"
#include "xls/dslx/python/cpp_ast.h"
#include "xls/dslx/python/errors.h"
#include "xls/ir/python/wrapper_types.h"

namespace py = pybind11;

namespace xls::dslx {

PYBIND11_MODULE(cpp_ir_converter, m) {
  ImportStatusModule();
  py::module::import("xls.ir.python.package");
  py::module::import("xls.ir.python.function_builder");
  py::module::import("xls.dslx.python.cpp_ast");
  py::module::import("xls.dslx.python.interp_value");

  m.def("mangle_dslx_name",
        [](absl::string_view function_name,
           const std::set<std::string>& free_keys, ModuleHolder m,
           absl::optional<SymbolicBindings> symbolic_bindings) {
          return MangleDslxName(
              function_name,
              absl::btree_set<std::string>(free_keys.begin(), free_keys.end()),
              &m.deref(),
              symbolic_bindings.has_value() && !symbolic_bindings->empty()
                  ? &*symbolic_bindings
                  : nullptr);
        });

  m.def(
      "convert_module_to_package",
      [](ModuleHolder module, const std::shared_ptr<TypeInfo>& type_info,
         ImportCache* import_cache, bool emit_positions,
         bool traverse_tests) -> absl::StatusOr<PackageHolder> {
        XLS_ASSIGN_OR_RETURN(
            std::unique_ptr<Package> package,
            ConvertModuleToPackage(&module.deref(), type_info, import_cache,
                                   emit_positions, traverse_tests));
        std::shared_ptr<Package> shared(std::move(package));
        return PackageHolder(shared.get(), std::move(shared));
      },
      py::arg("module"), py::arg("type_info"), py::arg("import_cacche"),
      py::arg("emit_positions") = true, py::arg("traverse_tests") = false);

  m.def(
      "convert_module",
      [](ModuleHolder module, const std::shared_ptr<TypeInfo>& type_info,
         absl::optional<ImportCache*> py_import_cache, bool emit_positions) {
        ImportCache* import_cache =
            py_import_cache ? *py_import_cache : nullptr;
        return ConvertModule(&module.deref(), type_info, import_cache,
                             emit_positions);
      },
      py::arg("module"), py::arg("type_info"),
      py::arg("import_cache") = absl::nullopt,
      py::arg("emit_positions") = true);

  m.def(
      "convert_one_function",
      [](ModuleHolder module, absl::string_view entry_function_name,
         const std::shared_ptr<TypeInfo>& type_info,
         absl::optional<ImportCache*> py_import_cache, bool emit_positions) {
        ImportCache* import_cache =
            py_import_cache ? *py_import_cache : nullptr;
        SymbolicBindings empty;
        return ConvertOneFunction(&module.deref(), entry_function_name,
                                  type_info, import_cache, &empty,
                                  emit_positions);
      },
      py::arg("module"), py::arg("entry_function_name"), py::arg("type_info"),
      py::arg("import_cache"), py::arg("emit_positions") = true);

  py::class_<ConversionRecord>(m, "ConversionRecord")
      .def_property_readonly("f",
                             [](ConversionRecord& self) -> FunctionHolder {
                               return FunctionHolder(
                                   self.f, self.f->owner()->shared_from_this());
                             })
      .def_property_readonly("m",
                             [](ConversionRecord& self) -> ModuleHolder {
                               return ModuleHolder(self.m,
                                                   self.m->shared_from_this());
                             })
      .def_property_readonly(
          "type_info",
          [](ConversionRecord& self) -> std::shared_ptr<TypeInfo> {
            return self.type_info;
          })
      .def_property_readonly("bindings",
                             [](ConversionRecord& self) -> SymbolicBindings {
                               return self.bindings;
                             });

  m.def(
      "get_conversion_order",
      [](ModuleHolder module, const std::shared_ptr<TypeInfo>& type_info,
         bool traverse_tests) {
        return GetOrder(&module.deref(), type_info, traverse_tests);
      },
      py::arg("module"), py::arg("type_info"),
      py::arg("traverse_tests") = false);
}

}  // namespace xls::dslx
