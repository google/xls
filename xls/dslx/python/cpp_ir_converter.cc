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
      "convert_one_function",
      [](PackageHolder package, ModuleHolder module, FunctionHolder function,
         const std::shared_ptr<TypeInfo>& type_info,
         absl::optional<ImportCache*> py_import_cache,
         absl::optional<const SymbolicBindings*> py_symbolic_bindings,
         bool emit_positions) {
        const SymbolicBindings* symbolic_bindings =
            py_symbolic_bindings ? *py_symbolic_bindings : nullptr;
        ImportCache* import_cache =
            py_import_cache ? *py_import_cache : nullptr;
        return ConvertOneFunction(&package.deref(), &module.deref(),
                                  &function.deref(), type_info, import_cache,
                                  symbolic_bindings, emit_positions);
      },
      py::arg("package"), py::arg("module"), py::arg("function"),
      py::arg("type_info"), py::arg("import_cache"),
      py::arg("symbolic_bindings"), py::arg("emit_positions"));
}

}  // namespace xls::dslx
