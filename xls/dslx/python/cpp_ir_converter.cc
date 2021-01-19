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

  py::class_<IrConverter>(m, "IrConverter")
      .def(py::init([](PackageHolder package, ModuleHolder module,
                       const std::shared_ptr<TypeInfo>& type_info,
                       ImportCache* import_cache, bool emit_positions) {
        return absl::make_unique<IrConverter>(package.package(),
                                              &module.deref(), type_info,
                                              import_cache, emit_positions);
      }))
      .def("add_constant_dep",
           [](IrConverter& self, ConstantDefHolder constant_def) {
             self.AddConstantDep(&constant_def.deref());
           })
      .def("visit_function",
           [](IrConverter& self, FunctionHolder fn,
              absl::optional<const SymbolicBindings*> symbolic_bindings)
               -> absl::StatusOr<xls::FunctionHolder> {
             const SymbolicBindings* p =
                 symbolic_bindings ? *symbolic_bindings : nullptr;
             XLS_ASSIGN_OR_RETURN(xls::Function * f,
                                  self.VisitFunction(&fn.deref(), p));
             return xls::FunctionHolder(f, self.package());
           });

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
}

}  // namespace xls::dslx
