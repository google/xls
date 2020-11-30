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

#include "xls/dslx/interpreter.h"

#include "absl/base/casts.h"
#include "absl/status/statusor.h"
#include "pybind11/functional.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "xls/common/status/status_macros.h"
#include "xls/common/status/statusor_pybind_caster.h"
#include "xls/dslx/python/callback_converters.h"
#include "xls/dslx/python/cpp_ast.h"
#include "xls/dslx/python/errors.h"
#include "xls/ir/python/wrapper_types.h"

namespace py = pybind11;

namespace xls::dslx {

template <typename K, typename V>
absl::flat_hash_map<K, V> ToAbsl(const std::unordered_map<K, V>& m) {
  return absl::flat_hash_map<K, V>(m.begin(), m.end());
}

PYBIND11_MODULE(interpreter, m) {
  ImportStatusModule();
  py::module::import("xls.dslx.python.cpp_ast");

  static py::exception<FailureError> failure_exc(m, "FailureError");

  py::register_exception_translator([](std::exception_ptr p) {
    try {
      if (p) std::rethrow_exception(p);
    } catch (const FailureError& e) {
      py::object& e_type = failure_exc;
      py::object instance = e_type();
      instance.attr("message") = e.what();
      instance.attr("span") = e.span();
      PyErr_SetObject(failure_exc.ptr(), instance.ptr());
    }
  });

  m.def("throw_fail_error",
        [](Span span, const std::string& s) { throw FailureError(s, span); });

  py::class_<Interpreter>(m, "Interpreter")
      .def(py::init([](ModuleHolder module,
                       const std::shared_ptr<TypeInfo>& type_info,
                       PyTypecheckFn typecheck, ImportCache* import_cache,
                       bool trace_all,
                       absl::optional<PackageHolder> ir_package) {
             Package* package = nullptr;
             if (ir_package.has_value()) {
               package = &ir_package->deref();
             }
             return absl::make_unique<Interpreter>(
                 &module.deref(), type_info, ToCppTypecheck(typecheck),
                 import_cache, trace_all, package);
           }),
           py::arg("module"), py::arg("type_info"), py::arg("typecheck"),
           py::arg("import_cache"), py::arg("trace_all") = false,
           py::arg("ir_package") = absl::nullopt)
      .def(
          "run_function",
          [](Interpreter* self, absl::string_view name,
             const std::vector<InterpValue>& args) {
            auto statusor = self->RunFunction(name, args);
            TryThrowFailureError(statusor.status());
            return statusor;
          },
          py::arg("name"), py::arg("args"))
      .def("run_test",
           [](Interpreter* self, absl::string_view name) {
             absl::Status result = self->RunTest(name);
             TryThrowFailureError(result);
             return result;
           })
      .def_property_readonly("module",
                             [](Interpreter* self) {
                               return ModuleHolder(
                                   self->module(),
                                   self->module()->shared_from_this());
                             })
      .def_static(
          "interpret_expr",
          [](ModuleHolder module, const std::shared_ptr<TypeInfo>& type_info,
             PyTypecheckFn py_typecheck, ImportCache* import_cache,
             const std::unordered_map<std::string, int64>& env,
             const std::unordered_map<std::string, int64>& bit_widths,
             ExprHolder expr, const FnCtx& fn_ctx) {
            auto statusor = Interpreter::InterpretExpr(
                &module.deref(), type_info, ToCppTypecheck(py_typecheck),
                import_cache, ToAbsl(env), ToAbsl(bit_widths), &expr.deref(),
                fn_ctx);
            TryThrowKeyError(statusor.status());
            return statusor;
          },
          py::arg("module"), py::arg("type_info"), py::arg("typecheck"),
          py::arg("import_cache"), py::arg("env"), py::arg("bit_widths"),
          py::arg("expr"), py::arg("fn_ctx"));
}

}  // namespace xls::dslx
