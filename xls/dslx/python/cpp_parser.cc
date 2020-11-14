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
#include "pybind11/functional.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "xls/common/status/status_macros.h"
#include "xls/common/status/statusor_pybind_caster.h"
#include "xls/dslx/parser.h"
#include "xls/dslx/python/cpp_ast.h"
#include "xls/dslx/python/errors.h"

namespace py = pybind11;

namespace xls::dslx {

PYBIND11_MODULE(cpp_parser, m) {
  ImportStatusModule();
  py::module::import("xls.dslx.python.cpp_ast");
  py::module::import("xls.dslx.python.cpp_scanner");

  static py::exception<CppParseError> parse_exc(m, "CppParseError");
  py::register_exception_translator([](std::exception_ptr p) {
    try {
      if (p) std::rethrow_exception(p);
    } catch (const CppParseError& e) {
      py::object& e_type = parse_exc;
      py::object instance = e_type();
      instance.attr("message") = e.what();
      instance.attr("span") = e.span();
      PyErr_SetObject(parse_exc.ptr(), instance.ptr());
    }
  });

  m.def("throw_parse_error", [](Span span, const std::string& s) {
    std::string message = absl::StrFormat("%s @ %s", s, span.ToString());
    throw CppParseError(std::move(span), std::move(message));
  });

  py::class_<Parser>(m, "Parser")
      .def(py::init([](Scanner* scanner, std::string module_name) {
             return Parser(module_name, scanner);
           }),
           py::keep_alive<1, 2>())
      .def(
          "parse_module",
          [](Parser* self, Bindings* bindings) -> absl::StatusOr<ModuleHolder> {
            absl::StatusOr<std::shared_ptr<Module>> module =
                self->ParseModule(bindings);
            TryThrowCppParseError(module.status());
            XLS_RETURN_IF_ERROR(module.status());
            return ModuleHolder(module.value().get(), module.value());
          },
          py::arg("bindings") = nullptr)
      .def(
          "parse_function",
          [](Parser* self, Bindings* bindings,
             bool is_public) -> absl::StatusOr<FunctionHolder> {
            absl::StatusOr<Function*> f_or =
                self->ParseFunction(is_public, bindings);
            TryThrowCppParseError(f_or.status());
            XLS_RETURN_IF_ERROR(f_or.status());
            return FunctionHolder(f_or.value(), self->module());
          },
          py::arg("outer_bindings") = nullptr, py::arg("is_public") = false)
      .def(
          "parse_proc",
          [](Parser* self, Bindings* bindings,
             bool is_public) -> absl::StatusOr<ProcHolder> {
            absl::StatusOr<Proc*> p_or = self->ParseProc(is_public, bindings);
            TryThrowCppParseError(p_or.status());
            XLS_RETURN_IF_ERROR(p_or.status());
            return ProcHolder(p_or.value(), self->module());
          },
          py::arg("outer_bindings") = nullptr, py::arg("is_public") = false)
      .def("parse_expression",
           [](Parser* self, Bindings* bindings) -> absl::StatusOr<ExprHolder> {
             absl::StatusOr<Expr*> e_or = self->ParseExpression(bindings);
             TryThrowCppParseError(e_or.status());
             XLS_RETURN_IF_ERROR(e_or.status());
             return ExprHolder(e_or.value(), self->module());
           });

  py::class_<Bindings>(m, "Bindings")
      .def(py::init([](Bindings* parent) { return Bindings(parent); }),
           py::arg("parent") = nullptr)
      // Note: returns an AnyNameDef.
      .def("resolve",
           [](Bindings& self, ModuleHolder module, absl::string_view name,
              const Span& span) -> absl::StatusOr<AstNodeHolder> {
             absl::StatusOr<AnyNameDef> name_def =
                 self.ResolveNameOrError(name, span);
             TryThrowCppParseError(name_def.status());
             XLS_RETURN_IF_ERROR(name_def.status());
             return AstNodeHolder(ToAstNode(name_def.value()), module.module());
           })
      // Note: returns an arbitrary BoundNode.
      .def("resolve_node",
           [](Bindings& self, ModuleHolder module, absl::string_view name,
              const Span& span) -> absl::StatusOr<AstNodeHolder> {
             absl::StatusOr<BoundNode> bn = self.ResolveNodeOrError(name, span);
             TryThrowCppParseError(bn.status());
             XLS_RETURN_IF_ERROR(bn.status());
             return AstNodeHolder(ToAstNode(bn.value()), module.module());
           })
      .def("resolve_node_or_none",
           [](Bindings& self, ModuleHolder module,
              absl::string_view name) -> absl::optional<AstNodeHolder> {
             absl::optional<BoundNode> result = self.ResolveNode(name);
             if (!result) {
               return absl::nullopt;
             }
             return AstNodeHolder(ToAstNode(*result), module.module());
           })
      .def("add",
           [](Bindings& bindings, std::string name,
              AstNodeHolder binding) -> absl::Status {
             absl::StatusOr<BoundNode> bn = ToBoundNode(&binding.deref());
             TryThrowCppParseError(bn.status());
             XLS_RETURN_IF_ERROR(bn.status());
             bindings.Add(std::move(name), bn.value());
             return absl::OkStatus();
           });
}

}  // namespace xls::dslx
