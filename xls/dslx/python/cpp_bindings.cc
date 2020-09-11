// Copyright 2020 Google LLC
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

#include "xls/dslx/cpp_bindings.h"

#include "absl/strings/match.h"
#include "absl/strings/str_split.h"
#include "absl/strings/strip.h"
#include "pybind11/functional.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "xls/common/status/status_macros.h"
#include "xls/common/status/statusor_pybind_caster.h"
#include "xls/dslx/python/cpp_ast.h"

namespace py = pybind11;

namespace xls::dslx {

class CppParseError : public std::exception {
 public:
  CppParseError(Span span, std::string message)
      : span_(std::move(span)), message_(std::move(message)) {}

  const char* what() const noexcept override { return message_.c_str(); }

  const Span& span() const { return span_; }

 private:
  Span span_;
  std::string message_;
};

void TryThrowCppParseError(const absl::Status& status) {
  absl::string_view s = status.message();
  if (absl::ConsumePrefix(&s, "ParseError: ")) {
    std::vector<absl::string_view> pieces =
        absl::StrSplit(s, absl::MaxSplits(" ", 1));
    if (pieces.size() < 2) {
      return;
    }
    xabsl::StatusOr<Span> pos = Span::FromString(pieces[0]);
    throw CppParseError(std::move(pos.value()), std::string(pieces[1]));
  }
}

PYBIND11_MODULE(cpp_bindings, m) {
  py::module::import("xls.dslx.python.cpp_ast");

  py::register_exception<CppParseError>(m, "CppParseError");

  m.def("get_parse_error_span",
        [](const CppParseError& e) { return e.span(); });

  // class Bindings
  py::class_<Bindings>(m, "Bindings")
      .def(py::init([](ModuleHolder module, absl::optional<Bindings*> parent) {
             return Bindings(module.module(),
                             parent.has_value() ? *parent : nullptr);
           }),
           py::arg("module"), py::arg("parent") = absl::nullopt)
      .def("add",
           [](Bindings& bindings, std::string name,
              AstNodeHolder binding) -> absl::Status {
             xabsl::StatusOr<BoundNode> bn = ToBoundNode(&binding.deref());
             TryThrowCppParseError(bn.status());
             XLS_RETURN_IF_ERROR(bn.status());
             bindings.Add(std::move(name), bn.value());
             return absl::OkStatus();
           })
      // Note: returns an AnyNameDef.
      .def("resolve",
           [](Bindings& self, absl::string_view name,
              const Span& span) -> xabsl::StatusOr<AstNodeHolder> {
             xabsl::StatusOr<AnyNameDef> name_def =
                 self.ResolveNameOrError(name, span);
             TryThrowCppParseError(name_def.status());
             XLS_RETURN_IF_ERROR(name_def.status());
             return AstNodeHolder(ToAstNode(name_def.value()), self.module());
           })
      // Note: returns an AnyNameDef.
      .def("resolve_or_none",
           [](Bindings& self,
              absl::string_view name) -> absl::optional<AstNodeHolder> {
             absl::optional<AnyNameDef> name_def =
                 self.ResolveNameOrNullopt(name);
             if (!name_def.has_value()) {
               return absl::nullopt;
             }
             return AstNodeHolder(ToAstNode(*name_def), self.module());
           })
      // Note: returns an arbitrary BoundNode.
      .def("resolve_node",
           [](Bindings& self, absl::string_view name,
              const Span& span) -> xabsl::StatusOr<AstNodeHolder> {
             xabsl::StatusOr<BoundNode> bn =
                 self.ResolveNodeOrError(name, span);
             TryThrowCppParseError(bn.status());
             XLS_RETURN_IF_ERROR(bn.status());
             return AstNodeHolder(ToAstNode(bn.value()), self.module());
           })
      .def("resolve_node_or_none",
           [](Bindings& self,
              absl::string_view name) -> absl::optional<AstNodeHolder> {
             absl::optional<BoundNode> result = self.ResolveNode(name);
             if (!result) {
               return absl::nullopt;
             }
             return AstNodeHolder(ToAstNode(*result), self.module());
           })
      .def("has_local_bindings", &Bindings::HasLocalBindings)
      .def("has_name", &Bindings::HasName)
      .def_property_readonly("module", [](Bindings& self) {
        return ModuleHolder(self.module().get(), self.module());
      });
}

}  // namespace xls::dslx
