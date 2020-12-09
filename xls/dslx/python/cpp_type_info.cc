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
#include "pybind11/functional.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "xls/common/status/status_macros.h"
#include "xls/common/status/statusor_pybind_caster.h"
#include "xls/dslx/python/cpp_ast.h"
#include "xls/dslx/python/errors.h"
#include "xls/dslx/type_info.h"

namespace py = pybind11;

namespace xls::dslx {

PYBIND11_MODULE(cpp_type_info, m) {
  ImportStatusModule();

  py::class_<SymbolicBinding>(m, "SymbolicBinding")
      .def("__getitem__",
           [](const SymbolicBinding& self,
              int64 i) -> absl::variant<std::string, int64> {
             switch (i) {
               case 0:
                 return self.identifier;
               case 1:
                 return self.value;
               default:
                 throw py::index_error("Index out of bounds");
             }
           })
      .def_property_readonly(
          "identifier",
          [](const SymbolicBinding& self) { return self.identifier; })
      .def_property_readonly(
          "value", [](const SymbolicBinding& self) { return self.value; });

  py::class_<SymbolicBindings>(m, "SymbolicBindings")
      .def(py::init<>())
      .def(
          py::init([](const std::vector<std::pair<std::string, int64>>& items) {
            return SymbolicBindings(items);
          }))
      .def("bindings", &SymbolicBindings::bindings)
      .def("to_dict",
           [](const SymbolicBindings& self) {
             std::unordered_map<std::string, int64> result;
             for (const SymbolicBinding& b : self.bindings()) {
               result[b.identifier] = b.value;
             }
             return result;
           })
      .def("__eq__", &SymbolicBindings::operator==)
      .def("__ne__", &SymbolicBindings::operator!=)
      .def("__len__", &SymbolicBindings::size)
      .def("__getitem__", [](const SymbolicBindings& self, int64 i) {
        if (i >= self.bindings().size() || i < 0) {
          throw py::index_error("Index out of bounds");
        }
        return self.bindings()[i];
      });

  static py::exception<TypeMissingError> type_missing_exc(m,
                                                          "TypeMissingError");

  py::register_exception_translator([](std::exception_ptr p) {
    try {
      if (p) std::rethrow_exception(p);
    } catch (const TypeMissingError& e) {
      type_missing_exc.attr("node") = AstNodeHolder(e.node(), e.module());
      type_missing_exc.attr("user") = absl::nullopt;
      type_missing_exc(e.what());
    }
  });

  py::class_<TypeInfo, std::shared_ptr<TypeInfo>>(m, "TypeInfo")
      .def(py::init([](ModuleHolder& module,
                       absl::optional<std::shared_ptr<TypeInfo>> parent) {
             return std::make_shared<TypeInfo>(
                 module.module(),
                 parent.has_value() ? parent.value() : nullptr);
           }),
           py::arg("module"), py::arg("parent") = absl::nullopt)
      .def("__contains__",
           [](const TypeInfo& self, AstNodeHolder n) {
             return self.Contains(&n.deref());
           })
      .def("__setitem__",
           [](TypeInfo& self, AstNodeHolder n, const ConcreteType& type) {
             self.SetItem(&n.deref(), type);
           })
      .def(
          "__getitem__",
          [](TypeInfo& self, AstNodeHolder n) -> std::unique_ptr<ConcreteType> {
            absl::optional<ConcreteType*> result = self.GetItem(&n.deref());
            if (result.has_value()) {
              return (*result)->CloneToUnique();
            }
            throw TypeMissingError(&n.deref());
          })
      .def("add_instantiation",
           [](TypeInfo& self, InvocationHolder invocation,
              const SymbolicBindings& caller,
              std::shared_ptr<TypeInfo> type_info) {
             self.AddInstantiation(&invocation.deref(), caller, type_info);
           })
      .def("add_invocation_symbolic_bindings",
           [](TypeInfo& self, InvocationHolder invocation,
              const SymbolicBindings& caller, const SymbolicBindings& callee) {
             self.AddInvocationSymbolicBindings(&invocation.deref(), caller,
                                                callee);
           })
      .def("get_invocation_symbolic_bindings",
           [](TypeInfo& self, InvocationHolder invocation,
              const SymbolicBindings& caller) -> const SymbolicBindings& {
             absl::optional<const SymbolicBindings*> result =
                 self.GetInvocationSymbolicBindings(&invocation.deref(),
                                                    caller);
             if (!result.has_value()) {
               throw py::key_error(
                   "Could not find symbolic bindings for invocation: " +
                   invocation.deref().ToString());
             }
             return **result;
           })
      .def("update",
           [](TypeInfo& self, const TypeInfo& other) { self.Update(other); })
      .def("has_instantiation",
           [](const TypeInfo& self, InvocationHolder invocation,
              const SymbolicBindings& caller) {
             return self.HasInstantiation(&invocation.deref(), caller);
           })
      .def("get_instantiation",
           [](const TypeInfo& self, InvocationHolder invocation,
              const SymbolicBindings& caller) {
             absl::optional<std::shared_ptr<TypeInfo>> result =
                 self.GetInstantiation(&invocation.deref(), caller);
             if (!result.has_value()) {
               throw py::key_error("Could not resolve instantiation.");
             }
             return *result;
           })
      .def("add_slice_start_width",
           [](TypeInfo& self, SliceHolder slice,
              const SymbolicBindings& symbolic_bindings,
              std::pair<int, int> start_width) {
             self.AddSliceStartAndWidth(
                 &slice.deref(), symbolic_bindings,
                 StartAndWidth{start_width.first, start_width.second});
           })
      .def("get_slice_start_width",
           [](const TypeInfo& self, SliceHolder slice,
              const SymbolicBindings& symbolic_bindings) {
             absl::optional<StartAndWidth> result =
                 self.GetSliceStartAndWidth(&slice.deref(), symbolic_bindings);
             if (!result.has_value()) {
               throw py::key_error("Could not resolve slice to TypeInfo data.");
             }
             return std::pair<int64, int64>{result->start, result->width};
           })
      .def("note_constant",
           [](TypeInfo& self, NameDefHolder name_def,
              ConstantDefHolder constant) {
             self.NoteConstant(&name_def.deref(), &constant.deref());
           })
      .def("get_imports",
           [](const TypeInfo& self) {
             std::vector<
                 std::pair<ImportHolder,
                           std::pair<ModuleHolder, std::shared_ptr<TypeInfo>>>>
                 result;
             auto add_import = [&](const auto& pair) {
               XLS_CHECK(pair.second.type_info != nullptr);
               result.push_back(
                   {ImportHolder(pair.first, self.module()),
                    {ModuleHolder(pair.second.module.get(), pair.second.module),
                     pair.second.type_info}});
             };
             for (const auto& pair : self.imports()) {
               add_import(pair);
             }
             if (self.parent() != nullptr) {
               for (const auto& pair : self.parent()->imports()) {
                 add_import(pair);
               }
             }
             return result;
           })
      .def("get_imported",
           [](const TypeInfo& self, ImportHolder import) {
             absl::optional<const ImportedInfo*> info =
                 self.GetImported(&import.deref());
             if (!info.has_value()) {
               throw py::key_error("Could not find information for import.");
             }
             return std::make_pair(
                 ModuleHolder((*info)->module.get(), (*info)->module),
                 (*info)->type_info);
           })
      .def("add_import",
           [](TypeInfo& self, ImportHolder import,
              std::pair<ModuleHolder, std::shared_ptr<TypeInfo>> info) {
             XLS_CHECK(info.second != nullptr);
             self.AddImport(&import.deref(), info.first.module(), info.second);
           })
      .def("get_const_int",
           [](const TypeInfo& self,
              NameDefHolder name_def) -> absl::optional<ExprHolder> {
             absl::optional<Expr*> e = self.GetConstInt(&name_def.deref());
             if (e.has_value()) {
               return ExprHolder(e.value(), self.module());
             }
             return absl::nullopt;
           })
      .def("clear_type_info_refs_for_gc",
           [](TypeInfo& self) { self.ClearTypeInfoRefsForGc(); })
      .def_property_readonly("module",
                             [](TypeInfo& self) {
                               return ModuleHolder(self.module().get(),
                                                   self.module());
                             })
      .def_property_readonly("parent", &TypeInfo::parent);
}

}  // namespace xls::dslx
