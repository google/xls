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
               XLS_CHECK_EQ(result.count(b.identifier), 0);
               result[b.identifier] = b.value;
             }
             return result;
           })
      .def("__str__", &SymbolicBindings::ToString)
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
      absl::optional<AstNodeHolder> user;
      if (e.user() != nullptr) {
        user = AstNodeHolder(e.user(), e.module());
      }
      type_missing_exc.attr("user") = user;
      type_missing_exc(e.what());
    }
  });

  py::class_<TypeInfo, std::shared_ptr<TypeInfo>>(m, "TypeInfo")
      .def("clear_type_info_refs_for_gc",
           [](TypeInfo& self) { self.ClearTypeInfoRefsForGc(); })
      .def(
          "get_type",
          [](TypeInfo& self, AstNodeHolder n) -> std::unique_ptr<ConcreteType> {
            absl::optional<ConcreteType*> result = self.GetItem(&n.deref());
            if (result.has_value()) {
              return (*result)->CloneToUnique();
            }
            throw TypeMissingError(&n.deref(), nullptr);
          });
}

}  // namespace xls::dslx
