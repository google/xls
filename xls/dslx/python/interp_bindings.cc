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

#include "xls/dslx/interp_bindings.h"

#include "absl/base/casts.h"
#include "pybind11/functional.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "xls/common/status/status_macros.h"
#include "xls/common/status/statusor_pybind_caster.h"
#include "xls/dslx/python/cpp_ast.h"

namespace py = pybind11;

namespace xls::dslx {

// If the status is "not found" throws a key error with the given status
// message.
void TryThrowKeyError(const absl::Status& status) {
  if (status.code() == absl::StatusCode::kNotFound) {
    throw py::key_error(std::string(status.message()));
  }
}

PYBIND11_MODULE(interp_bindings, m) {
  ImportStatusModule();

  using PySymbolicBindings = std::vector<std::pair<std::string, int64>>;

  py::class_<FnCtx>(m, "FnCtx")
      .def(py::init([](std::string module_name, std::string fn_name,
                       PySymbolicBindings sym_bindings) {
        return FnCtx{module_name, fn_name, SymbolicBindings(sym_bindings)};
      }))
      .def_property_readonly("module_name",
                             [](const FnCtx& self) { return self.module_name; })
      .def_property_readonly("fn_name",
                             [](const FnCtx& self) { return self.fn_name; })
      .def_property_readonly("sym_bindings", [](const FnCtx& self) {
        const SymbolicBindings& bindings = self.sym_bindings;
        py::tuple ret(bindings.size());
        for (int64 i = 0; i < bindings.size(); ++i) {
          const SymbolicBinding& b = bindings.bindings()[i];
          ret[i] = std::make_pair(b.identifier, b.value);
        }
        return ret;
      });

  py::class_<InterpBindings, std::shared_ptr<InterpBindings>>(m, "Bindings")
      .def(py::init<std::shared_ptr<InterpBindings>>(), py::arg("parent"))
      .def("add_value", &InterpBindings::AddValue)
      .def(
          "add_mod",
          [](InterpBindings& self, std::string identifier, ModuleHolder value) {
            self.AddMod(std::move(identifier), &value.deref());
          })
      .def("add_typedef",
           [](InterpBindings& self, std::string identifier,
              TypeDefHolder value) {
             self.AddTypeDef(std::move(identifier), &value.deref());
           })
      .def("add_enum",
           [](InterpBindings& self, std::string identifier,
              EnumDefHolder value) {
             self.AddEnum(std::move(identifier), &value.deref());
           })
      .def("add_struct",
           [](InterpBindings& self, std::string identifier,
              StructDefHolder value) {
             self.AddStruct(std::move(identifier), &value.deref());
           })
      .def("add_fn", &InterpBindings::AddFn)
      .def("resolve_value_from_identifier",
           [](const InterpBindings& self, absl::string_view identifier) {
             absl::StatusOr<InterpValue> v =
                 self.ResolveValueFromIdentifier(identifier);
             TryThrowKeyError(v.status());
             return v;
           })
      .def("resolve_value",
           [](const InterpBindings& self, NameRefHolder name_ref) {
             absl::StatusOr<InterpValue> result =
                 self.ResolveValue(&name_ref.deref());
             TryThrowKeyError(result.status());
             return result;
           })
      .def("resolve_type_definition",
           [](const InterpBindings& self, absl::string_view identifier,
              ModuleHolder m) -> absl::StatusOr<AstNodeHolder> {
             using Variant =
                 absl::variant<TypeAnnotation*, EnumDef*, StructDef*>;
             absl::StatusOr<Variant> v_or =
                 self.ResolveTypeDefinition(identifier);
             TryThrowKeyError(v_or.status());
             XLS_ASSIGN_OR_RETURN(Variant v, v_or);
             return AstNodeHolder(ToAstNode(v), m.module());
           })
      .def("resolve_mod",
           [](const InterpBindings& self, absl::string_view identifier,
              ModuleHolder m) -> absl::StatusOr<ModuleHolder> {
             absl::StatusOr<Module*> v_or = self.ResolveModule(identifier);
             TryThrowKeyError(v_or.status());
             XLS_ASSIGN_OR_RETURN(Module * v, v_or);
             return ModuleHolder(v, m.module());
           })
      .def("clone_with",
           [](std::shared_ptr<InterpBindings>& self,
              NameDefTreeHolder name_def_tree, InterpValue value) {
             return InterpBindings::CloneWith(self, &name_def_tree.deref(),
                                              value);
           })
      .def("keys",
           [](const InterpBindings& self) {
             absl::flat_hash_set<std::string> keys = self.GetKeys();
             return std::unordered_set<std::string>(keys.begin(), keys.end());
           })
      .def_property("fn_ctx", &InterpBindings::fn_ctx,
                    &InterpBindings::set_fn_ctx);
}

}  // namespace xls::dslx
