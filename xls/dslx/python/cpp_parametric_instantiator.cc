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

#include "xls/dslx/cpp_parametric_instantiator.h"

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
#include "xls/dslx/import_routines.h"
#include "xls/dslx/interp_bindings.h"
#include "xls/dslx/python/callback_converters.h"
#include "xls/dslx/python/cpp_ast.h"
#include "xls/dslx/python/errors.h"

namespace py = pybind11;

namespace xls::dslx {

PYBIND11_MODULE(cpp_parametric_instantiator, m) {
  ImportStatusModule();

  py::class_<TypeAndBindings>(m, "TypeAndBindings")
      .def("__getitem__",
           [](TypeAndBindings& self,
              int64 i) -> absl::variant<std::unique_ptr<ConcreteType>,
                                        SymbolicBindings> {
             switch (i) {
               case 0:
                 return self.type->CloneToUnique();
               case 1:
                 return self.symbolic_bindings;
               default:
                 throw py::index_error("Index out of bounds");
             }
           });

  static py::exception<ArgCountMismatchError> arg_count_exc(
      m, "ArgCountMismatchError");

  py::register_exception_translator([](std::exception_ptr p) {
    try {
      if (p) std::rethrow_exception(p);
    } catch (const ArgCountMismatchError& e) {
      arg_count_exc.attr("span") = e.span();
      arg_count_exc.attr("message") = e.message();
      arg_count_exc(e.what());
    }
  });

  // Unwraps vector of parametric bindings holders into vector of pointers.
  auto unwrap = +[](const std::vector<ParametricBindingHolder>& holders) {
    std::vector<ParametricBinding*> results;
    for (auto& holder : holders) {
      results.push_back(&holder.deref());
    }
    return results;
  };

  m.def("instantiate_function",
        [unwrap](const Span& span, const FunctionType& function_type,
                 const std::vector<const ConcreteType*>& arg_types,
                 DeduceCtx* ctx,
                 absl::optional<std::vector<ParametricBindingHolder>>
                     py_parametric_constraints,
                 absl::optional<std::unordered_map<std::string, int64>>
                     explicit_constraints) {
          absl::optional<absl::flat_hash_map<std::string, int64>>
              absl_explicit_constraints;
          absl::flat_hash_map<std::string, int64>* pexplicit_constraints =
              nullptr;
          if (explicit_constraints.has_value()) {
            absl_explicit_constraints.emplace(explicit_constraints->begin(),
                                              explicit_constraints->end());
            pexplicit_constraints = &*absl_explicit_constraints;
          }

          std::vector<ParametricBinding*> parametric_constraints;
          absl::optional<absl::Span<ParametricBinding* const>>
              pparametric_constraints;
          if (py_parametric_constraints.has_value()) {
            parametric_constraints = unwrap(*py_parametric_constraints);
            pparametric_constraints = absl::MakeSpan(parametric_constraints);
          }
          auto statusor = InstantiateFunction(
              span, function_type, CloneToUnique(absl::MakeSpan(arg_types)),
              ctx, pparametric_constraints, pexplicit_constraints);
          XLS_VLOG(5) << "instantiate_function status: " << statusor.status();
          TryThrowKeyError(statusor.status());
          TryThrowXlsTypeError(statusor.status());
          TryThrowArgCountMismatchError(statusor.status());
          return statusor;
        });
  m.def("instantiate_struct",
        [unwrap](const Span& span, const TupleType& struct_type,
                 const std::vector<const ConcreteType*>& arg_types,
                 const std::vector<const ConcreteType*>& member_types,
                 DeduceCtx* ctx,
                 absl::optional<std::vector<ParametricBindingHolder>>
                     py_parametric_bindings) {
          absl::optional<absl::Span<ParametricBinding* const>>
              pparametric_bindings;
          std::vector<ParametricBinding*> parametric_bindings;
          if (py_parametric_bindings.has_value()) {
            parametric_bindings = unwrap(*py_parametric_bindings);
            pparametric_bindings = absl::MakeSpan(parametric_bindings);
          }
          auto statusor = InstantiateStruct(
              span, struct_type, CloneToUnique(absl::MakeSpan(arg_types)),
              CloneToUnique(absl::MakeSpan(member_types)), ctx,
              pparametric_bindings);
          TryThrowKeyError(statusor.status());
          TryThrowXlsTypeError(statusor.status());
          TryThrowArgCountMismatchError(statusor.status());
          return statusor;
        });
}

}  // namespace xls::dslx
