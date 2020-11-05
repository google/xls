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

#include "xls/dslx/cpp_evaluate.h"

#include "absl/base/casts.h"
#include "absl/status/statusor.h"
#include "pybind11/functional.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "xls/common/status/status_macros.h"
#include "xls/common/status/statusor_pybind_caster.h"
#include "xls/dslx/python/callback_converters.h"
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

PYBIND11_MODULE(cpp_evaluate, m) {
  ImportStatusModule();

  m.def("evaluate_index_bitslice", [](TypeInfo* type_info, IndexHolder expr,
                                      InterpBindings* bindings,
                                      const Bits& bits) {
    return EvaluateIndexBitslice(type_info, &expr.deref(), bindings, bits);
  });
  m.def("evaluate_ConstRef", [](ConstRefHolder expr, InterpBindings* bindings,
                                ConcreteType* type_context) {
    auto statusor = EvaluateConstRef(&expr.deref(), bindings, type_context);
    TryThrowKeyError(statusor.status());
    return statusor;
  });
  m.def("evaluate_NameRef", [](NameRefHolder expr, InterpBindings* bindings,
                               ConcreteType* type_context) {
    auto statusor = EvaluateNameRef(&expr.deref(), bindings, type_context);
    TryThrowKeyError(statusor.status());
    return statusor;
  });
  m.def("make_top_level_bindings",
        [](ModuleHolder module, absl::optional<PyTypecheckFn> py_typecheck,
           const PyEvaluateFn& eval, const PyIsWipFn& is_wip,
           const PyNoteWipFn& note_wip, absl::optional<ImportCache*> py_cache) {
          TypecheckFn typecheck;
          if (py_typecheck.has_value()) {
            typecheck = ToCppTypecheck(*py_typecheck);
          }
          ImportCache* cache = py_cache.has_value() ? *py_cache : nullptr;
          return MakeTopLevelBindings(module.module(), typecheck,
                                      ToCppEval(eval), ToCppIsWip(is_wip),
                                      ToCppNoteWip(note_wip), cache);
        });
}

}  // namespace xls::dslx
