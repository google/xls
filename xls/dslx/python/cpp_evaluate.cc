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
#include "xls/dslx/concrete_type.h"
#include "xls/dslx/interp_bindings.h"
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

// Python version of the InterpCallbackData -- the std::functions contained in
// here need to be converted (via callback_converts.h helpers) to pass them to
// C++ routines with the appropriate interfaces.
struct PyInterpCallbackData {
  absl::optional<PyTypecheckFn> typecheck;
  PyEvaluateFn eval;
  PyIsWipFn is_wip;
  PyNoteWipFn note_wip;
  absl::optional<ImportCache*> cache;
};

// Converts a PyInterpCallbackData to a InterpCallbackData.
InterpCallbackData ToCpp(const PyInterpCallbackData& py) {
  TypecheckFn typecheck;
  if (py.typecheck.has_value()) {
    typecheck = ToCppTypecheck(py.typecheck.value());
  }
  ImportCache* cache = py.cache.has_value() ? py.cache.value() : nullptr;
  return InterpCallbackData{typecheck, ToCppEval(py.eval),
                            ToCppIsWip(py.is_wip), ToCppNoteWip(py.note_wip),
                            cache};
}

PYBIND11_MODULE(cpp_evaluate, m) {
  ImportStatusModule();

  py::class_<PyInterpCallbackData>(m, "InterpCallbackData")
      .def(py::init<absl::optional<PyTypecheckFn>, PyEvaluateFn, PyIsWipFn,
                    PyNoteWipFn, absl::optional<ImportCache*>>());

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
        [](ModuleHolder module, PyInterpCallbackData* py_callbacks) {
          InterpCallbackData callbacks = ToCpp(*py_callbacks);
          return MakeTopLevelBindings(module.module(), &callbacks);
        });
  m.def("concretize_type",
        [](TypeAnnotationHolder type, InterpBindings* bindings,
           PyInterpCallbackData* py_callbacks) {
          InterpCallbackData callbacks = ToCpp(*py_callbacks);
          return ConcretizeType(&type.deref(), bindings, &callbacks);
        });

  m.def("resolve_dim", &ResolveDim, py::arg("dim"), py::arg("bindings"));
  m.def(
      "evaluate_to_struct_or_enum_or_annotation",
      [](AstNodeHolder node, InterpBindings* bindings,
         PyInterpCallbackData* py_callbacks) -> absl::StatusOr<AstNodeHolder> {
        InterpCallbackData callbacks = ToCpp(*py_callbacks);
        XLS_ASSIGN_OR_RETURN(TypeDefinition td,
                             ToTypeDefinition(&node.deref()));
        XLS_ASSIGN_OR_RETURN(
            DerefVariant deref,
            EvaluateToStructOrEnumOrAnnotation(td, bindings, &callbacks));
        AstNode* deref_node = ToAstNode(deref);
        return AstNodeHolder(deref_node,
                             deref_node->owner()->shared_from_this());
      });
}

}  // namespace xls::dslx
