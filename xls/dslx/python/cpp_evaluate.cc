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
#include "xls/dslx/python/errors.h"

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
  PyGetTypeFn get_type_info;
  absl::optional<ImportCache*> cache;
};

// Converts a PyInterpCallbackData to a InterpCallbackData.
InterpCallbackData ToCpp(const PyInterpCallbackData& py) {
  TypecheckFn typecheck;
  if (py.typecheck.has_value()) {
    typecheck = ToCppTypecheck(py.typecheck.value());
  }
  ImportCache* cache = py.cache.has_value() ? py.cache.value() : nullptr;
  return InterpCallbackData{typecheck,
                            ToCppEval(py.eval),
                            ToCppIsWip(py.is_wip),
                            ToCppNoteWip(py.note_wip),
                            py.get_type_info,
                            cache};
}

PYBIND11_MODULE(cpp_evaluate, m) {
  ImportStatusModule();

  py::class_<PyInterpCallbackData>(m, "InterpCallbackData")
      .def(py::init<absl::optional<PyTypecheckFn>, PyEvaluateFn, PyIsWipFn,
                    PyNoteWipFn, PyGetTypeFn, absl::optional<ImportCache*>>());

  // Note: this could be more properly formulated as a generic lambda, but since
  // this code will all likely go away when the interpreter is fully ported to
  // C++ we hackily use a macro for now.
#define ADD_EVAL(__cls)                                                        \
  m.def(                                                                       \
      "evaluate_" #__cls,                                                      \
      [](__cls##Holder expr, InterpBindings* bindings,                         \
         ConcreteType* type_context, PyInterpCallbackData* py_callbacks) {     \
        InterpCallbackData callbacks = ToCpp(*py_callbacks);                   \
        auto statusor = Evaluate##__cls(&expr.deref(), bindings, type_context, \
                                        &callbacks);                           \
        TryThrowFailureError(statusor.status());                               \
        TryThrowKeyError(statusor.status());                                   \
        return statusor;                                                       \
      },                                                                       \
      py::arg("expr"), py::arg("bindings"), py::arg("type_context"),           \
      py::arg("callbacks"))

  ADD_EVAL(ConstRef);
  ADD_EVAL(NameRef);
  ADD_EVAL(EnumRef);
  ADD_EVAL(Unop);
  ADD_EVAL(Binop);
  ADD_EVAL(Ternary);
  ADD_EVAL(Attr);
  ADD_EVAL(Match);
  ADD_EVAL(Index);
  ADD_EVAL(StructInstance);
  ADD_EVAL(SplatStructInstance);
  ADD_EVAL(Number);
  ADD_EVAL(XlsTuple);
  ADD_EVAL(Let);
  ADD_EVAL(Cast);
  ADD_EVAL(Array);
  ADD_EVAL(For);
  ADD_EVAL(While);
  ADD_EVAL(ModRef);
  ADD_EVAL(Carry);
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
  m.def("concrete_type_accepts_value",
        [](const ConcreteType& type, const InterpValue& value) {
          auto statusor = ConcreteTypeAcceptsValue(type, value);
          TryThrowFailureError(statusor.status());
          return statusor;
        });
  m.def("concrete_type_convert_value",
        [](const ConcreteType& type, const InterpValue& value, const Span& span,
           absl::optional<std::vector<InterpValue>> enum_values,
           absl::optional<bool> enum_signed) {
          auto statusor = ConcreteTypeConvertValue(type, value, span,
                                                   enum_values, enum_signed);
          TryThrowFailureError(statusor.status());
          return statusor;
        });
  m.def("evaluate_derived_parametrics",
        [](FunctionHolder f, InterpBindings* bindings,
           PyInterpCallbackData* py_callbacks,
           const std::unordered_map<std::string, int64>& bound_dims) {
          InterpCallbackData callbacks = ToCpp(*py_callbacks);
          return EvaluateDerivedParametrics(
              &f.deref(), bindings, &callbacks,
              absl::flat_hash_map<std::string, int64>(bound_dims.begin(),
                                                      bound_dims.end()));
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
