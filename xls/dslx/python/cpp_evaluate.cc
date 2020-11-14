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

  m.def(
      "evaluate_ConstRef",
      [](ConstRefHolder expr, InterpBindings* bindings,
         ConcreteType* type_context, PyInterpCallbackData* py_callbacks) {
        auto statusor = EvaluateConstRef(&expr.deref(), bindings, type_context);
        TryThrowKeyError(statusor.status());
        return statusor;
      },
      py::arg("expr"), py::arg("bindings"), py::arg("type_context"),
      py::arg("callbacks"));
  m.def(
      "evaluate_NameRef",
      [](NameRefHolder expr, InterpBindings* bindings,
         ConcreteType* type_context, PyInterpCallbackData* py_callbacks) {
        auto statusor = EvaluateNameRef(&expr.deref(), bindings, type_context);
        TryThrowKeyError(statusor.status());
        return statusor;
      },
      py::arg("expr"), py::arg("bindings"), py::arg("type_context"),
      py::arg("callbacks"));
  m.def(
      "evaluate_EnumRef",
      [](EnumRefHolder expr, InterpBindings* bindings,
         ConcreteType* type_context, PyInterpCallbackData* py_callbacks) {
        InterpCallbackData callbacks = ToCpp(*py_callbacks);
        auto statusor =
            EvaluateEnumRef(&expr.deref(), bindings, type_context, &callbacks);
        TryThrowKeyError(statusor.status());
        return statusor;
      },
      py::arg("expr"), py::arg("bindings"), py::arg("type_context"),
      py::arg("callbacks"));
  m.def(
      "evaluate_Unop",
      [](UnopHolder expr, InterpBindings* bindings, ConcreteType* type_context,
         PyInterpCallbackData* py_callbacks) {
        InterpCallbackData callbacks = ToCpp(*py_callbacks);
        auto statusor =
            EvaluateUnop(&expr.deref(), bindings, type_context, &callbacks);
        TryThrowKeyError(statusor.status());
        return statusor;
      },
      py::arg("expr"), py::arg("bindings"), py::arg("type_context"),
      py::arg("callbacks"));
  m.def(
      "evaluate_Binop",
      [](BinopHolder expr, InterpBindings* bindings, ConcreteType* type_context,
         PyInterpCallbackData* py_callbacks) {
        InterpCallbackData callbacks = ToCpp(*py_callbacks);
        auto statusor =
            EvaluateBinop(&expr.deref(), bindings, type_context, &callbacks);
        TryThrowKeyError(statusor.status());
        return statusor;
      },
      py::arg("expr"), py::arg("bindings"), py::arg("type_context"),
      py::arg("callbacks"));
  m.def(
      "evaluate_Ternary",
      [](TernaryHolder expr, InterpBindings* bindings,
         ConcreteType* type_context, PyInterpCallbackData* py_callbacks) {
        InterpCallbackData callbacks = ToCpp(*py_callbacks);
        auto statusor =
            EvaluateTernary(&expr.deref(), bindings, type_context, &callbacks);
        TryThrowKeyError(statusor.status());
        return statusor;
      },
      py::arg("expr"), py::arg("bindings"), py::arg("type_context"),
      py::arg("callbacks"));
  m.def(
      "evaluate_Attr",
      [](AttrHolder expr, InterpBindings* bindings, ConcreteType* type_context,
         PyInterpCallbackData* py_callbacks) {
        InterpCallbackData callbacks = ToCpp(*py_callbacks);
        auto statusor =
            EvaluateAttr(&expr.deref(), bindings, type_context, &callbacks);
        TryThrowKeyError(statusor.status());
        return statusor;
      },
      py::arg("expr"), py::arg("bindings"), py::arg("type_context"),
      py::arg("callbacks"));
  m.def(
      "evaluate_Match",
      [](MatchHolder expr, InterpBindings* bindings, ConcreteType* type_context,
         PyInterpCallbackData* py_callbacks) {
        InterpCallbackData callbacks = ToCpp(*py_callbacks);
        auto statusor =
            EvaluateMatch(&expr.deref(), bindings, type_context, &callbacks);
        TryThrowFailureError(statusor.status());
        TryThrowKeyError(statusor.status());
        return statusor;
      },
      py::arg("expr"), py::arg("bindings"), py::arg("type_context"),
      py::arg("callbacks"));
  m.def(
      "evaluate_Index",
      [](IndexHolder expr, InterpBindings* bindings, ConcreteType* type_context,
         PyInterpCallbackData* py_callbacks) {
        InterpCallbackData callbacks = ToCpp(*py_callbacks);
        auto statusor =
            EvaluateIndex(&expr.deref(), bindings, type_context, &callbacks);
        TryThrowFailureError(statusor.status());
        TryThrowKeyError(statusor.status());
        return statusor;
      },
      py::arg("expr"), py::arg("bindings"), py::arg("type_context"),
      py::arg("callbacks"));
  m.def(
      "evaluate_StructInstance",
      [](StructInstanceHolder expr, InterpBindings* bindings,
         ConcreteType* type_context, PyInterpCallbackData* py_callbacks) {
        InterpCallbackData callbacks = ToCpp(*py_callbacks);
        auto statusor = EvaluateStructInstance(&expr.deref(), bindings,
                                               type_context, &callbacks);
        TryThrowFailureError(statusor.status());
        TryThrowKeyError(statusor.status());
        return statusor;
      },
      py::arg("expr"), py::arg("bindings"), py::arg("type_context"),
      py::arg("callbacks"));
  m.def(
      "evaluate_SplatStructInstance",
      [](SplatStructInstanceHolder expr, InterpBindings* bindings,
         ConcreteType* type_context, PyInterpCallbackData* py_callbacks) {
        InterpCallbackData callbacks = ToCpp(*py_callbacks);
        auto statusor = EvaluateSplatStructInstance(&expr.deref(), bindings,
                                                    type_context, &callbacks);
        TryThrowFailureError(statusor.status());
        TryThrowKeyError(statusor.status());
        return statusor;
      },
      py::arg("expr"), py::arg("bindings"), py::arg("type_context"),
      py::arg("callbacks"));
  m.def(
      "evaluate_Number",
      [](NumberHolder expr, InterpBindings* bindings,
         ConcreteType* type_context, PyInterpCallbackData* py_callbacks) {
        InterpCallbackData callbacks = ToCpp(*py_callbacks);
        auto statusor =
            EvaluateNumber(&expr.deref(), bindings, type_context, &callbacks);
        TryThrowFailureError(statusor.status());
        TryThrowKeyError(statusor.status());
        return statusor;
      },
      py::arg("expr"), py::arg("bindings"), py::arg("type_context"),
      py::arg("callbacks"));
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
