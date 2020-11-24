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

// Generally the callbacks we get from Python need to have exceptions quashed
// into StatusOr and "holder" types converted to raw pointers.

#ifndef XLS_DSLX_PYTHON_CALLBACK_CONVERTERS_H_
#define XLS_DSLX_PYTHON_CALLBACK_CONVERTERS_H_

#include "xls/dslx/cpp_evaluate.h"
#include "xls/dslx/interp_value.h"
#include "xls/dslx/python/cpp_ast.h"
#include "xls/dslx/type_info.h"

namespace xls::dslx {

using PySymbolicBindings = std::vector<std::pair<std::string, int64>>;

using PyTypecheckFn =
    std::function<std::shared_ptr<TypeInfo>(ModuleHolder module)>;
using PyIsWipFn = std::function<bool(ConstantDefHolder)>;
using PyNoteWipFn = std::function<absl::optional<InterpValue>(
    ConstantDefHolder, absl::optional<InterpValue>)>;
using PyEvaluateFn = std::function<InterpValue(ExprHolder, InterpBindings*,
                                               std::unique_ptr<ConcreteType>)>;
using PyCallValueFn = std::function<InterpValue(
    InterpValue, const std::vector<InterpValue>&, const Span&, InvocationHolder,
    absl::optional<PySymbolicBindings>)>;
using PyGetTypeFn = std::function<std::shared_ptr<TypeInfo>()>;

// Converts a Python typecheck callback into a "C++ signature" function.
TypecheckFn ToCppTypecheck(const PyTypecheckFn& py);

// Converts a Python evaluate callback into a "C++ signature" function.
EvaluateFn ToCppEval(const PyEvaluateFn& py);

// Converts a Python 'call function value' callback into a "C++ signature"
// function.
CallValueFn ToCppCallValue(const PyCallValueFn& py);

// Converts a Python "is_wip" callback into a "C++ signature" function.
IsWipFn ToCppIsWip(const PyIsWipFn& py);

// Converts a Python "is_wip" callback into a "C++ signature" function.
NoteWipFn ToCppNoteWip(const PyNoteWipFn& py);

// Python version of the InterpCallbackData -- the std::functions contained in
// here need to be converted (via callback_converts.h helpers) to pass them to
// C++ routines with the appropriate interfaces.
struct PyInterpCallbackData {
  absl::optional<PyTypecheckFn> typecheck;
  PyEvaluateFn eval;
  PyCallValueFn call_value;
  PyIsWipFn is_wip;
  PyNoteWipFn note_wip;
  PyGetTypeFn get_type_info;
  absl::optional<ImportCache*> cache;
};

// Converts a PyInterpCallbackData to a InterpCallbackData.
InterpCallbackData ToCpp(const PyInterpCallbackData& py);

}  // namespace xls::dslx

#endif  // XLS_DSLX_PYTHON_CALLBACK_CONVERTERS_H_
