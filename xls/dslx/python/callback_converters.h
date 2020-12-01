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

#include "pybind11/pybind11.h"
#include "xls/dslx/cpp_evaluate.h"
#include "xls/dslx/deduce.h"
#include "xls/dslx/interp_value.h"
#include "xls/dslx/python/cpp_ast.h"
#include "xls/dslx/type_info.h"

// Prevent symbols from being stripped from the callback_converters helper DSO
// if they appear to be unused.
#define XLS_DSLX_PY_PUBLIC __attribute__((visibility("default")))

namespace xls::dslx {

class DeduceCtx;

using PyTypecheckFn =
    std::function<std::shared_ptr<TypeInfo>(ModuleHolder module)>;

using PyTypecheckFunctionFn =
    std::function<void(FunctionHolder module, DeduceCtx*)>;

// Implementation note: we take a pybind11 object as the retval here so we can
// retain the reference to the unique ConcreteType (and clone it to unique) in
// our callback wrapper.
using PyDeduceFn = std::function<pybind11::object(AstNodeHolder, DeduceCtx*)>;

// Converts a Python typecheck callback into a "C++ signature" function.
XLS_DSLX_PY_PUBLIC TypecheckFn ToCppTypecheck(const PyTypecheckFn& py);

// Converts a Python deduce callback into a "C++ signature" function.
XLS_DSLX_PY_PUBLIC DeduceFn ToCppDeduce(const PyDeduceFn& py);

// Converts a C++ typecheck callback into a "Python signature" function.
XLS_DSLX_PY_PUBLIC PyTypecheckFn ToPyTypecheck(const TypecheckFn& cpp);

// Converts a Python "typecheck function" callback into a "C++ signature"
// function.
XLS_DSLX_PY_PUBLIC TypecheckFunctionFn
ToCppTypecheckFunction(const PyTypecheckFunctionFn& py);

}  // namespace xls::dslx

#endif  // XLS_DSLX_PYTHON_CALLBACK_CONVERTERS_H_
