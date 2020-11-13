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

#include "xls/dslx/python/callback_converters.h"

namespace xls::dslx {

TypecheckFn ToCppTypecheck(const PyTypecheckFn& py_typecheck) {
  // The typecheck callback we get from Python uses the "ModuleHolder"
  // type -- make a conversion lambda that turns a std::shared_ptr<Module>
  // into its "Holder" form so we can invoke the Python-provided
  // typechecking callback.
  return [py_typecheck](std::shared_ptr<Module> module)
             -> absl::StatusOr<std::shared_ptr<TypeInfo>> {
    ModuleHolder holder(module.get(), module);
    try {
      return py_typecheck(holder);
    } catch (std::exception& e) {
      // Note: normally we would throw a Python positional error, but
      // since this is a somewhat rare condition and everything is being
      // ported to C++ we don't do the super-nice thing and instead throw
      // a status error if you have a typecheck-failure-under-import.
      return absl::InternalError(e.what());
    }
  };
}

EvaluateFn ToCppEval(const PyEvaluateFn& py) {
  return [py](Expr* expr, InterpBindings* bindings,
              std::unique_ptr<ConcreteType> type_context) {
    return py(ExprHolder(expr, expr->owner()->shared_from_this()), bindings,
              std::move(type_context));
  };
}

IsWipFn ToCppIsWip(const PyIsWipFn& py) {
  return [py](ConstantDef* constant_def) -> bool {
    return py(ConstantDefHolder(constant_def,
                                constant_def->owner()->shared_from_this()));
  };
}

NoteWipFn ToCppNoteWip(const PyNoteWipFn& py) {
  return
      [py](ConstantDef* constant_def,
           absl::optional<InterpValue> value) -> absl::optional<InterpValue> {
        return py(ConstantDefHolder(constant_def,
                                    constant_def->owner()->shared_from_this()),
                  value);
      };
}

}  // namespace xls::dslx
