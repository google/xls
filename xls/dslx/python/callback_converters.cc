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

#include "xls/dslx/python/errors.h"

namespace xls::dslx {

static absl::Status ConvertException(std::exception& e, const char* caller) {
  absl::string_view message = e.what();
  XLS_VLOG(5) << "Caught exception from " << caller << "; message: \""
              << message.substr(0, 16) << "\"";
  if (absl::ConsumePrefix(&message, "KeyError: ")) {
    XLS_VLOG(5) << "Converted exception to NotFoundError";
    return absl::NotFoundError(message);
  }
  // Note: normally we would throw a Python positional error, but
  // since this is a somewhat rare condition and everything is being
  // ported to C++ we don't do the super-nice thing and instead throw
  // a status error if you have a typecheck-failure-under-import.
  return absl::InternalError(e.what());
}

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
    } catch (pybind11::error_already_set& e) {
      throw;
    } catch (std::exception& e) {
      return ConvertException(e, "PyTypecheckFn");
    }
  };
}

PyTypecheckFn ToPyTypecheck(const TypecheckFn& cpp) {
  return [cpp](ModuleHolder module) -> std::shared_ptr<TypeInfo> {
    return cpp(module.module()).value();
  };
}

TypecheckFunctionFn ToCppTypecheckFunction(const PyTypecheckFunctionFn& py) {
  return [py](Function* function, DeduceCtx* ctx) -> absl::Status {
    FunctionHolder holder(function, function->owner()->shared_from_this());
    try {
      py(holder, ctx);
    } catch (pybind11::error_already_set& e) {
      throw;
    } catch (std::exception& e) {
      return ConvertException(e, "PyTypecheckFunctionFn");
    }
    return absl::OkStatus();
  };
}

DeduceFn ToCppDeduce(const PyDeduceFn& py) {
  XLS_CHECK(py != nullptr);
  return [py](AstNode* node,
              DeduceCtx* ctx) -> absl::StatusOr<std::unique_ptr<ConcreteType>> {
    XLS_RET_CHECK(node != nullptr);
    AstNodeHolder holder(node, node->owner()->shared_from_this());
    try {
      pybind11::object retval = py(holder, ctx);
      return pybind11::cast<ConcreteType*>(*retval)->CloneToUnique();
    } catch (pybind11::error_already_set& e) {
      throw;
    } catch (std::exception& e) {
      return ConvertException(e, "PyDeduceFn");
    }
  };
}

}  // namespace xls::dslx
