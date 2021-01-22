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

#include "xls/dslx/interpreter.h"

#include "absl/base/casts.h"
#include "absl/status/statusor.h"
#include "pybind11/functional.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "xls/common/status/status_macros.h"
#include "xls/common/status/statusor_pybind_caster.h"
#include "xls/dslx/import_routines.h"
#include "xls/dslx/parser.h"
#include "xls/dslx/python/callback_converters.h"
#include "xls/dslx/python/cpp_ast.h"
#include "xls/dslx/python/errors.h"
#include "xls/dslx/scanner.h"
#include "xls/dslx/symbolic_bindings.h"
#include "xls/dslx/typecheck.h"
#include "xls/ir/python/wrapper_types.h"

namespace py = pybind11;

namespace xls::dslx {

template <typename K, typename V>
absl::flat_hash_map<K, V> ToAbsl(const std::unordered_map<K, V>& m) {
  return absl::flat_hash_map<K, V>(m.begin(), m.end());
}

PYBIND11_MODULE(interpreter, m) {
  ImportStatusModule();
  py::module::import("xls.dslx.python.cpp_ast");

  static py::exception<FailureError> failure_exc(m, "FailureError");

  py::register_exception_translator([](std::exception_ptr p) {
    try {
      if (p) std::rethrow_exception(p);
    } catch (const FailureError& e) {
      py::object& e_type = failure_exc;
      py::object instance = e_type();
      instance.attr("message") = e.what();
      instance.attr("span") = e.span();
      PyErr_SetObject(failure_exc.ptr(), instance.ptr());
    }
  });

  m.def(
      "run_batched",
      [](absl::string_view text, absl::string_view function_name,
         const std::vector<std::vector<InterpValue>> args_batch)
          -> absl::StatusOr<std::vector<InterpValue>> {
        ImportCache import_cache;
        Scanner scanner{"batched.x", std::string{text}};
        Parser parser("batched", &scanner);
        XLS_ASSIGN_OR_RETURN(std::shared_ptr<Module> module,
                             parser.ParseModule());
        XLS_ASSIGN_OR_RETURN(
            std::shared_ptr<TypeInfo> type_info,
            CheckModule(module.get(), &import_cache, /*dslx_paths=*/{}));

        XLS_ASSIGN_OR_RETURN(Function * f,
                             module->GetFunctionOrError(function_name));
        XLS_ASSIGN_OR_RETURN(FunctionType * fn_type,
                             type_info->GetItemAs<FunctionType>(f));

        Interpreter interpreter(module.get(), type_info, nullptr,
                                /*additional_search_paths=*/{}, &import_cache,
                                /*trace_all=*/false, /*package=*/nullptr);
        std::vector<InterpValue> results;
        results.reserve(args_batch.size());
        for (const std::vector<InterpValue>& unsigned_args : args_batch) {
          XLS_ASSIGN_OR_RETURN(std::vector<InterpValue> args,
                               SignConvertArgs(*fn_type, unsigned_args));
          XLS_ASSIGN_OR_RETURN(InterpValue result,
                               interpreter.RunFunction(function_name, args));
          results.push_back(result);
        }
        return results;
      },
      py::arg("text"), py::arg("function_name"), py::arg("args_batch"));
}

}  // namespace xls::dslx
