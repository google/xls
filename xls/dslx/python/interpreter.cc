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

#include "absl/base/casts.h"
#include "absl/status/statusor.h"
#include "pybind11/functional.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "xls/common/status/status_macros.h"
#include "xls/common/status/statusor_pybind_caster.h"
#include "xls/dslx/bytecode.h"
#include "xls/dslx/bytecode_emitter.h"
#include "xls/dslx/bytecode_interpreter.h"
#include "xls/dslx/create_import_data.h"
#include "xls/dslx/import_data.h"
#include "xls/dslx/import_routines.h"
#include "xls/dslx/interp_value_helpers.h"
#include "xls/dslx/ir_converter.h"
#include "xls/dslx/parse_and_typecheck.h"
#include "xls/dslx/parser.h"
#include "xls/dslx/python/errors.h"
#include "xls/dslx/symbolic_bindings.h"
#include "xls/dslx/typecheck.h"
#include "xls/ir/ir_parser.h"
#include "xls/ir/python/wrapper_types.h"

namespace py = pybind11;

namespace xls::dslx {

template <typename K, typename V>
absl::flat_hash_map<K, V> ToAbsl(const std::unordered_map<K, V>& m) {
  return absl::flat_hash_map<K, V>(m.begin(), m.end());
}

PYBIND11_MODULE(interpreter, m) {
  ImportStatusModule();

  m.def("ir_value_text_to_interp_value",
        [](absl::string_view text) -> absl::StatusOr<InterpValue> {
          XLS_ASSIGN_OR_RETURN(Value v, xls::Parser::ParseTypedValue(text));
          return ValueToInterpValue(v);
        });
  m.def(
      "run_batched",
      [](absl::string_view text, absl::string_view function_name,
         const std::vector<std::vector<InterpValue>> args_batch,
         absl::string_view dslx_stdlib_path)
          -> absl::StatusOr<std::vector<InterpValue>> {
        ImportData import_data(
            CreateImportData(std::string(dslx_stdlib_path),
                             /*additional_search_paths=*/{}));
        XLS_ASSIGN_OR_RETURN(
            TypecheckedModule tm,
            ParseAndTypecheck(text, "batched.x", "batched", &import_data));
        XLS_ASSIGN_OR_RETURN(
            Function * f, tm.module->GetMemberOrError<Function>(function_name));
        XLS_ASSIGN_OR_RETURN(FunctionType * fn_type,
                             tm.type_info->GetItemAs<FunctionType>(f));

        XLS_ASSIGN_OR_RETURN(
            std::unique_ptr<BytecodeFunction> bf,
            BytecodeEmitter::Emit(&import_data, tm.type_info, f,
                                  /*caller_bindings=*/{}));
        std::vector<InterpValue> results;
        results.reserve(args_batch.size());
        for (const std::vector<InterpValue>& unsigned_args : args_batch) {
          XLS_ASSIGN_OR_RETURN(std::vector<InterpValue> args,
                               SignConvertArgs(*fn_type, unsigned_args));
          XLS_ASSIGN_OR_RETURN(
              InterpValue result,
              BytecodeInterpreter::Interpret(&import_data, bf.get(), args));
          results.push_back(result);
        }
        return results;
      },
      py::arg("text"), py::arg("function_name"), py::arg("args_batch"),
      py::arg("dslx_stdlib_path"));
}

}  // namespace xls::dslx
