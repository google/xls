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
#include "absl/strings/ascii.h"
#include "pybind11/functional.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "xls/common/status/status_macros.h"
#include "xls/common/status/statusor_pybind_caster.h"
#include "xls/dslx/concrete_type.h"
#include "xls/dslx/deduce.h"
#include "xls/dslx/import_routines.h"
#include "xls/dslx/interp_bindings.h"
#include "xls/dslx/python/callback_converters.h"
#include "xls/dslx/python/cpp_ast.h"
#include "xls/dslx/python/errors.h"
#include "xls/dslx/typecheck.h"

namespace py = pybind11;

namespace xls::dslx {

static absl::Status TryThrowErrors(const absl::Status& status) {
  TryThrowTypeInferenceError(status);
  TryThrowXlsTypeError(status);
  TryThrowKeyError(status);
  TryThrowTypeMissingError(status);
  TryThrowArgCountMismatchError(status);
  return status;
}

PYBIND11_MODULE(cpp_typecheck, m) {
  ImportStatusModule();

  // Helper that sees if "status" is one of the various types of errors to be
  // thrown as explicit Python exceptions (vs a fallback StatusOr exception
  // type).
  m.def("check_test", [](TestHolder node, DeduceCtx* ctx) {
    auto status = CheckTest(&node.deref(), ctx);
    return TryThrowErrors(status);
  });

  m.def("check_function", [](FunctionHolder node, DeduceCtx* ctx) {
    auto status = CheckFunction(&node.deref(), ctx);
    return TryThrowErrors(status);
  });

  m.def("instantiate_builtin_parametric",
        [](BuiltinNameDefHolder builtin_name, InvocationHolder invocation,
           DeduceCtx* ctx) -> absl::StatusOr<absl::optional<NameDefHolder>> {
          auto statusor = InstantiateBuiltinParametric(
              &builtin_name.deref(), &invocation.deref(), ctx);
          TryThrowErrors(statusor.status());
          XLS_ASSIGN_OR_RETURN(NameDef * name_def, statusor);
          if (name_def == nullptr) {
            return absl::nullopt;
          }
          return NameDefHolder(name_def, name_def->owner()->shared_from_this());
        });

  m.def("check_top_node_in_module",
        [](FunctionHolder f, DeduceCtx* ctx) -> absl::Status {
          return TryThrowErrors(CheckTopNodeInModule(&f.deref(), ctx));
        });
  m.def("check_top_node_in_module",
        [](TestHolder f, DeduceCtx* ctx) -> absl::Status {
          return TryThrowErrors(CheckTopNodeInModule(&f.deref(), ctx));
        });
  m.def("check_top_node_in_module",
        [](StructDefHolder f, DeduceCtx* ctx) -> absl::Status {
          return TryThrowErrors(CheckTopNodeInModule(&f.deref(), ctx));
        });
  m.def("check_top_node_in_module",
        [](TypeDefHolder f, DeduceCtx* ctx) -> absl::Status {
          return TryThrowErrors(CheckTopNodeInModule(&f.deref(), ctx));
        });
}

}  // namespace xls::dslx
