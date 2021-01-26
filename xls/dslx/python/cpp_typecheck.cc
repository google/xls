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

  m.def(
      "check_module",
      [](ModuleHolder module, absl::optional<ImportCache*> import_cache,
         const std::vector<std::string>& additional_search_paths)
          -> absl::Status {
        ImportCache* pimport_cache =
            import_cache.has_value() ? import_cache.value() : nullptr;
        auto statusor = CheckModule(&module.deref(), pimport_cache,
                                    additional_search_paths);
        (void)TryThrowErrors(statusor.status());
        return statusor.status();
      },
      py::arg("module"), py::arg("import_cache"),
      py::arg("additional_search_paths"));
}

}  // namespace xls::dslx
