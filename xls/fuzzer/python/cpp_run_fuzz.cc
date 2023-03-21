// Copyright 2023 The XLS Authors
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

#include "xls/fuzzer/cpp_run_fuzz.h"

#include "absl/base/casts.h"
#include "absl/status/statusor.h"
#include "libs/json11/json11.hpp"
#include "pybind11/chrono.h"
#include "pybind11/functional.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "pybind11_abseil/statusor_caster.h"
#include "xls/common/status/import_status_module.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/dslx/interp_value_helpers.h"
#include "xls/fuzzer/sample.h"

namespace py = pybind11;

namespace xls {

PYBIND11_MODULE(cpp_run_fuzz, m) {
  ImportStatusModule();

  m.def(
      "minimize_ir",
      [](const Sample& smp, const std::string& run_dir,
         const std::optional<std::string>& inject_jit_result,
         const std::optional<std::chrono::microseconds>& timeout)
          -> absl::StatusOr<std::optional<std::string>> {
        std::optional<absl::Duration> absl_timeout;
        if (timeout.has_value()) {
          absl_timeout = absl::FromChrono(timeout.value());
        }
        XLS_ASSIGN_OR_RETURN(
            std::optional<std::filesystem::path> maybe_path,
            MinimizeIr(smp, run_dir, inject_jit_result, absl_timeout));
        if (!maybe_path.has_value()) {
          return std::nullopt;
        }
        return std::string{maybe_path.value()};
      },
      py::arg("smp"), py::arg("run_dir"),
      py::arg("inject_jit_result") = std::nullopt,
      py::arg("timeout") = std::nullopt);
}

}  // namespace xls
