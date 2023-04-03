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

#include "xls/fuzzer/cpp_sample_runner.h"

#include "absl/base/casts.h"
#include "absl/status/statusor.h"
#include "pybind11/chrono.h"
#include "pybind11/functional.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "pybind11_abseil/statusor_caster.h"
#include "xls/common/status/import_status_module.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/dslx/interp_value_helpers.h"

namespace py = pybind11;

namespace xls {

PYBIND11_MODULE(cpp_sample_runner, m) {
  ImportStatusModule();

  m.def(
      "compare_results_function",
      [](const std::unordered_map<std::string, std::vector<dslx::InterpValue>>&
             results,
         std::optional<ArgsBatch> maybe_args_batch) -> absl::Status {
        absl::flat_hash_map<std::string, absl::Span<const dslx::InterpValue>>
            absl_results(results.begin(), results.end());
        return CompareResultsFunction(
            absl_results,
            maybe_args_batch.has_value() ? &maybe_args_batch.value() : nullptr);
      },
      py::arg("results"), py::arg("args_batch") = std::nullopt);
}

}  // namespace xls
