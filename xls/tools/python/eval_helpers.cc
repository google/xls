// Copyright 2021 The XLS Authors
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

// Exposes AST generation capability (as is needed for fuzzing) to Python code
// (which currently drives the sampling / running process).

#include "xls/tools/eval_helpers.h"

#include "pybind11/functional.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "pybind11_abseil/absl_casters.h"
#include "pybind11_abseil/statusor_caster.h"
#include "xls/common/status/import_status_module.h"

namespace xls {

namespace py = pybind11;

PYBIND11_MODULE(eval_helpers, m) {
  ImportStatusModule();

  m.def("channel_values_to_string", &ChannelValuesToString);
  m.def("parse_channel_values", &ParseChannelValues,
        py::arg("all_channel_values"),
        py::arg("max_values_count") = absl::nullopt);
}

}  // namespace xls
