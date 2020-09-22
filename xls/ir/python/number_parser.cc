// Copyright 2020 Google LLC
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

#include "xls/ir/number_parser.h"

#include "pybind11/pybind11.h"
#include "xls/common/status/statusor_pybind_caster.h"

namespace py = pybind11;

namespace xls {

PYBIND11_MODULE(number_parser, m) {
  ImportStatusModule();
  py::module::import("xls.ir.python.bits");
  py::module::import("xls.ir.python.format_preference");

  m.def("bits_from_string", &ParseUnsignedNumberWithoutPrefix, py::arg("data"),
        py::arg("format") = FormatPreference::kHex,
        py::arg("bit_count") = kMinimumBitCount);
}

}  // namespace xls
