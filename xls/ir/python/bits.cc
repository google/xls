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

#include "xls/ir/bits.h"

#include "pybind11/pybind11.h"
#include "xls/common/status/statusor_pybind_caster.h"

namespace py = pybind11;

namespace xls {

PYBIND11_MODULE(bits, m) {
  ImportStatusModule();

  py::class_<Bits>(m, "Bits")
      .def(py::init<int64>())
      .def("bit_count", &Bits::bit_count)
      .def("to_uint", &Bits::ToUint64)
      .def("word_to_uint", &Bits::WordToUint64, py::arg("word_number") = 0)
      .def("to_int", &Bits::ToInt64);

  m.def("UBits", &UBitsWithStatus, py::arg("value"), py::arg("bit_count"));
  m.def("SBits", &SBitsWithStatus, py::arg("value"), py::arg("bit_count"));
}

}  // namespace xls
