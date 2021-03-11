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

#include "xls/ir/fileno.h"

#include "pybind11/pybind11.h"

namespace py = pybind11;

namespace xls {

PYBIND11_MODULE(fileno, m) {
  py::class_<Fileno>(m, "Fileno").def(py::init<int32_t>());

  py::class_<Lineno>(m, "Lineno").def(py::init<int32_t>());

  py::class_<Colno>(m, "Colno").def(py::init<int32_t>());
}

}  // namespace xls
