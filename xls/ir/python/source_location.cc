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

#include "xls/ir/source_location.h"

#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

namespace py = pybind11;

namespace xls {

PYBIND11_MODULE(source_location, m) {
  py::module::import("xls.ir.python.fileno");

  py::class_<SourceLocation>(m, "SourceLocation")
      .def(py::init<Fileno, Lineno, Colno>(), py::arg("fileno"),
           py::arg("lineno"), py::arg("colno"));

  py::class_<SourceInfo>(m, "SourceInfo")
      .def(py::init<std::vector<SourceLocation>>(), py::arg("locations"));
}

}  // namespace xls
