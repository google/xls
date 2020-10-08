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

#include "xls/dslx/cpp_pos.h"

#include "pybind11/functional.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "xls/common/status/statusor_pybind_caster.h"

namespace py = pybind11;

namespace xls::dslx {

std::string Repr(const Pos& pos) {
  return absl::StrFormat("Pos(\"%s\", %d, %d)", pos.filename(), pos.lineno(),
                         pos.colno());
}

PYBIND11_MODULE(cpp_pos, m) {
  ImportStatusModule();

  // class Pos
  py::class_<Pos>(m, "Pos")
      .def(py::init<std::string, int64, int64>(), py::arg("filename"),
           py::arg("lineno"), py::arg("colno"))
      .def("bump_col", &Pos::BumpCol)
      .def_property_readonly("filename", &Pos::filename)
      .def_property_readonly("lineno", &Pos::lineno)
      .def_property_readonly("colno", &Pos::colno)
      .def("__eq__", &Pos::operator==)
      .def("__ne__", &Pos::operator!=)
      .def("__lt__", &Pos::operator<)  // NOLINT
      .def("__ge__", &Pos::operator>=);

  // class Span
  py::class_<Span>(m, "Span")
      .def(py::init<Pos, Pos>())
      .def("__eq__", &Span::operator==)
      .def("__ne__", &Span::operator!=)
      .def("__str__", &Span::ToString)
      .def("__repr__",
           [](const Span& span) {
             return absl::StrFormat("Span(%s, %s)", Repr(span.start()),
                                    Repr(span.limit()));
           })
      .def("clone_with_limit", &Span::CloneWithLimit)
      .def_property_readonly("filename", &Span::filename)
      .def_property_readonly("start", &Span::start)
      .def_property_readonly("limit", &Span::limit);
}

}  // namespace xls::dslx
