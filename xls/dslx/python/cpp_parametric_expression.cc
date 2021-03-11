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
#include "pybind11/functional.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "xls/common/status/status_macros.h"
#include "xls/common/status/statusor_pybind_caster.h"
#include "xls/dslx/parametric_expression.h"

namespace py = pybind11;

namespace xls::dslx {

PYBIND11_MODULE(cpp_parametric_expression, m) {
  py::class_<ParametricExpression>(m, "ParametricExpression")
      .def("__eq__", &ParametricExpression::operator==)
      .def("__str__", &ParametricExpression::ToString)
      .def("__repr__", &ParametricExpression::ToRepr)
      .def("__add__", &ParametricExpression::Add)
      .def("__radd__", &ParametricExpression::Add)
      .def("get_freevars",
           [](const ParametricExpression& self) {
             auto freevars = self.GetFreeVariables();
             return std::unordered_set<std::string>(freevars.begin(),
                                                    freevars.end());
           })
      .def("evaluate",
           [](const ParametricExpression& self,
              const std::unordered_map<
                  std::string,
                  absl::variant<const ParametricExpression*, int64_t>>& env) {
             ParametricExpression::Env fhm_env(env.begin(), env.end());
             return self.Evaluate(fhm_env);
           });
  py::class_<ParametricAdd, ParametricExpression>(m, "ParametricAdd")
      .def(py::init(
          [](const ParametricExpression& lhs, const ParametricExpression& rhs) {
            return ParametricAdd(lhs.Clone(), rhs.Clone());
          }));
  py::class_<ParametricMul, ParametricExpression>(m, "ParametricMul")
      .def(py::init(
          [](const ParametricExpression& lhs, const ParametricExpression& rhs) {
            return ParametricMul(lhs.Clone(), rhs.Clone());
          }));
  py::class_<ParametricConstant, ParametricExpression>(m, "ParametricConstant")
      .def(py::init<int64_t>());
  py::class_<ParametricSymbol, ParametricExpression>(m, "ParametricSymbol")
      .def(py::init<std::string, Span>())
      .def_property_readonly("identifier", &ParametricSymbol::identifier);
}

}  // namespace xls::dslx
