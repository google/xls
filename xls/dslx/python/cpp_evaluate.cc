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

#include "xls/dslx/cpp_evaluate.h"

#include "absl/base/casts.h"
#include "absl/status/statusor.h"
#include "pybind11/functional.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "xls/common/status/status_macros.h"
#include "xls/common/status/statusor_pybind_caster.h"
#include "xls/dslx/concrete_type.h"
#include "xls/dslx/interp_bindings.h"
#include "xls/dslx/python/callback_converters.h"
#include "xls/dslx/python/cpp_ast.h"
#include "xls/dslx/python/errors.h"

namespace py = pybind11;

namespace xls::dslx {

PYBIND11_MODULE(cpp_evaluate, m) {
  ImportStatusModule();

  m.def("concrete_type_accepts_value",
        [](const ConcreteType& type, const InterpValue& value) {
          auto statusor = ConcreteTypeAcceptsValue(type, value);
          TryThrowFailureError(statusor.status());
          return statusor;
        });
  m.def("concrete_type_convert_value",
        [](const ConcreteType& type, const InterpValue& value, const Span& span,
           absl::optional<std::vector<InterpValue>> enum_values,
           absl::optional<bool> enum_signed) {
          auto statusor = ConcreteTypeConvertValue(type, value, span,
                                                   enum_values, enum_signed);
          TryThrowFailureError(statusor.status());
          return statusor;
        });

  m.def("resolve_dim", &ResolveDim, py::arg("dim"), py::arg("bindings"));
}

}  // namespace xls::dslx
