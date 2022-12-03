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

#include "xls/dslx/ast.h"

#include "absl/base/casts.h"
#include "absl/status/statusor.h"
#include "pybind11/functional.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "xls/common/status/status_macros.h"

namespace py = pybind11;

namespace xls::dslx {

PYBIND11_MODULE(ast, m) {
  py::enum_<BinopKind>(m, "BinopKind")
#define VALUE(A, B, ...) .value(B, BinopKind::A)
      XLS_DSLX_BINOP_KIND_EACH(VALUE)
#undef VALUE
          .export_values()
          .def_property_readonly(
              "value", [](BinopKind kind) { return BinopKindFormat(kind); })
          .def(py::init([](std::string_view s) {
            return BinopKindFromString(s).value();
          }));
}  // NOLINT(readability/fn_size)

}  // namespace xls::dslx
