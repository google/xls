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

#include "xls/ir/ir_parser.h"

#include "absl/status/statusor.h"
#include "pybind11/pybind11.h"
#include "pybind11_abseil/absl_casters.h"
#include "pybind11_abseil/statusor_caster.h"
#include "xls/common/status/import_status_module.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/python/wrapper_types.h"

namespace py = pybind11;

namespace xls {

PYBIND11_MODULE(ir_parser, m) {
  ImportStatusModule();
  py::module::import("xls.ir.python.package");
  py::module::import("xls.ir.python.type");
  py::module::import("xls.ir.python.value");

  py::class_<Parser>(m, "Parser")
      .def_static(
          "parse_package",
          [](std::string_view input_string,
             std::optional<std::string_view> filename)
              -> absl::StatusOr<PackageHolder> {
            XLS_ASSIGN_OR_RETURN(std::unique_ptr<Package> package,
                                 Parser::ParsePackage(input_string, filename));
            std::shared_ptr<Package> shared_package = std::move(package);
            return PackageHolder(shared_package.get(), shared_package);
          },
          py::arg("input_string"), py::arg("filename") = absl::nullopt)

      .def_static("parse_value", PyWrap(&Parser::ParseValue),
                  py::arg("input_string"), py::arg("expected_type"))

      .def_static("parse_typed_value", &Parser::ParseTypedValue,
                  py::arg("input_string"));
}

}  // namespace xls
