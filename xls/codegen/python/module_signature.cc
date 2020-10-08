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

#include "xls/codegen/module_signature.h"

#include "pybind11/pybind11.h"

namespace py = pybind11;

namespace xls {
namespace verilog {

PYBIND11_MODULE(module_signature, m) {
  py::class_<ModuleSignatureProto>(m, "ModuleSignatureProto");  // NOLINT

  py::class_<ModuleSignature>(m, "ModuleSignature")
      .def("as_text_proto", &ModuleSignature::AsTextProto);

  py::class_<ModuleGeneratorResult>(m, "ModuleGeneratorResult")
      .def_readonly("verilog_text", &ModuleGeneratorResult::verilog_text)
      .def_readonly("signature", &ModuleGeneratorResult::signature);
}

}  // namespace verilog
}  // namespace xls
