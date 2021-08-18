// Copyright 2021 The XLS Authors
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

#include "xls/public/runtime_build_actions.h"

#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "xls/common/status/statusor_pybind_caster.h"

namespace py = pybind11;

namespace xls {

PYBIND11_MODULE(runtime_build_actions, m) {
  ImportStatusModule();

  m.def(
      "convert_dslx_to_ir",
      [](absl::string_view dslx, absl::string_view path,
         absl::string_view package,
         const std::vector<absl::string_view>& additional_search_paths)
          -> absl::StatusOr<std::string> {
        return ConvertDslxToIr(
            dslx, path, package,
            std::vector<std::filesystem::path>(additional_search_paths.begin(),
                                               additional_search_paths.end()));
      },
      R"(Converts the specified DSLX text into XLS IR text.

Args:
 dslx: DSL (module) text to convert to IR.
 path: Path to use for source location information for the given text.
   Since text may be generated, an empty string or a pseudo path like
   "<generated>" is acceptable.
 module_name: Name of the DSL module, will be used in the name of the
   converted IR package text.
 additional_search_paths: Additional filesystem paths to search for imported
   modules.)",
      py::arg("dslx"), py::arg("path"), py::arg("package"),
      py::arg("additional_search_paths"));

  m.def(
      "convert_dslx_path_to_ir",
      [](absl::string_view path,
         const std::vector<absl::string_view>& additional_search_paths)
          -> absl::StatusOr<std::string> {
        return ConvertDslxPathToIr(path, std::vector<std::filesystem::path>(
                                             additional_search_paths.begin(),
                                             additional_search_paths.end()));
      },
      R"(As convert_dslx_to_ir, but uses a filesystem path to retrieve the DSLX module contents.
"path" should end with ".x" suffix, the path will determine the module name.)",
      py::arg("path"), py::arg("additional_search_paths"));

  m.def(
      "optimize_ir", &OptimizeIr,
      R"(Optimizes the generated XLS IR with the given entry point (which should be a
function present inside the IR text).)",
      py::arg("ir"), py::arg("entry"));

  m.def(
      "mangle_dslx_name", &MangleDslxName,
      R"(Mangles the given DSL module/function name combination so it can be resolved
as a corresponding symbol in converted IR.)",
      py::arg("module_name"), py::arg("function_name"));

  m.def("proto_to_dslx", &ProtoToDslx,
        R"(Converts protocol buffer data into its equivalent DSLX text, as a
module-level constant.

Note: this currently only supports "standalone" protobuf schemas; i.e. this
cannot translate schemas that import other `.proto` files. If this limitation
affects you, please file an issue at `github.com/google/xls/issues`

Args:
 proto_def: Protobuf schema (e.g. contents of `.proto` file).
 message_name: Name of the message type (inside the protobuf schema) to emit.
 text_proto: Protobuf text to translate to DSLX.
 binding_name: Name of the (const) DSLX binding (i.e. const variable name) to
   make in the output text.)",
        py::arg("proto_def"), py::arg("message_name"), py::arg("text_proto"),
        py::arg("binding_name"));

  m.def("convert_ir_to_combinational_verilog", &ConvertIrToCombinationalVerilog, R"(Converts the specified XLS IR text into a combinational Verilog module.
 Args:
  ir: Text for the XLS IR package.
  path: Path to use for source location information for the given text.
    Since text may be generated, an empty string or a pseudo path like
    "<generated>" is acceptable.
  entry: Entry function for the XLS IR package,
    if empty entry function of the package is used.
  module_name: Name of the Verilog module,
    if empty use the mangled IR function name.)",
        py::arg("ir"), py::arg("path"), py::arg("entry"),
        py::arg("module_name"));
}

}  // namespace xls
