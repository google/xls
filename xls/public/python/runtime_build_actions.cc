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

  m.def("convert_dslx_to_ir", [](absl::string_view dslx, absl::string_view path, absl::string_view package, const std::vector<absl::string_view>& additional_search_paths) -> absl::StatusOr<std::string> {
      return ConvertDslxToIr(dslx, path, package, std::vector<std::filesystem::path>(additional_search_paths.begin(), additional_search_paths.end()));
    });

  m.def("convert_dslx_path_to_ir", [](absl::string_view path, const std::vector<absl::string_view>& additional_search_paths) -> absl::StatusOr<std::string> {
      return ConvertDslxPathToIr(path, std::vector<std::filesystem::path>(additional_search_paths.begin(), additional_search_paths.end()));
    });

  m.def("optimize_ir", [](absl::string_view ir, absl::string_view entry) -> absl::StatusOr<std::string> {
      return OptimizeIr(ir, entry);
    });

  m.def("mangle_dslx_name", [](absl::string_view module_name, absl::string_view function_name) -> absl::StatusOr<std::string> {
      return MangleDslxName(module_name, function_name);
    });

  m.def("proto_to_dslx", [](absl::string_view proto_def, absl::string_view message_name, absl::string_view text_proto, absl::string_view binding_name) -> absl::StatusOr<std::string> {
      return ProtoToDslx(proto_def, message_name, text_proto, binding_name);
    });
}

}  // namespace xls
