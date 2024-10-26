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

#include <filesystem>  // NOLINT
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_split.h"
#include "absl/types/span.h"
#include "xls/common/exit_status.h"
#include "xls/common/file/filesystem.h"
#include "xls/common/init_xls.h"
#include "xls/common/status/status_macros.h"
#include "xls/dslx/cpp_transpiler/cpp_transpiler.h"
#include "xls/dslx/cpp_transpiler/cpp_type_generator.h"
#include "xls/dslx/create_import_data.h"
#include "xls/dslx/default_dslx_stdlib_path.h"
#include "xls/dslx/import_data.h"
#include "xls/dslx/parse_and_typecheck.h"
#include "xls/dslx/warning_kind.h"

ABSL_FLAG(std::string, output_header_path, "",
          "Path at which to write the generated header.");
ABSL_FLAG(std::string, output_source_path, "",
          "Path at which to write the generated source.");
ABSL_FLAG(std::string, namespaces, "",
          "Double-colon-delimited namespaces with which to wrap the "
          "generated code, e.g., \"my::namespace\" or "
          "\"::my::explicitly::top::level::namespace\".");
ABSL_FLAG(std::string, dslx_stdlib_path,
          std::string(xls::kDefaultDslxStdlibPath),
          "Path to DSLX standard library");
ABSL_FLAG(std::string, dslx_path, "",
          "Additional paths to search for modules (colon delimited).");

namespace xls {
namespace dslx {
namespace {

static constexpr std::string_view kUsage = R"(
Parses the specified module and emits corresponding C++ declarations for the
types therein. For each encountered struct type, functions are provided to
create a struct from a corresponding XLS Value and vice-versa.

At present, only a single module file is supported (i.e., no colon refs to other
modules).
)";

absl::Status RealMain(const std::filesystem::path& module_path,
                      const std::filesystem::path& dslx_stdlib_path,
                      absl::Span<const std::filesystem::path> dslx_paths,
                      std::string_view output_header_path,
                      std::string_view output_source_path,
                      std::string_view namespaces) {
  XLS_ASSIGN_OR_RETURN(std::string module_text, GetFileContents(module_path));

  ImportData import_data(CreateImportData(
      dslx_stdlib_path, /*additional_search_paths=*/dslx_paths,
      kDefaultWarningsSet, std::make_unique<RealFilesystem>()));
  XLS_ASSIGN_OR_RETURN(TypecheckedModule module,
                       ParseAndTypecheck(module_text, std::string(module_path),
                                         "source", &import_data));
  XLS_ASSIGN_OR_RETURN(
      CppSource sources,
      TranspileToCpp(module.module, &import_data, output_header_path,
                     std::string(namespaces)));

  XLS_RETURN_IF_ERROR(SetFileContents(output_header_path, sources.header));
  XLS_RETURN_IF_ERROR(SetFileContents(output_source_path, sources.source));
  return absl::OkStatus();
}

}  // namespace
}  // namespace dslx
}  // namespace xls

int main(int argc, char* argv[]) {
  std::vector<std::string_view> args =
      xls::InitXls(xls::dslx::kUsage, argc, argv);
  if (args.size() != 1) {
    LOG(QFATAL) << "A single module file must be specified.";
  }

  std::string output_header_path = absl::GetFlag(FLAGS_output_header_path);
  QCHECK(!output_header_path.empty())
      << "--output_header_path must be specified.";
  std::string output_source_path = absl::GetFlag(FLAGS_output_source_path);
  QCHECK(!output_source_path.empty())
      << "--output_source_path must be specified.";

  std::string dslx_path = absl::GetFlag(FLAGS_dslx_path);
  std::vector<std::string> dslx_path_strs = absl::StrSplit(dslx_path, ':');

  std::vector<std::filesystem::path> dslx_paths;
  dslx_paths.reserve(dslx_path_strs.size());
  for (const auto& path : dslx_path_strs) {
    dslx_paths.push_back(std::filesystem::path(path));
  }

  return xls::ExitStatus(xls::dslx::RealMain(
      args[0], absl::GetFlag(FLAGS_dslx_stdlib_path), dslx_paths,
      output_header_path, output_source_path, absl::GetFlag(FLAGS_namespaces)));

  return 0;
}
