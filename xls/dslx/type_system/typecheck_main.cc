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

#include <filesystem>  // NOLINT
#include <iostream>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"
#include "absl/types/span.h"
#include "xls/common/exit_status.h"
#include "xls/common/file/filesystem.h"
#include "xls/common/init_xls.h"
#include "xls/common/logging/logging.h"
#include "xls/common/status/status_macros.h"
#include "xls/dslx/command_line_utils.h"
#include "xls/dslx/create_import_data.h"
#include "xls/dslx/default_dslx_stdlib_path.h"
#include "xls/dslx/import_data.h"
#include "xls/dslx/parse_and_typecheck.h"
#include "xls/dslx/type_system/type_info_to_proto.h"
#include "xls/dslx/warning_kind.h"

ABSL_FLAG(std::string, dslx_path, "",
          "Additional paths to search for modules (colon delimited).");
ABSL_FLAG(std::string, dslx_stdlib_path, xls::kDefaultDslxStdlibPath,
          "Path to DSLX standard library");
ABSL_FLAG(std::string, output_path, "",
          "Path to dump the type information to as a protobin -- if not "
          "provided textual proto is given on stdout.");

namespace xls::dslx {
namespace {

const char* kUsage = R"(
Parses and typechecks the specified module and emits the type information that
was deduced.
)";

absl::Status RealMain(absl::Span<const std::filesystem::path> dslx_paths,
                      const std::filesystem::path& dslx_stdlib_path,
                      const std::filesystem::path& input_path,
                      std::optional<std::filesystem::path> output_path) {
  ImportData import_data(CreateImportData(
      dslx_stdlib_path,
      /*additional_search_paths=*/dslx_paths, kAllWarningsSet));
  XLS_ASSIGN_OR_RETURN(std::string input_contents, GetFileContents(input_path));
  XLS_ASSIGN_OR_RETURN(std::string module_name, PathToName(input_path.c_str()));
  absl::StatusOr<TypecheckedModule> tm_or = ParseAndTypecheck(
      input_contents, input_path.c_str(), module_name, &import_data);
  if (!tm_or.ok()) {
    if (TryPrintError(tm_or.status())) {
      return absl::InvalidArgumentError(
          "An error occurred during parsing / typechecking.");
    }
    return tm_or.status();
  }
  XLS_ASSIGN_OR_RETURN(TypeInfoProto tip, TypeInfoToProto(*tm_or->type_info));
  if (output_path.has_value()) {
    std::string output;
    QCHECK(tip.SerializeToString(&output));
    return SetFileContents(output_path->c_str(), output);
  }
  XLS_ASSIGN_OR_RETURN(std::string humanized, ToHumanString(tip, import_data));
  std::cout << humanized << '\n';
  return absl::OkStatus();
}

}  // namespace
}  // namespace xls::dslx

int main(int argc, char* argv[]) {
  std::vector<std::string_view> args =
      xls::InitXls(xls::dslx::kUsage, argc, argv);
  if (args.size() != 1) {
    XLS_LOG(QFATAL) << "Wrong number of command-line arguments; got "
                    << args.size() << ": `" << absl::StrJoin(args, " ")
                    << "`; want " << argv[0] << " <input-file>";
  }

  std::filesystem::path input_path(args[0]);
  std::optional<std::string> output_path;
  if (std::string flag = absl::GetFlag(FLAGS_output_path); !flag.empty()) {
    output_path = flag;
  }

  std::string dslx_path = absl::GetFlag(FLAGS_dslx_path);
  std::vector<std::string> dslx_path_strs = absl::StrSplit(dslx_path, ':');
  std::vector<std::filesystem::path> dslx_paths;
  dslx_paths.reserve(dslx_path_strs.size());
  for (const auto& path : dslx_path_strs) {
    dslx_paths.push_back(std::filesystem::path(path));
  }

  std::filesystem::path dslx_stdlib_path(absl::GetFlag(FLAGS_dslx_stdlib_path));

  return xls::ExitStatus(xls::dslx::RealMain(dslx_paths, dslx_stdlib_path,
                                             input_path, output_path));
}
