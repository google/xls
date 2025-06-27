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

#include <filesystem>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"
#include "absl/types/span.h"
#include "xls/common/exit_status.h"
#include "xls/common/file/filesystem.h"
#include "xls/common/init_xls.h"
#include "xls/common/status/status_macros.h"
#include "xls/dslx/command_line_utils.h"
#include "xls/dslx/create_import_data.h"
#include "xls/dslx/default_dslx_stdlib_path.h"
#include "xls/dslx/error_printer.h"
#include "xls/dslx/frontend/bindings.h"
#include "xls/dslx/import_data.h"
#include "xls/dslx/parse_and_typecheck.h"
#include "xls/dslx/type_system/type_info.pb.h"
#include "xls/dslx/type_system/type_info_to_proto.h"
#include "xls/dslx/virtualizable_file_system.h"
#include "xls/dslx/warning_kind.h"

ABSL_FLAG(std::string, dslx_path, "",
          "Additional paths to search for modules (colon delimited).");
ABSL_FLAG(std::string, dslx_stdlib_path,
          std::string(xls::kDefaultDslxStdlibPath),
          "Path to DSLX standard library");
ABSL_FLAG(std::string, output_path, "",
          "Path to dump the type information to as a protobin -- if not "
          "provided textual proto is given on stdout.");
ABSL_FLAG(bool, fatal_on_internal_error, false,
          "If true, internal errors will be fatal; this is useful for fuzzing "
          "without using a wrapper to check reported error invariants.");

// Warnings-oriented flags.
ABSL_FLAG(std::string, disable_warnings, "",
          "Comma-delimited list of warnings to disable from the default set of "
          "warnings used -- not generally "
          "recommended, but can be used in exceptional circumstances");
ABSL_FLAG(std::string, enable_warnings, "",
          "Comma-delimited list of warnings to enable that are disabled in the "
          "default set");
ABSL_FLAG(bool, warnings_as_errors, true,
          "Whether to fail early, as an error, if warnings are detected");

namespace xls::dslx {
namespace {

static constexpr std::string_view kUsage = R"(
Parses and typechecks the specified module and emits the type information that
was deduced.
)";

absl::Status RealMain(absl::Span<const std::filesystem::path> dslx_paths,
                      const std::filesystem::path& dslx_stdlib_path,
                      const std::filesystem::path& input_path,
                      std::optional<std::filesystem::path> output_path) {
  XLS_ASSIGN_OR_RETURN(
      WarningKindSet warnings,
      GetWarningsSetFromFlags(absl::GetFlag(FLAGS_enable_warnings),
                              absl::GetFlag(FLAGS_disable_warnings)));

  ImportData import_data(
      CreateImportData(dslx_stdlib_path,
                       /*additional_search_paths=*/dslx_paths, warnings,
                       std::make_unique<RealFilesystem>()));
  XLS_ASSIGN_OR_RETURN(std::string input_contents,
                       import_data.vfs().GetFileContents(input_path));
  XLS_ASSIGN_OR_RETURN(std::string module_name, PathToName(input_path.c_str()));
  absl::StatusOr<TypecheckedModule> tm = ParseAndTypecheck(
      input_contents, input_path.c_str(), module_name, &import_data);
  if (!tm.ok()) {
    if (absl::GetFlag(FLAGS_fatal_on_internal_error) &&
        !GetPositionalErrorData(tm.status(), std::nullopt,
                                import_data.file_table())
             .ok()) {
      LOG(QFATAL) << "Internal error: " << tm.status();
    }
    if (TryPrintError(tm.status(), import_data.file_table(),
                      import_data.vfs())) {
      return absl::InvalidArgumentError(
          "An error occurred during parsing / typechecking.");
    }
    return tm.status();
  }

  if (!tm->warnings.empty()) {
    PrintWarnings(tm->warnings, import_data.file_table(), import_data.vfs());

    if (absl::GetFlag(FLAGS_warnings_as_errors)) {
      return absl::InvalidArgumentError(
          "Warnings were detected; use --warnings_as_errors=false to allow "
          "them or --disable_warnings to disable them selectively.");
    }
  }

  XLS_ASSIGN_OR_RETURN(TypeInfoProto tip, TypeInfoToProto(*tm->type_info));
  if (output_path.has_value()) {
    std::string output;
    QCHECK(tip.SerializeToString(&output));
    return SetFileContents(output_path->c_str(), output);
  }
  XLS_ASSIGN_OR_RETURN(
      std::string humanized,
      ToHumanString(tip, import_data, import_data.file_table()));
  std::cout << humanized << '\n';
  return absl::OkStatus();
}

}  // namespace
}  // namespace xls::dslx

int main(int argc, char* argv[]) {
  std::vector<std::string_view> args =
      xls::InitXls(xls::dslx::kUsage, argc, argv);
  if (args.size() != 1) {
    LOG(QFATAL) << "Wrong number of command-line arguments; got " << args.size()
                << ": `" << absl::StrJoin(args, " ") << "`; want " << argv[0]
                << " <input-file>";
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
