// Copyright 2026 The XLS Authors
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

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
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
#include "xls/dslx/import_data.h"
#include "xls/dslx/parse_and_typecheck.h"
#include "xls/dslx/tabulate.h"
#include "xls/dslx/virtualizable_file_system.h"
#include "xls/dslx/warning_collector.h"
#include "xls/dslx/warning_kind.h"

ABSL_FLAG(std::string, function, "",
          "Name of the #[tabulate] function to evaluate (required).");
ABSL_FLAG(std::string, output_path, "",
          "Optional file path to write the generated Markdown table; if "
          "empty, writes to stdout.");
ABSL_FLAG(std::string, dslx_path, "",
          "Additional paths to search for modules (colon delimited).");
ABSL_FLAG(std::string, dslx_stdlib_path,
          std::string(xls::kDefaultDslxStdlibPath),
          "Path to DSLX standard library directory.");
ABSL_FLAG(std::string, disable_warnings, "",
          "Comma-delimited list of warnings to disable.");
ABSL_FLAG(std::string, enable_warnings, "",
          "Comma-delimited list of warnings to enable.");
ABSL_FLAG(bool, warnings_as_errors, true,
          "Whether to fail early if warnings are encountered.");

namespace xls::dslx {
namespace {

constexpr std::string_view kUsage = R"(
Evaluates a DSLX function annotated with #[tabulate] across its input domain
and emits a human-readable Markdown table.

Usage:
  dslx_to_md_main --function=<fn_name> [--output_path=<md_file>] <dslx_file>
)";

absl::Status RealMain(std::filesystem::path entry_module_path,
                      std::string_view fn_name, std::string_view output_path,
                      absl::Span<const std::filesystem::path> dslx_paths,
                      const std::filesystem::path& dslx_stdlib_path,
                      bool warnings_as_errors) {
  XLS_ASSIGN_OR_RETURN(
      WarningKindSet warnings,
      GetWarningsSetFromFlags(absl::GetFlag(FLAGS_enable_warnings),
                              absl::GetFlag(FLAGS_disable_warnings)));
  ImportData import_data =
      CreateImportData(dslx_stdlib_path, dslx_paths, warnings,
                       std::make_unique<RealFilesystem>());
  WarningCollector warning_collector(warnings);
  XLS_ASSIGN_OR_RETURN(std::string program,
                       import_data.vfs().GetFileContents(entry_module_path));
  XLS_ASSIGN_OR_RETURN(std::string module_name,
                       PathToName(entry_module_path.c_str()));

  absl::StatusOr<TypecheckedModule> tm = ParseAndTypecheck(
      program, entry_module_path.c_str(), module_name, &import_data);
  if (!tm.ok()) {
    TryPrintError(tm.status(), import_data.file_table(), import_data.vfs());
    return tm.status();
  }

  PrintWarnings(tm->warnings, import_data.file_table(), import_data.vfs());
  if (warnings_as_errors && !tm->warnings.empty()) {
    return absl::InvalidArgumentError(
        "Warnings encountered and warnings-as-errors is enabled.");
  }

  XLS_ASSIGN_OR_RETURN(std::string markdown,
                       TabulateFunction(&import_data, *tm, fn_name));

  if (!output_path.empty()) {
    return SetFileContents(output_path, markdown);
  }
  std::cout << markdown;
  return absl::OkStatus();
}

}  // namespace
}  // namespace xls::dslx

int main(int argc, char* argv[]) {
  std::vector<std::string_view> args =
      xls::InitXls(xls::dslx::kUsage, argc, argv);
  if (args.size() != 1) {
    std::cerr << "Exactly one input DSLX file must be specified.\n"
              << xls::dslx::kUsage;
    return EXIT_FAILURE;
  }

  std::string fn_name = absl::GetFlag(FLAGS_function);
  if (fn_name.empty()) {
    std::cerr << "--function must be specified.\n" << xls::dslx::kUsage;
    return EXIT_FAILURE;
  }

  std::string dslx_path = absl::GetFlag(FLAGS_dslx_path);
  std::vector<std::string> dslx_path_strs = absl::StrSplit(dslx_path, ':');
  std::vector<std::filesystem::path> dslx_paths;
  dslx_paths.reserve(dslx_path_strs.size());
  for (const auto& path : dslx_path_strs) {
    if (!path.empty()) {
      dslx_paths.push_back(std::filesystem::path(path));
    }
  }

  std::filesystem::path dslx_stdlib_path =
      absl::GetFlag(FLAGS_dslx_stdlib_path);
  bool warnings_as_errors = absl::GetFlag(FLAGS_warnings_as_errors);

  return xls::ExitStatus(
      xls::dslx::RealMain(args[0], fn_name, absl::GetFlag(FLAGS_output_path),
                          dslx_paths, dslx_stdlib_path, warnings_as_errors));
}
