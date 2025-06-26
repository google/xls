// Copyright 2025 The XLS Authors
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
#include <utility>
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
#include "xls/common/init_xls.h"
#include "xls/common/status/status_macros.h"
#include "xls/dslx/command_line_utils.h"
#include "xls/dslx/create_import_data.h"
#include "xls/dslx/default_dslx_stdlib_path.h"
#include "xls/dslx/parse_and_typecheck.h"
#include "xls/dslx/virtualizable_file_system.h"
#include "xls/dslx/warning_kind.h"

// TODO: https://github.com/google/xls/issues/2498 - Consider consolidating
// these common flags.
ABSL_FLAG(std::string, dslx_path, "",
          "Additional paths to search for modules (colon delimited).");
ABSL_FLAG(std::string, dslx_stdlib_path,
          std::string(xls::kDefaultDslxStdlibPath),
          "Path to DSLX standard library directory.");
ABSL_FLAG(std::string, disable_warnings, "",
          "Comma-delimited list of warnings to disable from the default set of "
          "warnings used -- not generally "
          "recommended, but can be used in exceptional circumstances");
ABSL_FLAG(std::string, enable_warnings, "",
          "Comma-delimited list of warnings to enable that are disabled in the "
          "default set");
ABSL_FLAG(bool, warnings_as_errors, true,
          "Whether to fail early, as an error, if warnings are detected");
ABSL_FLAG(bool, type_inference_v2, false,
          "Whether to use type system v2 when type checking the input.");

namespace xls::dslx {
namespace {

static constexpr std::string_view kUsage = R"(
Parses and typechecks a DSLX module.
)";

absl::Status RealMain(std::string_view entry_module_path,
                      absl::Span<const std::filesystem::path> dslx_paths,
                      const std::filesystem::path& dslx_stdlib_path,
                      bool warnings_as_errors) {
  XLS_ASSIGN_OR_RETURN(
      WarningKindSet warnings,
      GetWarningsSetFromFlags(absl::GetFlag(FLAGS_enable_warnings),
                              absl::GetFlag(FLAGS_disable_warnings)));
  bool type_inference_v2 = absl::GetFlag(FLAGS_type_inference_v2);

  std::unique_ptr<VirtualizableFilesystem> vfs =
      std::make_unique<RealFilesystem>();

  XLS_ASSIGN_OR_RETURN(std::string program,
                       vfs->GetFileContents(entry_module_path));
  XLS_ASSIGN_OR_RETURN(std::string module_name, PathToName(entry_module_path));

  ParseAndTypecheckOptions options = {.dslx_stdlib_path = dslx_stdlib_path,
                                      .dslx_paths = dslx_paths,
                                      .type_inference_v2 = type_inference_v2,
                                      .warnings_as_errors = warnings_as_errors,
                                      .warnings = warnings};

  auto import_data =
      CreateImportData(options.dslx_stdlib_path, options.dslx_paths,
                       options.warnings, std::move(vfs));

  absl::StatusOr<TypecheckedModule> tm =
      ParseAndTypecheck(program, entry_module_path, module_name, &import_data,
                        nullptr, options.type_inference_v2);
  if (!tm.ok()) {
    TryPrintError(tm.status(), import_data.file_table(), import_data.vfs());
  }
  return tm.status();
}

}  // namespace
}  // namespace xls::dslx

int main(int argc, char* argv[]) {
  std::vector<std::string_view> args =
      xls::InitXls(xls::dslx::kUsage, argc, argv);
  if (args.empty()) {
    LOG(QFATAL) << "Wrong number of command-line arguments; got " << args.size()
                << ": `" << absl::StrJoin(args, " ") << "`; want " << argv[0]
                << " <input-file>";
  }
  std::string dslx_path = absl::GetFlag(FLAGS_dslx_path);
  std::vector<std::string> dslx_path_strs = absl::StrSplit(dslx_path, ':');
  std::vector<std::filesystem::path> dslx_paths;
  dslx_paths.reserve(dslx_path_strs.size());
  for (const auto& path : dslx_path_strs) {
    dslx_paths.push_back(std::filesystem::path(path));
  }

  bool warnings_as_errors = absl::GetFlag(FLAGS_warnings_as_errors);

  std::filesystem::path dslx_stdlib_path =
      absl::GetFlag(FLAGS_dslx_stdlib_path);

  absl::Status status = xls::dslx::RealMain(
      args[0], dslx_paths, dslx_stdlib_path, warnings_as_errors);
  return xls::ExitStatus(status);
}
