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

#include <cstdlib>
#include <filesystem>  // NOLINT
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/status/status.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"
#include "absl/types/span.h"
#include "xls/common/exit_status.h"
#include "xls/common/init_xls.h"
#include "xls/common/logging/logging.h"
#include "xls/common/status/status_macros.h"
#include "xls/dslx/default_dslx_stdlib_path.h"
#include "xls/dslx/ir_convert/convert_options.h"
#include "xls/dslx/ir_convert/ir_converter.h"
#include "xls/dslx/warning_kind.h"
#include "xls/ir/package.h"

// LINT.IfChange
ABSL_FLAG(std::string, top, "",
          "The name of the top entity. When provided, the function/proc is the "
          "top entity in the generated IR. When not provided, all functions "
          "and procs are converted, there is no top entity defined in the "
          "generated IR.");
ABSL_FLAG(std::string, stdlib_path, xls::kDefaultDslxStdlibPath,
          "Path to DSLX standard library files.");
ABSL_FLAG(std::string, dslx_path, "",
          "Additional paths to search for modules (colon delimited).");
ABSL_FLAG(
    std::string, package_name, "",
    "Package name to use for output (required when multiple input .x files "
    "are given).");

ABSL_FLAG(bool, emit_fail_as_assert, true,
          "Feature flag for emitting fail!() in the DSL as an assert IR op.");
ABSL_FLAG(bool, verify, true,
          "If true, verifies the generated IR for correctness.");

ABSL_FLAG(std::string, disable_warnings, "",
          "Comma-delimited list of warnings to disable -- not generally "
          "recommended, but can be used in exceptional circumstances");
ABSL_FLAG(bool, warnings_as_errors, true,
          "Whether to fail early, as an error, if warnings are detected");
// LINT.ThenChange(//xls/build_rules/xls_ir_rules.bzl)

namespace xls::dslx {
namespace {

const char* kUsage = R"(
Converts a DSLX input file (and optional entry point) to XLS IR.

Successfully converted XLS IR is printed to stdout; errors are printed to
stderr.

Example invocation for a particular function:

  ir_converter_main -entry best_function path/to/frobulator.x

If no entry point is given all functions within the module are converted:

  ir_converter_main path/to/frobulator.x
)";

absl::Status RealMain(absl::Span<const std::string_view> paths,
                      std::optional<std::string_view> top,
                      std::optional<std::string_view> package_name,
                      const std::string& stdlib_path,
                      absl::Span<const std::filesystem::path> dslx_paths,
                      bool emit_fail_as_assert, bool verify_ir,
                      bool warnings_as_errors, bool* printed_error) {
  XLS_ASSIGN_OR_RETURN(
      WarningKindSet enabled_warnings,
      WarningKindSetFromDisabledString(absl::GetFlag(FLAGS_disable_warnings)));
  const ConvertOptions convert_options = {
      .emit_positions = true,
      .emit_fail_as_assert = emit_fail_as_assert,
      .verify_ir = verify_ir,
      .warnings_as_errors = warnings_as_errors,
      .enabled_warnings = enabled_warnings,
  };

  // The following checks are performed inside ConvertFilesToPackage(), but we
  // reproduce them here to give nicer error messages.
  if (!package_name.has_value()) {
    QCHECK_EQ(paths.size(), 1)
        << "-package_name *must* be given when multiple input paths are "
           "supplied";
  }
  if (paths.size() > 1) {
    QCHECK(!top.has_value())
        << "-entry cannot be supplied with multiple input paths (need a single "
           "input path to know where to resolve the entry function)";
  }

  XLS_ASSIGN_OR_RETURN(
      std::unique_ptr<xls::Package> package,
      ConvertFilesToPackage(paths, stdlib_path, dslx_paths, convert_options,
                            /*top=*/top,
                            /*package_name=*/package_name, printed_error));
  std::cout << package->DumpIr();

  return absl::OkStatus();
}

}  // namespace
}  // namespace xls::dslx

int main(int argc, char* argv[]) {
  std::vector<std::string_view> args =
      xls::InitXls(xls::dslx::kUsage, argc, argv);
  if (args.empty()) {
    XLS_LOG(QFATAL) << "Wrong number of command-line arguments; got "
                    << args.size() << ": `" << absl::StrJoin(args, " ")
                    << "`; want " << argv[0] << " <input-file>";
  }
  // "-" is a special path that is shorthand for /dev/stdin. Update here as
  // there isn't a better place later.
  for (auto& arg : args) {
    if (arg == "-") {
      arg = "/dev/stdin";
    }
  }

  std::string stdlib_path = absl::GetFlag(FLAGS_stdlib_path);
  std::string dslx_path = absl::GetFlag(FLAGS_dslx_path);
  std::vector<std::string> dslx_path_strs = absl::StrSplit(dslx_path, ':');

  std::vector<std::filesystem::path> dslx_paths;
  dslx_paths.reserve(dslx_path_strs.size());
  for (const auto& path : dslx_path_strs) {
    dslx_paths.push_back(std::filesystem::path(path));
  }

  std::optional<std::string> top;
  if (!absl::GetFlag(FLAGS_top).empty()) {
    top = absl::GetFlag(FLAGS_top);
  }

  std::optional<std::string> package_name;
  if (!absl::GetFlag(FLAGS_package_name).empty()) {
    package_name = absl::GetFlag(FLAGS_package_name);
  }

  bool emit_fail_as_assert = absl::GetFlag(FLAGS_emit_fail_as_assert);
  bool verify_ir = absl::GetFlag(FLAGS_verify);
  bool warnings_as_errors = absl::GetFlag(FLAGS_warnings_as_errors);
  bool printed_error = false;
  absl::Status status = xls::dslx::RealMain(
      args, top, package_name, stdlib_path, dslx_paths, emit_fail_as_assert,
      verify_ir, warnings_as_errors, &printed_error);
  if (printed_error) {
    return EXIT_FAILURE;
  }
  return xls::ExitStatus(status);
}
