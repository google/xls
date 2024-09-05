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
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"
#include "absl/types/span.h"
#include "google/protobuf/text_format.h"
#include "xls/common/exit_status.h"
#include "xls/common/file/filesystem.h"
#include "xls/common/init_xls.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/dslx/default_dslx_stdlib_path.h"
#include "xls/dslx/ir_convert/conversion_info.h"
#include "xls/dslx/ir_convert/convert_options.h"
#include "xls/dslx/ir_convert/ir_converter.h"
#include "xls/dslx/warning_kind.h"
#include "xls/ir/package.h"

ABSL_FLAG(std::optional<std::string>, output_file, std::nullopt,
          "Where to write the ir file. Defaults to stdout");
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
ABSL_FLAG(bool, convert_tests, false,
          "Feature flag for emitting test procs/functions to IR.");
ABSL_FLAG(bool, verify, true,
          "If true, verifies the generated IR for correctness.");

ABSL_FLAG(std::string, disable_warnings, "",
          "Comma-delimited list of warnings to disable -- not generally "
          "recommended, but can be used in exceptional circumstances");
ABSL_FLAG(bool, warnings_as_errors, true,
          "Whether to fail early, as an error, if warnings are detected");
ABSL_FLAG(std::optional<std::string>, interface_proto_file, std::nullopt,
          "File to store a xls.PackageInterfaceProto containing extern type "
          "information and interface specs");
// LINT.ThenChange(//xls/build_rules/xls_ir_rules.bzl)
ABSL_FLAG(std::optional<std::string>, interface_textproto_file, std::nullopt,
          "File to store a xls.PackageInterfaceProto containing extern type "
          "information and interface specs in textproto format");

namespace xls::dslx {
namespace {

static constexpr std::string_view kUsage = R"(
Converts a DSLX input file (and optional entry point) to XLS IR.

Successfully converted XLS IR is printed to stdout; errors are printed to
stderr.

Example invocation for a particular function:

  ir_converter_main -entry best_function path/to/frobulator.x

If no entry point is given all functions within the module are converted:

  ir_converter_main path/to/frobulator.x
)";

absl::Status RealMain(
    std::optional<std::filesystem::path> output_file,
    absl::Span<const std::string_view> paths,
    std::optional<std::string_view> top,
    std::optional<std::string_view> package_name,
    const std::string& stdlib_path,
    absl::Span<const std::filesystem::path> dslx_paths,
    bool emit_fail_as_assert, bool verify_ir, bool warnings_as_errors,
    bool* printed_error,
    std::optional<std::filesystem::path> interface_proto_file,
    std::optional<std::filesystem::path> interface_textproto_file,
    bool convert_tests) {
  XLS_ASSIGN_OR_RETURN(
      WarningKindSet enabled_warnings,
      WarningKindSetFromDisabledString(absl::GetFlag(FLAGS_disable_warnings)));
  const ConvertOptions convert_options = {
      .emit_positions = true,
      .emit_fail_as_assert = emit_fail_as_assert,
      .verify_ir = verify_ir,
      .warnings_as_errors = warnings_as_errors,
      .enabled_warnings = enabled_warnings,
      .convert_tests = convert_tests,
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
      PackageConversionData result,
      ConvertFilesToPackage(paths, stdlib_path, dslx_paths, convert_options,
                            /*top=*/top,
                            /*package_name=*/package_name, printed_error));
  if (output_file) {
    XLS_RETURN_IF_ERROR(SetFileContents(*output_file, result.DumpIr()));
  } else {
    std::cout << result.package->DumpIr();
  }
  if (interface_proto_file) {
    XLS_RETURN_IF_ERROR(SetFileContents(*interface_proto_file,
                                        result.interface.SerializeAsString()));
  }
  if (interface_textproto_file) {
    std::string res;
    XLS_RET_CHECK(google::protobuf::TextFormat::PrintToString(result.interface, &res));
    XLS_RETURN_IF_ERROR(SetFileContents(*interface_textproto_file, res));
  }

  return absl::OkStatus();
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
  // "-" is a special path that is shorthand for /dev/stdin. Update here as
  // there isn't a better place later.
  for (auto& arg : args) {
    if (arg == "-") {
      arg = "/dev/stdin";
    }
  }

  std::optional<std::filesystem::path> output_file =
      absl::GetFlag(FLAGS_output_file)
          ? std::make_optional<std::filesystem::path>(
                *absl::GetFlag(FLAGS_output_file))
          : std::nullopt;

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
      output_file, args, top, package_name, stdlib_path, dslx_paths,
      emit_fail_as_assert, verify_ir, warnings_as_errors, &printed_error,
      absl::GetFlag(FLAGS_interface_proto_file)
          ? std::make_optional<std::filesystem::path>(
                *absl::GetFlag(FLAGS_interface_proto_file))
          : std::nullopt,
      absl::GetFlag(FLAGS_interface_textproto_file)
          ? std::make_optional<std::filesystem::path>(
                *absl::GetFlag(FLAGS_interface_textproto_file))
          : std::nullopt,
      absl::GetFlag(FLAGS_convert_tests));
  if (printed_error) {
    return EXIT_FAILURE;
  }
  return xls::ExitStatus(status);
}
