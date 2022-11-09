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

#include <string>

#include "absl/flags/flag.h"
#include "absl/status/status.h"
#include "absl/strings/str_split.h"
#include "xls/common/file/filesystem.h"
#include "xls/common/init_xls.h"
#include "xls/dslx/command_line_utils.h"
#include "xls/dslx/create_import_data.h"
#include "xls/dslx/default_dslx_stdlib_path.h"
#include "xls/dslx/error_printer.h"
#include "xls/dslx/import_data.h"
#include "xls/dslx/ir_converter.h"
#include "xls/dslx/parser.h"
#include "xls/dslx/scanner.h"
#include "xls/dslx/typecheck.h"

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

absl::StatusOr<std::unique_ptr<Module>> ParseText(std::string_view text,
                                                  std::string_view module_name,
                                                  bool print_on_error,
                                                  std::string_view filename,
                                                  bool* printed_error) {
  Scanner scanner{std::string(filename), std::string(text)};
  Parser parser(std::string(module_name), &scanner);
  absl::StatusOr<std::unique_ptr<Module>> module_or = parser.ParseModule();
  *printed_error = TryPrintError(module_or.status());
  return module_or;
}

// Adds IR-converted symbols from the module specified by "path" to the given
// "package".
static absl::Status AddPathToPackage(
    std::string_view path, std::optional<std::string_view> entry,
    const ConvertOptions& convert_options, std::string stdlib_path,
    absl::Span<const std::filesystem::path> dslx_paths, Package* package,
    bool warnings_as_errors, bool* printed_error) {
  // Read the `.x` contents.
  XLS_ASSIGN_OR_RETURN(std::string text, GetFileContents(path));
  // Figure out what we name this module.
  XLS_ASSIGN_OR_RETURN(std::string module_name, PathToName(path));
  // Parse the module text.
  XLS_ASSIGN_OR_RETURN(std::unique_ptr<Module> module,
                       ParseText(text, module_name, /*print_on_error=*/true,
                                 /*filename=*/path, printed_error));

  // TODO(leary): 2021-07-21 We should be able to reuse the type checking if
  // there are overlapping nodes in the module DAG between files to process. For
  // now we throw it away for each file and re-derive it (we need to refactor to
  // make the modules outlive any given AddPathToPackage() if we want to
  // appropriately reuse things in ImportData).
  ImportData import_data(CreateImportData(std::move(stdlib_path), dslx_paths));
  WarningCollector warnings;
  absl::StatusOr<TypeInfo*> type_info_or =
      CheckModule(module.get(), &import_data, &warnings);
  if (!type_info_or.ok()) {
    *printed_error = TryPrintError(type_info_or.status());
    return type_info_or.status();
  }

  if (warnings_as_errors && !warnings.warnings().empty()) {
    *printed_error = true;
    PrintWarnings(warnings);
    return absl::InvalidArgumentError(
        "Warnings encountered and warnings-as-errors set.");
  }

  if (entry.has_value()) {
    XLS_RETURN_IF_ERROR(ConvertOneFunctionIntoPackage(
        module.get(), entry.value(), /*import_data=*/&import_data,
        /*symbolic_bindings=*/nullptr, convert_options, package));
  } else {
    XLS_RETURN_IF_ERROR(
        ConvertModuleIntoPackage(module.get(), &import_data, convert_options,
                                 /*traverse_tests=*/false, package));
  }
  return absl::OkStatus();
}

absl::Status RealMain(absl::Span<const std::string_view> paths,
                      std::optional<std::string_view> top,
                      std::optional<std::string_view> package_name,
                      const std::string& stdlib_path,
                      absl::Span<const std::filesystem::path> dslx_paths,
                      bool emit_fail_as_assert, bool verify_ir,
                      bool warnings_as_errors, bool* printed_error) {
  std::optional<xls::Package> package;
  if (package_name.has_value()) {
    package.emplace(package_name.value());
  } else {
    XLS_QCHECK_EQ(paths.size(), 1)
        << "-package_name *must* be given when multiple input paths are "
           "supplied";
    // Get it from the one module name (if package name was unspecified and we
    // just have one path).
    XLS_ASSIGN_OR_RETURN(std::string module_name, PathToName(paths[0]));
    package.emplace(module_name);
  }

  if (paths.size() > 1) {
    XLS_QCHECK(!top.has_value())
        << "-entry cannot be supplied with multiple input paths (need a single "
           "input path to know where to resolve the entry function)";
  }

  const ConvertOptions convert_options = {
      .emit_positions = true,
      .emit_fail_as_assert = emit_fail_as_assert,
      .verify_ir = verify_ir,
  };
  for (std::string_view path : paths) {
    if (path == "-") {
      path = "/dev/stdin";
    }
    XLS_RETURN_IF_ERROR(
        AddPathToPackage(path, top, convert_options, stdlib_path, dslx_paths,
                         &package.value(), warnings_as_errors, printed_error));
  }
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
  XLS_QCHECK_OK(status);
  return EXIT_SUCCESS;
}
