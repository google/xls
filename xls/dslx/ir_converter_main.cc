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
#include "xls/dslx/error_printer.h"
#include "xls/dslx/ir_converter.h"
#include "xls/dslx/parser.h"
#include "xls/dslx/scanner.h"
#include "xls/dslx/typecheck.h"

ABSL_FLAG(std::string, entry, "",
          "Entry function name for conversion; when not given, all functions "
          "are converted.");
ABSL_FLAG(std::string, dslx_path, "",
          "Additional paths to search for modules (colon delimited).");

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

absl::StatusOr<std::unique_ptr<Module>> ParseText(absl::string_view text,
                                                  absl::string_view module_name,
                                                  bool print_on_error,
                                                  absl::string_view filename,
                                                  bool* printed_error) {
  Scanner scanner{std::string(filename), std::string(text)};
  Parser parser(std::string(module_name), &scanner);
  absl::StatusOr<std::unique_ptr<Module>> module_or = parser.ParseModule();
  *printed_error = TryPrintError(module_or.status());
  return module_or;
}

absl::StatusOr<std::string> PathToName(absl::string_view path) {
  std::vector<absl::string_view> pieces = absl::StrSplit(path, '/');
  if (pieces.empty()) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Could not determine module name from path: %s", path));
  }
  absl::string_view last = pieces.back();
  std::vector<absl::string_view> dot_pieces = absl::StrSplit(last, '.');
  XLS_RET_CHECK(!dot_pieces.empty());
  return std::string(dot_pieces[0]);
}

absl::Status RealMain(absl::string_view path,
                      absl::optional<absl::string_view> entry,
                      absl::Span<const std::string> dslx_paths,
                      bool* printed_error) {
  XLS_ASSIGN_OR_RETURN(std::string text, GetFileContents(path));

  XLS_ASSIGN_OR_RETURN(std::string module_name, PathToName(path));
  XLS_ASSIGN_OR_RETURN(std::unique_ptr<Module> module,
                       ParseText(text, module_name, /*print_on_error=*/true,
                                 /*filename=*/path, printed_error));

  ImportCache import_cache;
  absl::StatusOr<TypeInfoOwner> type_info_or =
      CheckModule(module.get(), &import_cache, dslx_paths);
  if (!type_info_or.ok()) {
    *printed_error = TryPrintError(type_info_or.status());
    return type_info_or.status();
  }
  TypeInfo* type_info = type_info_or.value().primary();

  std::string converted;
  if (entry.has_value()) {
    XLS_ASSIGN_OR_RETURN(
        converted, ConvertOneFunction(module.get(), entry.value(), type_info,
                                      nullptr, /*emit_positions=*/true));
  } else {
    XLS_ASSIGN_OR_RETURN(converted, ConvertModule(module.get(), type_info));
  }
  std::cout << converted;

  return absl::OkStatus();
}

}  // namespace
}  // namespace xls::dslx

int main(int argc, char* argv[]) {
  std::vector<absl::string_view> args =
      xls::InitXls(xls::dslx::kUsage, argc, argv);
  if (args.empty()) {
    XLS_LOG(QFATAL) << "Wrong number of command-line arguments; got "
                    << args.size() << ": `" << absl::StrJoin(args, " ")
                    << "`; want " << argv[0] << " <input-file>";
  }
  std::string dslx_path = absl::GetFlag(FLAGS_dslx_path);
  std::vector<std::string> dslx_paths = absl::StrSplit(dslx_path, ':');
  absl::optional<std::string> entry;
  if (!absl::GetFlag(FLAGS_entry).empty()) {
    entry = absl::GetFlag(FLAGS_entry);
  }
  bool printed_error = false;
  absl::Status status =
      xls::dslx::RealMain(args[0], entry, dslx_paths, &printed_error);
  if (printed_error) {
    return EXIT_FAILURE;
  }
  XLS_QCHECK_OK(status);
  return EXIT_SUCCESS;
}
