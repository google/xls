// Copyright 2023 The XLS Authors
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

#include <algorithm>
#include <filesystem>  // NOLINT
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_split.h"
#include "absl/types/span.h"
#include "xls/common/exit_status.h"
#include "xls/common/file/filesystem.h"
#include "xls/common/init_xls.h"
#include "xls/common/status/status_macros.h"
#include "xls/dslx/command_line_utils.h"
#include "xls/dslx/create_import_data.h"
#include "xls/dslx/default_dslx_stdlib_path.h"
#include "xls/dslx/fmt/ast_fmt.h"
#include "xls/dslx/fmt/comments.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/frontend/comment_data.h"
#include "xls/dslx/import_data.h"
#include "xls/dslx/parse_and_typecheck.h"
#include "xls/dslx/virtualizable_file_system.h"
#include "xls/dslx/warning_kind.h"

// Note: we attempt to keep our command line interface similar to clang-format.
ABSL_FLAG(bool, i, false, "whether to modify the given path argument in-place");

ABSL_FLAG(bool, error_on_changes, false,
          "whether to error if the formatting changes the file contents");

ABSL_FLAG(std::string, dslx_path, "",
          "Additional paths to search for modules (colon delimited).");
ABSL_FLAG(std::string, mode, "autofmt",
          "whether to use reflowing auto-formatter; choices: autoformat|parse");
ABSL_FLAG(
    bool, opportunistic_postcondition, false,
    "whether to check the autoformatter postcondition for debug purposes (not "
    "a sound postcondition for all possible inputs, will make autofmt "
    "somewhat slower) -- note that this can flag false positive, e.g. in cases "
    "such as doubled parentheses that cannot be checked via regexp "
    "equivalence");

namespace xls::dslx {
namespace {

static constexpr std::string_view kUsage = R"(
Formats the DSLX source code present inside of a `.x` file.
)";

// Note: for "just typecheck the file" use the `typecheck_main` binary instead.
enum class Mode {
  // Uses the "simple AST autoformatter" that doesn't do reflow or anything
  // smart, useful mostly for
  // performance comparisons to the proper autoformatter.
  kParse,
  // Uses the proper autoformatter.
  kAutofmt,
};

absl::Status RunOnOneFile(std::string_view input_path, bool in_place,
                          bool error_on_changes,
                          bool opportunistic_postcondition, Mode mode) {
  std::filesystem::path path = input_path;

  ImportData import_data =
      CreateImportData(xls::kDefaultDslxStdlibPath,
                       /*additional_search_paths=*/{}, kNoWarningsSet,
                       std::make_unique<RealFilesystem>());

  XLS_ASSIGN_OR_RETURN(std::string module_name, PathToName(path.c_str()));
  XLS_ASSIGN_OR_RETURN(std::string contents,
                       import_data.vfs().GetFileContents(path));

  std::vector<CommentData> comments_vec;

  // Parse the module with comment collection enabled.
  absl::StatusOr<std::unique_ptr<Module>> module =
      ParseModule(contents, path.c_str(), module_name, import_data.file_table(),
                  &comments_vec);
  if (!module.ok()) {
    TryPrintError(module.status(), import_data.file_table(), import_data.vfs());
    return module.status();
  }

  // Format the module.
  std::string formatted;
  switch (mode) {
    case Mode::kAutofmt: {
      Comments comments = Comments::Create(comments_vec);
      XLS_ASSIGN_OR_RETURN(
          formatted,
          AutoFmt(import_data.vfs(), *module.value(), comments, contents));
      break;
    }
    case Mode::kParse: {
      formatted = module.value()->ToString();
      break;
    }
  }

  if (opportunistic_postcondition) {
    if (std::optional<AutoFmtPostconditionViolation> violation =
            ObeysAutoFmtOpportunisticPostcondition(contents, formatted);
        violation.has_value()) {
      std::cerr << "== Opportunistic Auto-formatter Postcondition Violation:\n";
      std::cerr << "original transformed: " << violation->original_transformed
                << "\n";
      std::cerr << "autofmt transformed:  " << violation->autofmt_transformed
                << "\n";
      return absl::InternalError(
          "Autoformatting failed its opportunistic postcondition test; "
          "autoformatted text may be buggy.");
    }
  }

  if (in_place) {
    XLS_RETURN_IF_ERROR(SetFileContents(path, formatted));
  } else {
    std::cout << formatted << std::flush;
  }

  if (error_on_changes && formatted != contents) {
    return absl::InternalError("Formatting changed the file contents.");
  }

  return absl::OkStatus();
}

absl::Status RealMain(absl::Span<const std::string_view> input_paths,
                      bool in_place, bool error_on_changes,
                      bool opportunistic_postcondition,
                      const std::string& mode_str) {
  // Note notable restrictions we place on the CLI, to avoid confusing results /
  // interactions:
  //
  // - If stdin is the input, it should be the only input.
  // - If we're *not* doing in-place formatting, there should be only one input
  // (otherwise things will be all jumbled together in stdout),
  // - If we error-on-changes or use the opportunistic postcondition, there
  //   should be only one input, to avoid semantic ambiguity when we're
  //   side-effecting the formatting of files and when we flag the error.

  bool has_stdin_arg =
      std::any_of(input_paths.begin(), input_paths.end(),
                  [](std::string_view path) { return path == "-"; });
  std::optional<std::vector<std::string_view>> stdin_input;
  if (has_stdin_arg) {
    if (input_paths.size() != 1) {
      return absl::InvalidArgumentError(
          "Cannot have stdin along with file arguments.");
    }
    if (in_place) {
      return absl::InvalidArgumentError(
          "Cannot format stdin with in-place formatting.");
    }
    stdin_input = std::vector<std::string_view>{"/dev/stdin"};
    input_paths = absl::MakeConstSpan(stdin_input.value());
  }

  if (error_on_changes && input_paths.size() > 1) {
    return absl::InvalidArgumentError(
        "Cannot have multiple input files when error-on-changes is enabled.");
  }

  if (opportunistic_postcondition && input_paths.size() > 1) {
    return absl::InvalidArgumentError(
        "Cannot have multiple input files when opportunistic-postcondition is "
        "enabled.");
  }

  if (!in_place && input_paths.size() > 1) {
    return absl::InvalidArgumentError(
        "Cannot have multiple input files when in-place formatting is "
        "disabled.");
  }

  Mode mode;
  if (mode_str == "autofmt") {
    mode = Mode::kAutofmt;
  } else if (mode_str == "parse") {
    mode = Mode::kParse;
  } else {
    return absl::InvalidArgumentError(
        absl::StrCat("Invalid mode: `", mode_str, "`"));
  }

  for (std::string_view input_path : input_paths) {
    XLS_RETURN_IF_ERROR(RunOnOneFile(input_path, in_place, error_on_changes,
                                     opportunistic_postcondition, mode));
  }

  return absl::OkStatus();
}

}  // namespace
}  // namespace xls::dslx

int main(int argc, char* argv[]) {
  std::vector<std::string_view> args =
      xls::InitXls(xls::dslx::kUsage, argc, argv);
  if (args.empty()) {
    LOG(QFATAL) << "No command-line arguments to format; want " << argv[0]
                << " <input-file>[, ...]";
  }
  std::string dslx_path = absl::GetFlag(FLAGS_dslx_path);
  std::vector<std::string> dslx_path_strs = absl::StrSplit(dslx_path, ':');

  absl::Status status = xls::dslx::RealMain(
      args,
      /*in_place=*/absl::GetFlag(FLAGS_i),
      /*error_on_changes=*/absl::GetFlag(FLAGS_error_on_changes),
      /*opportunistic_postcondition=*/
      absl::GetFlag(FLAGS_opportunistic_postcondition),
      /*mode_str=*/absl::GetFlag(FLAGS_mode));
  return xls::ExitStatus(status);
}
