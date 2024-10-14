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
#include "absl/strings/str_cat.h"
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
#include "xls/dslx/fmt/ast_fmt.h"
#include "xls/dslx/fmt/comments.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/frontend/comment_data.h"
#include "xls/dslx/import_data.h"
#include "xls/dslx/parse_and_typecheck.h"
#include "xls/dslx/warning_kind.h"

// Note: we attempt to keep our command line interface similar to clang-format.
ABSL_FLAG(bool, i, false, "whether to modify the given path argument in-place");

ABSL_FLAG(std::string, dslx_path, "",
          "Additional paths to search for modules (colon delimited).");
ABSL_FLAG(std::string, mode, "autofmt",
          "whether to use reflowing auto-formatter");
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

absl::Status RealMain(std::string_view input_path,
                      absl::Span<const std::filesystem::path> dslx_paths,
                      bool in_place, bool opportunistic_postcondition,
                      const std::string& mode) {
  if (input_path == "-") {
    input_path = "/dev/stdin";
  }
  std::filesystem::path path = input_path;

  XLS_ASSIGN_OR_RETURN(std::string module_name, PathToName(path.c_str()));
  XLS_ASSIGN_OR_RETURN(std::string contents, GetFileContents(path));

  ImportData import_data =
      CreateImportData(xls::kDefaultDslxStdlibPath,
                       /*additional_search_paths=*/{}, kNoWarningsSet);
  std::string formatted;
  if (mode == "autofmt") {
    std::vector<CommentData> comments;
    XLS_ASSIGN_OR_RETURN(std::unique_ptr<Module> module,
                         ParseModule(contents, path.c_str(), module_name,
                                     import_data.file_table(), &comments));
    formatted = AutoFmt(*module, Comments::Create(comments));
  } else if (mode == "typecheck") {
    // Note: we don't flag any warnings in this binary as we're just formatting
    // the text.
    XLS_ASSIGN_OR_RETURN(
        TypecheckedModule tm,
        ParseAndTypecheck(contents, path.c_str(), module_name, &import_data));
    formatted = tm.module->ToString();
  } else if (mode == "parse") {
    XLS_ASSIGN_OR_RETURN(
        std::unique_ptr<Module> module,
        ParseModule(contents, path.c_str(), module_name,
                    import_data.file_table(), /*comments=*/nullptr));
    formatted = module->ToString();
  } else {
    return absl::InvalidArgumentError(absl::StrCat("Invalid mode: ", mode));
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
  std::string dslx_path = absl::GetFlag(FLAGS_dslx_path);
  std::vector<std::string> dslx_path_strs = absl::StrSplit(dslx_path, ':');

  std::vector<std::filesystem::path> dslx_paths;
  dslx_paths.reserve(dslx_path_strs.size());
  for (const auto& path : dslx_path_strs) {
    dslx_paths.push_back(std::filesystem::path(path));
  }

  absl::Status status =
      xls::dslx::RealMain(args[0], dslx_paths,
                          /*in_place=*/absl::GetFlag(FLAGS_i),
                          /*opportunistic_postcondition=*/
                          absl::GetFlag(FLAGS_opportunistic_postcondition),
                          /*mode=*/absl::GetFlag(FLAGS_mode));
  return xls::ExitStatus(status);
}
