// Copyright 2022 The XLS Authors
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

// Tool to evaluate DSLX + args and report the result.
#include <filesystem>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_split.h"
#include "xls/common/exit_status.h"
#include "xls/common/file/filesystem.h"
#include "xls/common/init_xls.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/dslx/bytecode/bytecode.h"
#include "xls/dslx/bytecode/bytecode_emitter.h"
#include "xls/dslx/bytecode/bytecode_interpreter.h"
#include "xls/dslx/create_import_data.h"
#include "xls/dslx/default_dslx_stdlib_path.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/import_data.h"
#include "xls/dslx/interp_value.h"
#include "xls/dslx/interp_value_utils.h"
#include "xls/dslx/parse_and_typecheck.h"
#include "xls/dslx/virtualizable_file_system.h"
#include "xls/dslx/warning_kind.h"

static constexpr std::string_view kUsage = R"(
Evaluates a DSLX file with user-specified or random inputs using the DSLX
interpreter. Example invocations:

Evaluate DSLX with a single input:

   eval_dslx_main DSLX_FILE --input='bits[32]:42; (bits[7]:0, bits[20]:4)'
)";

ABSL_FLAG(std::string, dslx_paths, "",
          "Comma-separated list of paths to search for DSLX imports.");
ABSL_FLAG(std::string, entry, "", "Entry function to test. Must be specified.");
ABSL_FLAG(std::string, input, "",
          "The input to the function as a semicolon-separated list of typed "
          "values. For example: \"bits[32]:42; (bits[7]:0, bits[20]:4)\"");

namespace xls {

static absl::Status RealMain(
    const std::filesystem::path& dslx_path,
    const std::vector<std::filesystem::path>& additional_search_paths,
    std::string_view entry_fn_name, std::string_view args_text) {
  dslx::ImportData import_data(dslx::CreateImportData(
      kDefaultDslxStdlibPath, additional_search_paths,
      dslx::kDefaultWarningsSet, std::make_unique<dslx::RealFilesystem>()));

  XLS_ASSIGN_OR_RETURN(std::vector<dslx::InterpValue> args,
                       dslx::ParseArgs(args_text));

  XLS_ASSIGN_OR_RETURN(std::string dslx_text, GetFileContents(dslx_path));
  XLS_ASSIGN_OR_RETURN(
      dslx::TypecheckedModule tm,
      dslx::ParseAndTypecheck(dslx_text, std::string(dslx_path), "the_module",
                              &import_data));
  XLS_ASSIGN_OR_RETURN(
      dslx::Function * f,
      tm.module->GetMemberOrError<dslx::Function>(entry_fn_name));
  XLS_RET_CHECK(f != nullptr);
  if (f->params().size() != args.size()) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Incorrect number of arguments: wanted %d, got %d.",
                        f->params().size(), args.size()));
  }

  XLS_ASSIGN_OR_RETURN(std::unique_ptr<dslx::BytecodeFunction> bf,
                       dslx::BytecodeEmitter::Emit(&import_data, tm.type_info,
                                                   *f, std::nullopt));
  XLS_ASSIGN_OR_RETURN(
      dslx::InterpValue result,
      dslx::BytecodeInterpreter::Interpret(&import_data, bf.get(), args));
  std::cout << "Result: " << result.ToString() << "\n";

  return absl::OkStatus();
}

}  // namespace xls

int main(int argc, char* argv[]) {
  std::vector<std::string_view> positional_arguments =
      xls::InitXls(kUsage, argc, argv);
  QCHECK_EQ(positional_arguments.size(), 1) << absl::StreamFormat(
      "Expected invocation: %s <DSLX path> --input", argv[0]);

  std::string inputs = absl::GetFlag(FLAGS_input);
  QCHECK(!inputs.empty());

  std::string entry_fn_name = absl::GetFlag(FLAGS_entry);
  QCHECK(!entry_fn_name.empty()) << "--entry must be specified.";

  std::vector<std::string> pieces =
      absl::StrSplit(absl::GetFlag(FLAGS_dslx_paths), ',');
  std::vector<std::filesystem::path> additional_search_paths;
  additional_search_paths.reserve(pieces.size());
  for (const auto& piece : pieces) {
    additional_search_paths.push_back(piece);
  }

  return xls::ExitStatus(xls::RealMain(
      positional_arguments[0], additional_search_paths, entry_fn_name, inputs));
}
