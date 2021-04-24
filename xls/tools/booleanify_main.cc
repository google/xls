// Copyright 2020 The XLS Authors
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

// Utility binary to convert input XLS IR to a "fundamental"
// representation, i.e., consisting of only AND/OR/NOT ops.

#include <filesystem>
#include <string>

#include "absl/flags/flag.h"
#include "absl/status/status.h"
#include "absl/types/optional.h"
#include "xls/common/file/filesystem.h"
#include "xls/common/init_xls.h"
#include "xls/common/logging/logging.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/ir_parser.h"
#include "xls/tools/booleanifier.h"

ABSL_FLAG(
    std::string, function, "",
    "Name of function to convert to SMTLIB. If unspecified, a 'best guess' "
    "will be made to try to find the package's entry function. "
    "If that fails, an error will be returned.");
ABSL_FLAG(std::string, ir_path, "", "Path to the XLS IR to process.");
ABSL_FLAG(std::string, output_function_name, "",
          "Name of the booleanified function. If empty, then the name is the "
          "input function name with '_boolean' appended to it.");

namespace xls {

absl::Status RealMain(const std::filesystem::path& ir_path,
                      absl::optional<std::string> function_name) {
  XLS_ASSIGN_OR_RETURN(std::string ir_text, GetFileContents(ir_path));
  XLS_ASSIGN_OR_RETURN(auto package, Parser::ParsePackage(ir_text));
  Function* function;
  if (!function_name) {
    XLS_ASSIGN_OR_RETURN(function, package->EntryFunction());
  } else {
    XLS_ASSIGN_OR_RETURN(function, package->GetFunction(function_name.value()));
  }

  XLS_ASSIGN_OR_RETURN(
      function, Booleanifier::Booleanify(
                    function, absl::GetFlag(FLAGS_output_function_name)));
  std::cout << "package " << package->name() << "\n\n";
  std::cout << function->DumpIr() << "\n";
  return absl::OkStatus();
}

}  // namespace xls

int main(int argc, char* argv[]) {
  xls::InitXls(argv[0], argc, argv);

  absl::optional<std::string> function_name;
  if (!absl::GetFlag(FLAGS_function).empty()) {
    function_name = absl::GetFlag(FLAGS_function);
  }
  XLS_QCHECK_OK(xls::RealMain(absl::GetFlag(FLAGS_ir_path), function_name));
  return 0;
}
