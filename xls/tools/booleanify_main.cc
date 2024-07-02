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

#include <filesystem>  // NOLINT
#include <iostream>
#include <optional>
#include <string>

#include "absl/flags/flag.h"
#include "absl/status/status.h"
#include "xls/common/exit_status.h"
#include "xls/common/file/filesystem.h"
#include "xls/common/init_xls.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/function.h"
#include "xls/ir/ir_parser.h"
#include "xls/tools/booleanifier.h"

ABSL_FLAG(
    std::string, top, "",
    "The name of the top entity. Currently, only functions are supported. "
    "Name of function to convert to SMTLIB. If flag is not given, the top "
    "entity specified in the IR will be used or if no top is specified in the "
    "IR an error is returned.");

ABSL_FLAG(std::string, ir_path, "", "Path to the XLS IR to process.");
ABSL_FLAG(std::string, output_function_name, "",
          "Name of the booleanified function. If empty, then the name is the "
          "same as the provided/inferred input name.");

namespace xls {

static absl::Status RealMain(const std::filesystem::path& ir_path,
                             std::optional<std::string> function_name) {
  XLS_ASSIGN_OR_RETURN(std::string ir_text, GetFileContents(ir_path));
  XLS_ASSIGN_OR_RETURN(auto package, Parser::ParsePackage(ir_text));
  Function* function;
  if (!function_name) {
    XLS_ASSIGN_OR_RETURN(function, package->GetTopAsFunction());
  } else {
    XLS_ASSIGN_OR_RETURN(function, package->GetFunction(function_name.value()));
  }

  std::string boolean_function_name = absl::GetFlag(FLAGS_output_function_name);
  if (boolean_function_name.empty()) {
    boolean_function_name = function->name();
  }

  XLS_ASSIGN_OR_RETURN(
      function, Booleanifier::Booleanify(function, boolean_function_name));
  std::cout << "package " << package->name() << "\n\n";
  std::cout << "top " << function->DumpIr() << "\n";
  return absl::OkStatus();
}

}  // namespace xls

int main(int argc, char* argv[]) {
  xls::InitXls(argv[0], argc, argv);

  std::optional<std::string> top;
  if (!absl::GetFlag(FLAGS_top).empty()) {
    top = absl::GetFlag(FLAGS_top);
  }
  return xls::ExitStatus(xls::RealMain(absl::GetFlag(FLAGS_ir_path), top));
}
