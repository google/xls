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

// Utility binary to convert input XLS IR to SMTLIB2.
// Adds the handy option of converting the XLS IR into a "fundamental"
// representation, i.e., consisting of only AND/OR/NOT ops.

// TODO(rspringer): No array support yet. Should be pretty trivial to add.

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
#include "xls/solvers/z3_ir_translator.h"
#include "external/z3/src/api/z3_api.h"

ABSL_FLAG(
    std::string, top, "",
    "The name of the top entity. Currently, only functions are supported. "
    "Function to convert to SMTLIB. If unspecified, a 'best guess' "
    "will be made to try to find the package's entry function. "
    "If that fails, an error will be returned.");
ABSL_FLAG(std::string, ir_path, "", "Path to the XLS IR to process.");

namespace xls {

static absl::Status RealMain(const std::filesystem::path& ir_path,
                             std::optional<std::string> top) {
  XLS_ASSIGN_OR_RETURN(std::string ir_text, GetFileContents(ir_path));
  XLS_ASSIGN_OR_RETURN(auto package, Parser::ParsePackage(ir_text));
  Function* function;
  if (!top) {
    XLS_ASSIGN_OR_RETURN(function, package->GetTopAsFunction());
  } else {
    XLS_ASSIGN_OR_RETURN(function, package->GetFunction(top.value()));
  }

  XLS_ASSIGN_OR_RETURN(auto translator,
                       solvers::z3::IrTranslator::CreateAndTranslate(function));
  Z3_set_ast_print_mode(translator->ctx(), Z3_PRINT_SMTLIB2_COMPLIANT);
  std::cout << Z3_ast_to_string(translator->ctx(), translator->GetReturnNode())
            << '\n';
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
