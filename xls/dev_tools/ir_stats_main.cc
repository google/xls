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

// Prints summary information about an IR file to the terminal.
// Output will be added as needs warrant, so feel free to make additions!

#include <iostream>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "xls/common/exit_status.h"
#include "xls/common/file/filesystem.h"
#include "xls/common/init_xls.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/ir_parser.h"

ABSL_FLAG(
    std::string, top, "",
    "The name of the top entity. Currently, only functions are supported. "
    "If set, restrict dumping to the given function. "
    "The name should not be mangled with the Package name.");

namespace xls {

static absl::Status RealMain(std::string_view ir_path,
                             std::optional<std::string> restrict_fn) {
  XLS_ASSIGN_OR_RETURN(std::string contents, GetFileContents(ir_path));
  XLS_ASSIGN_OR_RETURN(auto package, Parser::ParsePackage(contents));

  std::cout << "Package \"" << package->name() << "\"" << '\n';
  for (const auto& f : package->functions()) {
    if (restrict_fn && restrict_fn.value() != f->name()) {
      continue;
    }
    std::cout << "  Function: \"" << f->name() << "\"" << '\n';
    std::cout << "    Signature: " << f->GetType()->ToString() << '\n';
    std::cout << "    Nodes: " << f->node_count() << '\n';
    std::cout << '\n';
  }
  return absl::OkStatus();
}

}  // namespace xls

int main(int argc, char** argv) {
  std::vector<std::string_view> positional_args =
      xls::InitXls(argv[0], argc, argv);
  QCHECK_EQ(positional_args.size(), 1);

  std::optional<std::string> restrict_fn;
  if (!absl::GetFlag(FLAGS_top).empty()) {
    restrict_fn = absl::GetFlag(FLAGS_top);
  }
  return xls::ExitStatus(xls::RealMain(positional_args[0], restrict_fn));
}
