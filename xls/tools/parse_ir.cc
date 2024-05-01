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

// Utility which parses files given specified as command-line arguments as XLS
// IR text. If no argument given reads from stdin. Returns non-zero value on
// failure and emits failing absl::Status message to stderr.

#include <iostream>
#include <iterator>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/status/status.h"
#include "absl/types/span.h"
#include "xls/common/exit_status.h"
#include "xls/common/file/filesystem.h"
#include "xls/common/init_xls.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/ir_parser.h"
#include "xls/ir/package.h"

ABSL_FLAG(bool, dump_ir, false,
          "If true, then dump the IR to stdout after parsing.");

namespace xls {
namespace tools {

static absl::Status RealMain(absl::Span<const std::string_view> args) {
  if (args.empty()) {
    // If no arguments are given, read from stdin.
    return Parser::ParsePackage(
               std::string{std::istreambuf_iterator<char>(std::cin),
                           std::istreambuf_iterator<char>()})
        .status();
  }
  for (std::string_view arg : args) {
    XLS_ASSIGN_OR_RETURN(std::string contents, GetFileContents(arg));
    XLS_ASSIGN_OR_RETURN(std::unique_ptr<Package> p,
                         Parser::ParsePackage(contents));
    if (absl::GetFlag(FLAGS_dump_ir)) {
      std::cout << p->DumpIr();
    }
  }
  return absl::OkStatus();
}

}  // namespace tools
}  // namespace xls

int main(int argc, char** argv) {
  std::vector<std::string_view> positional_arguments =
      xls::InitXls(argv[0], argc, argv);
  return xls::ExitStatus(xls::tools::RealMain(positional_arguments));
}
