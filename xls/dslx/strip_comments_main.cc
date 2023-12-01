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
#include <string>
#include <string_view>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/str_join.h"
#include "xls/common/exit_status.h"
#include "xls/common/file/filesystem.h"
#include "xls/common/init_xls.h"
#include "xls/common/logging/logging.h"
#include "xls/common/status/status_macros.h"
#include "xls/dslx/frontend/scanner.h"
#include "xls/dslx/frontend/token.h"

namespace xls::dslx {
namespace {

const char* kUsage = R"(
Emits the original DSLX source text with comment tokens stripped out.
)";

absl::Status RealMain(const std::filesystem::path& path) {
  XLS_ASSIGN_OR_RETURN(std::string contents, GetFileContents(path));
  Scanner s(path, contents, /*include_whitespace_and_comments=*/true);
  while (!s.AtEof()) {
    XLS_ASSIGN_OR_RETURN(Token t, s.Pop());
    if (t.kind() != TokenKind::kComment) {
      std::cout << t.ToString();
    }
  }
  return absl::OkStatus();
}

}  // namespace
}  // namespace xls::dslx

int main(int argc, char** argv) {
  std::vector<std::string_view> args =
      xls::InitXls(xls::dslx::kUsage, argc, argv);
  if (args.empty()) {
    XLS_LOG(QFATAL) << "Wrong number of command-line arguments; got "
                    << args.size() << ": `" << absl::StrJoin(args, " ")
                    << "`; want " << argv[0] << " <input-file>";
  }

  return xls::ExitStatus(xls::dslx::RealMain(std::filesystem::path(args[0])));
}
