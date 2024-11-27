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

#include <filesystem>  // NOLINT
#include <iostream>
#include <string>
#include <string_view>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "xls/common/exit_status.h"
#include "xls/common/init_xls.h"
#include "xls/common/status/status_macros.h"
#include "xls/dslx/frontend/builtins_metadata.h"
#include "xls/dslx/frontend/pos.h"
#include "xls/dslx/frontend/scanner.h"
#include "xls/dslx/frontend/token.h"
#include "xls/dslx/virtualizable_file_system.h"

namespace xls::dslx {
namespace {

static constexpr std::string_view kUsage = R"(
Emits an ANSI-highlighted version of a given DSLX file.
)";

// ANSI color codes.
constexpr std::string_view kRed = "\e[1;31m";
constexpr std::string_view kGreen = "\e[1;32m";
constexpr std::string_view kYellow = "\e[1;33m";
constexpr std::string_view kBlue = "\e[1;34m";
constexpr std::string_view kCyan = "\e[1;36m";
constexpr std::string_view kReset = "\e[1;0m";

std::string AnsiRed(std::string_view s) {
  return absl::StrCat(kRed, s, kReset);
}
std::string AnsiGreen(std::string_view s) {
  return absl::StrCat(kGreen, s, kReset);
}
std::string AnsiYellow(std::string_view s) {
  return absl::StrCat(kYellow, s, kReset);
}
std::string AnsiBlue(std::string_view s) {
  return absl::StrCat(kBlue, s, kReset);
}
std::string AnsiCyan(std::string_view s) {
  return absl::StrCat(kCyan, s, kReset);
}

std::string HandleKeyword(std::string_view s) { return AnsiYellow(s); }
std::string HandleNumber(std::string_view s) { return AnsiRed(s); }
std::string HandleComment(std::string_view s) { return AnsiBlue(s); }
std::string HandleBuiltin(std::string_view s) { return AnsiCyan(s); }
std::string HandleType(std::string_view s) { return AnsiGreen(s); }
std::string HandleOther(std::string_view s) { return std::string(s); }

std::string ToHighlightStr(const Token& t) {
  switch (t.kind()) {
    case TokenKind::kKeyword:
      if (GetTypeKeywords().contains(t.GetKeyword())) {
        return HandleType(KeywordToString(t.GetKeyword()));
      }
      return HandleKeyword(t.ToString());
    case TokenKind::kNumber:
      return HandleNumber(t.ToString());
    case TokenKind::kComment:
      return HandleComment(t.ToString());
    case TokenKind::kIdentifier: {
      const std::string& value = t.GetStringValue();
      if (IsNameParametricBuiltin(value)) {
        return HandleBuiltin(value);
      }
      return HandleOther(value);
    }
    default:
      return HandleOther(t.ToString());
  }
}

absl::Status RealMain(const std::filesystem::path& path) {
  RealFilesystem vfs;
  FileTable file_table;
  Fileno fileno = file_table.GetOrCreate(path.c_str());
  XLS_ASSIGN_OR_RETURN(std::string contents, vfs.GetFileContents(path));
  Scanner s(file_table, fileno, contents,
            /*include_whitespace_and_comments=*/true);
  while (!s.AtEof()) {
    XLS_ASSIGN_OR_RETURN(Token t, s.Pop());
    std::cout << ToHighlightStr(t);
  }
  return absl::OkStatus();
}

}  // namespace
}  // namespace xls::dslx

int main(int argc, char** argv) {
  std::vector<std::string_view> args =
      xls::InitXls(xls::dslx::kUsage, argc, argv);
  if (args.empty()) {
    LOG(QFATAL) << "Wrong number of command-line arguments; got " << args.size()
                << ": `" << absl::StrJoin(args, " ") << "`; want " << argv[0]
                << " <input-file>";
  }

  return xls::ExitStatus(xls::dslx::RealMain(std::filesystem::path(args[0])));
}
