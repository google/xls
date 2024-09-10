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

#include <cstdlib>
#include <filesystem>  // NOLINT
#include <fstream>
#include <iostream>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_join.h"
#include "xls/common/exit_status.h"
#include "xls/common/file/filesystem.h"
#include "xls/common/init_xls.h"
#include "xls/common/status/status_macros.h"
#include "xls/dslx/frontend/pos.h"
#include "xls/dslx/frontend/scanner.h"
#include "xls/dslx/frontend/token.h"

ABSL_FLAG(bool, original_on_error, false,
          "emit original source if a scan error is encountered");
ABSL_FLAG(std::optional<std::string>, output_path, std::nullopt,
          "output path to use in lieu of stdout");

namespace xls::dslx {
namespace {

static constexpr std::string_view kUsage = R"(
Emits the original DSLX source text with comment tokens stripped out.
)";

absl::StatusOr<std::string> RealMain(const std::filesystem::path& path,
                                     std::optional<std::string>* contents_out) {
  XLS_ASSIGN_OR_RETURN(*contents_out, GetFileContents(path));
  FileTable file_table;
  Fileno fileno = file_table.GetOrCreate(std::string{path});
  Scanner s(file_table, fileno, contents_out->value(),
            /*include_whitespace_and_comments=*/true);

  // We output to a string stream as an intermediary in case we encounter an
  // error in the process of emitting tokens.
  std::stringstream ss;
  while (!s.AtEof()) {
    XLS_ASSIGN_OR_RETURN(Token t, s.Pop());
    if (t.kind() != TokenKind::kComment) {
      ss << t.ToString();
    }
  }
  return ss.str();
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

  std::optional<std::ofstream> fs;
  std::ostream* os = nullptr;
  if (std::optional<std::string> output_path = absl::GetFlag(FLAGS_output_path);
      output_path.has_value()) {
    fs.emplace(output_path.value());
    os = &fs.value();
  } else {
    os = &std::cout;
  }

  std::optional<std::string> contents;
  absl::StatusOr<std::string> result =
      xls::dslx::RealMain(std::filesystem::path(args[0]), &contents);
  if (result.ok()) {
    *os << result.value();
  } else if (absl::GetFlag(FLAGS_original_on_error) && contents.has_value()) {
    *os << contents.value();
    return EXIT_SUCCESS;
  }

  return xls::ExitStatus(result.status());
}
