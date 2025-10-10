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

#include "xls/dslx/command_line_utils.h"

#include <stdio.h>  // NOLINT(modernize-deprecated-headers)
#include <unistd.h>

#include <iostream>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/ascii.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_split.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/dslx/error_printer.h"
#include "xls/dslx/frontend/bindings.h"
#include "xls/dslx/frontend/pos.h"
#include "xls/dslx/virtualizable_file_system.h"

namespace xls::dslx {

bool TryPrintError(const absl::Status& status, FileTable& file_table,
                   VirtualizableFilesystem& vfs) {
  if (status.ok()) {
    return false;
  }
  absl::StatusOr<PositionalErrorData> data =
      GetPositionalErrorData(status, std::nullopt, file_table);
  if (!data.ok()) {
    LOG(ERROR) << "Could not extract a textual position from error message: "
               << status << ": " << data.status();
    return false;
  }
  bool is_tty = isatty(fileno(stderr)) != 0;
  absl::Status print_status = PrintPositionalError(
      data->spans, absl::StrFormat("%s: %s", data->error_type, data->message),
      std::cerr,
      is_tty ? PositionalErrorColor::kErrorColor
             : PositionalErrorColor::kNoColor,
      file_table, vfs);
  if (!print_status.ok()) {
    LOG(ERROR) << "Could not print positional error: " << print_status;
  }
  return print_status.ok();
}

namespace {
// Replace any char which is not in the regex [a-zA-Z0-9_] to with
// __H0x<hex>__ where hex is the hex representation of the byte. Encoding is
// ignored.
std::string CanonicalizeName(std::string_view sv) {
  std::stringbuf buf;
  std::string result;
  for (char c : sv) {
    if (absl::ascii_isalnum(c) || c == '_') {
      result.push_back(c);
    } else {
      result.append(absl::StrFormat("__H0x%02X__", static_cast<int>(c)));
    }
  }
  return result;
}
}  // namespace

absl::StatusOr<std::string> RawNameFromPath(std::string_view path) {
  std::vector<std::string_view> pieces = absl::StrSplit(path, '/');
  if (pieces.empty()) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Could not determine module name from path: %s", path));
  }
  std::string_view last = pieces.back();
  std::vector<std::string_view> dot_pieces = absl::StrSplit(last, '.');
  XLS_RET_CHECK(!dot_pieces.empty());
  return std::string(dot_pieces[0]);
}
absl::StatusOr<std::string> PathToName(std::string_view path) {
  XLS_ASSIGN_OR_RETURN(std::string raw_name, RawNameFromPath(path));
  return CanonicalizeName(raw_name);
}

bool NameNeedsCanonicalization(std::string_view path) {
  XLS_ASSIGN_OR_RETURN(std::string raw_name, RawNameFromPath(path), true);
  return CanonicalizeName(raw_name) != raw_name;
}
}  // namespace xls::dslx
