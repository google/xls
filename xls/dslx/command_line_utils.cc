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

#include <stdio.h>
#include <unistd.h>

#include <cstdio>
#include <functional>
#include <iostream>
#include <string>
#include <string_view>
#include <vector>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_split.h"
#include "xls/common/status/ret_check.h"
#include "xls/dslx/error_printer.h"
#include "xls/dslx/frontend/bindings.h"

namespace xls::dslx {

bool TryPrintError(
    const absl::Status& status,
    const std::function<absl::StatusOr<std::string>(std::string_view)>&
        get_file_contents) {
  if (status.ok()) {
    return false;
  }
  absl::StatusOr<PositionalErrorData> data_or = GetPositionalErrorData(status);
  if (!data_or.ok()) {
    LOG(ERROR) << "Could not extract a textual position from error message: "
               << status << ": " << data_or.status();
    return false;
  }
  auto& data = data_or.value();
  bool is_tty = isatty(fileno(stderr)) != 0;
  absl::Status print_status = PrintPositionalError(
      data.span, absl::StrFormat("%s: %s", data.error_type, data.message),
      std::cerr, get_file_contents,
      is_tty ? PositionalErrorColor::kErrorColor
             : PositionalErrorColor::kNoColor);
  if (!print_status.ok()) {
    LOG(ERROR) << "Could not print positional error: " << print_status;
  }
  return print_status.ok();
}

absl::StatusOr<std::string> PathToName(std::string_view path) {
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

}  // namespace xls::dslx
