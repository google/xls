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

#include "xls/common/file/path.h"

#include <filesystem>
#include <system_error>
#include <utility>

#include "absl/status/statusor.h"
#include "xls/common/status/error_code_to_status.h"

namespace xls {

absl::StatusOr<std::filesystem::path> RelativizePath(
    const std::filesystem::path& path, const std::filesystem::path& reference) {
  std::error_code ec;
  std::filesystem::path result = std::filesystem::relative(path, reference, ec);
  if (ec) {
    return ErrorCodeToStatus(ec);
  }
  return std::move(result);
}

}  // namespace xls
