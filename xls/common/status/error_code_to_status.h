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

#ifndef XLS_COMMON_STATUS_ERROR_CODE_TO_STATUS_H_
#define XLS_COMMON_STATUS_ERROR_CODE_TO_STATUS_H_

#include <system_error>  // NOLINT(build/c++11)

#include "absl/status/status.h"
#include "absl/status/status_builder.h"
#include "absl/types/source_location.h"

namespace xls {

absl::StatusCode ErrorCodeToStatusCode(const std::error_code& ec);

absl::StatusBuilder ErrorCodeToStatus(
    const std::error_code& ec,
    absl::SourceLocation loc = absl::SourceLocation::current());

// Converts an `errno` value into an absl::Status.
absl::StatusBuilder ErrnoToStatus(
    int errno_value,
    absl::SourceLocation loc = absl::SourceLocation::current());

}  // namespace xls

#endif  // XLS_COMMON_STATUS_ERROR_CODE_TO_STATUS_H_
