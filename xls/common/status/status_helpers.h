// Copyright 2025 The XLS Authors
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

#ifndef XLS_COMMON_STATUS_STATUS_HELPERS_H_
#define XLS_COMMON_STATUS_STATUS_HELPERS_H_

#include <string_view>

#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "xls/common/source_location.h"

namespace xabsl {

template <typename T>
T ValueOrDie(absl::StatusOr<T> v,
             std::string_view message_prefix = "Unexpected error",
             xabsl::SourceLocation loc = xabsl::SourceLocation::current()) {
  if (!v.ok()) {
    LOG(FATAL).AtLocation(loc.file_name(), loc.line())
        << message_prefix << ": " << std::move(v).status();
  }
  return *std::move(v);
}

}  // namespace xabsl

#endif  // XLS_COMMON_STATUS_STATUS_HELPERS_H_
