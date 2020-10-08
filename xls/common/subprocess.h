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

#ifndef XLS_COMMON_SUBPROCESS_H_
#define XLS_COMMON_SUBPROCESS_H_

#include <filesystem>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"

namespace xls {

// Invokes a subprocess with the given argv. If 'cwd' is not empty the
// subprocess will be invoked in the given directory. Returns the
// stdout/stderr as a string pair.
absl::StatusOr<std::pair<std::string, std::string>> InvokeSubprocess(
    absl::Span<const std::string> argv, const std::filesystem::path& cwd = "");
}

#endif  // XLS_COMMON_SUBPROCESS_H_
