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

#ifndef XLS_COMMON_LOGGING_CAPTURE_STREAM_H_
#define XLS_COMMON_LOGGING_CAPTURE_STREAM_H_

#include <unistd.h>

#include <functional>
#include <string>

#include "absl/status/statusor.h"

namespace xls {
namespace testing {

// Capture what is printed to the provided file descriptor `fd` (should be
// STDOUT_FILENO or STDERR_FILENO) while `fn` is run. Returns a string with the
// captured data.
absl::StatusOr<std::string> CaptureStream(int fd, std::function<void()> fn);

}  // namespace testing
}  // namespace xls

#endif  // XLS_COMMON_LOGGING_CAPTURE_STREAM_H_
