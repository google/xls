// Copyright 2024 The XLS Authors
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

#include "xls/tools/tool_timeout.h"

#include <memory>
#include <optional>

#include "absl/flags/flag.h"
#include "absl/time/time.h"
#include "xls/common/timeout_support.h"

ABSL_FLAG(std::optional<absl::Duration>, timeout, std::nullopt,
          "How long to allow the process to run. After this timeout the "
          "process is forcefully terminated.");

namespace xls {

std::unique_ptr<TimeoutCleaner> StartTimeoutTimer() {
  if (absl::GetFlag(FLAGS_timeout)) {
    return SetupTimeoutThread(*absl::GetFlag(FLAGS_timeout));
  }
  return nullptr;
}

}  // namespace xls
