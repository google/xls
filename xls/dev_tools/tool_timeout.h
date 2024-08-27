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

#ifndef XLS_DEV_TOOLS_TOOL_TIMEOUT_H_
#define XLS_DEV_TOOLS_TOOL_TIMEOUT_H_

#include <memory>

#include "xls/common/timeout_support.h"
namespace xls {

// Start the timeout watchdog based on an absl flag.
//
// The timeout will be canceled when the returned cleaner is destroyed.
[[nodiscard]] std::unique_ptr<TimeoutCleaner> StartTimeoutTimer();

}  // namespace xls

#endif  // XLS_DEV_TOOLS_TOOL_TIMEOUT_H_
