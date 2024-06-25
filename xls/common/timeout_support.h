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

#ifndef XLS_COMMON_TIMEOUT_SUPPORT_H_
#define XLS_COMMON_TIMEOUT_SUPPORT_H_

#include <memory>
#include "absl/time/time.h"

namespace xls {

struct TimeoutCleaner {
 public:
  virtual ~TimeoutCleaner() = default;

 protected:
  explicit TimeoutCleaner() = default;
};

// Set a watchdog that will trigger after the given timeout and forcefully exit
// the process.
//
// The timeout will be canceled when the returned cleaner is destroyed.
std::unique_ptr<TimeoutCleaner> SetupTimeoutThread(absl::Duration duration);

}  // namespace xls

#endif  // XLS_COMMON_TIMEOUT_SUPPORT_H_
