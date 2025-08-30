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

#include "xls/ir/events.h"

#include "absl/status/status.h"

namespace xls {

absl::Status InterpreterEventsToStatus(const InterpreterEvents& events) {
  if (events.GetAssertMessages().empty()) {
    return absl::OkStatus();
  }

  // If an assertion has been raised, return the message from the first
  // assertion recorded, matching the behavior of short-circuit evaluation.
  return absl::AbortedError(events.GetAssertMessages().front());
}

}  // namespace xls
