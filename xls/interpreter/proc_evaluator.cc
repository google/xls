// Copyright 2022 The XLS Authors
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

#include "xls/interpreter/proc_evaluator.h"

#include "xls/common/logging/logging.h"

namespace xls {

bool TickResult::operator==(const TickResult& other) const {
  return execution_state == other.execution_state && channel == other.channel &&
         progress_made == other.progress_made;
}

bool TickResult::operator!=(const TickResult& other) const {
  return !(*this == other);
}

std::string TickResult::ToString() const {
  return absl::StrFormat(
      "{ state=%s, channel=%s, progress_made=%s }",
      ::xls::ToString(execution_state),
      channel.has_value() ? channel.value()->ToString() : "(none)",
      progress_made ? "true" : "false");
}

std::string ToString(TickExecutionState state) {
  switch (state) {
    case TickExecutionState::kCompleted:
      return "kCompleted";
    case TickExecutionState::kBlockedOnReceive:
      return "kBlockedOnReceive";
    case TickExecutionState::kSentOnChannel:
      return "kSentOnChannel";
  }
  XLS_CHECK(false) << "Internal Error";
}

std::ostream& operator<<(std::ostream& os, TickExecutionState state) {
  os << ToString(state);
  return os;
}

std::ostream& operator<<(std::ostream& os, const TickResult& result) {
  os << result.ToString();
  return os;
}

}  // namespace xls
