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

#include <cstdint>
#include <ostream>
#include <string>
#include <vector>

#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "xls/ir/node.h"
#include "xls/ir/nodes.h"
#include "xls/ir/proc.h"
#include "xls/ir/type.h"
#include "xls/ir/value.h"
#include "xls/ir/value_utils.h"

namespace xls {

absl::Status ProcContinuation::CheckConformsToStateType(
    const std::vector<Value>& v) const {
  // Check that v is the same size;
  if (v.size() != proc()->GetStateElementCount()) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "ProcContinuation::CheckConformsToStateType %s value is size %d, "
        "expected %d",
        proc_instance()->ToString(), v.size(), proc()->GetStateElementCount()));
  }

  // Check that v's elements are compatible with the type as what's in state_.
  for (int64_t i = 0; i < v.size(); ++i) {
    Type* type = proc()->GetStateElementType(i);
    if (!ValueConformsToType(v[i], type)) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "ProcContinuation::CheckConformsToStateType %s value[%d] is %s, "
          "expected type %s",
          proc_instance()->ToString(), i, v[i].ToString(), type->ToString()));
    }
  }

  return absl::OkStatus();
}

bool TickResult::operator==(const TickResult& other) const {
  return execution_state == other.execution_state &&
         channel_instance == other.channel_instance &&
         progress_made == other.progress_made;
}

bool TickResult::operator!=(const TickResult& other) const {
  return !(*this == other);
}

std::string TickResult::ToString() const {
  return absl::StrFormat("{ state=%s, channel_instance=%s, progress_made=%s }",
                         ::xls::ToString(execution_state),
                         channel_instance.has_value()
                             ? channel_instance.value()->ToString()
                             : "(none)",
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
  CHECK(false) << "Internal Error";
}

std::ostream& operator<<(std::ostream& os, TickExecutionState state) {
  os << ToString(state);
  return os;
}

std::ostream& operator<<(std::ostream& os, const TickResult& result) {
  os << result.ToString();
  return os;
}

ProcEvaluator::ProcEvaluator(Proc* proc, const EvaluatorOptions& options)
    : proc_(proc), has_io_operations_(false), options_(options) {
  for (Node* node : proc->nodes()) {
    if (node->Is<Send>() || node->Is<Receive>()) {
      has_io_operations_ = true;
      break;
    }
  }
}

}  // namespace xls
