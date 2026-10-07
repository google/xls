// Copyright 2026 The XLS Authors
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

#include "xls/passes/cached_state_element_query_engine.h"

#include <iterator>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/status/status.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/node.h"
#include "xls/ir/nodes.h"
#include "xls/ir/proc.h"
#include "xls/passes/manual_given_query_engine.h"
#include "xls/passes/proc_state_range_query_engine.h"

namespace xls {

absl::Status CachedStateElementQueryEngine::Compute() {
  if (!function_->IsProc()) {
    // Not actually able to do anything.
    XLS_RETURN_IF_ERROR(ManualGivenQueryEngine::ClearGivens());
    return absl::OkStatus();
  }
  ProcStateRangeQueryEngine proc_state_info;
  XLS_RETURN_IF_ERROR(proc_state_info.Populate(function_).status())
      << "Unable to populate the proc-state element information for "
      << function_->name();
  return ComputeWithStateInfo(proc_state_info);
}

absl::Status CachedStateElementQueryEngine::ComputeWithStateInfo(
    ProcStateRangeQueryEngine& proc_state_info) {
  XLS_RET_CHECK(function_->IsProc())
      << "CachedStateElementQueryEngine only supports procs.";
  std::vector<Node*> state_reads;
  state_reads.reserve(function_->AsProcOrDie()->GetStateElementCount());
  absl::c_copy_if(function_->nodes(), std::back_inserter(state_reads),
                  [](Node* node) { return node->Is<StateRead>(); });
  return SetGivens(state_reads, proc_state_info);
}

}  // namespace xls
