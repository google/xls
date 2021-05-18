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

#include "xls/passes/passes.h"

#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "xls/common/logging/logging.h"
#include "xls/common/status/status_macros.h"

namespace xls {

absl::StatusOr<bool> FunctionBasePass::RunOnFunctionBase(
    FunctionBase* f, const PassOptions& options, PassResults* results) const {
  XLS_VLOG(2) << absl::StreamFormat("Running %s on function_base %s [pass #%d]",
                                    long_name(), f->name(),
                                    results->invocations.size());
  XLS_VLOG(3) << "Before:";
  XLS_VLOG_LINES(3, f->DumpIr());

  XLS_ASSIGN_OR_RETURN(bool changed,
                       RunOnFunctionBaseInternal(f, options, results));

  XLS_VLOG(3) << absl::StreamFormat("After [changed = %d]:", changed);
  XLS_VLOG_LINES(3, f->DumpIr());
  return changed;
}

absl::StatusOr<bool> FunctionBasePass::RunInternal(Package* p,
                                                   const PassOptions& options,
                                                   PassResults* results) const {
  bool changed = false;
  for (FunctionBase* f : p->GetFunctionBases()) {
    XLS_ASSIGN_OR_RETURN(bool function_changed,
                         RunOnFunctionBaseInternal(f, options, results));
    changed |= function_changed;
  }
  return changed;
}

absl::StatusOr<bool> FunctionBasePass::TransformNodesToFixedPoint(
    FunctionBase* f,
    std::function<absl::StatusOr<bool>(Node*)> simplify_f) const {
  // Store nodes by id to avoid running afoul of Node* pointer values being
  // reused.
  absl::flat_hash_set<int64_t> simplified_node_ids;
  bool changed = false;
  bool changed_this_time = false;
  do {
    changed_this_time = false;
    auto node_it = f->nodes().begin();
    while (node_it != f->nodes().end()) {
      // Save the next iterator because node_it may be invalidated by the call
      // to simplify_f if simpplify_f ends up deleting 'node'.
      auto next_it = std::next(node_it);
      Node* node = *node_it;
      // If the node was previously simplified and is now dead, avoid running
      // simplification on it again to avoid inf-looping while simplifying the
      // same node over and over again.
      if (!node->IsDead() || !simplified_node_ids.contains(node->id())) {
        // Grab the node ID before simplifying because the node might be
        // removed when simplifying.
        int64_t node_id = node->id();
        XLS_ASSIGN_OR_RETURN(bool node_changed, simplify_f(node));
        if (node_changed) {
          simplified_node_ids.insert(node_id);
          changed_this_time = true;
          changed = true;
        }
      }
      node_it = next_it;
    }
  } while (changed_this_time);

  return changed;
}

absl::StatusOr<bool> ProcPass::RunOnProc(Proc* proc, const PassOptions& options,
                                         PassResults* results) const {
  XLS_VLOG(2) << absl::StreamFormat("Running %s on proc %s [pass #%d]",
                                    long_name(), proc->name(),
                                    results->invocations.size());
  XLS_VLOG(3) << "Before:";
  XLS_VLOG_LINES(3, proc->DumpIr());

  XLS_ASSIGN_OR_RETURN(bool changed, RunOnProcInternal(proc, options, results));

  XLS_VLOG(3) << absl::StreamFormat("After [changed = %d]:", changed);
  XLS_VLOG_LINES(3, proc->DumpIr());
  return changed;
}

absl::StatusOr<bool> ProcPass::RunInternal(Package* p,
                                           const PassOptions& options,
                                           PassResults* results) const {
  bool changed = false;
  for (const auto& proc : p->procs()) {
    XLS_ASSIGN_OR_RETURN(bool proc_changed,
                         RunOnProcInternal(proc.get(), options, results));
    changed |= proc_changed;
  }
  return changed;
}

}  // namespace xls
