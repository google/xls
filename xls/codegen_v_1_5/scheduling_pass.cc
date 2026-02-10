// Copyright 2025 The XLS Authors
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

#include "xls/codegen_v_1_5/scheduling_pass.h"

#include <memory>
#include <optional>
#include <string>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xls/codegen_v_1_5/block_conversion_pass.h"
#include "xls/codegen_v_1_5/block_conversion_utils.h"
#include "xls/common/casts.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/function.h"
#include "xls/ir/name_uniquer.h"
#include "xls/ir/node.h"
#include "xls/ir/nodes.h"
#include "xls/ir/op.h"
#include "xls/ir/package.h"
#include "xls/ir/proc.h"
#include "xls/ir/proc_instantiation.h"
#include "xls/ir/source_location.h"
#include "xls/passes/pass_base.h"
#include "xls/scheduling/pipeline_schedule.h"
#include "xls/scheduling/pipeline_schedule.pb.h"

namespace xls::codegen {
namespace {

template <typename ScheduledFunctionBase>
absl::Status ScheduleNodes(const BlockConversionPassOptions& options,
                           ScheduledFunctionBase* fb) {
  if (options.codegen_options.generate_combinational()) {
    fb->AddEmptyStages(1);
    for (Node* node : fb->nodes()) {
      XLS_ASSIGN_OR_RETURN(bool added, fb->AddNodeToStage(0, node));
      XLS_RET_CHECK(added) << "Failed to add node to stage: "
                           << node->ToString();
    }
    return absl::OkStatus();
  }

  XLS_ASSIGN_OR_RETURN(
      PipelineSchedule schedule,
      PipelineSchedule::FromProto(fb, options.package_schedule));
  fb->AddEmptyStages(schedule.length());
  for (Node* node : fb->nodes()) {
    if (schedule.IsScheduled(node)) {
      XLS_ASSIGN_OR_RETURN(bool added,
                           fb->AddNodeToStage(schedule.cycle(node), node));
      XLS_RET_CHECK(added) << "Failed to add node to stage: "
                           << node->ToString();
    }
  }
  return absl::OkStatus();
}

}  // namespace

absl::StatusOr<bool> SchedulingPass::RunInternal(
    Package* package, const BlockConversionPassOptions& options,
    PassResults* results) const {
  NameUniquer uniquer("__");
  std::optional<FunctionBase*> top = package->GetTop();
  std::string top_name;
  if (top.has_value()) {
    top_name = uniquer.GetSanitizedUniqueName(
        options.codegen_options.module_name().value_or((*top)->name()));
  }

  // Original proc to `scheduled_proc`.
  absl::flat_hash_map<Proc*, Proc*> proc_map;
  bool changed = false;

  for (FunctionBase* old_fb : package->GetFunctionBases()) {
    FunctionBase* new_fb = nullptr;
    if (old_fb->IsFunction()) {
      if (old_fb->ForeignFunctionData().has_value()) {
        // This isn't a real function; it's a placeholder for a foreign
        // function that will be replaced late in codegen.
        continue;
      }
      ScheduledFunction* new_fn =
          absl::down_cast<ScheduledFunction*>(package->AddFunction(
              std::make_unique<ScheduledFunction>(old_fb->name(), package)));
      new_fb = new_fn;
      new_fn->MoveFrom(*absl::down_cast<Function*>(old_fb));
      XLS_RETURN_IF_ERROR(
          ScheduleNodes(options, absl::down_cast<ScheduledFunction*>(new_fb)));

      Node* return_value = new_fn->return_value();
      if (new_fn->IsStaged(return_value) &&
          *new_fn->GetStageIndex(return_value) != new_fn->stages().size() - 1) {
        // The return value isn't in the last stage (so it's almost certainly a
        // Param); replace it with an identity node referencing the value from
        // the last stage to ensure it gets pipelined.
        XLS_ASSIGN_OR_RETURN(Node * staged_return_value,
                             new_fn->MakeNodeInStage<UnOp>(
                                 new_fn->stages().size() - 1, SourceInfo(),
                                 return_value, Op::kIdentity));
        XLS_RETURN_IF_ERROR(new_fn->set_return_value(staged_return_value));
      }
    } else if (old_fb->IsProc() &&
               (options.codegen_options.generate_combinational() ||
                options.package_schedule.schedules().contains(
                    old_fb->name()))) {
      Proc* new_proc = package->AddProc(
          std::make_unique<ScheduledProc>(old_fb->name(), package));
      new_fb = new_proc;
      new_proc->MoveFrom(*absl::down_cast<Proc*>(old_fb));
      proc_map.emplace(absl::down_cast<Proc*>(old_fb), new_proc);
      XLS_RETURN_IF_ERROR(
          ScheduleNodes(options, absl::down_cast<ScheduledProc*>(new_fb)));
    }

    if (new_fb != nullptr) {
      changed = true;
      if (top.has_value() && top == old_fb) {
        new_fb->SetName(top_name);
        XLS_RETURN_IF_ERROR(package->SetTop(new_fb));
      } else {
        new_fb->SetName(uniquer.GetSanitizedUniqueName(new_fb->name()));
      }

      if (!old_fb->IsProc()) {
        XLS_RETURN_IF_ERROR(package->RemoveFunctionBase(old_fb));
      }
    }
  }

  XLS_RETURN_IF_ERROR(UpdateProcInstantiationsAndRemoveOldProcs(proc_map));
  return changed;
}

}  // namespace xls::codegen
