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

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xls/codegen_v_1_5/block_conversion_pass.h"
#include "xls/common/casts.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/function.h"
#include "xls/ir/node.h"
#include "xls/ir/package.h"
#include "xls/ir/proc.h"
#include "xls/passes/pass_base.h"
#include "xls/scheduling/pipeline_schedule.h"
#include "xls/scheduling/pipeline_schedule.pb.h"

namespace xls::codegen {
namespace {

template <typename ScheduledFunctionBase>
absl::Status ScheduleNodes(const BlockConversionPassOptions& options,
                           ScheduledFunctionBase* fb) {
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
  for (FunctionBase* old_fb : package->GetFunctionBases()) {
    const auto schedule_it =
        options.package_schedule.schedules().find(old_fb->name());
    if (schedule_it == options.package_schedule.schedules().end()) {
      continue;
    }

    FunctionBase* new_fb = nullptr;
    if (old_fb->IsFunction()) {
      Function* new_fn = package->AddFunction(
          std::make_unique<ScheduledFunction>(old_fb->name(), package));
      new_fb = new_fn;
      new_fn->MoveFrom(*down_cast<Function*>(old_fb));
      XLS_RETURN_IF_ERROR(
          ScheduleNodes(options, down_cast<ScheduledFunction*>(new_fb)));
    } else if (old_fb->IsProc()) {
      Proc* new_proc = package->AddProc(
          std::make_unique<ScheduledProc>(old_fb->name(), package));
      new_fb = new_proc;
      new_proc->MoveFrom(*down_cast<Proc*>(old_fb));
      XLS_RETURN_IF_ERROR(
          ScheduleNodes(options, down_cast<ScheduledProc*>(new_fb)));
    }

    if (new_fb != nullptr) {
      std::optional<FunctionBase*> top = package->GetTop();
      if (top.has_value() && top == old_fb) {
        XLS_RETURN_IF_ERROR(package->SetTop(new_fb));
      }
      XLS_RETURN_IF_ERROR(package->RemoveFunctionBase(old_fb));
    }
  }

  return true;
}

}  // namespace xls::codegen
