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

#include "xls/jit/proc_jit.h"

#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "xls/common/status/status_macros.h"
#include "xls/interpreter/serial_proc_runtime.h"
#include "xls/ir/proc.h"
#include "xls/ir/type.h"
#include "xls/ir/value.h"
#include "xls/jit/jit_runtime.h"

namespace xls {

ProcJitContinuation::ProcJitContinuation(Proc* proc, int64_t temp_buffer_size,
                                         JitRuntime* jit_runtime)
    : proc_(proc), continuation_point_(0), jit_runtime_(jit_runtime) {
  // Pre-allocate input, output, and temporary buffers.
  for (Param* param : proc->params()) {
    int64_t param_size = jit_runtime_->GetTypeByteSize(param->GetType());
    input_buffers_.push_back(std::vector<uint8_t>(param_size));
    output_buffers_.push_back(std::vector<uint8_t>(param_size));
    input_ptrs_.push_back(input_buffers_.back().data());
    output_ptrs_.push_back(output_buffers_.back().data());
  }

  // Write initial state value to the input_buffer.
  for (Param* state_param : proc->StateParams()) {
    int64_t param_index = proc->GetParamIndex(state_param).value();
    int64_t state_index = proc->GetStateParamIndex(state_param).value();
    jit_runtime->BlitValueToBuffer(proc->GetInitValueElement(state_index),
                                   state_param->GetType(),
                                   absl::MakeSpan(input_buffers_[param_index]));
  }

  temp_buffer_.resize(temp_buffer_size);
}

std::vector<Value> ProcJitContinuation::GetState() const {
  std::vector<Value> state;
  for (Param* state_param : proc()->StateParams()) {
    int64_t param_index = proc()->GetParamIndex(state_param).value();
    state.push_back(jit_runtime_->UnpackBuffer(input_ptrs_[param_index],
                                               state_param->GetType(),
                                               /*unpoison=*/true));
  }
  return state;
}

void ProcJitContinuation::NextTick() {
  continuation_point_ = 0;
  {
    using std::swap;
    swap(input_buffers_, output_buffers_);
    swap(input_ptrs_, output_ptrs_);
  }
}

absl::StatusOr<std::unique_ptr<ProcJit>> ProcJit::Create(
    Proc* proc, JitRuntime* jit_runtime, JitChannelQueueManager* queue_mgr) {
  XLS_ASSIGN_OR_RETURN(std::unique_ptr<OrcJit> orc_jit, OrcJit::Create());
  auto jit =
      absl::WrapUnique(new ProcJit(proc, jit_runtime, std::move(orc_jit)));
  XLS_ASSIGN_OR_RETURN(jit->jitted_function_base_,
                       BuildProcFunction(proc, queue_mgr, jit->GetOrcJit()));
  return jit;
}

std::unique_ptr<ProcContinuation> ProcJit::NewContinuation() const {
  return std::make_unique<ProcJitContinuation>(
      proc(), jitted_function_base_.temp_buffer_size, jit_runtime_);
}

absl::StatusOr<TickResult> ProcJit::Tick(ProcContinuation& continuation) const {
  ProcJitContinuation* cont = dynamic_cast<ProcJitContinuation*>(&continuation);
  XLS_RET_CHECK_NE(cont, nullptr)
      << "ProcJit requires a continuation of type ProcJitContinuation";
  int64_t start_continuation_point = cont->GetContinuationPoint();

  // The jitted function returns the early exit point at which execution
  // halted. A return value of zero indicates that the tick completed.
  int64_t next_continuation_point = jitted_function_base_.function(
      cont->GetInputBuffers().data(), cont->GetOutputBuffers().data(),
      cont->GetTempBuffer().data(), &cont->GetEvents(),
      /*user_data=*/nullptr, runtime(), cont->GetContinuationPoint());

  if (next_continuation_point == 0) {
    // The proc successfully completed its tick.
    cont->NextTick();
    return TickResult{.execution_state = TickExecutionState::kCompleted,
                      .channel = std::nullopt,
                      .progress_made = true};
  }
  // The proc did not complete the tick. Determine at which node execution was
  // interrupted.
  cont->SetContinuationPoint(next_continuation_point);
  XLS_RET_CHECK(jitted_function_base_.continuation_points.contains(
      next_continuation_point));
  Node* early_exit_node =
      jitted_function_base_.continuation_points.at(next_continuation_point);
  if (early_exit_node->Is<Send>()) {
    // Execution exited after sending data on a channel.
    XLS_ASSIGN_OR_RETURN(Channel * sent_channel,
                         proc()->package()->GetChannel(
                             early_exit_node->As<Send>()->channel_id()));
    // The send executed so some progress should have been made.
    XLS_RET_CHECK_NE(next_continuation_point, start_continuation_point);
    return TickResult{.execution_state = TickExecutionState::kSentOnChannel,
                      .channel = sent_channel,
                      .progress_made = true};
  }
  XLS_RET_CHECK(early_exit_node->Is<Receive>());
  XLS_ASSIGN_OR_RETURN(Channel * blocked_channel,
                       proc()->package()->GetChannel(
                           early_exit_node->As<Receive>()->channel_id()));
  return TickResult{
      .execution_state = TickExecutionState::kBlockedOnReceive,
      .channel = blocked_channel,
      .progress_made = next_continuation_point != start_continuation_point};
}

}  // namespace xls
