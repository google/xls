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
    int64_t param_size =
        jit_runtime_->type_converter()->GetTypeByteSize(param->GetType());
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
    Proc* proc, JitChannelQueueManager* queue_mgr, int64_t opt_level) {
  auto jit = absl::WrapUnique(new ProcJit(proc));
  XLS_ASSIGN_OR_RETURN(jit->orc_jit_,
                       OrcJit::Create(opt_level, /*emit_object_code=*/false));
  jit->ir_runtime_ = std::make_unique<JitRuntime>(
      jit->orc_jit_->GetDataLayout(), &jit->orc_jit_->GetTypeConverter());
  XLS_ASSIGN_OR_RETURN(jit->jitted_function_base_,
                       BuildProcFunction(proc, queue_mgr, jit->GetOrcJit()));
  return jit;
}

std::unique_ptr<ProcContinuation> ProcJit::NewContinuation() const {
  return std::make_unique<ProcJitContinuation>(
      proc(), jitted_function_base_.temp_buffer_size, ir_runtime_.get());
}

absl::StatusOr<TickResult> ProcJit::Tick(ProcContinuation& continuation) const {
  ProcJitContinuation* cont = dynamic_cast<ProcJitContinuation*>(&continuation);
  XLS_RET_CHECK_NE(cont, nullptr)
      << "ProcJit requires a continuation of type ProcJitContinuation";
  std::vector<Channel*> sent_channels;

  InterpreterEvents events;

  int64_t start_continuation_point = cont->GetContinuationPoint();
  int64_t next_continuation_point = jitted_function_base_.function(
      cont->GetInputBuffers().data(), cont->GetOutputBuffers().data(),
      cont->GetTempBuffer().data(), &cont->GetEvents(),
      /*user_data=*/nullptr, runtime(), cont->GetContinuationPoint());

  std::optional<Channel*> blocked_channel;
  if (next_continuation_point == 0) {
    // The proc successfully completed its tick.
    cont->NextTick();
  } else {
    // The proc is blocked. Determine which node (and associated channel) the
    // node is blocke on.
    cont->SetContinuationPoint(next_continuation_point);
    XLS_RET_CHECK(jitted_function_base_.continuation_points.contains(
        next_continuation_point));
    Node* blocked_node =
        jitted_function_base_.continuation_points.at(next_continuation_point);
    XLS_RET_CHECK(blocked_node->Is<Receive>());
    XLS_ASSIGN_OR_RETURN(blocked_channel,
                         proc()->package()->GetChannel(
                             blocked_node->As<Receive>()->channel_id()));
  }
  return TickResult{
      .tick_complete = next_continuation_point == 0,
      .progress_made = next_continuation_point == 0 ||
                       next_continuation_point != start_continuation_point,
      .blocked_channel = blocked_channel,
      // TODO(meheff): 2022/09/23 Add the channels the proc sent on here, or
      // alternatively devise a different mechanism for determining which procs
      // to wake up.
      .sent_channels = {}};
}

}  // namespace xls
