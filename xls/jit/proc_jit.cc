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

#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "xls/common/logging/logging.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/interpreter/proc_evaluator.h"
#include "xls/ir/channel.h"
#include "xls/ir/elaboration.h"
#include "xls/ir/events.h"
#include "xls/ir/node.h"
#include "xls/ir/nodes.h"
#include "xls/ir/proc.h"
#include "xls/ir/value.h"
#include "xls/jit/function_base_jit.h"
#include "xls/jit/ir_builder_visitor.h"
#include "xls/jit/jit_buffer.h"
#include "xls/jit/jit_channel_queue.h"
#include "xls/jit/jit_runtime.h"
#include "xls/jit/observer.h"
#include "xls/jit/orc_jit.h"

namespace xls {

namespace {

// A continuation used by the ProcJit. Stores control and data state of proc
// execution for the JIT.
class ProcJitContinuation : public ProcContinuation {
 public:
  // Construct a new continuation. Execution of the proc begins with the state
  // set to its initial values with no proc nodes yet executed. `queues` is the
  // set of channel queues needed by this proc instance. The order of the queues
  // in the vector is determined at JIT compile time and stored in the
  // JittedFunctionBase.
  explicit ProcJitContinuation(ProcInstance* proc_instance,
                               JitRuntime* jit_runtime,
                               std::vector<JitChannelQueue*> queues,
                               const JittedFunctionBase& jit_func);

  ~ProcJitContinuation() override = default;

  std::vector<Value> GetState() const override;
  const InterpreterEvents& GetEvents() const override { return events_; }
  InterpreterEvents& GetEvents() override { return events_; }
  void ClearEvents() override { events_.Clear(); }

  bool AtStartOfTick() const override { return continuation_point_ == 0; }

  // Get/Set the point at which execution will resume in the proc in the next
  // call to Tick.
  int64_t GetContinuationPoint() const { return continuation_point_; }
  void SetContinuationPoint(int64_t value) { continuation_point_ = value; }

  // Return the various buffers passed to the top-level function implementing
  // the proc.
  JitArgumentSet& input() { return input_; }
  JitArgumentSet& output() { return output_; }
  JitTempBuffer& temp_buffer() { return temp_buffer_; }
  const JitArgumentSet& input() const { return input_; }
  const JitArgumentSet& output() const { return output_; }
  const JitTempBuffer& temp_buffer() const { return temp_buffer_; }

  // Sets the continuation to resume execution at the entry of the proc. Updates
  // state to the "next" value computed in the previous tick.
  absl::Status NextTick();

  InstanceContext* instance_context() { return &instance_context_; }

 private:
  int64_t continuation_point_;
  JitRuntime* jit_runtime_;

  InterpreterEvents events_;

  // Buffers to hold inputs, outputs, and temporary storage. This is allocated
  // once and then re-used with each invocation of Run. Not thread-safe.
  JitArgumentSet input_;
  JitArgumentSet output_;
  JitTempBuffer temp_buffer_;

  // Data structure passed to the JIT function which holds instance related
  // information.
  InstanceContext instance_context_;
};

ProcJitContinuation::ProcJitContinuation(ProcInstance* proc_instance,
                                         JitRuntime* jit_runtime,
                                         std::vector<JitChannelQueue*> queues,
                                         const JittedFunctionBase& jit_func)
    : ProcContinuation(proc_instance),
      continuation_point_(0),
      jit_runtime_(jit_runtime),
      input_(jit_func.CreateInputOutputBuffer().value()),
      output_(jit_func.CreateInputOutputBuffer().value()),
      temp_buffer_(jit_func.CreateTempBuffer()),
      instance_context_{.instance = proc_instance,
                        .channel_queues = std::move(queues)} {
  // Write initial state value to the input_buffer.
  for (Param* state_param : proc()->StateParams()) {
    int64_t param_index = proc()->GetParamIndex(state_param).value();
    int64_t state_index = proc()->GetStateParamIndex(state_param).value();
    jit_runtime->BlitValueToBuffer(
        proc()->GetInitValueElement(state_index), state_param->GetType(),
        absl::Span<uint8_t>(
            input_.pointers()[param_index],
            jit_runtime_->GetTypeByteSize(state_param->GetType())));
  }
}

std::vector<Value> ProcJitContinuation::GetState() const {
  std::vector<Value> state;
  for (Param* state_param : proc()->StateParams()) {
    int64_t param_index = proc()->GetParamIndex(state_param).value();
    state.push_back(jit_runtime_->UnpackBuffer(input_.pointers()[param_index],
                                               state_param->GetType()));
  }
  return state;
}

absl::Status ProcJitContinuation::NextTick() {
  for (auto& [param, active_next_values] :
       instance_context_.active_next_values) {
    if (active_next_values.size() > 1) {
      return absl::AlreadyExistsError(
          absl::StrFormat("Multiple active next values for param \"%s\" in a "
                          "single activation: %s",
                          param->name(),
                          absl::StrJoin(active_next_values, ", ",
                                        [](std::string* out, Next* next) {
                                          absl::StrAppend(out, next->GetName());
                                        })));
    }
  }
  instance_context_.active_next_values.clear();

  continuation_point_ = 0;
  {
    using std::swap;
    swap(input_, output_);
  }
  XLS_VLOG(4) << "New state: "
              << absl::StrJoin(GetState(), ", ",
                               [](std::string* out, const Value& value) {
                                 absl::StrAppend(out, value.ToString());
                               });

  if (!proc()->next_values().empty()) {
    // New-style state param evaluation; initialize the output state params to
    // be unchanged by default.
    for (int64_t state_index = 0; state_index < proc()->GetStateElementCount();
         ++state_index) {
      int64_t param_index = state_index + 1;
      memcpy(output_.pointers()[param_index], input_.pointers()[param_index],
             jit_runtime_->GetTypeByteSize(
                 proc()->GetStateElementType(state_index)));
    }
  }
  return absl::OkStatus();
}

}  // namespace

static absl::StatusOr<ChannelInstance*> GetChannelInstance(
    ProcInstance* proc_instance, std::string_view channel_name,
    JitChannelQueueManager* queue_mgr) {
  if (proc_instance->path().has_value()) {
    // New-style proc-scoped channels.
    return queue_mgr->elaboration().GetChannelInstance(channel_name,
                                                       *proc_instance->path());
  }
  // Old-style global channels.
  XLS_ASSIGN_OR_RETURN(
      Channel * channel,
      proc_instance->proc()->package()->GetChannel(channel_name));
  return queue_mgr->elaboration().GetUniqueInstance(channel);
}

absl::StatusOr<std::unique_ptr<ProcJit>> ProcJit::Create(
    Proc* proc, JitRuntime* jit_runtime, JitChannelQueueManager* queue_mgr,
    JitObserver* observer) {
  XLS_ASSIGN_OR_RETURN(std::unique_ptr<OrcJit> orc_jit, OrcJit::Create());
  orc_jit->SetJitObserver(observer);
  auto jit = absl::WrapUnique(
      new ProcJit(proc, jit_runtime, queue_mgr, std::move(orc_jit)));
  XLS_ASSIGN_OR_RETURN(jit->jitted_function_base_,
                       JittedFunctionBase::Build(proc, jit->GetOrcJit()));
  XLS_RET_CHECK(jit->jitted_function_base_.InputsAndOutputsAreEquivalent());

  for (ProcInstance* proc_instance :
       queue_mgr->elaboration().GetInstances(proc)) {
    jit->channel_queues_[proc_instance].resize(
        jit->jitted_function_base_.queue_indices().size());
    for (const auto& [channel_name, index] :
         jit->jitted_function_base_.queue_indices()) {
      XLS_ASSIGN_OR_RETURN(
          ChannelInstance * channel_instance,
          GetChannelInstance(proc_instance, channel_name, queue_mgr));
      jit->channel_queues_[proc_instance][index] =
          &queue_mgr->GetJitQueue(channel_instance);
    }
  }

  return jit;
}

std::unique_ptr<ProcContinuation> ProcJit::NewContinuation(
    ProcInstance* proc_instance) const {
  CHECK_EQ(proc_instance->proc(), proc());
  return std::make_unique<ProcJitContinuation>(
      proc_instance, jit_runtime_, channel_queues_.at(proc_instance),
      jitted_function_base_);
}

absl::StatusOr<TickResult> ProcJit::Tick(ProcContinuation& continuation) const {
  ProcJitContinuation* cont = dynamic_cast<ProcJitContinuation*>(&continuation);
  XLS_RET_CHECK_NE(cont, nullptr)
      << "ProcJit requires a continuation of type ProcJitContinuation";
  int64_t start_continuation_point = cont->GetContinuationPoint();

  // The jitted function returns the early exit point at which execution
  // halted. A return value of zero indicates that the tick completed.
  int64_t next_continuation_point = jitted_function_base_.RunJittedFunction(
      cont->input(), cont->output(), cont->temp_buffer(), &cont->GetEvents(),
      cont->instance_context(), runtime(), cont->GetContinuationPoint());

  if (next_continuation_point == 0) {
    // The proc successfully completed its tick.
    XLS_RETURN_IF_ERROR(cont->NextTick());
    return TickResult{.execution_state = TickExecutionState::kCompleted,
                      .channel_instance = std::nullopt,
                      .progress_made = true};
  }
  // The proc did not complete the tick. Determine at which node execution was
  // interrupted.
  cont->SetContinuationPoint(next_continuation_point);
  XLS_RET_CHECK(jitted_function_base_.continuation_points().contains(
      next_continuation_point));
  Node* early_exit_node =
      jitted_function_base_.continuation_points().at(next_continuation_point);
  if (early_exit_node->Is<Send>()) {
    // Execution exited after sending data on a channel.
    XLS_ASSIGN_OR_RETURN(
        ChannelInstance * channel_instance,
        GetChannelInstance(continuation.proc_instance(),
                           early_exit_node->As<Send>()->channel_name(),
                           queue_mgr_));

    // The send executed so some progress should have been made.
    XLS_RET_CHECK_NE(next_continuation_point, start_continuation_point);
    return TickResult{.execution_state = TickExecutionState::kSentOnChannel,
                      .channel_instance = channel_instance,
                      .progress_made = true};
  }
  XLS_RET_CHECK(early_exit_node->Is<Receive>());
  XLS_ASSIGN_OR_RETURN(
      ChannelInstance * channel_instance,
      GetChannelInstance(continuation.proc_instance(),
                         early_exit_node->As<Receive>()->channel_name(),
                         queue_mgr_));
  return TickResult{
      .execution_state = TickExecutionState::kBlockedOnReceive,
      .channel_instance = channel_instance,
      .progress_made = next_continuation_point != start_continuation_point};
}

}  // namespace xls
