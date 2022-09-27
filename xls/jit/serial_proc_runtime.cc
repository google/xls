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
#include "xls/jit/serial_proc_runtime.h"

#include "absl/status/status.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/proc.h"
#include "xls/jit/jit_channel_queue.h"
#include "xls/jit/proc_jit.h"

namespace xls {

absl::StatusOr<std::unique_ptr<SerialProcRuntime>> SerialProcRuntime::Create(
    Package* package) {
  auto runtime = absl::WrapUnique(new SerialProcRuntime(std::move(package)));
  XLS_RETURN_IF_ERROR(runtime->Init());
  return runtime;
}

SerialProcRuntime::SerialProcRuntime(Package* package) : package_(package) {}

absl::Status SerialProcRuntime::Init() {
  XLS_ASSIGN_OR_RETURN(queue_mgr_, JitChannelQueueManager::Create(package_));
  jit_runtime_ = nullptr;
  // Create a ProcJit and continuation for each proc.
  for (const std::unique_ptr<Proc>& proc : package_->procs()) {
    XLS_ASSIGN_OR_RETURN(proc_jits_[proc.get()],
                         ProcJit::Create(proc.get(), queue_mgr_.get()));
    continuations_[proc.get()] = proc_jits_.at(proc.get())->NewContinuation();
    if (jit_runtime_ == nullptr) {
      jit_runtime_ = proc_jits_[proc.get()]->runtime();
    }
  }
  XLS_RET_CHECK_NE(jit_runtime_, nullptr);

  // Enqueue initial values into channels.
  for (Channel* channel : package_->channels()) {
    for (const Value& value : channel->initial_values()) {
      XLS_RETURN_IF_ERROR(EnqueueValueToChannel(channel, value));
    }
  }
  return absl::OkStatus();
}

absl::Status SerialProcRuntime::Tick(bool print_traces) {
  bool progress_made = true;
  absl::flat_hash_set<Proc*> completed_procs;
  // In round-robin fashion, run each proc until every proc has either completed
  // their tick or are blocked on receives.
  while (progress_made) {
    progress_made = false;
    for (const std::unique_ptr<Proc>& proc : package_->procs()) {
      if (completed_procs.contains(proc.get())) {
        continue;
      }
      XLS_ASSIGN_OR_RETURN(
          TickResult result,
          proc_jits_.at(proc.get())->Tick(*continuations_.at(proc.get())));
      progress_made = progress_made || result.progress_made;
      if (result.tick_complete) {
        completed_procs.insert(proc.get());
      }
    }
  }
  return absl::OkStatus();
}

absl::Status SerialProcRuntime::EnqueueValueToChannel(Channel* channel,
                                                      const Value& value) {
  XLS_RET_CHECK_EQ(package_->GetTypeForValue(value), channel->type());
  Type* type = package_->GetTypeForValue(value);

  int64_t size = jit_runtime_->type_converter()->GetTypeByteSize(type);
  auto buffer = std::make_unique<uint8_t[]>(size);
  jit_runtime_->BlitValueToBuffer(value, type,
                                  absl::MakeSpan(buffer.get(), size));

  XLS_ASSIGN_OR_RETURN(JitChannelQueue * queue,
                       queue_mgr()->GetQueueById(channel->id()));
  queue->Send(buffer.get(), size);
  return absl::OkStatus();
}

absl::Status SerialProcRuntime::EnqueueBufferToChannel(
    Channel* channel, absl::Span<uint8_t const> buffer) {
  int64_t size =
      jit_runtime_->type_converter()->GetTypeByteSize(channel->type());
  XLS_RET_CHECK_GE(buffer.size(), size);
  XLS_ASSIGN_OR_RETURN(JitChannelQueue * queue,
                       queue_mgr()->GetQueueById(channel->id()));
  queue->Send(buffer.data(), size);
  return absl::OkStatus();
}

absl::StatusOr<std::optional<Value>> SerialProcRuntime::DequeueValueFromChannel(
    Channel* channel) {
  Type* type = channel->type();
  int64_t size = jit_runtime_->type_converter()->GetTypeByteSize(type);
  auto buffer = std::make_unique<uint8_t[]>(size);

  XLS_ASSIGN_OR_RETURN(JitChannelQueue * queue,
                       queue_mgr()->GetQueueById(channel->id()));
  bool had_data = queue->Recv(buffer.get(), size);
  if (!had_data) {
    return absl::nullopt;
  }

  return jit_runtime_->UnpackBuffer(buffer.get(), type);
}

absl::StatusOr<bool> SerialProcRuntime::DequeueBufferFromChannel(
    Channel* channel, absl::Span<uint8_t> buffer) {
  Type* type = channel->type();

  int64_t size = jit_runtime_->type_converter()->GetTypeByteSize(type);
  XLS_RET_CHECK_GE(buffer.size(), size);

  XLS_ASSIGN_OR_RETURN(JitChannelQueue * queue,
                       queue_mgr()->GetQueueById(channel->id()));
  bool had_data = queue->Recv(buffer.data(), size);
  if (!had_data) {
    return false;
  }
  return true;
}

absl::StatusOr<std::vector<Value>>
SerialProcRuntime::SerialProcRuntime::ProcState(Proc* proc) const {
  return continuations_.at(proc)->GetState();
}

void SerialProcRuntime::ResetState() {
  for (const std::unique_ptr<Proc>& proc : package_->procs()) {
    continuations_.at(proc.get()) =
        proc_jits_.at(proc.get())->NewContinuation();
  }
}

}  // namespace xls
