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
#include "xls/jit/function_builder_visitor.h"
#include "xls/jit/jit_channel_queue.h"
#include "xls/jit/proc_builder_visitor.h"

namespace xls {

void SerialProcRuntime::AwaitState(
    ThreadData* thread_data,
    const absl::flat_hash_set<ThreadData::State>& states) {
  struct AwaitData {
    ThreadData* thread_data;
    const absl::flat_hash_set<ThreadData::State>* states;
  };
  AwaitData await_data = {thread_data, &states};
  thread_data->mutex.AssertHeld();
  thread_data->mutex.Await(absl::Condition(
      +[](AwaitData* await_data) {
        await_data->thread_data->mutex.AssertReaderHeld();
        return await_data->states->contains(
            await_data->thread_data->thread_state);
      },
      &await_data));
}

void SerialProcRuntime::ThreadFn(ThreadData* thread_data) {
  absl::flat_hash_set<ThreadData::State> await_states(
      {ThreadData::State::kRunning, ThreadData::State::kCancelled});
  {
    absl::MutexLock lock(&thread_data->mutex);
    AwaitState(thread_data, await_states);
  }

  while (true) {
    // RunWithViews takes an array of arg view pointers - even if they're unused
    // during execution, tokens still occupy one of those spots.
    std::vector<uint8_t*> args = {nullptr, thread_data->proc_state.get()};
    XLS_CHECK_OK(thread_data->jit->RunWithViews(
        absl::MakeSpan(args),
        absl::MakeSpan(thread_data->proc_state.get(),
                       thread_data->proc_state_size),
        thread_data));

    absl::MutexLock lock(&thread_data->mutex);
    if (thread_data->thread_state == ThreadData::State::kCancelled) {
      break;
    }
    thread_data->thread_state = ThreadData::State::kDone;
    AwaitState(thread_data, await_states);
    if (thread_data->thread_state == ThreadData::State::kCancelled) {
      break;
    }
  }
}

// To implement Proc blocking receive semantics, RecvFn blocks if its associated
// queue is empty. The main thread unblocks it periodically (by changing its
// ThreadData::State) to try to receive again.
void SerialProcRuntime::RecvFn(JitChannelQueue* queue, Receive* recv,
                               uint8_t* data, int64_t data_bytes,
                               void* user_data) {
  ThreadData* thread_data = absl::bit_cast<ThreadData*>(user_data);
  absl::flat_hash_set<ThreadData::State> await_states(
      {ThreadData::State::kRunning, ThreadData::State::kCancelled});

  absl::MutexLock lock(&thread_data->mutex);
  while (queue->Empty()) {
    if (thread_data->thread_state == ThreadData::State::kCancelled) {
      return;
    }
    thread_data->thread_state = ThreadData::State::kBlocked;
    thread_data->blocking_channel = queue->channel_id();
    AwaitState(thread_data, await_states);
  }
  queue->Recv(data, data_bytes);
}

void SerialProcRuntime::SendFn(JitChannelQueue* queue, Send* send,
                               uint8_t* data, int64_t data_bytes,
                               void* user_data) {
  ThreadData* thread_data = absl::bit_cast<ThreadData*>(user_data);
  absl::MutexLock lock(&thread_data->mutex);
  thread_data->sent_data = true;
  queue->Send(data, data_bytes);
}

absl::StatusOr<std::unique_ptr<SerialProcRuntime>> SerialProcRuntime::Create(
    Package* package) {
  auto runtime = absl::WrapUnique(new SerialProcRuntime(std::move(package)));
  XLS_RETURN_IF_ERROR(runtime->Init());
  return runtime;
}

SerialProcRuntime::SerialProcRuntime(Package* package) : package_(package) {}

SerialProcRuntime::~SerialProcRuntime() {
  for (auto& thread_data : threads_) {
    {
      absl::MutexLock lock(&thread_data->mutex);
      thread_data->thread_state = ThreadData::State::kCancelled;
    }
    thread_data->thread->Join();
  }
}

absl::Status SerialProcRuntime::Init() {
  XLS_ASSIGN_OR_RETURN(queue_mgr_, JitChannelQueueManager::Create(package_));

  threads_.reserve(package_->procs().size());
  for (int i = 0; i < package_->procs().size(); i++) {
    auto thread = std::make_unique<ThreadData>();
    Proc* proc = package_->procs()[i].get();
    XLS_ASSIGN_OR_RETURN(thread->jit, IrJit::CreateProc(proc, queue_mgr_.get(),
                                                        &RecvFn, &SendFn));
    auto* jit = thread->jit.get();

    thread->proc_state_size = jit->GetReturnTypeSize();
    thread->proc_state = std::make_unique<uint8_t[]>(thread->proc_state_size);
    jit->runtime()->BlitValueToBuffer(
        proc->InitValue(),
        FunctionBuilderVisitor::GetEffectiveReturnValue(proc)->GetType(),
        absl::MakeSpan(thread->proc_state.get(), jit->GetReturnTypeSize()));

    absl::MutexLock lock(&thread->mutex);
    thread->sent_data = false;
    thread->thread_state = ThreadData::State::kPending;
    threads_.push_back(std::move(thread));

    // Start the thread - the first thing it does is wait until the state is
    // either running or cancelled, so it'll be waiting for us when we actually
    // call Tick().
    auto thread_ptr = threads_.back().get();
    thread_ptr->thread =
        std::make_unique<Thread>([thread_ptr]() { ThreadFn(thread_ptr); });
  }

  // Enqueue initial values into channels.
  for (Channel* channel : package_->channels()) {
    if (channel->kind() != ChannelKind::kStreaming) {
      return absl::UnimplementedError(
          "Only streaming channels are supported in serial proc runtime.");
    }
    for (const Value& value : channel->initial_values()) {
      XLS_RETURN_IF_ERROR(EnqueueValueToChannel(channel, value));
    }
  }

  return absl::OkStatus();
}

absl::Status SerialProcRuntime::Tick() {
  absl::flat_hash_set<ThreadData::State> await_states(
      {ThreadData::State::kBlocked, ThreadData::State::kDone});
  bool done = false;
  while (!done) {
    done = true;
    // True if any proc sent data during this pass/activation/partial cycle.
    bool data_sent = false;
    // True if the proc network is blocked waiting on data from "outside".
    bool blocked_by_external = false;
    for (auto& thread : threads_) {
      absl::MutexLock lock(&thread->mutex);
      if (thread->thread_state == ThreadData::State::kDone) {
        continue;
      }

      // Each blocked thread is stuck on a Condition waiting to be set to
      // kRunning before starting/resuming (so we can ensure serial operation).
      thread->sent_data = false;
      thread->thread_state = ThreadData::State::kRunning;
      AwaitState(thread.get(), await_states);
      if (thread->thread_state != ThreadData::State::kDone) {
        done = false;
      }

      if (thread->thread_state == ThreadData::State::kBlocked) {
        XLS_ASSIGN_OR_RETURN(Channel * chan,
                             package_->GetChannel(thread->blocking_channel));
        if (chan->supported_ops() == ChannelOps::kReceiveOnly) {
          blocked_by_external = true;
        }
      }

      data_sent |= thread->sent_data;
      thread->sent_data = false;
    }

    if (!done && !data_sent && !blocked_by_external) {
      return absl::AbortedError(
          "Deadlock detected; some procs were blocked with no data sent.");
    }
  }

  for (auto& thread : threads_) {
    // Reset state for the next Tick().
    absl::MutexLock lock(&thread->mutex);
    thread->sent_data = false;
    thread->thread_state = ThreadData::State::kPending;
  }

  return absl::OkStatus();
}

absl::Status SerialProcRuntime::EnqueueValueToChannel(Channel* channel,
                                                      const Value& value) {
  XLS_RET_CHECK_EQ(package_->GetTypeForValue(value), channel->type());
  Type* type = package_->GetTypeForValue(value);

  XLS_RET_CHECK(!threads_.empty());
  IrJit* jit = threads_.front()->jit.get();
  int64_t size = jit->type_converter()->GetTypeByteSize(type);
  auto buffer = absl::make_unique<uint8_t[]>(size);
  jit->runtime()->BlitValueToBuffer(value, type,
                                    absl::MakeSpan(buffer.get(), size));

  XLS_ASSIGN_OR_RETURN(JitChannelQueue * queue,
                       queue_mgr()->GetQueueById(channel->id()));
  queue->Send(buffer.get(), size);
  return absl::OkStatus();
}

absl::StatusOr<Value> SerialProcRuntime::DequeueValueFromChannel(
    Channel* channel) {
  Type* type = channel->type();

  XLS_RET_CHECK(!threads_.empty());
  IrJit* jit = threads_.front()->jit.get();
  int64_t size = jit->type_converter()->GetTypeByteSize(type);
  auto buffer = absl::make_unique<uint8_t[]>(size);

  XLS_ASSIGN_OR_RETURN(JitChannelQueue * queue,
                       queue_mgr()->GetQueueById(channel->id()));
  queue->Recv(buffer.get(), size);

  return jit->runtime()->UnpackBuffer(buffer.get(), type);
}

}  // namespace xls
