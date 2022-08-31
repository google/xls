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
    absl::StatusOr<InterpreterResult<std::vector<Value>>> next_state_or =
        thread_data->jit->Run(thread_data->proc_state, thread_data);
    XLS_CHECK_OK(next_state_or.status());

    absl::MutexLock lock(&thread_data->mutex);
    thread_data->proc_state = next_state_or.value().value;
    if (thread_data->thread_state == ThreadData::State::kCancelled) {
      break;
    }

    if (!next_state_or.value().events.trace_msgs.empty()) {
      XLS_LOG(INFO) << "Proc " << thread_data->jit->proc()->name()
                    << " trace messages:";
      for (const std::string& msg : next_state_or.value().events.trace_msgs) {
        XLS_LOG(INFO) << " - " << msg;
      }
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
bool SerialProcRuntime::RecvFn(JitChannelQueue* queue, Receive* recv,
                               uint8_t* data, int64_t data_bytes,
                               void* user_data) {
  if (!recv->is_blocking()) {
    return queue->Recv(data, data_bytes);
  }

  ThreadData* thread_data = absl::bit_cast<ThreadData*>(user_data);
  absl::flat_hash_set<ThreadData::State> await_states(
      {ThreadData::State::kRunning, ThreadData::State::kCancelled});

  absl::MutexLock lock(&thread_data->mutex);
  while (queue->Empty()) {
    if (thread_data->thread_state == ThreadData::State::kCancelled) {
      return false;
    }
    thread_data->thread_state = ThreadData::State::kBlocked;
    AwaitState(thread_data, await_states);
  }

  bool received_data = queue->Recv(data, data_bytes);
  if (XLS_VLOG_IS_ON(3)) {
    std::string data;
    for (int i = 0; i < data_bytes; i++) {
      absl::StrAppend(&data,
                      absl::StrFormat("0x%2x", static_cast<uint32_t>(data[i])));
    }
    XLS_LOG(INFO) << thread_data->jit->proc()->name()
                  << ": recv data: " << data;
  }

  return received_data;
}

void SerialProcRuntime::SendFn(JitChannelQueue* queue, Send* send,
                               uint8_t* data, int64_t data_bytes,
                               void* user_data) {
  ThreadData* thread_data = absl::bit_cast<ThreadData*>(user_data);
  absl::MutexLock lock(&thread_data->mutex);
  if (XLS_VLOG_IS_ON(3)) {
    std::string data;
    for (int i = 0; i < data_bytes; i++) {
      absl::StrAppend(&data,
                      absl::StrFormat("0x%2x", static_cast<uint32_t>(data[i])));
    }
    XLS_LOG(INFO) << thread_data->jit->proc()->name()
                  << ": send data: " << data;
  }
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
    XLS_ASSIGN_OR_RETURN(
        thread->jit, ProcJit::Create(proc, queue_mgr_.get(), &RecvFn, &SendFn));

    absl::MutexLock lock(&thread->mutex);
    thread->thread_state = ThreadData::State::kPending;
    thread->print_traces = false;
    threads_.push_back(std::move(thread));

    // Start the thread - the first thing it does is wait until the state is
    // either running or cancelled, so it'll be waiting for us when we actually
    // call Tick().
    auto thread_ptr = threads_.back().get();
    thread_ptr->thread =
        std::make_unique<Thread>([thread_ptr]() { ThreadFn(thread_ptr); });
  }

  ResetState();

  // Enqueue initial values into channels.
  for (Channel* channel : package_->channels()) {
    for (const Value& value : channel->initial_values()) {
      XLS_RETURN_IF_ERROR(EnqueueValueToChannel(channel, value));
    }
  }

  return absl::OkStatus();
}

absl::Status SerialProcRuntime::Tick(bool print_traces) {
  absl::flat_hash_set<ThreadData::State> await_states(
      {ThreadData::State::kBlocked, ThreadData::State::kDone});
  for (auto& thread : threads_) {
    absl::MutexLock lock(&thread->mutex);
    // Each blocked thread is stuck on a Condition waiting to be set to
    // kRunning before starting/resuming (so we can ensure serial operation).
    thread->thread_state = ThreadData::State::kRunning;
    thread->print_traces = print_traces;
    AwaitState(thread.get(), await_states);
  }

  return absl::OkStatus();
}

absl::Status SerialProcRuntime::EnqueueValueToChannel(Channel* channel,
                                                      const Value& value) {
  XLS_RET_CHECK_EQ(package_->GetTypeForValue(value), channel->type());
  Type* type = package_->GetTypeForValue(value);

  XLS_RET_CHECK(!threads_.empty());
  ProcJit* jit = threads_.front()->jit.get();
  int64_t size = jit->type_converter()->GetTypeByteSize(type);
  auto buffer = std::make_unique<uint8_t[]>(size);
  jit->runtime()->BlitValueToBuffer(value, type,
                                    absl::MakeSpan(buffer.get(), size));

  XLS_ASSIGN_OR_RETURN(JitChannelQueue * queue,
                       queue_mgr()->GetQueueById(channel->id()));
  queue->Send(buffer.get(), size);
  return absl::OkStatus();
}

absl::StatusOr<std::optional<Value>> SerialProcRuntime::DequeueValueFromChannel(
    Channel* channel) {
  Type* type = channel->type();

  XLS_RET_CHECK(!threads_.empty());
  ProcJit* jit = threads_.front()->jit.get();
  int64_t size = jit->type_converter()->GetTypeByteSize(type);
  auto buffer = std::make_unique<uint8_t[]>(size);

  XLS_ASSIGN_OR_RETURN(JitChannelQueue * queue,
                       queue_mgr()->GetQueueById(channel->id()));
  bool had_data = queue->Recv(buffer.get(), size);
  if (!had_data) {
    return absl::nullopt;
  }

  return jit->runtime()->UnpackBuffer(buffer.get(), type);
}

int64_t SerialProcRuntime::NumProcs() const { return threads_.size(); }

absl::StatusOr<Proc*> SerialProcRuntime::proc(int64_t index) const {
  if (index > threads_.size()) {
    return absl::InvalidArgumentError(
        absl::StrCat("Valid indices are 0 - ", threads_.size(), "."));
  }
  return dynamic_cast<Proc*>(threads_[index]->jit->proc());
}

absl::StatusOr<std::vector<Value>>
SerialProcRuntime::SerialProcRuntime::ProcState(int64_t index) const {
  if (index > threads_.size()) {
    return absl::InvalidArgumentError(
        absl::StrCat("Valid indices are 0 - ", threads_.size(), "."));
  }

  return threads_[index]->proc_state;
}

void SerialProcRuntime::ResetState() {
  for (int i = 0; i < package_->procs().size(); i++) {
    Proc* proc = package_->procs()[i].get();
    auto thread = threads_[i].get();
    absl::MutexLock lock(&thread->mutex);
    thread->proc_state = std::vector<Value>(proc->InitValues().begin(),
                                            proc->InitValues().end());
  }
}

}  // namespace xls
