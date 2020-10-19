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

#include <cstddef>
#include <thread>

#include "absl/status/status.h"
#include "xls/common/cleanup.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/proc.h"
#include "xls/jit/jit_channel_queue.h"
#include "xls/jit/proc_builder_visitor.h"

namespace xls {
namespace {

// Utility structure to hold the data needed for thread execution.
struct ThreadData {
  enum class State {
    kRunning,
    kBlocked,
    kDone,
    kCancelled,
  };

  std::thread thread;
  IrJit* proc_jit;

  // The value of the proc persistent state variable. Not to be confused with
  // "thread_state" below!
  // Owned by SerialProcRuntime::procs_, which should outlive any thread.
  Value* proc_state;

  absl::Mutex mutex;

  // The current run status of the thread, per the State enum above.
  State thread_state GUARDED_BY(mutex);

  // If thread_state is kBlocked, then this holds the ID of the blocking
  // channel.
  int64 blocking_channel GUARDED_BY(mutex);

  // True if this Thread sent data during its most recent activation.
  bool sent_data GUARDED_BY(mutex);
};

void ThreadFn(ThreadData* thread_data) {
  std::vector<Value> args({Value::Token(), *thread_data->proc_state});
  *thread_data->proc_state =
      thread_data->proc_jit->Run(args, thread_data).value().element(1);
  absl::MutexLock lock(&thread_data->mutex);
  thread_data->thread_state = ThreadData::State::kDone;
}

// To implement Proc blocking receive semantics, RecvFn blocks if its associated
// queue is empty. The main thread unblocks it periodically (by changing its
// ThreadData::State) to try to receive again.
void RecvFn(JitChannelQueue* queue, Receive* recv, uint8* buffer, int64 bytes,
            void* user_data) {
  ThreadData* thread_data = reinterpret_cast<ThreadData*>(user_data);
  absl::MutexLock lock(&thread_data->mutex);
  while (queue->Empty()) {
    if (thread_data->thread_state == ThreadData::State::kCancelled) {
      return;
    }
    thread_data->thread_state = ThreadData::State::kBlocked;
    thread_data->blocking_channel = queue->channel_id();
    thread_data->mutex.Await(absl::Condition(
        +[](ThreadData* thread_data) {
          thread_data->mutex.AssertReaderHeld();
          return thread_data->thread_state == ThreadData::State::kRunning ||
                 thread_data->thread_state == ThreadData::State::kCancelled;
        },
        thread_data));
  }
  queue->Recv(buffer, bytes);
}

void SendFn(JitChannelQueue* queue, Send* send, uint8* buffer, int64 bytes,
            void* user_data) {
  ThreadData* thread_data = reinterpret_cast<ThreadData*>(user_data);
  absl::MutexLock lock(&thread_data->mutex);
  thread_data->sent_data = true;
  queue->Send(buffer, bytes);
}

}  // namespace

absl::StatusOr<std::unique_ptr<SerialProcRuntime>> SerialProcRuntime::Create(
    Package* package) {
  auto runtime = absl::WrapUnique(new SerialProcRuntime(std::move(package)));
  XLS_RETURN_IF_ERROR(runtime->Init());
  return runtime;
}

SerialProcRuntime::SerialProcRuntime(Package* package)
    : package_(std::move(package)) {}

absl::Status SerialProcRuntime::Init() {
  XLS_ASSIGN_OR_RETURN(queue_mgr_, JitChannelQueueManager::Create(package_));

  for (const auto& proc : package_->procs()) {
    XLS_ASSIGN_OR_RETURN(
        auto jit,
        IrJit::CreateProc(proc.get(), queue_mgr_.get(), &RecvFn, &SendFn));
    procs_.push_back({std::move(jit), proc->InitValue()});
  }
  return absl::OkStatus();
}

absl::Status SerialProcRuntime::Tick() {
  absl::Mutex status_mutex;
  absl::CondVar status_condvar;

  std::vector<ThreadData> thread_data(procs_.size());
  auto cleanup = xabsl::MakeCleanup([&thread_data]() {
    for (ThreadData& thread_data : thread_data) {
      thread_data.mutex.Lock();
      thread_data.thread_state = ThreadData::State::kCancelled;
      thread_data.mutex.Unlock();
      thread_data.thread.join();
    }
  });
  bool done = true;
  for (int i = 0; i < procs_.size(); i++) {
    ThreadData& curr = thread_data[i];

    curr.proc_jit = procs_[i].jit.get();
    curr.proc_state = &procs_[i].state;
    absl::MutexLock lock(&curr.mutex);
    curr.sent_data = false;
    curr.thread_state = ThreadData::State::kRunning;
    curr.thread = std::thread([&curr]() { ThreadFn(&curr); });

    // Start each thread and wait for it to quiesce before starting another (to
    // keep execution serial).
    curr.mutex.Await(absl::Condition(
        +[](ThreadData* data) {
          data->mutex.AssertReaderHeld();
          return data->thread_state == ThreadData::State::kBlocked ||
                 data->thread_state == ThreadData::State::kDone;
        },
        &curr));

    if (thread_data.back().thread_state != ThreadData::State::kDone) {
      done = false;
    }
  }

  while (!done) {
    // Each blocked thread is stuck on a Condition waiting to be set to kRunning
    // before resuming (so we can ensure serial operation).
    done = true;
    // True if any proc sent data during this pass/activation/partial cycle.
    bool data_sent = false;
    // True if the proc network is blocked waiting on data from "outside".
    bool blocked_by_external = false;
    for (auto& thread : thread_data) {
      thread.mutex.Lock();

      if (thread.thread_state == ThreadData::State::kDone) {
        // REALLY, we don't need to hold the lock to inspect thread.state here,
        // since the thread hasn't run/taken the lock since we last looked at
        // the value, but I think it's less confusing this way (and this runtime
        // isn't performance-sensitive).
        thread.mutex.Unlock();
        continue;
      }

      thread.thread_state = ThreadData::State::kRunning;
      thread.mutex.Await(absl::Condition(
          +[](ThreadData* data) {
            data->mutex.AssertHeld();
            return data->thread_state == ThreadData::State::kBlocked ||
                   data->thread_state == ThreadData::State::kDone;
          },
          &thread));
      if (thread.thread_state != ThreadData::State::kDone) {
        done = false;
      }

      if (thread.thread_state == ThreadData::State::kBlocked) {
        auto status_or_chan = package_->GetChannel(thread.blocking_channel);
        if (!status_or_chan.ok()) {
          thread.mutex.Unlock();
          return status_or_chan.status();
        }

        if (status_or_chan.value()->kind() == ChannelKind::kReceiveOnly) {
          blocked_by_external = true;
        }
      }

      data_sent |= thread.sent_data;
      thread.sent_data = false;
      thread.mutex.Unlock();
    }

    if (!done && !data_sent && !blocked_by_external) {
      return absl::AbortedError(
          "Deadlock detected; some procs were blocked with no data sent.");
    }
  }

  return absl::OkStatus();
}

absl::StatusOr<Value> SerialProcRuntime::GetProcState(
    const std::string& proc_name) const {
  for (const auto& proc : procs_) {
    if (proc.jit->function()->name() == proc_name) {
      return proc.state;
    }
  }

  return absl::NotFoundError(
      absl::StrCat("Proc \"", proc_name, "\" not found in package."));
}

}  // namespace xls
