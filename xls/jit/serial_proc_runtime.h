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
#ifndef XLS_JIT_SERIAL_PROC_RUNTIME_H_
#define XLS_JIT_SERIAL_PROC_RUNTIME_H_

#include <memory>
#include <vector>

#include "absl/status/statusor.h"
#include "xls/common/thread.h"
#include "xls/ir/package.h"
#include "xls/jit/jit_channel_queue.h"
#include "xls/jit/proc_jit.h"

namespace xls {

// SerialProcRuntime is the "base case" for Proc runtimes. For each clock tick,
// it iterates through the procs in its package and runs them once.
// While basic, this enables steady progression so that a user can see how a
// Proc's internal state (or a proc network's internal state) evolves over time.
// To be able to block a Proc when waiting on input, we use a thread per
// Proc. When a receive is done on an empty queue, that Proc thread will
// conditional-wait until data becomes available, at which point it will
// continue execution.
class SerialProcRuntime {
 public:
  static absl::StatusOr<std::unique_ptr<SerialProcRuntime>> Create(
      Package* package);
  ~SerialProcRuntime();

  // Attempt to progress every proc in the network. Terminates when every
  // block has either completed execution of its `next` function or is blocked
  // on a recv operation.
  absl::Status Tick(bool print_traces = false);

  Package* package() { return package_; }
  JitChannelQueueManager* queue_mgr() { return queue_mgr_.get(); }

  // Enqueues the given set of values into the given channel. 'values' must
  // match the number and type of the data elements of the channel.
  absl::Status EnqueueValueToChannel(Channel* channel, const Value& value);

  // Dequeues a set of values into the given channel. The number and type of the
  // returned values matches the number and type of the data elements of the
  // channel. If the queue is empty, absl::nullopt is returned.
  absl::StatusOr<std::optional<Value>> DequeueValueFromChannel(
      Channel* channel);

  // Returns the current number of procs in this runtime.
  int64_t NumProcs() const;

  // Returns the n'th Proc being executed.
  absl::StatusOr<Proc*> proc(int64_t proc_index) const;

  // Returns the current state values in the given proc.
  absl::StatusOr<std::vector<Value>> ProcState(int64_t proc_index) const;

  void ResetState();

 private:
  // Utility structure to hold state needed by each proc thread.
  struct ThreadData {
    enum class State {
      kPending,
      kRunning,
      kBlocked,
      kDone,
      kCancelled,
    };
    std::unique_ptr<Thread> thread;
    std::unique_ptr<ProcJit> jit;


    absl::Mutex mutex;
    State thread_state ABSL_GUARDED_BY(mutex);
    bool print_traces ABSL_GUARDED_BY(mutex);
    // The Proc's carried state.
    std::vector<Value> proc_state ABSL_GUARDED_BY(mutex);
  };

  SerialProcRuntime(Package* package);
  absl::Status Init();
  static void ThreadFn(ThreadData* thread_data);

  // Proc Receive handler function.
  static bool RecvFn(JitChannelQueue* queue, Receive* recv, uint8_t* data,
                     int64_t data_bytes, void* user_data);

  // Proc Send handler function.
  static void SendFn(JitChannelQueue* queue, Send* send, uint8_t* data,
                     int64_t data_bytes, void* user_data);
  // Blocks the running thread until the given ThreadData is in one of the
  // states specified by "states".
  static void AwaitState(ThreadData* thread_data,
                         const absl::flat_hash_set<ThreadData::State>& states);

  Package* package_;
  std::vector<std::unique_ptr<ThreadData>> threads_;
  std::unique_ptr<JitChannelQueueManager> queue_mgr_;
};

}  // namespace xls

#endif  // XLS_JIT_SERIAL_PROC_RUNTIME_H_
