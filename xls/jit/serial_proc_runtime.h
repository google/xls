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
#include "xls/jit/ir_jit.h"
#include "xls/jit/jit_channel_queue.h"

namespace xls {

// SerialProcRuntime is the "base case" for Proc runtimes. For each clock tick,
// it iterates through the procs in its package and runs them once, all within a
// single thread. While basic, this enables steady progression so that a
// user can see how a Proc's internal state (or a proc network's internal state)
// evolves over time.
// To be able to block/suspect a Proc when waiting on input, we use a thread per
// Proc. When a receive is done on an empty queue, that Proc thread will
// conditional-wait until data becomes available, at which point it will
// continue execution. In this way, a single Tick() may span multiple thread
// activations and suspends, but will terminate once the cycle has completed or
// when a deadlock is detected.
class SerialProcRuntime {
 public:
  static absl::StatusOr<std::unique_ptr<SerialProcRuntime>> Create(
      Package* package);
  ~SerialProcRuntime();

  // Execute one cycle of every proc in the network.
  absl::Status Tick();

  Package* package() { return package_; }
  JitChannelQueueManager* queue_mgr() { return queue_mgr_.get(); }

  // Enqueues the given set of values into the given channel. 'values' must
  // match the number and type of the data elements of the channel.
  absl::Status EnqueueValueToChannel(Channel* channel, const Value& value);

  // Dequeues a set of values into the given channel. The number and type of the
  // returned values matches the number and type of the data elements of the
  // channel.
  absl::StatusOr<Value> DequeueValueFromChannel(Channel* channel);

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
    std::unique_ptr<IrJit> jit;

    // The size of and actual buffer used to hold the Proc's carried state.
    int64_t proc_state_size;
    std::unique_ptr<uint8_t[]> proc_state;

    absl::Mutex mutex;
    State thread_state ABSL_GUARDED_BY(mutex);

    // True if the proc sent out data during its last activation. Used to detect
    // network deadlock.
    bool sent_data ABSL_GUARDED_BY(mutex);

    // True if this proc is blocked on data coming from "outside" the network,
    // i.e., a receive_only channel. Stops network deadlock false positives.
    int64_t blocking_channel ABSL_GUARDED_BY(mutex);
  };

  SerialProcRuntime(Package* package);
  absl::Status Init();
  static void ThreadFn(ThreadData* thread_data);

  // Proc Receive handler function.
  static void RecvFn(JitChannelQueue* queue, Receive* recv, uint8_t* data,
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
