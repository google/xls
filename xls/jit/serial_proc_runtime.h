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
#include "xls/ir/package.h"
#include "xls/jit/jit_channel_queue.h"
#include "xls/jit/proc_jit.h"

namespace xls {

// SerialProcRuntime is the "base case" for Proc runtimes. For each clock tick,
// it iterates through the procs in its package and runs them once.
// While basic, this enables steady progression so that a user can see how a
// Proc's internal state (or a proc network's internal state) evolves over time.
// TODO(meheff): 2022/09/23 Merge this with the ProcNetworkInterpreter which is
// also serial once the JIT and interpreter share the same channel types.
class SerialProcRuntime {
 public:
  static absl::StatusOr<std::unique_ptr<SerialProcRuntime>> Create(
      Package* package);
  ~SerialProcRuntime() = default;

  // Attempt to progress every proc in the network. Terminates when every
  // block has either completed execution of its `next` function or is blocked
  // on a recv operation.
  absl::Status Tick(bool print_traces = false);

  Package* package() { return package_; }
  JitChannelQueueManager* queue_mgr() { return queue_mgr_.get(); }

  // Writes the values to the given channel. `value` must match the type of the
  // data element of the channel.
  absl::Status WriteValueToChannel(Channel* channel, const Value& value);
  absl::Status WriteBufferToChannel(Channel* channel,
                                    absl::Span<uint8_t const> buffer);

  // Reads a value from the given channel. The type of the returned value
  // matches the type of the data elements of the channel. If the queue is
  // empty, absl::nullopt is returned.
  absl::StatusOr<std::optional<Value>> ReadValueFromChannel(Channel* channel);
  absl::StatusOr<bool> ReadBufferFromChannel(Channel* channel,
                                             absl::Span<uint8_t> buffer);

  // Returns the current state values in the given proc.
  absl::StatusOr<std::vector<Value>> ProcState(Proc* proc) const;

  void ResetState();

  const InterpreterEvents& GetEvents(Proc* proc) const {
    return continuations_.at(proc)->GetEvents();
  }

  JitRuntime& jit_runtime() { return *jit_runtime_; }

 private:
  SerialProcRuntime(Package* package);
  absl::Status Init();

  Package* package_;
  std::unique_ptr<JitChannelQueueManager> queue_mgr_;
  std::unique_ptr<JitRuntime> jit_runtime_;

  absl::flat_hash_map<Proc*, std::unique_ptr<ProcJit>> proc_jits_;
  absl::flat_hash_map<Proc*, std::unique_ptr<ProcContinuation>> continuations_;
};

}  // namespace xls

#endif  // XLS_JIT_SERIAL_PROC_RUNTIME_H_
