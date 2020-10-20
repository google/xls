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
#include "xls/jit/ir_jit.h"
#include "xls/jit/jit_channel_queue.h"

namespace xls {

// SerialProcRuntime is the "base case" for Proc runtimes. For each clock tick,
// it iterates through the procs in its package and runs them once, all within a
// single thread. While basic, this enables steady progression so that a
// user can see how a proc's internal state (or a proc network's internal state)
// evolves over time.
class SerialProcRuntime {
 public:
  static absl::StatusOr<std::unique_ptr<SerialProcRuntime>> Create(
      Package* package);

  // Execute one cycle of every proc in the network.
  absl::Status Tick();

  Package* package() { return package_; }
  JitChannelQueueManager* queue_mgr() { return queue_mgr_.get(); }

 private:
  // Utility structure to bind a compiled proc with its current state.
  struct ProcData {
    std::unique_ptr<IrJit> jit;
    std::unique_ptr<uint8[]> value_buffer;
    int64 value_buffer_size;
  };

  SerialProcRuntime(Package* package);
  absl::Status Init();

  Package* package_;
  std::vector<ProcData> procs_;
  std::unique_ptr<JitChannelQueueManager> queue_mgr_;
};

}  // namespace xls

#endif  // XLS_JIT_SERIAL_PROC_RUNTIME_H_
