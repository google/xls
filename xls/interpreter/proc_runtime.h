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

#ifndef XLS_INTERPRETER_PROC_RUNTIME_H_
#define XLS_INTERPRETER_PROC_RUNTIME_H_

#include <memory>
#include <vector>

#include "absl/status/statusor.h"
#include "xls/interpreter/channel_queue.h"
#include "xls/interpreter/proc_evaluator.h"
#include "xls/ir/events.h"
#include "xls/ir/package.h"
#include "xls/jit/jit_channel_queue.h"

namespace xls {

// Abstract base class for interpreting the procs within a package.
class ProcRuntime {
 public:
  ProcRuntime(
      Package* package,
      absl::flat_hash_map<Proc*, std::unique_ptr<ProcEvaluator>>&& evaluators,
      std::unique_ptr<ChannelQueueManager>&& queue_manager);

  virtual ~ProcRuntime() = default;

  // Execute (up to) a single iteration of every proc in the package. If no
  // conditional send/receive nodes exist in the package then calling Tick will
  // execute exactly one iteration for all procs in the package. If conditional
  // send/receive nodes do exist, then some procs may be blocked in a state
  // where the iteration is partially complete. In this case, the call to Tick()
  // will not execute a complete iteration of the proc. Calling Tick() again
  // will resume these procs from their partially executed state. Returns an
  // error if no progress can be made due to a deadlock.
  absl::Status Tick();

  // Tick the proc network until some output channels have produced at least a
  // specified number of outputs as indicated by `output_counts`.
  // `output_counts` must only contain output channels and need not contain all
  // output channels. Returns the number of ticks executed before the conditions
  // were met. `max_ticks` is the maximum number of ticks of the proc network
  // before returning an error.
  absl::StatusOr<int64_t> TickUntilOutput(
      absl::flat_hash_map<Channel*, int64_t> output_counts,
      std::optional<int64_t> max_ticks = std::nullopt);

  // Tick until all procs are blocked on receive operations. `max_ticks` is the
  // maximum number of ticks of the proc network before returning an
  // error. Note: some proc networks are not guaranteed to block even if given
  // no inputs. `max_ticks` is the maximum number of ticks of the proc network
  // before returning an error.
  absl::StatusOr<int64_t> TickUntilBlocked(
      std::optional<int64_t> max_ticks = std::nullopt);

  ChannelQueueManager& queue_manager() { return *queue_manager_; }

  // If the contained Channel queue manager is a JitChannelQueueManager then
  // return it. Otherwise raise an error.
  // TODO(meheff): 2022/10/26 Determine if there is a way of eliminating this
  // method. Perhaps adding "raw" read/write methods to the ChannelQueue API
  // making JitChannelQueues interface the same as a ChannelQueue
  absl::StatusOr<JitChannelQueueManager*> GetJitChannelQueueManager();

  // Returns the state values for a proc in the network.
  std::vector<Value> ResolveState(Proc* proc) const {
    return evaluator_contexts_.at(proc).continuation->GetState();
  }

  // Reset the state of all of the procs to their initial state.
  void ResetState();

  // Returns the events for each proc in the network.
  const InterpreterEvents& GetInterpreterEvents(Proc* proc) const {
    return evaluator_contexts_.at(proc).continuation->GetEvents();
  }

 protected:
  // Execute (up to) a single iteration of every proc in the package.
  struct NetworkTickResult {
    bool progress_made;
    std::vector<Channel*> blocked_channels;
  };
  virtual absl::StatusOr<NetworkTickResult> TickInternal() = 0;

  Package* package_;
  std::unique_ptr<ChannelQueueManager> queue_manager_;
  struct EvaluatorContext {
    std::unique_ptr<ProcEvaluator> evaluator;
    std::unique_ptr<ProcContinuation> continuation;
  };
  absl::flat_hash_map<Proc*, EvaluatorContext> evaluator_contexts_;
};

}  // namespace xls

#endif  // XLS_INTERPRETER_PROC_RUNTIME_H_
