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

#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "xls/interpreter/channel_queue.h"
#include "xls/interpreter/evaluator_options.h"
#include "xls/interpreter/observer.h"
#include "xls/interpreter/proc_evaluator.h"
#include "xls/ir/events.h"
#include "xls/ir/package.h"
#include "xls/ir/proc_elaboration.h"
#include "xls/ir/value.h"
#include "xls/jit/jit_channel_queue.h"

namespace xls {

// Abstract base class for interpreting the procs within a package.
class ProcRuntime {
 public:
  ProcRuntime(
      absl::flat_hash_map<Proc*, std::unique_ptr<ProcEvaluator>>&& evaluators,
      std::unique_ptr<ChannelQueueManager>&& queue_manager,
      const EvaluatorOptions& options);

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

  // Tick the proc network until some output channels (channel instances) have
  // produced at least a specified number of outputs as indicated by
  // `output_counts`. `output_counts` must only contain output channels and need
  // not contain all output channels. Returns the number of ticks executed
  // before the conditions were met. `max_ticks` is the maximum number of ticks
  // of the proc network before returning an error.
  absl::StatusOr<int64_t> TickUntilOutput(
      const absl::flat_hash_map<Channel*, int64_t>& output_counts,
      std::optional<int64_t> max_ticks = std::nullopt);
  absl::StatusOr<int64_t> TickUntilOutput(
      const absl::flat_hash_map<ChannelInstance*, int64_t>& output_counts,
      std::optional<int64_t> max_ticks = std::nullopt);

  // Tick until all procs with IO (send or receive nodes) are blocked on receive
  // operations. `max_ticks` is the maximum number of ticks of the proc network
  // before returning an error. Note: some proc networks are not guaranteed to
  // block even if given no inputs. `max_ticks` is the maximum number of ticks
  // of the proc network before returning an error.
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
  std::vector<Value> ResolveState(ProcInstance* instance) const {
    return continuations_.at(instance)->GetState();
  }
  std::vector<Value> ResolveState(Proc* proc) const {
    return continuations_.at(elaboration().GetUniqueInstance(proc).value())
        ->GetState();
  }

  // Updates the state values for a proc in the network.
  absl::Status SetState(ProcInstance* instance, std::vector<Value> v) {
    return continuations_.at(instance)->SetState(std::move(v));
  }
  absl::Status SetState(Proc* proc, std::vector<Value> v) {
    return continuations_.at(elaboration().GetUniqueInstance(proc).value())
        ->SetState(std::move(v));
  }

  // Reset the state of all of the procs to their initial state.
  void ResetState();

  // Returns the events for each proc in the network.
  const InterpreterEvents& GetInterpreterEvents(ProcInstance* instance) const {
    return continuations_.at(instance)->GetEvents();
  }
  const InterpreterEvents& GetInterpreterEvents(Proc* proc) const {
    return continuations_.at(elaboration().GetUniqueInstance(proc).value())
        ->GetEvents();
  }

  // Return the events which are not associated with any particular proc (e.g.,
  // trace messages for channel activity).
  InterpreterEvents GetGlobalEvents() const;
  void ClearInterpreterEvents();

  const ProcElaboration& elaboration() const {
    return queue_manager_->elaboration();
  }

  Package* package() const { return elaboration().package(); }

  void ClearObserver();

  // Set the callbacks for node calculation. Only one may be set at a time. If
  // this execution environment cannot support the observer api an
  // absl::UnimplementedError will be returned.
  absl::Status SetObserver(EvaluationObserver* obs);

  // Does this execution environment support the observer api. If false then
  // setting an observer might fail and callbacks might not always occur or
  // could cause crashes.
  bool SupportsObservers() const;

 protected:
  friend class ChannelTraceRecorder;
  void AddTraceMessage(TraceMessage message);

  // Execute (up to) a single iteration of every proc in the package.
  struct NetworkTickResult {
    // Whether any instruction on any proc executed.
    bool progress_made;

    // Whether any instruction on a proc with IO executed
    bool progress_made_on_io_procs;

    std::vector<ChannelInstance*> blocked_channel_instances;
  };
  virtual absl::StatusOr<NetworkTickResult> TickInternal() = 0;

  std::unique_ptr<ChannelQueueManager> queue_manager_;
  absl::flat_hash_map<Proc*, std::unique_ptr<ProcEvaluator>> evaluators_;
  absl::flat_hash_map<ProcInstance*, std::unique_ptr<ProcContinuation>>
      continuations_;

  mutable absl::Mutex global_events_mutex_;
  InterpreterEvents global_events_ ABSL_GUARDED_BY(global_events_mutex_);

  EvaluatorOptions options_;
  std::optional<EvaluationObserver*> observer_ = std::nullopt;
};

}  // namespace xls

#endif  // XLS_INTERPRETER_PROC_RUNTIME_H_
