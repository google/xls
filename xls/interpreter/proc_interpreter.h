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

#ifndef XLS_INTERPRETER_PROC_INTERPRETER_H_
#define XLS_INTERPRETER_PROC_INTERPRETER_H_

#include <deque>
#include <limits>
#include <memory>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xls/interpreter/channel_queue.h"
#include "xls/ir/channel.h"
#include "xls/ir/events.h"
#include "xls/ir/proc.h"
#include "xls/ir/value.h"

namespace xls {

// Abstract base class representing a continuation for the evaluation of a
// Proc. The continuation captures the control and data state of the execution
// of a proc tick.
class ProcContinuation {
 public:
  virtual ~ProcContinuation() = default;

  // Returns the Proc state at the beginning of the tick currently being
  // executed.
  virtual std::vector<Value> GetState() const = 0;

  // Returns the events recorded during execution of this continuation.
  virtual const InterpreterEvents& GetEvents() const = 0;
  virtual InterpreterEvents& GetEvents() = 0;

  // Returns true if the point of execution of this continuation is at the start
  // of a tick, rather than, for example, blocked on a receive in the middle of
  // a tick execution.
  virtual bool AtStartOfTick() const = 0;
};

// Data structure holding the result of a single call to Tick.
struct TickResult {
  // Whether the proc completed executing for this tick. Execution is not
  // completed if (and only if) a receive operation is blocked.
  bool tick_complete;

  // Whether any progress was made (at least one instruction was executed).
  bool progress_made;

  // If the proc is blocked on receive (iteration complete is false), this
  // vector includes the channels which have blocked operations.
  std::optional<Channel*> blocked_channel;

  // Vector of channels which were sent on by this proc in this invocation of
  // Tick.
  std::vector<Channel*> sent_channels;

  bool operator==(const TickResult& other) const;
  bool operator!=(const TickResult& other) const;

  std::string ToString() const;
};

// Abstract base class for evaluators of procs (e.g., interpreter or JIT).
class ProcEvaluator {
 public:
  virtual ~ProcEvaluator() = default;

  // Creates and returns a new continuation for the proc. The continuation is
  // initialized to start execution at the beginning of the proc with state set
  // to its initial value.
  virtual std::unique_ptr<ProcContinuation> NewContinuation() const = 0;

  // Runs the proc from the given continuation until the tick is complete or
  // execution is blocked on a receive operation. The continuation is updated in
  // place to reflect the new execution point. If the proc tick completes, the
  // continuation is set to execute the next tick on the subsequent invocation
  // of Tick.
  virtual absl::StatusOr<TickResult> Tick(
      ProcContinuation& continuation) const = 0;

  virtual Proc* proc() const = 0;
};

std::ostream& operator<<(std::ostream& os, const TickResult& result);

// A continuation used by the ProcInterpreter.
class ProcInterpreterContinuation : public ProcContinuation {
 public:
  // Construct a new continuation. Execution the proc begins with the state set
  // to its initial values with no proc nodes yet executed.
  explicit ProcInterpreterContinuation(Proc* proc)
      : node_index_(0),
        state_(proc->InitValues().begin(), proc->InitValues().end()) {}

  ~ProcInterpreterContinuation() override = default;

  std::vector<Value> GetState() const override { return state_; }
  const InterpreterEvents& GetEvents() const override { return events_; }
  InterpreterEvents& GetEvents() override { return events_; }
  bool AtStartOfTick() const override { return node_index_ == 0; }

  // Resets the continuation so it will start executing at the beginning of the
  // proc with the given state values.
  void NextTick(std::vector<Value>&& next_state) {
    node_index_ = 0;
    state_ = next_state;
    node_values_.clear();
  }

  // Gets/sets the index of the node to be executed next. This index refers to a
  // place in a topological sort of the proc nodes held by the ProcInterpreter.
  int64_t GetNodeExecutionIndex() const { return node_index_; }
  void SetNodeExecutionIndex(int64_t index) { node_index_ = index; }

  // Returns the map of node values computed in the tick so far.
  absl::flat_hash_map<Node*, Value>& GetNodeValues() { return node_values_; }
  const absl::flat_hash_map<Node*, Value>& GetNodeValues() const {
    return node_values_;
  }

 private:
  int64_t node_index_;
  std::vector<Value> state_;

  InterpreterEvents events_;
  absl::flat_hash_map<Node*, Value> node_values_;
};

// A interpreter for an individual proc. Incrementally executes Procs a single
// tick at a time. Data is fed to the proc via ChannelQueues.  ProcInterpreters
// are thread-save if called with different continuations.
class ProcInterpreter : public ProcEvaluator {
 public:
  ProcInterpreter(Proc* proc, ChannelQueueManager* queue_manager);
  ProcInterpreter(const ProcInterpreter&) = delete;
  ProcInterpreter operator=(const ProcInterpreter&) = delete;

  virtual ~ProcInterpreter() = default;

  std::unique_ptr<ProcContinuation> NewContinuation() const override;
  absl::StatusOr<TickResult> Tick(
      ProcContinuation& continuation) const override;
  Proc* proc() const override { return proc_; }

 private:
  Proc* proc_;
  ChannelQueueManager* queue_manager_;

  // A topological sort of the nodes of the proc which determines the execution
  // order of the proc.
  std::vector<Node*> execution_order_;
};

}  // namespace xls

#endif  // XLS_INTERPRETER_PROC_INTERPRETER_H_
