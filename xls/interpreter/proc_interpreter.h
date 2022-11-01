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
#include "xls/interpreter/proc_evaluator.h"
#include "xls/interpreter/serial_proc_runtime.h"
#include "xls/ir/events.h"
#include "xls/ir/proc.h"
#include "xls/ir/value.h"

namespace xls {

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
// are thread-safe if called with different continuations.
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
