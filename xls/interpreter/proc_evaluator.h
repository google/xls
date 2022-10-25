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

#ifndef XLS_INTERPRETER_PROC_EVALUATOR_H_
#define XLS_INTERPRETER_PROC_EVALUATOR_H_

#include <memory>
#include <vector>

#include "absl/status/statusor.h"
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

// The execution state that a proc may be left in after callin Tick.
enum class TickExecutionState {
  // The proc tick completed.
  kCompleted,
  // The proc tick was blocked on a blocking receive.
  kBlockedOnReceive,
  // The proc tick exited early because it sent data a channel. The proc is not
  // blocked and execution can resume.
  kSentOnChannel,
};

std::string ToString(TickExecutionState state);

// Data structure holding the result of a single call to Tick.
struct TickResult {
  TickExecutionState execution_state;

  // If tick state is kBlockedOnReceive of kSentOnChannel then this field holds
  // the respective channel.
  std::optional<Channel*> channel;

  // Whether any progress was made (at least one instruction was executed).
  bool progress_made;

  bool operator==(const TickResult& other) const;
  bool operator!=(const TickResult& other) const;

  std::string ToString() const;
};

std::ostream& operator<<(std::ostream& os, TickExecutionState state);
std::ostream& operator<<(std::ostream& os, const TickResult& result);

// Abstract base class for evaluators of procs (e.g., interpreter or JIT).
class ProcEvaluator {
 public:
  virtual ~ProcEvaluator() = default;

  // Creates and returns a new continuation for the proc. The continuation is
  // initialized to start execution at the beginning of the proc with state set
  // to its initial value.
  virtual std::unique_ptr<ProcContinuation> NewContinuation() const = 0;

  // Runs the proc from the given continuation until the tick is complete or
  // execution exits early (e.g., blocked on a receive operation). The
  // continuation is updated in place to reflect the new execution point. If the
  // proc tick completes, the continuation is set to execute the next tick on
  // the subsequent invocation of Tick.
  virtual absl::StatusOr<TickResult> Tick(
      ProcContinuation& continuation) const = 0;

  virtual Proc* proc() const = 0;
};

}  // namespace xls

#endif  // XLS_INTERPRETER_PROC_EVALUATOR_H_
