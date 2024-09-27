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

#include <cstdint>
#include <memory>
#include <optional>
#include <ostream>
#include <string>
#include <vector>

#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xls/interpreter/observer.h"
#include "xls/ir/events.h"
#include "xls/ir/proc.h"
#include "xls/ir/proc_elaboration.h"
#include "xls/ir/value.h"

namespace xls {

// Abstract base class representing a continuation for the evaluation of a
// Proc. The continuation captures the control and data state of the execution
// of a proc tick.
class ProcContinuation {
 public:
  explicit ProcContinuation(ProcInstance* proc_instance)
      : proc_instance_(proc_instance) {}
  virtual ~ProcContinuation() = default;

  // Returns the Proc state at the beginning of the tick currently being
  // executed.
  virtual std::vector<Value> GetState() const = 0;

  // Sets the internal state.
  // Note: Calling when the proc is not AtStartOfTick() may result in
  // incorrect proc behavior.
  virtual absl::Status SetState(std::vector<Value> v) = 0;

  // Returns the events recorded during execution of this continuation.
  virtual const InterpreterEvents& GetEvents() const = 0;
  virtual InterpreterEvents& GetEvents() = 0;
  virtual void ClearEvents() = 0;

  // Returns true if the point of execution of this continuation is at the start
  // of a tick, rather than, for example, blocked on a receive in the middle of
  // a tick execution.
  virtual bool AtStartOfTick() const = 0;

  ProcInstance* proc_instance() const { return proc_instance_; }
  Proc* proc() const { return proc_instance_->proc(); }

  const std::optional<EvaluationObserver*>& GetObserver() const {
    return observer_;
  }

  virtual void ClearObserver() { observer_ = std::nullopt; }

  // Set the callbacks for node calculation. Only one may be set at a time. If
  // this execution environment cannot support the observer api an
  // absl::UnimplementedError will be returned.
  virtual absl::Status SetObserver(EvaluationObserver* observer) {
    CHECK(observer != nullptr);
    observer_ = observer;
    return absl::OkStatus();
  }

  // Does this execution environment support the observer api. If false then
  // setting an observer might fail and callbacks might not always occur or
  // could cause crashes.
  virtual bool SupportsObservers() const { return true; }

 protected:
  absl::Status CheckConformsToStateType(const std::vector<Value>& v) const;

 private:
  ProcInstance* proc_instance_;

  // If not-null an observer for each node in the proc being evaluated.
  std::optional<EvaluationObserver*> observer_ = std::nullopt;
};

// The execution state that a proc may be left in after callin Tick.
enum class TickExecutionState : int8_t {
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

  // If tick state is kBlockedOnReceive or kSentOnChannel then this field holds
  // the respective channel instance.
  std::optional<ChannelInstance*> channel_instance;

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
  explicit ProcEvaluator(Proc* proc);
  virtual ~ProcEvaluator() = default;

  // Creates and returns a new continuation for the proc. The continuation is
  // initialized to start execution at the beginning of the proc with state set
  // to its initial value.
  virtual std::unique_ptr<ProcContinuation> NewContinuation(
      ProcInstance* proc_instance) const = 0;

  // Runs the proc from the given continuation until the tick is complete or
  // execution exits early (e.g., blocked on a receive operation). The
  // continuation is updated in place to reflect the new execution point. If the
  // proc tick completes, the continuation is set to execute the next tick on
  // the subsequent invocation of Tick.
  virtual absl::StatusOr<TickResult> Tick(
      ProcContinuation& continuation) const = 0;

  Proc* proc() const { return proc_; }

  // Returns true if the proc has any send or receive nodes.
  bool ProcHasIoOperations() const { return has_io_operations_; }

 private:
  Proc* proc_;
  bool has_io_operations_;
};

}  // namespace xls

#endif  // XLS_INTERPRETER_PROC_EVALUATOR_H_
