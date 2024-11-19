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

#include "xls/interpreter/proc_interpreter.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/interpreter/channel_queue.h"
#include "xls/interpreter/ir_interpreter.h"
#include "xls/interpreter/observer.h"
#include "xls/interpreter/proc_evaluator.h"
#include "xls/ir/bits.h"
#include "xls/ir/events.h"
#include "xls/ir/node.h"
#include "xls/ir/nodes.h"
#include "xls/ir/proc.h"
#include "xls/ir/proc_elaboration.h"
#include "xls/ir/state_element.h"
#include "xls/ir/topo_sort.h"
#include "xls/ir/value.h"
#include "xls/ir/value_utils.h"

namespace xls {
namespace {

// A continuation used by the ProcInterpreter.
class ProcInterpreterContinuation : public ProcContinuation {
 public:
  // Construct a new continuation. Execution the proc begins with the state set
  // to its initial values with no proc nodes yet executed.
  explicit ProcInterpreterContinuation(ProcInstance* proc_instance)
      : ProcContinuation(proc_instance), node_index_(0) {
    state_.reserve(proc()->GetStateElementCount());
    for (StateElement* state_element : proc()->StateElements()) {
      state_.push_back(state_element->initial_value());
    }
  }

  ~ProcInterpreterContinuation() override = default;

  std::vector<Value> GetState() const override { return state_; }

  absl::Status SetState(std::vector<Value> v) override {
    XLS_RETURN_IF_ERROR(CheckConformsToStateType(v));
    state_ = std::move(v);
    return absl::OkStatus();
  }

  const InterpreterEvents& GetEvents() const override { return events_; }
  InterpreterEvents& GetEvents() override { return events_; }
  void ClearEvents() override { events_.Clear(); }
  bool AtStartOfTick() const override { return node_index_ == 0; }

  const absl::flat_hash_map<StateElement*, std::vector<Next*>>&
  GetActiveNextValues() const {
    return active_next_values_;
  }
  absl::flat_hash_map<StateElement*, std::vector<Next*>>&
  GetActiveNextValues() {
    return active_next_values_;
  }
  void ClearActiveNextValues() { active_next_values_.clear(); }

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
  absl::flat_hash_map<StateElement*, std::vector<Next*>> active_next_values_;
};

// A visitor for interpreting procs. Adds handlers for send and receive
// communicate via ChannelQueues.
class ProcIrInterpreter : public IrInterpreter {
 public:
  // Constructor args:
  //   proc_instance: the instance of the proc which is being interpreted.
  //   state: is the value to use for the proc state in the tick being
  //     interpreted.
  //   node_values: map from Node to Value for already computed values in this
  //     tick of the proc. Used for continuations.
  //   events: events object to record events in (e.g, traces).
  //   queue_manager: manager for channel queues.
  ProcIrInterpreter(ProcInstance* proc_instance, absl::Span<const Value> state,
                    absl::flat_hash_map<Node*, Value>* node_values,
                    InterpreterEvents* events,
                    ChannelQueueManager* queue_manager,
                    absl::flat_hash_map<StateElement*, std::vector<Next*>>*
                        active_next_values,
                    std::optional<EvaluationObserver*> observer)
      : IrInterpreter(node_values, events, observer),
        proc_instance_(proc_instance),
        state_(state.begin(), state.end()),
        queue_manager_(queue_manager),
        active_next_values_(active_next_values) {}

  absl::Status HandleReceive(Receive* receive) override {
    XLS_ASSIGN_OR_RETURN(ChannelQueue * queue,
                         GetChannelQueue(receive->channel_name()));

    if (receive->predicate().has_value()) {
      const Bits& pred = ResolveAsBits(receive->predicate().value());
      if (pred.IsZero()) {
        // If the predicate is false, nothing is read from the channel.
        // Rather the result of the receive is the zero values of the
        // respective type.
        return SetValueResult(receive, ZeroOfType(receive->GetType()));
      }
    }

    std::optional<Value> value = queue->Read();
    if (!value.has_value()) {
      if (receive->is_blocking()) {
        // Record the channel this receive instruction is blocked on and exit.
        blocked_channel_instance_ = queue->channel_instance();
        return absl::OkStatus();
      }
      // A non-blocking receive returns a zero data value with a zero valid bit
      // if the queue is empty.
      return SetValueResult(receive, ZeroOfType(receive->GetType()));
    }

    if (receive->is_blocking()) {
      return SetValueResult(receive, Value::Tuple({Value::Token(), *value}));
    }

    return SetValueResult(
        receive, Value::Tuple({Value::Token(), *value, Value(UBits(1, 1))}));
  }

  absl::Status HandleSend(Send* send) override {
    XLS_ASSIGN_OR_RETURN(ChannelQueue * queue,
                         GetChannelQueue(send->channel_name()));
    if (send->predicate().has_value()) {
      const Bits& pred = ResolveAsBits(send->predicate().value());
      if (pred.IsZero()) {
        return SetValueResult(send, Value::Token());
      }
    }
    // Indicate that data is sent on this channel.
    sent_channel_instance_ = queue->channel_instance();

    XLS_RETURN_IF_ERROR(queue->Write(ResolveAsValue(send->data())));

    // The result of a send is simply a token.
    return SetValueResult(send, Value::Token());
  }

  absl::Status HandleStateRead(StateRead* state_read) override {
    XLS_ASSIGN_OR_RETURN(
        int64_t index,
        state_read->function_base()->AsProcOrDie()->GetStateElementIndex(
            state_read->state_element()));
    return SetValueResult(state_read, state_[index]);
  }

  absl::Status HandleNext(Next* next) override {
    if (next->predicate().has_value()) {
      Value predicate = ResolveAsValue(*next->predicate());
      XLS_RET_CHECK(predicate.IsBits());
      XLS_RET_CHECK_EQ(predicate.bits().bit_count(), 1);
      if (predicate.bits().IsZero()) {
        // No change.
        return SetValueResult(next, Value::Tuple({}));
      }
    }
    (*active_next_values_)[next->state_read()->As<StateRead>()->state_element()]
        .push_back(next);
    return SetValueResult(next, Value::Tuple({}));
  }

  // Executes a single node and return whether the node is blocked on a channel
  // (for receive nodes) or whether data was sent on a channel (for send nodes).
  struct NodeResult {
    std::optional<ChannelInstance*> blocked_channel_instance;
    std::optional<ChannelInstance*> sent_channel_instance;
  };
  absl::StatusOr<NodeResult> ExecuteNode(Node* node) {
    // Send/Receive handlers might set these values so clear them before hand.
    blocked_channel_instance_ = std::nullopt;
    sent_channel_instance_ = std::nullopt;
    XLS_RETURN_IF_ERROR(node->VisitSingleNode(this));
    return NodeResult{.blocked_channel_instance = blocked_channel_instance_,
                      .sent_channel_instance = sent_channel_instance_};
  }

 private:
  // Get the channel queue for the channel or channel reference of the given
  // name.
  absl::StatusOr<ChannelQueue*> GetChannelQueue(std::string_view name) {
    if (proc_instance_->path().has_value()) {
      // New-style proc-scoped channel.
      XLS_ASSIGN_OR_RETURN(ChannelInstance * channel_instance,
                           queue_manager_->elaboration().GetChannelInstance(
                               name, *proc_instance_->path()));
      return &queue_manager_->GetQueue(channel_instance);
    }
    // Old-style global channel.
    return queue_manager_->GetQueueByName(name);
  }

  ProcInstance* proc_instance_;
  std::vector<Value> state_;
  ChannelQueueManager* queue_manager_;

  absl::flat_hash_map<StateElement*, std::vector<Next*>>* active_next_values_;

  // Ephemeral values set by the send/receive handlers indicating the channel
  // execution is blocked on or the channel on which data was sent.
  std::optional<ChannelInstance*> blocked_channel_instance_;
  std::optional<ChannelInstance*> sent_channel_instance_;
};

}  // namespace

ProcInterpreter::ProcInterpreter(Proc* proc, ChannelQueueManager* queue_manager)
    : ProcEvaluator(proc),
      queue_manager_(queue_manager),
      execution_order_(TopoSort(proc)) {}

std::unique_ptr<ProcContinuation> ProcInterpreter::NewContinuation(
    ProcInstance* proc_instance) const {
  return std::make_unique<ProcInterpreterContinuation>(proc_instance);
}

absl::StatusOr<TickResult> ProcInterpreter::Tick(
    ProcContinuation& continuation) const {
  ProcInterpreterContinuation* cont =
      dynamic_cast<ProcInterpreterContinuation*>(&continuation);
  XLS_RET_CHECK_NE(cont, nullptr) << "ProcInterpreter requires a continuation "
                                     "of type ProcInterpreterContinuation";

  ProcIrInterpreter ir_interpreter(cont->proc_instance(), cont->GetState(),
                                   &cont->GetNodeValues(), &cont->GetEvents(),
                                   queue_manager_, &cont->GetActiveNextValues(),
                                   continuation.GetObserver());

  // Resume execution at the node indicated in the continuation
  // (NodeExecutionIndex).
  int64_t starting_index = cont->GetNodeExecutionIndex();
  for (int64_t i = starting_index; i < execution_order_.size(); ++i) {
    Node* node = execution_order_[i];
    XLS_ASSIGN_OR_RETURN(ProcIrInterpreter::NodeResult result,
                         ir_interpreter.ExecuteNode(node));
    if (result.sent_channel_instance.has_value()) {
      // Early exit: proc sent on a channel. Execution should resume _after_ the
      // send.
      cont->SetNodeExecutionIndex(i + 1);
      // Raise a status error if interpreter events indicate failure such as a
      // failed assert.
      XLS_RETURN_IF_ERROR(InterpreterEventsToStatus(cont->GetEvents()));
      return TickResult{
          .execution_state = TickExecutionState::kSentOnChannel,
          .channel_instance = result.sent_channel_instance,
          .progress_made = cont->GetNodeExecutionIndex() != starting_index};
    }
    if (result.blocked_channel_instance.has_value()) {
      // Early exit: proc is blocked at a receive node waiting for data on a
      // channel. Execution should resume at the send.
      cont->SetNodeExecutionIndex(i);
      // Raise a status error if interpreter events indicate failure such as a
      // failed assert.
      XLS_RETURN_IF_ERROR(InterpreterEventsToStatus(cont->GetEvents()));
      return TickResult{
          .execution_state = TickExecutionState::kBlockedOnReceive,
          .channel_instance = result.blocked_channel_instance,
          .progress_made = cont->GetNodeExecutionIndex() != starting_index};
    }
  }

  // Proc completed execution of the Tick. Pass the next proc state to the
  // continuation.
  //
  // TODO: Simplify this once fully transitioned over to `next_value` nodes.
  std::vector<Value> next_state;
  next_state.resize(proc()->GetStateElementCount());
  for (int64_t index = 0; index < proc()->NextState().size(); ++index) {
    next_state[index] =
        ir_interpreter.ResolveAsValue(proc()->GetNextStateElement(index));
  }
  for (const auto& [state_element, next_values] : cont->GetActiveNextValues()) {
    if (next_values.size() > 1) {
      return absl::AlreadyExistsError(absl::StrFormat(
          "Multiple active next values for state element %d (\"%s\") in a "
          "single activation: %s",
          *proc()->GetStateElementIndex(state_element), state_element->name(),
          absl::StrJoin(next_values, ", ", [](std::string* out, Next* next) {
            absl::StrAppend(out, next->GetName());
          })));
    }

    XLS_ASSIGN_OR_RETURN(int64_t index,
                         proc()->GetStateElementIndex(state_element));
    next_state[index] = ir_interpreter.ResolveAsValue(next_values[0]->value());
  }
  cont->ClearActiveNextValues();
  cont->NextTick(std::move(next_state));

  // Raise a status error if interpreter events indicate failure such as a
  // failed assert.
  XLS_RETURN_IF_ERROR(InterpreterEventsToStatus(cont->GetEvents()));

  return TickResult{.execution_state = TickExecutionState::kCompleted,
                    .channel_instance = std::nullopt,
                    .progress_made = true};
}

}  // namespace xls
