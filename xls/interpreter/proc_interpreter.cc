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

#include "absl/status/statusor.h"
#include "absl/strings/str_join.h"
#include "xls/common/logging/log_lines.h"
#include "xls/ir/node_iterator.h"
#include "xls/ir/proc.h"
#include "xls/ir/value_helpers.h"

namespace xls {
namespace {

// A visitor for interpreting procs. Adds handlers for send and receive
// communcate via ChannelQueues.
class ProcIrInterpreter : public IrInterpreter {
 public:
  // "state" is the value to use for the proc state during interpretation.
  ProcIrInterpreter(absl::Span<const Value> state,
                    ChannelQueueManager* queue_manager)
      : IrInterpreter(),
        state_(state.begin(), state.end()),
        queue_manager_(queue_manager) {}

  absl::Status HandleReceive(Receive* receive) override {
    XLS_ASSIGN_OR_RETURN(ChannelQueue * queue,
                         queue_manager_->GetQueueById(receive->channel_id()));

    if (receive->predicate().has_value()) {
      const Bits& pred = ResolveAsBits(receive->predicate().value());
      if (pred.IsZero()) {
        // If the predicate is false, nothing is dequeued from the channel.
        // Rather the result of the receive is the zero values of the
        // respective type.
        return SetValueResult(receive, ZeroOfType(receive->GetType()));
      }
    }

    // If this is a non-blocking queue, if the queue is empty, then
    // return a value of zero for that type.
    if (!receive->is_blocking() && queue->empty()) {
      return SetValueResult(receive, ZeroOfType(receive->GetType()));
    }

    XLS_ASSIGN_OR_RETURN(Value value, queue->Dequeue());

    if (receive->is_blocking()) {
      return SetValueResult(receive, Value::Tuple({Value::Token(), value}));
    }

    return SetValueResult(
        receive, Value::Tuple({Value::Token(), value, Value(UBits(1, 1))}));
  }

  absl::Status HandleSend(Send* send) override {
    XLS_ASSIGN_OR_RETURN(ChannelQueue * queue,
                         queue_manager_->GetQueueById(send->channel_id()));
    if (send->predicate().has_value()) {
      const Bits& pred = ResolveAsBits(send->predicate().value());
      if (pred.IsZero()) {
        return SetValueResult(send, Value::Token());
      }
    }
    XLS_RETURN_IF_ERROR(queue->Enqueue(ResolveAsValue(send->data())));

    // The result of a send is simply a token.
    return SetValueResult(send, Value::Token());
  }

  absl::Status HandleParam(Param* param) override {
    XLS_ASSIGN_OR_RETURN(int64_t index,
                         param->function_base()->GetParamIndex(param));
    if (index == 0) {
      return SetValueResult(param, Value::Token());
    }
    // Params from 1 on are state.
    return SetValueResult(param, state_[index - 1]);
  }

 private:
  std::vector<Value> state_;

  ChannelQueueManager* queue_manager_;
};

}  // namespace

bool ProcInterpreter::RunResult::operator==(
    const ProcInterpreter::RunResult& other) const {
  return iteration_complete == other.iteration_complete &&
         progress_made == other.progress_made &&
         blocked_channels == other.blocked_channels;
}

bool ProcInterpreter::RunResult::operator!=(
    const ProcInterpreter::RunResult& other) const {
  return !(*this == other);
}

ProcInterpreter::ProcInterpreter(Proc* proc, ChannelQueueManager* queue_manager)
    : proc_(proc),
      queue_manager_(queue_manager),
      topo_sort_(TopoSort(proc)),
      current_iteration_(0) {}

bool ProcInterpreter::IsIterationComplete() const {
  return visitor_ == nullptr ||
         (std::all_of(proc_->NextState().begin(), proc_->NextState().end(),
                      [&](Node* n) { return visitor_->IsVisited(n); }) &&
          visitor_->IsVisited(proc_->NextToken()));
}

absl::StatusOr<std::vector<Value>> ProcInterpreter::ResolveState() const {
  std::vector<Value> results;
  for (Node* next_node : proc_->NextState()) {
    if (!visitor_->HasResult(next_node)) {
      return absl::NotFoundError(absl::StrFormat(
          "Proc next state has not been computed: %s", next_node->GetName()));
    }
    results.push_back(visitor_->ResolveAsValue(next_node));
  }
  return results;
}

absl::StatusOr<ProcInterpreter::RunResult>
ProcInterpreter::RunIterationUntilCompleteOrBlocked() {
  XLS_VLOG(3) << absl::StreamFormat(
      "%s iteration %d of proc %s",
      (IsIterationComplete() ? "Running" : "Resuming"), current_iteration_,
      proc_->name());
  XLS_VLOG_LINES(4, proc_->DumpIr());

  if (IsIterationComplete()) {
    // Previous iteration was complete or this the first time this method has
    // been called. Create a new visitor for evaluating the nodes this
    // iteration.
    if (visitor_ == nullptr) {
      // This is the first time the proc has run. Proc state is the init value.
      ResetState();
    } else {
      XLS_ASSIGN_OR_RETURN(std::vector<Value> next_state, ResolveState());
      visitor_ =
          std::make_unique<ProcIrInterpreter>(next_state, queue_manager_);
    }
  }

  RunResult result{.iteration_complete = true,
                   .progress_made = false,
                   .blocked_channels = {}};
  auto executed_this_iteration = [&](Node* node) {
    return visitor_->IsVisited(node);
  };

  // TODO(meheff): Iterating through all the nodes every time is
  // inefficient. It'd be better to continue from some checkpointed state.
  for (Node* node : topo_sort_) {
    if (executed_this_iteration(node)) {
      continue;
    }
    if (std::all_of(node->operands().begin(), node->operands().end(),
                    executed_this_iteration)) {
      // Check to see if this is a receive node which is blocked.
      if (node->Is<Receive>() && node->As<Receive>()->is_blocking()) {
        Receive* receive = node->As<Receive>();
        bool predicate = !receive->predicate().has_value() ||
                         visitor_->ResolveAsValue(receive->predicate().value())
                             .bits()
                             .IsOne();
        XLS_ASSIGN_OR_RETURN(
            ChannelQueue * queue,
            queue_manager_->GetQueueById(node->As<Receive>()->channel_id()));
        if (predicate && queue->empty()) {
          // Queue is empty, receive is blocked.
          XLS_VLOG(4) << absl::StreamFormat(
              "Receive node %s blocked on channel with ID %d", node->GetName(),
              node->As<Receive>()->channel_id());
          result.blocked_channels.push_back(queue->channel());
          result.iteration_complete = false;
          continue;
        }
      }

      // Node is ready to execute.
      XLS_VLOG(4) << absl::StreamFormat("Node %s executing", node->GetName());
      XLS_RETURN_IF_ERROR(node->VisitSingleNode(visitor_.get()));
      visitor_->MarkVisited(node);

      result.progress_made = true;
    } else {
      XLS_VLOG(4) << absl::StreamFormat("Node %s not ready to execute",
                                        node->GetName());
    }
  }

  // Raise a status error if interpreter events indicate failure such as a
  // failed assert.
  XLS_RETURN_IF_ERROR(
      InterpreterEventsToStatus(visitor_->GetInterpreterEvents()));

  // Sort blocked_channels vector by channel id.
  std::sort(result.blocked_channels.begin(), result.blocked_channels.end(),
            [](Channel* a, Channel* b) { return a->id() < b->id(); });

  XLS_VLOG(3) << absl::StreamFormat("Proc %s run result: %s", proc_->name(),
                                    result.ToString());
  if (result.iteration_complete) {
    ++current_iteration_;
  }
  return result;
}

void ProcInterpreter::ResetState() {
  visitor_ =
      std::make_unique<ProcIrInterpreter>(proc_->InitValues(), queue_manager_);
}

std::string ProcInterpreter::RunResult::ToString() const {
  return absl::StrFormat(
      "{ iteration_complete=%s, progress_made=%s, "
      "blocked_channels={%s} }",
      iteration_complete ? "true" : "false", progress_made ? "true" : "false",
      absl::StrJoin(blocked_channels, ", ", ChannelFormatter));
}

std::ostream& operator<<(std::ostream& os,
                         const ProcInterpreter::RunResult& result) {
  os << result.ToString();
  return os;
}

}  // namespace xls
