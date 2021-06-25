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
  ProcIrInterpreter(const Value& state, ChannelQueueManager* queue_manager)
      : IrInterpreter(), state_(state), queue_manager_(queue_manager) {}

  absl::Status HandleReceive(Receive* receive) override {
    XLS_ASSIGN_OR_RETURN(ChannelQueue * queue,
                         queue_manager_->GetQueueById(receive->channel_id()));
    XLS_ASSIGN_OR_RETURN(Value value, queue->Dequeue());
    return SetValueResult(receive, Value::Tuple({Value::Token(), value}));
  }

  absl::Status HandleReceiveIf(ReceiveIf* receive_if) override {
    const Bits& pred = ResolveAsBits(receive_if->predicate());
    if (pred.IsZero()) {
      // If the predicate is false, nothing is dequeued from the channel. Rather
      // the result of the receive_if is the zero values of the respective
      // type.
      return SetValueResult(receive_if, ZeroOfType(receive_if->GetType()));
    }

    XLS_ASSIGN_OR_RETURN(ChannelQueue * queue, queue_manager_->GetQueueById(
                                                   receive_if->channel_id()));
    XLS_ASSIGN_OR_RETURN(Value value, queue->Dequeue());
    return SetValueResult(receive_if, Value::Tuple({Value::Token(), value}));
  }

  absl::Status HandleSend(Send* send) override {
    XLS_ASSIGN_OR_RETURN(ChannelQueue * queue,
                         queue_manager_->GetQueueById(send->channel_id()));
    XLS_RETURN_IF_ERROR(queue->Enqueue(ResolveAsValue(send->data())));

    // The result of a send is simply a token.
    return SetValueResult(send, Value::Token());
  }

  absl::Status HandleSendIf(SendIf* send_if) override {
    const Bits& pred = ResolveAsBits(send_if->predicate());
    if (pred.IsOne()) {
      XLS_ASSIGN_OR_RETURN(ChannelQueue * queue,
                           queue_manager_->GetQueueById(send_if->channel_id()));
      XLS_RETURN_IF_ERROR(queue->Enqueue(ResolveAsValue(send_if->data())));
    }

    // The result of a send_if is simply a token.
    return SetValueResult(send_if, Value::Token());
  }

  absl::Status HandleParam(Param* param) override {
    XLS_ASSIGN_OR_RETURN(int64_t index,
                         param->function_base()->GetParamIndex(param));
    XLS_RET_CHECK(index == 0 || index == 1);
    if (index == 0) {
      return SetValueResult(param, Value::Token());
    } else {
      return SetValueResult(param, state_);
    }
  }

 private:
  Value state_;

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
  return visitor_ == nullptr || (visitor_->IsVisited(proc_->NextState()) &&
                                 visitor_->IsVisited(proc_->NextToken()));
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
      visitor_ = absl::make_unique<ProcIrInterpreter>(proc_->InitValue(),
                                                      queue_manager_);
    } else {
      const Value& next_state = visitor_->ResolveAsValue(proc_->NextState());
      visitor_ =
          absl::make_unique<ProcIrInterpreter>(next_state, queue_manager_);
      ++current_iteration_;
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
      if (node->Is<Receive>()) {
        XLS_ASSIGN_OR_RETURN(
            ChannelQueue * queue,
            queue_manager_->GetQueueById(node->As<Receive>()->channel_id()));
        if (queue->empty()) {
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
  // Sort blocked_channels vector by channel id.
  std::sort(result.blocked_channels.begin(), result.blocked_channels.end(),
            [](Channel* a, Channel* b) { return a->id() < b->id(); });

  XLS_VLOG(3) << absl::StreamFormat("Proc %s run result: %s", proc_->name(),
                                    result.ToString());
  return result;
}

std::string ProcInterpreter::RunResult::ToString() const {
  return absl::StrFormat(
      "{ iteration_complete=%s, progress_made=%s, "
      "blocked_channels={%s} }",
      iteration_complete ? "true" : "false", progress_made ? "true" : "false",
      absl::StrJoin(blocked_channels, ", ", [](std::string* out, Channel* ch) {
        absl::StrAppend(out, ch->name());
      }));
}

std::ostream& operator<<(std::ostream& os,
                         const ProcInterpreter::RunResult& result) {
  os << result.ToString();
  return os;
}

}  // namespace xls
