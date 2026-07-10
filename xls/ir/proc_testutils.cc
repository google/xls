// Copyright 2024 The XLS Authors
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

#include "xls/ir/proc_testutils.h"

#include <algorithm>
#include <cstdint>
#include <iterator>
#include <optional>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "cppitertools/sorted.hpp"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/bits.h"
#include "xls/ir/channel.h"
#include "xls/ir/dfs_visitor.h"
#include "xls/ir/function_base.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/name_uniquer.h"
#include "xls/ir/node.h"
#include "xls/ir/node_util.h"
#include "xls/ir/nodes.h"
#include "xls/ir/package.h"
#include "xls/ir/proc.h"
#include "xls/ir/source_location.h"
#include "xls/ir/state_element.h"
#include "xls/ir/type.h"
#include "xls/ir/value.h"
#include "xls/ir/value_utils.h"
#include "xls/passes/dce_pass.h"
#include "xls/passes/inlining_pass.h"
#include "xls/passes/optimization_pass.h"
#include "xls/passes/pass_base.h"

namespace xls {

namespace {
// A struct which represents a single IO action, either a send or a receive.
struct IOAction {
  // Whether this action is a send or a receive.
  bool is_send;
  // The data that was sent or received.
  BValue data;
  // A boolean value which is whether the action is attempted (eg for a
  // send_if).
  BValue executed;
  // A boolean value which is whether the action completes, whether or not a
  // value is actually returned/sent. If false then the action is blocked due to
  // fifo full/empty.
  BValue completes;

  // Additional optional BValue that the input/output gen can fill in for use
  // wherever it is required.
  std::optional<BValue> context;
};

struct ActivationAction {
  // What activation we are on. The first activation is index 0.
  int64_t activation_index;
  // State vector after this activation. Values are only valid if
  // 'activation_complete' is true.
  absl::flat_hash_map<StateElement*, BValue> end_state;
  // Values which are sent out from this activation.
  absl::flat_hash_map<SendChannelRef, IOAction> sent_values;
  // Map from a receive node to the IO action describing the receive.
  absl::flat_hash_map<ReceiveChannelRef, IOAction> receives_finished;
  // Boolean value which is true if the activation completed and false if it
  // blocked on a send/recv.
  //
  // This is the and of all the IOAction.completes values.
  BValue activation_complete;
};

// A struct which represents the unrolled proc.
struct UnrolledProc {
  // State at the end of each activation.
  std::vector<ActivationAction> activations;
  // Initial state of the proc.
  absl::flat_hash_map<StateElement*, BValue> initial_state;
};

// A concept for a function that can generate inputs for a given receive.
//
// Take the receive node, the function builder, the argument set of the recieve
// and the current activation index, and the previous IOAction (if any). It
// returns an IOAction representing the receive and holding the resulting value,
// and whether or not it completed.
template <typename T>
concept InputGeneratorConcept = std::is_invocable_r_v<
    absl::StatusOr<IOAction>, T, /*receive=*/Receive*,
    /*fb=*/FunctionBuilder&, /*args=*/absl::Span<BValue const>,
    /*activation_index=*/int64_t,
    /*prev_action=*/std::optional<IOAction>, /*active=*/BValue>;

// A concept for a function that can consume the output of a send.
//
// Takes the send node, the function builder, the value to be sent, and the
// current activation index, and the previous IOAction (if any). It returns an
// IOAction representing the send and holding whether or not it completed. The
// returned IOAction also unconditionally holds the data that was sent.
template <typename T>
concept OutputConsumerConcept = std::is_invocable_r_v<
    absl::StatusOr<IOAction>, T, /*send=*/Send*, /*fb=*/FunctionBuilder&,
    /*args=*/absl::Span<BValue const>, /*activation_index=*/int64_t,
    /*prev_action=*/std::optional<IOAction>, /*active=*/BValue>;

// This visitor basically inlines one activation of the proc into a function
// collecting sent values (and optionally state elements) as return values.
// Receives are provided by function inputs.
template <typename InputGenerator, typename OutputConsumer>
  requires(InputGeneratorConcept<InputGenerator> &&
           OutputConsumerConcept<OutputConsumer>)
// TODO(allight): Need to track for each node whether its executed or not and
// pass that down to the consumer/generator.
class UnrollProcVisitor final : public DfsVisitorWithDefault {
 public:
  UnrollProcVisitor(FunctionBuilder& fb,
                    const ActivationAction& prev_activation, Value token_value,
                    InputGenerator& input_gen, OutputConsumer& output_consumer)
      : DfsVisitorWithDefault(),
        fb_(fb),
        prev_activation_(prev_activation),
        activation_(prev_activation.activation_index + 1),
        token_value_(std::move(token_value)),
        input_generator_(input_gen),
        output_consumer_(output_consumer) {}
  UnrollProcVisitor(const UnrollProcVisitor&) = delete;
  UnrollProcVisitor(UnrollProcVisitor&&) = delete;
  UnrollProcVisitor& operator=(const UnrollProcVisitor&) = delete;
  UnrollProcVisitor& operator=(UnrollProcVisitor&&) = delete;

  absl::StatusOr<ActivationAction> RunActivation(Proc* p) && {
    XLS_RETURN_IF_ERROR(fb_.GetError());
    XLS_RETURN_IF_ERROR(p->Accept(this));
    XLS_RETURN_IF_ERROR(fb_.GetError());
    XLS_ASSIGN_OR_RETURN((absl::flat_hash_map<StateElement*, BValue> end_state),
                         GetStateValuesAfterActivation(p));
    std::vector<BValue> ios_complete{prev_activation_.activation_complete};
    ios_complete.reserve(send_state_.size() + recv_state_.size());
    for (const auto& [_, io_action] : send_state_) {
      ios_complete.push_back(io_action.completes);
    }
    for (const auto& [_, io_action] : recv_state_) {
      ios_complete.push_back(io_action.completes);
    }
    return ActivationAction{
        .activation_index = activation_,
        .end_state = std::move(end_state),
        .sent_values = std::move(send_state_),
        .receives_finished = std::move(recv_state_),
        .activation_complete =
            ios_complete.size() == 1
                ? ios_complete.front()
                : fb_.And(ios_complete, SourceInfo(),
                          /*name=*/absl::StrCat("act_complete_", activation_)),
    };
  }

  absl::StatusOr<absl::flat_hash_map<StateElement*, BValue>>
  GetStateValuesAfterActivation(Proc* p) {
    absl::flat_hash_map<StateElement*, BValue> states;
    for (StateElement* state_element : p->StateElements()) {
      absl::Span<StateRead* const> reads =
          p->GetStateReadsByStateElement(state_element);
      XLS_RET_CHECK(!reads.empty()) << "No reads for " << state_element;

      BValue state_val;
      std::vector<BValue> cases;
      std::vector<BValue> selectors;
      for (Next* nxt : p->next_values(state_element)) {
        if (nxt->predicate()) {
          selectors.push_back(values_[nxt->predicate().value()]);
          XLS_RET_CHECK_EQ(nxt->predicate().value()->BitCountOrDie(), 1)
              << nxt->predicate().value()->ToString();
        }
        cases.push_back(values_[nxt->value()]);
      }
      if (selectors.empty()) {
        XLS_RET_CHECK_EQ(cases.size(), 1) << "no cases for " << state_element;
        state_val = cases.front();
      } else if (cases.front().GetType()->IsBits() &&
                 cases.front().GetType()->GetFlatBitCount() == 0) {
        // Special case to avoid creating non-trivial uses of zero-len bit
        // vectors.
        state_val = fb_.Literal(UBits(0, 0));
      } else {
        XLS_RET_CHECK_EQ(cases.size(), selectors.size());
        // materialize the next values into a select.
        // Need to reverse to keep the LSB is case 0 etc.
        absl::c_reverse(selectors);
        state_val =
            fb_.PrioritySelect(fb_.Concat(selectors), cases,
                               /*default_value=*/values_[reads.front()]);
      }
      states[state_element] = state_val;
    }
    return states;
  }

  BValue AndReduce(const absl::flat_hash_set<BValue>& values) {
    if (values.empty()) {
      return fb_.Literal(UBits(1, 1));
    }
    if (values.size() == 1) {
      return *values.begin();
    }
    return fb_.And(std::vector<BValue>(values.begin(), values.end()));
  }

  absl::Status DefaultHandler(Node* n) override {
    XLS_RETURN_IF_ERROR(fb_.GetError());
    absl::flat_hash_set<BValue> active;
    std::vector<Node*> new_ops;
    for (Node* old_op : n->operands()) {
      XLS_RET_CHECK(values_.contains(old_op)) << n << " @" << old_op;
      auto* new_op = values_[old_op].node();
      XLS_RET_CHECK(new_op != nullptr) << n << " @" << old_op;
      XLS_RET_CHECK(TypeHasToken(old_op->GetType()) ||
                    new_op->GetType()->IsEqualTo(old_op->GetType()))
          << "Type mismatch for " << old_op << " vs " << new_op << " in " << n;
      new_ops.push_back(new_op);
      active.insert(node_active_[old_op]);
    }
    XLS_ASSIGN_OR_RETURN(Node * new_node,
                         n->CloneInNewFunction(new_ops, fb_.function()));
    values_[n] = BValue(new_node, &fb_);
    node_active_[n] = AndReduce(active);
    return absl::OkStatus();
  }

  // TODO(allight): It might be interesting to expose trace values too as
  // IOActions.
  absl::Status HandleTrace(Trace* t) override {
    XLS_RETURN_IF_ERROR(fb_.GetError());
    absl::flat_hash_set<BValue> active;
    std::vector<Node*> new_ops{fb_.Literal(Value::Token()).node()};
    for (Node* v : t->operands().subspan(1)) {
      new_ops.push_back(values_[v].node());
      active.insert(node_active_[v]);
    }
    XLS_ASSIGN_OR_RETURN(Node * new_node,
                         t->CloneInNewFunction(new_ops, fb_.function()));
    values_[t] = BValue(new_node, &fb_);
    node_active_[t] = AndReduce(active);
    return absl::OkStatus();
  }

  // TODO(allight): It might be interesting to expose cover values too as
  // IOActions.
  absl::Status HandleCover(Cover* c) override {
    XLS_RETURN_IF_ERROR(DefaultHandler(c));
    Cover* new_cover = values_[c].node()->As<Cover>();
    new_cover->set_label(
        absl::StrFormat("%s_act%d_cover", c->GetName(), activation_));
    return absl::OkStatus();
  }

  // Asserts are kept around but the token argument is just dropped.
  // TODO(allight): It might be interesting to expose assert values too as
  // IOActions.
  absl::Status HandleAssert(Assert* n) override {
    XLS_RETURN_IF_ERROR(fb_.GetError());
    std::vector<Node*> new_ops{fb_.Literal(Value::Token()).node(),
                               values_[n->condition()].node()};
    XLS_ASSIGN_OR_RETURN(Node * new_node,
                         n->CloneInNewFunction(new_ops, fb_.function()));
    // Give the assert a new label.
    if (n->label()) {
      new_node->As<Assert>()->set_label(absl::StrFormat(
          "%s_act%d_assert", n->label().value_or(n->GetName()), activation_));
    }
    values_[n] = BValue(new_node, &fb_);
    BValue active = node_active_[n->condition()];
    node_active_[n] = active;
    return absl::OkStatus();
  }

  absl::Status HandleStateRead(StateRead* state_read) override {
    XLS_RETURN_IF_ERROR(fb_.GetError());
    absl::flat_hash_set<BValue> active;
    if (state_read->GetType()->IsToken()) {
      values_[state_read] = fb_.Literal(token_value_);
      node_active_[state_read] = prev_activation_.activation_complete;
      return absl::OkStatus();
    }
    if (!state_read->predicate()) {
      values_[state_read] =
          prev_activation_.end_state.at(state_read->state_element());
      node_active_[state_read] = prev_activation_.activation_complete;
    } else {
      BValue predicate_value = values_[state_read->predicate().value()];
      values_[state_read] =
          fb_.Select(predicate_value,
                     prev_activation_.end_state.at(state_read->state_element()),
                     fb_.Literal(ZeroOfType(state_read->GetType())));
      node_active_[state_read] =
          fb_.And({prev_activation_.activation_complete, predicate_value,
                   node_active_[state_read->predicate().value()]});
    }
    return absl::OkStatus();
  }

  absl::Status HandleSend(Send* s) override {
    XLS_RETURN_IF_ERROR(fb_.GetError());
    // The value of the send is always just a new token.
    values_[s] = fb_.Literal(token_value_);
    XLS_ASSIGN_OR_RETURN(SendChannelRef send_channel_ref,
                         s->GetSendChannelRef());
    std::optional<IOAction> prev_action =
        send_state_.contains(send_channel_ref)
            ? std::make_optional(send_state_.at(send_channel_ref))
            : std::nullopt;
    BValue token = values_[s->token()];
    BValue data = values_[s->data()];
    BValue active =
        AndReduce({node_active_[s->token()], node_active_[s->data()]});
    IOAction action;
    if (s->predicate()) {
      BValue predicate_value = values_[s->predicate().value()];
      active = fb_.And({active, node_active_[s->predicate().value()]});
      XLS_ASSIGN_OR_RETURN(
          action, output_consumer_(s, fb_, {token, data, predicate_value},
                                   activation_, prev_action, active));
    } else {
      XLS_ASSIGN_OR_RETURN(action,
                           output_consumer_(s, fb_, {token, data}, activation_,
                                            prev_action, active));
    }
    send_state_[send_channel_ref] = std::move(action);
    node_active_[s] = fb_.And(action.completes, active);
    return absl::OkStatus();
  }

  absl::Status HandleNext(Next* n) override {
    // This is handled separately
    XLS_RETURN_IF_ERROR(fb_.GetError());
    return absl::OkStatus();
  }

  absl::Status HandleReceive(Receive* r) override {
    XLS_RETURN_IF_ERROR(fb_.GetError());
    BValue real_data;
    XLS_ASSIGN_OR_RETURN(ReceiveChannelRef recv_channel_ref,
                         r->GetReceiveChannelRef());
    std::optional<IOAction> prev_action =
        recv_state_.contains(recv_channel_ref)
            ? std::make_optional(recv_state_.at(recv_channel_ref))
            : std::nullopt;
    BValue token = values_[r->token()];
    BValue active = node_active_[r->token()];
    IOAction action;
    if (r->predicate()) {
      active = fb_.And({active, node_active_[r->predicate().value()]});
      BValue predicate_value = values_[r->predicate().value()];
      XLS_ASSIGN_OR_RETURN(action,
                           input_generator_(r, fb_, {token, predicate_value},
                                            activation_, prev_action, active));
    } else {
      XLS_ASSIGN_OR_RETURN(
          action,
          input_generator_(r, fb_, {token}, activation_, prev_action, active));
    }
    std::vector<BValue> result_values{fb_.Literal(token_value_), action.data};
    if (!r->is_blocking()) {
      result_values.push_back(action.executed);
    }
    recv_state_[recv_channel_ref] = std::move(action);
    VLOG(2) << "got " << r << " -> " << recv_state_[recv_channel_ref].data;
    values_[r] = fb_.Tuple(std::move(result_values));
    node_active_[r] = fb_.And({active, action.completes});

    return absl::OkStatus();
  }

  absl::Status HandleAfterAll(AfterAll* aa) override {
    XLS_RETURN_IF_ERROR(fb_.GetError());
    // TODO: https://github.com/google/xls/issues/1375 - It would be nice to
    // record this for real. The issue is that we'd need to figure out some way
    // to flatten the tree in a consistent way.
    values_[aa] = fb_.Literal(token_value_);
    absl::flat_hash_set<BValue> active;
    for (Node* operand : aa->operands()) {
      active.insert(node_active_[operand]);
    }
    node_active_[aa] = AndReduce(active);
    return absl::OkStatus();
  }

  // Get the sent value tuple (or nullopt if no value was sent)
  std::vector<IOAction> GetSentValues() const {
    if (send_state_.empty()) {
      return {};
    }
    std::vector<std::pair<SendChannelRef, IOAction>> results(
        send_state_.begin(), send_state_.end());
    absl::c_sort(results,
                 [](const std::pair<SendChannelRef, BValue>& lhs,
                    const std::pair<SendChannelRef, BValue>& rhs) -> bool {
                   return ChannelRefName(lhs.first) < ChannelRefName(rhs.first);
                 });
    std::vector<IOAction> ordered;
    absl::c_transform(results, std::back_inserter(ordered),
                      [](auto& v) { return v.second; });
    return ordered;
  }

 private:
  // The function we are building to do verification on.
  FunctionBuilder& fb_;
  // The previous activation.
  const ActivationAction& prev_activation_;
  // A map of each nodes on a particular activation to the node that implements
  // the same operation in the test function.
  absl::flat_hash_map<Node*, BValue> values_;
  // Map to a bool bvalue which denotes if the translated node (values_[node])
  // is still active or if the function has blocked earlier.
  absl::flat_hash_map<Node*, BValue> node_active_;
  // A map of channel names to values sent on the most recent activation.
  absl::flat_hash_map<SendChannelRef, IOAction> send_state_;
  // A map of channel names to values received on the most recent activation.
  absl::flat_hash_map<ReceiveChannelRef, IOAction> recv_state_;
  // Which activation are we inlining.
  int64_t activation_;
  // What value should we use for a token.
  Value token_value_;
  // The input generator to use for this activation.
  InputGenerator& input_generator_;
  // The output consumer to use for this activation.
  OutputConsumer& output_consumer_;
};

// Unrolls the proc 'activation_count' times, returning the unrolled function
// data.
//
// InputGenerator is a function that takes a Receive node, the function
// builder, the current activation index, and the current token value, and
// returns the value to be used as the input for that receive.
template <typename InputGen, typename OutputConsumer>
  requires(InputGeneratorConcept<InputGen> &&
           OutputConsumerConcept<OutputConsumer>)
absl::StatusOr<UnrolledProc> UnrollProcToFunctionInternal(
    Proc* p, FunctionBuilder& fb, int64_t activation_count,
    const Value& token_value, InputGen& input_gen,
    OutputConsumer& output_consumer) {
  XLS_RET_CHECK_GT(activation_count, 0)
      << "At least one activation is required.";
  UnrolledProc result;
  for (StateElement* state_element : p->StateElements()) {
    result.initial_state[state_element] =
        fb.Literal(state_element->initial_value());
  }
  ActivationAction initial_state{
      .activation_index = -1,
      .end_state = result.initial_state,
      .activation_complete = fb.Literal(UBits(1, 1)),
  };
  for (int64_t i = 0; i < activation_count; ++i) {
    // Fixup and collect state elements
    UnrollProcVisitor<InputGen, OutputConsumer> upv(
        fb, i == 0 ? initial_state : result.activations.back(), token_value,
        input_gen, output_consumer);
    XLS_ASSIGN_OR_RETURN(ActivationAction action,
                         std::move(upv).RunActivation(p));
    result.activations.push_back(std::move(action));
  }
  return result;
}

class InlineSpecificPass : public InliningPass {
 public:
  explicit InlineSpecificPass(Function* to_inline)
      : InliningPass(InliningPass::InlineDepth::kFull, "inline_specific",
                     "Inline specific function"),
        to_inline_(to_inline) {}

 protected:
  bool ShouldInlineInto(FunctionBase* caller) const override {
    return caller == to_inline_;
  }

 private:
  Function* to_inline_;
};

absl::Status CleanupFunction(Function* f) {
  // Clean up the function a bit. The tracking values might be dangling so we
  // might as well get rid of them to avoid making z3 scan through them.
  OptimizationFixedPointCompoundPass pass("cleanup", "UnrollProc cleanup");
  pass.Add<InlineSpecificPass>(f);
  pass.Add<DeadCodeEliminationPass>();
  DeadCodeEliminationPass dce;
  PassResults pass_results;
  OptimizationContext context;
  return pass.Run(f->package(), {}, &pass_results, context).status();
}

}  // namespace

absl::StatusOr<Function*> UnrollProcToFunction(Proc* p,
                                               int64_t activation_count,
                                               bool include_state,
                                               const Value& token_value) {
  XLS_RET_CHECK_GT(activation_count, 0)
      << "At least one activation is required.";
  if (include_state) {
    XLS_RET_CHECK(
        !p->StateElements().empty() ||
        absl::c_any_of(p->nodes(), [](Node* n) { return n->Is<Send>(); }))
        << "No state or sends means returned function would return a single "
           "constant value";
  } else {
    XLS_RET_CHECK(
        absl::c_any_of(p->nodes(), [](Node* n) { return n->Is<Send>(); }))
        << "No sends means returned function would return a single constant "
           "value";
  }
  Package* pkg = p->package();
  FunctionBuilder fb(
      absl::StrFormat("%s_x%d_function", p->name(), activation_count), pkg);
  absl::flat_hash_map<std::pair<ReceiveChannelRef, int64_t>, BValue>
      recvd_value;
  auto recv_gen = [&recvd_value](Receive* r, FunctionBuilder& fb,
                                 absl::Span<BValue const> args, int64_t act_idx,
                                 std::optional<IOAction> prev_action,
                                 BValue active) -> absl::StatusOr<IOAction> {
    BValue does_execute;
    if (r->predicate() && r->is_blocking()) {
      // The receive executes if the predicate is true.
      does_execute = args.back();
    } else if (r->predicate() && !r->is_blocking()) {
      // The receive executes if the predicate is true and we actually
      // send something.
      does_execute =
          fb.And(args.back(),
                 fb.Param(NodeNameFormat("recv_executed_%d_%s", act_idx, r),
                          r->GetPayloadType()));
    } else if (!r->predicate() && !r->is_blocking()) {
      does_execute = fb.Param(NodeNameFormat("recv_executed_%d_%s", act_idx, r),
                              fb.package()->GetBitsType(1));
    } else {
      does_execute = fb.Literal(UBits(1, 1));
    }
    XLS_ASSIGN_OR_RETURN(ReceiveChannelRef recv_channel_ref,
                         r->GetReceiveChannelRef());
    BValue recv_data;
    if (recvd_value.contains({recv_channel_ref, act_idx})) {
      recv_data = recvd_value.at({recv_channel_ref, act_idx});
    } else {
      recv_data = fb.Param(NodeNameFormat("recv_data_act%d_%s", act_idx,
                                          ChannelRefName(recv_channel_ref)),
                           r->GetPayloadType());
      recvd_value[{recv_channel_ref, act_idx}] = recv_data;
    }
    if (r->predicate()) {
      recv_data =
          fb.Select(does_execute, recv_data,
                    fb.Literal(ZeroOfType(recv_data.node()->GetType())));
    }
    return IOAction{
        .is_send = false,
        .data = recv_data,
        .executed = does_execute,
        .completes = active,
        .context = std::nullopt,
    };
  };
  auto send_gen = [](Send* s, FunctionBuilder& fb,
                     absl::Span<BValue const> args, int64_t act_idx,
                     std::optional<IOAction> prev_action,
                     BValue active) -> absl::StatusOr<IOAction> {
    return IOAction{
        .is_send = true,
        .data = args[Send::kDataOperand],
        .executed = s->predicate() ? fb.And(active, args.back()) : active,
        .completes = fb.Literal(UBits(1, 1)),
        .context = std::nullopt,
    };
  };
  XLS_ASSIGN_OR_RETURN(
      UnrolledProc unrolled,
      UnrollProcToFunctionInternal(p, fb, activation_count, token_value,
                                   /*input_gen=*/recv_gen,
                                   /*output_consumer=*/send_gen));
  std::vector<BValue> return_values;
  for (const ActivationAction& action : unrolled.activations) {
    std::vector<BValue> act_values;
    if (include_state) {
      std::vector<BValue> state;
      for (const StateElement* element :
           iter::sorted(p->StateElements(),
                        [](const StateElement* a, const StateElement* b) {
                          return a->name() < b->name();
                        })) {
        state.push_back(action.end_state.at(element));
      }
      act_values.push_back(fb.Tuple(state));
    }
    std::vector<BValue> chan;
    chan.reserve(action.sent_values.size());
    for (const auto& [st, send] :
         iter::sorted(action.sent_values, [](const auto& a, const auto& b) {
           return ChannelRefName(a.first) < ChannelRefName(b.first);
         })) {
      chan.push_back(fb.Tuple(
          {send.executed,
           fb.Select(send.executed, send.data,
                     fb.Literal(ZeroOfType(send.data.node()->GetType())))}));
    }
    act_values.push_back(fb.Tuple(chan));
    if (act_values.size() == 1) {
      return_values.push_back(act_values.front());
    } else {
      return_values.push_back(fb.Tuple(act_values));
    }
  }
  XLS_ASSIGN_OR_RETURN(Function * result,
                       fb.BuildWithReturnValue(fb.Tuple(return_values)));
  XLS_RETURN_IF_ERROR(CleanupFunction(result));
  VLOG(2) << "Proc: \n" << p->DumpIr() << "To Func: \n" << result->DumpIr();
  return result;
}

namespace {

// Returns new state, new fill_amnt, whether the fifo push occurred, and if the
// fifo had space for another element.
// fn push<N: u32>(st: TY[N], data: TY, fill_amnt: u8, enable: bool) -> (TY[N],
// u8, bool, bool)
absl::StatusOr<Function*> CreateFifoPush(Package* p, NameUniquer& name_uniquer,
                                         Type* type, int64_t fifo_depth,
                                         std::string_view name_base) {
  FunctionBuilder fb(
      name_uniquer.GetSanitizedUniqueName(absl::StrCat(name_base, "_push")), p);
#if 0
  not_full = fill_amnt < fifo_depth;
  if (enable && not_full) {
    st[fill_amnt] = data;
    fill_amnt += 1;
    occurred = true;
  } else {
    occurred = false;
  }
#endif
  BValue state = fb.Param("st", p->GetArrayType(fifo_depth, type));
  BValue data = fb.Param("data", type);
  BValue fill_amnt = fb.Param("fill_amnt", p->GetBitsType(8));
  BValue enable = fb.Param("enable", p->GetBitsType(1));
  BValue full_depth = fb.Literal(UBits(fifo_depth, 8));

  BValue not_full = fb.ULt(fill_amnt, full_depth);
  BValue can_push = fb.And(enable, not_full);
  fb.Select(can_push,
            fb.Tuple({fb.ArrayUpdate(state, data, {fill_amnt}),
                      fb.Add(fill_amnt, fb.Literal(UBits(1, 8)), SourceInfo(),
                             absl::StrCat(name_base, "_push_next_fill_amnt")),
                      fb.Literal(UBits(1, 1)), fb.Literal(UBits(1, 1))}),
            fb.Tuple({state, fill_amnt, fb.Literal(UBits(0, 1)), not_full}));
  return fb.Build();
}
// Returns value, new state, new fill_amnt, whether the fifo pop occurred and if
// the fifo was non-empty before starting.
// fn push<N: u32>(st: TY[N], fill_amnt: u8, enable: bool) -> (TY, u8,
// bool, bool)
absl::StatusOr<Function*> CreateFifoPop(Package* pkg, NameUniquer& name_uniquer,
                                        Type* type, int64_t fifo_depth,
                                        std::string_view name_base) {
  FunctionBuilder fb(
      name_uniquer.GetSanitizedUniqueName(absl::StrCat(name_base, "_pop")),
      pkg);

#if 0
  non_empty = fill_amnt < fifo_depth;
  if (enable && non_empty) {
    value = st[fill_amnt];
    fill_amnt += 1;
    occurred = true;
  } else {
    occurred = false;
  }
#endif
  BValue state = fb.Param("st", pkg->GetArrayType(fifo_depth, type));
  BValue fill_amnt = fb.Param("fill_amnt", pkg->GetBitsType(8));
  BValue enable = fb.Param("enable", pkg->GetBitsType(1));
  BValue full_depth = fb.Literal(UBits(fifo_depth, 8));
  BValue not_empty = fb.ULt(fill_amnt, full_depth, SourceInfo(),
                            absl::StrCat(name_base, "_pop_not_empty"));
  BValue can_pop = fb.And({enable, not_empty}, SourceInfo(),
                          absl::StrCat(name_base, "_pop_can_pop"));
  fb.Select(can_pop,
            fb.Tuple({fb.ArrayIndex(state, {fill_amnt}),
                      fb.Add(fill_amnt, fb.Literal(UBits(1, 8)), SourceInfo(),
                             absl::StrCat(name_base, "_pop_next_fill_amnt")),
                      fb.Literal(UBits(1, 1)), fb.Literal(UBits(1, 1))}),
            fb.Tuple({fb.Literal(ZeroOfType(type)), fill_amnt,
                      fb.Literal(UBits(0, 1)), not_empty}),
            SourceInfo(), absl::StrCat(name_base, "_pop_result"));
  return fb.Build();
}
class UntimedInputGen {
 public:
  // available_values is a map of channel to the values that are available (as
  // an xls array value) be received on that channel.
  explicit UntimedInputGen(
      const absl::flat_hash_map<ReceiveChannelRef, BValue>& available_values,
      const absl::flat_hash_map<ReceiveChannelRef, Function*>& fifo_pops)
      : available_values_(std::move(available_values)), fifo_pops_(fifo_pops) {}

  // Map of
  absl::flat_hash_map<ReceiveChannelRef, BValue> final_recv_count() const {
    absl::flat_hash_map<ReceiveChannelRef, BValue> out;
    int64_t max_cycle = 0;
    for (const auto& [key, value] : recv_count_) {
      max_cycle = std::max(max_cycle, key.second);
    }
    for (const auto& [key, value] : recv_count_) {
      if (key.second == max_cycle) {
        out[key.first] = value;
      }
    }
    return out;
  }

  absl::StatusOr<IOAction> operator()(Receive* r, FunctionBuilder& fb,
                                      absl::Span<BValue const> args,
                                      int64_t act_idx,
                                      std::optional<IOAction> prev_action,
                                      BValue active) {
    XLS_ASSIGN_OR_RETURN(ReceiveChannelRef recv_chan_ref,
                         r->GetReceiveChannelRef());
    XLS_RET_CHECK(act_idx == 0 ||
                  recv_count_.contains({recv_chan_ref, act_idx - 1}))
        << " act_idx: " << act_idx;
    XLS_RET_CHECK(fifo_pops_.contains(recv_chan_ref));
    BValue prev_recv_count = act_idx != 0
                                 ? recv_count_.at({recv_chan_ref, act_idx - 1})
                                 : fb.Literal(UBits(0, 8));
    BValue enable_signal =
        r->predicate() ? fb.And(active, args.back()) : active;
    BValue pop_result = fb.Invoke(
        {available_values_.at(recv_chan_ref), prev_recv_count, enable_signal},
        fifo_pops_.at(recv_chan_ref));
    BValue recvd_value =
        fb.TupleIndex(pop_result, 0, SourceInfo(),
                      absl::StrFormat("recv_data_%s_act%d",
                                      ChannelRefName(recv_chan_ref), act_idx));
    BValue next_count =
        fb.TupleIndex(pop_result, 1, SourceInfo(),
                      absl::StrFormat("next_count_%s_act%d",
                                      ChannelRefName(recv_chan_ref), act_idx));
    BValue did_pop =
        fb.TupleIndex(pop_result, 2, SourceInfo(),
                      absl::StrFormat("did_pop_%s_act%d",
                                      ChannelRefName(recv_chan_ref), act_idx));
    BValue was_not_empty =
        fb.TupleIndex(pop_result, 3, SourceInfo(),
                      absl::StrFormat("was_not_empty_%s_act%d",
                                      ChannelRefName(recv_chan_ref), act_idx));
    recv_count_[{recv_chan_ref, act_idx}] = next_count;
    return IOAction{
        .is_send = false,
        .data = recvd_value,
        .executed = did_pop,
        .completes = r->is_blocking() ? fb.And(active, was_not_empty) : active,
        .context = std::nullopt,
    };
  }

 private:
  const absl::flat_hash_map<ReceiveChannelRef, BValue>& available_values_;
  absl::flat_hash_map<std::pair<ReceiveChannelRef, int64_t>, BValue>
      recv_count_;
  const absl::flat_hash_map<ReceiveChannelRef, Function*>& fifo_pops_;
};

class UntimedOutputConsumer {
 public:
  explicit UntimedOutputConsumer(
      const absl::flat_hash_map<SendChannelRef, Function*>& fifo_pushs)
      : fifo_pushes_(fifo_pushs) {}

  absl::StatusOr<IOAction> operator()(Send* s, FunctionBuilder& fb,
                                      absl::Span<BValue const> args,
                                      int64_t act_idx,
                                      std::optional<IOAction> prev_action,
                                      BValue active) {
    XLS_ASSIGN_OR_RETURN(SendChannelRef send_chan_ref, s->GetSendChannelRef());
    XLS_RET_CHECK(act_idx == 0 ||
                  send_count_.contains({send_chan_ref, act_idx - 1}))
        << " act_idx: " << act_idx;
    XLS_RET_CHECK(fifo_pushes_.contains(send_chan_ref));
    BValue send_count = act_idx != 0
                            ? send_count_.at({send_chan_ref, act_idx - 1})
                            : fb.Literal(UBits(0, 8));
    BValue enable_signal =
        s->predicate() ? fb.And(args.back(), active) : active;
    BValue send_val = args[Send::kDataOperand];
    BValue prev_state =
        act_idx != 0 ? send_output_fifo_.at({send_chan_ref, act_idx - 1})
                     : fb.Literal(ZeroOfType(fifo_pushes_.at(send_chan_ref)
                                                 ->GetType()
                                                 ->parameter_type(0)),
                                  SourceInfo(),
                                  absl::StrFormat("zero_send_data_%s_act%d",
                                                  ChannelRefName(send_chan_ref),
                                                  act_idx));
    BValue push_result =
        fb.Invoke({prev_state, send_val, send_count, enable_signal},
                  fifo_pushes_.at(send_chan_ref));
    BValue next_data =
        fb.TupleIndex(push_result, 0, SourceInfo(),
                      absl::StrFormat("sent_data_%s_act%d",
                                      ChannelRefName(send_chan_ref), act_idx));
    BValue next_count =
        fb.TupleIndex(push_result, 1, SourceInfo(),
                      absl::StrFormat("next_count_%s_act%d",
                                      ChannelRefName(send_chan_ref), act_idx));
    BValue did_push =
        fb.TupleIndex(push_result, 2, SourceInfo(),
                      absl::StrFormat("did_push_%s_act%d",
                                      ChannelRefName(send_chan_ref), act_idx));
    BValue could_push =
        fb.TupleIndex(push_result, 3, SourceInfo(),
                      absl::StrFormat("could_push_%s_act%d",
                                      ChannelRefName(send_chan_ref), act_idx));
    send_count_[{send_chan_ref, act_idx}] = next_count;
    send_output_fifo_[{send_chan_ref, act_idx}] = next_data;
    return IOAction{
        .is_send = true,
        .data = send_val,
        .executed = did_push,
        .completes = fb.And(active, could_push),
        .context = next_data,
    };
  }

  BValue send_count(SendChannelRef send_channel_ref, int64_t act_idx) const {
    return send_count_.at({send_channel_ref, act_idx});
  }

 private:
  absl::flat_hash_map<std::pair<SendChannelRef, int64_t>, BValue> send_count_;
  absl::flat_hash_map<std::pair<SendChannelRef, int64_t>, BValue>
      send_output_fifo_;
  const absl::flat_hash_map<SendChannelRef, Function*>& fifo_pushes_;
};

}  // namespace
absl::StatusOr<Function*> UnrollProcToUntimedFunction(
    Proc* p, int64_t activation_count, int64_t input_value_count,
    int64_t output_value_count, bool count_recvs) {
  Package* pkg = p->package();
  NameUniquer name_uniquer = pkg->NameUniquerForPackage();
  FunctionBuilder fb(
      name_uniquer.GetSanitizedUniqueName(absl::StrFormat(
          "%s_x%d_untimed_function", p->name(), activation_count)),
      pkg);
  // We need to be careful to create the params in a predictable order.
  absl::flat_hash_map<ReceiveChannelRef, BValue> available_values;
  absl::flat_hash_map<ReceiveChannelRef, Type*> available_recv_types;
  absl::flat_hash_map<ReceiveChannelRef, Function*> fifo_pops;
  absl::flat_hash_map<SendChannelRef, Type*> available_send_types;
  absl::flat_hash_map<SendChannelRef, Function*> fifo_pushes;
  for (Node* n : p->nodes()) {
    if (n->Is<Receive>()) {
      XLS_ASSIGN_OR_RETURN(ReceiveChannelRef recv_channel_ref,
                           n->As<Receive>()->GetReceiveChannelRef());
      if (available_recv_types.contains(recv_channel_ref)) {
        XLS_RET_CHECK_EQ(available_recv_types[recv_channel_ref],
                         n->As<Receive>()->GetPayloadType());
        continue;
      }
      available_recv_types[recv_channel_ref] = pkg->GetArrayType(
          input_value_count, n->As<Receive>()->GetPayloadType());
      XLS_ASSIGN_OR_RETURN(
          fifo_pops[recv_channel_ref],
          CreateFifoPop(pkg, name_uniquer, n->As<Receive>()->GetPayloadType(),
                        input_value_count, ChannelRefName(recv_channel_ref)));
    }
    if (n->Is<Send>()) {
      XLS_ASSIGN_OR_RETURN(SendChannelRef send_channel_ref,
                           n->As<Send>()->GetSendChannelRef());
      if (available_send_types.contains(send_channel_ref)) {
        XLS_RET_CHECK_EQ(available_send_types[send_channel_ref],
                         n->As<Send>()->GetPayloadType());
        continue;
      }
      available_send_types[send_channel_ref] = pkg->GetArrayType(
          output_value_count, n->As<Send>()->GetPayloadType());
      XLS_ASSIGN_OR_RETURN(
          fifo_pushes[send_channel_ref],
          CreateFifoPush(pkg, name_uniquer, n->As<Send>()->GetPayloadType(),
                         output_value_count, ChannelRefName(send_channel_ref)));
    }
  }
  // Need to ensure inputs are ordered consistently
  for (const auto& [recv_channel_ref, type] :
       iter::sorted(available_recv_types, [](const auto& a, const auto& b) {
         return ChannelRefName(a.first) < ChannelRefName(b.first);
       })) {
    available_values[recv_channel_ref] = fb.Param(
        absl::StrFormat("available_%s", ChannelRefName(recv_channel_ref)),
        type);
  }
  UntimedInputGen input_gen(available_values, fifo_pops);
  UntimedOutputConsumer output_consumer(fifo_pushes);
  XLS_ASSIGN_OR_RETURN(
      UnrolledProc unrolled,
      UnrollProcToFunctionInternal(p, fb, activation_count,
                                   Value::Tuple({Value(UBits(0xdeadbeef, 32))}),
                                   input_gen, output_consumer));
  // Output is ((input_a_cnt, input_b_cnt, ...), ((chan_a_array, send_cnt),
  // (chan_b_array, send_cnt), ...)) Channels out sorted by name too.
  std::vector<BValue> results;
  for (const auto& [send_channel_ref, value] :
       iter::sorted(unrolled.activations.back().sent_values,
                    [](const auto& a, const auto& b) {
                      return ChannelRefName(a.first) < ChannelRefName(b.first);
                    })) {
    XLS_RET_CHECK(value.context.has_value())
        << "Send " << ChannelRefName(send_channel_ref) << " has no context";
    results.push_back(fb.Tuple(
        {value.context.value(),
         output_consumer.send_count(send_channel_ref, activation_count - 1)}));
  }
  BValue ret_value;
  if (count_recvs) {
    std::vector<BValue> recv_cnts;
    for (const auto& [recv_channel_ref, value] : iter::sorted(
             input_gen.final_recv_count(), [](const auto& a, const auto& b) {
               return ChannelRefName(a.first) < ChannelRefName(b.first);
             })) {
      recv_cnts.push_back(value);
    }
    ret_value = fb.Tuple({fb.Tuple(recv_cnts), fb.Tuple(results)});
  } else {
    ret_value = fb.Tuple(results);
  }
  XLS_ASSIGN_OR_RETURN(Function * result, fb.BuildWithReturnValue(ret_value));
  XLS_RETURN_IF_ERROR(CleanupFunction(result));
  VLOG(2) << "Proc: \n" << p->DumpIr() << "To Func: \n" << result->DumpIr();
  return result;
}

}  // namespace xls
