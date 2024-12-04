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

#include <cstdint>
#include <iterator>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/bits.h"
#include "xls/ir/dfs_visitor.h"
#include "xls/ir/function_base.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/node.h"
#include "xls/ir/nodes.h"
#include "xls/ir/package.h"
#include "xls/ir/proc.h"
#include "xls/ir/source_location.h"
#include "xls/ir/state_element.h"
#include "xls/ir/value.h"
#include "xls/ir/value_utils.h"
#include "xls/passes/dce_pass.h"
#include "xls/passes/pass_base.h"

namespace xls {

namespace {
struct NodeActivation {
  Node* node;
  int64_t activation;

  bool operator==(const NodeActivation& o) const {
    return node == o.node && activation == o.activation;
  }
  template <typename H>
  friend H AbslHashValue(H h, const NodeActivation& c) {
    return H::combine(std::move(h), c.node, c.activation);
  }
};

// This visitor basically inlines one activation of the proc into a function
// collecting sent values (and optionally state elements) as return values.
// Receives are provided by function inputs.
class UnrollProcVisitor final : public DfsVisitorWithDefault {
 public:
  UnrollProcVisitor(FunctionBuilder& fb,
                    absl::flat_hash_map<NodeActivation, BValue>& values,
                    int64_t activation, Value token_value)
      : DfsVisitorWithDefault(),
        fb_(fb),
        values_(values),
        activation_(activation),
        token_value_(std::move(token_value)) {}

  absl::Status DefaultHandler(Node* n) override {
    XLS_RETURN_IF_ERROR(fb_.GetError());
    std::vector<Node*> new_ops;
    for (Node* old_op : n->operands()) {
      XLS_RET_CHECK(values_.contains({old_op, activation_}))
          << n << " @" << old_op;
      auto* old_node = values_[{old_op, activation_}].node();
      XLS_RET_CHECK(old_node != nullptr) << n << " @" << old_op;
      new_ops.push_back(old_node);
    }
    XLS_ASSIGN_OR_RETURN(Node * new_node,
                         n->CloneInNewFunction(new_ops, fb_.function()));
    values_[{n, activation_}] = BValue(new_node, &fb_);
    return absl::OkStatus();
  }

  absl::Status HandleStateRead(StateRead* state_read) override {
    XLS_RETURN_IF_ERROR(fb_.GetError());
    if (state_read->GetType()->IsToken()) {
      values_[{state_read, activation_}] = fb_.Literal(token_value_);
      return absl::OkStatus();
    }
    XLS_RET_CHECK(values_.contains({state_read, activation_}))
        << "State value not created for activation " << activation_ << ": "
        << state_read;
    return absl::OkStatus();
  }

  absl::Status HandleSend(Send* s) override {
    XLS_RETURN_IF_ERROR(fb_.GetError());
    values_[{s, activation_}] = fb_.Literal(token_value_);
    BValue predicate_value;
    BValue data;
    if (s->predicate()) {
      predicate_value = values_[{s->predicate().value(), activation_}];
      data = fb_.Select(predicate_value,
                        {fb_.Literal(ZeroOfType(s->data()->GetType())),
                         values_[{s->data(), activation_}]});
    } else {
      predicate_value = fb_.Literal(UBits(1, 1));
      data = values_[{s->data(), activation_}];
    }
    send_state_[s->channel_name()] = fb_.Tuple({predicate_value, data});
    return absl::OkStatus();
  }

  absl::Status HandleNext(Next* n) override {
    XLS_RETURN_IF_ERROR(fb_.GetError());
    return absl::OkStatus();
  }

  absl::Status HandleReceive(Receive* r) override {
    XLS_RETURN_IF_ERROR(fb_.GetError());
    BValue real_data;
    if (recv_state_.contains({r->channel_name(), activation_})) {
      real_data = recv_state_.at({r->channel_name(), activation_});
    } else {
      real_data = fb_.Param(
          absl::StrFormat("%s_act%d_read", r->channel_name(), activation_),
          r->GetPayloadType());
      recv_state_[{r->channel_name(), activation_}] = real_data;
    }
    std::vector<BValue> result_values{fb_.Literal(token_value_)};
    if (r->predicate()) {
      result_values.push_back(fb_.Select(
          values_[{r->predicate().value(), activation_}],
          {fb_.Literal(ZeroOfType(r->GetPayloadType())), real_data}));
    } else {
      result_values.push_back(real_data);
    }
    if (!r->is_blocking()) {
      // valid is an input.
      result_values.push_back(
          fb_.Param(absl::StrFormat("%s_act%d_read_valid", r->channel_name(),
                                    activation_),
                    fb_.package()->GetBitsType(1)));
    }
    values_[{r, activation_}] = fb_.Tuple(std::move(result_values));
    VLOG(2) << "got " << r << " -> " << values_[{r, activation_}];
    return absl::OkStatus();
  }

  absl::Status HandleAssert(Assert* a) override {
    return absl::UnimplementedError(
        "UnrollProcVisitor: assert is not supported");
  }
  absl::Status HandleCover(Cover* c) override {
    return absl::UnimplementedError(
        "UnrollProcVisitor: cover is not supported");
  }

  absl::Status HandleAfterAll(AfterAll* aa) override {
    XLS_RETURN_IF_ERROR(fb_.GetError());
    // TODO: https://github.com/google/xls/issues/1375 - It would be nice to
    // record this for real. The issue is that we'd need to figure out some way
    // to flatten the tree in a consistent way.
    values_[{aa, activation_}] = fb_.Literal(token_value_);
    return absl::OkStatus();
  }

  // Get the sent value tuple (or nullopt if no value was sent)
  std::optional<BValue> GetSentValues() const {
    if (send_state_.empty()) {
      return std::nullopt;
    }
    std::vector<std::pair<std::string, BValue>> results(send_state_.begin(),
                                                        send_state_.end());
    absl::c_sort(results,
                 [](const std::pair<std::string, BValue>& lhs,
                    const std::pair<std::string, BValue>& rhs) -> bool {
                   return lhs.first < rhs.first;
                 });
    std::vector<BValue> ordered;
    absl::c_transform(results, std::back_inserter(ordered),
                      [](auto& v) { return v.second; });
    return fb_.Tuple(
        ordered, SourceInfo(),
        absl::StrFormat("send_values_for_activation_%d", activation_));
  }

 private:
  // The function we are building to do verification on.
  FunctionBuilder& fb_;
  // A map of each nodes on a particular activation to the node that implements
  // the same operation in the test function.
  absl::flat_hash_map<NodeActivation, BValue>& values_;
  // A map of channel names to values sent on the most recent activation.
  absl::flat_hash_map<std::string, BValue> send_state_;
  // A map of channel names & activation to the values received on that
  // activation.
  absl::flat_hash_map<std::pair<std::string, int64_t>, BValue> recv_state_;
  // Which activation are we inlining.
  int64_t activation_;
  // What value should we use for a token.
  Value token_value_;
};
// Pull out all the state values before the given activation has started. Also
// update 'values' to include these in the map.
absl::StatusOr<std::vector<BValue>> GetStateValuesBeforeActivation(
    Proc* p, int64_t activation, FunctionBuilder& fb,
    absl::flat_hash_map<NodeActivation, BValue>& values) {
  std::vector<BValue> states;
  for (StateElement* state_element : p->StateElements()) {
    StateRead* state_read = p->GetStateRead(state_element);
    if (activation == 0) {
      values[{state_read, 0}] =
          fb.Literal(state_element->initial_value(), SourceInfo(),
                     absl::StrFormat("%s_initial_value", p->name()));
    } else {
      std::vector<BValue> cases;
      std::vector<BValue> selectors;
      for (Next* nxt : p->next_values(state_read)) {
        if (nxt->predicate()) {
          selectors.push_back(
              values[{nxt->predicate().value(), activation - 1}]);
        }
        cases.push_back(values[{nxt->value(), activation - 1}]);
      }
      if (selectors.empty()) {
        XLS_RET_CHECK_EQ(cases.size(), 1) << "no cases for " << state_element;
        values[{state_read, activation}] = cases.front();
      } else if (cases.front().GetType()->IsBits() &&
                 cases.front().GetType()->GetFlatBitCount() == 0) {
        // Special case to avoid creating non-trivial uses of zero-len bit
        // vectors.
        values[{state_read, activation}] = fb.Literal(UBits(0, 0));
      } else {
        XLS_RET_CHECK_EQ(cases.size(), selectors.size());
        // materialize the next values into a select.
        // Need to reverse to keep the LSB is case 0 etc.
        absl::c_reverse(selectors);
        values[{state_read, activation}] = fb.PrioritySelect(
            fb.Concat(selectors), cases,
            /*default_value=*/values[{state_read, activation - 1}]);
      }
    }
    states.push_back(values[{state_read, activation}]);
  }
  return states;
}
}  // namespace

absl::StatusOr<Function*> UnrollProcToFunction(Proc* p,
                                               int64_t activation_count,
                                               bool include_state,
                                               const Value& token_value) {
  XLS_RET_CHECK_GT(activation_count, 0)
      << "At least one activation is required.";
  XLS_RET_CHECK(!p->next_values().empty() || p->NextState().empty())
      << "Only procs using 'next-node' style are supported.";
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
  absl::flat_hash_map<NodeActivation, BValue> values;
  std::vector<std::optional<BValue>> sends;
  std::vector<std::vector<BValue>> start_states;
  for (int64_t i = 0; i < activation_count; ++i) {
    // Fixup and collect state elements
    XLS_ASSIGN_OR_RETURN(std::vector<BValue> act_state,
                         GetStateValuesBeforeActivation(p, i, fb, values));
    start_states.push_back(std::move(act_state));
    UnrollProcVisitor upv(fb, values, i, token_value);
    XLS_RETURN_IF_ERROR(p->Accept(&upv));
    sends.push_back(upv.GetSentValues());
  }

  BValue return_val;
  // Get the final state.
  XLS_ASSIGN_OR_RETURN(
      std::vector<BValue> last_state,
      GetStateValuesBeforeActivation(p, activation_count, fb, values));
  start_states.push_back(std::move(last_state));
  absl::Span<std::vector<BValue> const> states_after_activation =
      absl::MakeConstSpan(start_states).subspan(1);
  // Collect the return values.
  std::vector<BValue> each_activation;
  XLS_RET_CHECK_EQ(states_after_activation.size(), sends.size());
  for (int64_t i = 0; i < sends.size(); ++i) {
    if (include_state && sends[i].has_value() && !p->StateElements().empty()) {
      each_activation.push_back(
          fb.Tuple({*sends[i], fb.Tuple(states_after_activation[i])}));
    } else if (!p->StateElements().empty() && include_state) {
      // Nothing is actually sent so avoid the empty tuple that z3 doesn't
      // like
      each_activation.push_back(fb.Tuple(states_after_activation[i]));
    } else {
      each_activation.push_back(*sends[i]);
    }
    return_val = fb.Tuple(each_activation);
  }

  XLS_ASSIGN_OR_RETURN(Function * result, fb.BuildWithReturnValue(return_val));

  // Clean up the function a bit. The tracking values might be dangling so we
  // might as well get rid of them to avoid making z3 scan through them.
  DeadCodeEliminationPass dce;
  PassResults pass_results;
  XLS_RETURN_IF_ERROR(
      dce.RunOnFunctionBase(result, {}, &pass_results).status());

  VLOG(2) << "Proc: \n" << p->DumpIr() << "To Func: \n" << result->DumpIr();

  return result;
}

}  // namespace xls
