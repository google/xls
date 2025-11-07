// Copyright 2025 The XLS Authors
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

#include "xls/ir/block_testutils.h"

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
#include "cppitertools/sorted.hpp"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/bits.h"
#include "xls/ir/block.h"
#include "xls/ir/channel.h"
#include "xls/ir/dfs_visitor.h"
#include "xls/ir/function.h"
#include "xls/ir/function_base.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/node.h"
#include "xls/ir/nodes.h"
#include "xls/ir/package.h"
#include "xls/ir/register.h"
#include "xls/ir/source_location.h"
#include "xls/ir/value.h"
#include "xls/ir/value_utils.h"
#include "xls/passes/dce_pass.h"
#include "xls/passes/optimization_pass.h"
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

// This visitor basically inlines one activation of the block into a function
// collecting sent values (and optionally state elements) as return values.
// Receives are provided by function inputs.
class UnrollBlockVisitor final : public DfsVisitorWithDefault {
 public:
  UnrollBlockVisitor(FunctionBuilder& fb,
                     absl::flat_hash_map<NodeActivation, BValue>& values,
                     int64_t activation, bool zero_invalid_outputs)
      : DfsVisitorWithDefault(),
        fb_(fb),
        values_(values),
        activation_(activation),
        zero_invalid_outputs_(zero_invalid_outputs) {}

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

  absl::Status HandleRegisterRead(RegisterRead* read) override {
    XLS_RETURN_IF_ERROR(fb_.GetError());
    XLS_RET_CHECK(values_.contains({read, activation_}))
        << "reg value not created for activation " << activation_ << ": "
        << read;
    return absl::OkStatus();
  }

  absl::Status HandleRegisterWrite(RegisterWrite* write) override {
    XLS_RETURN_IF_ERROR(fb_.GetError());
    values_[{write, activation_}] = fb_.Literal(Value::Tuple({}));
    return absl::OkStatus();
  }

  absl::Status HandleOutputPort(OutputPort* port) override {
    XLS_RETURN_IF_ERROR(fb_.GetError());
    values_[{port, activation_}] = fb_.Literal(Value::Tuple({}));
    output_state_[port->name()] = values_[{port->output_source(), activation_}];
    return absl::OkStatus();
  }

  absl::Status HandleInputPort(InputPort* port) override {
    XLS_RETURN_IF_ERROR(fb_.GetError());
    values_[{port, activation_}] =
        fb_.Param(absl::StrFormat("%s_act%d_input", port->name(), activation_),
                  port->GetType());
    return absl::OkStatus();
  }

  absl::Status HandleStateRead(StateRead* state_read) override {
    return absl::FailedPreconditionError("Blocks cannot have state read!");
  }

  absl::Status HandleSend(Send* s) override {
    return absl::FailedPreconditionError("Blocks cannot have send!");
  }

  absl::Status HandleNext(Next* n) override {
    return absl::FailedPreconditionError("Blocks cannot have next!");
  }

  absl::Status HandleReceive(Receive* r) override {
    return absl::FailedPreconditionError("Blocks cannot have recv!");
  }

  absl::Status HandleAssert(Assert* a) override {
    return absl::UnimplementedError(
        "UnrollBlockVisitor: assert is not supported");
  }
  absl::Status HandleCover(Cover* c) override {
    return absl::UnimplementedError(
        "UnrollBlockVisitor: cover is not supported");
  }

  absl::Status HandleAfterAll(AfterAll* aa) override {
    return absl::FailedPreconditionError("Blocks cannot have after-all!");
  }

  absl::Status HandleInvoke(Invoke* i) override {
    return absl::FailedPreconditionError("Blocks cannot have invoke!");
  }

  absl::Status HandleInstantiationInput(InstantiationInput* ii) override {
    return absl::UnimplementedError(
        "UnrollBlockVisitor: instantiation inputs not supported");
  }
  absl::Status HandleInstantiationOutput(InstantiationOutput* ii) override {
    return absl::UnimplementedError(
        "UnrollBlockVisitor: instantiation outputs not supported");
  }

  // Get the sent value tuple (or nullopt if no value was sent)
  absl::StatusOr<std::optional<BValue>> GetSentValues(Block* b) const {
    if (output_state_.empty()) {
      return std::nullopt;
    }
    std::vector<std::pair<std::string, BValue>> results(output_state_.begin(),
                                                        output_state_.end());
    absl::c_sort(results,
                 [](const std::pair<std::string, BValue>& lhs,
                    const std::pair<std::string, BValue>& rhs) -> bool {
                   return lhs.first < rhs.first;
                 });
    std::vector<BValue> ordered;
    // Perform final cleanup to zero-out invalid output ports if configured to
    // do so.
    absl::flat_hash_map<std::string, std::string> port_to_valid_port;
    for (const auto& [chan, dir] : b->GetChannelsWithMappedPorts()) {
      if (dir == ChannelDirection::kReceive) {
        continue;
      }
      XLS_ASSIGN_OR_RETURN(ChannelPortMetadata metadata,
                           b->GetChannelPortMetadata(chan, dir));
      if (metadata.valid_port && metadata.data_port) {
        port_to_valid_port[*metadata.data_port] = *metadata.valid_port;
      }
    }
    ordered.reserve(results.size());
    for (const auto& [port_name, val] : results) {
      auto valid = port_to_valid_port.find(port_name);
      if (!zero_invalid_outputs_ || valid == port_to_valid_port.end()) {
        ordered.push_back(val);
        continue;
      }
      XLS_RET_CHECK(output_state_.contains(valid->second));
      BValue valid_signal = output_state_.at(valid->second);
      ordered.push_back(
          fb_.Select(valid_signal, /*on_true=*/val,
                     /*on_false=*/fb_.Literal(ZeroOfType(val.GetType()))));
    }
    return fb_.Tuple(
        ordered, SourceInfo(),
        absl::StrFormat("output_values_for_activation_%d", activation_));
  }

 private:
  // The function we are building to do verification on.
  FunctionBuilder& fb_;
  // A map of each nodes on a particular activation to the node that implements
  // the same operation in the test function.
  absl::flat_hash_map<NodeActivation, BValue>& values_;
  // A map of port names to values sent on the most recent activation.
  absl::flat_hash_map<std::string, BValue> output_state_;
  // A map of port names & activation to the values received on that
  // activation.
  absl::flat_hash_map<std::pair<std::string, int64_t>, BValue> input_state_;
  // Which activation are we inlining.
  int64_t activation_;
  // Should we use channel metadata to zero-output ports with valid unset.
  bool zero_invalid_outputs_;
};

// Pull out all the state values before the given activation has started. Also
// update 'values' to include these in the map.
absl::StatusOr<std::vector<BValue>> GetRegisterValuesBeforeActivation(
    Block* b, int64_t activation, FunctionBuilder& fb,
    absl::flat_hash_map<NodeActivation, BValue>& values) {
  bool has_reset_input = b->GetResetPort().has_value();
  std::vector<BValue> states;
  for (Register* reg : iter::sorted(b->GetRegisters(),
                                    [](Register* lhs, Register* rhs) -> bool {
                                      return lhs->name() < rhs->name();
                                    })) {
    XLS_ASSIGN_OR_RETURN(RegisterRead * state_read, b->GetRegisterRead(reg));
    if (activation == 0) {
      values[{state_read, 0}] = fb.Literal(
          reg->reset_value().value_or(ZeroOfType(reg->type())), SourceInfo(),
          absl::StrFormat("%s_initial_value", reg->name()));
    } else {
      std::vector<BValue> cases;
      std::vector<BValue> selectors;
      XLS_ASSIGN_OR_RETURN(absl::Span<RegisterWrite* const> writes,
                           b->GetRegisterWrites(reg));
      std::optional<Node*> reset_line;
      for (RegisterWrite* write : writes) {
        if (write->load_enable()) {
          selectors.push_back(
              values[{write->load_enable().value(), activation - 1}]);
        }
        if (reset_line) {
          XLS_RET_CHECK(!write->reset() || reset_line == write->reset())
              << "Multiple different reset signals for register " << reg;
        }
        reset_line = write->reset();
        cases.push_back(values[{write->data(), activation - 1}]);
      }
      if (!reset_line && has_reset_input) {
        reset_line = b->GetResetPort();
      }
      if (reset_line) {
        // Add a case at the very front for reset.
        cases.insert(
            cases.begin(),
            fb.Literal(reg->reset_value().value_or(ZeroOfType(reg->type())),
                       SourceInfo(),
                       absl::StrFormat("%s_reset_value_act_%d", reg->name(),
                                       activation)));
        BValue is_reset = values[{*reset_line, activation - 1}];
        if (b->GetResetBehavior()->active_low) {
          is_reset = fb.Not(is_reset);
        }
        selectors.insert(selectors.begin(), is_reset);
        if (selectors.empty()) {
          cases.push_back(fb.Not(is_reset));
        }
      }
      XLS_RET_CHECK(selectors.size() == cases.size() ||
                    (selectors.empty() && cases.size() == 1))
          << "Mix of conditional and unconditional writes to register "
          << reg->ToString();
      if (selectors.empty()) {
        XLS_RET_CHECK_EQ(cases.size(), 1) << "no cases for " << reg;
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

absl::StatusOr<Function*> UnrollBlockToFunction(Block* b,
                                                int64_t activation_count,
                                                bool include_state,
                                                bool zero_invalid_outputs) {
  XLS_RET_CHECK_GT(activation_count, 0)
      << "At least one activation is required.";
  if (include_state) {
    XLS_RET_CHECK(!b->GetRegisters().empty() || !b->GetOutputPorts().empty())
        << "No register or output means returned function would return a "
           "single constant value";
  } else {
    XLS_RET_CHECK(!b->GetOutputPorts().empty())
        << "No output ports means returned function would return a single "
           "constant value";
  }
  Package* pkg = b->package();
  FunctionBuilder fb(
      absl::StrFormat("%s_x%d_function", b->name(), activation_count), pkg);
  absl::flat_hash_map<NodeActivation, BValue> values;
  std::vector<std::optional<BValue>> outputs;
  std::vector<std::vector<BValue>> start_states;
  for (int64_t i = 0; i < activation_count; ++i) {
    // Fixup and collect state elements
    XLS_ASSIGN_OR_RETURN(std::vector<BValue> act_state,
                         GetRegisterValuesBeforeActivation(b, i, fb, values));
    start_states.push_back(std::move(act_state));
    UnrollBlockVisitor ubv(fb, values, i, zero_invalid_outputs);
    XLS_RETURN_IF_ERROR(b->Accept(&ubv));
    XLS_ASSIGN_OR_RETURN(std::back_inserter(outputs), ubv.GetSentValues(b));
  }

  BValue return_val;
  // Get the final state.
  XLS_ASSIGN_OR_RETURN(
      std::vector<BValue> last_state,
      GetRegisterValuesBeforeActivation(b, activation_count, fb, values));
  start_states.push_back(std::move(last_state));
  absl::Span<std::vector<BValue> const> states_after_activation =
      absl::MakeConstSpan(start_states).subspan(1);
  // Collect the return values.
  std::vector<BValue> each_activation;
  XLS_RET_CHECK_EQ(states_after_activation.size(), outputs.size());
  for (int64_t i = 0; i < outputs.size(); ++i) {
    if (include_state && outputs[i].has_value() && !b->GetRegisters().empty()) {
      each_activation.push_back(
          fb.Tuple({*outputs[i], fb.Tuple(states_after_activation[i])}));
    } else if (!b->GetRegisters().empty() && include_state) {
      // Nothing is actually sent so avoid the empty tuple that z3 doesn't
      // like
      each_activation.push_back(fb.Tuple(states_after_activation[i]));
    } else {
      each_activation.push_back(*outputs[i]);
    }
    return_val = fb.Tuple(each_activation);
  }

  XLS_ASSIGN_OR_RETURN(Function * result, fb.BuildWithReturnValue(return_val));

  // Clean up the function a bit. The tracking values might be dangling so we
  // might as well get rid of them to avoid making z3 scan through them.
  DeadCodeEliminationPass dce;
  PassResults pass_results;
  OptimizationContext context;
  XLS_RETURN_IF_ERROR(
      dce.RunOnFunctionBase(result, {}, &pass_results, context).status());

  VLOG(2) << "Block: \n" << b->DumpIr() << "To Func: \n" << result->DumpIr();

  return result;
}
}  // namespace xls
