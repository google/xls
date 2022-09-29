// Copyright 2021 The XLS Authors
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

#include "xls/passes/proc_loop_folding.h"
#include <memory>
#include <string>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/types/optional.h"
#include "xls/common/logging/logging.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/bits.h"
#include "xls/ir/node.h"
#include "xls/ir/node_iterator.h"
#include "xls/ir/node_util.h"
#include "xls/ir/nodes.h"
#include "xls/ir/op.h"
#include "xls/ir/value.h"
#include "xls/ir/value_helpers.h"

namespace xls {

// This enum is used to demarcate what goes into each Tuple index for the new
// proc state we will create.
enum {
  kOriginalState,
  kLoopCounter,
  kInductionVariable,
  kLoopCarry,
  kReceiveData,
  kInvariantArgStart,
};

RollIntoProcPass::RollIntoProcPass(
    std::optional<int64_t> unroll_factor) :
  ProcPass("roll_into_proc", "Re-roll an iterative set of nodes into a proc"),
  unroll_factor_(unroll_factor) {}

// We will use this function to unroll the CountedFor loop body by some factor.
// For example, if unroll_factor is 2, we will clone the loopbody once into the
// same function and wire it up as needed.
// If unroll_factor does not divide the CountedFor trip count evenly, then we
// do not perform any unrolling.
absl::StatusOr<CountedFor*> RollIntoProcPass::UnrollCountedForBody(
    Proc* proc, CountedFor* countedfor, int64_t unroll_factor) const {
  int64_t trip_count = countedfor->trip_count();
  // Perform the validity checks.
  if (unroll_factor < 2) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Unroll factor should be 2 or larger, is %d",
                        unroll_factor));
  }
  if (unroll_factor > trip_count) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Unroll factor should be less than or equal to the trip"
                        " count (%d), is %d", trip_count, unroll_factor));
  }
  if (trip_count % unroll_factor != 0) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Unroll factor should evenly divide the trip count (%d)"
                        " is %d", trip_count, unroll_factor));
  }

  Function* countedfor_loopbody = countedfor->body();
  // Clone the function so we don't change the original.
  std::string_view loopbody_name = countedfor_loopbody->name();
  XLS_ASSIGN_OR_RETURN(countedfor_loopbody, countedfor_loopbody->Clone(
      absl::StrFormat("%s_unrolled", loopbody_name),
      countedfor_loopbody->package()));

  // Create a map of original to cloned nodes. This is so we can wire up the
  // nodes more effectively later on.
  absl::flat_hash_map<Node*, Node*> old_to_cloned;

  // We need to keep track of a few things. The first is the parameters, as
  // these can change as we unroll the loop.
  Node* loop_induction_variable;
  Node* loop_carry;
  Node* return_val = countedfor_loopbody->return_value();

  for (Param* param : countedfor_loopbody->params()) {
    // If node is a parameter, check which one it is using the index. If it is
    // index 1, then it's the loop iterator. index 2 is the carry data. index 3+
    // are invariants.
    XLS_ASSIGN_OR_RETURN(int param_index,
                         countedfor_loopbody->GetParamIndex(param));
    switch (param_index) {
      case 0: {
        loop_induction_variable = param;
        old_to_cloned[param] = loop_induction_variable;
        break;
      }
      case 1: {
        loop_carry = param;
        old_to_cloned[param] = loop_carry;
        break;
      }
      default: {
        old_to_cloned[param] = param;
        break;
      }
    }
  }

  // Collect the loop body nodes (aside from params).
  std::vector<Node*> loopbody_nodes;
  for (Node* node : TopoSort(countedfor_loopbody)) {
    if (!node->Is<Param>()) {
      loopbody_nodes.push_back(node);
    }
  }

  // Create a literal for the CountedFor stride. We will need it to update
  // the loop induction variable.
  int64_t countedfor_stride = countedfor->stride();
  int64_t induction_bitcount = countedfor_loopbody->param(0)->BitCountOrDie();
  SourceInfo countedfor_loc = countedfor->loc();
  XLS_ASSIGN_OR_RETURN(auto stride_literal,
                       countedfor_loopbody->MakeNode<Literal>(
                           countedfor_loc, Value(UBits(countedfor_stride,
                                                       induction_bitcount))));

  // Clone the nodes into the loop body function the specified number of times.
  for (int64_t idx = 0; idx < unroll_factor - 1; idx++) {
    // We need to update the loop induction variable and the loop carry value.
    old_to_cloned[loop_carry] = return_val;
    // Add the stride to the induction variable.
    XLS_ASSIGN_OR_RETURN(auto induction_value_next,
                         countedfor_loopbody->MakeNode<BinOp>(
                             countedfor_loc,
                             old_to_cloned.at(loop_induction_variable),
                             stride_literal, Op::kAdd));
    old_to_cloned[loop_induction_variable] = induction_value_next;
    Node* new_return_val = nullptr;
    for (Node* original_node : loopbody_nodes) {
      std::vector<Node*> new_operands;
      for (Node* operand : original_node->operands()) {
        new_operands.push_back(old_to_cloned.at(operand));
      }
      XLS_ASSIGN_OR_RETURN(Node* cloned_node,
                           original_node->Clone(new_operands));
      old_to_cloned[original_node] = cloned_node;
      new_return_val = cloned_node;
    }
    // New return value will be the last node processed
    return_val = new_return_val;
  }

  // Update the return value of the function.
  XLS_RETURN_IF_ERROR(countedfor_loopbody->set_return_value(return_val));

  // Rebuild the CountedFor.
  int64_t new_trip_count = countedfor->trip_count() / unroll_factor;
  int64_t new_stride = countedfor->stride() * unroll_factor;
  XLS_ASSIGN_OR_RETURN(auto new_countedfor, proc->MakeNode<CountedFor>(
      countedfor_loc, countedfor->initial_value(), countedfor->invariant_args(),
      new_trip_count, new_stride, countedfor_loopbody));

  XLS_RETURN_IF_ERROR(countedfor->ReplaceUsesWith(new_countedfor));

  return new_countedfor;
}

absl::StatusOr<Value> RollIntoProcPass::CreateInitialState(
    Proc* proc, CountedFor* countedfor, Receive* recv) const {

  // Get important values out of the countedfor.
  Function* loopbody = countedfor->body();
  Node* initial_value = countedfor->initial_value();
  absl::Span<Node* const> loop_invariants = countedfor->invariant_args();
  Value initial_induction_var = ZeroOfType(loopbody->param(0)->GetType());
  Value initial_trip_counter = ZeroOfType(loopbody->param(0)->GetType());
  std::vector<Value> new_state_vals;

  // Order the new state in the following way:
  // 0 - Original state
  // 1 - Loop counter
  // 2 - Induction variable
  // 3 - Loop carry
  // 4 - Receive data
  // 5, 6, 7, ... - Invariant 1, 2, 3, ...
  new_state_vals.push_back(proc->GetInitValueElement(0));
  new_state_vals.push_back(ZeroOfType(loopbody->param(0)->GetType()));
  new_state_vals.push_back(ZeroOfType(loopbody->param(0)->GetType()));
  new_state_vals.push_back(ZeroOfType(initial_value->GetType()));
  XLS_ASSIGN_OR_RETURN(Channel* recv_channel,
                      proc->package()->GetChannel(recv->channel_id()));
  new_state_vals.push_back(ZeroOfType(recv_channel->type()));
  // Naively push the loop invariants into the state.
  for (Node* invariant : loop_invariants) {
    // We only want to handle invariants that aren't Literals.
    if (!invariant->Is<Literal>()) {
      new_state_vals.push_back(ZeroOfType(invariant->GetType()));
    }
  }
  return Value::Tuple(new_state_vals);
}

absl::StatusOr<std::vector<Node*>> RollIntoProcPass::SelectBetween(
  Proc* proc, CountedFor* countedfor, Node* selector,
    absl::Span<Node* const> on_false, absl::Span<Node* const> on_true) const {
  XLS_RET_CHECK_EQ(on_false.size(), on_true.size());
  int64_t num_selects = on_false.size();
  std::vector<Node*> selects(num_selects);
  for (int64_t i = 0; i < num_selects; i++) {
    std::vector<Node*> select_choices = {on_false.at(i), on_true.at(i)};
    XLS_ASSIGN_OR_RETURN(auto select_invariant,
                         proc->MakeNode<Select>(countedfor->loc(), selector,
                                                select_choices, absl::nullopt));
    selects[i] = select_invariant;
    // on_true contains the original invariant nodes, whose uses will need to
    // be replaced by the newly created Select.
    XLS_RETURN_IF_ERROR(on_true.at(i)->ReplaceUsesWith(select_invariant));
  }
  return selects;
}

absl::StatusOr<Node*> RollIntoProcPass::CloneCountedFor(
    Proc* proc, CountedFor* countedfor, Node* loop_induction_variable,
    Node* loop_carry) const {
  Function* loopbody = countedfor->body();
  absl::Span<Node* const> loop_invariants = countedfor->invariant_args();
  absl::flat_hash_map<Node*, Node*> old_to_new;

  for (Param* param : loopbody->params()) {
    // If node is a parameter, check which one it is using the ID. If it is
    // ID 1, then it's the loop iterator. ID 2 is the carry data. ID 3+ are
    // invariants.
    XLS_ASSIGN_OR_RETURN(int param_index,
                         loopbody->GetParamIndex(param));
    switch (param_index) {
      case 0: {
        old_to_new[param] = loop_induction_variable;
        break;
      }
      case 1: {
        old_to_new[param] = loop_carry;
        break;
      }
      default: {
        old_to_new[param] = loop_invariants.at(param_index - 2);
        break;
      }
    }
  }

  for (Node* node : TopoSort(loopbody)) {
    // Params were already taken care of above.
    if (!node->Is<Param>()) {
      std::vector<Node*> new_operands;
      for (Node* operand : node->operands()) {
        new_operands.push_back(old_to_new.at(operand));
      }
      XLS_ASSIGN_OR_RETURN(Node* cloned_node, node->CloneInNewFunction(
          new_operands, proc));
      old_to_new[node] = cloned_node;
    }
  }
  return old_to_new[loopbody->return_value()];
}

absl::StatusOr<Node*> RollIntoProcPass::ReplaceReceiveWithConditionalReceive(
    Proc* proc, Receive* original_receive, Node* receive_condition,
    Node* on_condition_false) const {
  XLS_ASSIGN_OR_RETURN(
      auto new_receive,
      proc->MakeNode<Receive>(original_receive->loc(), proc->TokenParam(),
                              receive_condition, original_receive->channel_id(),
                              original_receive->is_blocking()));
  XLS_ASSIGN_OR_RETURN(
      auto new_receive_token,
      proc->MakeNode<TupleIndex>(original_receive->loc(), new_receive, 0));
  XLS_ASSIGN_OR_RETURN(
      auto new_receive_value,
      proc->MakeNode<TupleIndex>(original_receive->loc(), new_receive, 1));
  XLS_ASSIGN_OR_RETURN(
      auto receive_select_value,
      proc->MakeNode<Select>(
          original_receive->loc(), receive_condition,
          std::vector<Node*>{on_condition_false, new_receive_value},
          absl::nullopt));
  XLS_ASSIGN_OR_RETURN(
      auto receive_select_token,
      proc->MakeNode<AfterAll>(
          original_receive->loc(),
          std::vector<Node*>{new_receive->token(), new_receive_token}));
  XLS_ASSIGN_OR_RETURN(
      auto receive_select,
      proc->MakeNode<Tuple>(
          original_receive->loc(),
          std::vector<Node*>{receive_select_token, receive_select_value}));
  XLS_RETURN_IF_ERROR(original_receive->ReplaceUsesWith(receive_select));
  XLS_RETURN_IF_ERROR(proc->RemoveNode(original_receive));
  return receive_select;
}

absl::StatusOr<bool> RollIntoProcPass::RunOnProcInternal(
    Proc* proc, const PassOptions& options, PassResults* results) const {

  // Find Send, Receive, and CountedFor nodes.
  CountedFor* counted_for_node = nullptr;
  Receive* receive_node = nullptr;
  Send* send_node = nullptr;
  for (Node* node : proc->nodes()) {
    if (node->Is<CountedFor>()) {
      // If more than one CountedFor, Receive or Send, give up.
      if (counted_for_node != nullptr) {
        return false;
      }
      counted_for_node = node->As<CountedFor>();
    } else if (node->Is<Receive>()) {
      if (receive_node != nullptr) {
        return false;
      }
      receive_node = node->As<Receive>();
    } else if (node->Is<Send>()) {
      if (send_node != nullptr) {
        return false;
      }
      send_node = node->As<Send>();
    }
  }

  // Check if we found the CountedFor, ReceiveNode and SendNode.
  if (counted_for_node == nullptr || receive_node == nullptr ||
      send_node == nullptr) {
    return false;
  }

  if (unroll_factor_.has_value()) {
    XLS_ASSIGN_OR_RETURN(counted_for_node,
                         UnrollCountedForBody(
                             proc, counted_for_node, unroll_factor_.value()));
  }

  // Get information from the CountedFor node.
  Node* initial_value = counted_for_node->initial_value();
  Function* loopbody = counted_for_node->body();
  absl::Span<Node* const> loop_invariants = counted_for_node->invariant_args();
  int64_t trip_count_value = counted_for_node->trip_count();
  int64_t stride = counted_for_node->stride();

  // Create a Tuple for the new state.
  Value initial_induction_var = ZeroOfType(loopbody->param(0)->GetType());
  Value initial_trip_counter = ZeroOfType(loopbody->param(0)->GetType());
  XLS_ASSIGN_OR_RETURN(Value init_state_new,
                      CreateInitialState(proc, counted_for_node, receive_node));

  // The CountedFor iterator and loop value need to be rolled up into the Proc
  // state. We also need to roll up the invariants, as well as keep track of the
  // receive value. So we need to create a Tuple with all of these values
  // included, along with the old proc state as well.

  // Need to keep track of the old proc next state. We will use this when
  // making the new proc state at the end of this pass.
  Node* old_proc_next = proc->GetNextStateElement(0);
  XLS_ASSIGN_OR_RETURN(
      Literal * dummy_state_new,
      proc->MakeNode<Literal>(proc->GetStateParam(0)->loc(), init_state_new));

  // Anything that used the old proc state now uses the tuple at index 0.
  XLS_RETURN_IF_ERROR(
      proc->GetStateParam(0)
          ->ReplaceUsesWithNew<TupleIndex>(dummy_state_new, kOriginalState)
          .status());

  // We want to mux the invariants with their proc state versions so we can
  // allow for CountedFor loops where an invariant is dependent on the receive
  // node. The mux select line will be current trip counter == 0.
  SourceInfo countedfor_loc = counted_for_node->loc();
  XLS_ASSIGN_OR_RETURN(
      auto trip_counter,
      proc->MakeNode<TupleIndex>(SourceInfo(), dummy_state_new, kLoopCounter));

  XLS_ASSIGN_OR_RETURN(auto zero, proc->MakeNode<Literal>(
      countedfor_loc, ZeroOfType(loopbody->param(0)->GetType())));
  XLS_ASSIGN_OR_RETURN(auto is_first_iteration,
                       proc->MakeNode<CompareOp>(countedfor_loc, trip_counter,
                                               zero, Op::kEq));

  // Need to replace the invariants with a Select each, to choose between the
  // proc state, or a modified version of the invariants when loop iteration
  // equals 0. These will feed the CountedFor nodes as well as the next state.
  int64_t proc_state_tuple_idx = kInvariantArgStart;
  std::vector<Node*> select_true;
  std::vector<Node*> select_false;
  std::vector<Node*> invariant_selects;
  for (Node* invariant : loop_invariants) {
    // Only handle non-Literal invariants.
    if (!invariant->Is<Literal>()) {
      XLS_ASSIGN_OR_RETURN(auto tuple_idx,
                           proc->MakeNode<TupleIndex>(countedfor_loc,
                                                      dummy_state_new,
                                                      proc_state_tuple_idx));
      select_true.push_back(invariant);
      select_false.push_back(tuple_idx);
      proc_state_tuple_idx++;
    }
  }
  XLS_ASSIGN_OR_RETURN(invariant_selects,
                       SelectBetween(proc, counted_for_node, is_first_iteration,
                                     select_false, select_true));

  // Get the loop induction variable and loop carry variable from the new proc
  // state, as these are parameters to the CountedFor node.
  XLS_ASSIGN_OR_RETURN(auto loop_induction_variable,
                       proc->MakeNode<TupleIndex>(SourceInfo(), dummy_state_new,
                                                  kInductionVariable));
  XLS_ASSIGN_OR_RETURN(
      auto loop_carry,
      proc->MakeNode<TupleIndex>(SourceInfo(), dummy_state_new, kLoopCarry));

  // Clone the CountedFor nodes into the proc and get the CountedFor return
  // value, which we will use for the next proc state.
  XLS_ASSIGN_OR_RETURN(Node* countedfor_return, CloneCountedFor(
      proc, counted_for_node, loop_induction_variable, loop_carry));

  // Create the next CountedFor loop nodes.
  int64_t induction_bits = loopbody->param(0)->BitCountOrDie();
  XLS_ASSIGN_OR_RETURN(auto trip_count,
                       proc->MakeNode<Literal>(
                           countedfor_loc, Value(UBits(
                               trip_count_value-1, induction_bits))));
  XLS_ASSIGN_OR_RETURN(auto is_final_iteration,
                       proc->MakeNode<CompareOp>(countedfor_loc,
                                               trip_counter, trip_count,
                                               Op::kEq));
  XLS_ASSIGN_OR_RETURN(auto initial_value_val,
                       proc->MakeNode<Literal>(
                           countedfor_loc, Value(UBits(0, induction_bits))));

  // Increment the induction value by stride each time.
  XLS_ASSIGN_OR_RETURN(auto induction_value_incr,
                       proc->MakeNode<Literal>(
                           countedfor_loc, Value(UBits(stride,
                                                       induction_bits))));
  XLS_ASSIGN_OR_RETURN(auto induction_value_next,
                       proc->MakeNode<BinOp>(countedfor_loc,
                                               loop_induction_variable,
                                               induction_value_incr, Op::kAdd));
  std::vector<Node*> next_induction_value_nodes = {induction_value_next,
                                                   initial_value_val};
  XLS_ASSIGN_OR_RETURN(auto next_loop_induction_value,
                       proc->MakeNode<Select>(countedfor_loc,
                                              is_final_iteration,
                                              next_induction_value_nodes,
                                              absl::nullopt));

  // Increment the trip counter by 1 each time.
  XLS_ASSIGN_OR_RETURN(auto trip_count_incr,
                       proc->MakeNode<Literal>(
                           countedfor_loc, Value(UBits(1, induction_bits))));
  XLS_ASSIGN_OR_RETURN(auto trip_counter_next,
                       proc->MakeNode<BinOp>(countedfor_loc, trip_counter,
                                             trip_count_incr, Op::kAdd));
  std::vector<Node*> next_loop_trip_count = {trip_counter_next,
                                             initial_value_val};
  XLS_ASSIGN_OR_RETURN(auto next_loop_trip_counter,
                       proc->MakeNode<Select>(countedfor_loc,
                                              is_final_iteration,
                                              next_loop_trip_count,
                                              absl::nullopt));

  // Create the next loop accumulator value.
  std::vector<Node*> next_loop_carry_sel = {countedfor_return, initial_value};
  XLS_ASSIGN_OR_RETURN(auto next_loop_carry,
                       proc->MakeNode<Select>(countedfor_loc,
                                              is_final_iteration,
                                              next_loop_carry_sel,
                                              absl::nullopt));

  // Create the "ReceiveIf" node.
  XLS_ASSIGN_OR_RETURN(
      auto receive_val_proc_state,
      proc->MakeNode<TupleIndex>(SourceInfo(), dummy_state_new, kReceiveData));
  XLS_ASSIGN_OR_RETURN(auto receive_select,
                       ReplaceReceiveWithConditionalReceive (
                           proc, receive_node, is_first_iteration,
                           receive_val_proc_state));

  // Create the "SendIf" node.
  XLS_ASSIGN_OR_RETURN(auto new_send,
                       proc->MakeNode<Send>(send_node->loc(),
                                            proc->TokenParam(),
                                            send_node->data(),
                                            is_final_iteration,
                                            send_node->channel_id()));
  XLS_RETURN_IF_ERROR(send_node->ReplaceUsesWith(new_send));
  XLS_RETURN_IF_ERROR(proc->RemoveNode(send_node));

  // Build the next state. We need to include the following:
  // - next loop iteration value
  // - new loop carry value
  // - all of the loop invariant select nodes
  // - a select node for the state from the original proc
  std::vector<Node*> next_state_tuple;

  // Select between the proc state and the old next state to update the state
  // from the original proc.
  XLS_ASSIGN_OR_RETURN(auto proc_state_old,
                       proc->MakeNode<TupleIndex>(SourceInfo(), dummy_state_new,
                                                  kOriginalState));
  std::vector<Node*> invariant_nodes = {proc_state_old, old_proc_next};
  XLS_ASSIGN_OR_RETURN(auto select_original_proc_state,
                       proc->MakeNode<Select>(countedfor_loc,
                                              is_first_iteration,
                                              invariant_nodes, absl::nullopt));
  next_state_tuple.push_back(select_original_proc_state);
  next_state_tuple.push_back(next_loop_trip_counter);
  next_state_tuple.push_back(next_loop_induction_value);
  next_state_tuple.push_back(next_loop_carry);
  XLS_ASSIGN_OR_RETURN(
      auto receive_select_val,
      proc->MakeNode<TupleIndex>(SourceInfo(), receive_select, 1));
  next_state_tuple.push_back(receive_select_val);
  for (Node* invariant : invariant_selects) {
    next_state_tuple.push_back(invariant);
  }

  XLS_ASSIGN_OR_RETURN(
      auto new_next_state,
      proc->MakeNode<Tuple>(proc->GetStateParam(0)->loc(), next_state_tuple));
  XLS_RETURN_IF_ERROR(proc->ReplaceStateElement(0,
                                                proc->GetStateParam(0)->name(),
                                                init_state_new, new_next_state)
                          .status());
  XLS_RETURN_IF_ERROR(dummy_state_new->ReplaceUsesWith(proc->GetStateParam(0)));
  XLS_RETURN_IF_ERROR(proc->RemoveNode(dummy_state_new));

  return true;
}

}  // namespace xls
