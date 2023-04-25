// Copyright 2023 The XLS Authors
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

#include "xls/scheduling/proc_clumping_pass.h"

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/bits_ops.h"
#include "xls/ir/node_iterator.h"
#include "xls/ir/value_helpers.h"

namespace xls {

namespace {

// Wraps a set of data dependencies around the back of the state register.
// Returns all of the nodes added in the process.
absl::StatusOr<std::vector<Node*>> DependenciesToStateEdge(
    Node* source, Node* condition, const absl::flat_hash_set<Node*>& sinks) {
  if (sinks.empty()) {
    return std::vector<Node*>();
  }

  if (source->GetType()->IsToken()) {
    return std::vector<Node*>();
  }

  Proc* proc = source->function_base()->AsProcOrDie();

  std::vector<Node*> added;

  std::string state_name = absl::StrFormat("%s_delayed", source->GetName());
  Value init_value = ZeroOfType(source->GetType());
  XLS_ASSIGN_OR_RETURN(
      Param * state,
      proc->AppendStateElement(state_name, init_value, std::nullopt));
  added.push_back(state);
  XLS_ASSIGN_OR_RETURN(int64_t state_index, proc->GetStateParamIndex(state));

  XLS_ASSIGN_OR_RETURN(
      Node * muxed_source,
      proc->MakeNode<Select>(SourceInfo(), condition,
                             std::vector<Node*>({state, source}),
                             /*default_value=*/std::nullopt));
  added.push_back(muxed_source);
  XLS_RETURN_IF_ERROR(proc->SetNextStateElement(state_index, muxed_source));

  for (Node* sink : sinks) {
    (void)sink->ReplaceOperand(source, state);
  }

  return added;
}

// Clones all nodes in the given proc in a given cycle into a separate proc,
// then schedules that proc with II = 1, and finally maps the computed schedule
// back to a schedule on the nodes of the given proc. This is used for the
// temporal scheduling of logical stages within physical stages.
absl::StatusOr<ScheduleCycleMap> ScheduleSubProc(
    Proc* proc, int64_t cycle, const ScheduleCycleMap& scm,
    const SchedulingPassOptions& options) {
  absl::flat_hash_map<Node*, Node*> original_to_clone;

  std::string new_name = absl::StrFormat("%s_cloned", proc->name());
  Proc* clone = proc->package()->AddProc(std::make_unique<Proc>(
      new_name, proc->TokenParam()->GetName(), proc->package()));
  original_to_clone[proc->TokenParam()] = clone->TokenParam();
  for (int64_t i = 0; i < proc->GetStateElementCount(); ++i) {
    if (scm.at(proc->GetNextStateElement(i)) == cycle) {
      XLS_ASSIGN_OR_RETURN(
          Param * cloned_param,
          clone->AppendStateElement(proc->GetStateParam(i)->GetName(),
                                    proc->GetInitValueElement(i)));
      original_to_clone[proc->GetStateParam(i)] = cloned_param;
    }
  }
  for (Node* node : TopoSort(proc)) {
    if (scm.at(node) != cycle) {
      continue;
    }
    if (node->Is<Param>()) {
      continue;
    }

    std::vector<Node*> cloned_operands;
    for (Node* operand : node->operands()) {
      if (!original_to_clone.contains(operand)) {
        XLS_ASSIGN_OR_RETURN(
            Literal * stub,
            clone->MakeNode<Literal>(SourceInfo(),
                                     Value(ZeroOfType(operand->GetType()))));
        cloned_operands.push_back(stub);
      } else {
        cloned_operands.push_back(original_to_clone.at(operand));
      }
    }

    XLS_ASSIGN_OR_RETURN(original_to_clone[node],
                         node->CloneInNewFunction(cloned_operands, clone));
  }

  SchedulingOptions scheduling_options = options.scheduling_options;
  scheduling_options.clear_constraints();
  for (const SchedulingConstraint& constraint :
       options.scheduling_options.constraints()) {
    if (std::holds_alternative<BackedgeConstraint>(constraint)) {
      continue;
    }
    // TODO(taktoa): map the nodes in these constraints to the cloned nodes
    XLS_CHECK(!std::holds_alternative<NodeInCycleConstraint>(constraint));
    XLS_CHECK(!std::holds_alternative<DifferenceConstraint>(constraint));
    scheduling_options.add_constraint(constraint);
  }

  XLS_ASSIGN_OR_RETURN(PipelineSchedule schedule,
                       PipelineSchedule::Run(clone, *options.delay_estimator,
                                             scheduling_options));

  absl::flat_hash_map<Node*, Node*> clone_to_original;
  for (const auto& [original, cloned] : original_to_clone) {
    clone_to_original[cloned] = original;
  }

  absl::flat_hash_map<Node*, int64_t> sub_schedule;
  for (int64_t c = 0; c < schedule.length(); ++c) {
    for (Node* node : schedule.nodes_in_cycle(c)) {
      if (clone_to_original.contains(node)) {
        sub_schedule[clone_to_original.at(node)] = c;
      }
    }
  }

  XLS_RETURN_IF_ERROR(clone->package()->RemoveProc(clone));

  return sub_schedule;
}

struct Counter {
  // All the nodes used to implement this counter FSM. Useful for setting the
  // schedule appropriately.
  std::vector<Node*> nodes;
  // predicates[i] is a 1-bit signal that will be high when the cycle count is
  // equal to i modulo predicates.size().
  std::vector<Node*> predicates;
};

absl::StatusOr<Counter> AddCounterFSM(Proc* proc, std::string_view name,
                                      int64_t limit) {
  XLS_CHECK_GE(limit, 0);
  int64_t bits = CeilOfLog2(limit);
  XLS_ASSIGN_OR_RETURN(
      Param * fsm_state,
      proc->AppendStateElement(name, Value(UBits(0, bits)), std::nullopt));
  XLS_ASSIGN_OR_RETURN(int64_t fsm_state_index,
                       proc->GetStateParamIndex(fsm_state));
  XLS_ASSIGN_OR_RETURN(Node * zero, proc->MakeNode<Literal>(
                                        SourceInfo(), Value(UBits(0, bits))));
  XLS_ASSIGN_OR_RETURN(
      Node * one, proc->MakeNode<Literal>(SourceInfo(), Value(UBits(1, bits))));
  XLS_ASSIGN_OR_RETURN(
      Node * added,
      proc->MakeNode<BinOp>(SourceInfo(), fsm_state, one, Op::kAdd));

  std::vector<Node*> nodes = {fsm_state, zero, one, added};

  Node* next_fsm_state = nullptr;
  if (limit == Exp2<int64_t>(static_cast<int>(bits))) {
    XLS_ASSIGN_OR_RETURN(Node * will_overflow,
                         proc->MakeNode<BitwiseReductionOp>(
                             SourceInfo(), fsm_state, Op::kAndReduce));
    nodes.push_back(will_overflow);
    XLS_ASSIGN_OR_RETURN(next_fsm_state, proc->MakeNode<Select>(
                                             SourceInfo(), will_overflow,
                                             std::vector<Node*>({added, zero}),
                                             /*default_value=*/std::nullopt));
  } else {
    XLS_ASSIGN_OR_RETURN(
        Node * limit_lit,
        proc->MakeNode<Literal>(SourceInfo(), Value(UBits(limit, bits))));
    XLS_ASSIGN_OR_RETURN(
        Node * equals_limit,
        proc->MakeNode<CompareOp>(SourceInfo(), added, limit_lit, Op::kEq));
    nodes.push_back(limit_lit);
    nodes.push_back(equals_limit);
    XLS_ASSIGN_OR_RETURN(next_fsm_state, proc->MakeNode<Select>(
                                             SourceInfo(), equals_limit,
                                             std::vector<Node*>({added, zero}),
                                             /*default_value=*/std::nullopt));
  }
  XLS_RETURN_IF_ERROR(
      proc->SetNextStateElement(fsm_state_index, next_fsm_state));
  nodes.push_back(next_fsm_state);

  std::vector<Node*> predicates;
  predicates.reserve(limit);
  for (int64_t i = 0; i < limit; ++i) {
    XLS_ASSIGN_OR_RETURN(
        Node * index,
        proc->MakeNode<Literal>(SourceInfo(), Value(UBits(i, bits))));
    XLS_ASSIGN_OR_RETURN(
        Node * pred,
        proc->MakeNode<CompareOp>(SourceInfo(), fsm_state, index, Op::kEq));
    predicates.push_back(pred);
    nodes.push_back(index);
    nodes.push_back(pred);
  }

  return Counter{nodes, predicates};
}

// Add the given condition as a predicate to the given side-effecting node.
// If the given node is not side-effecting then this has no effect.
// Returns the node that `node` was replaced with if it was.
absl::StatusOr<std::optional<Node*>> AddConditionToSideEffectingNode(
    Node* node, Node* condition, ScheduleCycleMap* scm) {
  XLS_CHECK_NE(scm, nullptr);

  FunctionBase* f = node->function_base();

  if (node->Is<Send>()) {
    Send* send = node->As<Send>();
    Node* new_pred = condition;
    if (std::optional<Node*> pred = node->As<Send>()->predicate()) {
      XLS_ASSIGN_OR_RETURN(
          new_pred,
          f->MakeNode<NaryOp>(SourceInfo(),
                              std::vector<Node*>({condition, pred.value()}),
                              Op::kAnd));
      int64_t s = scm->at(pred.value());
      (*scm)[new_pred] = s;
    }
    XLS_ASSIGN_OR_RETURN(
        Node * new_send,
        f->MakeNode<Send>(send->loc(), send->token(), send->data(), new_pred,
                          send->channel_id()));
    XLS_RETURN_IF_ERROR(send->ReplaceUsesWith(new_send));
    return new_send;
  }

  if (node->Is<Receive>()) {
    Receive* recv = node->As<Receive>();
    Node* new_pred = condition;
    if (std::optional<Node*> pred = node->As<Receive>()->predicate()) {
      XLS_ASSIGN_OR_RETURN(
          new_pred,
          f->MakeNode<NaryOp>(SourceInfo(),
                              std::vector<Node*>({condition, pred.value()}),
                              Op::kAnd));
      int64_t s = scm->at(pred.value());
      (*scm)[new_pred] = s;
    }
    XLS_ASSIGN_OR_RETURN(
        Node * new_recv,
        f->MakeNode<Receive>(recv->loc(), recv->token(), new_pred,
                             recv->channel_id(), recv->is_blocking()));
    XLS_RETURN_IF_ERROR(recv->ReplaceUsesWith(new_recv));
    return new_recv;
  }

  if (node->Is<Assert>()) {
    Assert* assert = node->As<Assert>();
    Node* new_cond = condition;
    if (std::optional<Node*> cond = node->As<Assert>()->condition()) {
      XLS_ASSIGN_OR_RETURN(
          new_cond,
          f->MakeNode<NaryOp>(SourceInfo(),
                              std::vector<Node*>({condition, cond.value()}),
                              Op::kAnd));
      int64_t s = scm->at(cond.value());
      (*scm)[new_cond] = s;
    }
    XLS_ASSIGN_OR_RETURN(
        Node * new_assert,
        f->MakeNode<Assert>(assert->loc(), assert->token(), new_cond,
                            assert->message(), assert->label()));
    XLS_RETURN_IF_ERROR(assert->ReplaceUsesWith(new_assert));
    return new_assert;
  }

  if (node->Is<Cover>()) {
    Cover* cover = node->As<Cover>();
    Node* new_cond = condition;
    if (std::optional<Node*> cond = node->As<Cover>()->condition()) {
      XLS_ASSIGN_OR_RETURN(
          new_cond,
          f->MakeNode<NaryOp>(SourceInfo(),
                              std::vector<Node*>({condition, cond.value()}),
                              Op::kAnd));
      int64_t s = scm->at(cond.value());
      (*scm)[new_cond] = s;
    }
    XLS_ASSIGN_OR_RETURN(Node * new_cover,
                         f->MakeNode<Cover>(cover->loc(), cover->token(),
                                            new_cond, cover->label()));
    XLS_RETURN_IF_ERROR(cover->ReplaceUsesWith(new_cover));
    return new_cover;
  }

  if (node->Is<Trace>()) {
    Trace* trace = node->As<Trace>();
    Node* new_cond = condition;
    if (std::optional<Node*> cond = node->As<Trace>()->condition()) {
      XLS_ASSIGN_OR_RETURN(
          new_cond,
          f->MakeNode<NaryOp>(SourceInfo(),
                              std::vector<Node*>({condition, cond.value()}),
                              Op::kAnd));
      int64_t s = scm->at(cond.value());
      (*scm)[new_cond] = s;
    }
    XLS_ASSIGN_OR_RETURN(
        Node * new_trace,
        f->MakeNode<Trace>(trace->loc(), trace->token(), new_cond,
                           trace->args(), trace->format()));
    XLS_RETURN_IF_ERROR(trace->ReplaceUsesWith(new_trace));
    return new_trace;
  }

  return std::nullopt;
}

}  // namespace

absl::StatusOr<bool> ProcClumpingPass::RunInternal(
    SchedulingUnit<>* unit, const SchedulingPassOptions& options,
    SchedulingPassResults* results) const {
  XLS_CHECK(unit->schedule.has_value())
      << "Proc clumping pass requires a pre-existing schedule";

  std::optional<FunctionBase*> top = unit->ir->GetTop();
  if (!top.has_value()) {
    return false;
  }
  FunctionBase* f = top.value();
  if (!f->IsProc()) {
    // Proc clumping pass does not currently support functions.
    return false;
  }
  Proc* proc = f->AsProcOrDie();

  int64_t ii = 1;
  if (proc->GetInitiationInterval().has_value()) {
    ii = proc->GetInitiationInterval().value();
  }

  if (ii == 1) {
    return false;
  }

  if (options.scheduling_options.pipeline_stages().has_value()) {
    XLS_RET_CHECK_EQ(ii, options.scheduling_options.pipeline_stages().value())
        << "For initiation interval > 1, initiation interval must currently "
        << "equal the number of pipeline stages.";
  }

  int64_t physical_length = unit->schedule.value().length();
  ScheduleCycleMap physical_schedule;
  for (int64_t c = 0; c < physical_length; ++c) {
    for (Node* node : unit->schedule.value().nodes_in_cycle(c)) {
      physical_schedule[node] = c;
    }
  }

  // A logical schedule is an assignment from nodes to "logical cycles", which
  // range from 0 to `(physical pipeline length * II) - 1`. For example, if the
  // II is 3 and the physical pipeline length is 5, then logical cycle 10 refers
  // to what executes in physical pipeline stage 3 when the FSM is equal to 1.
  ScheduleCycleMap logical_schedule;
  for (int64_t c = 0; c < unit->schedule.value().length(); ++c) {
    XLS_ASSIGN_OR_RETURN(ScheduleCycleMap sub_schedule,
                         ScheduleSubProc(proc, c, physical_schedule, options));
    for (const auto& [node, sub_cycle] : sub_schedule) {
      logical_schedule[node] = sub_cycle;
    }
  }

  absl::flat_hash_map<Node*, absl::flat_hash_set<Node*>> node_to_users;
  for (Node* node : TopoSort(proc)) {
    if (node->Is<Param>()) {
      continue;
    }

    node_to_users[node];
    for (Node* user : node->users()) {
      if ((logical_schedule.at(node) != logical_schedule.at(user)) ||
          (physical_schedule.at(node) != physical_schedule.at(user))) {
        node_to_users[node].insert(user);
      }
    }
  }

  XLS_ASSIGN_OR_RETURN(Counter counter,
                       AddCounterFSM(proc, "ii_fsm_state", ii));

  for (Node* node : counter.nodes) {
    physical_schedule[node] = 0;
  }

  for (const auto& [node, _users] : node_to_users) {
    if (!proc->HasImplicitUse(node)) {
      continue;
    }

    // If there are state backedges (implicit uses as a next node) on the given
    // node, we need to add a mux driven by the counter predicate corresponding
    // to the logical cycle of the node.
    int64_t physical_cycle = physical_schedule.at(node);
    int64_t logical_cycle = logical_schedule.at(node);
    Node* predicate = counter.predicates.at(logical_cycle);
    for (int64_t index : proc->GetNextStateIndices(node)) {
      Node* prev_state = proc->GetStateParam(index);
      XLS_ASSIGN_OR_RETURN(
          Node * mux,
          proc->MakeNode<Select>(SourceInfo(), predicate,
                                 std::vector<Node*>({prev_state, node}),
                                 /*default_value=*/std::nullopt));
      XLS_RETURN_IF_ERROR(proc->SetNextStateElement(index, mux));
      physical_schedule[mux] = physical_cycle;
    }
  }

  for (const auto& [node, users] : node_to_users) {
    int64_t physical_cycle = physical_schedule.at(node);
    int64_t logical_cycle = logical_schedule.at(node);
    Node* condition = counter.predicates.at(logical_cycle);

    // Create a state register for `node` and replace its uses with the state
    // register, and set the load enable of the register to the condition of
    // `node`.
    XLS_ASSIGN_OR_RETURN(std::vector<Node*> added,
                         DependenciesToStateEdge(node, condition, users));
    for (Node* added_node : added) {
      physical_schedule[added_node] = physical_cycle;
    }
  }

  for (Node* node : TopoSort(proc)) {
    // Skip over nodes that we added in this pass, since none of them are side
    // effecting so far.
    if (!logical_schedule.contains(node)) {
      continue;
    }

    int64_t physical_cycle = physical_schedule.at(node);
    int64_t logical_cycle = logical_schedule.at(node);
    Node* condition = counter.predicates.at(logical_cycle);

    // Add the counter predicate of the logical cycle of `node` as a condition
    // to `node` if it is side-effecting.
    XLS_ASSIGN_OR_RETURN(
        std::optional<Node*> new_node,
        AddConditionToSideEffectingNode(node, condition, &physical_schedule));
    if (new_node.has_value()) {
      XLS_RETURN_IF_ERROR(proc->RemoveNode(node));
      physical_schedule.erase(node);
      physical_schedule[new_node.value()] = physical_cycle;
    }
  }

  unit->schedule =
      PipelineSchedule(unit->schedule.value().function_base(),
                       physical_schedule, unit->schedule.value().length());

  return true;
}

}  // namespace xls
