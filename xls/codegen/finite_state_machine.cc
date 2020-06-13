// Copyright 2020 Google LLC
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

#include "xls/codegen/finite_state_machine.h"

#include "absl/algorithm/container.h"
#include "absl/strings/str_cat.h"
#include "xls/common/status/status_macros.h"

namespace xls {
namespace verilog {

void FsmBlockBase::EmitAssignments(StatementBlock* statement_block) const {
  for (const auto& assignment : assignments_) {
    statement_block->Add<BlockingAssignment>(assignment.lhs, assignment.rhs);
  }
  for (const ConditionalFsmBlock& cond_block : conditional_blocks_) {
    if (cond_block.HasAssignments()) {
      Conditional* conditional = statement_block->Add<Conditional>(
          statement_block->parent(), cond_block.condition());
      cond_block.EmitConditionalAssignments(conditional,
                                            conditional->consequent());
    }
  }
}

void FsmBlockBase::EmitStateTransitions(StatementBlock* statement_block,
                                        LogicRef* state_next_var) const {
  if (next_state_ != nullptr) {
    statement_block->Add<BlockingAssignment>(state_next_var,
                                             next_state_->state_value());
  }
  for (const ConditionalFsmBlock& cond_block : conditional_blocks_) {
    if (cond_block.HasStateTransitions()) {
      Conditional* conditional = statement_block->Add<Conditional>(
          statement_block->parent(), cond_block.condition());
      cond_block.EmitConditionalStateTransitions(
          conditional, conditional->consequent(), state_next_var);
    }
  }
}

bool FsmBlockBase::HasAssignments() const {
  return !assignments_.empty() ||
         absl::c_any_of(conditional_blocks_, [](const ConditionalFsmBlock& b) {
           return b.HasAssignments();
         });
}

bool FsmBlockBase::HasStateTransitions() const {
  return next_state_ != nullptr ||
         absl::c_any_of(conditional_blocks_, [](const ConditionalFsmBlock& b) {
           return b.HasStateTransitions();
         });
}

bool FsmBlockBase::HasAssignmentToOutput(const FsmOutput& output) const {
  return (
      absl::c_any_of(
          assignments_,
          [&](const Assignment& a) { return a.lhs == output.logic_ref; }) ||
      absl::c_any_of(conditional_blocks_, [&](const ConditionalFsmBlock& b) {
        return b.HasAssignmentToOutput(output);
      }));
}

ConditionalFsmBlock& ConditionalFsmBlock::ElseOnCondition(
    Expression* condition) {
  XLS_CHECK(next_alternate_ == nullptr && final_alternate_ == nullptr);
  next_alternate_ = absl::make_unique<ConditionalFsmBlock>(
      absl::StrFormat("%s else (%s)", debug_name_, condition->Emit()), file_,
      condition);
  return *next_alternate_;
}

UnconditionalFsmBlock& ConditionalFsmBlock::Else() {
  XLS_CHECK(next_alternate_ == nullptr && final_alternate_ == nullptr);
  final_alternate_ = absl::make_unique<UnconditionalFsmBlock>(
      absl::StrFormat("%s else", debug_name_), file_);
  return *final_alternate_;
}

void ConditionalFsmBlock::EmitConditionalAssignments(
    Conditional* conditional, StatementBlock* statement_block) const {
  EmitAssignments(statement_block);
  if (next_alternate_ != nullptr && next_alternate_->HasAssignments()) {
    next_alternate_->EmitConditionalAssignments(
        conditional, conditional->AddAlternate(next_alternate_->condition()));
  } else if (final_alternate_ != nullptr &&
             final_alternate_->HasAssignments()) {
    final_alternate_->EmitAssignments(conditional->AddAlternate());
  }
}

void ConditionalFsmBlock::EmitConditionalStateTransitions(
    Conditional* conditional, StatementBlock* statement_block,
    LogicRef* state_next_var) const {
  EmitStateTransitions(statement_block, state_next_var);
  if (next_alternate_ != nullptr && next_alternate_->HasStateTransitions()) {
    next_alternate_->EmitConditionalStateTransitions(
        conditional, conditional->AddAlternate(next_alternate_->condition()),
        state_next_var);
  } else if (final_alternate_ != nullptr &&
             final_alternate_->HasStateTransitions()) {
    final_alternate_->EmitStateTransitions(conditional->AddAlternate(),
                                           state_next_var);
  }
}

bool ConditionalFsmBlock::HasAssignments() const {
  return FsmBlock<ConditionalFsmBlock>::HasAssignments() ||
         (next_alternate_ != nullptr && next_alternate_->HasAssignments()) ||
         (final_alternate_ != nullptr && final_alternate_->HasAssignments());
}

bool ConditionalFsmBlock::HasStateTransitions() const {
  return FsmBlock<ConditionalFsmBlock>::HasStateTransitions() ||
         (next_alternate_ != nullptr &&
          next_alternate_->HasStateTransitions()) ||
         (final_alternate_ != nullptr &&
          final_alternate_->HasStateTransitions());
}

bool ConditionalFsmBlock::HasAssignmentToOutput(const FsmOutput& output) const {
  return FsmBlock<ConditionalFsmBlock>::HasAssignmentToOutput(output) ||
         (next_alternate_ != nullptr &&
          next_alternate_->HasAssignmentToOutput(output)) ||
         (final_alternate_ != nullptr &&
          final_alternate_->HasAssignmentToOutput(output));
}

LogicRef* FsmBuilder::AddRegDef(absl::string_view name, Expression* width,
                                RegInit init) {
  defs_.push_back(module_->parent()->Make<RegDef>(name, width, init));
  return module_->parent()->Make<LogicRef>(defs_.back());
}

FsmCounter* FsmBuilder::AddDownCounter(absl::string_view name, int64 width) {
  LogicRef* ref = AddRegDef(name, file_->PlainLiteral(width));
  LogicRef* ref_next =
      AddRegDef(absl::StrCat(name, "_next"), file_->PlainLiteral(width));
  counters_.push_back(FsmCounter{ref, ref_next, width});
  return &counters_.back();
}

FsmOutput* FsmBuilder::AddOutputAsExpression(absl::string_view name,
                                             Expression* width,
                                             Expression* default_value) {
  RegInit init = UninitializedSentinel();
  if (default_value != nullptr) {
    init = default_value;
  }
  LogicRef* logic_ref = AddRegDef(name, width, init);
  outputs_.push_back(FsmOutput{logic_ref, default_value});
  return &outputs_.back();
}

FsmOutput* FsmBuilder::AddExistingOutput(LogicRef* logic_ref,
                                         Expression* default_value) {
  outputs_.push_back(FsmOutput{logic_ref, default_value});
  return &outputs_.back();
}

FsmRegister* FsmBuilder::AddRegisterAsExpression(absl::string_view name,
                                                 Expression* width,
                                                 Expression* reset_value) {
  // A reset value can only be specified if the FSM has a reset signal.
  XLS_CHECK(reset_value == nullptr || reset_.has_value());
  LogicRef* logic_ref = reset_value == nullptr
                            ? AddRegDef(name, width)
                            : AddRegDef(name, width, reset_value);
  LogicRef* logic_ref_next = AddRegDef(absl::StrCat(name, "_next"), width);
  registers_.push_back(FsmRegister{logic_ref, logic_ref_next, reset_value});
  return &registers_.back();
}

FsmRegister* FsmBuilder::AddRegister(absl::string_view name, int64 width,
                                     absl::optional<int64> reset_value) {
  return AddRegisterAsExpression(
      name, file_->PlainLiteral(width),
      reset_value.has_value() ? file_->PlainLiteral(*reset_value) : nullptr);
}

FsmRegister* FsmBuilder::AddRegister1(absl::string_view name,
                                      absl::optional<bool> reset_value) {
  return AddRegisterAsExpression(
      name, /*width=*/file_->PlainLiteral(1),
      reset_value.has_value() ? file_->PlainLiteral(*reset_value) : nullptr);
}

FsmRegister* FsmBuilder::AddExistingRegister(LogicRef* reg) {
  LogicRef* logic_ref_next =
      AddRegDef(absl::StrCat(reg->GetName(), "_next"), reg->def()->width());
  registers_.push_back(FsmRegister{reg, logic_ref_next});
  return &registers_.back();
}

FsmState* FsmBuilder::AddState(absl::string_view name) {
  if (state_local_param_ == nullptr) {
    state_local_param_ = file_->Make<LocalParam>(file_);
  }
  XLS_CHECK(!absl::c_any_of(states_, [&](const FsmState& s) {
    return s.name() == name;
  })) << absl::StrFormat("State with name \"%s\" already exists.", name);
  Expression* state_value = state_local_param_->AddItem(
      absl::StrCat("State", name), file_->PlainLiteral(states_.size()));
  states_.emplace_back(name, file_, state_value);
  return &states_.back();
}

absl::Status FsmBuilder::BuildStateTransitionLogic(LogicRef* state,
                                                   LogicRef* state_next) {
  // Construct an always block encapsulating the combinational logic for
  // determining state transitions.
  module_->Add<BlankLine>();
  module_->Add<Comment>("FSM state transition logic.");
  AlwaysBase* ac;
  if (use_system_verilog_) {
    ac = module_->Add<AlwaysComb>(file_);
  } else {
    ac = module_->Add<Always>(file_, std::vector<SensitivityListElement>(
                                         {ImplicitEventExpression()}));
  }

  // Set default assignments.
  ac->statements()->Add<BlockingAssignment>(state_next, state);

  Case* case_statement = ac->statements()->Add<Case>(file_, state);
  for (const auto& fsm_state : states_) {
    fsm_state.EmitStateTransitions(
        case_statement->AddCaseArm(fsm_state.state_value()), state_next);
  }
  // If the number of states is not a power of two then add an unreachable
  // default case which sets the next state to X. This ensures all values of
  // the case are covered.
  if (states_.size() != 1 << StateRegisterWidth()) {
    StatementBlock* statements = case_statement->AddCaseArm(DefaultSentinel());
    statements->Add<BlockingAssignment>(
        state_next, file_->Make<XSentinel>(StateRegisterWidth()));
  }
  return absl::OkStatus();
}

UnconditionalFsmBlock& ConditionalFsmBlock::FindOrAddFinalAlternate() {
  ConditionalFsmBlock* alternate = this;
  while (alternate->next_alternate_ != nullptr) {
    alternate = alternate->next_alternate_.get();
  }
  if (alternate->final_alternate_ != nullptr) {
    return *alternate->final_alternate_;
  }
  return alternate->Else();
}

absl::Status FsmBlockBase::AddDefaultOutputAssignments(
    const FsmOutput& output) {
  XLS_VLOG(3) << absl::StreamFormat(
      "AddDefaultOutputAssignments for output %s in block \"%s\"",
      output.logic_ref->GetName(), debug_name());
  // The count of the assignments along any code path through the block.
  int64 assignment_count = 0;
  for (const Assignment& assignment : assignments_) {
    if (assignment.lhs == output.logic_ref) {
      XLS_VLOG(3) << absl::StreamFormat(
          "Output is unconditionally assigned %s in block \"%s\"",
          assignment.rhs->Emit(), debug_name());
      assignment_count++;
    }
  }
  for (ConditionalFsmBlock& cond_block : conditional_blocks_) {
    if (!cond_block.HasAssignmentToOutput(output)) {
      continue;
    }
    // The conditional block has an assignment to the output somewhere beneath
    // it. Make sure there is an assignment on each alternate of the
    // conditional.
    XLS_VLOG(3) << "Conditional block " << cond_block.debug_name()
                << " assigns output " << output.logic_ref->GetName();
    cond_block.FindOrAddFinalAlternate();
    XLS_RETURN_IF_ERROR(cond_block.ForEachAlternate(
        [&](FsmBlockBase* alternate) -> absl::Status {
          return alternate->AddDefaultOutputAssignments(output);
        }));
    assignment_count++;
  }
  if (assignment_count > 1) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Output \"%s\" may be assigned more than once along a code path.",
        output.logic_ref->GetName()));
  }
  if (assignment_count == 0) {
    XLS_VLOG(3) << absl::StreamFormat(
        "Adding assignment of %s to default value %s in block \"%s\"",
        output.logic_ref->Emit(), output.default_value->Emit(), debug_name());
    AddAssignment(output.logic_ref, output.default_value);
  }
  return absl::OkStatus();
}

absl::Status ConditionalFsmBlock::ForEachAlternate(
    std::function<absl::Status(FsmBlockBase*)> f) {
  for (ConditionalFsmBlock* alternate = this; alternate != nullptr;
       alternate = alternate->next_alternate_.get()) {
    XLS_RETURN_IF_ERROR(f(alternate));
    if (alternate->final_alternate_ != nullptr) {
      return f(alternate->final_alternate_.get());
    }
  }
  return absl::OkStatus();
}

absl::Status FsmBlockBase::RemoveAssignment(LogicRef* logic_ref) {
  int64 size_before = assignments_.size();
  assignments_.erase(
      std::remove_if(assignments_.begin(), assignments_.end(),
                     [&](const Assignment& a) { return a.lhs == logic_ref; }),
      assignments_.end());
  if (size_before == assignments_.size()) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Assignment to %s does not exist in block \"%s\".",
                        logic_ref->GetName(), debug_name()));
  }
  if (size_before > assignments_.size() + 1) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Multiple assignment to %s exist in block \"%s\".",
                        logic_ref->GetName(), debug_name()));
  }
  return absl::OkStatus();
}

namespace {

// Returns true iff the given expressions are all non-null and the same.
// Sameness means same pointer value or expressions are literals with same
// underlying Bits value. Literals require this special handled because literals
// are typically created on the fly for each use. So a 1'h1 in one part of the
// code will generally not refer to the same Literal object as a 1'h1 in another
// part of the code.
bool AllSameAndNonNull(absl::Span<Expression* const> exprs) {
  Expression* same_expr = nullptr;
  for (Expression* expr : exprs) {
    if (expr == nullptr) {
      return false;
    }
    if (same_expr == nullptr) {
      same_expr = expr;
      continue;
    }
    if (expr != same_expr && !(expr->IsLiteral() && same_expr->IsLiteral() &&
                               expr->AsLiteralOrDie()->bits() ==
                                   same_expr->AsLiteralOrDie()->bits())) {
      return false;
    }
  }
  return true;
}

}  // namespace

xabsl::StatusOr<Expression*> FsmBlockBase::HoistCommonConditionalAssignments(
    const FsmOutput& output) {
  XLS_VLOG(3) << absl::StreamFormat(
      "HoistCommonConditionalAssignments for output %s in block \"%s\"",
      output.logic_ref->GetName(), debug_name());

  for (const Assignment& assignment : assignments_) {
    if (assignment.lhs == output.logic_ref) {
      XLS_VLOG(3) << absl::StreamFormat(
          "Output is unconditionally assigned %s in block \"%s\"",
          assignment.rhs->Emit(), debug_name());
      return assignment.rhs;
    }
  }

  for (ConditionalFsmBlock& cond_block : conditional_blocks_) {
    if (!cond_block.HasAssignmentToOutput(output)) {
      continue;
    }
    XLS_VLOG(3) << absl::StreamFormat(
        "Conditional block \"%s\" assigns output %s", cond_block.debug_name(),
        output.logic_ref->GetName());
    std::vector<Expression*> rhses;
    XLS_RETURN_IF_ERROR(cond_block.ForEachAlternate(
        [&](FsmBlockBase* alternate) -> absl::Status {
          XLS_ASSIGN_OR_RETURN(
              Expression * rhs,
              alternate->HoistCommonConditionalAssignments(output));
          XLS_VLOG(3) << absl::StreamFormat(
              "Alternate block \"%s\" assigns output %s to %s",
              cond_block.debug_name(), output.logic_ref->GetName(),
              rhs == nullptr ? "nullptr" : rhs->Emit());
          rhses.push_back(rhs);
          return absl::OkStatus();
        }));
    if (!AllSameAndNonNull(rhses)) {
      XLS_VLOG(3) << absl::StreamFormat(
          "Not all conditional block assign output %s to same value",
          output.logic_ref->GetName());
      return nullptr;
    }

    XLS_VLOG(3) << absl::StreamFormat(
        "Conditional block assigns output %s to same value %s on all "
        "alternates",
        output.logic_ref->GetName(), rhses.front()->Emit());
    XLS_RETURN_IF_ERROR(cond_block.ForEachAlternate(
        [&](FsmBlockBase* alternate) -> absl::Status {
          return alternate->RemoveAssignment(output.logic_ref);
        }));
    AddAssignment(output.logic_ref, rhses.front());
    return rhses.front();
  }
  return nullptr;
}

absl::Status FsmBuilder::BuildOutputLogic(LogicRef* state) {
  if (registers_.empty() && outputs_.empty() && counters_.empty()) {
    return absl::OkStatus();
  }

  // Construct an always block encapsulating the combinational logic for
  // determining output values, next counter values, and next assignment values.
  module_->Add<BlankLine>();
  module_->Add<Comment>("FSM output logic.");

  AlwaysBase* ac;
  if (use_system_verilog_) {
    ac = module_->Add<AlwaysComb>(file_);
  } else {
    ac = module_->Add<Always>(file_, std::vector<SensitivityListElement>(
                                         {ImplicitEventExpression()}));
  }

  for (const FsmRegister& reg : registers_) {
    ac->statements()->Add<BlockingAssignment>(reg.next, reg.logic_ref);
  }

  // For each state there should exactly one assignment to each output along any
  // code path. This prevents infinite looping during simulation caused by the
  // "glitch" of assigning a value twice to the reg. See:
  // https://github.com/steveicarus/iverilog/issues/321. This single assignment
  // propertry is achieved by sinking the assignments to the default value into
  // conditional blocks to exactly cover those code paths which have no
  // assignment. Flopped regs such as the next state and next counter values do
  // not need this treatment because their value only changes on clock edges
  // which avoids any propagate of a multi-assignment glitch during simulation.
  for (FsmState& fsm_state : states_) {
    XLS_VLOG(3) << "Adding default assignments for state " << fsm_state.name();
    for (const FsmOutput& output : outputs_) {
      XLS_RETURN_IF_ERROR(fsm_state.AddDefaultOutputAssignments(output));
      XLS_RETURN_IF_ERROR(
          fsm_state.HoistCommonConditionalAssignments(output).status());
    }
  }

  for (const FsmCounter& counter : counters_) {
    ac->statements()->Add<BlockingAssignment>(
        counter.next, file_->Sub(counter.logic_ref, file_->PlainLiteral(1)));
  }
  Case* case_statement = ac->statements()->Add<Case>(file_, state);
  for (const auto& fsm_state : states_) {
    fsm_state.EmitAssignments(
        case_statement->AddCaseArm(fsm_state.state_value()));
  }

  // If the state vector is wide enough to allow values not encoded in the state
  // enum add a default case and assign outputs to the default value.
  if (states_.size() != 1 << StateRegisterWidth()) {
    StatementBlock* statements = case_statement->AddCaseArm(DefaultSentinel());
    for (const FsmOutput& output : outputs_) {
      statements->Add<BlockingAssignment>(output.logic_ref,
                                          output.default_value);
    }
  }

  return absl::OkStatus();
}

absl::Status FsmBuilder::Build() {
  if (is_built_) {
    return absl::InternalError("FSM has already been built.");
  }
  is_built_ = true;

  module_->Add<BlankLine>();
  module_->Add<Comment>(absl::StrCat(name_, " ", "FSM:"));

  LocalParamItemRef* state_bits = module_->Add<LocalParam>(file_)->AddItem(
      "StateBits", file_->PlainLiteral(StateRegisterWidth()));

  // For each state, define its numeric encoding as a LocalParam gathered in
  // state_values.
  module_->AddModuleMember(state_local_param_);

  Expression* initial_state_value = states_.front().state_value();
  LogicRef* state =
      module_->AddRegAsExpression("state", state_bits, initial_state_value);
  LogicRef* state_next = module_->AddRegAsExpression("state_next", state_bits,
                                                     initial_state_value);
  for (RegDef* def : defs_) {
    module_->AddModuleMember(def);
  }

  XLS_RETURN_IF_ERROR(BuildStateTransitionLogic(state, state_next));
  XLS_RETURN_IF_ERROR(BuildOutputLogic(state));

  AlwaysFlop* af = module_->Add<AlwaysFlop>(file_, clk_, reset_);
  if (reset_.has_value()) {
    af->AddRegister(state, state_next, /*reset_value=*/initial_state_value);
  } else {
    af->AddRegister(state, state_next);
  }
  for (const FsmRegister& reg : registers_) {
    af->AddRegister(reg.logic_ref, reg.next, /*reset_value=*/reg.reset_value);
  }
  for (const FsmCounter& counter : counters_) {
    af->AddRegister(counter.logic_ref, counter.next);
  }

  return absl::OkStatus();
}

}  // namespace verilog
}  // namespace xls
