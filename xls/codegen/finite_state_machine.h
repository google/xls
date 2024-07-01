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

#ifndef XLS_CODEGEN_FINITE_STATE_MACHINE_H_
#define XLS_CODEGEN_FINITE_STATE_MACHINE_H_

#include <algorithm>
#include <cstdint>
#include <functional>
#include <list>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xls/codegen/vast/vast.h"
#include "xls/common/casts.h"
#include "xls/ir/source_location.h"

namespace xls {
namespace verilog {

// Encapsulates an output signal driven by finite state machine logic.
struct FsmOutput {
  LogicRef* logic_ref;
  Expression* default_value;
};

// Encapsulates a registered output signal driven by finite state machine
// logic.
struct FsmRegister {
  LogicRef* logic_ref;

  // The value of the register in the next cycle.
  LogicRef* next;

  // The expression defining the reset state of the register. May be null.
  Expression* reset_value;
};

// Represents a down-counting cycle counter controlled by the finite state
// machine logic.
struct FsmCounter {
  // The value of the counter.
  LogicRef* logic_ref;

  // The value of the counter in the next cycle.
  LogicRef* next;

  // Width of the counter in bits.
  int64_t width;
};

// A single assignment of an FSM output to a value.
struct Assignment {
  LogicRef* lhs;
  Expression* rhs;
};

class FsmState;
class ConditionalFsmBlock;

// Abstraction representing a control-flow-equivalent block of logic in an FSM
// (i.e., a basic block).
class FsmBlockBase {
 public:
  explicit FsmBlockBase(std::string_view debug_name, VerilogFile* file)
      : debug_name_(debug_name), file_(file) {}
  virtual ~FsmBlockBase() = default;

  // Returns true if this block has any output assignments (or state
  // transitions). This includes any assignments (state transitions) in nested
  // conditional blocks.
  virtual bool HasAssignments() const;
  virtual bool HasStateTransitions() const;

  // Returns true if this block may assign a value to the given output. This
  // includes any assignments in nested conditional blocks.
  virtual bool HasAssignmentToOutput(const FsmOutput& output) const;

  // Emits the output assignments contained in this block as blocking
  // assignments in the given VAST StatementBlock including any nested
  // conditional assignments.
  void EmitAssignments(StatementBlock* statement_block) const;

  // Emits the state transition (if any) contained in this block as a blocking
  // assignment in the given VAST StatementBlock including any nested state
  // transitions.
  void EmitStateTransitions(StatementBlock* statement_block,
                            LogicRef* state_next_var) const;

 protected:
  friend class FsmBuilder;

  std::string debug_name() const { return debug_name_; }

  // Adds the assignment of 'logic_ref' to 'value' to the block.
  void AddAssignment(LogicRef* logic_ref, Expression* value) {
    for (const auto& assignment : assignments_) {
      CHECK_NE(logic_ref, assignment.lhs)
          << logic_ref->GetName() << " already assigned.";
    }
    assignments_.push_back(Assignment{logic_ref, value});
  }

  // Remove the assignment of the given LogicRef from the unconditional
  // assignments in this block. Does not recurse into contained conditional
  // blocks. Returns an error if there is not exactly one assignment to the
  // LogicRef in the block.
  absl::Status RemoveAssignment(LogicRef* logic_ref);

  // Adds assignments of the given output to its default value along all code
  // paths which do not have an assignment of the output. Returns an error if
  // the output may be assigned more than once on any code path. Upon completion
  // of this function, the output will be assigned exactly once on all code
  // paths through the block.
  absl::Status AddDefaultOutputAssignments(const FsmOutput& output);

  // Hoists conditional assignments which are identical along all paths in the
  // block into a single assignment. For example, given:
  //
  // if (foo) begin
  //   a = 1;
  //   b = 1;
  // end else begin
  //   a = 1;
  //   b = 0;
  // end
  //
  // After hoisting the code will look like:
  //
  // a = 1;
  // if (foo) begin
  //   b = 1;
  // end else begin
  //   b = 0;
  // end
  //
  absl::StatusOr<Expression*> HoistCommonConditionalAssignments(
      const FsmOutput& output);

  // An name which is used to uniquely identify the block in log messages. The
  // name does not affect the emitted Verilog.
  std::string debug_name_;
  VerilogFile* file_;

  FsmState* next_state_ = nullptr;
  std::vector<Assignment> assignments_;

  // The conditional blocks within this block (if any). This lowers to a
  // sequence of 'if' statements. A std::list is used for pointer stability.
  std::list<ConditionalFsmBlock> conditional_blocks_;
};

// Base class for curiously recurring template pattern to support polymorphic
// chaining. This class holds fluent style methods for constructing the
// finite state machine.
template <typename T>
class FsmBlock : public FsmBlockBase {
 public:
  explicit FsmBlock(std::string_view debug_name, VerilogFile* file)
      : FsmBlockBase(debug_name, file) {}
  ~FsmBlock() override = default;

  // Sets the next state to transition to.
  T& NextState(FsmState* next_state) {
    CHECK(next_state_ == nullptr);
    next_state_ = next_state;
    return down_cast<T&>(*this);
  }

  // Sets the given output to the given value. This occurs immediately and
  // asynchronously.
  T& SetOutput(FsmOutput* output, int64_t value) {
    return SetOutputAsExpression(output,
                                 file_->PlainLiteral(value, SourceInfo()));
  }
  T& SetOutputAsExpression(FsmOutput* output, Expression* value) {
    AddAssignment(output->logic_ref, value);
    return down_cast<T&>(*this);
  }

  // Sets the given register to the given value in the next cycle.
  T& SetRegisterNext(FsmRegister* reg, int64_t value) {
    return SetRegisterNextAsExpression(
        reg, file_->PlainLiteral(value, SourceInfo()));
  }
  T& SetRegisterNextAsExpression(FsmRegister* reg, Expression* value) {
    AddAssignment(reg->next, value);
    return down_cast<T&>(*this);
  }

  // Sets the given counter to the given value in the next cycle.
  T& SetCounter(FsmCounter* counter, int64_t value) {
    return SetCounterAsExpression(counter,
                                  file_->PlainLiteral(value, SourceInfo()));
  }
  T& SetCounterAsExpression(FsmCounter* counter, Expression* value) {
    AddAssignment(counter->next, value);
    return down_cast<T&>(*this);
  }

  // Adds a conditional statement using the given condition to the block.
  // Returns the resulting conditional block.
  ConditionalFsmBlock& OnCondition(Expression* condition) {
    conditional_blocks_.emplace_back(
        absl::StrFormat("%s: if (%s)", debug_name(), condition->Emit(nullptr)),
        file_, condition);
    return conditional_blocks_.back();
  }

  // Adds a conditional statement based on the given counter equal to
  // zero. Returns the resulting conditional block.
  ConditionalFsmBlock& OnCounterIsZero(FsmCounter* counter) {
    conditional_blocks_.emplace_back(
        absl::StrFormat("%s: if counter %s == 0", debug_name(),
                        counter->logic_ref->GetName()),
        file_,
        file_->Equals(counter->logic_ref, file_->PlainLiteral(0, SourceInfo()),
                      SourceInfo()));
    return conditional_blocks_.back();
  }
};

// An unconditional block of logic within an FSM state.
class UnconditionalFsmBlock : public FsmBlock<UnconditionalFsmBlock> {
 public:
  explicit UnconditionalFsmBlock(std::string_view debug_name, VerilogFile* file)
      : FsmBlock<UnconditionalFsmBlock>(debug_name, file) {}
};

// A conditional block of logic within an FSM state.
class ConditionalFsmBlock : public FsmBlock<ConditionalFsmBlock> {
 public:
  explicit ConditionalFsmBlock(std::string_view debug_name, VerilogFile* file,
                               Expression* condition)
      : FsmBlock<ConditionalFsmBlock>(debug_name, file),
        condition_(condition) {}

  // Appends an "else if" to the conditional ladder. Returns the resulting
  // conditional block.
  ConditionalFsmBlock& ElseOnCondition(Expression* condition);

  // Terminates the conditional ladder with an "else". Returns the resulting
  // block.
  UnconditionalFsmBlock& Else();

  Expression* condition() const { return condition_; }

  bool HasAssignments() const override;
  bool HasStateTransitions() const override;
  bool HasAssignmentToOutput(const FsmOutput& output) const override;

 protected:
  friend class FsmBlockBase;

  // Emits the VAST conditional ladder for this conditional block and the nested
  // assignments (state transitions). 'conditional' is the VAST conditional
  // statement corresponding to this conditional block.
  void EmitConditionalAssignments(Conditional* conditional,
                                  StatementBlock* statement_block) const;
  void EmitConditionalStateTransitions(Conditional* conditional,
                                       StatementBlock* statement_block,
                                       LogicRef* state_next_var) const;

  // Calls the given function on each alternate in this conditional block. For
  // example, if the conditional block represents:
  //
  // if (a) begin
  //   ...Block A...
  // end else if (b) begin
  //   ...Block B...
  // end else begin
  //   ...Block C...
  // end
  //
  // The function will be called on blocks A, B, and C.
  absl::Status ForEachAlternate(std::function<absl::Status(FsmBlockBase*)> f);

  // If the conditional block has a final alternate (unconditional else block)
  // then it is returned. Otherwise a final alternate is created and returned.
  UnconditionalFsmBlock& FindOrAddFinalAlternate();

  Expression* condition_;

  // The next alternate (else if) of the conditional ladder. Only one of
  // next_alternate_ or final_alternate_ may be non-null. Might be representable
  // as a std::variant but a std::variant of std::unique_ptrs is awkward to
  // manipulate.
  std::unique_ptr<ConditionalFsmBlock> next_alternate_;

  // The final alternate (else) of the conditional ladder.
  std::unique_ptr<UnconditionalFsmBlock> final_alternate_;
};

// Abstraction representing a state in the FSM. For convenience derives from
// UnconditionalFsmBlock which exposes the UnconditionalFsmBlock interface (eg,
// NextState). This enables code like the following:
//
//  auto st = fsm.NewState(...);
//  st->SetOutput(x, value).NextState(next_st);
class FsmState : public UnconditionalFsmBlock {
 public:
  explicit FsmState(std::string_view name, VerilogFile* file,
                    Expression* state_value)
      : UnconditionalFsmBlock(name, file),
        name_(name),
        state_value_(state_value) {}

  std::string name() const { return name_; }

  // The VAST expression of the numeric encoding of this state in the FSM state
  // variable.
  Expression* state_value() const { return state_value_; }

 protected:
  std::string name_;
  Expression* state_value_;
};

// Abstraction for building finite state machines in Verilog using VAST.
class FsmBuilder {
 public:
  FsmBuilder(std::string_view name, Module* module, LogicRef* clk,
             bool use_system_verilog, std::optional<Reset> reset = std::nullopt)
      : name_(name),
        module_(module),
        file_(module->file()),
        clk_(clk),
        use_system_verilog_(use_system_verilog),
        reset_(reset) {}

  // Adds an FSM-controled signal to the FSM with the given name. A RegDef named
  // 'name' is added to the module.
  FsmOutput* AddOutput(std::string_view name, int64_t width,
                       int64_t default_value) {
    return AddOutputAsExpression(
        name, file_->BitVectorType(width, SourceInfo()),
        file_->PlainLiteral(default_value, SourceInfo()));
  }
  FsmOutput* AddOutputAsExpression(std::string_view name, DataType* data_type,
                                   Expression* default_value);

  FsmOutput* AddOutput1(std::string_view name, int64_t default_value) {
    return AddOutputAsExpression(
        name, /*data_type=*/file_->ScalarType(SourceInfo()),
        file_->PlainLiteral(default_value, SourceInfo()));
  }

  // Overload which adds a previously defined reg as a FSM-controlled signal.
  FsmOutput* AddExistingOutput(LogicRef* logic_ref, Expression* default_value);

  // Adds a FSM-driven register with the given name. RegDefs named 'name' and
  // 'name_next' are added to the module.  The state of the register is affected
  // by calling SetRegisterNext.
  FsmRegister* AddRegister(std::string_view name, int64_t width,
                           std::optional<int64_t> reset_value = std::nullopt);
  FsmRegister* AddRegister(std::string_view name, DataType* data_type,
                           Expression* reset_value = nullptr);

  // Overload which adds a previously defined reg as an FSM-controlled signal. A
  // RegDef named "*_next" is added to the module where "*" is the name of the
  // given LogicRef.
  FsmRegister* AddExistingRegister(LogicRef* reg);

  // Add a cycle down-counter with the given name and width.
  FsmCounter* AddDownCounter(std::string_view name, int64_t width);

  // Add a new state to the FSM.
  FsmState* AddState(std::string_view name);

  // Builds the FSM in the module.
  absl::Status Build();

  // Configures FSM so that on reset, the state will be set to reset_state.
  void SetResetState(FsmState* reset_state) { reset_state_ = reset_state; }

 private:
  // Creates a RegDef of the given type and optional initial
  // value. Returns a LogicRef referring to it. The RegDef is added to the
  // module inline with the FSM logic when Build is called.
  LogicRef* AddRegDef(std::string_view name, DataType* data_type,
                      Expression* init = nullptr);

  // Build the always block containing the logic for state transitions.
  absl::Status BuildStateTransitionLogic(LogicRef* state, LogicRef* state_next);

  // Build the always block containing the logic for the FSM outputs.
  absl::Status BuildOutputLogic(LogicRef* state);

  // Returns the state register width.
  int64_t StateRegisterWidth() const {
    return std::max(int64_t{1}, Bits::MinBitCountUnsigned(states_.size() - 1));
  }

  std::string name_;
  Module* module_;
  VerilogFile* file_;
  LogicRef* clk_;
  bool use_system_verilog_;
  std::optional<Reset> reset_;

  // Output and registers defined by the FSM prior to called Build (such as
  // AddOutput and AddRegister). These are added to the module when Build is
  // called. Delaying insertion of the RegDefs enables placing them inline with
  // the rest of the FSM logic.
  std::vector<RegDef*> defs_;

  // The localparam statement defining the concrete values for each state.
  LocalParam* state_local_param_ = nullptr;

  // Whether the build method has been called on this FsmBuilder. The build
  // method may only be called once.
  bool is_built_ = false;

  FsmState* reset_state_ = nullptr;
  std::list<FsmState> states_;
  std::list<FsmCounter> counters_;
  std::list<FsmOutput> outputs_;
  std::list<FsmRegister> registers_;
};

}  // namespace verilog
}  // namespace xls

#endif  // XLS_CODEGEN_FINITE_STATE_MACHINE_H_
