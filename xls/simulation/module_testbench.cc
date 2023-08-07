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

#include "xls/simulation/module_testbench.h"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/types/variant.h"
#include "xls/codegen/vast.h"
#include "xls/common/logging/log_lines.h"
#include "xls/common/logging/logging.h"
#include "xls/common/source_location.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/common/visitor.h"
#include "xls/ir/bits.h"
#include "xls/ir/bits_ops.h"
#include "xls/ir/number_parser.h"
#include "xls/simulation/verilog_simulator.h"
#include "re2/re2.h"

namespace xls {
namespace verilog {
namespace {

// The number of cycles that the design under test (DUT) is being reset.
static constexpr int64_t kResetCycles = 5;

// Clock period in Verilog time units. Must be even number greater than 3.
static constexpr int64_t kClockPeriod = 10;
static_assert(kClockPeriod > 3);
static_assert(kClockPeriod % 2 == 0);

// Upper limit on the length of simulation. Simulation terminates with
// $finish after this many cycles. This is to avoid test timeouts in cases
// that the simulation logic does not terminate.
static constexpr int64_t kSimulationCycleLimit = 10000;

static_assert(kResetCycles < kSimulationCycleLimit,
              "Reset cycles must be less than the simulation cycle limit.");

// Name of the testbench internal signal which is asserted in the last cycle
// that the DUT is in reset. This can be used to trigger the driving of signals
// for the first cycle out of reset.
static constexpr std::string_view kLastResetCycleSignal =
    "__last_cycle_of_reset";

std::string GetTimeoutMessage() {
  return absl::StrFormat("ERROR: timeout, simulation ran too long (%d cycles).",
                         kSimulationCycleLimit);
}

std::string ToString(const xabsl::SourceLocation& loc) {
  return absl::StrFormat("%s:%d", loc.file_name(), loc.line());
}

// Emits a Verilog delay statement (e.g., `#42;`).
void EmitDelay(StatementBlock* statement_block, int32_t amount) {
  statement_block->Add<DelayStatement>(
      SourceInfo(),
      statement_block->file()->PlainLiteral(amount, SourceInfo()));
}

// Emits a Verilog statement which waits for the clock posedge.
void WaitForClockPosEdge(LogicRef* clk, StatementBlock* statement_block) {
  VerilogFile& file = *statement_block->file();
  Expression* posedge_clk = file.Make<PosEdge>(SourceInfo(), clk);
  statement_block->Add<EventControl>(SourceInfo(), posedge_clk);
}

// Insert statements into statement block which delay the simulation for the
// given number of cycles. Regardless of delay, simulation resumes one time unit
// after the posedge of the clock.
void WaitNCycles(LogicRef* clk, StatementBlock* statement_block,
                 int64_t n_cycles) {
  XLS_CHECK_GT(n_cycles, 0);
  XLS_CHECK_NE(statement_block, nullptr);
  VerilogFile& file = *statement_block->file();
  Expression* posedge_clk = file.Make<PosEdge>(SourceInfo(), clk);
  if (n_cycles == 1) {
    WaitForClockPosEdge(clk, statement_block);
  } else {
    statement_block->Add<RepeatStatement>(
        SourceInfo(), file.PlainLiteral(n_cycles, SourceInfo()),
        file.Make<EventControl>(SourceInfo(), posedge_clk));
  }
  EmitDelay(statement_block, 1);
}

// Emit $display statements into the given block which print the value of the
// given signals.
void EmitDisplayStatements(
    absl::Span<const SignalCapture> signal_captures,
    StatementBlock* statement_block,
    const absl::flat_hash_map<std::string, LogicRef*>& signal_refs) {
  for (const SignalCapture& signal_capture : signal_captures) {
    statement_block->Add<Display>(
        SourceInfo(),
        std::vector<Expression*>{
            statement_block->file()->Make<QuotedString>(
                SourceInfo(), absl::StrFormat("%%t OUTPUT %s = %d'h%%0x (#%d)",
                                              signal_capture.signal.name,
                                              signal_capture.signal.width,
                                              signal_capture.instance_id)),
            statement_block->file()->Make<SystemFunctionCall>(SourceInfo(),
                                                              "time"),
            signal_refs.at(signal_capture.signal.name)});
  }
}

// Emits Verilog implementing the given action.
void EmitAction(
    const Action& action, StatementBlock* statement_block, LogicRef* clk,
    const absl::flat_hash_map<std::string, LogicRef*>& signal_refs) {
  // Inserts a delay which waits until one unit before the clock
  // posedge. Assumes that simulation is currently one time unit after posedge
  // of clock which is the point at which every action starts.
  auto delay_to_right_before_posedge = [&](StatementBlock* sb) {
    EmitDelay(sb, kClockPeriod - 2);
  };
  VerilogFile& file = *statement_block->file();
  statement_block->Add<BlankLine>(SourceInfo());
  Visitor visitor{
      [&](const AdvanceCycle& a) {
        statement_block->Add<Comment>(
            SourceInfo(), absl::StrFormat("Wait %d cycle(s).", a.amount));
        if (a.amount > 1) {
          WaitNCycles(clk, statement_block, a.amount - 1);
        }
        if (a.end_of_cycle_event != nullptr &&
            !a.end_of_cycle_event->signal_captures().empty()) {
          // Capture signals one time unit before posedge of the clock.
          delay_to_right_before_posedge(statement_block);
          EmitDisplayStatements(a.end_of_cycle_event->signal_captures(),
                                statement_block, signal_refs);
        }
        WaitForClockPosEdge(clk, statement_block);
        EmitDelay(statement_block, 1);
      },
      [&](const SetSignal& s) {
        if (s.value.bit_count() > 0) {
          statement_block->Add<BlockingAssignment>(
              SourceInfo(), signal_refs.at(s.signal_name),
              file.Literal(s.value, SourceInfo()));
        }
      },
      [&](const SetSignalX& s) {
        if (s.width > 0) {
          statement_block->Add<BlockingAssignment>(
              SourceInfo(), signal_refs.at(s.signal_name),
              file.Make<XSentinel>(SourceInfo(), s.width));
        }
      },
      [&](const WaitForSignals& w) {
        // WaitForSignal waits until the signal equals a certain value at the
        // posedge of the clock. Use a while loop to sample every cycle at
        // the posedge of the clock.
        // TODO(meheff): If we switch to SystemVerilog this is better handled
        // using a clocking block.
        statement_block->Add<Comment>(SourceInfo(), w.comment);
        Expression* cond = w.any_or_all == AnyOrAll::kAll
                               ? file.Literal1(1, SourceInfo())
                               : file.Literal1(0, SourceInfo());
        for (const SignalValue& signal_value : w.signal_values) {
          Expression* element;
          if (std::holds_alternative<Bits>(signal_value.value)) {
            element = file.Equals(
                signal_refs.at(signal_value.signal_name),
                file.Literal(std::get<Bits>(signal_value.value), SourceInfo()),
                SourceInfo());
          } else if (std::holds_alternative<IsX>(signal_value.value)) {
            // To test whether any bit is X do an XOR reduce of the bits. If any
            // bit is X the result will be X.
            element = file.EqualsX(
                file.XorReduce(signal_refs.at(signal_value.signal_name),
                               SourceInfo()),
                SourceInfo());
          } else {
            XLS_CHECK(std::holds_alternative<IsNotX>(signal_value.value));
            // To test whether all bits are not X do an XOR reduce of the
            // bits and test that it does not equal X.
            element = file.NotEqualsX(
                file.XorReduce(signal_refs.at(signal_value.signal_name),
                               SourceInfo()),
                SourceInfo());
          }
          cond = w.any_or_all == AnyOrAll::kAll
                     ? file.LogicalAnd(cond, element, SourceInfo())
                     : file.LogicalOr(cond, element, SourceInfo());
        }
        // Always sample signals on the posedge of the clock minus one unit.
        delay_to_right_before_posedge(statement_block);
        auto whle = statement_block->Add<WhileStatement>(
            SourceInfo(), file.LogicalNot(cond, SourceInfo()));
        // Test condition once per clock right before the posedge.
        EmitDelay(whle->statements(), kClockPeriod);

        // Currently at the posedge of the clock minus one unit. Emit any
        // display statements.
        if (w.end_of_cycle_event != nullptr) {
          EmitDisplayStatements(w.end_of_cycle_event->signal_captures(),
                                statement_block, signal_refs);
        }

        // Every action should terminate one unit after the posedge of the
        // clock.
        WaitForClockPosEdge(clk, statement_block);
        EmitDelay(statement_block, 1);
      }};
  absl::visit(visitor, action);
}

}  // namespace

TestbenchMetadata::TestbenchMetadata(const ModuleSignature& signature) {
  dut_module_name_ = signature.module_name();

  auto add_input_signal = [&](std::string_view name, int64_t width) {
    XLS_CHECK_OK(AddSignal(name, width, TestbenchSignalType::kInputPort));
  };
  auto add_output_signal = [&](std::string_view name, int64_t width) {
    XLS_CHECK_OK(AddSignal(name, width, TestbenchSignalType::kOutputPort));
  };

  if (signature.proto().has_clock_name()) {
    clk_name_ = signature.proto().clock_name();
    add_input_signal(signature.proto().clock_name(), 1);
  }
  if (signature.proto().has_reset()) {
    reset_proto_ = signature.proto().reset();
    add_input_signal(signature.proto().reset().name(), 1);
  }

  for (const PortProto& port : signature.data_inputs()) {
    add_input_signal(port.name(), port.width());
  }
  for (const PortProto& port : signature.data_outputs()) {
    add_output_signal(port.name(), port.width());
  }

  if (signature.proto().has_pipeline() &&
      signature.proto().pipeline().has_pipeline_control()) {
    // Module has pipeline register control.
    if (signature.proto().pipeline().pipeline_control().has_valid()) {
      // Add the valid input and optional valid output signals.
      const ValidProto& valid =
          signature.proto().pipeline().pipeline_control().valid();
      add_input_signal(valid.input_name(), 1);
      if (!valid.output_name().empty()) {
        add_output_signal(valid.output_name(), 1);
      }
    } else {
      XLS_CHECK(!signature.proto().pipeline().pipeline_control().has_manual())
          << "Manual register control not supported";
    }
  }
}

absl::Status TestbenchMetadata::CheckIsReadableSignal(
    std::string_view name) const {
  if (!signals_by_name_.contains(name)) {
    return absl::NotFoundError(absl::StrFormat(
        "`%s` is not a signal of module `%s`", name, dut_module_name()));
  }
  if (clk_name_.has_value() && clk_name_.value() == name) {
    return absl::InternalError(
        absl::StrFormat("Clock signal `%s` is readable", name));
  }
  return absl::OkStatus();
}

absl::Status TestbenchMetadata::AddSignal(std::string_view name, int64_t width,
                                          TestbenchSignalType type) {
  XLS_RET_CHECK(!signals_by_name_.contains(name))
      << absl::StrFormat("Signal `%s` already exists", name);
  signals_.push_back(
      TestbenchSignal{.name = std::string{name}, .width = width, .type = type});
  signals_by_name_[name] = signals_.size() - 1;
  return absl::OkStatus();
}

ModuleTestbenchThread& ModuleTestbenchThread::NextCycle() {
  return AdvanceNCycles(1);
}

ModuleTestbenchThread& ModuleTestbenchThread::AdvanceNCycles(int64_t n_cycles) {
  actions_.push_back(AdvanceCycle{n_cycles});
  return *this;
}

ModuleTestbenchThread& ModuleTestbenchThread::WaitForCycleAfter(
    std::string_view signal_name) {
  XLS_CHECK_EQ(metadata_->GetSignalWidth(signal_name), 1);
  actions_.push_back(WaitForSignals{
      .any_or_all = AnyOrAll::kAll,
      .signal_values = {SignalValue{std::string(signal_name), UBits(1, 1)}},
      .comment = absl::StrFormat("Wait for cycle after `%s` is asserted",
                                 signal_name)});
  return *this;
}

ModuleTestbenchThread& ModuleTestbenchThread::WaitForCycleAfterNot(
    std::string_view signal_name) {
  XLS_CHECK_EQ(metadata_->GetSignalWidth(signal_name), 1);
  actions_.push_back(WaitForSignals{
      .any_or_all = AnyOrAll::kAll,
      .signal_values = {SignalValue{std::string(signal_name), UBits(0, 1)}},
      .comment = absl::StrFormat("Wait for cycle after `%s` is de-asserted",
                                 signal_name)});
  return *this;
}

ModuleTestbenchThread& ModuleTestbenchThread::WaitForCycleAfterX(
    std::string_view signal_name) {
  actions_.push_back(WaitForSignals{
      .any_or_all = AnyOrAll::kAll,
      .signal_values = {SignalValue{std::string(signal_name), IsX()}},
      .comment =
          absl::StrFormat("Wait for cycle after `%s` is X", signal_name)});
  return *this;
}

ModuleTestbenchThread& ModuleTestbenchThread::WaitForCycleAfterNotX(
    std::string_view signal_name) {
  actions_.push_back(WaitForSignals{
      .any_or_all = AnyOrAll::kAll,
      .signal_values = {SignalValue{std::string(signal_name), IsNotX()}},
      .comment =
          absl::StrFormat("Wait for cycle after `%s` is not X", signal_name)});
  return *this;
}

ModuleTestbenchThread& ModuleTestbenchThread::WaitForCycleAfterAll(
    absl::Span<const std::string> signal_names) {
  std::vector<SignalValue> signal_values;
  for (const std::string& name : signal_names) {
    signal_values.push_back(SignalValue{std::string(name), UBits(1, 1)});
  }
  actions_.push_back(WaitForSignals{
      .any_or_all = AnyOrAll::kAll,
      .signal_values = signal_values,
      .comment = absl::StrFormat("Wait for cycle after all asserted: %s",
                                 absl::StrJoin(signal_names, ", "))});
  return *this;
}

ModuleTestbenchThread& ModuleTestbenchThread::WaitForCycleAfterAny(
    absl::Span<const std::string> signal_names) {
  std::vector<SignalValue> signal_values;
  for (const std::string& name : signal_names) {
    signal_values.push_back(SignalValue{std::string(name), UBits(1, 1)});
  }
  actions_.push_back(WaitForSignals{
      .any_or_all = AnyOrAll::kAny,
      .signal_values = signal_values,
      .comment = absl::StrFormat("Wait for cycle after any asserted: %s",
                                 absl::StrJoin(signal_names, ", "))});
  return *this;
}

ModuleTestbenchThread& ModuleTestbenchThread::Set(std::string_view signal_name,
                                                  const Bits& value) {
  CheckCanDriveSignal(signal_name);
  actions_.push_back(SetSignal{std::string(signal_name), value});
  return *this;
}

ModuleTestbenchThread& ModuleTestbenchThread::Set(std::string_view signal_name,
                                                  uint64_t value) {
  CheckCanDriveSignal(signal_name);
  return Set(signal_name, UBits(value, metadata_->GetSignalWidth(signal_name)));
}

ModuleTestbenchThread& ModuleTestbenchThread::SetX(
    std::string_view signal_name) {
  CheckCanDriveSignal(signal_name);
  actions_.push_back(SetSignalX{std::string(signal_name),
                                metadata_->GetSignalWidth(signal_name)});
  return *this;
}

SignalCapture SignalCaptureManager::Capture(const TestbenchSignal& signal,
                                            Bits* bits) {
  int64_t instance = signal_captures_.size();
  signal_captures_.push_back(
      SignalCapture{.signal = signal,
                    .action = TestbenchCapture{.bits = bits},
                    .instance_id = instance});
  return signal_captures_.back();
}

SignalCapture SignalCaptureManager::ExpectEq(const TestbenchSignal& signal,
                                             const Bits& bits,
                                             xabsl::SourceLocation loc) {
  int64_t instance = signal_captures_.size();
  signal_captures_.push_back(SignalCapture{
      .signal = signal,
      .action = TestbenchExpectation{.expected = bits, .loc = loc},
      .instance_id = instance});
  return signal_captures_.back();
}

SignalCapture SignalCaptureManager::ExpectX(const TestbenchSignal& signal,
                                            xabsl::SourceLocation loc) {
  int64_t instance = signal_captures_.size();
  signal_captures_.push_back(SignalCapture{
      .signal = signal,
      .action = TestbenchExpectation{.expected = IsX(), .loc = loc},
      .instance_id = instance});
  return signal_captures_.back();
}

EndOfCycleEvent& EndOfCycleEvent::Capture(std::string_view signal_name,
                                          Bits* value) {
  XLS_CHECK_OK(metadata_->CheckIsReadableSignal(signal_name));
  if (metadata_->GetSignalWidth(signal_name) > 0) {
    signal_captures_.push_back(
        capture_manager_->Capture(metadata_->GetSignal(signal_name), value));
  }
  return *this;
}

EndOfCycleEvent& EndOfCycleEvent::ExpectEq(std::string_view signal_name,
                                           const Bits& expected,
                                           xabsl::SourceLocation loc) {
  XLS_CHECK_OK(metadata_->CheckIsReadableSignal(signal_name));
  if (metadata_->GetSignalWidth(signal_name) > 0) {
    signal_captures_.push_back(capture_manager_->ExpectEq(
        metadata_->GetSignal(signal_name), expected, loc));
  }
  return *this;
}

EndOfCycleEvent& EndOfCycleEvent::ExpectEq(std::string_view signal_name,
                                           uint64_t expected,
                                           xabsl::SourceLocation loc) {
  XLS_CHECK_OK(metadata_->CheckIsReadableSignal(signal_name));
  return ExpectEq(signal_name,
                  UBits(expected, metadata_->GetSignalWidth(signal_name)), loc);
}

EndOfCycleEvent& EndOfCycleEvent::ExpectX(std::string_view signal_name,
                                          xabsl::SourceLocation loc) {
  XLS_CHECK_OK(metadata_->CheckIsReadableSignal(signal_name));
  if (metadata_->GetSignalWidth(signal_name) > 0) {
    signal_captures_.push_back(
        capture_manager_->ExpectX(metadata_->GetSignal(signal_name), loc));
  }
  return *this;
}

EndOfCycleEvent& ModuleTestbenchThread::AtEndOfCycle() {
  AdvanceCycle advance_cycle{
      .amount = 1,
      .end_of_cycle_event =
          std::make_unique<EndOfCycleEvent>(metadata_, capture_manager_)};
  EndOfCycleEvent* end_of_cycle_event = advance_cycle.end_of_cycle_event.get();
  actions_.push_back(std::move(advance_cycle));
  return *end_of_cycle_event;
}

EndOfCycleEvent& ModuleTestbenchThread::AtEndOfCycleWhenSignalsEq(
    AnyOrAll any_or_all, std::vector<SignalValue> signal_values,
    std::string_view comment) {
  WaitForSignals wait_for_signals{
      .any_or_all = any_or_all,
      .signal_values = std::move(signal_values),
      .end_of_cycle_event =
          std::make_unique<EndOfCycleEvent>(metadata_, capture_manager_),
      .comment = std::string{comment}};
  EndOfCycleEvent* end_of_cycle_event =
      wait_for_signals.end_of_cycle_event.get();
  actions_.push_back(std::move(wait_for_signals));
  return *end_of_cycle_event;
}

EndOfCycleEvent& ModuleTestbenchThread::AtEndOfCycleWhen(
    std::string_view signal_name) {
  XLS_CHECK_EQ(metadata_->GetSignalWidth(signal_name), 1);
  return AtEndOfCycleWhenSignalsEq(
      AnyOrAll::kAll, {SignalValue{std::string(signal_name), UBits(1, 1)}},
      absl::StrFormat("Wait for `%s` to be asserted and capture output",
                      signal_name));
}

EndOfCycleEvent& ModuleTestbenchThread::AtEndOfCycleWhenNot(
    std::string_view signal_name) {
  XLS_CHECK_EQ(metadata_->GetSignalWidth(signal_name), 1);
  return AtEndOfCycleWhenSignalsEq(
      AnyOrAll::kAll, {SignalValue{std::string(signal_name), UBits(0, 1)}},
      absl::StrFormat("Wait for `%s` to be de-asserted and capture output",
                      signal_name));
}

EndOfCycleEvent& ModuleTestbenchThread::AtEndOfCycleWhenAll(
    absl::Span<const std::string> signal_names) {
  std::vector<SignalValue> signal_values;
  for (const std::string& name : signal_names) {
    signal_values.push_back(SignalValue{std::string(name), UBits(1, 1)});
  }
  return AtEndOfCycleWhenSignalsEq(
      AnyOrAll::kAll, signal_values,
      absl::StrFormat("Wait for all asserted (and capture output): %s",
                      absl::StrJoin(signal_names, ", ")));
}

EndOfCycleEvent& ModuleTestbenchThread::AtEndOfCycleWhenAny(
    absl::Span<const std::string> signal_names) {
  std::vector<SignalValue> signal_values;
  for (const std::string& name : signal_names) {
    signal_values.push_back(SignalValue{std::string(name), UBits(1, 1)});
  }
  return AtEndOfCycleWhenSignalsEq(
      AnyOrAll::kAny, signal_values,
      absl::StrFormat("Wait for any asserted (and capture output): %s",
                      absl::StrJoin(signal_names, ", ")));
}

EndOfCycleEvent& ModuleTestbenchThread::AtEndOfCycleWhenX(
    std::string_view signal_name) {
  return AtEndOfCycleWhenSignalsEq(
      AnyOrAll::kAll, {SignalValue{std::string(signal_name), IsX()}},
      absl::StrFormat("Wait for `%s` to be X and capture output", signal_name));
}

EndOfCycleEvent& ModuleTestbenchThread::AtEndOfCycleWhenNotX(
    std::string_view signal_name) {
  return AtEndOfCycleWhenSignalsEq(
      AnyOrAll::kAll, {SignalValue{std::string(signal_name), IsNotX()}},
      absl::StrFormat("Wait for `%s` to be not X and capture output",
                      signal_name));
}

ModuleTestbenchThread& ModuleTestbenchThread::ExpectTrace(
    std::string_view trace_message) {
  expected_traces_.push_back(std::string(trace_message));
  return *this;
}

void ModuleTestbenchThread::CheckCanDriveSignal(std::string_view name) {
  XLS_CHECK(driven_signal_names_.contains(name)) << absl::StrFormat(
      "'%s' is not a signal that the thread is designated to drive.", name);
}

void ModuleTestbenchThread::CheckCanReadSignal(std::string_view name) {
  // Any signal except the clock is readable by any thread.
  XLS_CHECK(metadata_->HasSignal(name) && !metadata_->IsClock(name))
      << absl::StrFormat("'%s' is not a readable signal", name);
}

void ModuleTestbenchThread::EmitInto(
    StructuredProcedure* procedure, LogicRef* clk,
    const absl::flat_hash_map<std::string, LogicRef*>& signal_refs) {
  XLS_CHECK_NE(procedure, nullptr);

  // Some values must be initialize after a reset, for example control signals.
  for (const DrivenSignal& driven_signal : driven_signals_) {
    if (std::holds_alternative<Bits>(driven_signal.initial_value)) {
      EmitAction(SetSignal{driven_signal.signal_name,
                           std::get<Bits>(driven_signal.initial_value)},
                 procedure->statements(), clk, signal_refs);
    } else {
      EmitAction(
          SetSignalX{driven_signal.signal_name,
                     metadata_->GetSignalWidth(driven_signal.signal_name)},
          procedure->statements(), clk, signal_refs);
    }
  }

  // All actions assume they start at one time unit after the posedge of the
  // clock. This can be done with the advance one cycle action.
  EmitAction(AdvanceCycle{.amount = 1, .end_of_cycle_event = nullptr},
             procedure->statements(), clk, signal_refs);

  // The thread must wait for reset (if present). Specifically, the thread waits
  // until the last cycle of reset before emitting the actions. This enables the
  // first action (e.g., Set) to occur in the first cycle out of reset.
  if (metadata_->reset_proto().has_value()) {
    EmitAction(
        WaitForSignals{.any_or_all = AnyOrAll::kAll,
                       .signal_values = {SignalValue{
                           .signal_name = std::string{kLastResetCycleSignal},
                           .value = UBits(1, 1)}},
                       .end_of_cycle_event = nullptr,
                       .comment = "Wait for last cycle of reset"},
        procedure->statements(), clk, signal_refs);
  }
  // All actions occur one time unit after the pos edge of the clock.
  for (const Action& action : actions_) {
    EmitAction(action, procedure->statements(), clk, signal_refs);
  }
  // The last statement must be to assert the done signal of the thread.
  if (done_signal_name_.has_value()) {
    EmitAction(SetSignal{done_signal_name_.value(), UBits(1, 1)},
               procedure->statements(), clk, signal_refs);
  }
}

static absl::StatusOr<ModuleSignature> GenerateModuleSignature(
    Module* module, std::optional<std::string_view> clk_name,
    std::optional<ResetProto> reset) {
  ModuleSignatureBuilder b(module->name());
  for (const Port& port : module->ports()) {
    // The clock and reset should not be added as a data input.
    if ((clk_name.has_value() && port.name() == clk_name.value()) ||
        (reset.has_value() && port.name() == reset->name())) {
      continue;
    }
    const int64_t width = port.wire->data_type()->WidthAsInt64().value();
    if (port.direction == Direction::kInput) {
      b.AddDataInputAsBits(port.name(), width);
    } else {
      XLS_CHECK_EQ(port.direction, Direction::kOutput);
      b.AddDataOutputAsBits(port.name(), width);
    }
  }
  if (clk_name.has_value()) {
    b.WithClock(clk_name.value());
  }
  if (reset.has_value()) {
    b.WithReset(reset->name(), reset->asynchronous(), reset->active_low());
  }
  return b.Build();
}

ModuleTestbench::ModuleTestbench(Module* module,
                                 const VerilogSimulator* simulator,
                                 std::optional<std::string_view> clk_name,
                                 std::optional<ResetProto> reset,
                                 absl::Span<const VerilogInclude> includes)
    : ModuleTestbench(
          module->file()->Emit(),
          module->file()->use_system_verilog() ? FileType::kSystemVerilog
                                               : FileType::kVerilog,
          GenerateModuleSignature(module, clk_name, std::move(reset)).value(),
          simulator, includes) {}

ModuleTestbench::ModuleTestbench(std::string_view verilog_text,
                                 FileType file_type,
                                 const ModuleSignature& signature,
                                 const VerilogSimulator* simulator,
                                 absl::Span<const VerilogInclude> includes)
    : verilog_text_(verilog_text),
      file_type_(file_type),
      simulator_(simulator),
      includes_(includes),
      metadata_(signature) {
  if (metadata_.reset_proto().has_value()) {
    XLS_CHECK_OK(metadata_.AddInternalSignal(kLastResetCycleSignal, 1));
  }
}

absl::StatusOr<ModuleTestbenchThread*> ModuleTestbench::CreateThread(
    std::optional<absl::flat_hash_map<std::string, std::optional<Bits>>>
        owned_signals_to_drive,
    bool emit_done_signal) {
  std::vector<DrivenSignal> driven_signals;
  std::optional<std::string> done_signal_name = std::nullopt;
  if (emit_done_signal) {
    done_signal_name = absl::StrFormat("__is_thread_%d_done", threads_.size());
    XLS_RETURN_IF_ERROR(
        metadata_.AddInternalSignal(done_signal_name.value(), /*width=*/1));
  }
  if (owned_signals_to_drive.has_value()) {
    for (auto [name, initial_value] : owned_signals_to_drive.value()) {
      driven_signals.push_back(
          DrivenSignal{.signal_name = name,
                       .initial_value = initial_value.has_value()
                                            ? BitsOrX(initial_value.value())
                                            : BitsOrX(IsX())});
    }
    // Threads always drive their own done signal.
    if (done_signal_name.has_value()) {
      driven_signals.push_back(
          DrivenSignal{.signal_name = done_signal_name.value(),
                       .initial_value = UBits(0, 1)});
    }
  } else {
    // No owned signals specified. This thread is responsible for driving all
    // inputs.
    XLS_RET_CHECK(thread_owned_signals_.empty())
        << "ModuleTestbenchThread cannot drive all inputs as some inputs are "
           "being driven by existing threads.";
    for (const TestbenchSignal& signal : metadata_.signals()) {
      if (metadata_.IsClock(signal.name) ||
          signal.type == TestbenchSignalType::kOutputPort) {
        continue;
      }
      BitsOrX initial_value = IsX();
      // Initial value of the done signal must be zero for the testbench to work
      // properly.
      if (done_signal_name.has_value() &&
          signal.name == done_signal_name.value()) {
        initial_value = UBits(0, 1);
      }
      driven_signals.push_back(DrivenSignal{.signal_name = signal.name,
                                            .initial_value = initial_value});
    }
  }

  for (const DrivenSignal& driven_signal : driven_signals) {
    auto [it, inserted] =
        thread_owned_signals_.insert(driven_signal.signal_name);
    XLS_RET_CHECK(inserted) << absl::StreamFormat(
        "%s is being already being driven by an existing thread",
        driven_signal.signal_name);
  }

  // The driven signals were constructed from a map. Given them a deterministic
  // order.
  std::sort(driven_signals.begin(), driven_signals.end(),
            [](const DrivenSignal& a, const DrivenSignal& b) {
              return a.signal_name < b.signal_name;
            });

  threads_.push_back(std::make_unique<ModuleTestbenchThread>(
      &metadata_, &capture_manager_, driven_signals, done_signal_name));
  return threads_.back().get();
}

// Scans the given simulation stdout and finds the $display statement outputs
// associated with captured signals. Returns the signal values as Bits (or X)
// for each output found in a map indexed by id of the capture instance.
static absl::StatusOr<absl::flat_hash_map<int64_t, std::vector<BitsOrX>>>
ExtractSignalValues(std::string_view stdout_str) {
  // Scan the simulator output and pick out the OUTPUT lines holding the value
  // of module signal.
  absl::flat_hash_map<int64_t, std::vector<BitsOrX>> parsed_values;

  // Example output lines for a bits value:
  //
  //   5 OUTPUT out0 = 16'h12ab (#1)
  //
  // And a value with one or more X's:
  //
  //   5 OUTPUT out0 = 16'hxxab (#1)
  RE2 re(
      R"(\s+[0-9]+\s+OUTPUT\s(\w+)\s+=\s+([0-9]+)'h([0-9a-fA-FxX]+)\s+\(#([0-9]+)\))");
  std::string output_name;
  std::string output_width;
  std::string output_value;
  std::string instance_str;
  std::string_view piece(stdout_str);
  while (RE2::FindAndConsume(&piece, re, &output_name, &output_width,
                             &output_value, &instance_str)) {
    int64_t width;
    XLS_RET_CHECK(absl::SimpleAtoi(output_width, &width));
    int64_t instance;
    XLS_RET_CHECK(absl::SimpleAtoi(instance_str, &instance));

    XLS_VLOG(1) << absl::StreamFormat(
        "Found output `%s` width %d value %s instance #%d", output_name, width,
        output_value, instance);

    if (absl::StrContains(output_value, "x") ||
        absl::StrContains(output_value, "X")) {
      parsed_values[instance].push_back(IsX());
    } else {
      XLS_ASSIGN_OR_RETURN(
          Bits value, ParseUnsignedNumberWithoutPrefix(output_value,
                                                       FormatPreference::kHex));
      XLS_RET_CHECK_GE(width, value.bit_count());
      parsed_values[instance].push_back(bits_ops::ZeroExtend(value, width));
    }
  }
  return parsed_values;
}

absl::Status ModuleTestbench::CaptureOutputsAndCheckExpectations(
    std::string_view stdout_str) const {
  // Check for timeout.
  if (absl::StrContains(stdout_str, GetTimeoutMessage())) {
    return absl::DeadlineExceededError(
        absl::StrFormat("Simulation exceeded maximum length of %d cycles.",
                        kSimulationCycleLimit));
  }

  absl::flat_hash_map<int64_t, std::vector<BitsOrX>> outputs;
  XLS_ASSIGN_OR_RETURN(outputs, ExtractSignalValues(stdout_str));

  for (const SignalCapture& signal_capture :
       capture_manager_.signal_captures()) {
    if (!outputs.contains(signal_capture.instance_id)) {
      return absl::NotFoundError(absl::StrFormat(
          "Output `%s`, instance #%d not found in Verilog simulator output.",
          signal_capture.signal.name, signal_capture.instance_id));
    }
    XLS_RET_CHECK_EQ(outputs.at(signal_capture.instance_id).size(), 1);
    const BitsOrX& bits_or_x = outputs.at(signal_capture.instance_id).front();

    if (std::holds_alternative<TestbenchCapture>(signal_capture.action)) {
      // Write out module signal values to pointers passed in via Capture
      // calls.
      const TestbenchCapture& bits_capture =
          std::get<TestbenchCapture>(signal_capture.action);
      if (std::holds_alternative<IsX>(bits_or_x)) {
        return absl::NotFoundError(absl::StrFormat(
            "Output `%s`, instance #%d holds X value in "
            "Verilog simulator output.",
            signal_capture.signal.name, signal_capture.instance_id));
      }
      *bits_capture.bits = std::get<Bits>(bits_or_x);
    } else {
      // Check the signal value against any expectations.
      const TestbenchExpectation& expectation =
          std::get<TestbenchExpectation>(signal_capture.action);
      auto get_source_location = [&]() {
        return absl::StrFormat("%s@%d", expectation.loc.file_name(),
                               expectation.loc.line());
      };
      if (std::holds_alternative<Bits>(expectation.expected)) {
        const Bits& expected_bits = std::get<Bits>(expectation.expected);
        if (std::holds_alternative<IsX>(bits_or_x)) {
          return absl::FailedPreconditionError(absl::StrFormat(
              "%s: expected output `%s`, instance #%d to "
              "have value: %s, has X",
              get_source_location(), signal_capture.signal.name,
              signal_capture.instance_id, expected_bits.ToString()));
        }
        const Bits& actual_bits = std::get<Bits>(bits_or_x);
        if (actual_bits != expected_bits) {
          return absl::FailedPreconditionError(absl::StrFormat(
              "%s: expected output `%s`, instance #%d to have value: %s, "
              "actual: %s",
              get_source_location(), signal_capture.signal.name,
              signal_capture.instance_id, expected_bits.ToString(),
              actual_bits.ToString()));
        }
      } else {
        XLS_CHECK(std::holds_alternative<IsX>(expectation.expected));
        if (std::holds_alternative<Bits>(bits_or_x)) {
          return absl::FailedPreconditionError(absl::StrFormat(
              "%s: expected output `%s`, instance #%d to have X value, has "
              "non X value: %s",
              get_source_location(), signal_capture.signal.name,
              signal_capture.instance_id,
              std::get<Bits>(bits_or_x).ToString()));
        }
      }
    }
  }

  // Look for the expected trace messages in the simulation output.
  size_t search_pos = 0;
  for (const std::string& message : GatherExpectedTraces()) {
    size_t found_pos = stdout_str.find(message, search_pos);
    if (found_pos == std::string_view::npos) {
      return absl::NotFoundError(absl::StrFormat(
          "Expected trace \"%s\" not found in Verilog simulator output.",
          message));
    }
    search_pos = found_pos + message.length();
  }

  return absl::OkStatus();
}

std::string ModuleTestbench::GenerateVerilog() {
  VerilogFile file(file_type_);
  Module* m = file.AddModule("testbench", SourceInfo());

  LogicRef* clk = nullptr;
  absl::flat_hash_map<std::string, LogicRef*> signal_refs;
  std::vector<Connection> connections;
  for (const TestbenchSignal& signal : metadata_.signals()) {
    if (signal.width == 0) {
      // Skip zero-width inputs (e.g., empty tuples) as these have no actual
      // port in the Verilog module.
      continue;
    }
    LogicRef* ref;
    if (signal.type == TestbenchSignalType::kInputPort) {
      ref =
          m->AddReg(signal.name, file.BitVectorType(signal.width, SourceInfo()),
                    SourceInfo());
    } else if (signal.type == TestbenchSignalType::kOutputPort) {
      ref = m->AddWire(signal.name,
                       file.BitVectorType(signal.width, SourceInfo()),
                       SourceInfo());
    } else {
      // Internal signal. This should not be connected to the DUT instantiation.
      XLS_CHECK(signal.type == TestbenchSignalType::kInternal);
      continue;
    }
    signal_refs[signal.name] = ref;
    connections.push_back(Connection{signal.name, ref});

    if (metadata_.IsClock(signal.name)) {
      clk = ref;
    }
  }

  // If DUT has no clock, create a clock reg (not connected to the DUT) as the
  // testbench requires a clock for sequencing.
  if (clk == nullptr) {
    clk = m->AddReg("clk", file.ScalarType(SourceInfo()), SourceInfo());
    signal_refs["clk"] = clk;
  }

  // Instantiate the device under test module.
  const char kInstantiationName[] = "dut";
  m->Add<Instantiation>(
      SourceInfo(), metadata_.dut_module_name(), kInstantiationName,
      /*parameters=*/absl::Span<const Connection>(), connections);

  std::vector<LogicRef*> done_signal_refs;
  std::vector<std::string> done_signal_names;
  for (const auto& thread : threads_) {
    if (!thread->done_signal_name().has_value()) {
      continue;
    }
    LogicRef* ref =
        m->AddReg(thread->done_signal_name().value(),
                  file.BitVectorType(1, SourceInfo()), SourceInfo());
    done_signal_names.push_back(thread->done_signal_name().value());
    done_signal_refs.push_back(ref);
    signal_refs[thread->done_signal_name().value()] = ref;
  }

  {
    // Generate the clock. It has a frequency of ten time units. Start clk at 0
    // at time zero to avoid any races with rising edge of clock and any
    // initialization.
    m->Add<BlankLine>(SourceInfo());
    m->Add<Comment>(SourceInfo(), "Clock generator.");
    Initial* initial = m->Add<Initial>(SourceInfo());
    initial->statements()->Add<BlockingAssignment>(
        SourceInfo(), clk, file.PlainLiteral(0, SourceInfo()));
    initial->statements()->Add<Forever>(
        SourceInfo(),
        file.Make<DelayStatement>(
            SourceInfo(), file.PlainLiteral(kClockPeriod / 2, SourceInfo()),
            file.Make<BlockingAssignment>(SourceInfo(), clk,
                                          file.LogicalNot(clk, SourceInfo()))));
  }

  {
    // Global reset controller. If a reset is present, the threads will wait
    // on this reset prior to starting their execution.
    if (metadata_.reset_proto().has_value()) {
      m->Add<BlankLine>(SourceInfo());
      m->Add<Comment>(SourceInfo(), "Reset generator.");
      LogicRef* last_cycle_of_reset =
          m->AddReg(kLastResetCycleSignal, file.BitVectorType(1, SourceInfo()),
                    SourceInfo());
      signal_refs[kLastResetCycleSignal] = last_cycle_of_reset;
      LogicRef* reset = signal_refs.at(metadata_.reset_proto()->name());
      Expression* zero = file.Literal(UBits(0, 1), SourceInfo());
      Expression* one = file.Literal(UBits(1, 1), SourceInfo());
      Expression* reset_start_value =
          metadata_.reset_proto()->active_low() ? zero : one;
      Expression* reset_end_value =
          metadata_.reset_proto()->active_low() ? one : zero;
      Initial* initial = m->Add<Initial>(SourceInfo());
      initial->statements()->Add<BlockingAssignment>(SourceInfo(), reset,
                                                     reset_start_value);
      initial->statements()->Add<BlockingAssignment>(SourceInfo(),
                                                     last_cycle_of_reset, zero);
      WaitNCycles(clk, initial->statements(), kResetCycles - 1);
      initial->statements()->Add<BlockingAssignment>(SourceInfo(),
                                                     last_cycle_of_reset, one);
      WaitNCycles(clk, initial->statements(), 1);
      initial->statements()->Add<BlockingAssignment>(SourceInfo(), reset,
                                                     reset_end_value);
      initial->statements()->Add<BlockingAssignment>(SourceInfo(),
                                                     last_cycle_of_reset, zero);
    }
  }

  {
    // Add a watchdog which stops the simulation after a long time.
    m->Add<BlankLine>(SourceInfo());
    m->Add<Comment>(SourceInfo(), "Watchdog timer.");
    Initial* initial = m->Add<Initial>(SourceInfo());
    WaitNCycles(clk, initial->statements(), kSimulationCycleLimit);
    initial->statements()->Add<Display>(
        SourceInfo(), std::vector<Expression*>{file.Make<QuotedString>(
                          SourceInfo(), GetTimeoutMessage())});
    initial->statements()->Add<Finish>(SourceInfo());
  }

  {
    // Add a monitor statement which prints out all the port values.
    m->Add<BlankLine>(SourceInfo());
    m->Add<Comment>(SourceInfo(), "Monitor for input/output ports.");
    Initial* initial = m->Add<Initial>(SourceInfo());
    initial->statements()->Add<Display>(
        SourceInfo(),
        std::vector<Expression*>{file.Make<QuotedString>(
            SourceInfo(),
            absl::StrFormat("Starting. Clock rises at %d, %d, %d, ...:",
                            kClockPeriod / 2, kClockPeriod + kClockPeriod / 2,
                            2 * kClockPeriod + kClockPeriod / 2))});
    std::string monitor_fmt = "%t";
    std::vector<Expression*> monitor_args = {
        file.Make<SystemFunctionCall>(SourceInfo(), "time")};
    for (const Connection& connection : connections) {
      if (metadata_.IsClock(connection.port_name)) {
        continue;
      }
      absl::StrAppend(&monitor_fmt, " ", connection.port_name, ": %d");
      monitor_args.push_back(connection.expression);
    }
    monitor_args.insert(monitor_args.begin(),
                        file.Make<QuotedString>(SourceInfo(), monitor_fmt));
    initial->statements()->Add<Monitor>(SourceInfo(), monitor_args);
  }

  int64_t thread_number = 0;
  for (const auto& thread : threads_) {
    // TODO(vmirian) : Consider lowering the thread to an 'always' block.
    m->Add<BlankLine>(SourceInfo());
    std::vector<std::string> driven_signal_names;
    for (const DrivenSignal& driven_signal : thread->driven_signals()) {
      driven_signal_names.push_back(driven_signal.signal_name);
    }
    m->Add<Comment>(
        SourceInfo(),
        absl::StrFormat("Thread %d. Drives signals: %s", thread_number,
                        driven_signal_names.empty()
                            ? "<none>"
                            : absl::StrJoin(driven_signal_names, ", ")));
    Initial* initial = m->Add<Initial>(SourceInfo());
    thread->EmitInto(initial, clk, signal_refs);
    ++thread_number;
  }
  {
    // Add a finish statement when all threads are complete.
    m->Add<BlankLine>(SourceInfo());
    m->Add<Comment>(SourceInfo(), "Thread completion monitor.");

    Initial* initial = m->Add<Initial>(SourceInfo());

    // All actions assume they start one unit after the clock posedge.
    WaitForClockPosEdge(clk, initial->statements());
    EmitDelay(initial->statements(), 1);

    if (metadata_.reset_proto().has_value()) {
      EmitAction(
          WaitForSignals{
              .any_or_all = AnyOrAll::kAll,
              .signal_values = {SignalValue{
                  .signal_name = metadata_.reset_proto()->name(),
                  .value =
                      UBits(metadata_.reset_proto()->active_low() ? 1 : 0, 1)}},
              .end_of_cycle_event = nullptr,
              .comment = "Wait for reset deasserted"},
          initial->statements(), clk, signal_refs);
    }
    std::vector<SignalValue> done_signal_values;
    done_signal_values.reserve(done_signal_names.size());
    for (const std::string& done_signal : done_signal_names) {
      done_signal_values.push_back(
          SignalValue{.signal_name = done_signal, .value = UBits(1, 1)});
    }
    EmitAction(WaitForSignals{.any_or_all = AnyOrAll::kAll,
                              .signal_values = done_signal_values,
                              .end_of_cycle_event = nullptr,
                              .comment = "Wait for all threads to complete"},
               initial->statements(), clk, signal_refs);
    initial->statements()->Add<Finish>(SourceInfo());
  }

  // Concatenate the module Verilog with the testbench verilog to create the
  // verilog text to pass to the simulator.
  return absl::StrCat(verilog_text_, "\n\n", file.Emit());
}

std::vector<std::string> ModuleTestbench::GatherExpectedTraces() const {
  std::vector<std::string> expected_traces;
  for (const std::unique_ptr<ModuleTestbenchThread>& thread : threads_) {
    for (const std::string& expected_trace : thread->expected_traces()) {
      expected_traces.push_back(expected_trace);
    }
  }
  return expected_traces;
}

absl::Status ModuleTestbench::Run() {
  std::string verilog_text = GenerateVerilog();
  XLS_VLOG_LINES(3, verilog_text);

  std::pair<std::string, std::string> stdout_stderr;
  XLS_ASSIGN_OR_RETURN(stdout_stderr,
                       simulator_->Run(verilog_text, file_type_, includes_));

  XLS_VLOG(2) << "Verilog simulator stdout:\n" << stdout_stderr.first;
  XLS_VLOG(2) << "Verilog simulator stderr:\n" << stdout_stderr.second;

  const std::string& stdout_str = stdout_stderr.first;
  return CaptureOutputsAndCheckExpectations(stdout_str);
}

}  // namespace verilog
}  // namespace xls
