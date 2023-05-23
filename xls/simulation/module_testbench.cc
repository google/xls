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
    absl::Span<const DisplaySignal> display_signals,
    StatementBlock* statement_block,
    const absl::flat_hash_map<std::string, LogicRef*>& signal_refs) {
  for (const DisplaySignal& display_signal : display_signals) {
    statement_block->Add<Display>(
        SourceInfo(),
        std::vector<Expression*>{
            statement_block->file()->Make<QuotedString>(
                SourceInfo(),
                absl::StrFormat("%%t OUTPUT %s = %d'h%%0x (#%d)",
                                display_signal.signal_name,
                                display_signal.width, display_signal.instance)),
            statement_block->file()->Make<SystemFunctionCall>(SourceInfo(),
                                                              "time"),
            signal_refs.at(display_signal.signal_name)});
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
            !a.end_of_cycle_event->display_signals().empty()) {
          // Capture signals one time unit before posedge of the clock.
          delay_to_right_before_posedge(statement_block);
          EmitDisplayStatements(a.end_of_cycle_event->display_signals(),
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
          EmitDisplayStatements(w.end_of_cycle_event->display_signals(),
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

ModuleTestbenchThread& ModuleTestbenchThread::NextCycle() {
  return AdvanceNCycles(1);
}

ModuleTestbenchThread& ModuleTestbenchThread::AdvanceNCycles(int64_t n_cycles) {
  actions_.push_back(AdvanceCycle{n_cycles});
  return *this;
}

ModuleTestbenchThread& ModuleTestbenchThread::WaitForCycleAfter(
    std::string_view signal_name) {
  XLS_CHECK_EQ(GetSignalWidth(signal_name), 1);
  actions_.push_back(WaitForSignals{
      .any_or_all = AnyOrAll::kAll,
      .signal_values = {SignalValue{std::string(signal_name), UBits(1, 1)}},
      .comment = absl::StrFormat("Wait for cycle after `%s` is asserted",
                                 signal_name)});
  return *this;
}

ModuleTestbenchThread& ModuleTestbenchThread::WaitForCycleAfterNot(
    std::string_view signal_name) {
  XLS_CHECK_EQ(GetSignalWidth(signal_name), 1);
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
  return Set(signal_name, UBits(value, GetSignalWidth(signal_name)));
}

ModuleTestbenchThread& ModuleTestbenchThread::SetX(
    std::string_view signal_name) {
  CheckCanDriveSignal(signal_name);
  actions_.push_back(
      SetSignalX{std::string(signal_name), GetSignalWidth(signal_name)});
  return *this;
}

EndOfCycleEvent& EndOfCycleEvent::Capture(std::string_view signal_name,
                                          Bits* value) {
  thread_->CheckCanReadSignal(signal_name);
  if (thread_->GetSignalWidth(signal_name) > 0) {
    int64_t instance = thread_->next_instance_++;
    display_signals_.push_back(
        DisplaySignal{.signal_name = std::string(signal_name),
                      .width = thread_->GetSignalWidth(signal_name),
                      .instance = instance});
    auto key = std::make_pair(instance, std::string(signal_name));
    thread_->captures_[key] = value;
  }
  return *this;
}

EndOfCycleEvent& EndOfCycleEvent::ExpectEq(std::string_view signal_name,
                                           const Bits& expected,
                                           xabsl::SourceLocation loc) {
  thread_->CheckCanReadSignal(signal_name);
  int64_t instance = thread_->next_instance_++;
  auto key = std::make_pair(instance, std::string(signal_name));
  XLS_CHECK(thread_->expectations_.find(key) == thread_->expectations_.end());
  thread_->expectations_[key] = Expectation{.expected = expected, .loc = loc};
  display_signals_.push_back(
      DisplaySignal{.signal_name = std::string(signal_name),
                    .width = thread_->GetSignalWidth(signal_name),
                    .instance = instance});
  return *this;
}

EndOfCycleEvent& EndOfCycleEvent::ExpectEq(std::string_view signal_name,
                                           uint64_t expected,
                                           xabsl::SourceLocation loc) {
  thread_->CheckCanReadSignal(signal_name);
  return ExpectEq(signal_name,
                  UBits(expected, thread_->GetSignalWidth(signal_name)), loc);
}

EndOfCycleEvent& EndOfCycleEvent::ExpectX(std::string_view signal_name,
                                          xabsl::SourceLocation loc) {
  thread_->CheckCanReadSignal(signal_name);
  int64_t instance = thread_->next_instance_++;
  auto key = std::make_pair(instance, std::string(signal_name));
  XLS_CHECK(thread_->expectations_.find(key) == thread_->expectations_.end());
  thread_->expectations_[key] = Expectation{.expected = IsX(), .loc = loc};
  display_signals_.push_back(
      DisplaySignal{.signal_name = std::string(signal_name),
                    .width = thread_->GetSignalWidth(signal_name),
                    .instance = instance});
  return *this;
}

EndOfCycleEvent& ModuleTestbenchThread::AtEndOfCycle() {
  AdvanceCycle advance_cycle{
      .amount = 1,
      .end_of_cycle_event = std::make_unique<EndOfCycleEvent>(this)};
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
      .end_of_cycle_event = std::make_unique<EndOfCycleEvent>(this),
      .comment = std::string{comment}};
  EndOfCycleEvent* end_of_cycle_event =
      wait_for_signals.end_of_cycle_event.get();
  actions_.push_back(std::move(wait_for_signals));
  return *end_of_cycle_event;
}

EndOfCycleEvent& ModuleTestbenchThread::AtEndOfCycleWhen(
    std::string_view signal_name) {
  XLS_CHECK_EQ(GetSignalWidth(signal_name), 1);
  return AtEndOfCycleWhenSignalsEq(
      AnyOrAll::kAll, {SignalValue{std::string(signal_name), UBits(1, 1)}},
      absl::StrFormat("Wait for `%s` to be asserted and capture output",
                      signal_name));
}

EndOfCycleEvent& ModuleTestbenchThread::AtEndOfCycleWhenNot(
    std::string_view signal_name) {
  XLS_CHECK_EQ(GetSignalWidth(signal_name), 1);
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

int64_t ModuleTestbenchThread::GetSignalWidth(std::string_view name) {
  if (done_signal_name_.has_value() && name == done_signal_name_.value()) {
    return 1;
  }
  if (shared_data_->input_port_widths.contains(name)) {
    return shared_data_->input_port_widths.at(name);
  }
  return shared_data_->output_port_widths.at(name);
}

void ModuleTestbenchThread::CheckCanDriveSignal(std::string_view name) {
  XLS_CHECK(
      owned_signals_to_drive_.contains(name) ||
      (done_signal_name_.has_value() && name == done_signal_name_.value()))
      << absl::StrFormat(
             "'%s' is not a signal that the thread is designated to drive.",
             name);
}

void ModuleTestbenchThread::CheckCanReadSignal(std::string_view name) {
  XLS_CHECK(shared_data_->output_port_widths.contains(name) ||
            (shared_data_->reset.has_value() &&
             shared_data_->reset.value().name() == name))
      << absl::StrFormat(
             "'%s' is not a signal that the thread is designated to read.",
             name);
}

absl::Status ModuleTestbenchThread::CheckOutput(
    std::string_view stdout_str,
    const absl::flat_hash_map<InstanceSignalName, BitsOrX>& parsed_values)
    const {
  // Write out module signal values to pointers passed in via Capture
  // calls.
  for (const auto& pair : captures_) {
    auto cycle_signal = pair.first;
    Bits* value_ptr = pair.second;
    if (!parsed_values.contains(cycle_signal)) {
      return absl::NotFoundError(absl::StrFormat(
          "Output %s, instance #%d not found in Verilog simulator output.",
          cycle_signal.second, cycle_signal.first));
    }
    if (std::holds_alternative<IsX>(parsed_values.at(cycle_signal))) {
      return absl::NotFoundError(absl::StrFormat(
          "Output %s, instance #%d holds X value in Verilog simulator output.",
          cycle_signal.second, cycle_signal.first));
    }
    *value_ptr = std::get<Bits>(parsed_values.at(cycle_signal));
  }

  // Check the signal value against any expectations.
  for (const auto& pair : expectations_) {
    auto cycle_signal = pair.first;
    const Expectation& expectation = pair.second;
    auto get_source_location = [&]() {
      return absl::StrFormat("%s@%d", expectation.loc.file_name(),
                             expectation.loc.line());
    };
    if (!parsed_values.contains(cycle_signal)) {
      return absl::NotFoundError(absl::StrFormat(
          "%s: output '%s', instance #%d @ %s not found in Verilog "
          "simulator output.",
          get_source_location(), cycle_signal.second, cycle_signal.first,
          ToString(expectation.loc)));
    }
    if (std::holds_alternative<Bits>(expectation.expected)) {
      const Bits& expected_bits = std::get<Bits>(expectation.expected);
      if (std::holds_alternative<IsX>(parsed_values.at(cycle_signal))) {
        return absl::FailedPreconditionError(absl::StrFormat(
            "%s: expected output '%s', instance #%d to have value: %s, has X",
            get_source_location(), cycle_signal.second, cycle_signal.first,
            expected_bits.ToString()));
      }
      const Bits& actual_bits = std::get<Bits>(parsed_values.at(cycle_signal));
      if (actual_bits != expected_bits) {
        return absl::FailedPreconditionError(absl::StrFormat(
            "%s: expected output '%s', instance #%d to have value: %s, actual: "
            "%s",
            get_source_location(), cycle_signal.second, cycle_signal.first,
            expected_bits.ToString(), actual_bits.ToString()));
      }
    } else {
      XLS_CHECK(std::holds_alternative<IsX>(expectation.expected));
      if (std::holds_alternative<Bits>(parsed_values.at(cycle_signal))) {
        return absl::FailedPreconditionError(absl::StrFormat(
            "%s: expected output '%s', instance #%d to have X value, has non X "
            "value: %s",
            get_source_location(), cycle_signal.second, cycle_signal.first,
            std::get<Bits>(parsed_values.at(cycle_signal)).ToString()));
      }
    }
  }

  // Look for the expected trace messages in the simulation output.
  size_t search_pos = 0;
  for (const std::string& message : expected_traces_) {
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

std::vector<std::string> ModuleTestbenchThread::GetThreadOwnedSignals() const {
  std::vector<std::string> vec;
  for (auto [name, _] : owned_signals_to_drive_) {
    vec.push_back(name);
  }
  std::sort(vec.begin(), vec.end());
  return vec;
}

void ModuleTestbenchThread::EmitInto(
    StructuredProcedure* procedure, LogicRef* clk,
    const absl::flat_hash_map<std::string, LogicRef*>& signal_refs) {
  XLS_CHECK_NE(procedure, nullptr);

  // Before doing anything, initialize signals. The first statement must be to
  // deassert the done signal of the thread.
  if (done_signal_name_.has_value()) {
    EmitAction(SetSignal{done_signal_name_.value(), UBits(0, 1)},
               procedure->statements(), clk, signal_refs);
  }

  // Some values must be initialize after a reset, for example control signals.
  for (const auto& [signal_name, value] : owned_signals_to_drive_) {
    if (value.has_value()) {
      EmitAction(SetSignal{signal_name, value.value()}, procedure->statements(),
                 clk, signal_refs);
    } else {
      EmitAction(SetSignalX{signal_name, GetSignalWidth(signal_name)},
                 procedure->statements(), clk, signal_refs);
    }
  }

  // All actions assume they start at one time unit after the posedge of the
  // clock. This can be done with the advance one cycle action.
  EmitAction(AdvanceCycle{.amount = 1, .end_of_cycle_event = nullptr},
             procedure->statements(), clk, signal_refs);

  // The thread must wait for reset (if present).
  if (shared_data_->reset.has_value()) {
    EmitAction(
        WaitForSignals{
            .any_or_all = AnyOrAll::kAll,
            .signal_values = {SignalValue{
                .signal_name = shared_data_->reset->name(),
                .value = UBits(shared_data_->reset.value().active_low() ? 1 : 0,
                               1)}},
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

ModuleTestbench::ModuleTestbench(Module* module,
                                 const VerilogSimulator* simulator,
                                 std::optional<std::string_view> clk_name,
                                 std::optional<ResetProto> reset,
                                 absl::Span<const VerilogInclude> includes)
    // Emit the entire file because the module may instantiate other modules.
    : verilog_text_(module->file()->Emit()),
      file_type_(module->file()->use_system_verilog() ? FileType::kSystemVerilog
                                                      : FileType::kVerilog),
      simulator_(simulator),
      includes_(includes) {
  XLS_VLOG(3) << "Building ModuleTestbench for Verilog module:";
  XLS_VLOG_LINES(3, verilog_text_);
  shared_data_.dut_module_name = module->name();
  absl::btree_map<std::string, int64_t>& input_port_widths =
      shared_data_.input_port_widths;
  absl::btree_map<std::string, int64_t>& output_port_widths =
      shared_data_.output_port_widths;
  shared_data_.reset = reset;

  // Define default clock name to control the testbench.
  if (!clk_name.has_value()) {
    shared_data_.clk_name = "clk";
  } else {
    shared_data_.clk_name = clk_name.value();
  }

  for (const Port& port : module->ports()) {
    const int64_t width = port.wire->data_type()->WidthAsInt64().value();
    if (port.direction == Direction::kInput) {
      input_port_widths[port.name()] = width;
    } else {
      XLS_CHECK_EQ(port.direction, Direction::kOutput);
      output_port_widths[port.name()] = width;
    }
  }
  if (reset.has_value()) {
    input_port_widths[reset->name()] = 1;
  }
}

ModuleTestbench::ModuleTestbench(std::string_view verilog_text,
                                 FileType file_type,
                                 const ModuleSignature& signature,
                                 const VerilogSimulator* simulator,
                                 absl::Span<const VerilogInclude> includes)
    : verilog_text_(verilog_text),
      file_type_(file_type),
      simulator_(simulator),
      includes_(includes) {
  XLS_VLOG(3) << "Building ModuleTestbench for Verilog module:";
  XLS_VLOG_LINES(3, verilog_text_);
  XLS_VLOG(3) << "With signature:";
  XLS_VLOG_LINES(3, signature.ToString());
  shared_data_.dut_module_name = signature.module_name();
  absl::btree_map<std::string, int64_t>& input_port_widths =
      shared_data_.input_port_widths;
  absl::btree_map<std::string, int64_t>& output_port_widths =
      shared_data_.output_port_widths;

  for (const PortProto& port : signature.data_inputs()) {
    input_port_widths[port.name()] = port.width();
  }
  for (const PortProto& port : signature.data_outputs()) {
    output_port_widths[port.name()] = port.width();
  }

  // Add in any non-data ports.
  // Define default clock name to control the testbench.
  if (!signature.proto().has_clock_name()) {
    shared_data_.clk_name = "clk";
  } else {
    shared_data_.clk_name = signature.proto().clock_name();
    input_port_widths[shared_data_.clk_name] = 1;
  }
  if (signature.proto().has_reset()) {
    shared_data_.reset = signature.proto().reset();
    input_port_widths[signature.proto().reset().name()] = 1;
  }
  if (signature.proto().has_pipeline() &&
      signature.proto().pipeline().has_pipeline_control()) {
    // Module has pipeline register control.
    if (signature.proto().pipeline().pipeline_control().has_valid()) {
      // Add the valid input and optional valid output signals.
      const ValidProto& valid =
          signature.proto().pipeline().pipeline_control().valid();
      input_port_widths[valid.input_name()] = 1;
      if (!valid.output_name().empty()) {
        output_port_widths[valid.output_name()] = 1;
      }
    } else if (signature.proto().pipeline().pipeline_control().has_manual()) {
      // Add the manual pipeline register load enable signal.
      input_port_widths[signature.proto()
                            .pipeline()
                            .pipeline_control()
                            .manual()
                            .input_name()] =
          signature.proto().pipeline().latency();
    }
  }
}

absl::StatusOr<ModuleTestbenchThread*> ModuleTestbench::CreateThread(
    std::optional<absl::flat_hash_map<std::string, std::optional<Bits>>>
        owned_signals_to_drive,
    bool emit_done_signal) {
  std::optional<std::string> done_signal_name = std::nullopt;
  if (emit_done_signal) {
    done_signal_name =
        absl::StrFormat("is_thread_%s_done", std::to_string(threads_.size()));
  }
  if (!owned_signals_to_drive.has_value()) {
    // No owned signals specified. This thread is responsible for driving all
    // inputs.
    XLS_RET_CHECK(thread_owned_signals_.empty())
        << "ModuleTestbenchThread cannot drive all inputs as some inputs are "
           "being driven by existing threads.";
    owned_signals_to_drive =
        absl::flat_hash_map<std::string, std::optional<Bits>>();
    for (const auto& [name, width] : shared_data_.input_port_widths) {
      if (name == shared_data_.clk_name) {
        continue;
      }
      (*owned_signals_to_drive)[name] = std::nullopt;
    }
  }
  for (auto [signal, _] : owned_signals_to_drive.value()) {
    auto [it, inserted] = thread_owned_signals_.insert(signal);
    XLS_RET_CHECK(inserted) << absl::StreamFormat(
        "%s is being already being driven by an existing thread", signal);
  }

  threads_.push_back(std::make_unique<ModuleTestbenchThread>(
      &shared_data_, owned_signals_to_drive.value(), done_signal_name));
  return threads_.back().get();
}

absl::Status ModuleTestbench::CheckOutput(std::string_view stdout_str) const {
  // Check for timeout.
  if (absl::StrContains(stdout_str, GetTimeoutMessage())) {
    return absl::DeadlineExceededError(
        absl::StrFormat("Simulation exceeded maximum length of %d cycles.",
                        kSimulationCycleLimit));
  }

  // Scan the simulator output and pick out the OUTPUT lines holding the value
  // of module signal.
  absl::flat_hash_map<ModuleTestbenchThread::InstanceSignalName, BitsOrX>
      parsed_values;

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
    XLS_VLOG(1) << absl::StreamFormat(
        "Found output %s width %s value %s instance %s", output_name,
        output_width, output_value, instance_str);
    int64_t width;
    XLS_RET_CHECK(absl::SimpleAtoi(output_width, &width));
    int64_t instance;
    XLS_RET_CHECK(absl::SimpleAtoi(instance_str, &instance));
    if (absl::StrContains(output_value, "x") ||
        absl::StrContains(output_value, "X")) {
      parsed_values[{instance, output_name}] = IsX();
    } else {
      XLS_ASSIGN_OR_RETURN(
          Bits value, ParseUnsignedNumberWithoutPrefix(output_value,
                                                       FormatPreference::kHex));
      XLS_RET_CHECK_GE(width, value.bit_count());
      parsed_values[{instance, output_name}] =
          bits_ops::ZeroExtend(value, width);
    }
  }

  for (const auto& thread : threads_) {
    XLS_RETURN_IF_ERROR(thread->CheckOutput(stdout_str, parsed_values));
  }
  return absl::OkStatus();
}

std::string ModuleTestbench::GenerateVerilog() {
  VerilogFile file(file_type_);
  Module* m = file.AddModule("testbench", SourceInfo());

  absl::flat_hash_map<std::string, LogicRef*> signal_refs;
  std::vector<Connection> connections;
  for (const auto& pair : shared_data_.input_port_widths) {
    if (pair.second == 0) {
      // Skip zero-width inputs (e.g., empty tuples) as these have no actual
      // port in the Verilog module.
      continue;
    }
    const std::string& port_name = pair.first;
    LogicRef* ref = m->AddReg(
        port_name, file.BitVectorType(pair.second, SourceInfo()), SourceInfo());
    signal_refs[port_name] = ref;
    connections.push_back(Connection{port_name, ref});
  }

  for (const auto& [port_name, width] : shared_data_.output_port_widths) {
    if (width == 0) {
      // Skip zero-width outputs (e.g., empty tuples) as these have no actual
      // port in the Verilog module.
      continue;
    }
    LogicRef* ref = m->AddWire(
        port_name, file.BitVectorType(width, SourceInfo()), SourceInfo());
    signal_refs[port_name] = ref;
    connections.push_back(Connection{port_name, ref});
  }

  // For combinational modules define, but do not connect a clock signal.
  if (!signal_refs.contains(shared_data_.clk_name)) {
    signal_refs[shared_data_.clk_name] =
        m->AddReg("clk", file.ScalarType(SourceInfo()), SourceInfo());
  }
  LogicRef* clk = signal_refs[shared_data_.clk_name];

  // Instantiate the device under test module.
  const char kInstantiationName[] = "dut";
  m->Add<Instantiation>(
      SourceInfo(), shared_data_.dut_module_name, kInstantiationName,
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
    if (shared_data_.reset.has_value()) {
      m->Add<BlankLine>(SourceInfo());
      m->Add<Comment>(SourceInfo(), "Reset generator.");
      LogicRef* reset = signal_refs.at(shared_data_.reset->name());
      Expression* zero = file.Literal(UBits(0, 1), SourceInfo());
      Expression* one = file.Literal(UBits(1, 1), SourceInfo());
      Expression* reset_start_value =
          shared_data_.reset->active_low() ? zero : one;
      Expression* reset_end_value =
          shared_data_.reset->active_low() ? one : zero;
      Initial* initial = m->Add<Initial>(SourceInfo());
      initial->statements()->Add<BlockingAssignment>(SourceInfo(), reset,
                                                     reset_start_value);

      WaitNCycles(clk, initial->statements(), kResetCycles);
      initial->statements()->Add<BlockingAssignment>(SourceInfo(), reset,
                                                     reset_end_value);
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
      if (connection.port_name == shared_data_.clk_name) {
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
    std::vector<std::string> inputs = thread->GetThreadOwnedSignals();
    m->Add<Comment>(
        SourceInfo(),
        absl::StrFormat(
            "Thread %d. Drives inputs: %s", thread_number,
            inputs.empty() ? "<none>" : absl::StrJoin(inputs, ", ")));
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

    if (shared_data_.reset.has_value()) {
      EmitAction(
          WaitForSignals{
              .any_or_all = AnyOrAll::kAll,
              .signal_values = {SignalValue{
                  .signal_name = shared_data_.reset->name(),
                  .value = UBits(
                      shared_data_.reset.value().active_low() ? 1 : 0, 1)}},
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

absl::Status ModuleTestbench::Run() {
  std::string verilog_text = GenerateVerilog();
  XLS_VLOG_LINES(3, verilog_text);

  std::pair<std::string, std::string> stdout_stderr;
  XLS_ASSIGN_OR_RETURN(stdout_stderr,
                       simulator_->Run(verilog_text, file_type_, includes_));

  XLS_VLOG(2) << "Verilog simulator stdout:\n" << stdout_stderr.first;
  XLS_VLOG(2) << "Verilog simulator stderr:\n" << stdout_stderr.second;

  const std::string& stdout_str = stdout_stderr.first;
  return CheckOutput(stdout_str);
}

}  // namespace verilog
}  // namespace xls
