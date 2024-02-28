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

#include "xls/simulation/module_testbench_thread.h"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/log/die_if_null.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "absl/types/variant.h"
#include "xls/codegen/vast.h"
#include "xls/common/logging/logging.h"
#include "xls/common/visitor.h"
#include "xls/ir/bits.h"
#include "xls/ir/source_location.h"
#include "xls/simulation/testbench_metadata.h"
#include "xls/simulation/testbench_signal_capture.h"
#include "xls/simulation/testbench_stream.h"

namespace xls {
namespace verilog {
namespace {

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
  CHECK_GT(n_cycles, 0);
  CHECK_NE(statement_block, nullptr);
  VerilogFile& file = *statement_block->file();
  Expression* posedge_clk = file.Make<PosEdge>(SourceInfo(), clk);
  if (n_cycles == 1) {
    WaitForClockPosEdge(clk, statement_block);
  } else {
    verilog::RepeatStatement* repeat_statement =
        statement_block->Add<verilog::RepeatStatement>(
            SourceInfo(),
            file.PlainLiteral(static_cast<int32_t>(n_cycles), SourceInfo()));
    repeat_statement->statements()->Add<EventControl>(SourceInfo(),
                                                      posedge_clk);
  }
  EmitDelay(statement_block, 1);
}

// Emit $display statements and file I/O into the given block which sample the
// value of the given signals.
void EmitSignalCaptures(
    absl::Span<const SignalCapture> signal_captures,
    StatementBlock* statement_block,
    const absl::flat_hash_map<std::string, LogicRef*>& signal_refs,
    const absl::flat_hash_map<std::string, VastStreamEmitter>&
        stream_emitters) {
  for (const SignalCapture& signal_capture : signal_captures) {
    if (std::holds_alternative<const TestbenchStream*>(signal_capture.action)) {
      const TestbenchStream* stream =
          std::get<const TestbenchStream*>(signal_capture.action);
      CHECK_GT(stream->width, 0);
      stream_emitters.at(stream->name)
          .EmitWrite(statement_block,
                     signal_refs.at(signal_capture.signal_name));
      continue;
    }
    if (signal_capture.signal_width == 0) {
      // Zero-width signals are not actually represented in the Verilog though
      // they may appear in the module signature. Call $display to print a
      // constant 0 value in this case.
      statement_block->Add<Display>(
          SourceInfo(),
          std::vector<Expression*>{
              statement_block->file()->Make<QuotedString>(
                  SourceInfo(), absl::StrFormat("%%t OUTPUT %s = 0'h0 (#%d)",
                                                signal_capture.signal_name,
                                                signal_capture.instance_id)),
              statement_block->file()->Make<SystemFunctionCall>(SourceInfo(),
                                                                "time")});
    } else {
      statement_block->Add<Display>(
          SourceInfo(),
          std::vector<Expression*>{
              statement_block->file()->Make<QuotedString>(
                  SourceInfo(),
                  absl::StrFormat("%%t OUTPUT %s = %d'h%%0x (#%d)",
                                  signal_capture.signal_name,
                                  signal_capture.signal_width,
                                  signal_capture.instance_id)),
              statement_block->file()->Make<SystemFunctionCall>(SourceInfo(),
                                                                "time"),
              signal_refs.at(signal_capture.signal_name)});
    }
  }
}

// Emits Verilog implementing the given action.
void EmitAction(const Action& action, StatementBlock* statement_block,
                LogicRef* clk,
                const absl::flat_hash_map<std::string, LogicRef*>& signal_refs,
                const absl::flat_hash_map<std::string, VastStreamEmitter>&
                    stream_emitters) {
  // Inserts a delay which waits until one unit before the clock
  // posedge. Assumes that simulation is currently one time unit after posedge
  // of clock which is the point at which every action starts.
  auto delay_to_right_before_posedge = [&](StatementBlock* sb) {
    EmitDelay(sb, kClockPeriod - 2);
  };
  VerilogFile& file = *statement_block->file();
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
          EmitSignalCaptures(a.end_of_cycle_event->signal_captures(),
                             statement_block, signal_refs, stream_emitters);
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
      [&](const SetSignalFromStream& s) {
        statement_block->Add<Comment>(
            SourceInfo(),
            absl::StrFormat("Reading value from stream `%s`", s.stream->name));
        stream_emitters.at(s.stream->name)
            .EmitRead(statement_block, signal_refs.at(s.signal_name));
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
            CHECK(std::holds_alternative<IsNotX>(signal_value.value));
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
          EmitSignalCaptures(w.end_of_cycle_event->signal_captures(),
                             statement_block, signal_refs, stream_emitters);
        }

        // Every action should terminate one unit after the posedge of the
        // clock.
        WaitForClockPosEdge(clk, statement_block);
        EmitDelay(statement_block, 1);
      },
      [&](const DisplayAction& d) {
        statement_block->Add<Display>(
            SourceInfo(), std::vector<Expression*>({file.Make<QuotedString>(
                              SourceInfo(), d.message)}));
      },
      [&](const FinishAction& f) {
        statement_block->Add<Finish>(SourceInfo());
      },
  };
  absl::visit(visitor, action);
}

}  // namespace

const TestbenchMetadata& SequentialBlock::metadata() const {
  return testbench_thread_->metadata();
}

SignalCaptureManager& SequentialBlock::signal_capture_manager() const {
  return testbench_thread_->signal_capture_manager();
}

SequentialBlock& SequentialBlock::NextCycle() { return AdvanceNCycles(1); }

SequentialBlock& SequentialBlock::AdvanceNCycles(int64_t n_cycles) {
  statements_.push_back(AdvanceCycle{n_cycles});
  return *this;
}

SequentialBlock& SequentialBlock::WaitForCycleAfter(
    std::string_view signal_name) {
  CHECK_EQ(metadata().GetPortWidth(signal_name), 1);
  statements_.push_back(WaitForSignals{
      .any_or_all = AnyOrAll::kAll,
      .signal_values = {SignalValue{std::string(signal_name), UBits(1, 1)}},
      .comment = absl::StrFormat("Wait for cycle after `%s` is asserted",
                                 signal_name)});
  return *this;
}

SequentialBlock& SequentialBlock::WaitForCycleAfterNot(
    std::string_view signal_name) {
  CHECK_EQ(metadata().GetPortWidth(signal_name), 1);
  statements_.push_back(WaitForSignals{
      .any_or_all = AnyOrAll::kAll,
      .signal_values = {SignalValue{std::string(signal_name), UBits(0, 1)}},
      .comment = absl::StrFormat("Wait for cycle after `%s` is de-asserted",
                                 signal_name)});
  return *this;
}

SequentialBlock& SequentialBlock::WaitForCycleAfterX(
    std::string_view signal_name) {
  statements_.push_back(WaitForSignals{
      .any_or_all = AnyOrAll::kAll,
      .signal_values = {SignalValue{std::string(signal_name), IsX()}},
      .comment =
          absl::StrFormat("Wait for cycle after `%s` is X", signal_name)});
  return *this;
}

SequentialBlock& SequentialBlock::WaitForCycleAfterNotX(
    std::string_view signal_name) {
  statements_.push_back(WaitForSignals{
      .any_or_all = AnyOrAll::kAll,
      .signal_values = {SignalValue{std::string(signal_name), IsNotX()}},
      .comment =
          absl::StrFormat("Wait for cycle after `%s` is not X", signal_name)});
  return *this;
}

SequentialBlock& SequentialBlock::WaitForCycleAfterAll(
    absl::Span<const std::string> signal_names) {
  std::vector<SignalValue> signal_values;
  for (const std::string& name : signal_names) {
    signal_values.push_back(SignalValue{std::string(name), UBits(1, 1)});
  }
  statements_.push_back(WaitForSignals{
      .any_or_all = AnyOrAll::kAll,
      .signal_values = signal_values,
      .comment = absl::StrFormat("Wait for cycle after all asserted: %s",
                                 absl::StrJoin(signal_names, ", "))});
  return *this;
}

SequentialBlock& SequentialBlock::WaitForCycleAfterAny(
    absl::Span<const std::string> signal_names) {
  std::vector<SignalValue> signal_values;
  for (const std::string& name : signal_names) {
    signal_values.push_back(SignalValue{std::string(name), UBits(1, 1)});
  }
  statements_.push_back(WaitForSignals{
      .any_or_all = AnyOrAll::kAny,
      .signal_values = signal_values,
      .comment = absl::StrFormat("Wait for cycle after any asserted: %s",
                                 absl::StrJoin(signal_names, ", "))});
  return *this;
}

SequentialBlock& SequentialBlock::Set(std::string_view signal_name,
                                      const Bits& value) {
  testbench_thread_->CheckCanDriveSignal(signal_name);
  statements_.push_back(SetSignal{std::string(signal_name), value});
  return *this;
}

SequentialBlock& SequentialBlock::Set(std::string_view signal_name,
                                      uint64_t value) {
  testbench_thread_->CheckCanDriveSignal(signal_name);
  return Set(signal_name, UBits(value, metadata().GetPortWidth(signal_name)));
}

SequentialBlock& SequentialBlock::SetX(std::string_view signal_name) {
  testbench_thread_->CheckCanDriveSignal(signal_name);
  statements_.push_back(SetSignalX{std::string(signal_name),
                                   metadata().GetPortWidth(signal_name)});
  return *this;
}

SequentialBlock& SequentialBlock::ReadFromStreamAndSet(
    std::string_view signal_name, const TestbenchStream* stream) {
  testbench_thread_->CheckCanDriveSignal(signal_name);
  statements_.push_back(SetSignalFromStream{std::string(signal_name), stream});
  return *this;
}

EndOfCycleEvent& SequentialBlock::AtEndOfCycle() {
  AdvanceCycle advance_cycle{
      .amount = 1,
      .end_of_cycle_event = std::make_unique<EndOfCycleEvent>(
          &metadata(), &signal_capture_manager())};
  EndOfCycleEvent* end_of_cycle_event = advance_cycle.end_of_cycle_event.get();
  statements_.push_back(std::move(advance_cycle));
  return *end_of_cycle_event;
}

EndOfCycleEvent& SequentialBlock::AtEndOfCycleWhenSignalsEq(
    AnyOrAll any_or_all, std::vector<SignalValue> signal_values,
    std::string_view comment) {
  WaitForSignals wait_for_signals{
      .any_or_all = any_or_all,
      .signal_values = std::move(signal_values),
      .end_of_cycle_event = std::make_unique<EndOfCycleEvent>(
          &metadata(), &signal_capture_manager()),
      .comment = std::string{comment}};
  EndOfCycleEvent* end_of_cycle_event =
      wait_for_signals.end_of_cycle_event.get();
  statements_.push_back(std::move(wait_for_signals));
  return *end_of_cycle_event;
}

EndOfCycleEvent& SequentialBlock::AtEndOfCycleWhen(
    std::string_view signal_name) {
  CHECK_EQ(metadata().GetPortWidth(signal_name), 1);
  return AtEndOfCycleWhenSignalsEq(
      AnyOrAll::kAll, {SignalValue{std::string(signal_name), UBits(1, 1)}},
      absl::StrFormat("Wait for `%s` to be asserted and capture output",
                      signal_name));
}

EndOfCycleEvent& SequentialBlock::AtEndOfCycleWhenNot(
    std::string_view signal_name) {
  CHECK_EQ(metadata().GetPortWidth(signal_name), 1);
  return AtEndOfCycleWhenSignalsEq(
      AnyOrAll::kAll, {SignalValue{std::string(signal_name), UBits(0, 1)}},
      absl::StrFormat("Wait for `%s` to be de-asserted and capture output",
                      signal_name));
}

EndOfCycleEvent& SequentialBlock::AtEndOfCycleWhenAll(
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

EndOfCycleEvent& SequentialBlock::AtEndOfCycleWhenAny(
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

EndOfCycleEvent& SequentialBlock::AtEndOfCycleWhenX(
    std::string_view signal_name) {
  return AtEndOfCycleWhenSignalsEq(
      AnyOrAll::kAll, {SignalValue{std::string(signal_name), IsX()}},
      absl::StrFormat("Wait for `%s` to be X and capture output", signal_name));
}

EndOfCycleEvent& SequentialBlock::AtEndOfCycleWhenNotX(
    std::string_view signal_name) {
  return AtEndOfCycleWhenSignalsEq(
      AnyOrAll::kAll, {SignalValue{std::string(signal_name), IsNotX()}},
      absl::StrFormat("Wait for `%s` to be not X and capture output",
                      signal_name));
}

SequentialBlock& SequentialBlock::RepeatForever() {
  statements_.push_back(RepeatStatement{
      .count = std::nullopt,
      .sequential_block =
          std::make_unique<SequentialBlock>(testbench_thread_)});
  return *std::get<RepeatStatement>(statements_.back()).sequential_block;
}
SequentialBlock& SequentialBlock::Repeat(int64_t count) {
  statements_.push_back(RepeatStatement{
      .count = count,
      .sequential_block =
          std::make_unique<SequentialBlock>(testbench_thread_)});
  return *std::get<RepeatStatement>(statements_.back()).sequential_block;
}

void SequentialBlock::Display(std::string_view message) {
  statements_.push_back(DisplayAction{.message = std::string{message}});
}

void SequentialBlock::Finish() { statements_.push_back(FinishAction()); }

void SequentialBlock::Emit(
    StatementBlock* statement_block, LogicRef* clk,
    const absl::flat_hash_map<std::string, LogicRef*>& signal_refs,
    const absl::flat_hash_map<std::string, VastStreamEmitter>&
        stream_emitters) {
  VerilogFile& file = *statement_block->file();
  for (const Statement& statement : statements_) {
    if (std::holds_alternative<Action>(statement)) {
      EmitAction(std::get<Action>(statement), statement_block, clk, signal_refs,
                 stream_emitters);
    } else {
      const RepeatStatement& repeat_statement =
          std::get<RepeatStatement>(statement);
      if (repeat_statement.count.has_value()) {
        verilog::RepeatStatement* verilog_statement =
            statement_block->Add<verilog::RepeatStatement>(
                SourceInfo(),
                file.PlainLiteral(
                    static_cast<int32_t>(repeat_statement.count.value()),
                    SourceInfo()));
        repeat_statement.sequential_block->Emit(
            verilog_statement->statements(), clk, signal_refs, stream_emitters);
      } else {
        verilog::WhileStatement* while_statement =
            statement_block->Add<verilog::WhileStatement>(
                SourceInfo(), file.PlainLiteral(1, SourceInfo()));
        repeat_statement.sequential_block->Emit(
            while_statement->statements(), clk, signal_refs, stream_emitters);
      }
    }
  }
}

ModuleTestbenchThread::ModuleTestbenchThread(
    std::string_view name, const TestbenchMetadata* metadata,
    SignalCaptureManager* capture_manager,
    absl::Span<const DutInput> dut_inputs, bool generate_done_signal,
    bool wait_for_reset)
    : name_(name),
      metadata_(ABSL_DIE_IF_NULL(metadata)),
      capture_manager_(capture_manager),
      dut_inputs_(dut_inputs.begin(), dut_inputs.end()),
      wait_for_reset_(wait_for_reset) {
  for (const DutInput& signal : dut_inputs) {
    drivable_signals_.push_back(signal.port_name);
  }
  if (generate_done_signal) {
    std::string done_signal_name =
        absl::StrFormat("__thread_%s_done", SanitizeIdentifier(name));
    done_signal_ = TestbenchSignal{
        .name = done_signal_name, .width = 1, .initial_value = UBits(0, 1)};
    DeclareInternalSignal(done_signal_->name, done_signal_->width,
                          done_signal_->initial_value);
  }
  main_block_ = std::make_unique<SequentialBlock>(this);
}

ModuleTestbenchThread& ModuleTestbenchThread::ExpectTrace(
    std::string_view trace_message) {
  expected_traces_.push_back(std::string(trace_message));
  return *this;
}

void ModuleTestbenchThread::DeclareInternalSignal(
    std::string_view name, int64_t width, const BitsOrX& initial_value) {
  drivable_signals_.push_back(std::string{name});
  TestbenchSignal signal{.name = std::string{name},
                         .width = width,
                         .initial_value = initial_value};
  internal_signals_.push_back(signal);
}

void ModuleTestbenchThread::EmitInto(
    Module* m, LogicRef* clk,
    absl::flat_hash_map<std::string, LogicRef*>* signal_refs,
    const absl::flat_hash_map<std::string, VastStreamEmitter>&
        stream_emitters) {
  for (const TestbenchSignal& signal : internal_signals_) {
    (*signal_refs)[signal.name] = m->AddReg(
        signal.name, m->file()->BitVectorType(signal.width, SourceInfo()),
        SourceInfo(), /*init=*/nullptr);
  }

  StructuredProcedure* procedure = m->Add<Initial>(SourceInfo());

  // Initialize the DUT input ports driven by this thread and any internal
  // signals.
  for (const DutInput& dut_input : dut_inputs_) {
    if (std::holds_alternative<Bits>(dut_input.initial_value)) {
      EmitAction(SetSignal{dut_input.port_name,
                           std::get<Bits>(dut_input.initial_value)},
                 procedure->statements(), clk, *signal_refs, stream_emitters);
    } else {
      EmitAction(SetSignalX{dut_input.port_name,
                            metadata_->GetPortWidth(dut_input.port_name)},
                 procedure->statements(), clk, *signal_refs, stream_emitters);
    }
  }
  for (const TestbenchSignal& signal : internal_signals_) {
    if (std::holds_alternative<Bits>(signal.initial_value)) {
      EmitAction(SetSignal{signal.name, std::get<Bits>(signal.initial_value)},
                 procedure->statements(), clk, *signal_refs, stream_emitters);
    } else {
      EmitAction(SetSignalX{signal.name, signal.width}, procedure->statements(),
                 clk, *signal_refs, stream_emitters);
    }
  }

  procedure->statements()->Add<BlankLine>(SourceInfo());

  // All actions assume they start at one time unit after the posedge of the
  // clock. This can be done with the advance one cycle action.
  EmitAction(AdvanceCycle{.amount = 1, .end_of_cycle_event = nullptr},
             procedure->statements(), clk, *signal_refs, stream_emitters);

  // The thread must wait for reset (if present and specified). Specifically,
  // the thread waits until the last cycle of reset before emitting the actions.
  // This enables the first action (e.g., Set) to occur in the first cycle out
  // of reset.
  if (wait_for_reset_ && metadata_->reset_proto().has_value()) {
    EmitAction(
        WaitForSignals{.any_or_all = AnyOrAll::kAll,
                       .signal_values = {SignalValue{
                           .signal_name = std::string{kLastResetCycleSignal},
                           .value = UBits(1, 1)}},
                       .end_of_cycle_event = nullptr,
                       .comment = "Wait for last cycle of reset"},
        procedure->statements(), clk, *signal_refs, stream_emitters);
  }
  // All actions occur one time unit after the pos edge of the clock.
  main_block_->Emit(procedure->statements(), clk, *signal_refs,
                    stream_emitters);

  // The last statement must be to assert the done signal of the thread.
  if (done_signal_.has_value()) {
    procedure->statements()->Add<BlankLine>(SourceInfo());
    procedure->statements()->Add<BlockingAssignment>(
        SourceInfo(), (*signal_refs)[done_signal_->name],
        procedure->file()->Literal(UBits(1, 1), SourceInfo()));
  }
}

void ModuleTestbenchThread::CheckCanDriveSignal(std::string_view signal_name) {
  CHECK(std::find(drivable_signals_.begin(), drivable_signals_.end(),
                  signal_name) != drivable_signals_.end())
      << absl::StrFormat(
             "'%s' is not a signal that thread `%s is designated to drive.",
             signal_name, name());
}

}  // namespace verilog
}  // namespace xls
