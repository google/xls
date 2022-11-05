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
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include "absl/status/status.h"
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

// Insert statements into statement block which delay the simulation for the
// given number of cycles. Regardless of delay, simulation resumes on the
// falling edge of the clock.
void WaitNCycles(LogicRef* clk, StatementBlock* statement_block,
                 int64_t n_cycles) {
  XLS_CHECK_GT(n_cycles, 0);
  XLS_CHECK_NE(statement_block, nullptr);
  VerilogFile& file = *statement_block->file();
  Expression* posedge_clk = file.Make<PosEdge>(SourceInfo(), clk);
  if (n_cycles == 1) {
    statement_block->Add<EventControl>(SourceInfo(), posedge_clk);
  } else {
    statement_block->Add<RepeatStatement>(
        SourceInfo(), file.PlainLiteral(n_cycles, SourceInfo()),
        file.Make<EventControl>(SourceInfo(), posedge_clk));
  }
  statement_block->Add<EventControl>(SourceInfo(),
                                     file.Make<NegEdge>(SourceInfo(), clk));
}

}  // namespace

ModuleTestbenchThread& ModuleTestbenchThread::NextCycle() {
  return AdvanceNCycles(1);
}

ModuleTestbenchThread& ModuleTestbenchThread::AdvanceNCycles(int64_t n_cycles) {
  actions_.push_back(AdvanceCycle{n_cycles});
  return *this;
}

ModuleTestbenchThread& ModuleTestbenchThread::WaitFor(
    std::string_view signal_name) {
  XLS_CHECK_EQ(GetSignalWidth(signal_name), 1);
  actions_.push_back(WaitForSignal{std::string(signal_name), UBits(1, 1)});
  return *this;
}

ModuleTestbenchThread& ModuleTestbenchThread::WaitForNot(
    std::string_view signal_name) {
  XLS_CHECK_EQ(GetSignalWidth(signal_name), 1);
  actions_.push_back(WaitForSignal{std::string(signal_name), UBits(0, 1)});
  return *this;
}

ModuleTestbenchThread& ModuleTestbenchThread::WaitForX(
    std::string_view signal_name) {
  actions_.push_back(WaitForSignal{std::string(signal_name), IsX{}});
  return *this;
}

ModuleTestbenchThread& ModuleTestbenchThread::WaitForNotX(
    std::string_view signal_name) {
  actions_.push_back(WaitForSignal{std::string(signal_name), IsNotX{}});
  return *this;
}
ModuleTestbenchThread& ModuleTestbenchThread::WaitForEvent(
    std::string_view signal_name, Bits value) {
  XLS_CHECK_EQ(GetSignalWidth(signal_name), value.bit_count());
  actions_.push_back(WaitForSignalEvent{std::string(signal_name), value,
                                        /*is_comparison_equal=*/true});
  return *this;
}

ModuleTestbenchThread& ModuleTestbenchThread::WaitForEventNot(
    std::string_view signal_name, Bits value) {
  XLS_CHECK_EQ(GetSignalWidth(signal_name), value.bit_count());
  actions_.push_back(WaitForSignalEvent{std::string(signal_name), value,
                                        /*is_comparison_equal=*/false});
  return *this;
}

ModuleTestbenchThread& ModuleTestbenchThread::WaitForEventX(
    std::string_view signal_name) {
  actions_.push_back(WaitForSignalEvent{std::string(signal_name), IsX{},
                                        /*is_comparison_equal=*/true});
  return *this;
}

ModuleTestbenchThread& ModuleTestbenchThread::WaitForEventNotX(
    std::string_view signal_name) {
  actions_.push_back(WaitForSignalEvent{std::string(signal_name), IsX{},
                                        /*is_comparison_equal=*/false});
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
  actions_.push_back(SetSignalX{std::string(signal_name)});
  return *this;
}

ModuleTestbenchThread& ModuleTestbenchThread::ExpectEq(
    std::string_view signal_name, const Bits& expected,
    xabsl::SourceLocation loc) {
  CheckCanReadSignal(signal_name);
  int64_t instance = next_instance_++;
  auto key = std::make_pair(instance, std::string(signal_name));
  XLS_CHECK(expectations_.find(key) == expectations_.end());
  expectations_[key] = Expectation{.expected = expected, .loc = loc};
  actions_.push_back(DisplaySignal{std::string(signal_name), instance});
  return *this;
}

ModuleTestbenchThread& ModuleTestbenchThread::ExpectEq(
    std::string_view signal_name, uint64_t expected,
    xabsl::SourceLocation loc) {
  CheckCanReadSignal(signal_name);
  return ExpectEq(signal_name, UBits(expected, GetSignalWidth(signal_name)),
                  loc);
}

ModuleTestbenchThread& ModuleTestbenchThread::ExpectX(
    std::string_view signal_name, xabsl::SourceLocation loc) {
  CheckCanReadSignal(signal_name);
  int64_t instance = next_instance_++;
  auto key = std::make_pair(instance, std::string(signal_name));
  XLS_CHECK(expectations_.find(key) == expectations_.end());
  expectations_[key] = Expectation{.expected = IsX(), .loc = loc};
  actions_.push_back(DisplaySignal{std::string(signal_name), instance});
  return *this;
}

ModuleTestbenchThread& ModuleTestbenchThread::ExpectTrace(
    std::string_view trace_message) {
  expected_traces_.push_back(std::string(trace_message));
  return *this;
}

ModuleTestbenchThread& ModuleTestbenchThread::Capture(
    std::string_view signal_name, Bits* value) {
  CheckCanReadSignal(signal_name);
  if (GetSignalWidth(signal_name) > 0) {
    int64_t instance = next_instance_++;
    actions_.push_back(DisplaySignal{std::string(signal_name), instance});
    auto key = std::make_pair(instance, std::string(signal_name));
    captures_[key] = value;
  }
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
    const absl::flat_hash_map<InstanceSignalName, std::variant<Bits, IsX>>&
        parsed_values) const {
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

void ModuleTestbenchThread::EmitInto(
    StructuredProcedure* procedure,
    const absl::flat_hash_map<std::string, LogicRef*>& signal_refs) {
  XLS_CHECK_NE(procedure, nullptr);
  VerilogFile& file = *procedure->file();
  LogicRef* clk = signal_refs.at(shared_data_->clk_name);
  Visitor visitor{
      [&](const AdvanceCycle& a) {
        WaitNCycles(clk, procedure->statements(), a.amount);
      },
      [&](const SetSignal& s) {
        if (GetSignalWidth(s.signal_name) > 0) {
          procedure->statements()->Add<NonblockingAssignment>(
              SourceInfo(), signal_refs.at(s.signal_name),
              file.Literal(s.value, SourceInfo()));
        }
      },
      [&](const SetSignalX& s) {
        if (GetSignalWidth(s.signal_name) > 0) {
          procedure->statements()->Add<NonblockingAssignment>(
              SourceInfo(), signal_refs.at(s.signal_name),
              file.Make<XSentinel>(SourceInfo(),
                                   GetSignalWidth(s.signal_name)));
        }
      },
      [&](const WaitForSignal& w) {
        // WaitForSignal waits until the signal equals a certain value
        // at the falling edge of the clock. Use a while loop to
        // sample every cycle at the falling edge of the clock.
        // TODO(meheff): If we switch to SystemVerilog this is better
        // handled using a clocking block.
        Expression* cmp;
        if (std::holds_alternative<Bits>(w.value)) {
          cmp = file.NotEquals(
              signal_refs.at(w.signal_name),
              file.Literal(std::get<Bits>(w.value), SourceInfo()),
              SourceInfo());
        } else if (std::holds_alternative<IsX>(w.value)) {
          cmp = file.NotEqualsX(signal_refs.at(w.signal_name), SourceInfo());
        } else {
          XLS_CHECK(std::holds_alternative<IsNotX>(w.value));
          cmp = file.EqualsX(signal_refs.at(w.signal_name), SourceInfo());
        }
        auto whle =
            procedure->statements()->Add<WhileStatement>(SourceInfo(), cmp);
        WaitNCycles(clk, whle->statements(), 1);
      },
      [&](const WaitForSignalEvent& w) {
        Expression* cmp;
        if (std::holds_alternative<Bits>(w.value)) {
          if (w.is_comparison_equal) {
            cmp =
                file.Equals(signal_refs.at(w.signal_name),
                            file.Literal(std::get<Bits>(w.value), SourceInfo()),
                            SourceInfo());
          } else {
            cmp = file.NotEquals(
                signal_refs.at(w.signal_name),
                file.Literal(std::get<Bits>(w.value), SourceInfo()),
                SourceInfo());
          }
        } else {
          XLS_CHECK(std::holds_alternative<IsX>(w.value));
          if (w.is_comparison_equal) {
            cmp = file.EqualsX(signal_refs.at(w.signal_name), SourceInfo());
          } else {
            cmp = file.NotEqualsX(signal_refs.at(w.signal_name), SourceInfo());
          }
        }
        procedure->statements()->Add<WaitStatement>(SourceInfo(), cmp);
      },
      [&](const DisplaySignal& c) {
        // Use $strobe rather than $display to print value after all
        // assignments in the simulator time slot and avoid any
        // potential race conditions.
        procedure->statements()->Add<Strobe>(
            SourceInfo(),
            std::vector<Expression*>{
                file.Make<QuotedString>(
                    SourceInfo(),
                    absl::StrFormat("%%t OUTPUT %s = %d'h%%0x (#%d)",
                                    c.signal_name,
                                    GetSignalWidth(c.signal_name), c.instance)),
                file.Make<SystemFunctionCall>(SourceInfo(), "time"),
                signal_refs.at(c.signal_name)});
      }};
  // The first statement must be to deassert the done signal of the thread.
  if (done_signal_name_.has_value()) {
    absl::visit(visitor,
                Action{SetSignal{done_signal_name_.value(), UBits(0, 1)}});
  }
  // Wait for a cycle is part of original protocol.
  absl::visit(visitor, Action{AdvanceCycle{1}});

  // Some values must be initialize after a reset, for example control signals.
  for (const auto& [signal_name, value] : owned_signals_to_drive_) {
    if (value.has_value()) {
      absl::visit(visitor, Action{SetSignal{signal_name, value.value()}});
    } else {
      absl::visit(visitor, Action{SetSignalX{signal_name}});
    }
  }
  // The thread must wait for reset (if present).
  if (shared_data_->reset.has_value()) {
    absl::visit(visitor,
                Action{WaitForSignalEvent{
                    std::string(shared_data_->reset.value().name()),
                    UBits(shared_data_->reset.value().active_low() ? 1 : 0, 1),
                    /*is_comparison_equal=*/true}});
  }
  // All actions occur at the falling edge of the clock to avoid races with
  // signals changing at the rising edge of the clock.
  for (const Action& action : actions_) {
    absl::visit(visitor, action);
  }
  // The last statement must be to assert the done signal of the thread.
  if (done_signal_name_.has_value()) {
    absl::visit(visitor,
                Action{SetSignal{done_signal_name_.value(), UBits(1, 1)}});
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
  XLS_VLOG(3) << "With signature:\n" << signature;
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

ModuleTestbenchThread& ModuleTestbench::CreateThread(
    std::optional<absl::flat_hash_map<std::string, std::optional<Bits>>>
        owned_signals_to_drive,
    bool emit_done_signal) {
  std::optional<std::string> done_signal_name = std::nullopt;
  if (emit_done_signal) {
    done_signal_name =
        absl::StrFormat("is_thread_%s_done", std::to_string(threads_.size()));
  }
  if (!owned_signals_to_drive.has_value()) {
    owned_signals_to_drive =
        absl::flat_hash_map<std::string, std::optional<Bits>>();
    for (const auto& [name, width] : shared_data_.input_port_widths) {
      if (name == shared_data_.clk_name) {
        continue;
      }
      (*owned_signals_to_drive)[name] = std::nullopt;
    }
  }
  threads_.push_back(std::make_unique<ModuleTestbenchThread>(
      &shared_data_, owned_signals_to_drive.value(), done_signal_name));
  ModuleTestbenchThread& thread = *threads_.back();
  return thread;
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
  absl::flat_hash_map<ModuleTestbenchThread::InstanceSignalName,
                      std::variant<Bits, ModuleTestbenchThread::IsX>>
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
      parsed_values[{instance, output_name}] = ModuleTestbenchThread::IsX();
    } else {
      XLS_ASSIGN_OR_RETURN(
          Bits value, ParseUnsignedNumberWithoutPrefix(output_value,
                                                       FormatPreference::kHex));
      XLS_RET_CHECK_GE(width, value.bit_count());
      parsed_values[{instance, output_name}] =
          bits_ops::ZeroExtend(value, width);
    }
  }

  for (int64_t index = 0; index < threads_.size(); ++index) {
    XLS_RETURN_IF_ERROR(
        threads_[index]->CheckOutput(stdout_str, parsed_values));
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
  for (int64_t index = 0; index < threads_.size(); ++index) {
    if (!threads_[index]->done_signal_name().has_value()) {
      continue;
    }
    LogicRef* ref =
        m->AddReg(threads_[index]->done_signal_name().value(),
                  file.BitVectorType(1, SourceInfo()), SourceInfo());
    done_signal_refs.push_back(ref);
    signal_refs[threads_[index]->done_signal_name().value()] = ref;
  }

  {
    // Generate the clock. It has a frequency of two time units. Start clk at 0
    // at time zero to avoid any races with rising edge of clock and
    // any initialization.
    Initial* initial = m->Add<Initial>(SourceInfo());
    initial->statements()->Add<NonblockingAssignment>(
        SourceInfo(), clk, file.PlainLiteral(0, SourceInfo()));
    initial->statements()->Add<Forever>(
        SourceInfo(),
        file.Make<DelayStatement>(
            SourceInfo(), file.PlainLiteral(1, SourceInfo()),
            file.Make<BlockingAssignment>(SourceInfo(), clk,
                                          file.LogicalNot(clk, SourceInfo()))));
  }

  {
    // Global reset controller. If a reset is present, the threads will wait
    // on this reset prior to starting their execution.
    if (shared_data_.reset.has_value()) {
      LogicRef* reset = signal_refs.at(shared_data_.reset->name());
      const Bits reset_start_value =
          UBits(shared_data_.reset->active_low() ? 0 : 1, 1);
      const Bits reset_end_value =
          UBits(shared_data_.reset->active_low() ? 1 : 0, 1);
      Initial* initial = m->Add<Initial>(SourceInfo());
      initial->statements()->Add<NonblockingAssignment>(
          SourceInfo(), reset, file.Literal(reset_start_value, SourceInfo()));
      WaitNCycles(clk, initial->statements(), kResetCycles);
      initial->statements()->Add<NonblockingAssignment>(
          SourceInfo(), reset, file.Literal(reset_end_value, SourceInfo()));
      WaitNCycles(clk, initial->statements(), 1);
    }
  }

  {
    // Add a watchdog which stops the simulation after a long time.
    Initial* initial = m->Add<Initial>(SourceInfo());
    WaitNCycles(clk, initial->statements(), kSimulationCycleLimit);
    initial->statements()->Add<Display>(
        SourceInfo(), std::vector<Expression*>{file.Make<QuotedString>(
                          SourceInfo(), GetTimeoutMessage())});
    initial->statements()->Add<Finish>(SourceInfo());
  }

  {
    // Add a monitor statement which prints out all the port values.
    Initial* initial = m->Add<Initial>(SourceInfo());
    initial->statements()->Add<Display>(
        SourceInfo(),
        std::vector<Expression*>{file.Make<QuotedString>(
            SourceInfo(),
            "Starting. Clock rises at start of odd time units:")});
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

  for (int64_t index = 0; index < threads_.size(); ++index) {
    // TODO(vmirian) : Consider lowering the thread to an 'always' block.
    Initial* initial = m->Add<Initial>(SourceInfo());
    threads_[index]->EmitInto(initial, signal_refs);
  }

  {
    // Add a finish statement when all threads are complete.

    // The sensitivity_list must also include the clock to trigger the always
    // block since we are using conditions that verifies the level of the clock.
    std::vector<SensitivityListElement> sensitivity_list;
    sensitivity_list.push_back(clk);
    for (LogicRef* done_signal_ref : done_signal_refs) {
      sensitivity_list.push_back(done_signal_ref);
    }
    Always* always = m->Add<Always>(SourceInfo(), sensitivity_list);

    Expression* posedge_clk =
        file.Equals(clk, file.Literal(UBits(1, 1), SourceInfo()), SourceInfo());
    Expression* negedge_clk =
        file.Equals(clk, file.Literal(UBits(0, 1), SourceInfo()), SourceInfo());
    // Wait for a cycle to mimic the startup protocol in the threads.
    always->statements()->Add<WaitStatement>(SourceInfo(), posedge_clk);
    always->statements()->Add<WaitStatement>(SourceInfo(), negedge_clk);
    for (LogicRef* done_signal_ref : done_signal_refs) {
      Expression* cmp =
          file.Equals(done_signal_ref, file.Literal(UBits(1, 1), SourceInfo()),
                      SourceInfo());
      always->statements()->Add<WaitStatement>(SourceInfo(), cmp);
    }

    // Add one final wait for a cycle. This ensures that any $strobe from
    // DisplaySignal runs before the final $finish.
    always->statements()->Add<WaitStatement>(SourceInfo(), posedge_clk);
    always->statements()->Add<WaitStatement>(SourceInfo(), negedge_clk);

    always->statements()->Add<Finish>(SourceInfo());
  }
  // Concatentate the module Verilog with the testbench verilog to create the
  // verilog text to pass to the simulator.
  return absl::StrCat(verilog_text_, "\n\n", file.Emit());
}

absl::Status ModuleTestbench::Run() {
  std::string verilog_text = GenerateVerilog();
  XLS_VLOG(2) << verilog_text;

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
