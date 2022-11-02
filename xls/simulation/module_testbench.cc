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
    std::string_view output_port) {
  XLS_CHECK_EQ(GetPortWidth(output_port), 1);
  actions_.push_back(WaitForOutput{std::string(output_port), UBits(1, 1)});
  return *this;
}

ModuleTestbenchThread& ModuleTestbenchThread::WaitForNot(
    std::string_view output_port) {
  XLS_CHECK_EQ(GetPortWidth(output_port), 1);
  actions_.push_back(WaitForOutput{std::string(output_port), UBits(0, 1)});
  return *this;
}

ModuleTestbenchThread& ModuleTestbenchThread::WaitForX(
    std::string_view output_port) {
  actions_.push_back(WaitForOutput{std::string(output_port), IsX{}});
  return *this;
}

ModuleTestbenchThread& ModuleTestbenchThread::WaitForNotX(
    std::string_view output_port) {
  actions_.push_back(WaitForOutput{std::string(output_port), IsNotX{}});
  return *this;
}

ModuleTestbenchThread& ModuleTestbenchThread::Set(std::string_view input_port,
                                                  const Bits& value) {
  CheckIsMyInput(input_port);
  CheckIsInput(input_port);
  actions_.push_back(SetInput{std::string(input_port), value});
  return *this;
}

ModuleTestbenchThread& ModuleTestbenchThread::Set(std::string_view input_port,
                                                  uint64_t value) {
  CheckIsMyInput(input_port);
  CheckIsInput(input_port);
  return Set(input_port, UBits(value, GetPortWidth(input_port)));
}

ModuleTestbenchThread& ModuleTestbenchThread::SetX(
    std::string_view input_port) {
  CheckIsMyInput(input_port);
  CheckIsInput(input_port);
  actions_.push_back(SetInputX{std::string(input_port)});
  return *this;
}

ModuleTestbenchThread& ModuleTestbenchThread::ExpectEq(
    std::string_view output_port, const Bits& expected,
    xabsl::SourceLocation loc) {
  CheckIsOutput(output_port);
  int64_t instance = next_instance_++;
  auto key = std::make_pair(instance, std::string(output_port));
  XLS_CHECK(expectations_.find(key) == expectations_.end());
  expectations_[key] = Expectation{.expected = expected, .loc = loc};
  actions_.push_back(DisplayOutput{std::string(output_port), instance});
  return *this;
}

ModuleTestbenchThread& ModuleTestbenchThread::ExpectEq(
    std::string_view output_port, uint64_t expected,
    xabsl::SourceLocation loc) {
  CheckIsOutput(output_port);
  return ExpectEq(output_port, UBits(expected, GetPortWidth(output_port)), loc);
}

ModuleTestbenchThread& ModuleTestbenchThread::ExpectX(
    std::string_view output_port, xabsl::SourceLocation loc) {
  CheckIsOutput(output_port);
  int64_t instance = next_instance_++;
  auto key = std::make_pair(instance, std::string(output_port));
  XLS_CHECK(expectations_.find(key) == expectations_.end());
  expectations_[key] = Expectation{.expected = IsX(), .loc = loc};
  actions_.push_back(DisplayOutput{std::string(output_port), instance});
  return *this;
}

ModuleTestbenchThread& ModuleTestbenchThread::ExpectTrace(
    std::string_view trace_message) {
  expected_traces_.push_back(std::string(trace_message));
  return *this;
}

ModuleTestbenchThread& ModuleTestbenchThread::Capture(
    std::string_view output_port, Bits* value) {
  CheckIsOutput(output_port);
  if (GetPortWidth(output_port) > 0) {
    int64_t instance = next_instance_++;
    actions_.push_back(DisplayOutput{std::string(output_port), instance});
    auto key = std::make_pair(instance, std::string(output_port));
    captures_[key] = value;
  }
  return *this;
}

int64_t ModuleTestbenchThread::GetPortWidth(std::string_view port) {
  if (shared_data_->input_port_widths.contains(port)) {
    return shared_data_->input_port_widths.at(port);
  }
  return shared_data_->output_port_widths.at(port);
}

void ModuleTestbenchThread::CheckIsMyInput(std::string_view name) {
  if (inputs_to_drive_.has_value()) {
    absl::Span<const std::string> inputs = inputs_to_drive_.value();
    XLS_CHECK(std::find(inputs.begin(), inputs.end(), name) != inputs.end())
        << absl::StrFormat(
               "'%s' is not an input port that the thread is designated to "
               "drive.",
               name);
  }
}

void ModuleTestbenchThread::CheckIsInput(std::string_view name) {
  XLS_CHECK(shared_data_->input_port_widths.contains(name))
      << absl::StrFormat("'%s' is not an input port of module '%s'", name,
                         shared_data_->dut_module_name);
}

void ModuleTestbenchThread::CheckIsOutput(std::string_view name) {
  XLS_CHECK(shared_data_->output_port_widths.contains(name))
      << absl::StrFormat("'%s' is not an output port of module '%s'", name,
                         shared_data_->dut_module_name);
}

absl::Status ModuleTestbenchThread::CheckOutput(
    std::string_view stdout_str,
    const absl::flat_hash_map<InstancePort, std::variant<Bits, IsX>>&
        parsed_values) const {
  // Write out module output port values to pointers passed in via Capture
  // calls.
  for (const auto& pair : captures_) {
    auto cycle_port = pair.first;
    Bits* value_ptr = pair.second;
    if (!parsed_values.contains(cycle_port)) {
      return absl::NotFoundError(absl::StrFormat(
          "Output %s, instance #%d not found in Verilog simulator output.",
          cycle_port.second, cycle_port.first));
    }
    if (std::holds_alternative<IsX>(parsed_values.at(cycle_port))) {
      return absl::NotFoundError(absl::StrFormat(
          "Output %s, instance #%d holds X value in Verilog simulator output.",
          cycle_port.second, cycle_port.first));
    }
    *value_ptr = std::get<Bits>(parsed_values.at(cycle_port));
  }

  // Check the module output port value against any expectations.
  for (const auto& pair : expectations_) {
    auto cycle_port = pair.first;
    const Expectation& expectation = pair.second;
    auto get_source_location = [&]() {
      return absl::StrFormat("%s@%d", expectation.loc.file_name(),
                             expectation.loc.line());
    };
    if (!parsed_values.contains(cycle_port)) {
      return absl::NotFoundError(absl::StrFormat(
          "%s: output '%s', instance #%d @ %s not found in Verilog "
          "simulator output.",
          get_source_location(), cycle_port.second, cycle_port.first,
          ToString(expectation.loc)));
    }
    if (std::holds_alternative<Bits>(expectation.expected)) {
      const Bits& expected_bits = std::get<Bits>(expectation.expected);
      if (std::holds_alternative<IsX>(parsed_values.at(cycle_port))) {
        return absl::FailedPreconditionError(absl::StrFormat(
            "%s: expected output '%s', instance #%d to have value: %s, has X",
            get_source_location(), cycle_port.second, cycle_port.first,
            expected_bits.ToString()));
      }
      const Bits& actual_bits = std::get<Bits>(parsed_values.at(cycle_port));
      if (actual_bits != expected_bits) {
        return absl::FailedPreconditionError(absl::StrFormat(
            "%s: expected output '%s', instance #%d to have value: %s, actual: "
            "%s",
            get_source_location(), cycle_port.second, cycle_port.first,
            expected_bits.ToString(), actual_bits.ToString()));
      }
    } else {
      XLS_CHECK(std::holds_alternative<IsX>(expectation.expected));
      if (std::holds_alternative<Bits>(parsed_values.at(cycle_port))) {
        return absl::FailedPreconditionError(absl::StrFormat(
            "%s: expected output '%s', instance #%d to have X value, has non X "
            "value: %s",
            get_source_location(), cycle_port.second, cycle_port.first,
            std::get<Bits>(parsed_values.at(cycle_port)).ToString()));
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
    StructuredProcedure* procedure, LogicRef* done_signal,
    const absl::flat_hash_map<std::string, LogicRef*>& port_refs, LogicRef* clk,
    std::optional<LogicRef*> reset, std::optional<Bits> reset_end_value) {
  XLS_CHECK_NE(procedure, nullptr);
  VerilogFile& file = *procedure->file();
  // The first statement must be to deassert the done signal of the thread.
  procedure->statements()->Add<NonblockingAssignment>(
      SourceInfo(), done_signal, file.Literal(UBits(0, 1), SourceInfo()));
  WaitNCycles(clk, procedure->statements(), 1);

  // Some values must be initialize after a reset, for example control signals.
  for (const auto& [port_name, value] : init_values_after_reset_) {
    procedure->statements()->Add<NonblockingAssignment>(
        SourceInfo(), port_refs.at(port_name),
        file.Literal(value, SourceInfo()));
  }
  // The thread must wait for reset (if present).
  if (reset.has_value() && reset_end_value.has_value()) {
    Expression* reset_finished = file.Equals(
        reset.value(), file.Literal(reset_end_value.value(), SourceInfo()),
        SourceInfo());
    procedure->statements()->Add<WaitStatement>(SourceInfo(), reset_finished);
  }
  // All actions occur at the falling edge of the clock to avoid races with
  // signals changing at the rising edge of the clock.
  for (const Action& action : actions_) {
    absl::visit(
        Visitor{[&](const AdvanceCycle& a) {
                  WaitNCycles(clk, procedure->statements(), a.amount);
                },
                [&](const SetInput& s) {
                  if (GetPortWidth(s.port) > 0) {
                    procedure->statements()->Add<NonblockingAssignment>(
                        SourceInfo(), port_refs.at(s.port),
                        file.Literal(s.value, SourceInfo()));
                  }
                },
                [&](const SetInputX& s) {
                  if (GetPortWidth(s.port) > 0) {
                    procedure->statements()->Add<NonblockingAssignment>(
                        SourceInfo(), port_refs.at(s.port),
                        file.Make<XSentinel>(SourceInfo(),
                                             GetPortWidth(s.port)));
                  }
                },
                [&](const WaitForOutput& w) {
                  // WaitForOutput waits until the signal equals a certain value
                  // at the falling edge of the clock. Use a while loop to
                  // sample every cycle at the falling edge of the clock.
                  // TODO(meheff): If we switch to SystemVerilog this is better
                  // handled using a clocking block.
                  Expression* cmp;
                  if (std::holds_alternative<Bits>(w.value)) {
                    cmp = file.NotEquals(
                        port_refs.at(w.port),
                        file.Literal(std::get<Bits>(w.value), SourceInfo()),
                        SourceInfo());
                  } else if (std::holds_alternative<IsX>(w.value)) {
                    cmp = file.NotEqualsX(port_refs.at(w.port), SourceInfo());
                  } else {
                    XLS_CHECK(std::holds_alternative<IsNotX>(w.value));
                    cmp = file.EqualsX(port_refs.at(w.port), SourceInfo());
                  }
                  auto whle = procedure->statements()->Add<WhileStatement>(
                      SourceInfo(), cmp);
                  WaitNCycles(clk, whle->statements(), 1);
                },
                [&](const DisplayOutput& c) {
                  // Use $strobe rather than $display to print value after all
                  // assignments in the simulator time slot and avoid any
                  // potential race conditions.
                  procedure->statements()->Add<Strobe>(
                      SourceInfo(),
                      std::vector<Expression*>{
                          file.Make<QuotedString>(
                              SourceInfo(),
                              absl::StrFormat("%%t OUTPUT %s = %d'h%%0x (#%d)",
                                              c.port, GetPortWidth(c.port),
                                              c.instance)),
                          file.Make<SystemFunctionCall>(SourceInfo(), "time"),
                          port_refs.at(c.port)});
                }},
        action);
  }
  // The last statement must be to assert the done signal of the thread.
  procedure->statements()->Add<NonblockingAssignment>(
      SourceInfo(), done_signal, file.Literal(UBits(1, 1), SourceInfo()));
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
      clk_name_(clk_name),
      reset_(reset),
      includes_(includes) {
  XLS_VLOG(3) << "Building ModuleTestbench for Verilog module:";
  XLS_VLOG_LINES(3, verilog_text_);
  shared_data_.dut_module_name = module->name();
  absl::btree_map<std::string, int64_t>& input_port_widths =
      shared_data_.input_port_widths;
  absl::btree_map<std::string, int64_t>& output_port_widths =
      shared_data_.output_port_widths;

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
  if (signature.proto().has_clock_name()) {
    clk_name_ = signature.proto().clock_name();
    input_port_widths[*clk_name_] = 1;
  }
  if (signature.proto().has_reset()) {
    reset_ = signature.proto().reset();
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
    absl::flat_hash_map<std::string, Bits> init_values_after_reset,
    std::optional<std::vector<std::string>> inputs_to_drive) {
  threads_.push_back(std::make_unique<ModuleTestbenchThread>(
      &shared_data_, init_values_after_reset, inputs_to_drive));
  return *threads_.back();
}

absl::Status ModuleTestbench::CheckOutput(std::string_view stdout_str) const {
  // Check for timeout.
  if (absl::StrContains(stdout_str, GetTimeoutMessage())) {
    return absl::DeadlineExceededError(
        absl::StrFormat("Simulation exceeded maximum length of %d cycles.",
                        kSimulationCycleLimit));
  }

  // Scan the simulator output and pick out the OUTPUT lines holding the value
  // of module output ports.
  absl::flat_hash_map<ModuleTestbenchThread::InstancePort,
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

  std::optional<LogicRef*> reset = std::nullopt;
  std::optional<Bits> reset_end_value = std::nullopt;

  absl::flat_hash_map<std::string, LogicRef*> port_refs;
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
    port_refs[port_name] = ref;
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
    port_refs[port_name] = ref;
    connections.push_back(Connection{port_name, ref});
  }
  // For combinational modules define, but do not connect a clock signal.
  LogicRef* clk =
      clk_name_.has_value()
          ? port_refs.at(*clk_name_)
          : m->AddReg("clk", file.ScalarType(SourceInfo()), SourceInfo());

  // Instantiate the device under test module.
  const char kInstantiationName[] = "dut";
  m->Add<Instantiation>(
      SourceInfo(), shared_data_.dut_module_name, kInstantiationName,
      /*parameters=*/absl::Span<const Connection>(), connections);

  std::vector<LogicRef*> done_signal_refs;
  for (int64_t index = 0; index < threads_.size(); ++index) {
    std::string done_signal_name =
        absl::StrFormat("is_thread_%s_done", std::to_string(index));
    LogicRef* ref = m->AddReg(
        done_signal_name, file.BitVectorType(1, SourceInfo()), SourceInfo());
    done_signal_refs.push_back(ref);
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
    if (reset_.has_value()) {
      reset = port_refs.at(reset_->name());
      const Bits reset_start_value = UBits(reset_->active_low() ? 0 : 1, 1);
      reset_end_value = UBits(reset_->active_low() ? 1 : 0, 1);
      Initial* initial = m->Add<Initial>(SourceInfo());
      initial->statements()->Add<NonblockingAssignment>(
          SourceInfo(), reset.value(),
          file.Literal(reset_start_value, SourceInfo()));
      WaitNCycles(clk, initial->statements(), kResetCycles);
      initial->statements()->Add<NonblockingAssignment>(
          SourceInfo(), reset.value(),
          file.Literal(reset_end_value.value(), SourceInfo()));
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
      if (connection.port_name == clk_name_) {
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
    threads_[index]->EmitInto(initial, done_signal_refs[index], port_refs, clk,
                              reset, reset_end_value);
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
    // DisplayOutput runs before the final $finish.
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
