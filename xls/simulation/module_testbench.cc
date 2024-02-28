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

#include <cstddef>
#include <cstdint>
#include <filesystem>  // NOLINT
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/ascii.h"
#include "absl/strings/match.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "absl/types/variant.h"
#include "xls/codegen/module_signature.h"
#include "xls/codegen/vast.h"
#include "xls/common/file/temp_directory.h"
#include "xls/common/logging/log_lines.h"
#include "xls/common/logging/logging.h"
#include "xls/common/source_location.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/bits.h"
#include "xls/ir/bits_ops.h"
#include "xls/ir/format_preference.h"
#include "xls/ir/number_parser.h"
#include "xls/ir/source_location.h"
#include "xls/simulation/module_testbench_thread.h"
#include "xls/simulation/testbench_metadata.h"
#include "xls/simulation/testbench_signal_capture.h"
#include "xls/simulation/testbench_stream.h"
#include "xls/simulation/verilog_simulator.h"
#include "xls/tools/verilog_include.h"
#include "re2/re2.h"

namespace xls {
namespace verilog {
namespace {

// The number of cycles that the design under test (DUT) is being reset.
static constexpr int64_t kResetCycles = 5;

std::string GetTimeoutMessage(int64_t cycle_limit) {
  return absl::StrFormat("ERROR: timeout, simulation ran too long (%d cycles).",
                         cycle_limit);
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
      CHECK_EQ(port.direction, Direction::kOutput);
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

// Returns all of the DUT inputs for the DUT described by `metadata`. Excludes
// clock and optionally reset.
std::vector<DutInput> GetAllDutInputs(const TestbenchMetadata& metadata,
                                      ZeroOrX initial_values,
                                      bool exclude_reset) {
  std::vector<DutInput> dut_inputs;
  auto get_initial_value = [&](int64_t width) {
    return initial_values == ZeroOrX::kZero ? BitsOrX{UBits(0, width)}
                                            : BitsOrX{IsX()};
  };
  for (const std::string& port : metadata.dut_input_ports()) {
    // Exclude the clock as an input and reset if `exclude_reset` is true.
    if ((metadata.clk_name().has_value() &&
         metadata.clk_name().value() == port) ||
        (exclude_reset && metadata.reset_proto().has_value() &&
         metadata.reset_proto()->name() == port)) {
      continue;
    }
    dut_inputs.push_back(DutInput{
        .port_name = port,
        .initial_value = get_initial_value(metadata.GetPortWidth(port))});
  }
  return dut_inputs;
}

}  // namespace

absl::StatusOr<std::unique_ptr<ModuleTestbench>>
ModuleTestbench::CreateFromVastModule(
    Module* module, const VerilogSimulator* simulator,
    std::optional<std::string_view> clk_name,
    const std::optional<ResetProto>& reset,
    absl::Span<const VerilogInclude> includes,
    std::optional<int64_t> simulation_cycle_limit) {
  XLS_ASSIGN_OR_RETURN(ModuleSignature signature,
                       GenerateModuleSignature(module, clk_name, reset));
  TestbenchMetadata metadata(signature);
  auto tb = absl::WrapUnique(new ModuleTestbench(
      module->file()->Emit(),
      module->file()->use_system_verilog() ? FileType::kSystemVerilog
                                           : FileType::kVerilog,
      simulator, metadata, /*reset_dut=*/reset.has_value(), includes,
      simulation_cycle_limit));
  XLS_RETURN_IF_ERROR(tb->CreateInitialThreads());
  return std::move(tb);
}

absl::StatusOr<std::unique_ptr<ModuleTestbench>>
ModuleTestbench::CreateFromVerilogText(
    std::string_view verilog_text, FileType file_type,
    const ModuleSignature& signature, const VerilogSimulator* simulator,
    bool reset_dut, absl::Span<const VerilogInclude> includes,
    std::optional<int64_t> simulation_cycle_limit) {
  TestbenchMetadata metadata(signature);
  auto tb = absl::WrapUnique(
      new ModuleTestbench(verilog_text, file_type, simulator, metadata,
                          reset_dut, includes, simulation_cycle_limit));
  XLS_RETURN_IF_ERROR(tb->CreateInitialThreads());
  return std::move(tb);
}

ModuleTestbench::ModuleTestbench(std::string_view verilog_text,
                                 FileType file_type,
                                 const VerilogSimulator* simulator,
                                 const TestbenchMetadata& metadata,
                                 bool reset_dut,
                                 absl::Span<const VerilogInclude> includes,
                                 std::optional<int64_t> simulation_cycle_limit)
    : verilog_text_(verilog_text),
      file_type_(file_type),
      simulator_(simulator),
      metadata_(metadata),
      reset_dut_(reset_dut),
      includes_(includes.begin(), includes.end()),
      simulation_cycle_limit_(simulation_cycle_limit),
      capture_manager_(&metadata_) {}

absl::Status ModuleTestbench::CreateInitialThreads() {
  // Global reset controller. If a reset is present, the threads will wait
  // on this reset prior to starting their execution.
  bool has_reset_thread = reset_dut_ && metadata_.reset_proto().has_value();
  if (has_reset_thread) {
    uint64_t reset_asserted_value =
        metadata_.reset_proto()->active_low() ? 0 : 1;
    uint64_t reset_unasserted_value =
        metadata_.reset_proto()->active_low() ? 1 : 0;

    XLS_ASSIGN_OR_RETURN(
        ModuleTestbenchThread * reset_thread,
        CreateThread("reset_controller",
                     {DutInput{std::string{metadata_.reset_proto()->name()},
                               UBits(reset_asserted_value, 1)}},
                     /*wait_until_done=*/false, /*wait_for_reset=*/false));
    reset_thread->DeclareInternalSignal(kLastResetCycleSignal, 1, UBits(0, 1));
    SequentialBlock& seq = reset_thread->MainBlock();
    seq.AdvanceNCycles(kResetCycles - 1);
    seq.Set(kLastResetCycleSignal, UBits(1, 1)).NextCycle();
    seq.Set(kLastResetCycleSignal, UBits(0, 1))
        .Set(metadata_.reset_proto()->name(), reset_unasserted_value);
  }

  if (simulation_cycle_limit_.has_value()) {
    XLS_ASSIGN_OR_RETURN(ModuleTestbenchThread * watchdog_thread,
                         CreateThread("watchdog",
                                      /*dut_inputs=*/{},
                                      /*wait_until_done=*/false,
                                      /*wait_for_reset=*/false));
    SequentialBlock& seq = watchdog_thread->MainBlock();
    seq.AdvanceNCycles(simulation_cycle_limit_.value() +
                       (has_reset_thread ? kResetCycles : 0));
    seq.Display(GetTimeoutMessage(simulation_cycle_limit_.value()));
    seq.Finish();
  }

  return absl::OkStatus();
}

absl::StatusOr<ModuleTestbenchThread*>
ModuleTestbench::CreateThreadDrivingAllInputs(std::string_view thread_name,
                                              ZeroOrX initial_value,
                                              bool wait_until_done) {
  return CreateThread(thread_name,
                      GetAllDutInputs(metadata_, initial_value,
                                      /*exclude_reset=*/reset_dut_),
                      wait_until_done,
                      /*wait_for_reset=*/true);
}

absl::StatusOr<ModuleTestbenchThread*> ModuleTestbench::CreateThread(
    std::string_view thread_name, absl::Span<const DutInput> dut_inputs,
    bool wait_until_done) {
  return CreateThread(thread_name, dut_inputs, wait_until_done,
                      /*wait_for_reset=*/true);
}

absl::StatusOr<ModuleTestbenchThread*> ModuleTestbench::CreateThread(
    std::string_view thread_name, absl::Span<const DutInput> dut_inputs,
    bool wait_until_done, bool wait_for_reset) {
  //  Verify thread name is unique.
  for (const std::unique_ptr<ModuleTestbenchThread>& t : threads_) {
    if (thread_name == t->name()) {
      return absl::InvalidArgumentError(
          absl::StrFormat("Already a thread named `%s`", thread_name));
    }
  }
  for (const DutInput& dut_input : dut_inputs) {
    // Verify port_name is valid.
    if (!metadata_.HasInputPortNamed(dut_input.port_name)) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "`%s` is not a input port on the DUT", dut_input.port_name));
    }
    if (claimed_dut_inputs_.contains(dut_input.port_name)) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "`%s` is already being drive by thread `%s`; cannot also be drive by "
          "thread `%s`",
          dut_input.port_name,
          claimed_dut_inputs_.at(dut_input.port_name)->name(), thread_name));
    }
  }

  auto testbench_thread = std::make_unique<ModuleTestbenchThread>(
      thread_name, &metadata_, &capture_manager_, dut_inputs, wait_until_done,
      wait_for_reset);
  for (const DutInput& dut_input : dut_inputs) {
    claimed_dut_inputs_[dut_input.port_name] = testbench_thread.get();
  }

  threads_.push_back(std::move(testbench_thread));
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

// Writes the captured sequence of values of the given signal/instance into the
// given Bits vector. `outputs` is index by instance_id as returned by
// `ExtractSignalValues`.
static absl::Status RecordCapturedBitsVector(
    std::string_view signal_name, int64_t instance_id,
    const absl::flat_hash_map<int64_t, std::vector<BitsOrX>>& outputs,
    std::vector<Bits>* bits_vector) {
  bits_vector->clear();
  if (outputs.contains(instance_id)) {
    for (int64_t i = 0; i < outputs.at(instance_id).size(); ++i) {
      const BitsOrX& output = outputs.at(instance_id)[i];
      if (std::holds_alternative<IsX>(output)) {
        return absl::NotFoundError(absl::StrFormat(
            "Output `%s`, instance #%d, recurrence %d holds X value in "
            "Verilog simulator output.",
            signal_name, instance_id, i));
      }
      bits_vector->push_back(std::get<Bits>(output));
    }
  }
  return absl::OkStatus();
}

// Writes the captured single value of the given signal/instance into the given
// Bits. `outputs` is index by instance_id as returned by `ExtractSignalValues`.
static absl::Status RecordCapturedBits(
    std::string_view signal_name, int64_t instance_id,
    const absl::flat_hash_map<int64_t, std::vector<BitsOrX>>& outputs,
    Bits* bits) {
  if (!outputs.contains(instance_id)) {
    return absl::NotFoundError(absl::StrFormat(
        "Output `%s`, instance #%d not found in Verilog simulator output.",
        signal_name, instance_id));
  }
  XLS_RET_CHECK_EQ(outputs.at(instance_id).size(), 1);
  const BitsOrX& bits_or_x = outputs.at(instance_id).front();
  // Write out module signal values to pointers passed in via Capture
  // calls.
  if (std::holds_alternative<IsX>(bits_or_x)) {
    return absl::NotFoundError(
        absl::StrFormat("Output `%s`, instance #%d holds X value in "
                        "Verilog simulator output.",
                        signal_name, instance_id));
  }
  *bits = std::get<Bits>(bits_or_x);
  return absl::OkStatus();
}

// Checks all of the captured values of the given signal/instance against the
// given expectation. `outputs` is index by instance_id as returned by
// `ExtractSignalValues`.
static absl::Status CheckCapturedSignalAgainstExpectation(
    std::string_view signal_name, int64_t instance_id,
    const TestbenchExpectation& expectation,
    const absl::flat_hash_map<int64_t, std::vector<BitsOrX>>& outputs) {
  // The captured signal may appear zero or more times in the output. Check
  // every instance.
  if (outputs.contains(instance_id)) {
    auto get_source_location = [&]() {
      return absl::StrFormat("%s@%d", expectation.loc.file_name(),
                             expectation.loc.line());
    };
    for (int64_t i = 0; i < outputs.at(instance_id).size(); ++i) {
      const BitsOrX& bits_or_x = outputs.at(instance_id)[i];
      std::string instance_name =
          absl::StrFormat("output `%s`, instance #%d, recurrence %d",
                          signal_name, instance_id, i);
      if (std::holds_alternative<Bits>(expectation.expected)) {
        const Bits& expected_bits = std::get<Bits>(expectation.expected);
        if (std::holds_alternative<IsX>(bits_or_x)) {
          return absl::FailedPreconditionError(absl::StrFormat(
              "%s: expected %s to have value: %v, has X", get_source_location(),
              instance_name, expected_bits));
        }
        const Bits& actual_bits = std::get<Bits>(bits_or_x);
        if (actual_bits != expected_bits) {
          return absl::FailedPreconditionError(
              absl::StrFormat("%s: expected %s to have value: %v, actual: %v",
                              get_source_location(), instance_name,
                              expected_bits, actual_bits));
        }
      } else {
        CHECK(std::holds_alternative<IsX>(expectation.expected));
        if (std::holds_alternative<Bits>(bits_or_x)) {
          return absl::FailedPreconditionError(absl::StrFormat(
              "%s: expected %s to have X value, has non X value: %v",
              get_source_location(), instance_name, std::get<Bits>(bits_or_x)));
        }
      }
    }
  }
  return absl::OkStatus();
}

absl::Status ModuleTestbench::CaptureOutputsAndCheckExpectations(
    std::string_view stdout_str) const {
  // Check for timeout.
  if (simulation_cycle_limit_.has_value() &&
      absl::StrContains(stdout_str,
                        GetTimeoutMessage(simulation_cycle_limit_.value()))) {
    return absl::DeadlineExceededError(
        absl::StrFormat("Simulation exceeded maximum length of %d cycles.",
                        simulation_cycle_limit_.value()));
  }

  absl::flat_hash_map<int64_t, std::vector<BitsOrX>> outputs;
  XLS_ASSIGN_OR_RETURN(outputs, ExtractSignalValues(stdout_str));

  for (const SignalCapture& signal_capture :
       capture_manager_.signal_captures()) {
    if (std::holds_alternative<const TestbenchStream*>(signal_capture.action)) {
      continue;
    }
    if (std::holds_alternative<std::vector<Bits>*>(signal_capture.action)) {
      // Capture multiple instances of the same signal.
      XLS_RETURN_IF_ERROR(RecordCapturedBitsVector(
          signal_capture.signal_name, signal_capture.instance_id, outputs,
          std::get<std::vector<Bits>*>(signal_capture.action)));
      continue;
    }

    if (std::holds_alternative<Bits*>(signal_capture.action)) {
      // Capture a single instance of a signal.
      XLS_RETURN_IF_ERROR(RecordCapturedBits(
          signal_capture.signal_name, signal_capture.instance_id, outputs,
          std::get<Bits*>(signal_capture.action)));
      continue;
    }

    // Check the signal value against any expectations.
    XLS_RET_CHECK(
        std::holds_alternative<TestbenchExpectation>(signal_capture.action));
    XLS_RETURN_IF_ERROR(CheckCapturedSignalAgainstExpectation(
        signal_capture.signal_name, signal_capture.instance_id,
        std::get<TestbenchExpectation>(signal_capture.action), outputs));
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

std::string ModuleTestbench::GenerateVerilog() const {
  VerilogFile file(file_type_);
  Module* m = file.AddModule("testbench", SourceInfo());

  LogicRef* clk = nullptr;
  absl::flat_hash_map<std::string, LogicRef*> signal_refs;
  std::vector<Connection> connections;

  // Create a `reg` for each DUT input port.
  for (const std::string& port_name : metadata_.dut_input_ports()) {
    if (metadata_.GetPortWidth(port_name) == 0) {
      // Skip zero-width inputs (e.g., empty tuples) as these have no actual
      // port in the Verilog module.
      continue;
    }
    LogicRef* ref = m->AddReg(
        port_name,
        file.BitVectorType(metadata_.GetPortWidth(port_name), SourceInfo()),
        SourceInfo());
    signal_refs[port_name] = ref;
    if (metadata_.IsClock(port_name)) {
      clk = ref;
    }
    connections.push_back(Connection{port_name, ref});
  }

  // Create a `wire` for each DUT output port.
  for (const std::string& port_name : metadata_.dut_output_ports()) {
    if (metadata_.GetPortWidth(port_name) == 0) {
      // Skip zero-width inputs (e.g., empty tuples) as these have no actual
      // port in the Verilog module.
      continue;
    }
    LogicRef* ref = m->AddWire(
        port_name,
        file.BitVectorType(metadata_.GetPortWidth(port_name), SourceInfo()),
        SourceInfo());
    signal_refs[port_name] = ref;
    connections.push_back(Connection{port_name, ref});
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

  // Create emitters for emitting Verilog code for handling file I/O. Add any
  // declarations for handling IO to/from streams. And emit code to open files.
  absl::flat_hash_map<std::string, VastStreamEmitter> stream_emitters;
  if (!streams_.empty()) {
    m->Add<BlankLine>(SourceInfo());
    m->Add<Comment>(SourceInfo(),
                    "Variable declarations for supporting streaming I/O.");
    // Declare variables required for performing IO.
    for (const std::unique_ptr<TestbenchStream>& stream : streams_) {
      stream_emitters.insert(
          {stream->name, VastStreamEmitter::Create(*stream, m)});
    }

    m->Add<BlankLine>(SourceInfo());
    m->Add<Comment>(SourceInfo(), "Open files for I/O.");
    Initial* initial = m->Add<Initial>(SourceInfo());
    for (const std::unique_ptr<TestbenchStream>& stream : streams_) {
      stream_emitters.at(stream->name).EmitOpen(initial->statements());
    }
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
            absl::StrFormat("Clock rises at %d, %d, %d, ....", kClockPeriod / 2,
                            kClockPeriod + kClockPeriod / 2,
                            2 * kClockPeriod + kClockPeriod / 2))});
    initial->statements()->Add<Display>(
        SourceInfo(),
        std::vector<Expression*>{file.Make<QuotedString>(
            SourceInfo(), "Signals driven one time unit after rising clock.")});
    initial->statements()->Add<Display>(
        SourceInfo(),
        std::vector<Expression*>{file.Make<QuotedString>(
            SourceInfo(),
            "Signals sampled one time unit before rising clock.")});
    initial->statements()->Add<Display>(
        SourceInfo(),
        std::vector<Expression*>{file.Make<QuotedString>(
            SourceInfo(), "Starting simulation. Monitor output:")});
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

  // Emit threads.
  std::vector<LogicRef*> thread_done_signals;
  for (const auto& thread : threads_) {
    m->Add<BlankLine>(SourceInfo());
    std::vector<std::string> dut_input_names;
    for (const DutInput& dut_input : thread->dut_inputs()) {
      dut_input_names.push_back(dut_input.port_name);
    }
    m->Add<Comment>(
        SourceInfo(),
        absl::StrFormat("Thread `%s`. Drives signals: %s", thread->name(),
                        dut_input_names.empty()
                            ? "<none>"
                            : absl::StrJoin(dut_input_names, ", ")));
    thread->EmitInto(m, clk, &signal_refs, stream_emitters);
    if (thread->done_signal_name().has_value()) {
      thread_done_signals.push_back(
          signal_refs.at(thread->done_signal_name().value()));
    }
  }

  {
    // Add a finish statement when all threads are complete.
    m->Add<BlankLine>(SourceInfo());
    m->Add<Comment>(SourceInfo(), "Thread completion monitor.");
    Initial* initial = m->Add<Initial>(SourceInfo());
    Expression* posedge_clk = file.Make<PosEdge>(SourceInfo(), clk);
    if (!thread_done_signals.empty()) {
      // Sample at one cycle before clock posedge.
      initial->statements()->Add<EventControl>(SourceInfo(), posedge_clk);
      initial->statements()->Add<DelayStatement>(
          SourceInfo(), file.PlainLiteral(kClockPeriod - 1, SourceInfo()));
      Expression* all_done = thread_done_signals.front();
      for (int64_t i = 1; i < thread_done_signals.size(); ++i) {
        all_done =
            file.LogicalAnd(all_done, thread_done_signals[i], SourceInfo());
      }
      auto whle = initial->statements()->Add<WhileStatement>(
          SourceInfo(), file.LogicalNot(all_done, SourceInfo()));
      whle->statements()->Add<EventControl>(SourceInfo(), posedge_clk);
      whle->statements()->Add<DelayStatement>(
          SourceInfo(), file.PlainLiteral(kClockPeriod - 1, SourceInfo()));
    }

    initial->statements()->Add<EventControl>(SourceInfo(), posedge_clk);
    initial->statements()->Add<DelayStatement>(
        SourceInfo(), file.PlainLiteral(1, SourceInfo()));

    // Close any open files.
    for (const std::unique_ptr<TestbenchStream>& stream : streams_) {
      stream_emitters.at(stream->name).EmitClose(initial->statements());
    }

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

absl::Status ModuleTestbench::Run() const {
  XLS_VLOG(1) << "ModuleTestbench::Run()";
  if (!streams_.empty()) {
    return absl::InvalidArgumentError(
        "Testbenches with streaming IO should be run with RunWithStreamingIO");
  }
  std::string verilog_text = GenerateVerilog();
  XLS_VLOG_LINES(3, verilog_text);
  std::pair<std::string, std::string> stdout_stderr;
  XLS_ASSIGN_OR_RETURN(stdout_stderr,
                       simulator_->Run(verilog_text, file_type_,
                                       /*macro_definitions=*/{}, includes_));
  XLS_VLOG(2) << "Verilog simulator stdout:\n" << stdout_stderr.first;
  XLS_VLOG(2) << "Verilog simulator stderr:\n" << stdout_stderr.second;
  const std::string& stdout_str = stdout_stderr.first;
  return CaptureOutputsAndCheckExpectations(stdout_str);
}

absl::Status ModuleTestbench::RunWithStreamingIo(
    const absl::flat_hash_map<std::string, TestbenchStreamThread::Producer>&
        input_producers,
    const absl::flat_hash_map<std::string, TestbenchStreamThread::Consumer>&
        output_consumers) const {
  XLS_VLOG(1) << "ModuleTestbench::RunWithStreamingIo()";

  // Verify all inputs and output consumer/producers are there.
  for (const std::unique_ptr<TestbenchStream>& stream : streams_) {
    if (stream->direction == TestbenchStreamDirection::kInput) {
      if (!input_producers.contains(stream->name)) {
        return absl::InvalidArgumentError(absl::StrFormat(
            "Missing producer for input stream `%s`", stream->name));
      }
    } else {
      if (!output_consumers.contains(stream->name)) {
        return absl::InvalidArgumentError(absl::StrFormat(
            "Missing consumer for output stream `%s`", stream->name));
      }
    }
  }
  if (input_producers.size() + output_consumers.size() > streams_.size()) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Too many producers/consumers specified (%d). Expected %d.",
        input_producers.size() + output_consumers.size(), streams_.size()));
  }

  std::string verilog_text = GenerateVerilog();
  XLS_VLOG_LINES(3, verilog_text);

  XLS_ASSIGN_OR_RETURN(TempDirectory temp_dir, TempDirectory::Create());
  std::vector<VerilogSimulator::MacroDefinition> macro_definitions;

  std::vector<TestbenchStreamThread> stream_threads;
  stream_threads.reserve(streams_.size());
  for (const std::unique_ptr<TestbenchStream>& stream : streams_) {
    std::filesystem::path stream_path = temp_dir.path() / stream->name;
    XLS_ASSIGN_OR_RETURN(TestbenchStreamThread thread,
                         TestbenchStreamThread::Create(*stream, stream_path));
    stream_threads.push_back(std::move(thread));
    if (stream->direction == TestbenchStreamDirection::kInput) {
      stream_threads.back().RunInputStream(input_producers.at(stream->name));
    } else {
      stream_threads.back().RunOutputStream(output_consumers.at(stream->name));
    }
    macro_definitions.push_back(VerilogSimulator::MacroDefinition{
        stream->path_macro_name,
        absl::StrFormat("\"%s\"", stream_path.string())});
  }
  XLS_VLOG(1) << "Starting simulation.";
  std::pair<std::string, std::string> stdout_stderr;
  XLS_ASSIGN_OR_RETURN(
      stdout_stderr,
      simulator_->Run(verilog_text, file_type_, macro_definitions, includes_));

  XLS_VLOG(1) << "Simulation done.";

  for (TestbenchStreamThread& thread : stream_threads) {
    XLS_RETURN_IF_ERROR(thread.Join());
  }

  XLS_VLOG(2) << "Verilog simulator stdout:\n" << stdout_stderr.first;
  XLS_VLOG(2) << "Verilog simulator stderr:\n" << stdout_stderr.second;

  const std::string& stdout_str = stdout_stderr.first;
  return CaptureOutputsAndCheckExpectations(stdout_str);
}

static std::string GetPipePathMacroName(std::string_view stream_name) {
  return absl::StrFormat("__%s_PIPE_PATH", absl::AsciiStrToUpper(stream_name));
}

absl::StatusOr<const TestbenchStream*> ModuleTestbench::CreateInputStream(
    std::string_view name, int64_t width) {
  if (stream_names_.contains(name)) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Already a I/O stream named `%s`", name));
  }
  stream_names_.insert(std::string{name});
  streams_.push_back(absl::WrapUnique(
      new TestbenchStream{.name = std::string{name},
                          .direction = TestbenchStreamDirection::kInput,
                          .path_macro_name = GetPipePathMacroName(name),
                          .width = width}));
  return streams_.back().get();
}

absl::StatusOr<const TestbenchStream*> ModuleTestbench::CreateOutputStream(
    std::string_view name, int64_t width) {
  if (stream_names_.contains(name)) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Already a I/O stream named `%s`", name));
  }
  stream_names_.insert(std::string{name});
  streams_.push_back(absl::WrapUnique(
      new TestbenchStream{.name = std::string{name},
                          .direction = TestbenchStreamDirection::kOutput,
                          .path_macro_name = GetPipePathMacroName(name),
                          .width = width}));
  return streams_.back().get();
}

}  // namespace verilog
}  // namespace xls
