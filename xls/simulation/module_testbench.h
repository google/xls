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

#ifndef XLS_CODEGEN_MODULE_TESTBENCH_H_
#define XLS_CODEGEN_MODULE_TESTBENCH_H_

#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include "absl/container/btree_map.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "xls/codegen/module_signature.h"
#include "xls/codegen/vast.h"
#include "xls/common/logging/logging.h"
#include "xls/common/source_location.h"
#include "xls/ir/bits.h"
#include "xls/simulation/verilog_simulator.h"

namespace xls {
namespace verilog {

struct ModuleTestbenchData {
  // The design-under-test module name.
  std::string_view dut_module_name;
  // Map of each input/output port name to its width.
  absl::btree_map<std::string, int64_t> input_port_widths;
  absl::btree_map<std::string, int64_t> output_port_widths;
};

// Provides a fluent interface for driving inputs, capturing outputs, and
// setting expectations.
class ModuleTestbenchThread {
 public:
  // The shared_data is data that is shared amongst all threads. Cannot be
  // nullptr.
  //
  // The init_values_after_reset is a name-value map with the name of the
  // input signals mapped to their initial value after reset.
  //
  // The inputs_to_drive are the input signals of the design under test
  // (DUT) that the thread is capable of driving, std::nullopt signifies all
  // inputs (no constraint on the inputs).
  // TODO(vmirian) : Consider merging init_values_after_reset and
  // inputs_to_drive.
  ModuleTestbenchThread(
      const ModuleTestbenchData* shared_data,
      absl::flat_hash_map<std::string, Bits> init_values_after_reset = {},
      std::optional<std::vector<std::string>> inputs_to_drive = std::nullopt)
      : shared_data_(XLS_DIE_IF_NULL(shared_data)),
        init_values_after_reset_(std::move(init_values_after_reset)),
        inputs_to_drive_(std::move(inputs_to_drive)) {}

  // Sets the given module input port to the given value in the current
  // cycle. The value is sticky and remains driven to this value across cycle
  // boundaries until it is Set again (if ever).
  ModuleTestbenchThread& Set(std::string_view input_port, const Bits& value);
  ModuleTestbenchThread& Set(std::string_view input_port, uint64_t value);

  // Sets the given module input to the unknown value in the current cycle. As
  // with Set() this is sticky.
  ModuleTestbenchThread& SetX(std::string_view input_port);

  // Advances the simulation the given number of cycles.
  ModuleTestbenchThread& AdvanceNCycles(int64_t n_cycles);

  // Wait for a given single-bit output port to be asserted (unasserted). If
  // the signal is already asserted (unasserted), this action takes no simulator
  // time.
  ModuleTestbenchThread& WaitFor(std::string_view output_port);
  ModuleTestbenchThread& WaitForNot(std::string_view output_port);

  // Wait for the given outputs to have X or non-X values. The output is
  // considered to have an X value if *any* bit is X. The outputs may have
  // arbitrary width.
  ModuleTestbenchThread& WaitForX(std::string_view output_port);
  ModuleTestbenchThread& WaitForNotX(std::string_view output_port);

  // Advances the simulation a single cycle. Equivalent to AdvanceNCycles(1).
  ModuleTestbenchThread& NextCycle();

  // Captures the value of the output port at the current cycle. The given
  // pointer value is written with the output port value when Run is called.
  ModuleTestbenchThread& Capture(std::string_view output_port, Bits* value);

  // Expects the given output port is the given value (or X) in the current
  // cycle. An error is returned during Run if this expectation is not met.
  //
  // "loc" indicates the source position in the test where the expectation was
  // created, and is displayed on expectation failure.
  ModuleTestbenchThread& ExpectEq(
      std::string_view output_port, const Bits& expected,
      xabsl::SourceLocation loc = xabsl::SourceLocation::current());
  ModuleTestbenchThread& ExpectEq(
      std::string_view output_port, uint64_t expected,
      xabsl::SourceLocation loc = xabsl::SourceLocation::current());
  // Similar to ExpectEq, but expects the given output port to be X. For this
  // purpose an output port is considered to have the value X if *any* bit is X.
  ModuleTestbenchThread& ExpectX(
      std::string_view output_port,
      xabsl::SourceLocation loc = xabsl::SourceLocation::current());

  // Expect to find a particular string in the simulation output,
  // typically generated by a trace in the dut.
  // Trace strings must be found in the order of the ExpectTrace calls.
  // TODO(amfv): 2021-09-02 Figure out how to associate dut traces with
  // significant events during the test (e.g. the activation of a trace
  // because an input changed).
  ModuleTestbenchThread& ExpectTrace(std::string_view trace_message);

  // Sentinel type for indicating an "X value, in lieu of some real bits value.
  struct IsX {};

  // A pair of instance number and port name used as a key for associating a
  // output display statement with a particular Capture or ExpectEq.
  using InstancePort = std::pair<int64_t, std::string>;

  // Checks the stdout of a simulation run against expectations.
  absl::Status CheckOutput(
      std::string_view stdout_str,
      const absl::flat_hash_map<InstancePort, std::variant<Bits, IsX>>&
          parsed_values) const;

  // Emit the thread contents into the verilog file with the contents specified.
  void EmitInto(StructuredProcedure* procedure, LogicRef* done_signal,
                const absl::flat_hash_map<std::string, LogicRef*>& port_refs,
                LogicRef* clk, std::optional<LogicRef*> reset,
                std::optional<Bits> reset_end_value);

 private:
  // Returns the width of the given port.
  int64_t GetPortWidth(std::string_view port);

  // CHECKs whether the given name is an input port that the thread is
  // designated to drive.
  void CheckIsMyInput(std::string_view name);
  // CHECKs whether the given name is an input/output port.
  void CheckIsInput(std::string_view name);
  void CheckIsOutput(std::string_view name);

  const ModuleTestbenchData* shared_data_;
  absl::flat_hash_map<std::string, Bits> init_values_after_reset_;
  std::optional<std::vector<std::string>> inputs_to_drive_;

  // The following structs define the actions which are set to occur during
  // simulation. This list of actions is built up by calling the methods on
  // ModuleTestbench (e.g., Set()).

  // Advances the current cycle a certain amount.
  struct AdvanceCycle {
    int64_t amount;
  };

  // Drives the module input to a concrete value.
  struct SetInput {
    std::string port;
    Bits value;
  };

  // Drives the module input to an unknown value.
  struct SetInputX {
    std::string port;
  };

  // Sentinel type for indicating an "not X" value, in lieu of some real bits
  // value.
  struct IsNotX {};

  // Waits for an output port to equal a certain value.
  struct WaitForOutput {
    std::string port;
    std::variant<Bits, IsX, IsNotX> value;
  };

  // Inserts a Verilog display statement which prints the value of the given
  // port.
  struct DisplayOutput {
    std::string port;

    // A unique identifier which associates this display statement with a
    // particular Capture or ExpectEq call.
    int64_t instance;
  };

  // The list of actions to perform during simulation.
  using Action = std::variant<AdvanceCycle, SetInput, SetInputX, WaitForOutput,
                              DisplayOutput>;
  std::vector<Action> actions_;

  // A map containing the pointers passed in to each Capture call. Use std::map
  // for stable iteration order.
  std::map<InstancePort, Bits*> captures_;

  // A map containing the expected values passed in to each ExpectEq call. Use
  // std::map for stable iteration order.
  struct Expectation {
    std::variant<Bits, IsX> expected;
    xabsl::SourceLocation loc;
  };
  std::map<InstancePort, Expectation> expectations_;

  // A increasing counter which is used to generate unique instance identifiers
  // for DisplayOutput and InstantPort objects.
  int64_t next_instance_ = 0;

  std::vector<std::string> expected_traces_;
};

// Test class which does a cycle-by-cycle simulation of a Verilog module.
class ModuleTestbench {
 public:
  // Constructor for testing a VAST-defined module with the given clock and
  // reset signal.
  ModuleTestbench(Module* module, const VerilogSimulator* simulator,
                  std::optional<std::string_view> clk_name = absl::nullopt,
                  std::optional<ResetProto> reset = absl::nullopt,
                  absl::Span<const VerilogInclude> includes = {});

  // Constructor for testing a module defined in Verilog text with an interface
  // described with a ModuleSignature.
  ModuleTestbench(std::string_view verilog_text, FileType file_type,
                  const ModuleSignature& signature,
                  const VerilogSimulator* simulator,
                  absl::Span<const VerilogInclude> includes = {});

  // Returns a reference to a newly created thread to execute in the testbench.
  //
  // The init_values_after_reset is a name-value map with the name of the
  // signals mapped to their initial value after reset.
  //
  // The inputs_to_drive are the input signals of the design under test (DUT)
  // that the thread is capable of driving, std::nullopt signifies all inputs
  // (no constraint on the inputs).
  ModuleTestbenchThread& CreateThread(
      absl::flat_hash_map<std::string, Bits> init_values_after_reset = {},
      std::optional<std::vector<std::string>> inputs_to_drive = std::nullopt);

  // Generates the Verilog representation of the testbench.
  std::string GenerateVerilog();

  // Runs the simulation.
  absl::Status Run();

 private:
  // Checks the stdout of a simulation run against expectations.
  absl::Status CheckOutput(std::string_view stdout_str) const;

  std::string verilog_text_;
  FileType file_type_;
  const VerilogSimulator* simulator_;
  std::optional<std::string> clk_name_;
  std::optional<ResetProto> reset_;
  absl::Span<const VerilogInclude> includes_;

  ModuleTestbenchData shared_data_;

  // A list of blocks that execute concurrently in the testbench, a.k.a.
  // 'threads'. The xls::verilog::ModuleTestbench::CreateThread function returns
  // a reference to a ModuleTestbenchThread, as a result the threads use
  // std::unique_ptr to avoid bad referencing when vector is resized.
  std::vector<std::unique_ptr<ModuleTestbenchThread>> threads_;
};

}  // namespace verilog
}  // namespace xls

#endif  // XLS_CODEGEN_MODULE_TESTBENCH_H_
