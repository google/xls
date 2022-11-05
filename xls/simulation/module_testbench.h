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
  // In a ModuleTestBench, the port names also mirror the signal name used in a
  // connection.
  absl::btree_map<std::string, int64_t> input_port_widths;
  absl::btree_map<std::string, int64_t> output_port_widths;
  // The clock and reset names are global.
  std::string clk_name;
  std::optional<ResetProto> reset;
};

// Provides a fluent interface for driving inputs, capturing outputs, and
// setting expectations.
class ModuleTestbenchThread {
 public:
  // The shared_data is data that is shared amongst all threads. Cannot be
  // nullptr.
  //
  // The owned_signals_to_drive_ is a name-value map with the name of the
  // input signals mapped to their initial value after reset. A value of
  // std::nullopt signifies an 'X' in Verilog. The names in the map are also the
  // input signals of the design under test (DUT) that the thread is capable of
  // driving.
  //
  // The done_signal_name is the name of the thread's done signal. The signal is
  // used to notify the testbench that the thread is done with its execution. If
  // its value is std::nullopt, then the thread does not emit the done signal.
  ModuleTestbenchThread(
      const ModuleTestbenchData* shared_data,
      absl::flat_hash_map<std::string, std::optional<Bits>>
          owned_signals_to_drive,
      std::optional<std::string> done_signal_name = std::nullopt)
      : shared_data_(XLS_DIE_IF_NULL(shared_data)),
        owned_signals_to_drive_(std::move(owned_signals_to_drive)),
        done_signal_name_(std::move(done_signal_name)) {}

  std::optional<std::string> done_signal_name() const {
    return done_signal_name_;
  }

  // Sets the given signal to the given value in the current
  // cycle. The value is sticky and remains driven to this value across cycle
  // boundaries until it is Set again (if ever).
  ModuleTestbenchThread& Set(std::string_view signal_name, const Bits& value);
  ModuleTestbenchThread& Set(std::string_view signal_name, uint64_t value);

  // Sets the given signal to the unknown value in the current cycle. As
  // with Set() this is sticky.
  ModuleTestbenchThread& SetX(std::string_view signal_name);

  // Advances the simulation the given number of cycles.
  ModuleTestbenchThread& AdvanceNCycles(int64_t n_cycles);

  // Wait for a given single-bit signal to be asserted (unasserted). If
  // the signal is already asserted (unasserted), this action takes no simulator
  // time.
  ModuleTestbenchThread& WaitFor(std::string_view signal_name);
  ModuleTestbenchThread& WaitForNot(std::string_view signal_name);

  // Wait for the given signal to have X or non-X values. The signal is
  // considered to have an X value if *any* bit is X. The signals may have
  // arbitrary width.
  ModuleTestbenchThread& WaitForX(std::string_view signal_name);
  ModuleTestbenchThread& WaitForNotX(std::string_view signal_name);

  // The wait for a given signal to be equal/not equal to a value or 'X'. In
  // contrast to the 'WaitFor*' functions above, these function treat the
  // expression in the wait statement as an event and are triggered on
  // immediately.
  ModuleTestbenchThread& WaitForEvent(std::string_view signal_name, Bits value);
  ModuleTestbenchThread& WaitForEventNot(std::string_view signal_name,
                                         Bits value);
  ModuleTestbenchThread& WaitForEventX(std::string_view signal_name);
  ModuleTestbenchThread& WaitForEventNotX(std::string_view signal_name);

  // Advances the simulation a single cycle. Equivalent to AdvanceNCycles(1).
  ModuleTestbenchThread& NextCycle();

  // Captures the value of the signal at the current cycle. The given
  // pointer value is written with the signal value when Run is called.
  ModuleTestbenchThread& Capture(std::string_view signal_name, Bits* value);

  // Expects the given signal is the given value (or X) in the current
  // cycle. An error is returned during Run if this expectation is not met.
  //
  // "loc" indicates the source position in the test where the expectation was
  // created, and is displayed on expectation failure.
  ModuleTestbenchThread& ExpectEq(
      std::string_view signal_name, const Bits& expected,
      xabsl::SourceLocation loc = xabsl::SourceLocation::current());
  ModuleTestbenchThread& ExpectEq(
      std::string_view signal_name, uint64_t expected,
      xabsl::SourceLocation loc = xabsl::SourceLocation::current());
  // Similar to ExpectEq, but expects the given signal to be X. For this
  // purpose an signal is considered to have the value X if *any* bit is X.
  ModuleTestbenchThread& ExpectX(
      std::string_view signal_name,
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

  // A pair of instance number and signal name used as a key for associating a
  // output display statement with a particular Capture or ExpectEq.
  using InstanceSignalName = std::pair<int64_t, std::string>;

  // Checks the stdout of a simulation run against expectations.
  absl::Status CheckOutput(
      std::string_view stdout_str,
      const absl::flat_hash_map<InstanceSignalName, std::variant<Bits, IsX>>&
          parsed_values) const;

  // Emit the thread contents into the verilog file with the contents specified.
  void EmitInto(StructuredProcedure* procedure,
                const absl::flat_hash_map<std::string, LogicRef*>& signal_refs);

 private:
  // Returns the width of the given signal.
  int64_t GetSignalWidth(std::string_view name);

  // CHECKs whether the given name is a signal that the thread is
  // designated to drive.
  void CheckCanDriveSignal(std::string_view name);
  // CHECKs whether the given name is a signal that the thread can read.
  void CheckCanReadSignal(std::string_view name);

  const ModuleTestbenchData* shared_data_;
  // The owned_signals_to_drive_ is a name-value map with the name of the
  // input signals mapped to their initial value after reset. A value of
  // std::nullopt signifies an 'X' in Verilog. The names in the map are also the
  // input signals of the design under test (DUT) that the thread is capable of
  // driving.
  absl::flat_hash_map<std::string, std::optional<Bits>> owned_signals_to_drive_;
  // The name of the thread's done signal. The signal is used to notify the
  // testbench that the thread is done with its execution. If its value is
  // std::nullopt, then the thread does not emit the done signal.
  std::optional<std::string> done_signal_name_;

  // The following structs define the actions which are set to occur during
  // simulation. This list of actions is built up by calling the methods on
  // ModuleTestbench (e.g., Set()).

  // Advances the current cycle a certain amount.
  struct AdvanceCycle {
    int64_t amount;
  };

  // Drives the module input to a concrete value.
  struct SetSignal {
    std::string signal_name;
    Bits value;
  };

  // Drives the module input to an unknown value.
  struct SetSignalX {
    std::string signal_name;
  };

  // Sentinel type for indicating an "not X" value, in lieu of some real bits
  // value.
  struct IsNotX {};

  // Waits for a signal to equal a certain value.
  struct WaitForSignal {
    std::string signal_name;
    std::variant<Bits, IsX, IsNotX> value;
  };

  // Waits for a signal event to equal a certain value.
  struct WaitForSignalEvent {
    std::string signal_name;
    std::variant<Bits, IsX> value;
    bool is_comparison_equal;
  };

  // Inserts a Verilog display statement which prints the value of the given
  // signal.
  struct DisplaySignal {
    std::string signal_name;

    // A unique identifier which associates this display statement with a
    // particular Capture or ExpectEq call.
    int64_t instance;
  };

  // The list of actions to perform during simulation.
  using Action = std::variant<AdvanceCycle, SetSignal, SetSignalX,
                              WaitForSignal, DisplaySignal, WaitForSignalEvent>;
  std::vector<Action> actions_;

  // A map containing the pointers passed in to each Capture call. Use std::map
  // for stable iteration order.
  std::map<InstanceSignalName, Bits*> captures_;

  // A map containing the expected values passed in to each ExpectEq call. Use
  // std::map for stable iteration order.
  struct Expectation {
    std::variant<Bits, IsX> expected;
    xabsl::SourceLocation loc;
  };
  std::map<InstanceSignalName, Expectation> expectations_;

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
  // The owned_signals_to_drive is a name-value map with the name of the
  // input signals mapped to their initial value after reset. A value of
  // std::nullopt signifies an 'X' in Verilog. The names in the map are also the
  // input signals of the design under test (DUT) that the thread is capable of
  // driving. A std::nullopt value for the map signifies all inputs (no
  // constraint on the inputs). The latter is the default value of the map.
  ModuleTestbenchThread& CreateThread(
      std::optional<absl::flat_hash_map<std::string, std::optional<Bits>>>
          owned_signals_to_drive = std::nullopt,
      bool emit_done_signal = true);

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
