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
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include "absl/container/btree_map.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "xls/codegen/module_signature.h"
#include "xls/codegen/vast.h"
#include "xls/common/logging/logging.h"
#include "xls/common/source_location.h"
#include "xls/ir/bits.h"
#include "xls/simulation/verilog_simulator.h"

namespace xls {
namespace verilog {

// Sentinel type for indicating an "X" value, in lieu of some real bits value.
struct IsX {};

// Sentinel type for indicating an "not X" value, in lieu of some real bits
// value.
struct IsNotX {};

using BitsOrX = std::variant<Bits, IsX>;

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

// Represents signal to display during simulation using a Verilog $display
// statement.
struct DisplaySignal {
  std::string signal_name;
  int64_t width;
  // A unique identifier which associates this display statement with a
  // particular Capture or ExpectEq call.
  int64_t instance;
};

struct Expectation {
  BitsOrX expected;
  xabsl::SourceLocation loc;
};

class ModuleTestbenchThread;

// Data-structure representing the end of a cycle (one time unit before the
// rising edge of the clock). In the ModuleTestbench infrastructure signals are
// only sampled at the end of a cycle. The ModuleTestbenchThread API returns
// this object to enable capturing signals and `expect`ing their values. For
// example, in the following code `AtEndOfCycleWhen` returns a EndOfCycleEvent
// corresponding to the end of the cycle when `foo_valid` is first asserted, and
// at this point the value of `bar` is captured.
//
//    Bits bar;
//    testbench_thread.AtEndOfCycleWhen("foo_valid").Capture("bar", &bar);
class EndOfCycleEvent {
 public:
  explicit EndOfCycleEvent(ModuleTestbenchThread* thread) : thread_(thread) {}

  // Captures the value of the signal. The given pointer value is written with
  // the signal value when Run is called.
  EndOfCycleEvent& Capture(std::string_view signal_name, Bits* value);

  // Expects the given signal is the given value (or X). An error is returned
  // during Run if this expectation is not met.
  //
  // "loc" indicates the source position in the test where the expectation was
  // created, and is displayed on expectation failure.
  EndOfCycleEvent& ExpectEq(
      std::string_view signal_name, const Bits& expected,
      xabsl::SourceLocation loc = xabsl::SourceLocation::current());
  EndOfCycleEvent& ExpectEq(
      std::string_view signal_name, uint64_t expected,
      xabsl::SourceLocation loc = xabsl::SourceLocation::current());
  EndOfCycleEvent& ExpectX(
      std::string_view signal_name,
      xabsl::SourceLocation loc = xabsl::SourceLocation::current());

  absl::Span<const DisplaySignal> display_signals() const {
    return display_signals_;
  }

 private:
  ModuleTestbenchThread* thread_;

  // Set of signals to display during simulation.
  std::vector<DisplaySignal> display_signals_;
};

// The following structs define the actions which are set to occur during
// simulation. This list of actions is built up by calling the methods on
// ModuleTestbenchThread (e.g., Set()).

// Advances the current cycle a certain amount.
struct AdvanceCycle {
  int64_t amount;

  // Things to perform at the end of the cycle one time unit before posedge of
  // the clock.
  std::unique_ptr<EndOfCycleEvent> end_of_cycle_event;
};

// Drives the module input to a concrete value.
struct SetSignal {
  std::string signal_name;
  Bits value;
};

// Drives the module input to an unknown value.
struct SetSignalX {
  std::string signal_name;
  int64_t width;
};

// Waits for signals to equal a certain value.
enum class AnyOrAll { kAny, kAll };
struct SignalValue {
  std::string signal_name;
  std::variant<Bits, IsX, IsNotX> value;
};
struct WaitForSignals {
  AnyOrAll any_or_all;
  std::vector<SignalValue> signal_values;

  // Things to perform at the end of the cycle one time unit before posedge of
  // the clock when the condition is met.
  std::unique_ptr<EndOfCycleEvent> end_of_cycle_event;

  std::string comment;
};

using Action =
    std::variant<AdvanceCycle, SetSignal, SetSignalX, WaitForSignals>;

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
      const absl::flat_hash_map<std::string, std::optional<Bits>>&
          owned_signals_to_drive,
      std::optional<std::string> done_signal_name = std::nullopt)
      : shared_data_(XLS_DIE_IF_NULL(shared_data)),
        owned_signals_to_drive_(owned_signals_to_drive.begin(),
                                owned_signals_to_drive.end()),
        done_signal_name_(std::move(done_signal_name)) {}

  std::optional<std::string> done_signal_name() const {
    return done_signal_name_;
  }

  // Sets the given signal to the given value (or X). The value is sticky and
  // remains driven to this value across cycle boundaries until it is Set again
  // (if ever). Signals are always set one time unit after the posedge of the
  // clock.
  ModuleTestbenchThread& Set(std::string_view signal_name, const Bits& value);
  ModuleTestbenchThread& Set(std::string_view signal_name, uint64_t value);
  ModuleTestbenchThread& SetX(std::string_view signal_name);

  // Advances the simulation the given number of cycles.
  ModuleTestbenchThread& AdvanceNCycles(int64_t n_cycles);

  // Advances the simulation a single cycle. Equivalent to AdvanceNCycles(1).
  ModuleTestbenchThread& NextCycle();

  // `WaitForCycleAfter...` methods sample signals at the end of the cycle right
  // before the rising edge of the clock. These methods advance the thread to
  // the cycle *after* the condition is satisfied.

  // Wait for a given single-bit signal to be asserted (unasserted).
  ModuleTestbenchThread& WaitForCycleAfter(std::string_view signal_name);
  ModuleTestbenchThread& WaitForCycleAfterNot(std::string_view signal_name);

  // Waits for all/any of the given single-bit signals to be asserted.
  ModuleTestbenchThread& WaitForCycleAfterAll(
      absl::Span<const std::string> signal_names);
  ModuleTestbenchThread& WaitForCycleAfterAny(
      absl::Span<const std::string> signal_names);

  // Wait for the given signal to have X or non-X values. The signal is
  // considered to have an X value if *any* bit is X. The signals may have
  // arbitrary width.
  ModuleTestbenchThread& WaitForCycleAfterX(std::string_view signal_name);
  ModuleTestbenchThread& WaitForCycleAfterNotX(std::string_view signal_name);

  // Returns a EndOfCycleEvent for the end of the current cycle. Can be used to
  // sample signals. Advances the testbench thread one cycle.
  EndOfCycleEvent& AtEndOfCycle();

  // Returns a EndOfCycleEvent for the end of the cycle in which the given
  // signal is asserted (not asserted, is X, is not X). Advances the testbench
  // thread to one cycle *after* the condition is satisfied.
  EndOfCycleEvent& AtEndOfCycleWhen(std::string_view signal_name);
  EndOfCycleEvent& AtEndOfCycleWhenNot(std::string_view signal_name);
  EndOfCycleEvent& AtEndOfCycleWhenX(std::string_view signal_name);
  EndOfCycleEvent& AtEndOfCycleWhenNotX(std::string_view signal_name);

  // Returns a EndOfCycleEvent for the end of the cycle in which
  // all (any) of the single-bit signals in `signal_names` are asserted.
  // Advances the testbench thread to one cycle *after* the condition is
  // satisfied.
  EndOfCycleEvent& AtEndOfCycleWhenAll(
      absl::Span<const std::string> signal_names);
  EndOfCycleEvent& AtEndOfCycleWhenAny(
      absl::Span<const std::string> signal_names);

  // Expect to find a particular string in the simulation output, typically
  // generated by a trace in the dut.  Trace strings must be found in the order
  // of the ExpectTrace calls.
  // TODO(amfv): 2021-09-02 Figure out how to associate dut traces with
  // significant events during the test (e.g. the activation of a trace because
  // an input changed).
  ModuleTestbenchThread& ExpectTrace(std::string_view trace_message);

  // A pair of instance number and signal name used as a key for associating a
  // output display statement with a particular Capture or ExpectEq.
  using InstanceSignalName = std::pair<int64_t, std::string>;

  // Checks the stdout of a simulation run against expectations.
  absl::Status CheckOutput(
      std::string_view stdout_str,
      const absl::flat_hash_map<InstanceSignalName, BitsOrX>& parsed_values)
      const;

  // Emit the thread contents into the verilog file with the contents specified.
  void EmitInto(StructuredProcedure* procedure, LogicRef* clk,
                const absl::flat_hash_map<std::string, LogicRef*>& signal_refs);

  // Returns a sorted list of the input ports driven by this thread.
  std::vector<std::string> GetThreadOwnedSignals() const;

 private:
  friend class EndOfCycleEvent;

  // Returns the width of the given signal.
  int64_t GetSignalWidth(std::string_view name);

  // CHECKs whether the given name is a signal that the thread is
  // designated to drive.
  void CheckCanDriveSignal(std::string_view name);
  // CHECKs whether the given name is a signal that the thread can read.
  void CheckCanReadSignal(std::string_view name);

  // Returns the EndOfCycleEvent when any (all) of the given signals have the
  // given values.
  EndOfCycleEvent& AtEndOfCycleWhenSignalsEq(
      AnyOrAll any_or_all, std::vector<SignalValue> signal_values,
      std::string_view comment = "");

  const ModuleTestbenchData* shared_data_;
  // The owned_signals_to_drive_ is a name-value map with the name of the
  // input signals mapped to their initial value after reset. A value of
  // std::nullopt signifies an 'X' in Verilog. The names in the map are also
  // the input signals of the design under test (DUT) that the thread is
  // capable of driving. Use absl::btree_map for stable iteration order.
  absl::btree_map<std::string, std::optional<Bits>> owned_signals_to_drive_;
  // The name of the thread's done signal. The signal is used to notify the
  // testbench that the thread is done with its execution. If its value is
  // std::nullopt, then the thread does not emit the done signal.
  std::optional<std::string> done_signal_name_;

  // The list of actions to perform during simulation.
  std::vector<Action> actions_;

  // A map containing the pointers passed in to each Capture call. Use
  // absl::btree_map for stable iteration order.
  absl::btree_map<InstanceSignalName, Bits*> captures_;

  // A map containing the expected values passed in to each ExpectEq call. Use
  // absl::btree_map for stable iteration order.
  absl::btree_map<InstanceSignalName, Expectation> expectations_;

  // A increasing counter which is used to generate unique instance
  // identifiers for DisplayOutput and InstantPort objects.
  int64_t next_instance_ = 0;

  std::vector<std::string> expected_traces_;
};

// Test class which does a cycle-by-cycle simulation of a Verilog module.
class ModuleTestbench {
 public:
  // Constructor for testing a VAST-defined module with the given clock and
  // reset signal.
  ModuleTestbench(Module* module, const VerilogSimulator* simulator,
                  std::optional<std::string_view> clk_name = std::nullopt,
                  std::optional<ResetProto> reset = std::nullopt,
                  absl::Span<const VerilogInclude> includes = {});

  // Constructor for testing a module defined in Verilog text with an
  // interface described with a ModuleSignature.
  ModuleTestbench(std::string_view verilog_text, FileType file_type,
                  const ModuleSignature& signature,
                  const VerilogSimulator* simulator,
                  absl::Span<const VerilogInclude> includes = {});

  // Returns a reference to a newly created thread to execute in the
  // testbench.
  //
  // The owned_signals_to_drive is a name-value map with the name of the
  // input signals mapped to their initial value after reset. A value of
  // std::nullopt signifies an 'X' in Verilog. The names in the map are also
  // the input signals of the design under test (DUT) that the thread is
  // capable of driving. A std::nullopt value for the map signifies all inputs
  // (no constraint on the inputs). The latter is the default value of the
  // map.
  absl::StatusOr<ModuleTestbenchThread*> CreateThread(
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
  // 'threads'. The xls::verilog::ModuleTestbench::CreateThread function
  // returns a reference to a ModuleTestbenchThread, as a result the threads
  // use std::unique_ptr to avoid bad referencing when vector is resized.
  std::vector<std::unique_ptr<ModuleTestbenchThread>> threads_;

  // The set of input signals which have been claimed by a thread. An input
  // can only be claimed by a single thread (the thread which is responsible
  // for driving the signal).
  absl::flat_hash_set<std::string> thread_owned_signals_;
};

}  // namespace verilog
}  // namespace xls

#endif  // XLS_CODEGEN_MODULE_TESTBENCH_H_
