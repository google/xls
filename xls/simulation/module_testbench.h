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

enum class TestbenchSignalType {
  // Input port of the DUT.
  kInputPort,

  // Output port of the DUT.
  kOutputPort,

  // Internal signal in the testbench.
  kInternal
};

struct TestbenchSignal {
  std::string name;
  int64_t width;
  TestbenchSignalType type;
};

// Metadata about the testbench and the underlying device-under-test.
class TestbenchMetadata {
 public:
  explicit TestbenchMetadata(const ModuleSignature& signature);

  std::string dut_module_name() const { return dut_module_name_; }
  absl::Span<const TestbenchSignal> signals() const { return signals_; }
  std::optional<std::string> clk_name() const { return clk_name_; }
  std::optional<ResetProto> reset_proto() const { return reset_proto_; }

  const TestbenchSignal& GetSignal(std::string_view name) const {
    return signals_.at(signals_by_name_.at(name));
  }
  int64_t GetSignalWidth(std::string_view name) const {
    return GetSignal(name).width;
  }
  bool HasSignal(std::string_view name) const {
    return signals_by_name_.contains(name);
  }

  bool IsClock(std::string_view name) const {
    return clk_name_.has_value() && name == clk_name_.value();
  }

  // Returns a status error if the given signal is *not* readable by the
  // testbench logic.
  absl::Status CheckIsReadableSignal(std::string_view name) const;

  // Adds information about an internal testbench signal to the metadata.
  absl::Status AddInternalSignal(std::string_view name, int64_t width) {
    return AddSignal(name, width, TestbenchSignalType::kInternal);
  }

 private:
  absl::Status AddSignal(std::string_view name, int64_t width,
                         TestbenchSignalType type);

  std::string dut_module_name_;
  std::vector<TestbenchSignal> signals_;

  // Map from signal name to index in `signals_`.
  absl::flat_hash_map<std::string, int64_t> signals_by_name_;

  std::optional<std::string> clk_name_;
  std::optional<ResetProto> reset_proto_;
};

struct TestbenchExpectation {
  BitsOrX expected;
  xabsl::SourceLocation loc;
};

struct TestbenchCapture {
  Bits* bits;
};

using SignalCaptureAction =
    std::variant<TestbenchExpectation, TestbenchCapture>;

// Represents a single instance of a signal capture. Each corresponds to a
// particular $display statement in the testbench.
struct SignalCapture {
  TestbenchSignal signal;

  SignalCaptureAction action;

  // A unique identifier which associates this capture instance with a
  // particular line of simulation output. This integer is emitted along side
  // the captured value in the $display-ed string during simulation and is used
  // to associate the value back to a particular `Capture` (or `ExpectEq`, etc)
  // instance.
  int64_t instance_id;
};

// Data structure which allocates capture instances so that the instance ids are
// unique across all threads in the testbench.
class SignalCaptureManager {
 public:
  // Return a capture instance associated with a Capture/ExpectEq/ExpectX
  // action.
  SignalCapture Capture(const TestbenchSignal& signal, Bits* bits);
  SignalCapture ExpectEq(const TestbenchSignal& signal, const Bits& bits,
                         xabsl::SourceLocation loc);
  SignalCapture ExpectX(const TestbenchSignal& signal,
                        xabsl::SourceLocation loc);

  absl::Span<const SignalCapture> signal_captures() const {
    return signal_captures_;
  }

 private:
  std::vector<SignalCapture> signal_captures_;
};

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
  explicit EndOfCycleEvent(const TestbenchMetadata* shared_data,
                           SignalCaptureManager* capture_manager)
      : metadata_(XLS_DIE_IF_NULL(shared_data)),
        capture_manager_(capture_manager) {}

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

  absl::Span<const SignalCapture> signal_captures() const {
    return signal_captures_;
  }

 private:
  const TestbenchMetadata* metadata_;
  SignalCaptureManager* capture_manager_;

  // Set of instances of signal captures. Each corresponds to a particular
  // $display statement in the testbench.
  std::vector<SignalCapture> signal_captures_;
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

struct DrivenSignal {
  std::string signal_name;
  BitsOrX initial_value;
};

// Provides a fluent interface for driving inputs, capturing outputs, and
// setting expectations.
class ModuleTestbenchThread {
 public:
  // `driven_signals` is the set of signals which this thread can drive. A
  // signal can only be driven by one thread.
  //
  // The done_signal_name is the name of the thread's done signal. The signal is
  // used to notify the testbench that the thread is done with its execution. If
  // its value is std::nullopt, then the thread does not emit the done signal.
  ModuleTestbenchThread(
      const TestbenchMetadata* metadata, SignalCaptureManager* capture_manager,
      absl::Span<const DrivenSignal> driven_signals,
      std::optional<std::string> done_signal_name = std::nullopt)
      : metadata_(XLS_DIE_IF_NULL(metadata)),
        capture_manager_(capture_manager),
        driven_signals_(driven_signals.begin(), driven_signals.end()),
        done_signal_name_(std::move(done_signal_name)) {
    for (const DrivenSignal& signal : driven_signals) {
      driven_signal_names_.insert(signal.signal_name);
    }
  }

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

  // Emit the thread contents into the verilog file with the contents specified.
  void EmitInto(StructuredProcedure* procedure, LogicRef* clk,
                const absl::flat_hash_map<std::string, LogicRef*>& signal_refs);

  absl::Span<const DrivenSignal> driven_signals() const {
    return driven_signals_;
  }

  absl::Span<const std::string> expected_traces() const {
    return expected_traces_;
  }

 private:
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

  const TestbenchMetadata* metadata_;
  SignalCaptureManager* capture_manager_;

  // The signals which this thread drives. This can include internal signals of
  // the testbench of inputs to the DUT.
  std::vector<DrivenSignal> driven_signals_;

  // Set of driven signal names for easy membership testing.
  absl::flat_hash_set<std::string> driven_signal_names_;

  // The name of the thread's done signal. The signal is used to notify the
  // testbench that the thread is done with its execution. If its value is
  // std::nullopt, then the thread does not emit the done signal.
  std::optional<std::string> done_signal_name_;

  // The list of actions to perform during simulation.
  std::vector<Action> actions_;

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
  absl::Status CaptureOutputsAndCheckExpectations(
      std::string_view stdout_str) const;

  std::vector<std::string> GatherExpectedTraces() const;

  std::string verilog_text_;
  FileType file_type_;
  const VerilogSimulator* simulator_;
  absl::Span<const VerilogInclude> includes_;

  TestbenchMetadata metadata_;
  SignalCaptureManager capture_manager_;

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
