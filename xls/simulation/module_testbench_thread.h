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

#ifndef XLS_SIMULATION_MODULE_TESTBENCH_THREAD_H_
#define XLS_SIMULATION_MODULE_TESTBENCH_THREAD_H_

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/types/span.h"
#include "xls/codegen/vast/vast.h"
#include "xls/ir/bits.h"
#include "xls/simulation/testbench_metadata.h"
#include "xls/simulation/testbench_signal_capture.h"
#include "xls/simulation/testbench_stream.h"

namespace xls {
namespace verilog {

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

// Drives the module input from a value popped from a stream (read
// desctructively).
struct SetSignalFromStream {
  std::string signal_name;
  const TestbenchStream* stream;
};

// Waits for signals to equal a certain value.
enum class AnyOrAll : int8_t { kAny, kAll };
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

// Emits a $display statement with the specified message.
struct DisplayAction {
  std::string message;
};

// Emits a $finish statement to terminate the simulation.
struct FinishAction {};

using Action =
    std::variant<AdvanceCycle, SetSignal, SetSignalX, SetSignalFromStream,
                 WaitForSignals, DisplayAction, FinishAction>;

class ModuleTestbenchThread;

// Abstraction representing a sequence of statements within the testbench. These
// statements can include driving signals and capturing signals. Sequential
// blocks can be nested. For example, one sequential block may contain a loop
// (`Repeat`) as a statement which itself contains a sequential block.
class SequentialBlock {
 public:
  explicit SequentialBlock(ModuleTestbenchThread* testbench_thread)
      : testbench_thread_(testbench_thread) {}

  // Sets the given signal to the given value (or X). The value is sticky and
  // remains driven to this value across cycle boundaries until it is Set again
  // (if ever). Signals are always set one time unit after the posedge of the
  // clock.
  SequentialBlock& Set(std::string_view signal_name, const Bits& value);
  SequentialBlock& Set(std::string_view signal_name, uint64_t value);
  SequentialBlock& SetX(std::string_view signal_name);

  // Reads a value from the given stream and sets `signal_name` to it.
  SequentialBlock& ReadFromStreamAndSet(std::string_view signal_name,
                                        const TestbenchStream* stream);

  // Advances the simulation the given number of cycles.
  SequentialBlock& AdvanceNCycles(int64_t n_cycles);

  // Advances the simulation a single cycle. Equivalent to AdvanceNCycles(1).
  SequentialBlock& NextCycle();

  // `WaitForCycleAfter...` methods sample signals at the end of the cycle right
  // before the rising edge of the clock. These methods advance the thread to
  // the cycle *after* the condition is satisfied.

  // Wait for a given single-bit signal to be asserted (unasserted).
  SequentialBlock& WaitForCycleAfter(std::string_view signal_name);
  SequentialBlock& WaitForCycleAfterNot(std::string_view signal_name);

  // Waits for all/any of the given single-bit signals to be asserted.
  SequentialBlock& WaitForCycleAfterAll(
      absl::Span<const std::string> signal_names);
  SequentialBlock& WaitForCycleAfterAny(
      absl::Span<const std::string> signal_names);

  // Wait for the given signal to have X or non-X values. The signal is
  // considered to have an X value if *any* bit is X. The signals may have
  // arbitrary width.
  SequentialBlock& WaitForCycleAfterX(std::string_view signal_name);
  SequentialBlock& WaitForCycleAfterNotX(std::string_view signal_name);

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

  // Add a loop (`while (1)` or `repeat` Verilog statement). Returns the
  // SequentialBlock of the loop body.
  SequentialBlock& RepeatForever();
  SequentialBlock& Repeat(int64_t count);

  // Add a $display statement with the given message.
  void Display(std::string_view message);

  // Add a $finish statement to terminate the simulation.
  void Finish();

  void Emit(StatementBlock* statement_block, LogicRef* clk,
            const absl::flat_hash_map<std::string, LogicRef*>& signal_refs,
            const absl::flat_hash_map<std::string, VastStreamEmitter>&
                stream_emitters);

 private:
  // Returns the EndOfCycleEvent when any (all) of the given signals have the
  // given values.
  EndOfCycleEvent& AtEndOfCycleWhenSignalsEq(
      AnyOrAll any_or_all, std::vector<SignalValue> signal_values,
      std::string_view comment = "");

  const TestbenchMetadata& metadata() const;
  SignalCaptureManager& signal_capture_manager() const;

  // The list of actions to perform during simulation.
  struct RepeatStatement {
    std::optional<int64_t> count;
    std::unique_ptr<SequentialBlock> sequential_block;
  };

  ModuleTestbenchThread* testbench_thread_;

  using Statement = std::variant<RepeatStatement, Action>;
  std::vector<Statement> statements_;
};

// Abstraction representing a device-under-test (DUT) input port and a initial
// value to drive on the port in the testbench.
struct DutInput {
  std::string port_name;
  BitsOrX initial_value;
};

// Provides a fluent interface for driving inputs, capturing outputs, and
// setting expectations.
class ModuleTestbenchThread {
 public:
  // `dut_inputs` is the set of DUT input ports that this thread can drive (and
  // initial values to drive). A DUT input port signal can only be driven by one
  // thread.
  //
  // If `generate_done_signal` is true then a testbench-internal signal is
  // declared which will be asserted when the thread finishes executing.
  //
  // If `wait_for_reset` is true then the actions of the thread will not start
  // until reset is asserted, then deasserted (handled in ModuleTestbench).
  ModuleTestbenchThread(std::string_view name,
                        const TestbenchMetadata* metadata,
                        SignalCaptureManager* capture_manager,
                        absl::Span<const DutInput> dut_inputs,
                        bool generate_done_signal = true,
                        bool wait_for_reset = true);

  // Returns the name of the testbench-internal signal which is asserted when
  // the thread is done executing. Returns std::nullopt if
  // `generate_done_signal` was set to false when the thread was constructed.
  std::optional<std::string> done_signal_name() const {
    if (done_signal_.has_value()) {
      return done_signal_->name;
    }
    return std::nullopt;
  }

  // Returns the top-level block of the thread.
  SequentialBlock& MainBlock() { return *main_block_; }

  // Expect to find a particular string in the simulation output, typically
  // generated by a trace in the dut.  Trace strings must be found in the order
  // of the ExpectTrace calls.
  // TODO(amfv): 2021-09-02 Figure out how to associate dut traces with
  // significant events during the test (e.g. the activation of a trace because
  // an input changed).
  ModuleTestbenchThread& ExpectTrace(std::string_view trace_message);

  // Declare a testbench-internal signal. This signal can be driven by the
  // thread using SequentialBlock::Set* methods.
  void DeclareInternalSignal(std::string_view name, int64_t width,
                             const BitsOrX& initial_value);

  // Emit the thread contents into the verilog file with the contents specified.
  absl::Status EmitInto(
      Module* m, LogicRef* clk,
      absl::flat_hash_map<std::string, LogicRef*>* signal_refs,
      const absl::flat_hash_map<std::string, VastStreamEmitter>&
          stream_emitters);

  absl::Span<const DutInput> dut_inputs() const { return dut_inputs_; }
  const TestbenchMetadata& metadata() const { return *metadata_; }
  SignalCaptureManager& signal_capture_manager() const {
    return *capture_manager_;
  }

  absl::Span<const std::string> expected_traces() const {
    return expected_traces_;
  }

  // CHECKs whether the given name is a signal that the thread is designated to
  // drive.
  void CheckCanDriveSignal(std::string_view signal_name);

  std::string_view name() const { return name_; }

 private:
  struct TestbenchSignal {
    std::string name;
    int64_t width;
    BitsOrX initial_value;
  };

  std::string name_;
  const TestbenchMetadata* metadata_;
  SignalCaptureManager* capture_manager_;

  // The DUT input ports this thread drives. This can include internal signals
  // of the testbench of inputs to the DUT.
  std::vector<DutInput> dut_inputs_;

  // The sequential block of the top scope in the thread.
  std::unique_ptr<SequentialBlock> main_block_;

  // Set of driven signal names for easy membership testing.
  std::vector<std::string> drivable_signals_;

  // The name of the thread's done signal. The signal is used to notify the
  // testbench that the thread is done with its execution. If its value is
  // std::nullopt, then the thread does not emit the done signal.
  std::optional<TestbenchSignal> done_signal_;

  bool wait_for_reset_;

  std::vector<std::string> expected_traces_;
  std::vector<TestbenchSignal> internal_signals_;
};

}  // namespace verilog
}  // namespace xls

#endif  // XLS_SIMULATION_MODULE_TESTBENCH_THREAD_H_
