// Copyright 2020 Google LLC
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

#include <list>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "absl/types/variant.h"
#include "xls/codegen/module_signature.h"
#include "xls/codegen/vast.h"
#include "xls/common/integral_types.h"
#include "xls/ir/bits.h"
#include "xls/simulation/verilog_simulator.h"

namespace xls {
namespace verilog {

// Test class which does a cycle-by-cycle simulation of a Verilog module.
// Provides a fluent interface for driving inputs, capturing outputs, and
// setting expectations.
class ModuleTestbench {
 public:
  // Constructor for testing a VAST-defined module with the given clock and
  // reset signal.
  explicit ModuleTestbench(
      Module* module, const VerilogSimulator* simulator,
      absl::optional<absl::string_view> clk_name = absl::nullopt,
      absl::optional<ResetProto> reset = absl::nullopt);

  // Constructor for testing a module defined in Verilog text with an interface
  // described with a ModuleSignature.
  ModuleTestbench(absl::string_view verilog_text,
                  const ModuleSignature& signature,
                  const VerilogSimulator* simulator);

  // Sets the given module input port to the given value in the current
  // cycle. The value is sticky and remains driven to this value across cycle
  // boundaries until it is Set again (if ever).
  ModuleTestbench& Set(absl::string_view input_port, const Bits& value);
  ModuleTestbench& Set(absl::string_view input_port, uint64 value);

  // Sets the given module input to the unknown value in the current cycle. As
  // with Set() this is sticky.
  ModuleTestbench& SetX(absl::string_view input_port);

  // Advances the simulation the given number of cycles.
  ModuleTestbench& AdvanceNCycles(int64 n_cycles);

  // Wait for an given single-bit output port to be asserted (unasserted). If
  // the signal is already asserted (unasserted), this action takes no simulator
  // time.
  ModuleTestbench& WaitFor(absl::string_view output_port);
  ModuleTestbench& WaitForNot(absl::string_view output_port);

  // Wait for the given outputs to have X or non-X values. The output is
  // considered to have an X value if *any* bit is X. The outputs may have
  // arbitrary width.
  ModuleTestbench& WaitForX(absl::string_view output_port);
  ModuleTestbench& WaitForNotX(absl::string_view output_port);

  // Advances the simulation a single cycle. Equivalent to AdvanceNCycles(1).
  ModuleTestbench& NextCycle();

  // Captures the value of the output port at the current cycle. The given
  // pointer value is written with the output port value when Run is called.
  ModuleTestbench& Capture(absl::string_view output_port, Bits* value);

  // Expects the given output port is the given value (or X) in the current
  // cycle. An error is returned during Run if this expectation is not met.
  //
  // "loc" indicates the source position in the test where the expectation was
  // created, and is displayed on expectation failure.
  ModuleTestbench& ExpectEq(
      absl::string_view output_port, const Bits& expected,
      xabsl::SourceLocation loc = xabsl::SourceLocation::current());
  ModuleTestbench& ExpectEq(
      absl::string_view output_port, uint64 expected,
      xabsl::SourceLocation loc = xabsl::SourceLocation::current());
  // Similar to ExpectEq, but expects the given output port to be X. For this
  // purpose an output port is considered to have the value X if *any* bit is X.
  ModuleTestbench& ExpectX(
      absl::string_view output_port,
      xabsl::SourceLocation loc = xabsl::SourceLocation::current());

  // Runs the simulation.
  absl::Status Run();

 private:
  // Checks the stdout of a simulation run against expectations.
  absl::Status CheckOutput(absl::string_view stdout_str) const;

  // Returns the width of the given port.
  int64 GetPortWidth(absl::string_view port);

  // CHECKs whether the given name is an input/output port.
  void CheckIsInput(absl::string_view name);
  void CheckIsOutput(absl::string_view name);

  std::string verilog_text_;
  std::string module_name_;
  const VerilogSimulator* simulator_;
  absl::optional<std::string> clk_name_;

  // Map of each input/output port name to its width.
  absl::flat_hash_map<std::string, int64> input_port_widths_;
  absl::flat_hash_map<std::string, int64> output_port_widths_;

  // The following structs define the actions which are set to occur during
  // simulation. This list of actions is built up by calling the methods on
  // ModuleTestbench (e.g., Set()).

  // Advances the current cycle a certain amount.
  struct AdvanceCycle {
    int64 amount;
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

  // Sentinel type for indicating an "X or "not X" value, in lieu of some real
  // bits value.
  struct IsX {};
  struct IsNotX {};

  // Waits for an output port to equal a certain value.
  struct WaitForOutput {
    std::string port;
    absl::variant<Bits, IsX, IsNotX> value;
  };

  // Inserts a Verilog display statement which prints the value of the given
  // port.
  struct DisplayOutput {
    std::string port;

    // A unique identifier which associates this display statement with a
    // particular Capture or ExpectEq call.
    int64 instance;
  };

  // The list of actions to perform during simulation.
  using Action = absl::variant<AdvanceCycle, SetInput, SetInputX, WaitForOutput,
                               DisplayOutput>;
  std::vector<Action> actions_;

  // A pair of instance number and port name used as a key for associating a
  // output display statement with a particular Capture or ExpectEq.
  using InstancePort = std::pair<int64, std::string>;

  // A map containing the pointers passed in to each Capture call. Use std::map
  // for stable iteration order.
  std::map<InstancePort, Bits*> captures_;

  // A map containing the expected values passed in to each ExpectEq call. Use
  // std::map for stable iteration order.
  struct Expectation {
    absl::variant<Bits, IsX> expected;
    xabsl::SourceLocation loc;
  };
  std::map<InstancePort, Expectation> expectations_;

  // A increasing counter which is used to generate unique instance identifiers
  // for DisplayOutput and InstantPort objects.
  int64 next_instance_ = 0;
};

}  // namespace verilog
}  // namespace xls

#endif  // XLS_CODEGEN_MODULE_TESTBENCH_H_
