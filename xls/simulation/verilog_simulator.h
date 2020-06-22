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

#ifndef XLS_SIMULATION_VERILOG_SIMULATOR_H_
#define XLS_SIMULATION_VERILOG_SIMULATOR_H_

#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xls/codegen/name_to_bit_count.h"
#include "xls/common/status/statusor.h"
#include "xls/ir/bits.h"
#include "xls/tools/verilog_include.h"

namespace xls {
namespace verilog {

struct Observation {
  int64 time;
  std::string name;
  Bits value;
};

// Interface wrapping a Verilog simulator such Icarus verilog.
class VerilogSimulator {
 public:
  virtual ~VerilogSimulator() = default;

  // Runs the simulator with the given Verilog text as input and returns the
  // stdout/stderr as a string pair.
  virtual xabsl::StatusOr<std::pair<std::string, std::string>> Run(
      absl::string_view text,
      absl::Span<const VerilogInclude> includes) const = 0;
  xabsl::StatusOr<std::pair<std::string, std::string>> Run(
      absl::string_view text) const;

  // Runs the simulator to check the Verilog syntax. Does not run simulation.
  virtual absl::Status RunSyntaxChecking(
      absl::string_view text,
      absl::Span<const VerilogInclude> includes) const = 0;
  absl::Status RunSyntaxChecking(absl::string_view text) const;

  // Simulation runner harness: runs the given Verilog text using the verilog
  // simulator infrastructure and returns observations of data values that arose
  // during simulation.
  //
  // This routine expects the simulation (via the provided Verilog text) to emit
  // data to stdout of the form:
  //
  //    $time: $name = %h; $name = %h; ...
  //
  // Which the routine turns into a sequence of the Observation structure shown
  // above.
  xabsl::StatusOr<std::vector<Observation>> SimulateCombinational(
      absl::string_view text, const NameToBitCount& to_observe) const;
};

// An abstraction which holds multiple VerilogSimulator objects organized by
// name.
class VerilogSimulatorManager {
 public:
  // Returns the delay estimator with the given name, or returns an error if no
  // such estimator exists.
  xabsl::StatusOr<VerilogSimulator*> GetVerilogSimulator(
      absl::string_view name) const;

  // Adds a VerilogSimulator to the manager and associates it with the given
  // name.
  absl::Status RegisterVerilogSimulator(
      absl::string_view name, std::unique_ptr<VerilogSimulator> simulator);

  // Returns a list of the names of available simulators in this manager.
  absl::Span<const std::string> simulator_names() const {
    return simulator_names_;
  }

 private:
  absl::flat_hash_map<std::string, std::unique_ptr<VerilogSimulator>>
      simulators_;
  std::vector<std::string> simulator_names_;
};

// Returns the singleton manager which holds the
VerilogSimulatorManager& GetVerilogSimulatorManagerSingleton();

}  // namespace verilog
}  // namespace xls

#endif  // XLS_SIMULATION_VERILOG_SIMULATOR_H_
