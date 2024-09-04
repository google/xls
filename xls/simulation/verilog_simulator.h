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

#ifndef XLS_SIMULATION_VERILOG_SIMULATOR_H_
#define XLS_SIMULATION_VERILOG_SIMULATOR_H_

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "xls/codegen/name_to_bit_count.h"
#include "xls/codegen/vast/vast.h"
#include "xls/ir/bits.h"
#include "xls/simulation/verilog_include.h"

namespace xls {
namespace verilog {

struct Observation {
  int64_t time;
  std::string name;
  Bits value;
};

// Interface wrapping a Verilog simulator such Icarus verilog.
class VerilogSimulator {
 public:
  virtual ~VerilogSimulator() = default;

  // Abstraction representing a macro definition (tick define). Passing in a
  // MacroDefinition is equivalent to the following if value is specified:
  //   `define NAME VALUE
  // or the following if value is std::nullopt:
  //   `define NAME
  struct MacroDefinition {
    std::string name;
    std::optional<std::string> value;
  };
  static constexpr char kSimulationMacroName[] = "SIMULATION";

  // Runs the simulator with the given Verilog text as input and returns the
  // stdout/stderr as a string pair.
  absl::StatusOr<std::pair<std::string, std::string>> Run(
      std::string_view text, FileType file_type) const;
  absl::StatusOr<std::pair<std::string, std::string>> Run(
      std::string_view text, FileType file_type,
      absl::Span<const MacroDefinition> macro_definitions) const;
  virtual absl::StatusOr<std::pair<std::string, std::string>> Run(
      std::string_view text, FileType file_type,
      absl::Span<const MacroDefinition> macro_definitions,
      absl::Span<const VerilogInclude> includes) const = 0;

  // Runs the simulator to check the Verilog syntax. Does not run simulation.
  absl::Status RunSyntaxChecking(std::string_view text,
                                 FileType file_type) const;
  absl::Status RunSyntaxChecking(
      std::string_view text, FileType file_type,
      absl::Span<const MacroDefinition> macro_definitions) const;
  virtual absl::Status RunSyntaxChecking(
      std::string_view text, FileType file_type,
      absl::Span<const MacroDefinition> macro_definitions,
      absl::Span<const VerilogInclude> includes) const = 0;

  // Features supported by simulator.
  virtual bool DoesSupportSystemVerilog() const = 0;
  virtual bool DoesSupportAssertions() const = 0;

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
  absl::StatusOr<std::vector<Observation>> SimulateCombinational(
      std::string_view text, FileType file_type,
      const NameToBitCount& to_observe) const;

 protected:
  std::string GetTopFileName(FileType file_type) const {
    return absl::StrCat("top.",
                        file_type == FileType::kSystemVerilog ? "sv" : "v");
  }
};

// An abstraction which holds multiple VerilogSimulator objects organized by
// name.
class VerilogSimulatorManager {
 public:
  // Returns the delay estimator with the given name, or returns an error if no
  // such estimator exists.
  absl::StatusOr<VerilogSimulator*> GetVerilogSimulator(
      std::string_view name) const;

  // Adds a VerilogSimulator to the manager and associates it with the given
  // name.
  absl::Status RegisterVerilogSimulator(
      std::string_view name, std::unique_ptr<VerilogSimulator> simulator);

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
