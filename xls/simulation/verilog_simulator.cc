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

#include "xls/simulation/verilog_simulator.h"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/base/no_destructor.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/ascii.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"
#include "absl/types/span.h"
#include "xls/codegen/name_to_bit_count.h"
#include "xls/codegen/vast/vast.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/bits.h"
#include "re2/re2.h"

namespace xls {
namespace verilog {
namespace {

absl::StatusOr<std::vector<Observation>> StdoutToObservations(
    std::string_view output, const NameToBitCount& to_observe) {
  std::vector<Observation> result;
  auto error = [](std::string_view line, std::string_view message) {
    return absl::InternalError(
        absl::StrCat("Simulation produced invalid monitoring line: \"", line,
                     "\" :: ", message));
  };
  for (std::string_view line : absl::StrSplit(output, '\n')) {
    line = absl::StripAsciiWhitespace(line);
    if (line.empty()) {
      continue;
    }

    if (!RE2::FullMatch(line, R"(^\s*[0-9]+\s*:.*)")) {
      // Skip lines which do not begin with a numeric time value followed by a
      // colon.
      continue;
    }

    std::vector<std::string_view> pieces = absl::StrSplit(line, ':');
    if (pieces.size() != 2) {
      return error(line, "missing time-delimiting ':'");
    }

    int64_t time;
    if (!absl::SimpleAtoi(pieces[0], &time)) {
      return error(line, "invalid simulation time value");
    }

    // Turn all of the print-outs at this time into "observations".
    std::vector<std::string_view> observed = absl::StrSplit(pieces[1], ';');
    for (std::string_view observation : observed) {
      std::string name;
      uint64_t value;
      if (!RE2::FullMatch(observation, "\\s*(\\w+) = ([0-9A-Fa-f]+)\\s*", &name,
                          RE2::Hex(&value))) {
        return error(line, "monitoring line did not match expected pattern");
      }
      auto it = to_observe.find(name);
      if (it == to_observe.end()) {
        continue;
      }
      int64_t bit_count = it->second;
      result.push_back(Observation{time, name, UBits(value, bit_count)});
    }
  }
  return result;
}

}  // namespace

absl::StatusOr<std::pair<std::string, std::string>> VerilogSimulator::Run(
    std::string_view text, FileType file_type) const {
  return Run(text, file_type, /*macro_definitions=*/{}, /*includes=*/{});
}

absl::StatusOr<std::pair<std::string, std::string>> VerilogSimulator::Run(
    std::string_view text, FileType file_type,
    absl::Span<const MacroDefinition> macro_definitions) const {
  return Run(text, file_type, macro_definitions, /*includes=*/{});
}

absl::Status VerilogSimulator::RunSyntaxChecking(std::string_view text,
                                                 FileType file_type) const {
  return RunSyntaxChecking(text, file_type, /*macro_definitions=*/{},
                           /*includes=*/{});
}

absl::Status VerilogSimulator::RunSyntaxChecking(
    std::string_view text, FileType file_type,
    absl::Span<const MacroDefinition> macro_definitions) const {
  return RunSyntaxChecking(text, file_type, macro_definitions, /*includes=*/{});
}

absl::StatusOr<std::vector<Observation>>
VerilogSimulator::SimulateCombinational(
    std::string_view text, FileType file_type,
    const NameToBitCount& to_observe) const {
  std::pair<std::string, std::string> stdout_stderr;
  XLS_ASSIGN_OR_RETURN(stdout_stderr, Run(text, file_type));
  return StdoutToObservations(stdout_stderr.first, to_observe);
}

VerilogSimulatorManager& GetVerilogSimulatorManagerSingleton() {
  static absl::NoDestructor<VerilogSimulatorManager> manager;
  return *manager;
}

absl::StatusOr<std::unique_ptr<VerilogSimulator>>
VerilogSimulatorManager::GetVerilogSimulator(std::string_view name) const {
  if (!simulator_generators_.contains(name)) {
    if (simulator_names_.empty()) {
      return absl::NotFoundError(
          absl::StrFormat("No simulator found named \"%s\". No "
                          "simulators are registered. Was InitXls called?",
                          name));
    }
    return absl::NotFoundError(absl::StrFormat(
        "No simulator found named \"%s\". Available simulators: %s", name,
        absl::StrJoin(simulator_names_, ", ")));
  }
  return simulator_generators_.at(name)();
}

absl::Status VerilogSimulatorManager::RegisterVerilogSimulator(
    std::string_view name,
    VerilogSimulatorManager::SimulatorGenerator simulator_generator) {
  if (simulator_generators_.contains(name)) {
    return absl::InternalError(
        absl::StrFormat("Simulator named %s already exists", name));
  }
  simulator_generators_[name] = std::move(simulator_generator);
  simulator_names_.push_back(std::string(name));
  std::sort(simulator_names_.begin(), simulator_names_.end());

  return absl::OkStatus();
}

}  // namespace verilog
}  // namespace xls
