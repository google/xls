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

#include "xls/simulation/default_verilog_simulator.h"

#include <memory>
#include <string>
#include <utility>

#include "absl/flags/flag.h"
#include "absl/log/check.h"
#include "absl/status/statusor.h"
#include "xls/simulation/verilog_simulator.h"
#include "xls/simulation/verilog_simulators.h"

ABSL_FLAG(std::string, verilog_simulator, "iverilog",
          "The Verilog simulator to use. If not specified, the default "
          "simulator is used.");

namespace xls {
namespace verilog {

std::unique_ptr<VerilogSimulator> GetDefaultVerilogSimulator() {
  const std::string simulator_name = absl::GetFlag(FLAGS_verilog_simulator);
  auto simulator = GetVerilogSimulator(simulator_name);
  QCHECK_OK(simulator) << "Unknown simulator --verilog_simulator="
                       << simulator_name;
  return std::move(simulator.value());
}

}  // namespace verilog
}  // namespace xls
