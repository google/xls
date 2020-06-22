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

#ifndef XLS_SIMULATION_VERILOG_SIMULATORS_H_
#define XLS_SIMULATION_VERILOG_SIMULATORS_H_

#include <memory>
#include <string>
#include <vector>

#include "absl/strings/string_view.h"
#include "xls/common/status/statusor.h"
#include "xls/simulation/verilog_simulator.h"

namespace xls {
namespace verilog {

// Returns the registered Verilog simulator with the given name.
xabsl::StatusOr<VerilogSimulator*> GetVerilogSimulator(absl::string_view name);

// Returns a reference to a default verilog simulator.
const VerilogSimulator& GetDefaultVerilogSimulator();

}  // namespace verilog
}  // namespace xls

#endif  // XLS_SIMULATION_VERILOG_SIMULATORS_H_
