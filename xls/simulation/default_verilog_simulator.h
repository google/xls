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

#ifndef XLS_SIMULATION_DEFAULT_VERILOG_SIMULATOR_H_
#define XLS_SIMULATION_DEFAULT_VERILOG_SIMULATOR_H_


#include "xls/simulation/verilog_simulator.h"

namespace xls {
namespace verilog {

// Returns a reference to a default verilog simulator named by
// the --verilog_simulator flag.
const VerilogSimulator& GetDefaultVerilogSimulator();

}  // namespace verilog
}  // namespace xls

#endif  // XLS_SIMULATION_DEFAULT_VERILOG_SIMULATOR_H_
