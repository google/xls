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

#include "xls/simulation/verilog_simulators.h"

#include "absl/flags/flag.h"

namespace xls {
namespace verilog {

xabsl::StatusOr<VerilogSimulator*> GetVerilogSimulator(absl::string_view name) {
  return GetVerilogSimulatorManagerSingleton().GetVerilogSimulator(name);
}

// TODO(meheff): Remove this function.
const VerilogSimulator& GetDefaultVerilogSimulator() {
  const char kDefaultSimulator[] = "iverilog";
  return *GetVerilogSimulator(kDefaultSimulator).value();
}

}  // namespace verilog
}  // namespace xls
