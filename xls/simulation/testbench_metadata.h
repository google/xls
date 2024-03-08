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

#ifndef XLS_SIMULATION_TESTBENCH_METADATA_H_
#define XLS_SIMULATION_TESTBENCH_METADATA_H_

#include <algorithm>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/types/span.h"
#include "xls/codegen/module_signature.h"

namespace xls {
namespace verilog {

// Name of the testbench internal signal which is asserted in the last cycle
// that the DUT is in reset. This can be used to trigger the driving of signals
// for the first cycle out of reset.
constexpr std::string_view kLastResetCycleSignal = "__last_cycle_of_reset";

// Clock period in Verilog time units. Must be even number greater than 3.
constexpr int64_t kClockPeriod = 10;
static_assert(kClockPeriod > 3);
static_assert(kClockPeriod % 2 == 0);

// Metadata about the testbench and the underlying device-under-test.
class TestbenchMetadata {
 public:
  explicit TestbenchMetadata(const ModuleSignature& signature);

  std::string dut_module_name() const { return dut_module_name_; }
  absl::Span<const std::string> dut_input_ports() const {
    return dut_input_ports_;
  }
  absl::Span<const std::string> dut_output_ports() const {
    return dut_output_ports_;
  }

  bool HasInputPortNamed(std::string_view port_name) const {
    return std::find(dut_input_ports_.begin(), dut_input_ports_.end(),
                     port_name) != dut_input_ports_.end();
  }
  bool HasOutputPortNamed(std::string_view port_name) const {
    return std::find(dut_output_ports_.begin(), dut_output_ports_.end(),
                     port_name) != dut_output_ports_.end();
  }
  bool HasPortNamed(std::string_view port_name) const {
    return port_widths_.contains(port_name);
  }

  int64_t GetPortWidth(std::string_view port_name) const {
    return port_widths_.at(port_name);
  }

  //  absl::Span<const TestbenchSignal> signals() const { return signals_; }
  std::optional<std::string> clk_name() const { return clk_name_; }
  std::optional<ResetProto> reset_proto() const { return reset_proto_; }

  bool IsClock(std::string_view name) const {
    return clk_name_.has_value() && name == clk_name_.value();
  }

 private:
  std::string dut_module_name_;
  std::optional<std::string> clk_name_;
  std::optional<ResetProto> reset_proto_;

  std::vector<std::string> dut_input_ports_;
  std::vector<std::string> dut_output_ports_;

  absl::flat_hash_map<std::string, int64_t> port_widths_;
};

}  // namespace verilog
}  // namespace xls

#endif  // XLS_SIMULATION_TESTBENCH_METADATA_H_
