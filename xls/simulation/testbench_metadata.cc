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

#include "xls/simulation/testbench_metadata.h"

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "xls/codegen/module_signature.h"
#include "xls/codegen/module_signature.pb.h"

namespace xls {
namespace verilog {

TestbenchMetadata::TestbenchMetadata(const ModuleSignature& signature) {
  dut_module_name_ = signature.module_name();

  auto add_input_port = [&](std::string_view name, int64_t width) {
    port_widths_[name] = width;
    dut_input_ports_.push_back(std::string{name});
  };
  auto add_output_port = [&](std::string_view name, int64_t width) {
    port_widths_[name] = width;
    dut_output_ports_.push_back(std::string{name});
  };

  if (signature.proto().has_clock_name()) {
    clk_name_ = signature.proto().clock_name();
    add_input_port(signature.proto().clock_name(), 1);
  }
  if (signature.proto().has_reset()) {
    reset_proto_ = signature.proto().reset();
    add_input_port(signature.proto().reset().name(), 1);
  }

  for (const PortProto& port : signature.data_inputs()) {
    add_input_port(port.name(), port.width());
  }
  for (const PortProto& port : signature.data_outputs()) {
    add_output_port(port.name(), port.width());
  }

  if (signature.proto().has_pipeline() &&
      signature.proto().pipeline().has_pipeline_control()) {
    // Module has pipeline register control.
    if (signature.proto().pipeline().pipeline_control().has_valid()) {
      // Add the valid input and optional valid output signals.
      const ValidProto& valid =
          signature.proto().pipeline().pipeline_control().valid();
      add_input_port(valid.input_name(), 1);
      if (!valid.output_name().empty()) {
        add_output_port(valid.output_name(), 1);
      }
    } else {
      CHECK(!signature.proto().pipeline().pipeline_control().has_manual())
          << "Manual register control not supported";
    }
  }
}

}  // namespace verilog
}  // namespace xls
