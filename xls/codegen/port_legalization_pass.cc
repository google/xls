// Copyright 2021 The XLS Authors
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

#include "xls/codegen/port_legalization_pass.h"

#include "xls/common/logging/logging.h"
#include "xls/ir/block.h"
#include "xls/ir/value_helpers.h"

namespace xls::verilog {

absl::StatusOr<bool> PortLegalizationPass::RunInternal(
    CodegenPassUnit* unit, const CodegenPassOptions& options,
    PassResults* results) const {
  bool changed = false;
  Block* block = unit->block;
  std::vector<Block::Port> ports(block->GetPorts().begin(),
                                 block->GetPorts().end());
  for (const Block::Port& port : ports) {
    // Remove zero-width input ports and output ports.
    if (absl::holds_alternative<InputPort*>(port)) {
      InputPort* input_port = absl::get<InputPort*>(port);
      if (input_port->GetType()->GetFlatBitCount() == 0) {
        XLS_VLOG(4) << "Removing zero-width input port " << input_port->name();
        XLS_RETURN_IF_ERROR(input_port
                                ->ReplaceUsesWithNew<xls::Literal>(
                                    ZeroOfType(input_port->GetType()))
                                .status());
        XLS_RETURN_IF_ERROR(block->RemoveNode(input_port));
        changed = true;
      }
    } else if (absl::holds_alternative<OutputPort*>(port)) {
      OutputPort* output_port = absl::get<OutputPort*>(port);
      if (output_port->operand(0)->GetType()->GetFlatBitCount() == 0) {
        XLS_VLOG(4) << "Removing zero-width output port "
                    << output_port->name();
        XLS_RETURN_IF_ERROR(block->RemoveNode(output_port));
        changed = true;
      }
    }
  }

  return changed;
}

}  // namespace xls::verilog
