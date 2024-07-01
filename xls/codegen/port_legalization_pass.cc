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

#include <memory>
#include <variant>
#include <vector>

#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "xls/codegen/codegen_pass.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/block.h"
#include "xls/ir/node.h"
#include "xls/ir/nodes.h"
#include "xls/ir/value_utils.h"

namespace xls::verilog {
namespace {
bool IsZeroWidthInstantiationPort(Node* node) {
  if (node->Is<InstantiationInput>()) {
    return node->As<InstantiationInput>()
               ->data()
               ->GetType()
               ->GetFlatBitCount() == 0;
  }
  if (node->Is<InstantiationOutput>()) {
    return node->GetType()->GetFlatBitCount() == 0;
  }
  return false;
}
}  // namespace

absl::StatusOr<bool> PortLegalizationPass::RunInternal(
    CodegenPassUnit* unit, const CodegenPassOptions& options,
    CodegenPassResults* results) const {
  bool changed = false;
  std::vector<Node*> to_remove;
  for (std::unique_ptr<Block>& block : unit->package->blocks()) {
    // Remove instantiation inputs/outputs with zero width.
    for (Node* node : block->nodes()) {
      if (IsZeroWidthInstantiationPort(node)) {
        XLS_RETURN_IF_ERROR(
            node->ReplaceUsesWithNew<xls::Literal>(ZeroOfType(node->GetType()))
                .status());
        to_remove.push_back(node);
        changed = true;
      }
    }
    for (Node* node : to_remove) {
      XLS_RETURN_IF_ERROR(block->RemoveNode(node));
    }
    to_remove.clear();

    std::vector<Block::Port> ports(block->GetPorts().begin(),
                                   block->GetPorts().end());
    // Remove zero-width input ports and output ports.
    for (const Block::Port& port : ports) {
      if (std::holds_alternative<InputPort*>(port)) {
        InputPort* input_port = std::get<InputPort*>(port);
        if (input_port->GetType()->GetFlatBitCount() == 0) {
          VLOG(4) << "Removing zero-width input port " << input_port->name();
          XLS_RETURN_IF_ERROR(input_port
                                  ->ReplaceUsesWithNew<xls::Literal>(
                                      ZeroOfType(input_port->GetType()))
                                  .status());
          XLS_RETURN_IF_ERROR(block->RemoveNode(input_port));
          changed = true;
        }
      } else if (std::holds_alternative<OutputPort*>(port)) {
        OutputPort* output_port = std::get<OutputPort*>(port);
        if (output_port->operand(0)->GetType()->GetFlatBitCount() == 0) {
          VLOG(4) << "Removing zero-width output port " << output_port->name();
          XLS_RETURN_IF_ERROR(block->RemoveNode(output_port));
          changed = true;
        }
      }
    }
  }

  if (changed) {
    unit->GcMetadata();
  }

  return changed;
}

}  // namespace xls::verilog
