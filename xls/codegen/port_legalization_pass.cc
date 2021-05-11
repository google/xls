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

#include "xls/ir/channel.h"
#include "xls/ir/value_helpers.h"

namespace xls::verilog {

// Renumbers the ports of the proc to be valid (consectively numbered from
// zero). To be used after removing one or more ports. If ports are not numbered
// then this function does nothing.
static absl::Status RenumberPorts(Proc* proc) {
  XLS_ASSIGN_OR_RETURN(std::vector<Proc::Port> ports, proc->GetPorts());
  int64_t current_pos = 0;
  for (const Proc::Port& port : ports) {
    if (!port.channel->GetPosition().has_value()) {
      // Ports are either all numbered or none of them are numbered (checked by
      // the verifier).
      return absl::OkStatus();
    }
    // GetPorts returns the ports in order by position.
    XLS_RET_CHECK_LE(current_pos, port.channel->GetPosition().value());
    port.channel->SetPosition(current_pos++);
  }
  return absl::OkStatus();
}

absl::StatusOr<bool> PortLegalizationPass::RunInternal(
    CodegenPassUnit* unit, const CodegenPassOptions& options,
    PassResults* results) const {
  bool changed = false;
  Proc* proc = unit->top;
  XLS_ASSIGN_OR_RETURN(std::vector<Proc::Port> ports, unit->top->GetPorts());

  for (int64_t i = 0; i < ports.size(); ++i) {
    const Proc::Port& port = ports[i];
    // Remove zero-width ports.
    if (port.channel->type()->GetFlatBitCount() == 0) {
      if (port.direction == Proc::PortDirection::kOutput) {
        // Output ports are sends over a port channel. Delete the send
        // node. The send node produces a token value. Replace any uses of that
        // token value with the token operand of the send.
        XLS_RET_CHECK(port.node->Is<Send>());
        XLS_RETURN_IF_ERROR(
            port.node->ReplaceUsesWith(port.node->As<Send>()->token()));
      } else {
        // Input ports are receives over a port channel. Delete the receive
        // node. The receive node produces a tuple containing (token,
        // data). Replace with a tuple containing the operand token and a dummy
        // literal value for the data.
        XLS_RET_CHECK(port.node->Is<Receive>());
        XLS_ASSIGN_OR_RETURN(
            xls::Literal * dummy_value,
            proc->MakeNode<xls::Literal>(port.node->loc(),
                                         ZeroOfType(port.channel->type())));
        std::vector<Node*> tuple_elements = {port.node->As<Receive>()->token(),
                                             dummy_value};
        XLS_RETURN_IF_ERROR(
            port.node->ReplaceUsesWithNew<Tuple>(tuple_elements).status());
      }
      XLS_RETURN_IF_ERROR(proc->RemoveNode(port.node));
      XLS_RETURN_IF_ERROR(proc->package()->RemoveChannel(port.channel));
      changed = true;
    }
  }
  if (changed) {
    XLS_RETURN_IF_ERROR(RenumberPorts(proc));
  }
  return changed;
}

}  // namespace xls::verilog
