// Copyright 2024 The XLS Authors
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

#include "xls/codegen/block_inlining_pass.h"

#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "xls/codegen/codegen_pass.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/block.h"
#include "xls/ir/block_elaboration.h"
#include "xls/ir/elaborated_block_dfs_visitor.h"
#include "xls/ir/instantiation.h"
#include "xls/ir/node.h"
#include "xls/ir/nodes.h"
#include "xls/ir/register.h"
#include "xls/passes/pass_base.h"

namespace xls::verilog {

namespace {

class InlineVisitor : public ElaboratedBlockDfsVisitorWithDefault {
 public:
  explicit InlineVisitor(Block* out) : new_block_(out) {}

  const absl::flat_hash_map<std::pair<Register*, BlockInstance*>, Register*>&
  reg_map() const {
    return reg_map_;
  }
  absl::Status DefaultHandler(const ElaboratedNode& n) final {
    std::vector<Node*> new_ops;
    new_ops.reserve(n.node->operand_count());
    for (Node* op : n.node->operands()) {
      new_ops.push_back(
          old_to_new_[ElaboratedNode{.node = op, .instance = n.instance}]);
    }
    XLS_ASSIGN_OR_RETURN(Node * new_node,
                         n.node->CloneInNewFunction(new_ops, new_block_));
    old_to_new_[n] = new_node;
    return absl::OkStatus();
  }

  // NB Call order is InstantiationInput -> Input
  absl::Status HandleInstantiationInput(InstantiationInput* ii,
                                        BlockInstance* instance) final {
    if (ii->instantiation()->kind() != InstantiationKind::kBlock) {
      // Copy a new instantiation
      XLS_ASSIGN_OR_RETURN(
          Instantiation * new_inst,
          GetCopiedInstantiation(ii->instantiation(), instance));
      XLS_ASSIGN_OR_RETURN(
          (old_to_new_[{.node = ii, .instance = instance}]),
          new_block_->MakeNodeWithName<InstantiationInput>(
              ii->loc(),
              old_to_new_[{.node = ii->data(), .instance = instance}], new_inst,
              ii->port_name(), ii->GetName()));
      return absl::OkStatus();
    }
    XLS_RET_CHECK(
        old_to_new_.contains({.node = ii->data(), .instance = instance}))
        << "Unable to find value set for " << ii->data();
    // Create the 'InputPort' node associated with this and map it to the input.
    // Done here to avoid having to keep track of the value across the function
    // boundary.
    XLS_ASSIGN_OR_RETURN(BlockInstantiation * bi,
                         ii->instantiation()->AsBlockInstantiation());
    XLS_ASSIGN_OR_RETURN(
        InputPort * mapped_port,
        bi->instantiated_block()->GetInputPort(ii->port_name()),
        _ << "Unable to find referenced port " << ii->port_name());
    XLS_RET_CHECK(
        instance->instantiation_to_instance().contains(ii->instantiation()));
    BlockInstance* ii_instance =
        instance->instantiation_to_instance().at(ii->instantiation());
    old_to_new_[{.node = mapped_port, .instance = ii_instance}] =
        old_to_new_[{.node = ii->data(), .instance = instance}];
    return absl::OkStatus();
  }

  absl::Status HandleInputPort(InputPort* port,
                               BlockInstance* instance) final {
    if (!instance->parent_instance()) {
      // Top block, no linking of ports required.
      XLS_ASSIGN_OR_RETURN(
          InputPort * new_input,
          new_block_->AddInputPort(port->name(), port->GetType(), port->loc()));
      old_to_new_[{.node = port, .instance = instance}] = new_input;
      std::optional<InputPort*> old_reset_port =
          port->function_base()->AsBlockOrDie()->GetResetPort();
      if (old_reset_port.has_value() && old_reset_port.value() == port) {
        XLS_RETURN_IF_ERROR(new_block_->SetResetPort(
            new_input,
            *port->function_base()->AsBlockOrDie()->GetResetBehavior()));
      }
      return absl::OkStatus();
    }
    // The InstantiationInput node has already filled in the ops map so no need
    // to do anything.
    XLS_RET_CHECK(old_to_new_.contains({.node = port, .instance = instance}));
    return absl::OkStatus();
  }

  // NB Call order is Output -> InstantiationOutput
  absl::Status HandleOutputPort(OutputPort* port,
                                BlockInstance* instance) final {
    if (!instance->parent_instance()) {
      // Top block, no linking of ports required.
      XLS_ASSIGN_OR_RETURN(
          Node * new_output,
          new_block_->AddOutputPort(
              port->name(),
              old_to_new_[{.node = port->operand(OutputPort::kOperandOperand),
                           .instance = instance}],
              port->loc()));
      old_to_new_[{.node = port, .instance = instance}] = new_output;
      return absl::OkStatus();
    }
    // Just save the output_port -> the mapped value.
    old_to_new_[{.node = port, .instance = instance}] =
        old_to_new_[{.node = port->operand(OutputPort::kOperandOperand),
                     .instance = instance}];
    return absl::OkStatus();
  }

  absl::Status HandleInstantiationOutput(InstantiationOutput* ii,
                                         BlockInstance* instance) final {
    if (ii->instantiation()->kind() != InstantiationKind::kBlock) {
      // Copy a new instantiation
      XLS_ASSIGN_OR_RETURN(
          Instantiation * new_inst,
          GetCopiedInstantiation(ii->instantiation(), instance));
      XLS_ASSIGN_OR_RETURN(
          (old_to_new_[{.node = ii, .instance = instance}]),
          new_block_->MakeNodeWithName<InstantiationOutput>(
              ii->loc(), new_inst, ii->port_name(), ii->GetName()));
      return absl::OkStatus();
    }
    // Take the value the output-port is mapped to.
    XLS_ASSIGN_OR_RETURN(BlockInstantiation * bi,
                         ii->instantiation()->AsBlockInstantiation());
    XLS_ASSIGN_OR_RETURN(
        OutputPort * mapped_port,
        bi->instantiated_block()->GetOutputPort(ii->port_name()),
        _ << "Unable to find referenced port " << ii->port_name());
    XLS_RET_CHECK(
        instance->instantiation_to_instance().contains(ii->instantiation()));
    BlockInstance* ii_instance =
        instance->instantiation_to_instance().at(ii->instantiation());
    XLS_RET_CHECK(
        old_to_new_.contains({.node = mapped_port, .instance = ii_instance}))
        << "Unable to find value set for " << mapped_port << " in "
        << ii_instance;
    old_to_new_[{.node = ii, .instance = instance}] =
        old_to_new_[{.node = mapped_port, .instance = ii_instance}];
    return absl::OkStatus();
  }

  absl::Status HandleRegisterRead(RegisterRead* rr,
                                  BlockInstance* instance) final {
    XLS_ASSIGN_OR_RETURN(Register * new_reg,
                         GetCopiedRegister(rr->GetRegister(), instance));
    XLS_ASSIGN_OR_RETURN(Node * new_node,
                         new_block_->MakeNodeWithName<RegisterRead>(
                             rr->loc(), new_reg, rr->GetName()));
    old_to_new_[{.node = rr, .instance = instance}] = new_node;
    return absl::OkStatus();
  }

  absl::Status HandleRegisterWrite(RegisterWrite* rw,
                                   BlockInstance* instance) final {
    XLS_ASSIGN_OR_RETURN(Register * new_reg,
                         GetCopiedRegister(rw->GetRegister(), instance));
    Node* new_data = old_to_new_[{.node = rw->data(), .instance = instance}];
    std::optional<Node*> new_le =
        rw->load_enable()
            ? std::make_optional(old_to_new_[{.node = *rw->load_enable(),
                                              .instance = instance}])
            : std::nullopt;
    std::optional<Node*> new_reset =
        rw->reset()
            ? std::make_optional(
                  old_to_new_[{.node = *rw->reset(), .instance = instance}])
            : std::nullopt;
    XLS_ASSIGN_OR_RETURN(
        Node * new_node,
        new_block_->MakeNodeWithName<RegisterWrite>(
            rw->loc(), new_data, new_le, new_reset, new_reg, rw->GetName()));
    old_to_new_[{.node = rw, .instance = instance}] = new_node;
    return absl::OkStatus();
  }

 private:
  // verilog::Instantiation from vast is also visible.
  using Instantiation = xls::Instantiation;
  absl::StatusOr<Instantiation*> GetCopiedInstantiation(
      Instantiation* old_inst, BlockInstance* instance) {
    XLS_RET_CHECK_NE(old_inst->kind(), InstantiationKind::kBlock);
    if (inst_map_.contains({old_inst, instance})) {
      return inst_map_[{old_inst, instance}];
    }
    // Create a new Instance
    Instantiation* res;
    std::string new_name =
        absl::StrFormat("%s%s", instance->RegisterPrefix(), old_inst->name());
    if (old_inst->kind() == InstantiationKind::kExtern) {
      XLS_ASSIGN_OR_RETURN(ExternInstantiation * ext,
                           old_inst->AsExternInstantiation());
      XLS_ASSIGN_OR_RETURN(res,
                           new_block_->AddInstantiation(
                               new_name, std::make_unique<ExternInstantiation>(
                                             new_name, ext->function())));
    } else {
      XLS_ASSIGN_OR_RETURN(FifoInstantiation * fifo,
                           old_inst->AsFifoInstantiation());
      XLS_ASSIGN_OR_RETURN(res, new_block_->AddFifoInstantiation(
                                    new_name, fifo->fifo_config(),
                                    fifo->data_type(), fifo->channel_name()));
    }
    inst_map_[{old_inst, instance}] = res;
    return res;
  }
  absl::StatusOr<Register*> GetCopiedRegister(Register* old_reg,
                                              BlockInstance* instance) {
    if (reg_map_.contains({old_reg, instance})) {
      return reg_map_[{old_reg, instance}];
    }
    // create a register
    XLS_ASSIGN_OR_RETURN(
        Register * new_reg,
        new_block_->AddRegister(InstanceRegisterName(old_reg, instance),
                                old_reg->type(), old_reg->reset_value()));
    reg_map_[{old_reg, instance}] = new_reg;
    return new_reg;
  }

  std::string InstanceRegisterName(Register* reg, BlockInstance* instance) {
    return absl::StrFormat("%s%s", instance->RegisterPrefix(), reg->name());
  }

  Block* new_block_;
  absl::flat_hash_map<ElaboratedNode, Node*> old_to_new_;
  absl::flat_hash_map<std::pair<Register*, BlockInstance*>, Register*> reg_map_;
  absl::flat_hash_map<std::pair<Instantiation*, BlockInstance*>, Instantiation*>
      inst_map_;
};

absl::StatusOr<Block*> InlineElaboration(
    const BlockElaboration& elab,
    absl::flat_hash_map<std::string, std::string>& reg_renames) {
  // Create a new block we will stitch everything into. This block takes over
  // the name and top-ness of the old top block.
  XLS_RET_CHECK(elab.top()->block()) << "Top is a fifo.";
  Block* old_top = *elab.top()->block();
  std::string top_name = old_top->name();
  old_top->SetName(absl::StrFormat("old_top_%s", old_top->name()));
  Block* stitched = elab.package()->AddBlock(
      std::make_unique<Block>(top_name, elab.package()));
  absl::Span<Block* const> all_blocks = elab.blocks();

  auto clk_it = absl::c_find_if(
      all_blocks, [](Block* b) { return b->GetClockPort().has_value(); });
  if (clk_it != all_blocks.cend()) {
    XLS_RETURN_IF_ERROR(
        stitched->AddClockPort((*clk_it)->GetClockPort()->name));
  }
  if (elab.package()->GetTop() == old_top) {
    XLS_RETURN_IF_ERROR(elab.package()->SetTop(stitched));
  }

  InlineVisitor vis(stitched);
  XLS_RETURN_IF_ERROR(elab.Accept(vis));

  // Record new register names.
  for (const auto& [orig_reg_and_instance, new_reg] : vis.reg_map()) {
    const auto& [orig_reg, instance] = orig_reg_and_instance;
    if (instance->parent_instance()) {
      reg_renames[absl::StrFormat("%s%s", instance->RegisterPrefix(),
                                  orig_reg->name())] = new_reg->name();
    }
  }

  return stitched;
}
}  // namespace

absl::StatusOr<bool> BlockInliningPass::RunInternal(
    Package* package, const CodegenPassOptions& options, PassResults* results,
    CodegenContext& context) const {
  // No need to inline blocks when we don't have 2+ blocks.
  if (package->blocks().size() < 2) {
    return false;
  }
  XLS_ASSIGN_OR_RETURN(BlockElaboration elab,
                       BlockElaboration::Elaborate(context.top_block()));
  XLS_ASSIGN_OR_RETURN(Block * new_top_block,
                       InlineElaboration(elab, context.register_renames()));
  context.SetTopBlock(new_top_block);

  return true;
}

}  // namespace xls::verilog
