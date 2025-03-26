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

#include "xls/codegen/conversion_utils.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "xls/codegen/codegen_checker.h"
#include "xls/codegen/codegen_options.h"
#include "xls/codegen/codegen_pass.h"
#include "xls/codegen/codegen_wrapper_pass.h"
#include "xls/codegen/register_legalization_pass.h"
#include "xls/codegen/vast/vast.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/block.h"
#include "xls/ir/channel.h"
#include "xls/ir/instantiation.h"
#include "xls/ir/node.h"
#include "xls/ir/node_util.h"
#include "xls/ir/nodes.h"
#include "xls/ir/op.h"
#include "xls/ir/package.h"
#include "xls/ir/register.h"
#include "xls/ir/source_location.h"
#include "xls/ir/xls_ir_interface.pb.h"
#include "xls/passes/dataflow_simplification_pass.h"
#include "xls/passes/dce_pass.h"
#include "xls/passes/optimization_pass.h"
#include "xls/passes/pass_base.h"
#include "re2/re2.h"

namespace xls::verilog {

std::optional<PackageInterfaceProto::Function> FindFunctionInterface(
    const std::optional<PackageInterfaceProto>& src,
    std::string_view func_name) {
  if (!src) {
    return std::nullopt;
  }
  auto it = absl::c_find_if(src->functions(),
                            [&](const PackageInterfaceProto::Function& f) {
                              return f.base().name() == func_name;
                            });
  if (it != src->functions().end()) {
    return *it;
  }
  return std::nullopt;
}

std::optional<PackageInterfaceProto::Channel> FindChannelInterface(
    const std::optional<PackageInterfaceProto>& src,
    std::string_view chan_name) {
  if (!src) {
    return std::nullopt;
  }
  auto it = absl::c_find_if(src->channels(),
                            [&](const PackageInterfaceProto::Channel& f) {
                              return f.name() == chan_name;
                            });
  if (it != src->channels().end()) {
    return *it;
  }
  return std::nullopt;
}

std::string PipelineSignalName(std::string_view root, int64_t stage) {
  std::string base;
  // Strip any existing pipeline prefix from the name.
  static constexpr LazyRE2 kPipelinePrefix = {.pattern_ = R"(^p\d+_(.+))"};
  if (!RE2::PartialMatch(root, *kPipelinePrefix, &base)) {
    base = root;
  }
  return absl::StrFormat("p%d_%s", stage, SanitizeIdentifier(base));
}

absl::StatusOr<std::vector<Node*>> MakeInputReadyPortsForOutputChannels(
    std::vector<std::vector<StreamingOutput>>& streaming_outputs,
    int64_t stage_count, std::string_view ready_suffix, Block* block) {
  std::vector<Node*> result;

  // Add a ready input port for each streaming output. Gather the ready signals
  // into a vector. Ready signals from streaming outputs generated from Send
  // operations are conditioned upon the optional predicate value.
  for (Stage stage = 0; stage < stage_count; ++stage) {
    std::vector<Node*> active_readys;
    for (StreamingOutput& streaming_output : streaming_outputs[stage]) {
      if (streaming_output.GetPredicate().has_value()) {
        // Logic for the active ready signal for a Send operation with a
        // predicate `pred`.
        //
        //   active = !pred | pred && ready
        //          = !pred | ready
        XLS_ASSIGN_OR_RETURN(
            Node * not_pred,
            block->MakeNode<UnOp>(SourceInfo(),
                                  streaming_output.GetPredicate().value(),
                                  Op::kNot));
        // If predicate has an assigned name, let the not expression get
        // inlined. Otherwise, give a descriptive name.
        if (!streaming_output.GetPredicate().value()->HasAssignedName()) {
          not_pred->SetName(absl::StrFormat(
              "%s_not_pred", ChannelRefName(streaming_output.GetChannel())));
        }
        std::vector<Node*> operands{not_pred, streaming_output.GetReadyPort()};
        XLS_ASSIGN_OR_RETURN(
            Node * active_ready,
            block->MakeNode<NaryOp>(SourceInfo(), operands, Op::kOr));
        // not_pred will have an assigned name or be inlined, so only check
        // the ready port. If it has an assigned name, just let everything
        // inline. Otherwise, give a descriptive name.
        if (!streaming_output.GetReadyPort()->HasAssignedName()) {
          active_ready->SetName(
              absl::StrFormat("%s_active_ready",
                              ChannelRefName(streaming_output.GetChannel())));
        }
        active_readys.push_back(active_ready);
      } else {
        active_readys.push_back(streaming_output.GetReadyPort());
      }
    }

    // And reduce all the active ready signals. This signal is true iff all
    // active outputs are ready.
    XLS_ASSIGN_OR_RETURN(
        Node * all_active_outputs_ready,
        NaryAndIfNeeded(block, active_readys,
                        PipelineSignalName("all_active_outputs_ready", stage)));
    result.push_back(all_active_outputs_ready);
  }

  return result;
}

// For each input streaming channel add a corresponding valid port (input port).
// Combinationally combine those valid signals with their predicates
// to generate an all_active_inputs_valid signal.
//
// Upon success returns a Node* to the all_active_inputs_valid signal.
absl::StatusOr<std::vector<Node*>> MakeInputValidPortsForInputChannels(
    std::vector<std::vector<StreamingInput>>& streaming_inputs,
    int64_t stage_count, std::string_view valid_suffix, Block* block) {
  std::vector<Node*> result;

  for (Stage stage = 0; stage < stage_count; ++stage) {
    // Add a valid input port for each streaming input. Gather the valid
    // signals into a vector. Valid signals from streaming inputs generated
    // from Receive operations are conditioned upon the optional predicate
    // value.
    std::vector<Node*> active_valids;
    for (StreamingInput& streaming_input : streaming_inputs[stage]) {
      // Input ports for input channels are already created during
      // HandleReceiveNode().
      XLS_RET_CHECK(streaming_input.GetSignalValid().has_value());
      Node* streaming_input_valid = *streaming_input.GetSignalValid();

      if (streaming_input.GetPredicate().has_value()) {
        // Logic for the active valid signal for a Receive operation with a
        // predicate `pred`.
        //
        //   active = !pred | pred && valid
        //          = !pred | valid
        XLS_ASSIGN_OR_RETURN(
            Node * not_pred,
            block->MakeNode<UnOp>(SourceInfo(),
                                  streaming_input.GetPredicate().value(),
                                  Op::kNot));

        // If predicate has an assigned name, let the not expression get
        // inlined. Otherwise, give a descriptive name.
        if (!streaming_input.GetPredicate().value()->HasAssignedName()) {
          not_pred->SetName(absl::StrFormat(
              "%s_not_pred", ChannelRefName(streaming_input.GetChannel())));
        }
        std::vector<Node*> operands = {not_pred, streaming_input_valid};
        XLS_ASSIGN_OR_RETURN(
            Node * active_valid,
            block->MakeNode<NaryOp>(SourceInfo(), operands, Op::kOr));
        // not_pred will have an assigned name or be inlined, so only check
        // the ready port. If it has an assigned name, just let everything
        // inline. Otherwise, give a descriptive name.
        if (!streaming_input_valid->HasAssignedName()) {
          active_valid->SetName(absl::StrFormat(
              "%s_active_valid", ChannelRefName(streaming_input.GetChannel())));
        }
        active_valids.push_back(active_valid);
      } else {
        // No predicate is the same as pred = true, so
        // active = !pred | valid = !true | valid = false | valid = valid
        active_valids.push_back(streaming_input_valid);
      }
    }

    // And reduce all the active valid signals. This signal is true iff all
    // active inputs are valid.
    XLS_ASSIGN_OR_RETURN(
        Node * all_active_inputs_valid,
        NaryAndIfNeeded(block, active_valids,
                        PipelineSignalName("all_active_inputs_valid", stage)));
    result.push_back(all_active_inputs_valid);
  }

  return result;
}

// Returns or makes a node that is 1 when the block is under reset,
// if said reset signal exists.
//
//   - If no reset exists, std::nullopt is returned
//   - Active low reset signals are inverted.
//
// See also MakeOrWithResetNode()
absl::StatusOr<std::optional<Node*>> ResetAsserted(Block* block) {
  std::optional<Node*> reset_node = block->GetResetPort();
  std::optional<ResetBehavior> reset_behavior = block->GetResetBehavior();
  if (!reset_node.has_value()) {
    XLS_RET_CHECK(!reset_behavior.has_value());
    return std::nullopt;
  }
  XLS_RET_CHECK(reset_behavior.has_value());
  if (reset_behavior->active_low) {
    return block->MakeNode<UnOp>(/*loc=*/SourceInfo(), *reset_node, Op::kNot);
  }

  return reset_node;
}

absl::StatusOr<std::optional<Node*>> ResetNotAsserted(Block* block) {
  std::optional<Node*> reset_node = block->GetResetPort();
  std::optional<ResetBehavior> reset_behavior = block->GetResetBehavior();
  if (!reset_node.has_value()) {
    XLS_RET_CHECK(!reset_behavior.has_value());
    return std::nullopt;
  }
  XLS_RET_CHECK(reset_behavior.has_value());
  if (reset_behavior->active_low) {
    return reset_node;
  }

  return block->MakeNode<UnOp>(/*loc=*/SourceInfo(), *reset_node, Op::kNot);
}

// Given a node returns a node that is OR'd with the reset signal.
// if said reset signal exists.  That node can be thought of as
//     1 - If being reset or if the src_node is 1
//     0 - otherwise.
//
//   - If no reset exists, the node is returned and the graph unchanged.
//   - Active low reset signals are inverted so that the resulting signal
//      OR(src_node, NOT(reset))
//
// This is used to drive load_enable signals of pipeline valid registers.
absl::StatusOr<Node*> MakeOrWithResetNode(Node* src_node,
                                          std::string_view result_name,
                                          Block* block) {
  Node* result = src_node;

  XLS_ASSIGN_OR_RETURN(std::optional<Node*> maybe_reset_node,
                       ResetAsserted(block));

  if (maybe_reset_node.has_value()) {
    Node* reset_node = maybe_reset_node.value();
    XLS_ASSIGN_OR_RETURN(result, block->MakeNodeWithName<NaryOp>(
                                     /*loc=*/SourceInfo(),
                                     std::vector<Node*>({result, reset_node}),
                                     Op::kOr, result_name));
  }

  return result;
}

// If options specify it, adds and returns an input for a reset signal.
absl::Status MaybeAddResetPort(Block* block, const CodegenOptions& options) {
  // TODO(tedhong): 2021-09-18 Combine this with AddValidSignal
  if (options.reset().has_value()) {
    XLS_RETURN_IF_ERROR(block
                            ->AddResetPort(options.reset()->name(),
                                           options.GetResetBehavior().value())
                            .status());
  }

  return absl::OkStatus();
}

// Send/receive nodes are not cloned from the proc into the block, but the
// network of tokens connecting these send/receive nodes *is* cloned. This
// function removes the token operations.
absl::Status RemoveDeadTokenNodes(Block* block, CodegenPassUnit* unit) {
  // Receive nodes produce a tuple of a token and a data value. In the block
  // this becomes a tuple of a token and an InputPort. Run tuple simplification
  // to disentangle the tuples so DCE can do its work and eliminate the token
  // network.

  // TODO: We really shouldn't be running passes like this during block
  // conversion. These should be fully in the pipeline. This is work for the
  // future.

  // To limit the scope of the optimizations to the block itself, call the pass
  // individually on the block.
  OptimizationPassOptions options;
  PassResults results;
  OptimizationContext context;
  DataflowSimplificationPass dataflow_pass;
  XLS_RETURN_IF_ERROR(
      dataflow_pass.RunOnFunctionBase(block, options, &results, context)
          .status());
  DataflowSimplificationPass dataflow_simp_pass;
  XLS_RETURN_IF_ERROR(
      dataflow_simp_pass.RunOnFunctionBase(block, options, &results, context)
          .status());
  DeadCodeEliminationPass dce_pass;
  XLS_RETURN_IF_ERROR(
      dce_pass.RunOnFunctionBase(block, options, &results, context).status());

  CodegenPassResults codegen_results;
  CodegenPassOptions codegen_options;
  RegisterLegalizationPass reg_legalization_pass;
  XLS_RETURN_IF_ERROR(
      reg_legalization_pass.Run(unit, codegen_options, &codegen_results)
          .status());

  XLS_RETURN_IF_ERROR(
      dce_pass.RunOnFunctionBase(block, options, &results, context).status());

  // Nodes like cover and assert have token types and will cause
  // a dangling token network remaining.
  //
  // TODO(tedhong): 2022-02-14, clean up dangling token
  // network to ensure that deleted nodes can't be accessed via normal
  // ir operations.

  return absl::OkStatus();
}

static absl::Status ReplacePortOrFifoInstantiationDataOperand(Node* node,
                                                              Node* value) {
  if (node->Is<OutputPort>()) {
    return node->ReplaceOperandNumber(OutputPort::kOperandOperand, value);
  }
  XLS_RET_CHECK(node->Is<InstantiationInput>());
  return node->ReplaceOperandNumber(InstantiationInput::kDataOperand, value);
}

// Make valid ports (output) for the output channel.
//
// A valid signal is asserted iff all active
// inputs valid signals are asserted and the predicate of the data channel (if
// any) is asserted.
absl::Status MakeOutputValidPortsForOutputChannels(
    absl::Span<Node* const> all_active_inputs_valid,
    absl::Span<Node* const> pipelined_valids,
    absl::Span<Node* const> next_stage_open,
    std::vector<std::vector<StreamingOutput>>& streaming_outputs,
    std::string_view valid_suffix, Block* block) {
  for (Stage stage = 0; stage < streaming_outputs.size(); ++stage) {
    for (StreamingOutput& streaming_output : streaming_outputs.at(stage)) {
      std::vector<Node*> operands{all_active_inputs_valid.at(stage),
                                  pipelined_valids.at(stage),
                                  next_stage_open.at(stage)};

      if (streaming_output.GetPredicate().has_value()) {
        operands.push_back(streaming_output.GetPredicate().value());
      }

      XLS_ASSIGN_OR_RETURN(Node * valid, block->MakeNode<NaryOp>(
                                             SourceInfo(), operands, Op::kAnd));
      // The valid port/instantiation-input is already created but is set
      // to a dummy value.
      XLS_RETURN_IF_ERROR(ReplacePortOrFifoInstantiationDataOperand(
          streaming_output.GetValidPort(), valid));
    }
  }

  return absl::OkStatus();
}

// Make ready ports (output) for each input channel.
//
// A ready signal is asserted iff all active
// output ready signals are asserted and the predicate of the data channel (if
// any) is asserted.
absl::Status MakeOutputReadyPortsForInputChannels(
    absl::Span<Node* const> all_active_outputs_ready,
    std::vector<std::vector<StreamingInput>>& streaming_inputs,
    std::string_view ready_suffix, Block* block) {
  for (Stage stage = 0; stage < streaming_inputs.size(); ++stage) {
    for (StreamingInput& streaming_input : streaming_inputs[stage]) {
      Node* ready = all_active_outputs_ready.at(stage);
      if (streaming_input.GetPredicate().has_value()) {
        std::vector<Node*> operands{streaming_input.GetPredicate().value(),
                                    all_active_outputs_ready.at(stage)};
        XLS_ASSIGN_OR_RETURN(
            ready, block->MakeNode<NaryOp>(SourceInfo(), operands, Op::kAnd));
      }

      // The ready port/instantiation-input is already created but is set to a
      // dummy value.
      XLS_RETURN_IF_ERROR(ReplacePortOrFifoInstantiationDataOperand(
          streaming_input.GetReadyPort(), ready));
    }
  }
  return absl::OkStatus();
}

}  // namespace xls::verilog
