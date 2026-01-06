// Copyright 2025 The XLS Authors
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

#include "xls/codegen_v_1_5/function_io_lowering_pass.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "cppitertools/reversed.hpp"
#include "xls/codegen/conversion_utils.h"
#include "xls/codegen_v_1_5/block_conversion_pass.h"
#include "xls/common/casts.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/block.h"
#include "xls/ir/function_base.h"
#include "xls/ir/node.h"
#include "xls/ir/node_util.h"
#include "xls/ir/nodes.h"
#include "xls/ir/op.h"
#include "xls/ir/package.h"
#include "xls/ir/register.h"
#include "xls/ir/source_location.h"
#include "xls/ir/value_utils.h"
#include "xls/passes/pass_base.h"

namespace xls::codegen {
namespace {

absl::StatusOr<Node*> FlopNode(
    ScheduledBlock* block, std::string_view name, Node* input,
    int64_t stage_index, std::optional<Node*> load_enable = std::nullopt) {
  XLS_ASSIGN_OR_RETURN(
      Register * input_flop,
      block->AddRegister(name, input->GetType(),
                         block->GetResetPort().has_value()
                             ? std::make_optional(ZeroOfType(input->GetType()))
                             : std::nullopt));
  XLS_RETURN_IF_ERROR(block
                          ->MakeNodeWithName<RegisterWrite>(
                              input->loc(), input, load_enable,
                              /*reset=*/block->GetResetPort(), input_flop,
                              absl::StrCat(name, "_write"))
                          .status());
  return block->MakeNodeWithNameInStage<RegisterRead>(
      stage_index, input->loc(), input_flop, absl::StrCat(name, "_read"));
}

}  // namespace

absl::StatusOr<bool> FunctionIOLoweringPass::LowerParams(
    ScheduledBlock* block, const BlockConversionPassOptions& options) const {
  if (block->source() == nullptr || !block->source()->IsFunction()) {
    return false;
  }
  Function* source = block->source()->AsFunctionOrDie();
  if (source->params().empty()) {
    return false;
  }

  std::optional<Node*> input_valid = std::nullopt;
  if (options.codegen_options.valid_control().has_value()) {
    if (options.codegen_options.valid_control()->input_name().empty()) {
      return absl::InvalidArgumentError(
          "Must specify input name of valid signal.");
    }

    XLS_ASSIGN_OR_RETURN(
        input_valid, block->AddInputPort(
                         options.codegen_options.valid_control()->input_name(),
                         block->package()->GetBitsType(1)));

    if (options.codegen_options.flop_inputs()) {
      XLS_ASSIGN_OR_RETURN(
          input_valid,
          FlopNode(block,
                   absl::StrCat(
                       options.codegen_options.valid_control()->input_name(),
                       "__input_flop"),
                   *input_valid,
                   /*stage_index=*/0));
    }

    // Update the active inputs valid signal to include the input-valid signal,
    // which will stop the pipeline from running new activations when the input
    // is invalid. This also automatically propagates validity to the output in
    // case an output-valid signal was requested.
    XLS_RETURN_IF_ERROR(ReplaceWithAnd(block->stages()[0].active_inputs_valid(),
                                       *input_valid, /*combine_literals=*/false)
                            .status());
  }

  std::vector<Param*> params_to_remove;
  params_to_remove.reserve(source->params().size());
  for (Param* param : source->params()) {
    std::string param_name = param->GetName();
    param->SetName(absl::StrCat("__temp_param_", param_name));
    XLS_ASSIGN_OR_RETURN(
        InputPort * port,
        block->AddInputPort(param_name, param->GetType(), param->loc()));
    XLS_ASSIGN_OR_RETURN(bool added, block->AddNodeToStage(0, port));
    XLS_RET_CHECK(added);
    if (std::optional<PackageInterfaceProto::Function> f =
            ::xls::verilog::FindFunctionInterface(
                options.codegen_options.package_interface(), block->name())) {
      // Record sv-type associated with this port.
      auto it = absl::c_find_if(
          f->parameters(), [&](const PackageInterfaceProto::NamedValue& p) {
            return p.name() == param_name;
          });
      if (it != f->parameters().end() && it->has_sv_type()) {
        port->set_system_verilog_type(it->sv_type());
      }
    }

    Node* input = port;
    if (options.codegen_options.flop_inputs()) {
      XLS_ASSIGN_OR_RETURN(
          input,
          FlopNode(block, absl::StrCat(param_name, "__input_flop"), input,
                   /*stage_index=*/0,
                   /*load_enable=*/input_valid));
    }

    XLS_RETURN_IF_ERROR(
        param->ReplaceUsesWith(input, /*replace_implicit_uses=*/false));
    if (block->source_return_value() == param) {
      block->SetSourceReturnValue(input);
    }
    XLS_RETURN_IF_ERROR(block->RemoveNodeFromStage(param).status());
    params_to_remove.push_back(param);
  }
  for (Param* param : iter::reversed(params_to_remove)) {
    XLS_RETURN_IF_ERROR(source->RemoveNode(param));
  }

  return true;
}

absl::StatusOr<bool> FunctionIOLoweringPass::LowerReturnValue(
    ScheduledBlock* block, const BlockConversionPassOptions& options) const {
  if (block->source() == nullptr || !block->source()->IsFunction() ||
      block->source_return_value() == nullptr) {
    return false;
  }

  if (options.codegen_options.split_outputs()) {
    return absl::UnimplementedError("Splitting outputs not supported.");
  }

  // The return value is always assumed to be in the last stage - or possibly
  // unscheduled, if (e.g.) returning a literal. If it's not in the last stage,
  // it must be a Param - so we need to introduce an identity node to ensure
  // that the value gets piped through the pipeline registers.
  Node* return_value = block->source_return_value();
  if (block->IsStaged(return_value)) {
    XLS_ASSIGN_OR_RETURN(int64_t stage_index,
                         block->GetStageIndex(return_value));
    const int64_t last_stage = block->stages().size() - 1;
    if (stage_index != last_stage) {
      XLS_ASSIGN_OR_RETURN(
          return_value,
          block->MakeNodeInStage<UnOp>(block->stages().size() - 1, SourceInfo(),
                                       return_value, Op::kIdentity));
    }
  }

  std::optional<Node*> output_valid = std::nullopt;
  if (options.codegen_options.valid_control().has_value() &&
      !options.codegen_options.valid_control()->output_name().empty()) {
    // If the valid signal is passed all the way through to an output port, then
    // the block must have a reset port. Otherwise, garbage will be passed out
    // of the valid out port until the pipeline flushes. If there is not a valid
    // output port, it's ok for the flopped valid to have garbage values because
    // it is only used as a term in load enables for power savings.
    if (!block->GetResetBehavior().has_value()) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "Block `%s` has valid signal output but no reset; this would produce "
          "an incorrect valid-output signal.",
          block->name()));
    }

    output_valid = block->stages().back().outputs_valid();
    if (options.codegen_options.flop_outputs()) {
      XLS_ASSIGN_OR_RETURN(
          output_valid,
          FlopNode(block,
                   absl::StrCat(
                       options.codegen_options.valid_control()->output_name(),
                       "__output_flop"),
                   *output_valid,
                   /*stage_index=*/block->stages().size() - 1));
    }
    XLS_ASSIGN_OR_RETURN(
        OutputPort * valid_output_port,
        block->AddOutputPort(
            options.codegen_options.valid_control()->output_name(),
            *output_valid));
    XLS_ASSIGN_OR_RETURN(
        bool added,
        block->AddNodeToStage(block->stages().size() - 1, valid_output_port));
    XLS_RET_CHECK(added);
  }

  Node* output = block->source_return_value();
  if (options.codegen_options.flop_outputs()) {
    XLS_ASSIGN_OR_RETURN(
        output,
        FlopNode(block,
                 absl::StrCat(options.codegen_options.output_port_name(),
                              "__output_flop"),
                 output,
                 /*stage_index=*/block->stages().size() - 1,
                 /*load_enable=*/block->stages().back().outputs_valid()));
  }
  XLS_ASSIGN_OR_RETURN(
      OutputPort * port,
      block->AddOutputPort(options.codegen_options.output_port_name(), output));
  XLS_ASSIGN_OR_RETURN(bool added,
                       block->AddNodeToStage(block->stages().size() - 1, port));
  XLS_RET_CHECK(added);
  if (std::optional<PackageInterfaceProto::Function> f =
          ::xls::verilog::FindFunctionInterface(
              options.codegen_options.package_interface(), block->name());
      f && f->has_sv_result_type()) {
    // Record sv-type associated with this port.
    port->set_system_verilog_type(f->sv_result_type());
  }

  // Clear the return value from the source function.
  block->SetSourceReturnValue(nullptr);

  return true;
}

absl::StatusOr<bool> FunctionIOLoweringPass::RunInternal(
    Package* package, const BlockConversionPassOptions& options,
    PassResults* results) const {
  bool changed = false;
  for (const std::unique_ptr<Block>& block : package->blocks()) {
    if (!block->IsScheduled()) {
      continue;
    }
    ScheduledBlock* scheduled_block = down_cast<ScheduledBlock*>(block.get());
    XLS_ASSIGN_OR_RETURN(bool changed_params,
                         LowerParams(scheduled_block, options));
    XLS_ASSIGN_OR_RETURN(bool changed_return_value,
                         LowerReturnValue(scheduled_block, options));
    changed |= changed_params || changed_return_value;
  }
  return changed;
}

}  // namespace xls::codegen
