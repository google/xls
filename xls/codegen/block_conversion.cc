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

#include "xls/codegen/block_conversion.h"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/base/nullability.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "xls/codegen/bdd_io_analysis.h"
#include "xls/codegen/clone_nodes_into_block_handler.h"
#include "xls/codegen/codegen_options.h"
#include "xls/codegen/codegen_pass.h"
#include "xls/codegen/concurrent_stage_groups.h"
#include "xls/codegen/conversion_utils.h"
#include "xls/codegen/mark_channel_fifos_pass.h"
#include "xls/codegen/proc_block_conversion.h"
#include "xls/common/logging/log_lines.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/bits.h"
#include "xls/ir/block.h"
#include "xls/ir/function_base.h"
#include "xls/ir/instantiation.h"
#include "xls/ir/name_uniquer.h"
#include "xls/ir/node.h"
#include "xls/ir/nodes.h"
#include "xls/ir/op.h"
#include "xls/ir/package.h"
#include "xls/ir/proc.h"
#include "xls/ir/proc_elaboration.h"
#include "xls/ir/register.h"
#include "xls/ir/source_location.h"
#include "xls/ir/state_element.h"
#include "xls/ir/topo_sort.h"
#include "xls/ir/type.h"
#include "xls/ir/value.h"
#include "xls/ir/xls_ir_interface.pb.h"
#include "xls/passes/pass_base.h"
#include "xls/scheduling/pipeline_schedule.h"
#include "xls/scheduling/scheduling_options.h"

namespace xls {
namespace verilog {
namespace {

struct ValidPorts {
  InputPort* input;
  OutputPort* output;
};

// Plumb valid signal through the pipeline stages, ANDing with a valid produced
// by each stage. Gather the pipelined valid signal in a vector where the
// zero-th element is the input port and subsequent elements are the pipelined
// valid signal from each stage.
static absl::StatusOr<std::vector<std::optional<Node*>>>
MakePipelineStagesForValid(
    Node* valid_input_port,
    absl::Span<const PipelineStageRegisters> pipeline_registers, Block* block,
    const CodegenOptions& options) {
  Type* u1 = block->package()->GetBitsType(1);

  std::vector<std::optional<Node*>> pipelined_valids(pipeline_registers.size() +
                                                     1);
  pipelined_valids[0] = valid_input_port;

  for (int64_t stage = 0; stage < pipeline_registers.size(); ++stage) {
    // If the valid signal is passed all the way through to an output port, then
    // the block must have a reset port. Otherwise, garbage will be passed out
    // of the valid out port until the pipeline flushes. If there is not a valid
    // output port, it's ok for the flopped valid to have garbage values because
    // it is only used as a term in load enables for power savings.
    if (!block->GetResetBehavior().has_value() &&
        options.valid_control().value().has_output_name()) {
      return absl::InternalError(absl::StrFormat(
          "Block `%s` has valid signal output but no reset", block->name()));
    }

    // Add valid register to each pipeline stage.
    Register* valid_reg;
    if (block->GetResetBehavior().has_value()) {
      XLS_ASSIGN_OR_RETURN(valid_reg,
                           block->AddRegisterWithZeroResetValue(
                               PipelineSignalName("valid", stage), u1));
    } else {
      XLS_ASSIGN_OR_RETURN(
          valid_reg,
          block->AddRegister(PipelineSignalName("valid", stage), u1));
    }
    XLS_RETURN_IF_ERROR(block
                            ->MakeNode<RegisterWrite>(
                                /*loc=*/SourceInfo(), *pipelined_valids[stage],
                                /*load_enable=*/std::nullopt,
                                /*reset=*/block->GetResetPort(), valid_reg)
                            .status());
    XLS_ASSIGN_OR_RETURN(pipelined_valids[stage + 1],
                         block->MakeNode<RegisterRead>(
                             /*loc=*/SourceInfo(), valid_reg));
  }

  return pipelined_valids;
}

static absl::StatusOr<ValidPorts> AddValidSignal(
    absl::Span<PipelineStageRegisters> pipeline_registers,
    const CodegenOptions& options, Block* block,
    std::vector<std::optional<Node*>>& pipelined_valids,
    absl::flat_hash_map<Node*, Stage>& node_to_stage_map) {
  // Add valid input port.
  XLS_RET_CHECK(options.valid_control().has_value());
  if (options.valid_control()->input_name().empty()) {
    return absl::InvalidArgumentError(
        "Must specify input name of valid signal.");
  }
  Type* u1 = block->package()->GetBitsType(1);
  XLS_ASSIGN_OR_RETURN(
      InputPort * valid_input_port,
      block->AddInputPort(options.valid_control()->input_name(), u1));

  // Plumb valid signal through the pipeline stages. Gather the pipelined valid
  // signal in a vector where the zero-th element is the input port and
  // subsequent elements are the pipelined valid signal from each stage.
  XLS_ASSIGN_OR_RETURN(
      pipelined_valids,
      MakePipelineStagesForValid(valid_input_port, pipeline_registers, block,
                                 options));

  // Use the pipelined valid signal as load enable for each datapath pipeline
  // register in each stage as a power optimization.
  for (int64_t stage = 0; stage < pipeline_registers.size(); ++stage) {
    // For each (non-valid-signal) pipeline register add `valid` or `valid ||
    // reset` (if reset exists) as a load enable. The `reset` term ensures the
    // pipeline flushes when reset is enabled.
    if (!pipeline_registers.at(stage).empty()) {
      XLS_ASSIGN_OR_RETURN(
          Node * load_enable,
          MakeOrWithResetNode(*pipelined_valids[stage],
                              PipelineSignalName("load_en", stage), block));

      for (PipelineRegister& pipeline_reg : pipeline_registers.at(stage)) {
        std::optional<Node*> reset_signal;
        if (block->GetResetPort().has_value() &&
            options.reset()->reset_data_path()) {
          reset_signal = block->GetResetPort();
        }
        XLS_ASSIGN_OR_RETURN(
            auto* new_write,
            block->MakeNode<RegisterWrite>(
                /*loc=*/SourceInfo(), pipeline_reg.reg_write->data(),
                /*load_enable=*/load_enable,
                /*reset=*/reset_signal, pipeline_reg.reg));
        if (node_to_stage_map.contains(pipeline_reg.reg_write)) {
          Stage s = node_to_stage_map[pipeline_reg.reg_write];
          node_to_stage_map.erase(pipeline_reg.reg_write);
          node_to_stage_map[new_write] = s;
        }
        XLS_RETURN_IF_ERROR(block->RemoveNode(pipeline_reg.reg_write));
        pipeline_reg.reg_write = new_write;
      }
    }
  }

  // Add valid output port.
  OutputPort* valid_output_port = nullptr;
  if (options.valid_control().has_value() &&
      options.valid_control()->has_output_name()) {
    if (options.valid_control()->output_name().empty()) {
      return absl::InvalidArgumentError(
          "Must specify output name of valid signal.");
    }
    XLS_ASSIGN_OR_RETURN(
        valid_output_port,
        block->AddOutputPort(options.valid_control()->output_name(),
                             pipelined_valids.back().value()));
  }

  return ValidPorts{valid_input_port, valid_output_port};
}

// Return a schedule based on the given schedule that adjusts the cycles of the
// nodes to introduce pipeline registers immediately after input ports or
// immediately before output ports based on the `flop_inputs` and `flop_outputs`
// options in CodegenOptions.
absl::StatusOr<PipelineSchedule> MaybeAddInputOutputFlopsToSchedule(
    const PipelineSchedule& schedule, const CodegenOptions& options) {
  // All function params must be scheduled in cycle 0.
  ScheduleCycleMap cycle_map;
  if (schedule.function_base()->IsFunction()) {
    for (Param* param : schedule.function_base()->params()) {
      XLS_RET_CHECK_EQ(schedule.cycle(param), 0);
      cycle_map[param] = 0;
    }
  }

  // If `flop_inputs` is true, adjust the cycle of all remaining nodes by one.
  int64_t cycle_offset = 0;
  if (options.flop_inputs()) {
    ++cycle_offset;
  }
  for (int64_t cycle = 0; cycle < schedule.length(); ++cycle) {
    for (Node* node : schedule.nodes_in_cycle(cycle)) {
      if (node->Is<Param>()) {
        continue;
      }
      cycle_map[node] = cycle + cycle_offset;
    }
  }

  // Add one more cycle to the schedule if `flop_outputs` is true. This only
  // changes the length of the schedule not the cycle that any node is placed
  // in. The final cycle is empty which effectively puts a pipeline register
  // between the nodes of the function and the output ports.
  if (options.flop_outputs()) {
    ++cycle_offset;
  }
  PipelineSchedule result(schedule.function_base(), cycle_map,
                          schedule.length() + cycle_offset);
  return std::move(result);
}

}  // namespace

absl::StatusOr<
    std::tuple<StreamingIOPipeline, std::optional<ConcurrentStageGroups>>>
CloneNodesIntoPipelinedBlock(
    FunctionBase* function_base, const PackageSchedule& package_schedule,
    const CodegenOptions& options, Block* block,
    const absl::flat_hash_map<FunctionBase*, Block*>& converted_blocks,
    std::optional<const ProcElaboration*> elab) {
  XLS_RET_CHECK(function_base->IsProc() || function_base->IsFunction());

  const PipelineSchedule& pipeline_schedule =
      package_schedule.GetSchedule(function_base);
  CloneNodesIntoBlockHandler cloner(function_base, pipeline_schedule.length(),
                                    options, block, &package_schedule);
  if (function_base->IsProc()) {
    XLS_RETURN_IF_ERROR(cloner.AddChannelPortsAndFifoInstantiations());
    XLS_RETURN_IF_ERROR(cloner.AddBlockInstantiations(converted_blocks));
  }
  for (int64_t stage = 0; stage < pipeline_schedule.length(); ++stage) {
    XLS_RET_CHECK_OK(
        cloner.CloneNodes(pipeline_schedule.nodes_in_cycle(stage), stage));
    XLS_RET_CHECK_OK(cloner.AddNextPipelineStage(stage));
  }

  XLS_RET_CHECK_OK(cloner.AddOutputPortsIfFunction(options.output_port_name()));
  XLS_RET_CHECK_OK(
      cloner.MarkMutualExclusiveStages(pipeline_schedule.length()));

  return std::make_tuple(cloner.GetResult(), cloner.GetConcurrentStages());
}

absl::StatusOr<std::string> StreamingIOName(Node* node) {
  switch (node->op()) {
    case Op::kInputPort:
      return std::string{node->As<InputPort>()->name()};
    case Op::kOutputPort:
      return std::string{node->As<OutputPort>()->name()};
    case Op::kInstantiationInput:
      return absl::StrCat(
          node->As<InstantiationInput>()->instantiation()->name(), "_",
          node->As<InstantiationInput>()->port_name());
    case Op::kInstantiationOutput:
      return absl::StrCat(
          node->As<InstantiationOutput>()->instantiation()->name(), "_",
          node->As<InstantiationOutput>()->port_name());
    default:
      return absl::InvalidArgumentError(absl::StrFormat(
          "Unsupported streaming operation %s.", OpToString(node->op())));
  }
}

absl::Status AddCombinationalFlowControl(
    std::vector<std::vector<StreamingInput>>& streaming_inputs,
    std::vector<std::vector<StreamingOutput>>& streaming_outputs,
    std::vector<std::optional<Node*>>& stage_valid,
    const CodegenOptions& options, Proc* proc, Block* block) {
  std::string_view valid_suffix = options.streaming_channel_valid_suffix();
  std::string_view ready_suffix = options.streaming_channel_ready_suffix();
  std::string_view data_suffix = options.streaming_channel_data_suffix();

  XLS_ASSIGN_OR_RETURN(
      std::vector<Node*> all_active_outputs_ready,
      MakeInputReadyPortsForOutputChannels(streaming_outputs, /*stage_count=*/1,
                                           ready_suffix, block));

  XLS_ASSIGN_OR_RETURN(
      std::vector<Node*> all_active_inputs_valid,
      MakeInputValidPortsForInputChannels(streaming_inputs, /*stage_count=*/1,
                                          valid_suffix, block));

  XLS_ASSIGN_OR_RETURN(Node * literal_1, block->MakeNode<xls::Literal>(
                                             SourceInfo(), Value(UBits(1, 1))));
  std::vector<Node*> pipelined_valids{literal_1};
  std::vector<Node*> next_stage_open{literal_1};
  XLS_RETURN_IF_ERROR(MakeOutputValidPortsForOutputChannels(
      all_active_inputs_valid, pipelined_valids, next_stage_open,
      streaming_outputs, valid_suffix, proc, {}, block));
  XLS_RETURN_IF_ERROR(MakeOutputDataPortsForOutputChannels(
      all_active_inputs_valid, pipelined_valids, next_stage_open,
      streaming_outputs, data_suffix, block));

  XLS_RETURN_IF_ERROR(MakeOutputReadyPortsForInputChannels(
      all_active_outputs_ready, streaming_inputs, ready_suffix, proc, {},
      block));

  XLS_RET_CHECK(stage_valid.empty());
  XLS_RET_CHECK_EQ(all_active_inputs_valid.size(), 1);
  stage_valid.push_back(all_active_inputs_valid.front());

  return absl::OkStatus();
}

absl::StatusOr<StreamingIOPipeline> CloneProcNodesIntoBlock(
    Proc* proc, const CodegenOptions& options, Block* block) {
  CloneNodesIntoBlockHandler cloner(proc, /*stage_count=*/0, options, block);
  XLS_RETURN_IF_ERROR(cloner.AddChannelPortsAndFifoInstantiations());
  XLS_RET_CHECK_OK(cloner.CloneNodes(TopoSort(proc), /*stage=*/0));
  return cloner.GetResult();
}

absl::Status SingleFunctionToPipelinedBlock(const PipelineSchedule& schedule,
                                            const CodegenOptions& options,
                                            CodegenContext& context,
                                            Function* f,
                                            Block* ABSL_NONNULL block) {
  if (options.manual_control().has_value()) {
    return absl::UnimplementedError("Manual pipeline control not implemented");
  }
  if (options.split_outputs()) {
    return absl::UnimplementedError("Splitting outputs not supported.");
  }
  if (options.manual_control().has_value()) {
    return absl::UnimplementedError("Manual pipeline control not implemented");
  }

  if (std::optional<int64_t> ii = f->GetInitiationInterval(); ii.has_value()) {
    block->SetInitiationInterval(*ii);
  }

  XLS_RET_CHECK_OK(MaybeAddResetPort(block, options));
  XLS_RET_CHECK_OK(MaybeAddClockPort(block, options));

  // Flopping inputs and outputs can be handled as a transformation to the
  // schedule. This makes the later code for creation of the pipeline simpler.
  // TODO(meheff): 2021/7/21 Add input/output flopping as an option to the
  // scheduler.
  XLS_ASSIGN_OR_RETURN(PipelineSchedule transformed_schedule,
                       MaybeAddInputOutputFlopsToSchedule(schedule, options));

  PackageSchedule package_schedule(std::move(transformed_schedule));
  absl::flat_hash_map<FunctionBase*, Block*> converted_blocks;
  XLS_ASSIGN_OR_RETURN(
      (auto [streaming_io_and_pipeline, concurrent_stages]),
      CloneNodesIntoPipelinedBlock(f, package_schedule, options, block,
                                   converted_blocks));

  std::optional<ValidPorts> valid_ports;
  if (options.valid_control().has_value()) {
    XLS_ASSIGN_OR_RETURN(
        valid_ports,
        AddValidSignal(
            absl::MakeSpan(streaming_io_and_pipeline.pipeline_registers),
            options, block, streaming_io_and_pipeline.pipeline_valid,
            streaming_io_and_pipeline.node_to_stage_map));
  }

  // Reorder the ports of the block to the following:
  //   - clk
  //   - reset (optional)
  //   - input valid (optional)
  //   - function inputs
  //   - output valid (optional)
  //   - function output
  // This is solely a cosmetic change to improve readability.
  std::vector<std::string> port_order;
  port_order.push_back(std::string{options.clock_name().value()});
  if (block->GetResetPort().has_value()) {
    port_order.push_back(block->GetResetPort().value()->GetName());
  }
  if (valid_ports.has_value()) {
    port_order.push_back(valid_ports->input->GetName());
  }
  for (Param* param : f->params()) {
    port_order.push_back(param->GetName());
  }
  if (valid_ports.has_value() && valid_ports->output != nullptr) {
    port_order.push_back(valid_ports->output->GetName());
  }
  port_order.push_back(std::string{options.output_port_name()});
  XLS_RETURN_IF_ERROR(block->ReorderPorts(port_order));

  context.SetMetadataForBlock(
      block,
      CodegenMetadata{
          .streaming_io_and_pipeline = std::move(streaming_io_and_pipeline),
          .concurrent_stages = std::move(concurrent_stages),
      });

  return absl::OkStatus();
}

// Adds a register between the node and all its downstream users.
// Returns the new register added.
absl::StatusOr<RegisterRead*> AddRegisterAfterNode(
    std::string_view name_prefix, std::optional<Node*> load_enable,
    Node* node) {
  Block* block = node->function_base()->AsBlockOrDie();

  Type* node_type = node->GetType();
  std::string name = absl::StrFormat("__%s_reg", name_prefix);

  XLS_ASSIGN_OR_RETURN(Register * reg,
                       block->AddRegisterWithZeroResetValue(name, node_type));

  XLS_ASSIGN_OR_RETURN(RegisterRead * reg_read,
                       block->MakeNodeWithName<RegisterRead>(
                           /*loc=*/node->loc(),
                           /*reg=*/reg,
                           /*name=*/name));

  XLS_RETURN_IF_ERROR(node->ReplaceUsesWith(reg_read));

  XLS_RETURN_IF_ERROR(block
                          ->MakeNode<RegisterWrite>(
                              /*loc=*/node->loc(),
                              /*data=*/node,
                              /*load_enable=*/load_enable,
                              /*reset=*/block->GetResetPort(),
                              /*reg=*/reg)
                          .status());

  return reg_read;
}

// Add a zero-latency buffer after a set of data/valid/ready signal.
//
// Logic will be inserted immediately after from_data and from node.
// Logic will be inserted immediately before from_rdy,
//   from_rdy must be a node with a single operand.
//
// Updates valid_nodes with the additional nodes associated with valid
// registers.
absl::StatusOr<Node*> AddZeroLatencyBufferToRDVNodes(
    Node* from_data, Node* from_valid, Node* from_rdy,
    std::string_view name_prefix, Block* block,
    std::vector<std::optional<Node*>>& valid_nodes) {
  CHECK_EQ(from_rdy->operand_count(), 1);

  // Add a node for load_enables (will be removed later).
  XLS_ASSIGN_OR_RETURN(Node * literal_1, block->MakeNode<xls::Literal>(
                                             SourceInfo(), Value(UBits(1, 1))));

  // Create data/valid and their skid counterparts.
  XLS_ASSIGN_OR_RETURN(
      RegisterRead * data_skid_reg_read,
      AddRegisterAfterNode(/*name_prefix=*/absl::StrCat(name_prefix, "_skid"),
                           /*load_enable=*/literal_1, from_data));

  XLS_ASSIGN_OR_RETURN(
      RegisterRead * data_valid_skid_reg_read,
      AddRegisterAfterNode(
          /*name_prefix=*/absl::StrCat(name_prefix, "_valid_skid"),
          /*load_enable=*/literal_1, from_valid));

  // If data_valid_skid_reg_read is 1, then data/valid outputs should
  // be selected from the skid set.
  XLS_ASSIGN_OR_RETURN(
      Node * to_valid,
      block->MakeNodeWithName<NaryOp>(
          /*loc=*/from_data->loc(),
          std::vector<Node*>{from_valid, data_valid_skid_reg_read}, Op::kOr,
          absl::StrCat(name_prefix, "_valid_or")));
  XLS_RETURN_IF_ERROR(data_valid_skid_reg_read->ReplaceUsesWith(to_valid));

  XLS_ASSIGN_OR_RETURN(
      Node * to_data,
      block->MakeNodeWithName<Select>(
          /*loc=*/from_data->loc(),
          /*selector=*/data_valid_skid_reg_read,
          /*cases=*/std::vector<Node*>{from_data, data_skid_reg_read},
          /*default_value=*/std::nullopt,
          /*name=*/absl::StrCat(name_prefix, "_select")));
  XLS_RETURN_IF_ERROR(data_skid_reg_read->ReplaceUsesWith(to_data));

  // Input can be accepted whenever the skid registers
  // are empty/invalid.
  Node* to_is_ready = from_rdy->operand(0);

  XLS_ASSIGN_OR_RETURN(
      Node * from_skid_rdy,
      block->MakeNodeWithName<UnOp>(
          /*loc=*/SourceInfo(), data_valid_skid_reg_read, Op::kNot,
          absl::StrCat(name_prefix, "_from_skid_rdy")));
  XLS_RETURN_IF_ERROR(from_rdy->ReplaceOperandNumber(0, from_skid_rdy));

  // Skid is loaded from 1st stage whenever
  //   a) the input is being read (input_ready_and_valid == 1) and
  //       --> which implies that the skid is invalid
  //   b) the output is not ready (to_is_ready == 0) and
  XLS_ASSIGN_OR_RETURN(Node * to_is_not_rdy,
                       block->MakeNodeWithName<UnOp>(
                           /*loc=*/SourceInfo(), to_is_ready, Op::kNot,
                           absl::StrCat(name_prefix, "_to_is_not_rdy")));

  XLS_ASSIGN_OR_RETURN(
      Node * skid_data_load_en,
      block->MakeNodeWithName<NaryOp>(
          /*loc=*/SourceInfo(),
          std::vector<Node*>{from_valid, from_skid_rdy, to_is_not_rdy},
          Op::kAnd, absl::StrCat(name_prefix, "_skid_data_load_en")));

  // Skid is reset (valid set to zero) to invalid whenever
  //   a) skid is valid and
  //   b) output is ready
  XLS_ASSIGN_OR_RETURN(
      Node * skid_valid_set_zero,
      block->MakeNodeWithName<NaryOp>(
          /*loc=*/SourceInfo(),
          std::vector<Node*>{data_valid_skid_reg_read, to_is_ready}, Op::kAnd,
          absl::StrCat(name_prefix, "_skid_valid_set_zero")));

  // Skid valid changes from 0 to 1 (load), or 1 to 0 (set zero).
  XLS_ASSIGN_OR_RETURN(
      Node * skid_valid_load_en,
      block->MakeNodeWithName<NaryOp>(
          /*loc=*/SourceInfo(),
          std::vector<Node*>{skid_data_load_en, skid_valid_set_zero}, Op::kOr,
          absl::StrCat(name_prefix, "_skid_valid_load_en")));

  XLS_ASSIGN_OR_RETURN(
      RegisterWrite * data_skid_reg_write,
      block->GetUniqueRegisterWrite(data_skid_reg_read->GetRegister()));
  XLS_RETURN_IF_ERROR(
      data_skid_reg_write->ReplaceExistingLoadEnable(skid_data_load_en));

  XLS_ASSIGN_OR_RETURN(
      RegisterWrite * data_valid_skid_reg_write,
      block->GetUniqueRegisterWrite(data_valid_skid_reg_read->GetRegister()));

  // If the skid valid is being set
  //   - If it's being set to 1, then the input is being read,
  //     and the prior data is being stored into the skid
  //   - If it's being set to 0, then the input is not being read
  //     and we are clearing the skid and sending the data to the output
  // this implies that
  //   skid_valid := skid_valid_load_en ? !skid_valid : skid_valid
  XLS_RETURN_IF_ERROR(
      data_valid_skid_reg_write->ReplaceExistingLoadEnable(skid_valid_load_en));
  XLS_RETURN_IF_ERROR(
      data_valid_skid_reg_write->ReplaceOperandNumber(0, from_skid_rdy));

  valid_nodes.push_back(to_valid);

  return to_data;
}

absl::StatusOr<std::vector<FunctionBase*>> GetBlockConversionOrder(
    Package* package, absl::Span<Proc* const> procs_to_convert,
    const std::optional<ProcElaboration>& proc_elab) {
  FunctionBase* top = *package->GetTop();
  if (top->IsFunction()) {
    XLS_RET_CHECK(procs_to_convert.empty());
    return std::vector<FunctionBase*>({top});
  }
  CHECK(top->IsProc());
  if (package->ChannelsAreProcScoped()) {
    XLS_RET_CHECK(procs_to_convert.empty());
    XLS_RET_CHECK(proc_elab.has_value());
    // The order of block conversion must be from the leaf up as *instantiated*
    // procs must be converted before *instantiating* proc.
    std::vector<FunctionBase*> order(proc_elab->procs().begin(),
                                     proc_elab->procs().end());
    std::reverse(order.begin(), order.end());
    return order;
  }
  // For the non-proc-scoped channels case, the set of procs to convert must be
  // given.
  XLS_RET_CHECK(!procs_to_convert.empty());
  XLS_RET_CHECK(!proc_elab.has_value());
  std::vector<FunctionBase*> order(procs_to_convert.begin(),
                                   procs_to_convert.end());
  std::sort(order.begin(), order.end(), FunctionBase::NameLessThan);
  return order;
}

absl::StatusOr<CodegenContext> PackageToPipelinedBlocks(
    const PackageSchedule& package_schedule, const CodegenOptions& options,
    Package* package) {
  XLS_RET_CHECK_GT(package_schedule.GetSchedules().size(), 0);
  VLOG(3) << "Converting package to pipelined blocks:";
  XLS_VLOG_LINES(3, package->DumpIr());

  if (options.manual_control().has_value()) {
    return absl::UnimplementedError("Manual pipeline control not implemented");
  }
  if (options.split_outputs()) {
    return absl::UnimplementedError("Splitting outputs not supported.");
  }

  XLS_RET_CHECK(package->GetTop().has_value());
  FunctionBase* top = *package->GetTop();

  std::vector<Proc*> procs_to_convert;
  if (!package->ChannelsAreProcScoped() &&
      package->GetTop().value()->IsProc()) {
    for (FunctionBase* fb : package_schedule.GetScheduledFunctionBases()) {
      if (fb->IsProc()) {
        procs_to_convert.push_back(fb->AsProcOrDie());
      }
    }
  }

  std::optional<ProcElaboration> proc_elab;
  if (top->IsProc() && package->ChannelsAreProcScoped()) {
    XLS_ASSIGN_OR_RETURN(proc_elab,
                         ProcElaboration::Elaborate(top->AsProcOrDie()));
  }

  XLS_ASSIGN_OR_RETURN(
      std::vector<FunctionBase*> conversion_order,
      GetBlockConversionOrder(package, procs_to_convert, proc_elab));

  // Make `unit` optional because we haven't created the top block yet. We will
  // create it on the first iteration and emplace `unit`.
  std::string module_name(options.module_name().value_or(top->name()));
  Block* top_block =
      package->AddBlock(std::make_unique<Block>(module_name, package));
  top_block->SetFunctionBaseProvenance(top);

  // We use a uniquer here because the top block name comes from the codegen
  // option's `module_name` field (if set). A non-top proc could have the same
  // name, so the name uniquer will ensure that the sub-block gets a suffix if
  // needed.
  NameUniquer block_name_uniquer("__");
  block_name_uniquer.GetSanitizedUniqueName(module_name);
  CodegenContext context(top_block);

  // Run codegen passes as appropriate
  {
    MarkChannelFifosPass mark_chans;
    const CodegenPassOptions cg_options = {
        .codegen_options = options,
    };
    PassResults results;
    XLS_RETURN_IF_ERROR(
        mark_chans.Run(package, cg_options, &results, context).status());
  }

  absl::flat_hash_map<FunctionBase*, Block*> converted_blocks;
  for (FunctionBase* fb : conversion_order) {
    XLS_RET_CHECK(package_schedule.GetSchedules().contains(fb))
        << absl::StrFormat("Missing schedule for functionbase `%s`",
                           fb->name());

    const PipelineSchedule& schedule = package_schedule.GetSchedule(fb);
    std::string sub_block_name =
        block_name_uniquer.GetSanitizedUniqueName(fb->name());
    Block* sub_block;
    if (fb == top) {
      sub_block = top_block;
    } else {
      sub_block =
          package->AddBlock(std::make_unique<Block>(sub_block_name, package));
      sub_block->SetFunctionBaseProvenance(fb);
    }
    if (fb->IsProc()) {
      absl::Span<ProcInstance* const> instances;
      if (package->ChannelsAreProcScoped()) {
        XLS_RET_CHECK(proc_elab.has_value());
        instances = proc_elab->GetInstances(fb->AsProcOrDie());
      }
      XLS_RETURN_IF_ERROR(SingleProcToPipelinedBlock(
          package_schedule, options, context, fb->AsProcOrDie(), instances,
          sub_block, converted_blocks,
          proc_elab.has_value() ? std::make_optional(&proc_elab.value())
                                : std::nullopt));
    } else if (fb->IsFunction()) {
      XLS_RET_CHECK_EQ(conversion_order.size(), 1);
      XLS_RET_CHECK_EQ(fb, top);
      XLS_RETURN_IF_ERROR(SingleFunctionToPipelinedBlock(
          schedule, options, context, fb->AsFunctionOrDie(), sub_block));
    } else {
      return absl::InvalidArgumentError(absl::StrFormat(
          "FunctionBase %s was not a function or proc.", fb->name()));
    }
    converted_blocks[fb] = sub_block;
  }

  // Avoid leaving any dangling pointers.
  context.GcMetadata();

  return context;
}

absl::StatusOr<CodegenContext> FunctionBaseToPipelinedBlock(
    const PipelineSchedule& schedule, const CodegenOptions& options,
    FunctionBase* f) {
  PackageSchedule package_schedule(schedule);
  std::optional<FunctionBase*> old_top = f->package()->GetTop();
  XLS_RETURN_IF_ERROR(f->package()->SetTop(f));

  // Don't return yet if there's an error- we need to restore old_top first.
  absl::StatusOr<CodegenContext> result_or_status =
      PackageToPipelinedBlocks(package_schedule, options, f->package());
  XLS_RETURN_IF_ERROR(f->package()->SetTop(old_top));
  return result_or_status;
}

absl::StatusOr<CodegenContext> FunctionToCombinationalBlock(
    Function* f, const CodegenOptions& options) {
  XLS_RET_CHECK(!options.valid_control().has_value())
      << "Combinational block generator does not support valid control.";
  std::string module_name(options.module_name().value_or(f->name()));
  Block* block = f->package()->AddBlock(
      std::make_unique<Block>(module_name, f->package()));
  block->SetFunctionBaseProvenance(f);

  // A map from the nodes in 'f' to their corresponding node in the block.
  absl::flat_hash_map<Node*, Node*> node_map;
  CodegenContext context(block);

  // Emit the parameters first to ensure their order is preserved in the
  // block.
  auto func_interface =
      FindFunctionInterface(options.package_interface(), f->name());
  for (Param* param : f->params()) {
    XLS_ASSIGN_OR_RETURN(
        InputPort * input,
        block->AddInputPort(param->GetName(), param->GetType(), param->loc()));
    node_map[param] = input;
    if (func_interface) {
      auto name =
          absl::c_find_if(func_interface->parameters(),
                          [&](const PackageInterfaceProto::NamedValue& p) {
                            return p.name() == param->name();
                          });
      if (name != func_interface->parameters().end() && name->has_sv_type()) {
        input->set_system_verilog_type(name->sv_type());
      }
    }
  }

  for (Node* node : TopoSort(f)) {
    if (node->Is<Param>()) {
      continue;
    }

    std::vector<Node*> new_operands;
    for (Node* operand : node->operands()) {
      new_operands.push_back(node_map.at(operand));
    }
    XLS_ASSIGN_OR_RETURN(Node * block_node,
                         node->CloneInNewFunction(new_operands, block));
    node_map[node] = block_node;
  }

  XLS_ASSIGN_OR_RETURN(OutputPort * output,
                       block->AddOutputPort(options.output_port_name(),
                                            node_map.at(f->return_value())));
  if (func_interface && func_interface->has_sv_result_type()) {
    output->set_system_verilog_type(func_interface->sv_result_type());
  }

  // Create empty metadata object for the block.
  context.SetMetadataForBlock(block, CodegenMetadata());

  context.GcMetadata();
  return context;
}

absl::StatusOr<CodegenContext> ProcToCombinationalBlock(
    Proc* proc, const CodegenOptions& options) {
  VLOG(3) << "Converting proc to combinational block:";
  XLS_VLOG_LINES(3, proc->DumpIr());

  // In a combinational module, the proc cannot have any state to avoid
  // combinational loops. That is, the only loop state must be empty tuples.
  if (proc->GetStateElementCount() > 1 &&
      !absl::c_all_of(proc->StateElements(), [&](StateElement* st) {
        return st->type() == proc->package()->GetTupleType({});
      })) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Proc must have no state (or state type is all empty tuples) when "
        "lowering to a combinational block. Proc state type is: {%s}",
        absl::StrJoin(proc->StateElements(), ", ",
                      [](std::string* out, StateElement* st) {
                        absl::StrAppend(out, st->type()->ToString());
                      })));
  }

  std::string module_name = std::string{options.module_name().has_value()
                                            ? options.module_name().value()
                                            : proc->name()};
  Block* block = proc->package()->AddBlock(
      std::make_unique<Block>(module_name, proc->package()));
  block->SetFunctionBaseProvenance(proc);

  XLS_ASSIGN_OR_RETURN(StreamingIOPipeline streaming_io,
                       CloneProcNodesIntoBlock(proc, options, block));

  int64_t number_of_outputs = 0;
  for (const auto& outputs : streaming_io.outputs) {
    number_of_outputs += outputs.size();
  }

  if (number_of_outputs > 1) {
    // TODO: do this analysis on a per-stage basis
    XLS_ASSIGN_OR_RETURN(bool streaming_outputs_mutually_exclusive,
                         AreStreamingOutputsMutuallyExclusive(proc));

    if (streaming_outputs_mutually_exclusive) {
      VLOG(3) << absl::StrFormat(
          "%d streaming outputs determined to be mutually exclusive",
          number_of_outputs);
    } else {
      return absl::UnimplementedError(absl::StrFormat(
          "Proc combinational generator only supports streaming "
          "output channels which can be determined to be mutually "
          "exclusive, got %d output channels which were not proven "
          "to be mutually exclusive",
          number_of_outputs));
    }
  }

  XLS_RET_CHECK_EQ(streaming_io.pipeline_registers.size(), 0);

  XLS_RETURN_IF_ERROR(AddCombinationalFlowControl(
      streaming_io.inputs, streaming_io.outputs, streaming_io.stage_valid,
      options, proc, block));

  // TODO(tedhong): 2021-09-23 Remove and add any missing functionality to
  //                codegen pipeline.
  CodegenContext context(block);
  context.SetMetadataForBlock(
      context.top_block(),
      CodegenMetadata{
          .streaming_io_and_pipeline = std::move(streaming_io),
          .concurrent_stages = std::nullopt,
      });
  XLS_RETURN_IF_ERROR(RemoveDeadTokenNodes(block, context));
  VLOG(3) << "After RemoveDeadTokenNodes";
  XLS_VLOG_LINES(3, block->package()->DumpIr());

  context.GcMetadata();
  return context;
}

absl::StatusOr<CodegenContext> FunctionBaseToCombinationalBlock(
    FunctionBase* f, const CodegenOptions& options) {
  if (f->IsFunction()) {
    return FunctionToCombinationalBlock(f->AsFunctionOrDie(), options);
  }
  if (f->IsProc()) {
    return ProcToCombinationalBlock(f->AsProcOrDie(), options);
  }
  return absl::InvalidArgumentError(absl::StrFormat(
      "FunctionBase %s was not a function or proc.", f->name()));
}

}  // namespace verilog
}  // namespace xls
