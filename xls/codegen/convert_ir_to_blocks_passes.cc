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

#include "xls/codegen/convert_ir_to_blocks_passes.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "xls/codegen/bdd_io_analysis.h"
#include "xls/codegen/block_conversion.h"
#include "xls/codegen/codegen_pass.h"
#include "xls/codegen/codegen_util.h"
#include "xls/codegen/conversion_utils.h"
#include "xls/codegen/proc_block_conversion.h"
#include "xls/common/logging/log_lines.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/block.h"
#include "xls/ir/function_base.h"
#include "xls/ir/nodes.h"
#include "xls/ir/op.h"
#include "xls/ir/package.h"
#include "xls/ir/proc.h"
#include "xls/ir/proc_elaboration.h"
#include "xls/ir/state_element.h"
#include "xls/ir/topo_sort.h"
#include "xls/ir/xls_ir_interface.pb.h"
#include "xls/passes/pass_base.h"
#include "xls/scheduling/pipeline_schedule.h"

// -- Implementations of
// ConvertFuncsToCombinationalBlocksPass
// ConvertProcsToCombinationalBlocksPass
// ConvertFuncsToPipelinedBlocksPass
// ConvertProcsToPipelinedBlocksPass

namespace xls::verilog {

absl::StatusOr<bool> ConvertFuncsToCombinationalBlocksPass::RunInternal(
    Package* package, const CodegenPassOptions& options, PassResults* results,
    CodegenContext& context) const {
  bool changed = false;

  for (Block* block : GetBlocksWithFunctionProvenance(package)) {
    XLS_ASSIGN_OR_RETURN(Function * f,
                         package->GetFunction(block->GetProvenance()->name));

    VLOG(3) << "Converting function to a combinationalblock:";
    XLS_VLOG_LINES(3, f->DumpIr());

    // A map from the nodes in 'f' to their corresponding node in the block.
    absl::flat_hash_map<Node*, Node*> nodes_function2block;

    // Emit the parameters first to ensure their order is preserved in the
    // block.
    auto func_interface = FindFunctionInterface(
        options.codegen_options.package_interface(), f->name());

    for (Param* param : f->params()) {
      XLS_ASSIGN_OR_RETURN(nodes_function2block[param],
                           block->AddInputPort(param->GetName(),
                                               param->GetType(), param->loc()));

      if (func_interface) {
        auto name =
            absl::c_find_if(func_interface->parameters(),
                            [&](const PackageInterfaceProto::NamedValue& p) {
                              return p.name() == param->name();
                            });
        if (name != func_interface->parameters().end() && name->has_sv_type()) {
          nodes_function2block[param]->As<InputPort>()->set_system_verilog_type(
              name->sv_type());
        }
      }
    }

    for (Node* node : TopoSort(f)) {
      if (node->Is<Param>()) {
        continue;
      }

      std::vector<Node*> new_operands;
      for (Node* operand : node->operands()) {
        new_operands.push_back(nodes_function2block.at(operand));
      }
      XLS_ASSIGN_OR_RETURN(Node * block_node,
                           node->CloneInNewFunction(new_operands, block));
      nodes_function2block[node] = block_node;
    }

    XLS_ASSIGN_OR_RETURN(
        OutputPort * output,
        block->AddOutputPort(options.codegen_options.output_port_name(),
                             nodes_function2block.at(f->return_value())));
    if (func_interface && func_interface->has_sv_result_type()) {
      output->set_system_verilog_type(func_interface->sv_result_type());
    }

    // Create empty metadata object for the block.
    context.SetMetadataForBlock(block, CodegenMetadata());

    changed = true;
  }

  if (changed) {
    context.GcMetadata();
  }
  return changed;
}

absl::StatusOr<bool> ConvertProcsToCombinationalBlocksPass::RunInternal(
    Package* package, const CodegenPassOptions& options, PassResults* results,
    CodegenContext& context) const {
  bool changed = false;

  for (Block* block : GetBlocksWithProcProvenance(package)) {
    XLS_ASSIGN_OR_RETURN(Proc * proc,
                         package->GetProc(block->GetProvenance()->name));

    VLOG(3) << "Converting proc to a pipelined block:";
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

    XLS_ASSIGN_OR_RETURN(
        StreamingIOPipeline streaming_io,
        CloneProcNodesIntoBlock(proc, options.codegen_options, block));

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
        options.codegen_options, proc, block));

    // TODO(tedhong): 2021-09-23 Remove and add any missing functionality to
    //                codegen pipeline.
    context.GetMetadataForBlock(block) = CodegenMetadata{
        .streaming_io_and_pipeline = std::move(streaming_io),
        .concurrent_stages = std::nullopt,
    };

    changed = true;
  }

  if (changed) {
    context.GcMetadata();
  }
  return changed;
}

absl::StatusOr<bool> ConvertFuncsToPipelinedBlocksPass::RunInternal(
    Package* package, const CodegenPassOptions& options, PassResults* results,
    CodegenContext& context) const {
  bool changed = false;

  for (Block* block : GetBlocksWithFunctionProvenance(package)) {
    XLS_ASSIGN_OR_RETURN(Function * f,
                         package->GetFunction(block->GetProvenance()->name));
    if (options.codegen_options.manual_control().has_value()) {
      return absl::UnimplementedError(
          "Manual pipeline control not implemented");
    }
    if (options.codegen_options.split_outputs()) {
      return absl::UnimplementedError("Splitting outputs not supported.");
    }

    VLOG(3) << "Converting function to a pipelined block:";
    XLS_VLOG_LINES(3, f->DumpIr());

    PipelineSchedule& schedule = context.function_base_to_schedule().at(f);
    XLS_RETURN_IF_ERROR(SingleFunctionToPipelinedBlock(
        schedule, options.codegen_options, context, f, block));

    changed = true;
  }

  if (changed) {
    context.GcMetadata();
  }
  return changed;
}

absl::StatusOr<bool> ConvertProcsToPipelinedBlocksPass::RunInternal(
    Package* package, const CodegenPassOptions& options, PassResults* results,
    CodegenContext& context) const {
  bool changed = false;

  bool top_is_proc =
      package->GetTop().has_value() && package->GetTop().value()->IsProc();
  std::vector<Proc*> procs_to_convert;
  std::optional<ProcElaboration> proc_elab;
  if (!package->ChannelsAreProcScoped() && top_is_proc) {
    for (const std::unique_ptr<Block>& block : package->blocks()) {
      if (!block->GetProvenance().has_value() ||
          block->GetProvenance()->kind != BlockProvenanceKind::kProc) {
        continue;
      }
      XLS_ASSIGN_OR_RETURN(Proc * proc,
                           package->GetProc(block->GetProvenance()->name));
      procs_to_convert.push_back(proc);
    }
  }
  if (package->ChannelsAreProcScoped() && top_is_proc) {
    XLS_ASSIGN_OR_RETURN(proc_elab, ProcElaboration::Elaborate(
                                        (*package->GetTop())->AsProcOrDie()));
  }
  XLS_ASSIGN_OR_RETURN(
      std::vector<FunctionBase*> conversion_order,
      GetBlockConversionOrder(package, procs_to_convert, proc_elab));

  absl::flat_hash_map<FunctionBase*, Block*> block_map;
  XLS_ASSIGN_OR_RETURN(block_map, GetBlockProvenanceMap(package));

  absl::flat_hash_map<FunctionBase*, Block*> converted_blocks;
  for (FunctionBase* fb : conversion_order) {
    if (!fb->IsProc()) {
      continue;
    }
    Block* block = block_map.at(fb);
    if (options.codegen_options.manual_control().has_value()) {
      return absl::UnimplementedError(
          "Manual pipeline control not implemented");
    }
    if (options.codegen_options.split_outputs()) {
      return absl::UnimplementedError("Splitting outputs not supported.");
    }

    Proc* const proc = fb->AsProcOrDie();

    VLOG(3) << "Converting proc to a pipelined block:";
    XLS_VLOG_LINES(3, proc->DumpIr());

    PipelineSchedule& schedule = context.function_base_to_schedule().at(fb);
    PackageSchedule package_schedule(schedule);
    absl::Span<ProcInstance* const> instances;
    if (proc_elab.has_value()) {
      instances = proc_elab->GetInstances(proc);
    }
    XLS_RETURN_IF_ERROR(SingleProcToPipelinedBlock(
        package_schedule, options.codegen_options, context, proc, instances,
        block, converted_blocks,
        proc_elab.has_value() ? std::make_optional(&proc_elab.value())
                              : std::nullopt));

    converted_blocks[fb] = block;
    changed = true;
  }

  if (changed) {
    context.GcMetadata();
  }
  return changed;
}

}  // namespace xls::verilog
