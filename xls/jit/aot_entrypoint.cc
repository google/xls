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

#include "xls/jit/aot_entrypoint.h"

#include "xls/common/status/ret_check.h"
#include "xls/dev_tools/extract_interface.h"
#include "xls/ir/function.h"
#include "xls/ir/function_base.h"
#include "xls/ir/nodes.h"
#include "xls/ir/package.h"
#include "xls/jit/function_base_jit.h"
#include "xls/jit/llvm_type_converter.h"

namespace xls {

absl::StatusOr<AotEntrypointProto> GenerateAotEntrypointProto(
    Package* package, const FunctionEntrypoint& entrypoint, bool include_msan,
    LlvmTypeConverter& type_converter) {
  FunctionBase* func = entrypoint.function;
  const JittedFunctionBase& object_code = entrypoint.jit_info;
  AotEntrypointProto proto;
  proto.set_has_msan(include_msan);
  if (func->IsFunction()) {
    proto.set_type(AotEntrypointProto::FUNCTION);
    proto.add_outputs_names("result");
    for (const Param* p : func->params()) {
      proto.add_inputs_names(p->name());
      *proto.mutable_inputs_layout()->add_layouts() =
          type_converter.CreateTypeLayout(p->GetType()).ToProto();
    }
    *proto.mutable_outputs_layout()->add_layouts() =
        type_converter
            .CreateTypeLayout(func->AsFunctionOrDie()->GetType()->return_type())
            .ToProto();
    AotEntrypointProto::FunctionMetadataProto* function_metadata_proto =
        proto.mutable_function_metadata();
    *function_metadata_proto->mutable_function_interface() =
        ExtractFunctionInterface(func->AsFunctionOrDie());
  } else if (func->IsProc()) {
    proto.set_type(AotEntrypointProto::PROC);
    for (const Param* p : func->params()) {
      proto.add_inputs_names(p->name());
      proto.add_outputs_names(p->name());
      auto layout_proto =
          type_converter.CreateTypeLayout(p->GetType()).ToProto();
      *proto.mutable_inputs_layout()->add_layouts() = layout_proto;
      *proto.mutable_outputs_layout()->add_layouts() = layout_proto;
    }
    AotEntrypointProto::ProcMetadataProto* proc_metadata_proto =
        proto.mutable_proc_metadata();
    proc_metadata_proto->mutable_continuation_point_node_ids()->insert(
        object_code.continuation_points().begin(),
        object_code.continuation_points().end());
    proc_metadata_proto->mutable_channel_queue_indices()->insert(
        object_code.queue_indices().begin(), object_code.queue_indices().end());
    *proc_metadata_proto->mutable_proc_interface() =
        ExtractProcInterface(func->AsProcOrDie());
  } else {
    XLS_RET_CHECK(func->IsBlock());
    proto.set_type(AotEntrypointProto::BLOCK);
    for (InputPort* p : func->AsBlockOrDie()->GetInputPorts()) {
      proto.add_inputs_names(p->name());
      auto layout_proto =
          type_converter.CreateTypeLayout(p->GetType()).ToProto();
      *proto.mutable_inputs_layout()->add_layouts() = layout_proto;
    }
    for (OutputPort* p : func->AsBlockOrDie()->GetOutputPorts()) {
      proto.add_outputs_names(p->name());
      auto layout_proto =
          type_converter.CreateTypeLayout(p->GetType()).ToProto();
      *proto.mutable_outputs_layout()->add_layouts() = layout_proto;
    }
    AotEntrypointProto::BlockMetadataProto* block_metadata_proto =
        proto.mutable_block_metadata();

    for (const auto& [orig, translated] : entrypoint.register_aliases) {
      block_metadata_proto->mutable_register_aliases()->insert(
          {orig, translated});
    }
    for (const auto& [reg, ty] : entrypoint.added_registers) {
      block_metadata_proto->mutable_added_registers()->insert(
          {reg, ty->ToProto()});
    }
    *block_metadata_proto->mutable_block_interface() =
        ExtractBlockInterface(func->AsBlockOrDie());
  }

  proto.set_xls_package_name(package->name());
  proto.set_xls_function_identifier(func->name());
  proto.set_function_symbol(object_code.function_name());
  if (object_code.HasPackedFunction()) {
    proto.set_packed_function_symbol(*object_code.packed_function_name());
  }
  for (const auto& metadata : object_code.GetInputBufferMetadata()) {
    proto.add_input_buffer_sizes(metadata.size);
    proto.add_input_buffer_alignments(metadata.preferred_alignment);
    proto.add_input_buffer_abi_alignments(metadata.abi_alignment);
    if (object_code.HasPackedFunction()) {
      proto.add_packed_input_buffer_sizes(metadata.packed_size);
    }
  }
  for (const auto& metadata : object_code.GetOutputBufferMetadata()) {
    proto.add_output_buffer_sizes(metadata.size);
    proto.add_output_buffer_alignments(metadata.preferred_alignment);
    proto.add_output_buffer_abi_alignments(metadata.abi_alignment);
    if (object_code.HasPackedFunction()) {
      proto.add_packed_output_buffer_sizes(metadata.packed_size);
    }
  }

  proto.set_temp_buffer_size(object_code.temp_buffer_size());
  proto.set_temp_buffer_alignment(object_code.temp_buffer_alignment());
  return proto;
}

}  // namespace xls
