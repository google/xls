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

#include "xls/codegen/signature_generation_pass.h"

#include "absl/base/casts.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "xls/codegen/codegen_pass.h"
#include "xls/codegen/module_signature.h"
#include "xls/codegen/module_signature.pb.h"
#include "xls/codegen/signature_generator.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/block.h"
#include "xls/ir/instantiation.h"
#include "xls/ir/package.h"
#include "xls/passes/pass_base.h"

namespace xls::verilog {
namespace {

// Returns `block`'s signature with `block_signature` embedded recursively.
ModuleSignatureProto ResolveEmbeddedSignature(Block* block) {
  ModuleSignatureProto proto = *block->GetSignature();
  for (InstantiationProto& instantiation : *proto.mutable_instantiations()) {
    if (!instantiation.has_block_instantiation()) {
      continue;
    }
    BlockInstantiationProto* block_instantiation =
        instantiation.mutable_block_instantiation();
    for (const ::xls::Instantiation* child_instantiation :
         block->GetInstantiations()) {
      if (child_instantiation->kind() != ::xls::InstantiationKind::kBlock ||
          child_instantiation->name() != block_instantiation->instance_name()) {
        continue;
      }
      Block* child_block =
          absl::down_cast<const BlockInstantiation*>(child_instantiation)
              ->instantiated_block();
      if (child_block->GetSignature().has_value()) {
        *block_instantiation->mutable_block_signature() =
            ResolveEmbeddedSignature(child_block);
      }
      break;
    }
  }
  return proto;
}

}  // namespace

absl::StatusOr<bool> SignatureGenerationPass::RunInternal(
    Package* package, const CodegenPassOptions& options, PassResults* results,
    CodegenContext& context) const {
  bool changed = false;
  VLOG(3) << absl::StreamFormat("Metadata has %d blocks",
                                context.metadata().size());
  for (auto& [block, metadata] : context.metadata()) {
    if (block->GetSignature().has_value()) {
      return absl::InvalidArgumentError("Signature already generated.");
    }
    XLS_ASSIGN_OR_RETURN(
        ModuleSignature signature,
        GenerateSignature(
            options.codegen_options, block,
            metadata.streaming_io_and_pipeline.node_to_stage_map));
    block->SetSignature(signature.proto());
    changed = true;
  }
  // All blocks now have their own signature; embed children recursively.
  for (auto& [block, metadata] : context.metadata()) {
    block->SetSignature(ResolveEmbeddedSignature(block));
  }
  return changed;
}

}  // namespace xls::verilog
