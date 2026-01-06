// Copyright 2026 The XLS Authors
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

#include "xls/codegen_v_1_5/ffi_instantiation_pass.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "cppitertools/zip.hpp"
#include "xls/codegen/vast/vast.h"
#include "xls/codegen_v_1_5/block_conversion_pass.h"
#include "xls/common/casts.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/block.h"
#include "xls/ir/code_template.h"
#include "xls/ir/function.h"
#include "xls/ir/instantiation.h"
#include "xls/ir/node.h"
#include "xls/ir/nodes.h"
#include "xls/ir/package.h"
#include "xls/ir/type.h"
#include "xls/passes/pass_base.h"

namespace xls::codegen {
namespace {

// Set of return* names extracted from template. Used in a couple of places
// below, so give it a name.
using NameSet = absl::flat_hash_set<std::string>;

template <typename NodeT, typename... Args>
  requires(std::is_base_of_v<Node, NodeT>)
absl::StatusOr<Node*> MakeNodeInStage(ScheduledBlock* block,
                                      std::optional<int64_t> stage_index,
                                      Args&&... args) {
  if (stage_index.has_value()) {
    return block->MakeNodeInStage<NodeT>(*stage_index,
                                         std::forward<Args>(args)...);
  } else {
    return block->MakeNode<NodeT>(std::forward<Args>(args)...);
  }
}

absl::Status MakeInstantiationInputs(ScheduledBlock* block,
                                     std::optional<int64_t> stage_index,
                                     Node* node,
                                     xls::Instantiation* instantiation,
                                     std::string_view base_name) {
  XLS_RETURN_IF_ERROR(
      MakeNodeInStage<InstantiationInput>(block, stage_index, node->loc(), node,
                                          instantiation, base_name)
          .status());
  // The user can access tuple elements in the template by index, so provide
  // recursively-expanded names here.
  if (node->GetType()->kind() == TypeKind::kTuple) {
    TupleType* const tuple_type = node->GetType()->AsTupleOrDie();
    for (int64_t i = 0; i < tuple_type->size(); ++i) {
      XLS_ASSIGN_OR_RETURN(Node * subnode,
                           MakeNodeInStage<TupleIndex>(block, stage_index,
                                                       node->loc(), node, i));
      XLS_RETURN_IF_ERROR(
          MakeInstantiationInputs(block, stage_index, subnode, instantiation,
                                  absl::StrCat(base_name, ".", i)));
    }
  }
  return absl::OkStatus();
}

absl::Status InvocationParamsToInstInputs(ScheduledBlock* block,
                                          std::optional<int64_t> stage_index,
                                          Invoke* invocation, Function* fun,
                                          xls::Instantiation* instantiation) {
  // We should have as many params in the IR function as nodes they are bound
  // to.
  CHECK_EQ(fun->params().size(), invocation->operands().size());

  // Creating InstantiationInput and Output in block will also wire them up.
  for (auto [param, operand] :
       iter::zip(fun->params(), invocation->operands())) {
    XLS_RETURN_IF_ERROR(MakeInstantiationInputs(block, stage_index, operand,
                                                instantiation, param->name()));
  }

  return absl::OkStatus();
}

// Create InstantiationOutputs from IR nodes, providing direct access to
// returned values if requested, and also extracting from tuples as needed.
absl::StatusOr<Node*> BuildInstantiationOutput(
    std::string_view prefix, ScheduledBlock* block,
    std::optional<int64_t> stage_index, xls::Instantiation* instantiation,
    Node* node, const NameSet& requested_names) {
  switch (node->GetType()->kind()) {
    case TypeKind::kBits:
      return MakeNodeInStage<InstantiationOutput>(
          block, stage_index, node->loc(), instantiation, prefix);
    case TypeKind::kTuple: {
      if (requested_names.contains(prefix)) {
        // If the template requests this tuple, we're done.
        return MakeNodeInStage<InstantiationOutput>(
            block, stage_index, node->loc(), instantiation, prefix);
      }
      TupleType* const tuple_type = node->GetType()->AsTupleOrDie();
      std::vector<Node*> inst_output_tuple_nodes;
      for (int64_t i = 0; i < tuple_type->size(); ++i) {
        XLS_ASSIGN_OR_RETURN(Node * subnode,
                             MakeNodeInStage<TupleIndex>(block, stage_index,
                                                         node->loc(), node, i));
        XLS_ASSIGN_OR_RETURN(
            Node * output_node,
            BuildInstantiationOutput(absl::StrCat(prefix, ".", i), block,
                                     stage_index, instantiation, subnode,
                                     requested_names));
        inst_output_tuple_nodes.push_back(output_node);
      }
      return MakeNodeInStage<Tuple>(block, stage_index, node->loc(),
                                    inst_output_tuple_nodes);
    }
    default:
      return absl::UnimplementedError(
          absl::StrFormat("Can't deal with FFI return type '%s' yet",
                          node->GetType()->ToString()));
  }
  return absl::OkStatus();
}

absl::StatusOr<Node*> InvocationReturnToInstOutputs(
    ScheduledBlock* node_factory, std::optional<int64_t> stage_index,
    Invoke* invocation, xls::Instantiation* instantiation,
    const NameSet& requested_names) {
  XLS_ASSIGN_OR_RETURN(
      Node * tuple_or_scalar,
      BuildInstantiationOutput("return", node_factory, stage_index,
                               instantiation, invocation, requested_names));
  XLS_RETURN_IF_ERROR(invocation->ReplaceUsesWith(tuple_or_scalar));
  return tuple_or_scalar;
}

// Extract all the template parameters that refer to return values.
absl::StatusOr<NameSet> ExtractReturnNames(std::string_view tpl) {
  XLS_ASSIGN_OR_RETURN(CodeTemplate code_template, CodeTemplate::Create(tpl));
  NameSet result;
  for (const std::string& name : code_template.Expressions()) {
    if (name == "return" || name.starts_with("return.")) {
      result.emplace(name);
    }
  }
  return result;
}
}  // namespace

// Public interface
absl::StatusOr<bool> FfiInstantiationPass::RunInternal(
    Package* package, const BlockConversionPassOptions& options,
    PassResults* results) const {
  bool changed = false;
  std::vector<Node*> to_remove;
  for (const std::unique_ptr<Block>& block : package->blocks()) {
    if (!block->IsScheduled()) {
      continue;
    }
    ScheduledBlock* scheduled_block = down_cast<ScheduledBlock*>(block.get());
    for (Node* node : block->nodes()) {
      if (!node->Is<Invoke>()) {
        continue;
      }
      Invoke* const invocation = node->As<Invoke>();
      Function* const fun = down_cast<Function*>(invocation->to_apply());
      if (!fun || !fun->ForeignFunctionData().has_value()) {
        return absl::InternalError(absl::StrFormat(
            "Detected non-FFI function call in IR; `%s` has no FFI data. "
            "Probable cause: IR was not run through optimizer (opt_main).",
            invocation->to_apply()->name()));
      }

      // TODO(hzeller): Better ways to generate a name ?
      const std::string inst_name = verilog::SanitizeVerilogIdentifier(
          absl::StrCat(fun->name(), "_", invocation->GetName(), "_inst"));
      XLS_ASSIGN_OR_RETURN(xls::Instantiation * instantiation,
                           block->AddInstantiation(
                               inst_name, std::make_unique<ExternInstantiation>(
                                              inst_name, fun)));

      XLS_ASSIGN_OR_RETURN(
          NameSet return_names,
          ExtractReturnNames(fun->ForeignFunctionData()->code_template()));

      std::optional<int64_t> stage_index =
          scheduled_block->IsStaged(invocation)
              ? std::make_optional(*scheduled_block->GetStageIndex(invocation))
              : std::nullopt;

      // Params and returns of the invocation become instantiation
      // inputs/outputs.
      XLS_RETURN_IF_ERROR(InvocationParamsToInstInputs(
          scheduled_block, stage_index, invocation, fun, instantiation));
      XLS_ASSIGN_OR_RETURN(Node * replacement,
                           InvocationReturnToInstOutputs(
                               scheduled_block, stage_index, invocation,
                               instantiation, return_names));

      if (invocation == scheduled_block->source_return_value()) {
        scheduled_block->SetSourceReturnValue(replacement);
      }
      to_remove.push_back(invocation);
    }

    for (Node* n : to_remove) {
      XLS_ASSIGN_OR_RETURN(bool removed_from_stage,
                           scheduled_block->RemoveNodeFromStage(n));
      CHECK(removed_from_stage);
      XLS_RETURN_IF_ERROR(block->RemoveNode(n));
      changed = true;
    }
  }

  return changed;
}

}  // namespace xls::codegen
