// Copyright 2023 The XLS Authors
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

#include "xls/codegen/ffi_instantiation_pass.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "xls/codegen/codegen_pass.h"
#include "xls/codegen/vast.h"
#include "xls/common/casts.h"
#include "xls/common/logging/logging.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/function.h"
#include "xls/ir/instantiation.h"
#include "xls/ir/node.h"
#include "xls/ir/nodes.h"
#include "xls/ir/type.h"
#include "xls/passes/pass_base.h"

namespace xls::verilog {

static absl::Status InvocationParamsToInstInputs(
    Block* block, Invoke* invocation, Function* fun,
    xls::Instantiation* instantiation) {
  // The names in the IR function
  const absl::Span<Param* const> fun_params = fun->params();

  // The IR expression nodes they are bound to.
  const absl::Span<Node* const> target_operands = invocation->operands();

  XLS_CHECK_EQ(fun_params.size(), target_operands.size());

  // Creating InstantiationInput and Output in block will also wire them up.
  for (int i = 0; i < fun_params.size(); ++i) {
    Node* const operand = target_operands[i];
    XLS_RETURN_IF_ERROR(
        block
            ->MakeNode<InstantiationInput>(invocation->loc(), operand,
                                           instantiation, fun_params[i]->name())
            .status());
  }

  return absl::OkStatus();
}

static absl::Status InvocationReturnToInstOutputs(
    Block* block, Invoke* invocation, Function* fun,
    xls::Instantiation* instantiation) {
  switch (invocation->GetType()->kind()) {
    case TypeKind::kBits:
      XLS_RETURN_IF_ERROR(
          invocation
              ->ReplaceUsesWithNew<InstantiationOutput>(instantiation, "return")
              .status());
      break;
    case TypeKind::kTuple: {
      // A tuple return requires multiple outputs that are mapped to
      // return.0, return.1 ... names in the FFI template.
      // TODO(hzeller): 2023-06-28 for nested tuples, build this recursively.
      TupleType* const node_type = invocation->GetType()->AsTupleOrDie();
      std::vector<Node*> inst_output_tuple_nodes;
      for (int64_t i = 0; i < node_type->size(); ++i) {
        XLS_ASSIGN_OR_RETURN(Node * tuple_element,
                             invocation->function_base()->MakeNode<TupleIndex>(
                                 invocation->loc(), invocation, i));
        XLS_ASSIGN_OR_RETURN(
            Node * output_node,
            tuple_element->ReplaceUsesWithNew<InstantiationOutput>(
                instantiation, absl::StrCat("return.", i)));
        inst_output_tuple_nodes.push_back(output_node);
      }

      // The original invocation becomes a tuple of InstantiationOutputs
      XLS_RETURN_IF_ERROR(
          invocation->ReplaceUsesWithNew<Tuple>(inst_output_tuple_nodes)
              .status());
      break;
    }
    default:
      return absl::UnimplementedError(
          absl::StrFormat("Can't deal with FFI return type '%s' yet",
                          invocation->GetType()->ToString()));
  }
  return absl::OkStatus();
}

absl::StatusOr<bool> FfiInstantiationPass::RunInternal(
    CodegenPassUnit* unit, const CodegenPassOptions& options,
    PassResults* results) const {
  Block* const block = unit->block;
  std::vector<Node*> to_remove;
  for (Node* node : block->nodes()) {
    if (!node->Is<Invoke>()) {
      continue;
    }
    Invoke* const invocation = node->As<Invoke>();
    Function* const fun = down_cast<Function*>(invocation->to_apply());
    if (!fun || !fun->ForeignFunctionData().has_value()) {
      return absl::InternalError(absl::StrCat("Only FFI invocations expected; ",
                                              invocation->to_apply()->name(),
                                              " is not."));
    }

    // TODO(hzeller): Better ways to generate a name ?
    const std::string inst_name = SanitizeIdentifier(
        absl::StrCat(fun->name(), "_", invocation->GetName(), "_inst"));
    XLS_ASSIGN_OR_RETURN(
        xls::Instantiation * instantiation,
        block->AddInstantiation(
            inst_name, std::make_unique<ExternInstantiation>(inst_name, fun)));

    // Params and returns of the invocation become instantiation inputs/outputs.
    XLS_RETURN_IF_ERROR(
        InvocationParamsToInstInputs(block, invocation, fun, instantiation));
    XLS_RETURN_IF_ERROR(
        InvocationReturnToInstOutputs(block, invocation, fun, instantiation));

    to_remove.push_back(invocation);
  }

  for (Node* n : to_remove) {
    XLS_RETURN_IF_ERROR(block->RemoveNode(n));
  }

  return !to_remove.empty();
}

}  // namespace xls::verilog
