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
#include <string_view>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "xls/codegen/codegen_pass.h"
#include "xls/codegen/vast/vast.h"
#include "xls/common/casts.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/code_template.h"
#include "xls/ir/function.h"
#include "xls/ir/instantiation.h"
#include "xls/ir/node.h"
#include "xls/ir/nodes.h"
#include "xls/ir/package.h"
#include "xls/ir/type.h"
#include "xls/passes/pass_base.h"

namespace xls::verilog {
namespace {
// Set of return* names extracted from template. Used in a couple of places
// below, so give it a name.
using NameSet = absl::flat_hash_set<std::string>;

static absl::Status MakeInstantiationInputs(Block* block, Node* node,
                                            xls::Instantiation* instantiation,
                                            std::string_view base_name) {
  XLS_RETURN_IF_ERROR(block
                          ->MakeNode<InstantiationInput>(
                              node->loc(), node, instantiation, base_name)
                          .status());
  // The user can access tuple-indexes in the template, so provide expanded
  // names a.0 here.
  if (node->GetType()->kind() == TypeKind::kTuple) {
    TupleType* const tuple_type = node->GetType()->AsTupleOrDie();
    for (int64_t i = 0; i < tuple_type->size(); ++i) {
      XLS_ASSIGN_OR_RETURN(Node * subnode,
                           block->MakeNode<TupleIndex>(node->loc(), node, i));
      XLS_RETURN_IF_ERROR(MakeInstantiationInputs(
          block, subnode, instantiation, absl::StrCat(base_name, ".", i)));
    }
  }
  return absl::OkStatus();
}

static absl::Status InvocationParamsToInstInputs(
    Block* block, Invoke* invocation, Function* fun,
    xls::Instantiation* instantiation) {
  // The names in the IR function
  const absl::Span<Param* const> fun_params = fun->params();

  // The IR expression nodes they are bound to.
  const absl::Span<Node* const> target_operands = invocation->operands();

  CHECK_EQ(fun_params.size(), target_operands.size());

  // Creating InstantiationInput and Output in block will also wire them up.
  for (int i = 0; i < fun_params.size(); ++i) {
    Node* const operand = target_operands[i];
    const std::string_view base_name = fun_params[i]->name();
    XLS_RETURN_IF_ERROR(
        MakeInstantiationInputs(block, operand, instantiation, base_name));
  }

  return absl::OkStatus();
}

// Create InstantiationOutputs from IR nodes, providing direct accesss to scalar
// values, if needed all the way unnested from tuples, but also provding
// access to entire tuples if so requestd.
static absl::StatusOr<Node*> BuildInstantiationOutput(
    std::string_view prefix, FunctionBase* node_factory,
    xls::Instantiation* instantiation, Node* node,
    const NameSet& requested_names) {
  switch (node->GetType()->kind()) {
    case TypeKind::kBits:
      return node->ReplaceUsesWithNew<InstantiationOutput>(instantiation,
                                                           prefix);
    case TypeKind::kTuple: {
      if (requested_names.contains(prefix)) {
        // If the template requests this tuple, we're done.
        return node->ReplaceUsesWithNew<InstantiationOutput>(instantiation,
                                                             prefix);
      }
      TupleType* const tuple_type = node->GetType()->AsTupleOrDie();
      std::vector<Node*> inst_output_tuple_nodes;
      for (int64_t i = 0; i < tuple_type->size(); ++i) {
        XLS_ASSIGN_OR_RETURN(Node * subnode, node_factory->MakeNode<TupleIndex>(
                                                 node->loc(), node, i));
        XLS_ASSIGN_OR_RETURN(
            Node * output_node,
            BuildInstantiationOutput(absl::StrCat(prefix, ".", i), node_factory,
                                     instantiation, subnode, requested_names));
        inst_output_tuple_nodes.push_back(output_node);
      }
      return node->ReplaceUsesWithNew<Tuple>(inst_output_tuple_nodes);
    }
    default:
      return absl::UnimplementedError(
          absl::StrFormat("Can't deal with FFI return type '%s' yet",
                          node->GetType()->ToString()));
  }
  return absl::OkStatus();
}

static absl::Status InvocationReturnToInstOutputs(
    FunctionBase* node_factory, Invoke* invocation,
    xls::Instantiation* instantiation, const NameSet& requested_names) {
  XLS_ASSIGN_OR_RETURN(
      Node * tuple_or_scalar,
      BuildInstantiationOutput("return", node_factory, instantiation,
                               invocation, requested_names));
  return invocation->ReplaceUsesWith(tuple_or_scalar);
}

// Extract all the template parameters that refer to return values.
static absl::StatusOr<NameSet> ExtractReturnNames(std::string_view tpl) {
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
    Package* package, const CodegenPassOptions& options, PassResults* results,
    CodegenContext& context) const {
  std::vector<Node*> to_remove;
  for (const std::unique_ptr<Block>& block : package->blocks()) {
    for (Node* node : block->nodes()) {
      if (!node->Is<Invoke>()) {
        continue;
      }
      Invoke* const invocation = node->As<Invoke>();
      Function* const fun = down_cast<Function*>(invocation->to_apply());
      if (!fun || !fun->ForeignFunctionData().has_value()) {
        return absl::InternalError(
            absl::StrCat("Detected function call in IR. Probable cause: IR was "
                         "not run through optimizer (opt_main) "
                         "(Only FFI invocations are allowed here; ",
                         invocation->to_apply()->name(), " is not)."));
      }

      // TODO(hzeller): Better ways to generate a name ?
      const std::string inst_name = SanitizeVerilogIdentifier(
          absl::StrCat(fun->name(), "_", invocation->GetName(), "_inst"));
      XLS_ASSIGN_OR_RETURN(xls::Instantiation * instantiation,
                           block->AddInstantiation(
                               inst_name, std::make_unique<ExternInstantiation>(
                                              inst_name, fun)));

      XLS_ASSIGN_OR_RETURN(
          NameSet return_names,
          ExtractReturnNames(fun->ForeignFunctionData()->code_template()));
      // Params and returns of the invocation become instantiation
      // inputs/outputs.
      XLS_RETURN_IF_ERROR(InvocationParamsToInstInputs(block.get(), invocation,
                                                       fun, instantiation));
      XLS_RETURN_IF_ERROR(InvocationReturnToInstOutputs(
          block.get(), invocation, instantiation, return_names));

      to_remove.push_back(invocation);
    }

    for (Node* n : to_remove) {
      XLS_RETURN_IF_ERROR(block->RemoveNode(n));
    }
  }
  if (!to_remove.empty()) {
    context.GcMetadata();
  }

  return !to_remove.empty();
}

}  // namespace xls::verilog
