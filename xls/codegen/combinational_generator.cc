// Copyright 2020 The XLS Authors
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

#include "xls/codegen/combinational_generator.h"

#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"
#include "xls/codegen/flattening.h"
#include "xls/codegen/module_builder.h"
#include "xls/codegen/node_expressions.h"
#include "xls/common/logging/log_lines.h"
#include "xls/common/logging/logging.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/dfs_visitor.h"
#include "xls/ir/node.h"
#include "xls/ir/node_iterator.h"

namespace xls {
namespace verilog {

absl::StatusOr<ModuleGeneratorResult> ToCombinationalModuleText(
    Function* func, bool use_system_verilog) {
  XLS_VLOG(2) << "Generating combinational module for function:";
  XLS_VLOG_LINES(2, func->DumpIr());

  VerilogFile f;
  ModuleBuilder mb(func->name(), &f, /*use_system_verilog=*/use_system_verilog);

  // Build the module signature.
  ModuleSignatureBuilder sig_builder(mb.module()->name());
  for (Param* param : func->params()) {
    sig_builder.AddDataInput(param->name(),
                             param->GetType()->GetFlatBitCount());
  }
  const int64 output_width = func->return_value()->GetType()->GetFlatBitCount();
  // Don't use the assigned name if this is a parameter or there will be ports
  // with duplicate names.
  const char kOutputPortName[] = "out";
  sig_builder.AddDataOutput(kOutputPortName, output_width);
  sig_builder.WithFunctionType(func->GetType());
  sig_builder.WithCombinationalInterface();
  XLS_ASSIGN_OR_RETURN(ModuleSignature signature, sig_builder.Build());

  // Map from Node* to the Verilog expression representing its value.
  absl::flat_hash_map<Node*, Expression*> node_exprs;

  // Add parameters explicitly so the input ports are added in the order they
  // appear in the parameters of the function.
  for (Param* param : func->params()) {
    if (param->GetType()->GetFlatBitCount() == 0) {
      XLS_RET_CHECK_EQ(param->users().size(), 0);
      continue;
    }
    XLS_ASSIGN_OR_RETURN(
        node_exprs[param],
        mb.AddInputPort(param->As<Param>()->name(), param->GetType()));
  }

  for (Node* node : TopoSort(func)) {
    if (node->Is<Param>()) {
      // Parameters are added in the above loop.
      continue;
    }

    // Verilog has no zero-bit data types so elide such types. They should have
    // no uses.
    if (node->GetType()->GetFlatBitCount() == 0) {
      XLS_RET_CHECK_EQ(node->users().size(), 0);
      continue;
    }

    // Emit non-bits-typed literals as module-level constants because in general
    // these complicated types cannot be handled inline, and constructing them
    // in Verilog may require a sequence of assignments.
    if (node->Is<xls::Literal>() && !node->GetType()->IsBits()) {
      XLS_ASSIGN_OR_RETURN(
          node_exprs[node],
          mb.DeclareModuleConstant(node->GetName(),
                                   node->As<xls::Literal>()->value()));
      continue;
    }

    std::vector<Expression*> inputs;
    for (Node* operand : node->operands()) {
      inputs.push_back(node_exprs.at(operand));
    }
    // If the node has an assigned name then don't emit as an inline
    // expression. This ensures the name appears in the generated Verilog.
    if (node->HasAssignedName() || node->users().size() > 1 ||
        node == func->return_value() || !mb.CanEmitAsInlineExpression(node)) {
      XLS_ASSIGN_OR_RETURN(node_exprs[node],
                           mb.EmitAsAssignment(node->GetName(), node, inputs));
    } else {
      XLS_ASSIGN_OR_RETURN(node_exprs[node],
                           mb.EmitAsInlineExpression(node, inputs));
    }
  }

  // Skip adding an output port to the Verilog module if the output is
  // zero-width.
  if (output_width > 0) {
    XLS_RETURN_IF_ERROR(mb.AddOutputPort(kOutputPortName,
                                         func->return_value()->GetType(),
                                         node_exprs.at(func->return_value())));
  }
  std::string text = f.Emit();

  XLS_VLOG(2) << "Verilog output:";
  XLS_VLOG_LINES(2, text);

  return ModuleGeneratorResult{text, signature};
}

}  // namespace verilog
}  // namespace xls
