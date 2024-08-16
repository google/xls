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
#include "xls/dslx/ir_convert/proc_config_ir_converter.h"

#include <optional>
#include <variant>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/meta/type_traits.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/dslx/constexpr_evaluator.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/import_data.h"
#include "xls/dslx/interp_value.h"
#include "xls/dslx/ir_convert/channel_scope.h"
#include "xls/dslx/ir_convert/conversion_info.h"
#include "xls/dslx/ir_convert/extract_conversion_order.h"
#include "xls/dslx/type_system/deduce_utils.h"
#include "xls/dslx/type_system/parametric_env.h"
#include "xls/dslx/type_system/type_info.h"
#include "xls/ir/channel.h"
#include "xls/ir/package.h"
#include "xls/ir/value.h"

namespace xls::dslx {
namespace {

ProcConfigValue ChannelOrArrayToProcConfigValue(
    ChannelOrArray channel_or_array) {
  return std::get<Channel*>(channel_or_array);
}

}  // namespace

ProcConfigIrConverter::ProcConfigIrConverter(
    PackageConversionData* conversion_info, Function* f, TypeInfo* type_info,
    ImportData* import_data, ProcConversionData* proc_data,
    ChannelScope* channel_scope, const ParametricEnv& bindings,
    const ProcId& proc_id)
    : conversion_info_(conversion_info),
      f_(f),
      type_info_(type_info),
      import_data_(import_data),
      proc_data_(proc_data),
      channel_scope_(channel_scope),
      bindings_(bindings),
      proc_id_(proc_id),
      final_tuple_(nullptr) {
  proc_data->id_to_members[proc_id_] = {};
}

absl::Status ProcConfigIrConverter::Finalize() {
  XLS_RET_CHECK(f_->proc().has_value());
  Proc* p = f_->proc().value();
  if (final_tuple_ == nullptr) {
    XLS_RET_CHECK(p->members().empty());
    return absl::OkStatus();
  }

  XLS_RET_CHECK_EQ(p->members().size(), final_tuple_->members().size());
  for (int i = 0; i < p->members().size(); i++) {
    ProcMember* member = p->members()[i];
    proc_data_->id_to_members.at(proc_id_)[member->identifier()] =
        node_to_ir_.at(final_tuple_->members()[i]);
  }

  return absl::OkStatus();
}

absl::Status ProcConfigIrConverter::HandleStatementBlock(
    const StatementBlock* node) {
  VLOG(4) << "ProcConfigIrConverter::HandleStatementBlock: " << node->ToString()
          << " : " << node->span().ToString();
  for (const Statement* statement : node->statements()) {
    XLS_RETURN_IF_ERROR(statement->Accept(this));
  }
  return absl::OkStatus();
}

absl::Status ProcConfigIrConverter::HandleStatement(const Statement* node) {
  return ToAstNode(node->wrapped())->Accept(this);
}

absl::Status ProcConfigIrConverter::HandleChannelDecl(const ChannelDecl* node) {
  VLOG(4) << "ProcConfigIrConverter::HandleChannelDecl: " << node->ToString()
          << " : " << node->span().ToString();
  XLS_ASSIGN_OR_RETURN(ChannelOrArray channel_or_array,
                       channel_scope_->DefineChannelOrArray(node));
  node_to_ir_[node] = ChannelOrArrayToProcConfigValue(channel_or_array);
  return absl::OkStatus();
}

absl::Status ProcConfigIrConverter::HandleColonRef(const ColonRef* node) {
  VLOG(4) << "ProcConfigIrConverter::HandleColonRef: " << node->ToString();
  XLS_ASSIGN_OR_RETURN(InterpValue const_value, type_info_->GetConstExpr(node));
  XLS_ASSIGN_OR_RETURN(auto ir_value, const_value.ConvertToIr());
  node_to_ir_[node] = ir_value;
  return absl::OkStatus();
}

absl::Status ProcConfigIrConverter::HandleFunction(const Function* node) {
  for (Param* p : node->params()) {
    XLS_RETURN_IF_ERROR(p->Accept(this));
  }

  return node->body()->Accept(this);
}

absl::Status ProcConfigIrConverter::HandleInvocation(const Invocation* node) {
  VLOG(4) << "ProcConfigIrConverter::HandleInvocation: " << node->ToString();
  XLS_ASSIGN_OR_RETURN(InterpValue const_value, type_info_->GetConstExpr(node));
  XLS_ASSIGN_OR_RETURN(auto ir_value, const_value.ConvertToIr());
  node_to_ir_[node] = ir_value;
  return absl::OkStatus();
}

absl::Status ProcConfigIrConverter::HandleLet(const Let* node) {
  VLOG(4) << "ProcConfigIrConverter::HandleLet : " << node->ToString();
  XLS_RETURN_IF_ERROR(node->rhs()->Accept(this));

  if (ChannelDecl* decl = dynamic_cast<ChannelDecl*>(node->rhs());
      decl != nullptr) {
    std::vector<NameDefTree::Leaf> leaves = node->name_def_tree()->Flatten();
    CHECK_EQ(leaves.size(), 2);
    for (int i = 0; i < 2; i++) {
      NameDef* name_def = std::get<NameDef*>(leaves[i]);
      XLS_ASSIGN_OR_RETURN(
          ChannelOrArray target,
          channel_scope_->AssociateWithExistingChannelOrArray(name_def, decl));
      node_to_ir_[name_def] = ChannelOrArrayToProcConfigValue(target);
    }
  } else {
    if (!node->name_def_tree()->is_leaf()) {
      return absl::UnimplementedError(
          "Destructuring let bindings are not yet supported in Proc configs.");
    }

    // A leaf on the LHS of a Let will always be a NameDef.
    NameDef* def = std::get<NameDef*>(node->name_def_tree()->leaf());
    if (!node_to_ir_.contains(node->rhs())) {
      return absl::InternalError(
          absl::StrCat("Let RHS not evaluated as constexpr: ", def->ToString(),
                       " : ", node->rhs()->ToString()));
    }
    auto value = node_to_ir_.at(node->rhs());
    node_to_ir_[def] = value;
  }

  return absl::OkStatus();
}

absl::Status ProcConfigIrConverter::HandleNameRef(const NameRef* node) {
  VLOG(4) << "ProcConfigIrConverter::HandleNameRef : " << node->ToString();
  const NameDef* name_def = std::get<const NameDef*>(node->name_def());
  auto iter = node_to_ir_.find(name_def);
  if (iter == node_to_ir_.end()) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Could not find converted value for node \"", node->ToString(), "\"."));
  }
  node_to_ir_[node] = ProcConfigValue(iter->second);
  return absl::OkStatus();
}

absl::Status ProcConfigIrConverter::HandleNumber(const Number* node) {
  XLS_ASSIGN_OR_RETURN(InterpValue const_value, type_info_->GetConstExpr(node));
  XLS_ASSIGN_OR_RETURN(auto ir_value, const_value.ConvertToIr());
  node_to_ir_[node] = ir_value;
  return absl::OkStatus();
}

absl::Status ProcConfigIrConverter::HandleParam(const Param* node) {
  // Matches a param AST node to the actual arg for this Proc instance.
  VLOG(4) << "ProcConfigIrConverter::HandleParam: " << node->ToString();

  int param_index = -1;
  for (int i = 0; i < f_->params().size(); i++) {
    if (f_->params()[i] == node) {
      param_index = i;
      break;
    }
  }
  XLS_RET_CHECK_NE(param_index, -1);
  if (!proc_data_->id_to_config_args.contains(proc_id_)) {
    return absl::InternalError(absl::StrCat(
        "Proc ID \"", proc_id_.ToString(), "\" was not found in arg mapping."));
  }

  ProcConfigValue value =
      proc_data_->id_to_config_args.at(proc_id_)[param_index];
  // TODO: https://github.com/google/xls/issues/704 - Associate array aliases,
  // similar to what we do for channels below.
  if (std::holds_alternative<Channel*>(value)) {
    XLS_RETURN_IF_ERROR(channel_scope_->AssociateWithExistingChannel(
        node->name_def(), std::get<Channel*>(value)));
  }
  node_to_ir_[node->name_def()] = value;
  return absl::OkStatus();
}

absl::Status ProcConfigIrConverter::HandleSpawn(const Spawn* node) {
  VLOG(4) << "ProcConfigIrConverter::HandleSpawn : " << node->ToString();
  std::vector<ProcConfigValue> config_args;
  XLS_ASSIGN_OR_RETURN(Proc * p, ResolveProc(node->callee(), type_info_));
  ProcId new_id = proc_id_factory_.CreateProcId(proc_id_, p);
  for (const auto& arg : node->config()->args()) {
    XLS_RETURN_IF_ERROR(arg->Accept(this));
    config_args.push_back(node_to_ir_.at(arg));
  }
  proc_data_->id_to_config_args[new_id] = config_args;

  if (!node->next()->args().empty()) {
    // Note: warning collect is nullptr since all warnings should have been
    // flagged in typechecking.
    XLS_ASSIGN_OR_RETURN(
        InterpValue value,
        ConstexprEvaluator::EvaluateToValue(
            import_data_, type_info_, /*warning_collector=*/nullptr, bindings_,
            node->next()->args()[0], nullptr));
    XLS_ASSIGN_OR_RETURN(auto ir_value, value.ConvertToIr());
    proc_data_->id_to_initial_value[new_id] = ir_value;
  }

  return absl::OkStatus();
}

absl::Status ProcConfigIrConverter::HandleStructInstance(
    const StructInstance* node) {
  VLOG(3) << "ProcConfigIrConverter::HandleStructInstance: "
          << node->ToString();
  XLS_ASSIGN_OR_RETURN(InterpValue const_value, type_info_->GetConstExpr(node));
  XLS_ASSIGN_OR_RETURN(auto ir_value, const_value.ConvertToIr());
  node_to_ir_[node] = ir_value;
  return absl::OkStatus();
}

absl::Status ProcConfigIrConverter::HandleXlsTuple(const XlsTuple* node) {
  for (const auto& element : node->members()) {
    XLS_RETURN_IF_ERROR(element->Accept(this));
  }
  final_tuple_ = node;
  return absl::OkStatus();
}

}  // namespace xls::dslx
