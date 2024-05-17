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

#include <cstdint>
#include <optional>
#include <string>
#include <variant>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/log/log.h"
#include "absl/meta/type_traits.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/dslx/constexpr_evaluator.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/import_data.h"
#include "xls/dslx/interp_value.h"
#include "xls/dslx/interp_value_utils.h"
#include "xls/dslx/ir_convert/extract_conversion_order.h"
#include "xls/dslx/ir_convert/ir_conversion_utils.h"
#include "xls/dslx/type_system/deduce_utils.h"
#include "xls/dslx/type_system/parametric_env.h"
#include "xls/dslx/type_system/type_info.h"
#include "xls/ir/channel.h"
#include "xls/ir/channel_ops.h"
#include "xls/ir/package.h"
#include "xls/ir/value.h"

namespace xls::dslx {
namespace {

std::string ProcStackToId(const std::vector<Proc*>& stack) {
  return absl::StrJoin(stack, "->", [](std::string* out, const Proc* p) {
    out->append(p->identifier());
  });
}

}  // namespace

ProcConfigIrConverter::ProcConfigIrConverter(Package* package, Function* f,
                                             TypeInfo* type_info,
                                             ImportData* import_data,
                                             ProcConversionData* proc_data,
                                             const ParametricEnv& bindings,
                                             const ProcId& proc_id)
    : package_(package),
      f_(f),
      type_info_(type_info),
      import_data_(import_data),
      channel_name_uniquer_(/*separator=*/"__"),
      proc_data_(proc_data),
      bindings_(bindings),
      proc_id_(proc_id),
      final_tuple_(nullptr) {
  proc_data->id_to_members[proc_id_] = {};
  // Populate channel name uniquer with pre-existing channel names.
  for (Channel* channel : package_->channels()) {
    channel_name_uniquer_.GetSanitizedUniqueName(channel->name());
  }
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

absl::Status ProcConfigIrConverter::HandleBlock(const Block* node) {
  VLOG(4) << "ProcConfigIrConverter::HandleBlock: " << node->ToString() << " : "
          << node->span().ToString();
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
  XLS_ASSIGN_OR_RETURN(
      InterpValue name_interp_value,
      ConstexprEvaluator::EvaluateToValue(
          import_data_, type_info_, /*warning_collector=*/nullptr, bindings_,
          &node->channel_name_expr()));
  XLS_ASSIGN_OR_RETURN(std::string channel_name,
                       InterpValueAsString(name_interp_value));
  channel_name = channel_name_uniquer_.GetSanitizedUniqueName(
      absl::StrCat(package_->name(), "__", channel_name));
  auto maybe_type = type_info_->GetItem(node->type());
  XLS_RET_CHECK(maybe_type.has_value());
  XLS_ASSIGN_OR_RETURN(xls::Type * type,
                       TypeToIr(package_, *maybe_type.value(), bindings_));

  std::optional<int64_t> fifo_depth;
  if (node->fifo_depth().has_value()) {
    // Note: warning collect is nullptr since all warnings should have been
    // flagged in typechecking.
    XLS_ASSIGN_OR_RETURN(
        InterpValue iv,
        ConstexprEvaluator::EvaluateToValue(
            import_data_, type_info_, /*warning_collector=*/nullptr, bindings_,
            node->fifo_depth().value()));
    XLS_ASSIGN_OR_RETURN(Value fifo_depth_value, iv.ConvertToIr());
    if (!fifo_depth_value.IsBits()) {
      return absl::InvalidArgumentError(
          absl::StrFormat("Expected fifo depth to be bits type, got %s.",
                          fifo_depth_value.ToHumanString()));
    }
    XLS_ASSIGN_OR_RETURN(fifo_depth, fifo_depth_value.bits().ToInt64());
  }

  std::optional<FifoConfig> fifo_config;
  if (fifo_depth.has_value()) {
    // We choose bypass=true FIFOs by default and register push outputs (ready).
    // The idea is to avoid combo loops introduced by pop->push ready
    // combinational paths. For depth zero FIFOs, we do not register push
    // outputs as for now we think of these FIFOs as direct connections.
    // TODO: google/xls#1391 - we should have a better way to specify fifo
    // configuration.
    fifo_config.emplace(FifoConfig(
        /*depth=*/*fifo_depth,
        /*bypass=*/true,
        /*register_push_outputs=*/*fifo_depth != 0,
        /*register_pop_outputs=*/false));
  }
  XLS_ASSIGN_OR_RETURN(StreamingChannel * channel,
                       package_->CreateStreamingChannel(
                           channel_name, ChannelOps::kSendReceive, type,
                           /*initial_values=*/{},
                           /*fifo_config=*/fifo_config));
  node_to_ir_[node] = channel;
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
    Channel* channel = std::get<Channel*>(node_to_ir_.at(decl));
    std::vector<NameDefTree::Leaf> leaves = node->name_def_tree()->Flatten();
    node_to_ir_[std::get<NameDef*>(leaves[0])] = channel;
    node_to_ir_[std::get<NameDef*>(leaves[1])] = channel;
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

  node_to_ir_[node->name_def()] =
      proc_data_->id_to_config_args.at(proc_id_)[param_index];
  return absl::OkStatus();
}

absl::Status ProcConfigIrConverter::HandleSpawn(const Spawn* node) {
  VLOG(4) << "ProcConfigIrConverter::HandleSpawn : " << node->ToString();
  std::vector<ProcConfigValue> config_args;
  XLS_ASSIGN_OR_RETURN(Proc * p, ResolveProc(node->callee(), type_info_));
  std::vector<Proc*> new_stack = proc_id_.proc_stack;
  new_stack.push_back(p);
  ProcId new_id{new_stack, instances_[new_stack]++};
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
