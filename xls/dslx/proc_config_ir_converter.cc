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
#include "xls/dslx/proc_config_ir_converter.h"

#include "absl/status/status.h"
#include "xls/common/logging/logging.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/dslx/concrete_type.h"
#include "xls/dslx/deduce_ctx.h"
#include "xls/dslx/ir_conversion_utils.h"
#include "xls/dslx/symbolic_bindings.h"

namespace xls::dslx {
namespace {

std::string ProcStackToId(const std::vector<Proc*>& stack) {
  return absl::StrJoin(stack, "->", [](std::string* out, const Proc* p) {
    out->append(p->identifier());
  });
}

}  // namespace

absl::StatusOr<Proc*> ResolveProc(Expr* node, ImportData* import_data) {
  if (NameRef* name_ref = dynamic_cast<NameRef*>(node); name_ref != nullptr) {
    return name_ref->owner()->GetProcOrError(name_ref->identifier());
  }

  // Else colon ref.
  ColonRef* colon_ref = dynamic_cast<ColonRef*>(node);
  if (colon_ref == nullptr) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Spawn callee (\"", node->ToString(),
        "\" is neither a NameRef nor ColonRef: ", node->GetNodeTypeName()));
  }

  absl::optional<Import*> import = colon_ref->ResolveImportSubject();
  XLS_RET_CHECK(import.has_value())
      << "ColonRef did not refer to an import: " << colon_ref->ToString();
  XLS_ASSIGN_OR_RETURN(
      const ModuleInfo* mod_info,
      import_data->Get(ImportTokens(import.value()->subject())));
  return mod_info->module->GetProcOrError(colon_ref->attr());
}

ProcConfigIrConverter::ProcConfigIrConverter(
    Package* package, Function* f, TypeInfo* type_info, ImportData* import_data,
    absl::flat_hash_map<ProcId, std::vector<ProcConfigValue>>* proc_id_to_args,
    absl::flat_hash_map<ProcId, MemberNameToValue>* proc_id_to_members,
    const SymbolicBindings& bindings, const ProcId& proc_id)
    : package_(package),
      f_(f),
      type_info_(type_info),
      import_data_(import_data),
      proc_id_to_args_(proc_id_to_args),
      proc_id_to_members_(proc_id_to_members),
      bindings_(bindings),
      proc_id_(proc_id),
      final_tuple_(nullptr) {
  (*proc_id_to_members_)[proc_id_] = {};
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
    Param* member = p->members()[i];
    proc_id_to_members_->at(proc_id_)[member->identifier()] =
        node_to_ir_.at(final_tuple_->members()[i]);
  }

  return absl::OkStatus();
}

absl::Status ProcConfigIrConverter::HandleChannelDecl(ChannelDecl* node) {
  XLS_VLOG(4) << "ProcConfigIrConverter::HandlesChannelDecl: "
              << node->ToString();
  std::string name = absl::StrCat(ProcStackToId(proc_id_.proc_stack),
                                  "_chandecl_", node->span().ToString());
  name = absl::StrReplaceAll(
      name, {{":", "_"}, {".", "_"}, {"-", "_"}, {"/", "_"}, {"\\", "_"}});
  auto concrete_type = type_info_->GetItem(node->type());
  XLS_RET_CHECK(concrete_type.has_value());
  XLS_ASSIGN_OR_RETURN(xls::Type * type,
                       TypeToIr(package_, *concrete_type.value(), bindings_));

  XLS_ASSIGN_OR_RETURN(
      StreamingChannel * channel,
      package_->CreateStreamingChannel(name, ChannelOps::kSendReceive, type));
  node_to_ir_[node] = channel;
  return absl::OkStatus();
}

absl::Status ProcConfigIrConverter::HandleLet(Let* node) {
  XLS_VLOG(4) << "ProcConfigIrConverter::HandleLet : " << node->ToString();
  XLS_RETURN_IF_ERROR(node->rhs()->Accept(this));

  if (ChannelDecl* decl = dynamic_cast<ChannelDecl*>(node->rhs());
      decl != nullptr) {
    Channel* channel = absl::get<Channel*>(node_to_ir_.at(decl));
    std::vector<NameDefTree::Leaf> leaves = node->name_def_tree()->Flatten();
    node_to_ir_[absl::get<NameDef*>(leaves[0])] = channel;
    node_to_ir_[absl::get<NameDef*>(leaves[1])] = channel;

  } else {
    if (!node->name_def_tree()->is_leaf()) {
      return absl::UnimplementedError(
          "Destructuring let bindings are not yet supported in Proc configs.");
    }

    // A leaf on the LHS of a Let will always be a NameDef.
    NameDef* def = absl::get<NameDef*>(node->name_def_tree()->leaf());
    auto value = node_to_ir_[node->rhs()];
    node_to_ir_[def] = value;
  }

  XLS_RETURN_IF_ERROR(node->body()->Accept(this));

  return absl::OkStatus();
}

absl::Status ProcConfigIrConverter::HandleNameRef(NameRef* node) {
  XLS_VLOG(4) << "ProcConfigIrConverter::HandleNameRef : " << node->ToString();
  NameDef* name_def = absl::get<NameDef*>(node->name_def());
  auto rhs = node_to_ir_.at(name_def);
  node_to_ir_[node] = rhs;
  return absl::OkStatus();
}

absl::Status ProcConfigIrConverter::HandleParam(Param* node) {
  // Matches a param AST node to the actual arg for this Proc instance.
  XLS_VLOG(4) << "ProcConfigIrConverter::HandleParam: " << node->ToString();

  int param_index = -1;
  for (int i = 0; i < f_->params().size(); i++) {
    if (f_->params()[i] == node) {
      param_index = i;
      break;
    }
  }
  XLS_RET_CHECK_NE(param_index, -1);
  if (!proc_id_to_args_->contains(proc_id_)) {
    return absl::InternalError(absl::StrCat(
        "Proc ID \"", proc_id_.ToString(), "\" was not found in arg mapping."));
  }
  node_to_ir_[node] = proc_id_to_args_->at(proc_id_)[param_index];
  node_to_ir_[node->name_def()] = proc_id_to_args_->at(proc_id_)[param_index];
  return absl::OkStatus();
}

absl::Status ProcConfigIrConverter::HandleFunction(Function* node) {
  for (int i = 0; i < node->params().size(); i++) {
    XLS_RETURN_IF_ERROR(node->params()[i]->Accept(this));
  }

  return node->body()->Accept(this);
}

absl::Status ProcConfigIrConverter::HandleSpawn(Spawn* node) {
  XLS_VLOG(4) << "ProcConfigIrConverter::HandleSpawn : " << node->ToString();
  std::vector<ProcConfigValue> args;
  XLS_ASSIGN_OR_RETURN(Proc * p, ResolveProc(node->callee(), import_data_));
  std::vector<Proc*> new_stack = proc_id_.proc_stack;
  new_stack.push_back(p);
  ProcId new_id{new_stack, instances_[new_stack]++};
  for (const auto& arg : node->config()->args()) {
    XLS_RETURN_IF_ERROR(arg->Accept(this));
    args.push_back(node_to_ir_.at(arg));
  }

  (*proc_id_to_args_)[new_id] = args;

  if (node->body() != nullptr) {
    return node->body()->Accept(this);
  }

  return absl::OkStatus();
}

absl::Status ProcConfigIrConverter::HandleXlsTuple(XlsTuple* node) {
  for (const auto& element : node->members()) {
    XLS_RETURN_IF_ERROR(element->Accept(this));
  }
  final_tuple_ = node;
  return absl::OkStatus();
}

}  // namespace xls::dslx
