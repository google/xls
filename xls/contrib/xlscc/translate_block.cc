// Copyright 2022 The XLS Authors
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

#include <memory>

#include "absl/status/status.h"
#include "clang/include/clang/AST/Decl.h"
#include "clang/include/clang/AST/DeclCXX.h"
#include "xls/common/logging/logging.h"
#include "xls/common/status/status_macros.h"
#include "xls/contrib/xlscc/hls_block.pb.h"
#include "xls/contrib/xlscc/translator.h"
#include "xls/ir/function_builder.h"

using std::shared_ptr;
using std::string;
using std::vector;

namespace xlscc {

absl::Status Translator::GenerateExternalChannels(
    const HLSBlock& block, const std::list<ExternalChannelInfo>& top_decls,
    const xls::SourceInfo& loc) {
  for (const ExternalChannelInfo& top_decl : top_decls) {
    const clang::NamedDecl* decl = top_decl.decl;
    std::shared_ptr<CChannelType> channel_type = top_decl.channel_type;
    XLS_CHECK_NE(channel_type, nullptr);

    XLS_ASSIGN_OR_RETURN(xls::Type * data_type,
                         TranslateTypeToXLS(channel_type->GetItemType(), loc));

    xls::Channel* new_channel = nullptr;

    XLS_CHECK(channel_type->GetOpType() == OpType::kRecv ||
              channel_type->GetOpType() == OpType::kSend);
    const auto xls_channel_op = (channel_type->GetOpType() == OpType::kRecv)
                                    ? xls::ChannelOps::kReceiveOnly
                                    : xls::ChannelOps::kSendOnly;

    if (top_decl.interface_type == InterfaceType::kFIFO) {
      XLS_ASSIGN_OR_RETURN(
          new_channel, package_->CreateStreamingChannel(
                           decl->getNameAsString(), xls_channel_op, data_type,
                           /*initial_values=*/{}, /*fifo_depth=*/absl::nullopt,
                           xls::FlowControl::kReadyValid));
    } else if (top_decl.interface_type == InterfaceType::kDirect) {
      XLS_ASSIGN_OR_RETURN(
          new_channel, package_->CreateSingleValueChannel(
                           decl->getNameAsString(), xls_channel_op, data_type));
    } else {
      return absl::InvalidArgumentError(
          ErrorMessage(GetLoc(*decl), "Unknown interface type for channel %s",
                       decl->getNameAsString()));
    }
    XLS_CHECK(!external_channels_by_decl_.contains(decl));
    external_channels_by_decl_[decl] = new_channel;
  }
  return absl::OkStatus();
}

absl::StatusOr<xls::Proc*> Translator::GenerateIR_Block(
    xls::Package* package, const HLSBlock& block, int top_level_init_interval) {
  package_ = package;

  absl::flat_hash_map<std::string, HLSChannel> channels_by_name;
  for (const HLSChannel& channel : block.channels()) {
    if (!channel.has_name() || !channel.has_is_input() || !channel.has_type()) {
      return absl::InvalidArgumentError(
          absl::StrFormat("Channel is incomplete in proto"));
    }

    if (channels_by_name.contains(channel.name())) {
      return absl::InvalidArgumentError(
          absl::StrFormat("Duplicate channel name %s", channel.name()));
    }

    channels_by_name[channel.name()] = channel;
  }

  const clang::FunctionDecl* top_function = nullptr;

  XLS_CHECK_NE(parser_.get(), nullptr);
  XLS_ASSIGN_OR_RETURN(top_function, parser_->GetTopFunction());

  const clang::FunctionDecl* definition = nullptr;
  top_function->getBody(definition);
  const xls::SourceInfo body_loc = GetLoc(*definition);

  std::list<ExternalChannelInfo> top_decls;

  for (int pidx = 0; pidx < definition->getNumParams(); ++pidx) {
    const clang::ParmVarDecl* param = definition->getParamDecl(pidx);

    if (!channels_by_name.contains(param->getNameAsString())) {
      return absl::InvalidArgumentError(
          absl::StrFormat("Parameter %s doesn't name a channel",
                          param->getNameAsString().c_str()));
    }

    const HLSChannel& channel_spec =
        channels_by_name.at(param->getNameAsString());

    XLS_CHECK(channel_spec.type() == ChannelType::DIRECT_IN ||
              channel_spec.type() == ChannelType::FIFO);

    const OpType op_type_spec =
        channel_spec.is_input() ? OpType::kRecv : OpType::kSend;

    ExternalChannelInfo channel_info = {.decl = param};

    if (channel_spec.type() == ChannelType::DIRECT_IN) {
      channel_info.interface_type = InterfaceType::kDirect;
      XLS_ASSIGN_OR_RETURN(StrippedType stripped,
                           StripTypeQualifiers(param->getType()));
      XLS_ASSIGN_OR_RETURN(
          auto ctype, TranslateTypeFromClang(stripped.base, GetLoc(*param)));
      channel_info.channel_type =
          std::make_shared<CChannelType>(ctype, op_type_spec);
      channel_info.extra_return =
          stripped.is_ref && !stripped.base.isConstQualified();
    } else if (channel_spec.type() == ChannelType::FIFO) {
      channel_info.interface_type = InterfaceType::kFIFO;
      XLS_ASSIGN_OR_RETURN(
          channel_info.channel_type,
          GetChannelType(param->getType(), param->getASTContext(),
                         GetLoc(*param)));
      if (channel_info.channel_type->GetOpType() != OpType::kNull) {
        return absl::UnimplementedError(
            ErrorMessage(GetLoc(*param),
                         "Direction specifiers in parameter external channel "
                         "declarations not supported"));
      }
      channel_info.channel_type = std::make_shared<CChannelType>(
          channel_info.channel_type->GetItemType(), op_type_spec);
    } else {
      return absl::InvalidArgumentError(ErrorMessage(
          GetLoc(*param),
          "Don't know how to interpret channel type %i for param %s",
          (int)channel_spec.type(), param->getNameAsString()));
    }

    top_decls.push_back(channel_info);
  }

  return GenerateIR_Block(package, block, /*this_type=*/nullptr,
                          /*this_decl=*/nullptr, top_decls, body_loc,
                          top_level_init_interval,
                          /*force_static=*/true,
                          /*member_references_become_channels=*/false);
}

absl::StatusOr<xls::Proc*> Translator::GenerateIR_Block(
    xls::Package* package, const HLSBlock& block,
    std::shared_ptr<CStructType> this_type,
    const clang::CXXRecordDecl* this_decl,
    const std::list<ExternalChannelInfo>& top_decls,
    const xls::SourceInfo& body_loc, int top_level_init_interval,
    bool force_static, bool member_references_become_channels) {
  XLS_CHECK_NE(package_, nullptr);

  // Create external channels
  XLS_RETURN_IF_ERROR(
      CheckInitIntervalValidity(top_level_init_interval, body_loc));

  XLS_RETURN_IF_ERROR(GenerateIRBlockCheck(block, top_decls, body_loc));
  XLS_RETURN_IF_ERROR(GenerateExternalChannels(block, top_decls, body_loc));

  // Generate function without FIFO channel parameters
  // Force top function in block to be static.
  PreparedBlock prepared;
  XLS_ASSIGN_OR_RETURN(prepared.xls_func, GenerateIR_Top_Function(
                                              package, force_static,
                                              member_references_become_channels,
                                              top_level_init_interval));

  xls::ProcBuilder pb(block.name() + "_proc", /*token_name=*/"tkn", package);

  prepared.token = pb.GetTokenParam();

  XLS_RETURN_IF_ERROR(GenerateIRBlockPrepare(prepared, pb,
                                             /*next_return_index=*/0,
                                             /*next_state_index=*/0,
                                             /*this_type=*/this_type,
                                             /*this_decl=*/this_decl, top_decls,
                                             body_loc));

  XLS_ASSIGN_OR_RETURN(xls::BValue last_ret_val,
                       GenerateIOInvokes(prepared, pb, body_loc));

  // Create next state value
  std::vector<xls::BValue> static_next_values;
  XLS_CHECK_GE(prepared.xls_func->return_value_count,
               prepared.xls_func->static_values.size());

  XLS_CHECK(context().fb == &pb);

  if (this_decl != nullptr) {
    xls::BValue next_val = GetFlexTupleField(
        last_ret_val, prepared.return_index_for_static.at(this_decl),
        prepared.xls_func->return_value_count, body_loc);
    static_next_values.push_back(next_val);
  }

  for (const clang::NamedDecl* namedecl :
       prepared.xls_func->GetDeterministicallyOrderedStaticValues()) {
    xls::BValue next_val = GetFlexTupleField(
        last_ret_val, prepared.return_index_for_static[namedecl],
        prepared.xls_func->return_value_count, body_loc);

    XLS_ASSIGN_OR_RETURN(bool is_on_reset, DeclIsOnReset(namedecl));
    if (is_on_reset) {
      next_val = pb.Literal(xls::Value(xls::UBits(0, 1)), body_loc);
    }
    static_next_values.push_back(next_val);
  }
  XLS_CHECK_EQ(static_next_values.size(), prepared.state_init_count);
  // const xls::BValue next_state = pb.Tuple(static_next_values);

  return pb.Build(prepared.token, static_next_values);
}

absl::StatusOr<xls::Proc*> Translator::GenerateIR_BlockFromClass(
    xls::Package* package, HLSBlock* block_spec_out,
    int top_level_init_interval) {
  package_ = package;
  block_spec_out->Clear();

  // Create external channels
  const clang::FunctionDecl* top_function = nullptr;

  XLS_CHECK_NE(parser_.get(), nullptr);
  XLS_ASSIGN_OR_RETURN(top_function, parser_->GetTopFunction());

  if (!clang::isa<clang::CXXMethodDecl>(top_function)) {
    return absl::InvalidArgumentError(
        ErrorMessage(GetLoc(*top_function), "Top function %s isn't a method",
                     top_function->getQualifiedNameAsString().c_str()));
  }

  auto method = clang::dyn_cast<clang::CXXMethodDecl>(top_function);
  const clang::QualType& this_type = method->getThisType()->getPointeeType();

  XLS_CHECK(this_type->isRecordType());

  if (this_type.isConstQualified()) {
    return absl::UnimplementedError(
        ErrorMessage(GetLoc(*method), "Const top method unsupported"));
  }

  if (!top_function->getReturnType()->isVoidType()) {
    return absl::UnimplementedError(ErrorMessage(
        GetLoc(*method), "Non-void top method return unsupported"));
  }
  if (!top_function->parameters().empty()) {
    return absl::UnimplementedError(
        ErrorMessage(GetLoc(*method), "Top method parameters unsupported"));
  }

  const clang::CXXRecordDecl* record_decl = this_type->getAsCXXRecordDecl();

  XLS_RETURN_IF_ERROR(ScanStruct(record_decl));

  XLS_ASSIGN_OR_RETURN(
      shared_ptr<CType> this_ctype,
      TranslateTypeFromClang(this_type, GetLoc(*top_function)));

  XLS_ASSIGN_OR_RETURN(std::shared_ptr<CType> struct_ctype,
                       ResolveTypeInstance(this_ctype));

  auto struct_type = std::dynamic_pointer_cast<CStructType>(struct_ctype);
  XLS_CHECK_NE(struct_type, nullptr);

  block_spec_out->set_name(record_decl->getNameAsString());

  std::list<ExternalChannelInfo> top_decls;

  for (const std::shared_ptr<CField>& field : struct_type->fields()) {
    XLS_ASSIGN_OR_RETURN(std::shared_ptr<CType> resolved_field_type,
                         ResolveTypeInstanceDeeply(field->type()));

    if (auto channel_type =
            std::dynamic_pointer_cast<CChannelType>(field->type())) {
      if (channel_type->GetOpType() == OpType::kNull) {
        return absl::InvalidArgumentError(
            ErrorMessage(GetLoc(*field->name()),
                         "Direction unspecified for external channel %s",
                         field->name()->getNameAsString().c_str()));
      }

      xlscc::HLSChannel* channel_spec = block_spec_out->add_channels();
      channel_spec->set_name(field->name()->getNameAsString());
      channel_spec->set_width_in_bits(resolved_field_type->GetBitWidth());
      channel_spec->set_type(xlscc::FIFO);
      channel_spec->set_is_input(channel_type->GetOpType() == OpType::kRecv);

      ExternalChannelInfo channel_info = {
          .decl = field->name(),
          .channel_type = channel_type,
          .interface_type = InterfaceType::kFIFO};
      top_decls.push_back(channel_info);
    } else if (auto channel_type =
                   std::dynamic_pointer_cast<CReferenceType>(field->type())) {
      xlscc::HLSChannel* channel_spec = block_spec_out->add_channels();
      channel_spec->set_name(field->name()->getNameAsString());
      channel_spec->set_width_in_bits(resolved_field_type->GetBitWidth());
      channel_spec->set_type(xlscc::DIRECT_IN);
      channel_spec->set_is_input(true);

      ExternalChannelInfo channel_info = {
          .decl = field->name(),
          .channel_type = std::make_shared<CChannelType>(
              channel_type->GetPointeeType(), OpType::kRecv),
          .interface_type = InterfaceType::kDirect};
      top_decls.push_back(channel_info);
    }
  }

  return GenerateIR_Block(package, *block_spec_out, struct_type,
                          /*this_decl=*/record_decl, top_decls,
                          GetLoc(*record_decl), top_level_init_interval,
                          /*force_static=*/false,
                          /*member_references_become_channels=*/true);
}

absl::StatusOr<xls::BValue> Translator::GenerateIOInvokes(
    PreparedBlock& prepared, xls::ProcBuilder& pb,
    const xls::SourceInfo& body_loc) {
  XLS_CHECK(&pb == context().fb);

  XLS_CHECK_GE(prepared.xls_func->return_value_count,
               prepared.xls_func->io_ops.size());

  absl::flat_hash_map<const IOOp*, xls::BValue> op_tokens;

  std::list<const IOOp*> fan_ins_ordered;

  // The function is first invoked with defaults for any
  //  read() IO Ops.
  // If there are any read() IO Ops, then it will be invoked again
  //  for each read Op below.
  // Statics don't need to generate additional invokes, since they need not
  //  exchange any data with the outside world between iterations.
  xls::BValue last_ret_val =
      pb.Invoke(prepared.args, prepared.xls_func->xls_func, body_loc);
  for (const IOOp& op : prepared.xls_func->io_ops) {
    xls::Channel* xls_channel =
        prepared.xls_channel_by_function_channel.at(op.channel);
    const int return_index = prepared.return_index_for_op.at(&op);

    xls::SourceInfo op_loc = op.op_location;

    xls::BValue before_token = prepared.token;
    xls::BValue new_token;
    if (!op.after_ops.empty()) {
      std::vector<xls::BValue> after_tokens;
      for (const IOOp* op : op.after_ops) {
        after_tokens.push_back(op_tokens.at(op));
      }
      before_token = pb.AfterAll(after_tokens, body_loc);
    }

    if (op.op == OpType::kRecv) {
      const int arg_index = prepared.arg_index_for_op.at(&op);
      XLS_CHECK(arg_index >= 0 && arg_index < prepared.args.size());

      xls::BValue condition = GetFlexTupleField(
          last_ret_val, return_index, prepared.xls_func->return_value_count,
          op_loc, absl::StrFormat("%s_pred", xls_channel->name()));
      XLS_CHECK_EQ(condition.GetType()->GetFlatBitCount(), 1);
      xls::BValue receive;
      if (op.is_blocking) {
        receive = pb.ReceiveIf(xls_channel, before_token, condition, op_loc);
      } else {
        receive = pb.ReceiveIfNonBlocking(xls_channel, before_token, condition,
                                          op_loc);
      }
      new_token = pb.TupleIndex(receive, 0);

      xls::BValue in_val;
      if (op.is_blocking) {
        in_val = pb.TupleIndex(receive, 1);
      } else {
        in_val =
            pb.Tuple({pb.TupleIndex(receive, 1), pb.TupleIndex(receive, 2)});
      }
      prepared.args[arg_index] = in_val;

      // The function is invoked again with the value received from the channel
      //  for each read() Op. The final invocation will produce all complete
      //  outputs.
      last_ret_val =
          pb.Invoke(prepared.args, prepared.xls_func->xls_func, op_loc);
    } else if (op.op == OpType::kSend) {
      xls::BValue send_tup =
          GetFlexTupleField(last_ret_val, return_index,
                            prepared.xls_func->return_value_count, op_loc);
      xls::BValue val = pb.TupleIndex(send_tup, 0, op_loc);
      xls::BValue condition = pb.TupleIndex(
          send_tup, 1, op_loc, absl::StrFormat("%s_pred", xls_channel->name()));

      new_token =
          pb.SendIf(xls_channel, before_token, condition, {val}, op_loc);
    } else {
      XLS_CHECK("Unknown IOOp type" == nullptr);
    }

    XLS_CHECK(!op_tokens.contains(&op));
    op_tokens[&op] = new_token;

    fan_ins_ordered.push_back(&op);
  }

  if (!fan_ins_ordered.empty()) {
    std::vector<xls::BValue> fan_ins_tokens;
    for (const IOOp* op : fan_ins_ordered) {
      fan_ins_tokens.push_back(op_tokens.at(op));
    }
    prepared.token = pb.AfterAll(fan_ins_tokens, body_loc);
  }

  return last_ret_val;
}

absl::Status Translator::GenerateIRBlockCheck(
    const HLSBlock& block, const std::list<ExternalChannelInfo>& top_decls,
    const xls::SourceInfo& body_loc) {
  if (!block.has_name()) {
    return absl::InvalidArgumentError(absl::StrFormat("HLSBlock has no name"));
  }

  absl::flat_hash_set<string> channel_names_in_block;
  for (const HLSChannel& channel : block.channels()) {
    if (!channel.has_name() || !channel.has_is_input() || !channel.has_type()) {
      return absl::InvalidArgumentError(
          absl::StrFormat("Channel is incomplete in proto"));
    }

    channel_names_in_block.insert(channel.name());
  }

  if (top_decls.size() != block.channels_size()) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Top function has %i parameters, but block proto defines %i channels",
        top_decls.size(), block.channels_size()));
  }
  for (const ExternalChannelInfo& top_decl : top_decls) {
    const clang::NamedDecl* decl = top_decl.decl;

    if (!channel_names_in_block.contains(decl->getNameAsString())) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "Block proto does not contain channels '%s' in function prototype",
          decl->getNameAsString()));
    }
    channel_names_in_block.erase(decl->getNameAsString());
  }

  if (!channel_names_in_block.empty()) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Block proto contains %i channels not in function prototype",
        channel_names_in_block.size()));
  }

  return absl::OkStatus();
}

absl::StatusOr<xls::Value> Translator::GenerateTopClassInitValue(
    std::shared_ptr<CStructType> this_type,
    // Can be nullptr
    const clang::CXXRecordDecl* this_decl, const xls::SourceInfo& body_loc) {
  for (const clang::CXXConstructorDecl* ctor : this_decl->ctors()) {
    if (!ctor->isTrivial()) {
      ctor->dump();
      return absl::UnimplementedError(ErrorMessage(
          body_loc, "Non-trivial constructors in top class not yet supported"));
    }
  }

  std::vector<xls::BValue> field_bvals;

  // Check for side-effects
  for (const clang::FieldDecl* field_decl : this_decl->fields()) {
    XLS_ASSIGN_OR_RETURN(
        std::shared_ptr<CType> field_type,
        TranslateTypeFromClang(field_decl->getType(), GetLoc(*field_decl)));

    XLS_ASSIGN_OR_RETURN(bool contains_lvalues,
                         field_type->ContainsLValues(*this));

    xls::BValue field_bval;
    if (contains_lvalues || !field_decl->hasInClassInitializer()) {
      XLS_ASSIGN_OR_RETURN(field_bval,
                           CreateDefaultValue(field_type, GetLoc(*field_decl)));
    } else {
      PushContextGuard guard(*this, GetLoc(*field_decl));
      context().mask_side_effects = true;
      context().any_side_effects_requested = false;

      XLS_ASSIGN_OR_RETURN(CValue field_cval,
                           GenerateIR_Expr(field_decl->getInClassInitializer(),
                                           GetLoc(*field_decl)));
      field_bval = field_cval.rvalue();
      XLS_CHECK(field_bval.valid());
      XLS_CHECK(*field_cval.type() == *field_type);
    }

    if (context().any_side_effects_requested) {
      return absl::UnimplementedError(
          ErrorMessage(GetLoc(*field_decl),
                       "Side effects in initializer for top class field %s",
                       field_decl->getQualifiedNameAsString()));
    }
    field_bvals.push_back(field_bval);
  }

  xls::BValue this_bval = MakeStructXLS(field_bvals, *this_type, body_loc);

  XLS_CHECK(this_bval.valid());

  XLS_ASSIGN_OR_RETURN(xls::Value this_init_val,
                       EvaluateBVal(this_bval, body_loc));

  return this_init_val;
}

absl::Status Translator::GenerateIRBlockPrepare(
    PreparedBlock& prepared, xls::ProcBuilder& pb, int64_t next_return_index,
    int64_t next_state_index, std::shared_ptr<CStructType> this_type,
    const clang::CXXRecordDecl* this_decl,
    const std::list<ExternalChannelInfo>& top_decls,
    const xls::SourceInfo& body_loc) {
  // For defaults, updates, invokes
  context().fb = dynamic_cast<xls::BuilderBase*>(&pb);

  // This state and argument
  if (this_decl != nullptr) {
    XLS_ASSIGN_OR_RETURN(
        xls::Value this_init_val,
        GenerateTopClassInitValue(this_type, this_decl, body_loc));

    prepared.state_index_for_static[this_decl] = next_state_index++;
    pb.StateElement("this", this_init_val, body_loc);
    ++prepared.state_init_count;

    prepared.args.push_back(
        pb.GetStateParam(prepared.state_index_for_static.at(this_decl)));
  }

  for (const clang::NamedDecl* namedecl :
       prepared.xls_func->GetDeterministicallyOrderedStaticValues()) {
    const ConstValue& initval = prepared.xls_func->static_values.at(namedecl);

    pb.StateElement(XLSNameMangle(clang::GlobalDecl(namedecl)),
                    initval.rvalue(), body_loc);
    ++prepared.state_init_count;
  }

  // Add returns for static locals
  {
    for (const clang::NamedDecl* namedecl :
         prepared.xls_func->GetDeterministicallyOrderedStaticValues()) {
      prepared.return_index_for_static[namedecl] = next_return_index++;
      prepared.state_index_for_static[namedecl] = next_state_index++;
    }
  }

  // This return
  if (this_decl != nullptr) {
    prepared.return_index_for_static[this_decl] = next_return_index++;
  }

  // Prepare direct-ins
  for (const ExternalChannelInfo& top_decl : top_decls) {
    if (top_decl.interface_type != InterfaceType::kDirect) {
      continue;
    }

    xls::Channel* xls_channel = external_channels_by_decl_.at(top_decl.decl);

    xls::BValue receive = pb.Receive(xls_channel, prepared.token);
    prepared.token = pb.TupleIndex(receive, 0);
    xls::BValue direct_in_value = pb.TupleIndex(receive, 1);

    prepared.args.push_back(direct_in_value);

    // If it's const or not a reference, then there's no return
    if (top_decl.extra_return) {
      ++next_return_index;
    }
  }

  // Initialize parameters to defaults, handle direct-ins, create channels
  // Add channels in order of function prototype
  // Find return indices for ops
  for (const IOOp& op : prepared.xls_func->io_ops) {
    prepared.return_index_for_op[&op] = next_return_index++;

    if (op.channel->generated != nullptr) {
      prepared.xls_channel_by_function_channel[op.channel] =
          op.channel->generated;
      continue;
    }

    const clang::NamedDecl* decl =
        prepared.xls_func->decls_by_io_channel.at(op.channel);

    if (!prepared.xls_channel_by_function_channel.contains(op.channel)) {
      xls::Channel* xls_channel = external_channels_by_decl_.at(decl);
      prepared.xls_channel_by_function_channel[op.channel] = xls_channel;
    }
  }

  // Params
  for (const xlscc::SideEffectingParameter& param :
       prepared.xls_func->side_effecting_parameters) {
    switch (param.type) {
      case xlscc::SideEffectingParameterType::kIOOp: {
        const IOOp& op = *param.io_op;
        if (op.channel->channel_op_type == OpType::kRecv) {
          XLS_ASSIGN_OR_RETURN(
              xls::BValue val,
              CreateDefaultValue(op.channel->item_type, body_loc));
          if (!op.is_blocking) {
            val = pb.Tuple({val, pb.Literal(xls::UBits(1, 1))});
          }
          prepared.arg_index_for_op[&op] = prepared.args.size();
          prepared.args.push_back(val);
        }
        break;
      }
      case xlscc::SideEffectingParameterType::kStatic: {
        const uint64_t static_idx =
            prepared.state_index_for_static.at(param.static_value);
        prepared.args.push_back(pb.GetStateParam(static_idx));
        break;
      }
      default: {
        return absl::InternalError(
            ErrorMessage(body_loc, "Unknown type of SideEffectingParameter"));
        break;
      }
    }
  }

  return absl::OkStatus();
}

}  // namespace xlscc
