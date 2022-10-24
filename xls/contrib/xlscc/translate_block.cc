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

#include "xls/contrib/xlscc/translator.h"

using std::shared_ptr;
using std::string;
using std::vector;

namespace xlscc {

absl::Status Translator::GenerateExternalChannels(
    const absl::flat_hash_map<std::string, HLSChannel>& channels_by_name,
    const HLSBlock& block, const clang::FunctionDecl* definition,
    const xls::SourceInfo& loc) {
  for (int pidx = 0; pidx < definition->getNumParams(); ++pidx) {
    const clang::ParmVarDecl* param = definition->getParamDecl(pidx);

    xls::Channel* new_channel = nullptr;

    const HLSChannel& hls_channel =
        channels_by_name.at(param->getNameAsString());
    if (hls_channel.type() == ChannelType::FIFO) {
      XLS_ASSIGN_OR_RETURN(std::shared_ptr<CType> ctype,
                           GetChannelType(param, loc));
      XLS_ASSIGN_OR_RETURN(xls::Type * data_type,
                           TranslateTypeToXLS(ctype, loc));

      XLS_ASSIGN_OR_RETURN(
          new_channel,
          package_->CreateStreamingChannel(
              hls_channel.name(),
              hls_channel.is_input() ? xls::ChannelOps::kReceiveOnly
                                     : xls::ChannelOps::kSendOnly,
              data_type, /*initial_values=*/{}, /*fifo_depth=*/absl::nullopt,
              xls::FlowControl::kReadyValid));
    } else {
      XLS_ASSIGN_OR_RETURN(StrippedType stripped,
                           StripTypeQualifiers(param->getType()));
      XLS_ASSIGN_OR_RETURN(std::shared_ptr<CType> ctype,
                           TranslateTypeFromClang(stripped.base, loc));

      XLS_ASSIGN_OR_RETURN(xls::Type * data_type,
                           TranslateTypeToXLS(ctype, loc));
      XLS_ASSIGN_OR_RETURN(new_channel, package_->CreateSingleValueChannel(
                                            hls_channel.name(),
                                            hls_channel.is_input()
                                                ? xls::ChannelOps::kReceiveOnly
                                                : xls::ChannelOps::kSendOnly,
                                            data_type));
    }
    XLS_CHECK(!external_channels_by_decl_.contains(param));
    external_channels_by_decl_[param] = new_channel;
  }
  return absl::OkStatus();
}

absl::StatusOr<xls::Proc*> Translator::GenerateIR_Block(
    xls::Package* package, const HLSBlock& block, int top_level_init_interval) {
  // Create external channels
  const clang::FunctionDecl* top_function = nullptr;

  XLS_CHECK_NE(parser_.get(), nullptr);
  XLS_ASSIGN_OR_RETURN(top_function, parser_->GetTopFunction());

  const clang::FunctionDecl* definition = nullptr;
  top_function->getBody(definition);
  xls::SourceInfo body_loc = GetLoc(*definition);
  package_ = package;

  XLS_RETURN_IF_ERROR(
      CheckInitIntervalValidity(top_level_init_interval, body_loc));

  absl::flat_hash_map<std::string, HLSChannel> channels_by_name;

  XLS_RETURN_IF_ERROR(
      GenerateIRBlockCheck(channels_by_name, block, definition, body_loc));
  XLS_RETURN_IF_ERROR(
      GenerateExternalChannels(channels_by_name, block, definition, body_loc));

  // Generate function without FIFO channel parameters
  // Force top function in block to be static.
  PreparedBlock prepared;
  XLS_ASSIGN_OR_RETURN(
      prepared.xls_func,
      GenerateIR_Top_Function(package, true, top_level_init_interval));

  std::vector<xls::Value> static_init_values;
  for (const clang::NamedDecl* namedecl :
       prepared.xls_func->GetDeterministicallyOrderedStaticValues()) {
    const ConstValue& initval = prepared.xls_func->static_values.at(namedecl);
    static_init_values.push_back(initval.rvalue());
  }

  xls::ProcBuilder pb(block.name() + "_proc", /*token_name=*/"tkn", package);
  pb.StateElement("st", xls::Value::Tuple(static_init_values));

  prepared.token = pb.GetTokenParam();

  XLS_RETURN_IF_ERROR(GenerateIRBlockPrepare(prepared, pb,
                                             /*next_return_index=*/0,
                                             /*next_state_index=*/0, definition,
                                             &channels_by_name, body_loc));

  XLS_ASSIGN_OR_RETURN(xls::BValue last_ret_val,
                       GenerateIOInvokes(prepared, pb, body_loc));

  // Create next state value
  std::vector<xls::BValue> static_next_values;
  XLS_CHECK_GE(prepared.xls_func->return_value_count,
               prepared.xls_func->static_values.size());
  for (const clang::NamedDecl* namedecl :
       prepared.xls_func->GetDeterministicallyOrderedStaticValues()) {
    XLS_CHECK(context().fb == &pb);
    xls::BValue next_val = GetFlexTupleField(
        last_ret_val, prepared.return_index_for_static[namedecl],
        prepared.xls_func->return_value_count, body_loc);

    XLS_ASSIGN_OR_RETURN(bool is_on_reset, DeclIsOnReset(namedecl));
    if (is_on_reset) {
      next_val = pb.Literal(xls::Value(xls::UBits(0, 1)), body_loc);
    }
    static_next_values.push_back(next_val);
  }
  const xls::BValue next_state = pb.Tuple(static_next_values);

  return pb.Build(prepared.token, {next_state});
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
    absl::flat_hash_map<std::string, HLSChannel>& channels_by_name,
    const HLSBlock& block, const clang::FunctionDecl* definition,
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

    if (channels_by_name.contains(channel.name())) {
      return absl::InvalidArgumentError(
          absl::StrFormat("Duplicate channel name %s", channel.name()));
    }

    channels_by_name[channel.name()] = channel;
    channel_names_in_block.insert(channel.name());
  }

  if (definition->parameters().size() != block.channels_size()) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Top function has %i parameters, but block proto defines %i channels",
        definition->parameters().size(), block.channels_size()));
  }

  for (const clang::ParmVarDecl* param : definition->parameters()) {
    if (!channel_names_in_block.contains(param->getNameAsString())) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "Block proto does not contain channels '%s' in function prototype",
          param->getNameAsString()));
    }
    channel_names_in_block.erase(param->getNameAsString());
  }

  if (!channel_names_in_block.empty()) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Block proto contains %i channels not in function prototype",
        channel_names_in_block.size()));
  }

  return absl::OkStatus();
}

absl::Status Translator::GenerateIRBlockPrepare(
    PreparedBlock& prepared, xls::ProcBuilder& pb, int64_t next_return_index,
    int64_t next_state_index, const clang::FunctionDecl* definition,
    const absl::flat_hash_map<std::string, HLSChannel>* channels_by_name,
    const xls::SourceInfo& body_loc) {
  // For defaults, updates, invokes
  context().fb = dynamic_cast<xls::BuilderBase*>(&pb);

  // Add returns for static locals
  {
    for (const clang::NamedDecl* namedecl :
         prepared.xls_func->GetDeterministicallyOrderedStaticValues()) {
      prepared.return_index_for_static[namedecl] = next_return_index++;
      prepared.state_index_for_static[namedecl] = next_state_index++;
    }
  }

  // Prepare direct-ins
  if (definition != nullptr) {
    XLS_CHECK_NE(channels_by_name, nullptr);

    for (int pidx = 0; pidx < definition->getNumParams(); ++pidx) {
      const clang::ParmVarDecl* param = definition->getParamDecl(pidx);

      const HLSChannel& hls_channel =
          channels_by_name->at(param->getNameAsString());
      if (hls_channel.type() != ChannelType::DIRECT_IN) {
        continue;
      }

      XLS_ASSIGN_OR_RETURN(StrippedType stripped,
                           StripTypeQualifiers(param->getType()));

      xls::Channel* xls_channel = external_channels_by_decl_.at(param);

      xls::BValue receive = pb.Receive(xls_channel, prepared.token);
      prepared.token = pb.TupleIndex(receive, 0);
      xls::BValue direct_in_value = pb.TupleIndex(receive, 1);

      prepared.args.push_back(direct_in_value);

      // If it's const or not a reference, then there's no return
      if (stripped.is_ref && !stripped.base.isConstQualified()) {
        ++next_return_index;
      }
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

    const clang::NamedDecl* param =
        prepared.xls_func->decls_by_io_channel.at(op.channel);

    if (!prepared.xls_channel_by_function_channel.contains(op.channel)) {
      xls::Channel* xls_channel = external_channels_by_decl_.at(param);
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
          XLS_ASSIGN_OR_RETURN(xls::BValue val,
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
        prepared.args.push_back(
            pb.TupleIndex(pb.GetStateParam(0), static_idx, body_loc));
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
