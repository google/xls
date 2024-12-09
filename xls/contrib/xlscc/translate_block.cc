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

#include <algorithm>
#include <cstdint>
#include <iterator>
#include <list>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/container/btree_map.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "clang/include/clang/AST/Attr.h"
#include "clang/include/clang/AST/Attrs.inc"
#include "clang/include/clang/AST/Decl.h"
#include "clang/include/clang/AST/DeclBase.h"
#include "clang/include/clang/AST/DeclCXX.h"
#include "clang/include/clang/AST/GlobalDecl.h"
#include "clang/include/clang/AST/Type.h"
#include "clang/include/clang/Basic/LLVM.h"
#include "xls/common/math_util.h"
#include "xls/common/status/status_macros.h"
#include "xls/contrib/xlscc/cc_parser.h"
#include "xls/contrib/xlscc/hls_block.pb.h"
#include "xls/contrib/xlscc/translator.h"
#include "xls/contrib/xlscc/xlscc_logging.h"
#include "xls/ir/bits.h"
#include "xls/ir/channel.h"
#include "xls/ir/channel_ops.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/lsb_or_msb.h"
#include "xls/ir/nodes.h"
#include "xls/ir/op.h"
#include "xls/ir/package.h"
#include "xls/ir/source_location.h"
#include "xls/ir/state_element.h"
#include "xls/ir/value.h"
#include "xls/ir/value_utils.h"

using ::std::shared_ptr;
using ::std::string;
using ::std::vector;

namespace xlscc {

namespace {

inline std::string ToString(
    const absl::flat_hash_map<std::string, xls::ChannelStrictness>&
        channel_strictness_map) {
  return absl::StrJoin(
      channel_strictness_map, ",",
      [](std::string* out,
         const std::pair<std::string, xls::ChannelStrictness>& item) {
        absl::StrAppend(out, item.first, ":",
                        xls::ChannelStrictnessToString(item.second));
      });
}

}  // namespace

absl::StatusOr<xls::ChannelStrictness> Translator::GetChannelStrictness(
    const clang::NamedDecl& decl, const ChannelOptions& channel_options,
    absl::flat_hash_map<std::string, xls::ChannelStrictness>&
        unused_strictness_options) {
  std::optional<xls::ChannelStrictness> channel_strictness;
  for (const clang::AnnotateAttr* attr :
       decl.specific_attrs<clang::AnnotateAttr>()) {
    if (attr->getAnnotation() == "hls_channel_strictness") {
      if (attr->args_size() != 1) {
        return absl::InvalidArgumentError(
            "hls_channel_strictness must have exactly one argument");
      }
      clang::Expr::EvalResult result;
      if (!(*attr->args().begin())
               ->EvaluateAsInt(result, decl.getASTContext()) ||
          !result.Val.getInt().isNonNegative() ||
          !result.Val.getInt().isRepresentableByInt64()) {
        return absl::InvalidArgumentError(
            "hls_channel_strictness argument must be a xls::ChannelStrictness "
            "enum value");
      }
      channel_strictness = static_cast<xls::ChannelStrictness>(
          result.Val.getInt().getExtValue());
    }
  }

  if (auto it = channel_options.strictness_map.find(decl.getNameAsString());
      it != channel_options.strictness_map.end()) {
    if (channel_strictness.has_value() && *channel_strictness != it->second) {
      return absl::InvalidArgumentError(ErrorMessage(
          GetLoc(decl),
          "Command-line-specified channel strictness contradicts "
          "hls_channel_strictness annotation for channel: %s (command-line: "
          "%s, attribute: %s)",
          decl.getNameAsString(), xls::ChannelStrictnessToString(it->second),
          xls::ChannelStrictnessToString(channel_strictness.value())));
    }
    channel_strictness = it->second;

    // Record that we used this strictness-map entry.
    unused_strictness_options.erase(decl.getNameAsString());
  }
  return channel_strictness.value_or(channel_options.default_strictness);
}

absl::Status Translator::GenerateExternalChannels(
    std::list<ExternalChannelInfo>& top_decls,
    absl::flat_hash_map<const clang::NamedDecl*, ChannelBundle>*
        top_channel_injections,
    const xls::SourceInfo& loc) {
  CHECK_NE(top_channel_injections, nullptr);
  for (ExternalChannelInfo& top_decl : top_decls) {
    const clang::NamedDecl* decl = top_decl.decl;
    std::shared_ptr<CChannelType> channel_type = top_decl.channel_type;
    CHECK_NE(channel_type, nullptr);

    XLS_ASSIGN_OR_RETURN(xls::Type * data_type,
                         TranslateTypeToXLS(channel_type->GetItemType(), loc));

    ChannelBundle new_channel;

    if (top_decl.interface_type == InterfaceType::kFIFO) {
      const auto xls_channel_op = top_decl.is_input
                                      ? xls::ChannelOps::kReceiveOnly
                                      : xls::ChannelOps::kSendOnly;

      XLS_ASSIGN_OR_RETURN(
          new_channel.regular,
          package_->CreateStreamingChannel(
              decl->getNameAsString(), xls_channel_op, data_type,
              /*initial_values=*/{}, /*fifo_config=*/std::nullopt,
              xls::FlowControl::kReadyValid,
              /*strictness=*/top_decl.strictness));
      unused_xls_channel_ops_.push_back(
          {new_channel.regular, /*is_send=*/!top_decl.is_input});
    } else if (top_decl.interface_type == InterfaceType::kDirect) {
      CHECK(top_decl.is_input);
      XLS_ASSIGN_OR_RETURN(new_channel.regular,
                           package_->CreateSingleValueChannel(
                               decl->getNameAsString(),
                               xls::ChannelOps::kReceiveOnly, data_type));
      unused_xls_channel_ops_.push_back(
          {new_channel.regular, /*is_send=*/false});
    } else if (top_decl.interface_type == InterfaceType::kMemory) {
      const std::string& memory_name = top_decl.decl->getNameAsString();

      XLS_ASSIGN_OR_RETURN(
          xls::Type * read_request_type,
          top_decl.channel_type->GetReadRequestType(package_, data_type));
      XLS_ASSIGN_OR_RETURN(
          new_channel.read_request,
          package_->CreateStreamingChannel(
              memory_name + "_read_request", xls::ChannelOps::kSendOnly,
              read_request_type,
              /*initial_values=*/{}, /*fifo_config=*/std::nullopt,
              xls::FlowControl::kReadyValid,
              /*strictness=*/top_decl.strictness));
      unused_xls_channel_ops_.push_back(
          {new_channel.read_request, /*is_send=*/true});

      XLS_ASSIGN_OR_RETURN(
          xls::Type * read_response_type,
          top_decl.channel_type->GetReadResponseType(package_, data_type));
      XLS_ASSIGN_OR_RETURN(
          new_channel.read_response,
          package_->CreateStreamingChannel(
              memory_name + "_read_response", xls::ChannelOps::kReceiveOnly,
              read_response_type,
              /*initial_values=*/{}, /*fifo_config=*/std::nullopt,
              xls::FlowControl::kReadyValid,
              /*strictness=*/top_decl.strictness));
      unused_xls_channel_ops_.push_back(
          {new_channel.read_response, /*is_send=*/false});

      XLS_ASSIGN_OR_RETURN(
          xls::Type * write_request_type,
          top_decl.channel_type->GetWriteRequestType(package_, data_type));
      XLS_ASSIGN_OR_RETURN(
          new_channel.write_request,
          package_->CreateStreamingChannel(
              memory_name + "_write_request", xls::ChannelOps::kSendOnly,
              write_request_type,
              /*initial_values=*/{}, /*fifo_config=*/std::nullopt,
              xls::FlowControl::kReadyValid,
              /*strictness=*/top_decl.strictness));
      unused_xls_channel_ops_.push_back(
          {new_channel.write_request, /*is_send=*/true});

      XLS_ASSIGN_OR_RETURN(
          xls::Type * write_response_type,
          top_decl.channel_type->GetWriteResponseType(package_, data_type));
      XLS_ASSIGN_OR_RETURN(
          new_channel.write_response,
          package_->CreateStreamingChannel(
              memory_name + "_write_response", xls::ChannelOps::kReceiveOnly,
              write_response_type,
              /*initial_values=*/{}, /*fifo_config=*/std::nullopt,
              xls::FlowControl::kReadyValid,
              /*strictness=*/top_decl.strictness));
      unused_xls_channel_ops_.push_back(
          {new_channel.write_response, /*is_send=*/false});
    } else {
      return absl::InvalidArgumentError(
          ErrorMessage(GetLoc(*decl), "Unknown interface type for channel %s",
                       decl->getNameAsString()));
    }
    top_decl.external_channels = new_channel;
    if (top_decl.interface_type != InterfaceType::kDirect) {
      (*top_channel_injections)[decl] = new_channel;
    }
  }
  return absl::OkStatus();
}

absl::StatusOr<xls::Proc*> Translator::GenerateIR_Block(
    xls::Package* package, const HLSBlock& block, int top_level_init_interval,
    const ChannelOptions& channel_options) {
  package_ = package;

  absl::flat_hash_map<std::string, HLSChannel> channels_by_name;
  for (const HLSChannel& channel : block.channels()) {
    if (!channel.has_name() || !channel.has_type()) {
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

  CHECK_NE(parser_.get(), nullptr);
  XLS_ASSIGN_OR_RETURN(top_function, parser_->GetTopFunction());

  const clang::FunctionDecl* definition = nullptr;
  top_function->getBody(definition);
  const xls::SourceInfo body_loc = GetLoc(*definition);

  std::list<ExternalChannelInfo> top_decls;
  absl::flat_hash_map<std::string, xls::ChannelStrictness>
      unused_strictness_options = channel_options.strictness_map;
  for (int pidx = 0; pidx < definition->getNumParams(); ++pidx) {
    const clang::ParmVarDecl* param = definition->getParamDecl(pidx);

    if (!channels_by_name.contains(param->getNameAsString())) {
      return absl::InvalidArgumentError(
          absl::StrFormat("Parameter %s doesn't name a channel",
                          param->getNameAsString().c_str()));
    }

    const HLSChannel& channel_spec =
        channels_by_name.at(param->getNameAsString());

    CHECK(channel_spec.type() == ChannelType::DIRECT_IN ||
          channel_spec.type() == ChannelType::FIFO ||
          channel_spec.type() == ChannelType::MEMORY);

    ExternalChannelInfo channel_info = {.decl = param};
    XLS_ASSIGN_OR_RETURN(channel_info.strictness,
                         GetChannelStrictness(*param, channel_options,
                                              unused_strictness_options));
    if (channel_spec.type() == ChannelType::DIRECT_IN) {
      channel_info.interface_type = InterfaceType::kDirect;
      XLS_ASSIGN_OR_RETURN(StrippedType stripped,
                           StripTypeQualifiers(param->getType()));
      XLS_ASSIGN_OR_RETURN(
          auto ctype, TranslateTypeFromClang(stripped.base, GetLoc(*param)));
      channel_info.channel_type =
          std::make_shared<CChannelType>(ctype, /*memory_size=*/-1);
      channel_info.extra_return =
          stripped.is_ref && !stripped.base.isConstQualified();
      channel_info.is_input = channel_spec.is_input();
    } else if (channel_spec.type() == ChannelType::FIFO) {
      channel_info.interface_type = InterfaceType::kFIFO;
      XLS_ASSIGN_OR_RETURN(
          channel_info.channel_type,
          GetChannelType(param->getType(), param->getASTContext(),
                         GetLoc(*param)));

      channel_info.channel_type = std::make_shared<CChannelType>(
          channel_info.channel_type->GetItemType(),
          /*memory_size=*/-1);
      channel_info.is_input = channel_spec.is_input();
    } else if (channel_spec.type() == ChannelType::MEMORY) {
      channel_info.interface_type = InterfaceType::kMemory;
      XLS_ASSIGN_OR_RETURN(
          channel_info.channel_type,
          GetChannelType(param->getType(), param->getASTContext(),
                         GetLoc(*param)));

      CHECK_EQ(channel_spec.depth(),
               channel_info.channel_type->GetMemorySize());
    } else {
      return absl::InvalidArgumentError(ErrorMessage(
          GetLoc(*param),
          "Don't know how to interpret channel type %i for param %s",
          static_cast<int>(channel_spec.type()), param->getNameAsString()));
    }

    top_decls.push_back(channel_info);
  }
  if (!unused_strictness_options.empty()) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Unused channel strictness options: %s",
                        ToString(unused_strictness_options)));
  }

  return GenerateIR_Block(package, block, /*this_type=*/nullptr,
                          /*this_decl=*/nullptr, top_decls, body_loc,
                          top_level_init_interval,
                          /*force_static=*/true,
                          /*member_references_become_channels=*/false);
}

absl::StatusOr<xls::Proc*> Translator::GenerateIR_Block(
    xls::Package* package, const HLSBlock& block,
    const std::shared_ptr<CType>& this_type,
    const clang::CXXRecordDecl* this_decl,
    std::list<ExternalChannelInfo>& top_decls, const xls::SourceInfo& body_loc,
    int top_level_init_interval, bool force_static,
    bool member_references_become_channels) {
  CHECK_NE(package_, nullptr);

  // Create external channels
  XLS_RETURN_IF_ERROR(
      CheckInitIntervalValidity(top_level_init_interval, body_loc));

  XLS_RETURN_IF_ERROR(GenerateIRBlockCheck(block, top_decls, body_loc));
  absl::flat_hash_map<const clang::NamedDecl*, ChannelBundle>
      top_channel_injections;
  XLS_RETURN_IF_ERROR(
      GenerateExternalChannels(top_decls, &top_channel_injections, body_loc));

  return GenerateIR_Block(package, block.name(), top_channel_injections,
                          this_type, this_decl, top_decls, body_loc,
                          top_level_init_interval, force_static,
                          member_references_become_channels);
}

absl::Status Translator::GenerateIR_SubBlock(GeneratedFunction* gen_func,
                                             xls::Package* package,
                                             int top_level_init_interval) {
  if (clang::isa<clang::CXXMethodDecl>(gen_func->clang_decl)) {
    return absl::UnimplementedError(
        absl::StrFormat("Sub-block methods not supported for now"));
  }
  absl::flat_hash_map<
      int64_t, absl::flat_hash_map<const clang::NamedDecl*, ChannelBundle>>
      top_channel_injections_by_state;

  // Map channels
  for (const auto& [channels, state_index] :
       gen_func->state_index_by_called_channel_order) {
    int64_t num_channel_params = 0;
    for (int pi = 0; pi < gen_func->clang_decl->getNumParams(); ++pi) {
      const clang::ParmVarDecl* p = gen_func->clang_decl->getParamDecl(pi);

      XLS_ASSIGN_OR_RETURN(StrippedType stripped,
                           StripTypeQualifiers(p->getType()));

      // IO returns are later
      XLS_ASSIGN_OR_RETURN(bool is_channel,
                           TypeIsChannel(p->getType(), GetLoc(*p)));
      if (!is_channel) {
        CHECK(stripped.base.isConstQualified());
        continue;
      }

      top_channel_injections_by_state[state_index][p] =
          channels.at(num_channel_params);
      ++num_channel_params;
    }
    CHECK_EQ(channels.size(), num_channel_params);
  }

  if (top_channel_injections_by_state.size() != 1) {
    return absl::UnimplementedError(
        absl::StrFormat("Sub-block %s has %i states (combinations of "
                        "channels parameters), only 1 supported for now",
                        gen_func->clang_decl->getNameAsString(),
                        (int)top_channel_injections_by_state.size()));
  }

  // Save necessary global caller context and set up global callee context

  const clang::FunctionDecl* currently_generating_top_function_caller =
      currently_generating_top_function_;

  // Used for metadata, point to the actual implementation
  CHECK(xls_names_for_functions_generated_.contains(gen_func->clang_decl));
  xls_names_for_functions_generated_.erase(gen_func->clang_decl);

  // Many clang::NamedDecls will be re-encountered, since a stub was already
  // generated for the caller.
  auto unique_decl_ids_caller = std::move(unique_decl_ids_);
  unique_decl_ids_ = absl::flat_hash_set<const clang::NamedDecl*>();

  // Don't propagate sub-sub-blocks up,
  //  and don't generate the caller's sub-blocks again (infinite recursion)
  auto sub_block_stub_functions_caller = std::move(sub_block_stub_functions_);
  sub_block_stub_functions_ = std::list<GeneratedFunction*>();

  // IOChannel* pointers will be different in the sub-block context
  auto external_channels_by_internal_channel_caller =
      std::move(external_channels_by_internal_channel_);
  external_channels_by_internal_channel_ =
      absl::btree_multimap<const IOChannel*, ChannelBundle>();

  // The implementation function shares the same clang::NamedDecl as the
  // caller's stub function.
  CHECK(inst_functions_.contains(gen_func->clang_decl));
  std::unique_ptr<GeneratedFunction> caller_sub_function_caller =
      std::move(inst_functions_.at(gen_func->clang_decl));
  inst_functions_.erase(gen_func->clang_decl);

  XLS_RETURN_IF_ERROR(
      GenerateIR_Block(
          package,
          absl::StrCat(gen_func->clang_decl->getNameAsString(), "_proc"),
          top_channel_injections_by_state.begin()->second,
          /*this_type=*/std::shared_ptr<CType>(), /*this_decl=*/nullptr,
          /*top_decls=*/{}, GetLoc(*gen_func->clang_decl),
          top_level_init_interval,
          /*force_static=*/false,
          /*member_references_become_channels=*/false,
          /*caller_sub_function=*/gen_func)
          .status());

  // Restore caller's context
  inst_functions_[gen_func->clang_decl] = std::move(caller_sub_function_caller);
  unique_decl_ids_ = std::move(unique_decl_ids_caller);
  currently_generating_top_function_ = currently_generating_top_function_caller;
  sub_block_stub_functions_ = std::move(sub_block_stub_functions_caller);
  external_channels_by_internal_channel_ =
      std::move(external_channels_by_internal_channel_caller);
  return absl::OkStatus();
}

absl::StatusOr<xls::Proc*> Translator::GenerateIR_Block(
    xls::Package* package, std::string_view block_name,
    const absl::flat_hash_map<const clang::NamedDecl*, ChannelBundle>&
        top_channel_injections,
    const std::shared_ptr<CType>& this_type,
    const clang::CXXRecordDecl* this_decl,
    const std::list<ExternalChannelInfo>& top_decls,
    const xls::SourceInfo& body_loc, int top_level_init_interval,
    bool force_static, bool member_references_become_channels,
    const GeneratedFunction* caller_sub_function) {
  // Generate function without FIFO channel parameters
  // Force top function in block to be static.
  PreparedBlock prepared;

  XLS_ASSIGN_OR_RETURN(
      prepared.xls_func,
      GenerateIR_Top_Function(
          package,
          /*top_channel_injections=*/top_channel_injections, force_static,
          member_references_become_channels, top_level_init_interval,
          caller_sub_function != nullptr ? caller_sub_function->clang_decl
                                         : nullptr,
          "_impl"));

  for (GeneratedFunction* gen_func : sub_block_stub_functions_) {
    XLS_RETURN_IF_ERROR(
        GenerateIR_SubBlock(gen_func, package, top_level_init_interval));
  }

  xls::ProcBuilder pb(absl::StrCat(block_name, "_proc"), package);

  prepared.orig_token = pb.Literal(xls::Value::Token());
  prepared.token = prepared.orig_token;

  PushContextGuard pb_guard(*this, body_loc);

  XLS_ASSIGN_OR_RETURN(
      std::unique_ptr<GeneratedFunction> proc_func_generated_ownership,
      GenerateIRBlockPrepare(prepared, pb,
                             /*next_return_index=*/0, this_type, this_decl,
                             top_decls, caller_sub_function, body_loc));

  XLS_ASSIGN_OR_RETURN(
      GenerateFSMInvocationReturn fsm_ret,
      GenerateFSMInvocation(prepared, pb, /*nesting_level=*/0, body_loc));

  XLSCC_CHECK(
      fsm_ret.return_value.valid() && fsm_ret.returns_this_activation.valid(),
      body_loc);

  // Generate default ops for unused external channels
  // (Only in the top proc)
  if (caller_sub_function == nullptr) {
    XLS_RETURN_IF_ERROR(GenerateDefaultIOOps(prepared, pb, body_loc));
  }

  // Create next state values
  CHECK_GE(prepared.xls_func->return_value_count,
           prepared.xls_func->static_values.size());

  CHECK(context().fb == &pb);

  absl::btree_multimap<const xls::StateElement*, NextStateValue>
      next_state_values;

  std::vector<const clang::NamedDecl*> next_static_value_decls;

  if (this_decl != nullptr) {
    next_static_value_decls.push_back(this_decl);
  }

  for (const clang::NamedDecl* namedecl :
       prepared.xls_func->GetDeterministicallyOrderedStaticValues()) {
    next_static_value_decls.push_back(namedecl);
  }

  for (const clang::NamedDecl* namedecl : next_static_value_decls) {
    const int64_t ret_idx = prepared.return_index_for_static.at(namedecl);
    xls::BValue next_val =
        GetFlexTupleField(fsm_ret.return_value, ret_idx,
                          prepared.xls_func->return_value_count, body_loc);
    xls::StateElement* state_elem =
        prepared.state_element_for_variable.at(namedecl);
    xls::StateRead* state_read = pb.proc()->GetStateRead(state_elem);
    xls::BValue prev_val(state_read, &pb);

    XLS_ASSIGN_OR_RETURN(bool is_on_reset, DeclIsOnReset(namedecl));

    NextStateValue next_state_value = {.priority = 0,
                                       .extra_label = std::string{block_name}};

    if (!is_on_reset) {
      next_state_value.value = next_val;
      next_state_value.condition = fsm_ret.returns_this_activation;
    } else {
      next_state_value.value =
          pb.And(prev_val,
                 pb.Not(fsm_ret.returns_this_activation, body_loc,
                        /*name=*/"does_not_return_this_activation"),
                 body_loc, /*name=*/"next_on_reset");
    }
    next_state_values.insert({state_elem, next_state_value});
  }

  for (const auto& [state_elem, bval] : fsm_ret.extra_next_state_values) {
    next_state_values.insert({state_elem, bval});
  }

  return BuildWithNextStateValueMap(pb, prepared.token, next_state_values,
                                    body_loc);
}

absl::StatusOr<xls::Proc*> Translator::GenerateIR_BlockFromClass(
    xls::Package* package, HLSBlock* block_spec_out,
    int top_level_init_interval, const ChannelOptions& channel_options) {
  package_ = package;
  block_spec_out->Clear();

  // Create external channels
  const clang::FunctionDecl* top_function = nullptr;

  CHECK_NE(parser_.get(), nullptr);
  XLS_ASSIGN_OR_RETURN(top_function, parser_->GetTopFunction());

  if (!clang::isa<clang::CXXMethodDecl>(top_function)) {
    return absl::InvalidArgumentError(
        ErrorMessage(GetLoc(*top_function), "Top function %s isn't a method",
                     top_function->getQualifiedNameAsString().c_str()));
  }

  auto method = clang::dyn_cast<clang::CXXMethodDecl>(top_function);
  const clang::QualType& this_type = method->getThisType()->getPointeeType();

  CHECK(this_type->isRecordType());

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

  context().ast_context = &record_decl->getASTContext();

  XLS_RETURN_IF_ERROR(ScanStruct(record_decl));

  XLS_ASSIGN_OR_RETURN(
      shared_ptr<CType> this_ctype,
      TranslateTypeFromClang(this_type, GetLoc(*top_function)));

  XLS_ASSIGN_OR_RETURN(std::shared_ptr<CType> struct_ctype,
                       ResolveTypeInstance(this_ctype));

  auto struct_type = std::dynamic_pointer_cast<CStructType>(struct_ctype);
  CHECK_NE(struct_type, nullptr);

  block_spec_out->set_name(record_decl->getNameAsString());

  std::list<ExternalChannelInfo> top_decls;
  absl::flat_hash_map<std::string, xls::ChannelStrictness>
      unused_strictness_options = channel_options.strictness_map;
  for (const clang::FieldDecl* field_decl : record_decl->fields()) {
    std::shared_ptr<CField> field = struct_type->get_field(field_decl);
    std::shared_ptr<CType> field_type = field->type();

    // Only uninitialized are external channels
    if (field_decl->hasInClassInitializer()) {
      continue;
    }

    XLS_ASSIGN_OR_RETURN(std::shared_ptr<CType> resolved_field_type,
                         ResolveTypeInstanceDeeply(field->type()));

    if (auto channel_type =
            std::dynamic_pointer_cast<CChannelType>(field->type())) {
      if (channel_type->GetOpType() != OpType::kNull) {
        if (channel_type->GetOpType() == OpType::kSendRecv) {
          return absl::UnimplementedError(
              ErrorMessage(GetLoc(*field->name()),
                           "Internal (InOut) channels in top class"));
        }

        xlscc::HLSChannel* channel_spec = block_spec_out->add_channels();
        channel_spec->set_name(field->name()->getNameAsString());
        channel_spec->set_width_in_bits(resolved_field_type->GetBitWidth());
        channel_spec->set_type(xlscc::FIFO);
        channel_spec->set_is_input(channel_type->GetOpType() == OpType::kRecv);

        ExternalChannelInfo channel_info = {
            .decl = field->name(),
            .channel_type = channel_type,
            .interface_type = InterfaceType::kFIFO,
            .is_input = channel_type->GetOpType() == OpType::kRecv};
        XLS_ASSIGN_OR_RETURN(
            channel_info.strictness,
            GetChannelStrictness(*field->name(), channel_options,
                                 unused_strictness_options));
        top_decls.push_back(channel_info);
      } else if (channel_type->GetMemorySize() > 0) {
        xlscc::HLSChannel* channel_spec = block_spec_out->add_channels();
        channel_spec->set_name(field->name()->getNameAsString());
        channel_spec->set_width_in_bits(resolved_field_type->GetBitWidth());
        channel_spec->set_type(xlscc::MEMORY);
        channel_spec->set_depth(channel_type->GetMemorySize());

        ExternalChannelInfo channel_info = {
            .decl = field->name(),
            .channel_type = channel_type,
            .interface_type = InterfaceType::kMemory};
        XLS_ASSIGN_OR_RETURN(
            channel_info.strictness,
            GetChannelStrictness(*field->name(), channel_options,
                                 unused_strictness_options));
        top_decls.push_back(channel_info);
      } else {
        return absl::InvalidArgumentError(
            ErrorMessage(GetLoc(*field->name()),
                         "Direction or depth unspecified for external channel "
                         "or memory '%s'",
                         field->name()->getNameAsString().c_str()));
      }
    } else if (auto channel_type =
                   std::dynamic_pointer_cast<CReferenceType>(field->type())) {
      xlscc::HLSChannel* channel_spec = block_spec_out->add_channels();
      channel_spec->set_name(field->name()->getNameAsString());
      channel_spec->set_width_in_bits(resolved_field_type->GetBitWidth());
      channel_spec->set_type(xlscc::DIRECT_IN);
      channel_spec->set_is_input(true);

      ExternalChannelInfo channel_info = {
          .decl = field->name(),
          .channel_type =
              std::make_shared<CChannelType>(channel_type->GetPointeeType(),
                                             /*memory_size=*/-1),
          .interface_type = InterfaceType::kDirect,
          .is_input = true};
      XLS_ASSIGN_OR_RETURN(channel_info.strictness,
                           GetChannelStrictness(*field->name(), channel_options,
                                                unused_strictness_options));
      top_decls.push_back(channel_info);
    }
  }
  if (!unused_strictness_options.empty()) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Unused channel strictness options: %s",
                        ToString(unused_strictness_options)));
  }

  return GenerateIR_Block(package, *block_spec_out, this_ctype,
                          /*this_decl=*/record_decl, top_decls,
                          GetLoc(*record_decl), top_level_init_interval,
                          /*force_static=*/false,
                          /*member_references_become_channels=*/true);
}

absl::Status Translator::GenerateDefaultIOOp(
    xls::Channel* channel, bool is_send, xls::BValue token,
    std::vector<xls::BValue>& final_tokens, xls::ProcBuilder& pb,
    const xls::SourceInfo& loc) {
  xls::BValue final_token;

  xls::BValue pred_0 = pb.Literal(xls::UBits(0, 1), loc);

  xls::Value data_0_val = xls::ZeroOfType(channel->type());
  xls::BValue data_0 = pb.Literal(data_0_val, loc);

  if (!is_send) {
    XLSCC_CHECK(channel->CanReceive(), loc);
    xls::BValue tup = pb.ReceiveIf(
        channel, token, pred_0, loc,
        /*name=*/absl::StrFormat("%s_default_op", channel->name()));
    final_token = pb.TupleIndex(
        tup, 0, loc,
        /*name=*/absl::StrFormat("%s_default_op_token", channel->name()));
  } else if (is_send) {
    XLSCC_CHECK(channel->CanSend(), loc);
    final_token =
        pb.SendIf(channel, token, pred_0, data_0, loc,
                  /*name=*/absl::StrFormat("%s_default_op", channel->name()));
  } else {
    return absl::UnimplementedError(ErrorMessage(
        loc, "Don't know how to create default IO op for channel %s",
        channel->name()));
  }
  final_tokens.push_back(token);
  return absl::OkStatus();
}

absl::Status Translator::GenerateDefaultIOOps(PreparedBlock& prepared,
                                              xls::ProcBuilder& pb,
                                              const xls::SourceInfo& body_loc) {
  if (unused_xls_channel_ops_.empty()) {
    return absl::OkStatus();
  }

  std::vector<xls::BValue> final_tokens = {prepared.token};

  for (const auto& [channel, is_send] : unused_xls_channel_ops_) {
    XLS_RETURN_IF_ERROR(GenerateDefaultIOOp(
        channel, is_send, prepared.orig_token, final_tokens, pb, body_loc));
  }

  prepared.token =
      pb.AfterAll(final_tokens, body_loc, /*name=*/"after_default_io_ops");

  return absl::OkStatus();
}

absl::StatusOr<Translator::LayoutFSMStatesReturn> Translator::LayoutFSMStates(
    PreparedBlock& prepared, xls::ProcBuilder& pb,
    const xls::SourceInfo& body_loc) {
  absl::flat_hash_map<const IOOp*, const State*> state_by_io_op;
  std::vector<std::unique_ptr<State>> states;

  bool has_pipelined_loop = false;

  states.push_back(std::make_unique<State>());
  states.back()->index = states.size() - 1;

  for (const IOOp& op : prepared.xls_func->io_ops) {
    if (op.scheduling_option != xlscc::IOSchedulingOption::kNone) {
      return absl::UnimplementedError(
          absl::StrFormat("Generating IO ops with scheduling options (ASAP "
                          "etc) not yet supported"));
    }

    const PipelinedLoopSubProc* sub_proc = nullptr;

    if (prepared.xls_func->pipeline_loops_by_internal_channel.contains(
            op.channel)) {
      sub_proc =
          prepared.xls_func->pipeline_loops_by_internal_channel.at(op.channel);
      XLSCC_CHECK(sub_proc != nullptr, body_loc);
    }

    bool add_state = false;

    // Decide whether or not to add a state for a pipelined loop.
    if (generate_fsms_for_pipelined_loops_) {
      if (sub_proc != nullptr &&
          op.scheduling_option != IOSchedulingOption::kNone) {
        return absl::UnimplementedError(
            absl::StrFormat("Generating FSMs for pipelined loops with "
                            "scheduling options (ASAP etc)"));
      }
      if (merge_states_) {
        // Every pipelined loop goes in its own state
        if (sub_proc != nullptr && op.op == OpType::kSend &&
            states.back()->sub_proc != nullptr) {
          add_state = true;
        }
      } else {
        // Put regular IOs after a pipelined loop into their own state.
        // (such as another IO op not associated with a pipelined loop, null)
        if (sub_proc == nullptr && states.back()->sub_proc != nullptr) {
          add_state = true;
        }

        // Put each pipelined loop into its own state.
        // Subroutines can mean that there are multiple with the same statement.
        // (sub_proc != null only for pipelined loops)
        if (sub_proc != nullptr && op.op == OpType::kSend &&
            // Don't add a blank state
            (!states.back()->invokes_to_generate.empty() ||
             states.back()->sub_proc != nullptr)) {
          add_state = true;
        }
      }
      // If this mode is enabled, then don't allow ops on the same channel
      // in the same state, including in pipelined loop bodies
      if (split_states_on_channel_ops_) {
        std::set<ChannelBundle> channels_used_by_invoke =
            GetChannelsUsedByOp(op, sub_proc, body_loc);
        const std::set<ChannelBundle>& channels_used_by_state =
            states.back()->channels_used;
        std::set<ChannelBundle> channels_used_by_both;
        std::set_intersection(
            channels_used_by_invoke.begin(), channels_used_by_invoke.end(),
            channels_used_by_state.begin(), channels_used_by_state.end(),
            std::inserter(channels_used_by_both,
                          channels_used_by_both.begin()));

        if (!channels_used_by_both.empty()) {
          add_state = true;
        }
      }
    }

    if (add_state) {
      states.push_back(std::make_unique<State>());
      states.back()->index = states.size() - 1;
    }

    if (sub_proc) {
      states.back()->sub_proc = sub_proc;
      has_pipelined_loop = true;
    }

    XLSCC_CHECK(!states.empty(), body_loc);

    // Check that after_ops are not violated by the ordering of the states
    // state_by_io_op contains ops from previous states at this point
    for (const IOOp* after_op : op.after_ops) {
      XLSCC_CHECK(state_by_io_op.contains(after_op), body_loc);
    }

    states.back()->invokes_to_generate.push_back(
        {.op = op, .extra_condition = xls::BValue()});

    std::set<ChannelBundle> for_op =
        GetChannelsUsedByOp(op, sub_proc, body_loc);
    states.back()->channels_used.insert(for_op.begin(), for_op.end());

    state_by_io_op[&op] = states.back().get();
  }

  XLSCC_CHECK(!states.empty(), body_loc);

  if (debug_ir_trace_flags_ & DebugIrTraceFlags_FSMStates) {
    LOG(INFO) << absl::StrFormat("States for %s at %s:\n",
                                 prepared.xls_func->xls_func->name().c_str(),
                                 LocString(body_loc).c_str());
    for (const std::unique_ptr<State>& state : states) {
      LOG(INFO) << absl::StrFormat("-- state %li: sub_proc %p\n", state->index,
                                   state->sub_proc);
      for (const InvokeToGenerate& invoke : state->invokes_to_generate) {
        LOG(INFO) << absl::StrFormat(
            "---- invoke ch %s at %s\n",
            invoke.op.channel ? invoke.op.channel->unique_name.c_str()
                              : "(null)",
            LocString(invoke.op.op_location).c_str());
      }
    }
  }

  return LayoutFSMStatesReturn{.state_by_io_op = std::move(state_by_io_op),
                               .states = std::move(states),
                               .has_pipelined_loop = has_pipelined_loop};
}

absl::StatusOr<Translator::GenerateFSMInvocationReturn>
Translator::GenerateFSMInvocation(PreparedBlock& prepared, xls::ProcBuilder& pb,
                                  int nesting_level,
                                  const xls::SourceInfo& body_loc) {
  // Create a deterministic ordering for the last state elements
  // (These store the received inputs for IO operations)
  std::vector<int64_t> arg_indices_ordered_by_state_elems;
  std::vector<const IOOp*> io_ops_with_args_ordered;

  for (const IOOp& op : prepared.xls_func->io_ops) {
    // Don't copy direct-ins, statics, etc into FSM state
    if (!prepared.arg_index_for_op.contains(&op)) {
      continue;
    }
    // Don't copy context in/out for pipelined loops into the FSM state
    if (prepared.xls_func->pipeline_loops_by_internal_channel.contains(
            op.channel)) {
      continue;
    }

    arg_indices_ordered_by_state_elems.push_back(
        prepared.arg_index_for_op.at(&op));
    io_ops_with_args_ordered.push_back(&op);
  }

  // Lay out the states for this FSM
  XLS_ASSIGN_OR_RETURN(LayoutFSMStatesReturn layout_states_return,
                       LayoutFSMStates(prepared, pb, body_loc));

  absl::flat_hash_map<const IOOp*, const State*> state_by_io_op =
      std::move(layout_states_return.state_by_io_op);
  std::vector<std::unique_ptr<State>> states =
      std::move(layout_states_return.states);
  bool has_pipelined_loop = layout_states_return.has_pipelined_loop;

  const std::string fsm_prefix =
      absl::StrFormat("__fsm_%s", prepared.xls_func->xls_func->name());

  xls::BValue changed_state_last_activation;
  const bool states_can_have_multiple_parts =
      merge_states_ && has_pipelined_loop;

  if (states_can_have_multiple_parts) {
    xls::Value initial_exited_last_activation = xls::Value(xls::UBits(1, 1));
    changed_state_last_activation = pb.StateElement(
        absl::StrFormat("%s_exited_last_activation", fsm_prefix),
        initial_exited_last_activation, body_loc);
  }

  // Set up state with wrap-around, if needed
  xls::BValue state_index;
  int64_t state_bits = 0;
  xls::Value initial_state_index;
  xls::Type* args_type;
  xls::BValue args_from_last_state;
  xls::Value initial_args_val;

  xls::BValue state_read;

  // In merge states mode, FSM state elements are always created for storing
  // IO op inputs
  if (states.size() > 1 || states_can_have_multiple_parts) {
    prepared.contains_fsm = true;
    // Allow for valid IR with 1 state with 1 bit index
    state_bits = xls::CeilOfLog2(1 + states.size());
    xls::Type* state_type = package_->GetBitsType(state_bits);
    initial_state_index = xls::ZeroOfType(state_type);
    std::vector<xls::Type*> args_types;
    args_types.reserve(arg_indices_ordered_by_state_elems.size());
    for (int64_t arg_idx : arg_indices_ordered_by_state_elems) {
      args_types.push_back(prepared.args.at(arg_idx).GetType());
    }
    args_type = package_->GetTupleType(args_types);
    initial_args_val = xls::ZeroOfType(args_type);
    std::vector<xls::Value> initial_state_elements = {initial_state_index,
                                                      initial_args_val};

    xls::Value initial_state = xls::Value::Tuple(initial_state_elements);
    state_read = pb.StateElement(absl::StrFormat("%s_state", fsm_prefix),
                                 initial_state, body_loc);
    state_index =
        pb.TupleIndex(state_read, /*idx=*/0, body_loc,
                      /*name=*/absl::StrFormat("%s_state_index", fsm_prefix));

    args_from_last_state =
        pb.TupleIndex(state_read, /*idx=*/1, body_loc,
                      /*name=*/absl::StrFormat("%s_state_args", fsm_prefix));
  }

  xls::BValue origin_token = prepared.token;
  std::vector<xls::BValue> sink_tokens;

  std::vector<xls::BValue> next_args_by_state;
  next_args_by_state.resize(states.size());

  std::vector<xls::BValue> go_to_next_state_by_state;
  go_to_next_state_by_state.resize(states.size(),
                                   pb.Literal(xls::UBits(1, 1), body_loc));

  xls::BValue last_ret_val;

  auto set_args_for_state =
      [this, &prepared, &pb, io_ops_with_args_ordered, state_by_io_op](
          const State& state, auto apply_to_state_elem,
          const xls::BValue& args_from_last_state, const xls::SourceInfo& loc,
          const std::list<InvokeToGenerate>* filter_list =
              nullptr) -> absl::Status {
    XLSCC_CHECK(args_from_last_state.valid(), loc);

    absl::flat_hash_set<const IOOp*> filter_list_set;
    if (filter_list != nullptr) {
      for (const InvokeToGenerate& filter_invoke : *filter_list) {
        filter_list_set.insert(&filter_invoke.op);
      }
    }

    for (int64_t state_elem_idx = 0;
         state_elem_idx < io_ops_with_args_ordered.size(); ++state_elem_idx) {
      const IOOp* op_ptr = io_ops_with_args_ordered[state_elem_idx];
      if (state_by_io_op.at(op_ptr)->index != state.index) {
        continue;
      }
      if (filter_list != nullptr && !filter_list_set.contains(op_ptr)) {
        continue;
      }
      const int64_t arg_idx = prepared.arg_index_for_op.at(op_ptr);
      prepared.args[arg_idx] = apply_to_state_elem(
          pb,
          pb.TupleIndex(
              args_from_last_state, /*idx=*/state_elem_idx, loc, /*name=*/
              absl::StrFormat("arg_from_last_state_%s", Debug_OpName(*op_ptr))),
          op_ptr, loc);
    }
    return absl::OkStatus();
  };

  absl::btree_multimap<const xls::StateElement*, NextStateValue>
      sub_fsm_next_state_values;

  absl::flat_hash_set<const xls::Node*> prev_state_io_nodes;

  absl::flat_hash_map<const IOOp*, xls::BValue> op_tokens;

  // Generate IR for the individual states
  for (std::unique_ptr<State>& state : states) {
    XLSCC_CHECK_GE(state->index, 0, body_loc);

    auto last_node_iter = pb.proc()->nodes().begin();

    if (debug_ir_trace_flags_ & DebugIrTraceFlags_PrevStateIOReferences) {
      // How to do this better? -- not supported
      for (auto it = last_node_iter;; ++last_node_iter) {
        ++it;
        if (it == pb.proc()->nodes().end()) {
          break;
        }
      }
    }

    // Set all op tokens from previous states to the input of this state
    absl::flat_hash_map<const IOOp*, xls::BValue> op_tokens_prev = op_tokens;
    for (auto [op, _] : op_tokens_prev) {
      CHECK_NE(prepared.orig_token.node(), nullptr);
      op_tokens[op] = prepared.orig_token;
    }

    // If generating state machine, add extra predicates to IO ops
    // and save values from previous states
    if (state_index.valid()) {
      XLSCC_CHECK_GT(state_bits, 0, body_loc);
      xls::BValue this_state_index = pb.Literal(
          xls::UBits(state->index, state_bits), body_loc,
          absl::StrFormat("%s_state_%i_index", fsm_prefix, state->index));
      state->in_this_state = pb.Eq(
          state_index, this_state_index, body_loc,
          /*name=*/absl::StrFormat("%s_in_state_%i", fsm_prefix, state->index));

      for (InvokeToGenerate& invoke : state->invokes_to_generate) {
        invoke.extra_condition = state->in_this_state;
      }
    } else {
      state->in_this_state = pb.Literal(xls::UBits(1, 1), body_loc);
    }

    state->in_this_state =
        pb.And(state->in_this_state, context().full_condition_bval(body_loc),
               body_loc);

    // The function is first invoked with defaults for any
    //  read() IO Ops.
    // If there are any read() IO Ops, then it will be invoked again
    //  for each read Op below.
    // Statics don't need to generate additional invokes, since they need not
    //  exchange any data with the outside world between iterations.
    CHECK_EQ(prepared.args.size(),
             prepared.xls_func->xls_func->params().size());
    for (const xls::BValue& arg : prepared.args) {
      CHECK(arg.valid());
    }
    last_ret_val =
        pb.Invoke(prepared.args, prepared.xls_func->xls_func, body_loc,
                  /*name=*/
                  absl::StrFormat("%s_state_%i_default_invoke", fsm_prefix,
                                  state->index));
    XLSCC_CHECK(last_ret_val.valid(), body_loc);

    // States should be in parallel in the token graph
    prepared.token = origin_token;

    std::list<InvokeToGenerate> invokes_before_loop;
    std::list<InvokeToGenerate> invokes_for_loop;
    std::list<InvokeToGenerate> invokes_after_loop;

    auto invoke_it = state->invokes_to_generate.begin();

    for (; invoke_it != state->invokes_to_generate.end(); ++invoke_it) {
      if (generate_fsms_for_pipelined_loops_ &&
          prepared.xls_func->pipeline_loops_by_internal_channel.contains(
              invoke_it->op.channel)) {
        break;
      }
      invokes_before_loop.push_back(*invoke_it);
    }
    for (; invoke_it != state->invokes_to_generate.end(); ++invoke_it) {
      if (!prepared.xls_func->pipeline_loops_by_internal_channel.contains(
              invoke_it->op.channel)) {
        break;
      }
      invokes_for_loop.push_back(*invoke_it);
    }
    for (; invoke_it != state->invokes_to_generate.end(); ++invoke_it) {
      invokes_after_loop.push_back(*invoke_it);
    }

    XLSCC_CHECK(generate_fsms_for_pipelined_loops_ || invokes_for_loop.empty(),
                body_loc);
    XLSCC_CHECK(
        generate_fsms_for_pipelined_loops_ || invokes_after_loop.empty(),
        body_loc);

    xls::BValue state_token_out = prepared.token;
    xls::BValue first_iter = pb.Literal(xls::UBits(1, 1), body_loc);

    if (!invokes_before_loop.empty()) {
      PushContextGuard context_guard(*this, first_iter, body_loc);

      XLS_ASSIGN_OR_RETURN(
          state_token_out,
          GenerateIOInvokesWithAfterOps(
              IOSchedulingOption::kNone, state_token_out, invokes_before_loop,
              op_tokens, last_ret_val, prepared, pb, body_loc));

      XLSCC_CHECK(last_ret_val.valid(), body_loc);
      XLSCC_CHECK(state_token_out.valid(), body_loc);

      // Insert selects for prepared.args for later activations of this state
      if (states_can_have_multiple_parts) {
        // Check that changed_state_last_activation is safe to use
        XLSCC_CHECK(changed_state_last_activation.valid(), body_loc);
        XLSCC_CHECK(changed_state_last_activation.node()->Is<xls::StateRead>(),
                    body_loc);

        // This and is not strictly, logically necessary, however:
        // - It gives the compiler the information to remove unnecessary
        //   data dependencies.
        // - It should ensure that the logic remains inactive when its state
        //   is not active, which may save power.
        xls::BValue before_loop_args_selector = pb.And(
            changed_state_last_activation, state->in_this_state, body_loc,
            /*name=*/
            absl::StrFormat("%s_state_%i_before_loop_args_selector", fsm_prefix,
                            state->index));

        auto select_elem =
            [this, before_loop_args_selector, prepared, fsm_prefix, &state](
                xls::ProcBuilder& pb, const xls::BValue& state_elem,
                const IOOp* op_ptr, const xls::SourceInfo& loc) -> xls::BValue {
          const int64_t arg_idx = prepared.arg_index_for_op.at(op_ptr);
          xls::BValue ret = pb.Select(
              before_loop_args_selector,
              /*on_true=*/prepared.args.at(arg_idx),
              /*on_false=*/state_elem, loc,
              /*name=*/
              absl::StrFormat("%s_state_%i_select_%s_before_loop", fsm_prefix,
                              state->index, Debug_OpName(*op_ptr)));
          XLSCC_CHECK(ret.valid(), loc);
          return ret;
        };
        // Filter to produce less dead code
        XLS_RETURN_IF_ERROR(set_args_for_state(
            *state, select_elem, args_from_last_state, body_loc,
            /*filter_list=*/&invokes_before_loop));
      }
    }

    xls::BValue last_iter;

    if (!invokes_for_loop.empty()) {
      if (!invokes_before_loop.empty() && states_can_have_multiple_parts) {
        // Update inputs to loop FSM based on before IO op updates to
        // prepared.args
        last_ret_val = pb.Invoke(
            prepared.args, prepared.xls_func->xls_func, body_loc,
            /*name=*/
            absl::StrFormat("%s_state_%i_default_invoke_after_before_ops",
                            fsm_prefix, state->index));
        XLSCC_CHECK(last_ret_val.valid(), body_loc);
      }

      XLS_ASSIGN_OR_RETURN(
          SubFSMReturn sub_fsm_ret,
          GenerateSubFSM(prepared, invokes_for_loop, state_token_out, pb,
                         *state, fsm_prefix, op_tokens, last_ret_val,
                         /*outer_nesting_level=*/nesting_level, body_loc));

      for (const auto& [state_elem, bval] :
           sub_fsm_ret.extra_next_state_values) {
        sub_fsm_next_state_values.insert({state_elem, bval});
      }

      state_token_out = sub_fsm_ret.token_out;
      last_ret_val = sub_fsm_ret.return_value;

      if (!invokes_before_loop.empty() && states_can_have_multiple_parts) {
        xls::BValue first_iter_literal = first_iter;

        // The first_iter flag is a state element. It's safe to use,
        // as it can't depend on side effecting ops earlier in the token chain,
        // and can't create cycles.
        XLSCC_CHECK(sub_fsm_ret.first_iter.valid(), body_loc);
        XLSCC_CHECK(sub_fsm_ret.first_iter.node()->Is<xls::StateRead>(),
                    body_loc);
        XLSCC_CHECK(changed_state_last_activation.valid(), body_loc);
        XLSCC_CHECK(changed_state_last_activation.node()->Is<xls::StateRead>(),
                    body_loc);

        // With nested loops, the outer loop may stay in its first iteration
        // for several activations. However, the IO ops before the loop should
        // only be active in the first one.
        first_iter = pb.And(sub_fsm_ret.first_iter,
                            changed_state_last_activation, body_loc,
                            /*name=*/
                            absl::StrFormat("%s_state_%i_first_iter",
                                            fsm_prefix, state->index));

        XLSCC_CHECK(first_iter.valid(), body_loc);

        // Set condition for IO ops before loop body
        XLS_RETURN_IF_ERROR(
            first_iter_literal.node()->ReplaceUsesWith(first_iter.node()));
      }

      last_iter = sub_fsm_ret.exit_state_condition;

      XLSCC_CHECK(last_ret_val.valid(), body_loc);

      go_to_next_state_by_state[state->index] =
          sub_fsm_ret.exit_state_condition;

      if (generate_fsms_for_pipelined_loops_ &&
          (debug_ir_trace_flags_ & DebugIrTraceFlags_LoopControl)) {
        // op_tokens will skip path to sink
        xls::BValue literal_1 = pb.Literal(xls::UBits(1, 1), body_loc);
        xls::BValue trace_out;
        if (states_can_have_multiple_parts) {
          trace_out = pb.Trace(
              state_token_out, literal_1,
              /*args=*/
              {first_iter, sub_fsm_ret.first_iter,
               sub_fsm_ret.exit_state_condition, changed_state_last_activation,
               context().full_condition_bval(body_loc)},
              absl::StrFormat(
                  "-- %s state %i first_iter {:u} fsm.first {:u} "
                  "fsm.exit {:u} changed_state_last_activation {:u} cond {:u}",
                  fsm_prefix, state->index),
              /*verbosity=*/0, body_loc);
        } else {
          trace_out = pb.Trace(
              state_token_out, literal_1,
              /*args=*/
              {first_iter, sub_fsm_ret.first_iter,
               sub_fsm_ret.exit_state_condition,
               context().full_condition_bval(body_loc)},
              absl::StrFormat("-- %s state %i first_iter {:u} fsm.first {:u} "
                              "fsm.exit {:u} cond {:u}",
                              fsm_prefix, state->index),
              /*verbosity=*/0, body_loc);
        }
        XLSCC_CHECK(trace_out.valid(), body_loc);
        sink_tokens.push_back(trace_out);
      }

      // GenerateSubFSM() updates prepared.args, and the loop sub-FSM
      // keeps its outputs steady, so no need to update here
    }

    if (!invokes_after_loop.empty()) {
      PushContextGuard context_guard(*this, last_iter, body_loc);

      XLS_ASSIGN_OR_RETURN(
          state_token_out,
          GenerateIOInvokesWithAfterOps(
              IOSchedulingOption::kNone, state_token_out, invokes_after_loop,
              op_tokens, last_ret_val, prepared, pb, body_loc));

      XLSCC_CHECK(last_ret_val.valid(), body_loc);
      XLSCC_CHECK(state_token_out.valid(), body_loc);
    }

    sink_tokens.push_back(state_token_out);

    // Save return values for this state
    if (state_index.valid()) {
      XLSCC_CHECK(state->in_this_state.valid(), body_loc);
      std::vector<xls::BValue> next_args_by_state_elems;
      next_args_by_state_elems.reserve(
          arg_indices_ordered_by_state_elems.size());
      for (int64_t arg_idx : arg_indices_ordered_by_state_elems) {
        next_args_by_state_elems.push_back(prepared.args.at(arg_idx));
      }
      next_args_by_state.at(state->index) =
          pb.Tuple(next_args_by_state_elems, body_loc);
    }

    // Check for references to data from IO ops in previous states
    if (debug_ir_trace_flags_ & DebugIrTraceFlags_PrevStateIOReferences) {
      // Save, since prev_state_io_nodes is used in the loop
      auto next_prev_state_io_nodes = prev_state_io_nodes;

      for (; last_node_iter != pb.proc()->nodes().end(); ++last_node_iter) {
        const xls::Node* node = *last_node_iter;

        auto opt_path =
            Debug_DeeplyCheckOperandsFromPrev(node, prev_state_io_nodes);
        if (opt_path.has_value()) {
          LOG(WARNING) << absl::StrFormat(
              "This node in state %i references previous state's IO node\n",
              state->index);
          auto full_path = opt_path.value();
          full_path.push_front(node);
          for (const xls::Node* path_node : full_path) {
            LOG(WARNING) << absl::StrFormat("---- %s: %s\n",
                                            path_node->GetName(),
                                            xls::OpToString(path_node->op()));
          }
        }

        if (node->Is<xls::Receive>()) {
          next_prev_state_io_nodes.insert(node);
        }
      }

      prev_state_io_nodes = next_prev_state_io_nodes;
    }

    // Set args to state elements for later states to use
    // (With only 1 state, FSM state elements will not exist)
    if (states.size() > 1) {
      auto pass_through_elem =
          [](xls::ProcBuilder& pb, const xls::BValue& state_elem,
             const IOOp* op_ptr,
             const xls::SourceInfo& loc) -> xls::BValue { return state_elem; };
      XLS_RETURN_IF_ERROR(set_args_for_state(*state, pass_through_elem,
                                             args_from_last_state, body_loc));
    }
  }

  XLSCC_CHECK(!sink_tokens.empty(), body_loc);

  prepared.token =
      pb.AfterAll(sink_tokens, body_loc, /*name=*/"after_sink_tokens");

  xls::BValue returns_this_activation_vars = go_to_next_state_by_state.at(0);

  absl::btree_multimap<const xls::StateElement*, NextStateValue>
      fsm_next_state_values;

  xls::BValue go_to_next_state;

  // Construct the next FSM state
  if (state_index.valid()) {
    xls::BValue final_state_index =
        pb.Literal(xls::UBits(states.size() - 1, state_bits), body_loc,
                   absl::StrFormat("%s_final_state_index", fsm_prefix));
    xls::BValue in_last_state =
        pb.Eq(state_index, final_state_index, body_loc,
              absl::StrFormat("%s_in_final_state", fsm_prefix));
    xls::BValue state_one =
        pb.Literal(xls::UBits(1, state_bits), body_loc,
                   absl::StrFormat("%s_state_one", fsm_prefix));

    xls::BValue following_state_index = pb.Select(
        in_last_state,
        /*on_true=*/
        pb.Literal(initial_state_index, body_loc,
                   absl::StrFormat("%s_initial_state_index", fsm_prefix)),
        /*on_false=*/
        pb.Add(state_index, state_one, body_loc,
               absl::StrFormat("%s_state_plus_one", fsm_prefix)));

    std::optional<xls::BValue> default_go_to_next_state = std::nullopt;
    if (next_args_by_state.size() < (1 << state_bits)) {
      default_go_to_next_state =
          pb.Literal(xls::UBits(0, 1), body_loc,
                     absl::StrFormat("%s_default_next_state", fsm_prefix));
    }

    xls::BValue go_to_next_state_in_state = pb.Select(
        state_index,
        /*cases=*/go_to_next_state_by_state, default_go_to_next_state, body_loc,
        /*name=*/absl::StrFormat("%s_go_to_next_state_in_state", fsm_prefix));

    go_to_next_state =
        pb.And(go_to_next_state_in_state,
               context().full_condition_bval(body_loc), body_loc,
               /*name=*/absl::StrFormat("%s_go_to_next_state", fsm_prefix));

    xls::BValue next_state_index =
        pb.Select(go_to_next_state,
                  /*on_true=*/
                  following_state_index,
                  /*on_false=*/
                  state_index, body_loc,
                  /*name=*/absl::StrFormat("%s_next_state_index", fsm_prefix));

    returns_this_activation_vars =
        pb.And(in_last_state, go_to_next_state, body_loc,
               /*name=*/
               absl::StrFormat("%s_returns_this_activation_vars", fsm_prefix));

    std::optional<xls::BValue> initial_args_bval = std::nullopt;
    if (next_args_by_state.size() < (1 << state_bits)) {
      initial_args_bval =
          pb.Literal(initial_args_val, body_loc,
                     absl::StrFormat("%s_initial_args", fsm_prefix));
    }
    xls::BValue args_from_this_state =
        pb.Select(state_index,
                  /*cases=*/next_args_by_state,
                  /*default_value=*/initial_args_bval, body_loc,
                  absl::StrFormat("%s_next_args", fsm_prefix));

    std::vector<xls::BValue> next_state_elements = {next_state_index,
                                                    args_from_this_state};

    xls::BValue next_tuple = pb.Tuple(next_state_elements, body_loc);

    fsm_next_state_values.insert(
        {state_read.node()->As<xls::StateRead>()->state_element(),
         NextStateValue{.value = next_tuple}});
  } else {
    go_to_next_state = returns_this_activation_vars;
  }

  if (states_can_have_multiple_parts) {
    XLSCC_CHECK(returns_this_activation_vars.valid(), body_loc);
    // Only change changed last activation when active
    fsm_next_state_values.insert(
        {changed_state_last_activation.node()
             ->As<xls::StateRead>()
             ->state_element(),
         NextStateValue{
             .value =
                 pb.Select(context().full_condition_bval(body_loc),
                           /*on_true=*/go_to_next_state,
                           /*on_false=*/changed_state_last_activation, body_loc,
                           /*name=*/
                           absl::StrFormat("%s_next_exited_last_activation",
                                           fsm_prefix))}});
  }

  for (const auto& [state_elem, bval] : sub_fsm_next_state_values) {
    fsm_next_state_values.insert({state_elem, bval});
  }

  if (generate_fsms_for_pipelined_loops_ &&
      (debug_ir_trace_flags_ & DebugIrTraceFlags_LoopControl)) {
    xls::BValue literal_1 = pb.Literal(xls::UBits(1, 1), body_loc);
    if (state_index.valid()) {
      prepared.token = pb.Trace(
          prepared.token, literal_1,
          /*args=*/
          {state_index, returns_this_activation_vars,
           context().full_condition_bval(body_loc)},
          absl::StrFormat("-- %s state_index {:u} fsm_ret {:u} cond {:u}",
                          fsm_prefix),
          /*verbosity=*/0, body_loc);
    } else {
      prepared.token =
          pb.Trace(prepared.token, literal_1,
                   /*args=*/
                   {returns_this_activation_vars,
                    context().full_condition_bval(body_loc)},
                   absl::StrFormat("-- %s fsm_ret {:u} cond {:u}", fsm_prefix),
                   /*verbosity=*/0, body_loc);
    }

    XLSCC_CHECK(prepared.token.valid(), body_loc);
  }

  XLSCC_CHECK(last_ret_val.valid(), body_loc);

  return GenerateFSMInvocationReturn{
      .return_value = last_ret_val,
      .returns_this_activation = returns_this_activation_vars,
      .extra_next_state_values = fsm_next_state_values};
}

std::set<ChannelBundle> Translator::GetChannelsUsedByOp(
    const IOOp& op, const PipelinedLoopSubProc* sub_proc,
    const xls::SourceInfo& loc) {
  if (sub_proc != nullptr) {
    if (op.op != OpType::kSend) {
      // Don't split in the middle of a pipelined loop invocation
      return {};
    }

    std::set<ChannelBundle> ret;
    for (const IOOp& op : sub_proc->generated_func->io_ops) {
      const PipelinedLoopSubProc* sub_sub_proc = nullptr;

      if (sub_proc->generated_func->pipeline_loops_by_internal_channel.contains(
              op.channel)) {
        sub_sub_proc =
            sub_proc->generated_func->pipeline_loops_by_internal_channel.at(
                op.channel);
        XLSCC_CHECK(sub_sub_proc != nullptr, loc);
      }

      std::set<ChannelBundle> for_op =
          GetChannelsUsedByOp(op, sub_sub_proc, loc);
      ret.insert(for_op.begin(), for_op.end());
    }
    return ret;
  }
  std::optional<ChannelBundle> bundle_for_op = GetChannelBundleForOp(op, loc);
  if (!bundle_for_op.has_value()) {
    return {};
  }
  return {bundle_for_op.value()};
}

absl::StatusOr<Translator::SubFSMReturn> Translator::GenerateSubFSM(
    PreparedBlock& outer_prepared,
    const std::list<Translator::InvokeToGenerate>& invokes_to_generate,
    xls::BValue origin_token, xls::ProcBuilder& pb, const State& outer_state,
    const std::string& fsm_prefix,
    absl::flat_hash_map<const IOOp*, xls::BValue>& op_tokens,
    xls::BValue first_ret_val, int outer_nesting_level,
    const xls::SourceInfo& body_loc) {
  XLSCC_CHECK(outer_state.in_this_state.valid(), body_loc);

  const int nesting_level = outer_nesting_level + 1;

  // Find the sub proc for which to generate the sub FSM
  const PipelinedLoopSubProc* sub_proc_invoked = outer_state.sub_proc;

  XLSCC_CHECK(sub_proc_invoked != nullptr, body_loc);

  // Check that the invokes in this state are appropriate
  XLSCC_CHECK_EQ(invokes_to_generate.size(), 2, body_loc);
  auto it = invokes_to_generate.begin();
  const InvokeToGenerate& invoke_context_out = *(it++);
  XLSCC_CHECK(invoke_context_out.op.channel->generated.has_value(), body_loc);
  XLSCC_CHECK(sub_proc_invoked->context_out_channel->generated.has_value(),
              body_loc);
  XLSCC_CHECK_EQ(invoke_context_out.op.channel->generated.value(),
                 sub_proc_invoked->context_out_channel->generated.value(),
                 body_loc);
  const InvokeToGenerate& invoke_context_in = *(it++);
  XLSCC_CHECK(invoke_context_in.op.channel->generated.has_value(), body_loc);
  XLSCC_CHECK(sub_proc_invoked->context_in_channel->generated.has_value(),
              body_loc);
  XLSCC_CHECK_EQ(invoke_context_in.op.channel->generated.value(),
                 sub_proc_invoked->context_in_channel->generated.value(),
                 body_loc);

  const int64_t context_out_ret_idx =
      outer_prepared.return_index_for_op.at(&invoke_context_out.op);

  xls::BValue ret_io_value =
      GetFlexTupleField(first_ret_val, context_out_ret_idx,
                        outer_prepared.xls_func->return_value_count, body_loc);

  xls::BValue context_out = pb.TupleIndex(
      ret_io_value, /*idx=*/0, body_loc, /*name=*/
      absl::StrFormat("%s_context_out", sub_proc_invoked->name_prefix));
  xls::BValue enter_condition = pb.TupleIndex(
      ret_io_value, /*idx=*/1, body_loc, /*name=*/
      absl::StrFormat("%s_enter_condition", sub_proc_invoked->name_prefix));
  CHECK_EQ(enter_condition.GetType()->GetFlatBitCount(), 1);

  if (generate_fsms_for_pipelined_loops_ &&
      (debug_ir_trace_flags_ & DebugIrTraceFlags_LoopControl)) {
    xls::BValue literal_1 = pb.Literal(xls::UBits(1, 1), body_loc);
    origin_token =
        pb.Trace(origin_token, /*condition=*/literal_1,
                 /*args=*/{outer_state.in_this_state, enter_condition},
                 /*format_string=*/
                 absl::StrFormat("SubFSM %s outer in state {:u} enter {:u}",
                                 sub_proc_invoked->name_prefix));
  }

  // Generate inner FSM
  XLS_ASSIGN_OR_RETURN(
      PipelinedLoopContentsReturn contents_ret,
      GenerateIR_PipelinedLoopContents(
          *sub_proc_invoked, pb, origin_token, context_out,
          /*in_state_condition=*/
          pb.And(outer_state.in_this_state, enter_condition, body_loc,
                 /*name=*/
                 absl::StrFormat("%s_loop_contents_condition",
                                 sub_proc_invoked->name_prefix)),
          /*in_fsm=*/true,
          /*state_element_for_variable=*/
          &outer_prepared.state_element_for_variable,
          /*nesting_level=*/nesting_level));

  xls::BValue context_in = contents_ret.out_tuple;

  // Get context in from inner FSM
  const int64_t context_in_arg_idx =
      outer_prepared.arg_index_for_op.at(&invoke_context_in.op);
  XLSCC_CHECK(outer_prepared.args.at(context_in_arg_idx).valid(), body_loc);

  outer_prepared.args[context_in_arg_idx] = context_in;

  // Need an invoke for the context receive operation
  // (args have been updated)
  xls::BValue context_receive_ret_val = pb.Invoke(
      outer_prepared.args, outer_prepared.xls_func->xls_func, body_loc,
      /*name=*/
      absl::StrFormat("%s_state_%i_context_receive_invoke", fsm_prefix,
                      outer_state.index));

  xls::BValue not_enter_condition = pb.Not(
      enter_condition, body_loc,
      /*name=*/
      absl::StrFormat("%s_not_enter_condition", sub_proc_invoked->name_prefix));

  // Fill in op_tokens with keys for skipped operations in this state
  // For after_ops in other states
  for (const InvokeToGenerate& invoke : invokes_to_generate) {
    op_tokens[&invoke.op] = contents_ret.token_out;
  }

  return SubFSMReturn{
      .first_iter = contents_ret.first_iter,
      .exit_state_condition =
          pb.Or(not_enter_condition, contents_ret.do_break, body_loc,
                /*name=*/
                absl::StrFormat("%s_not_exit_state_condition",
                                sub_proc_invoked->name_prefix)),
      .return_value = context_receive_ret_val,
      .extra_next_state_values = contents_ret.extra_next_state_values,
      .token_out = contents_ret.token_out};
}

xls::BValue Translator::ConditionWithExtra(xls::BuilderBase& builder,
                                           xls::BValue condition,
                                           const InvokeToGenerate& invoke,
                                           const xls::SourceInfo& op_loc) {
  xls::BValue ret = condition;
  XLSCC_CHECK(ret.valid(), op_loc);
  if (invoke.extra_condition.valid()) {
    ret = builder.And(ret, invoke.extra_condition, op_loc);
  }
  if (context().full_condition.valid()) {
    ret = builder.And(ret, context().full_condition, op_loc);
  }
  return ret;
}

absl::StatusOr<xls::BValue> Translator::GenerateIOInvoke(
    const InvokeToGenerate& invoke, xls::BValue before_token,
    PreparedBlock& prepared, xls::BValue& last_ret_val, xls::ProcBuilder& pb) {
  const IOOp& op = invoke.op;
  xls::SourceInfo op_loc = op.op_location;
  const int64_t return_index = prepared.return_index_for_op.at(&op);

  xls::BValue ret_io_value = GetFlexTupleField(
      last_ret_val, return_index, prepared.xls_func->return_value_count, op_loc,
      /*op_name=*/
      absl::StrFormat("%s_ret_io_value", op.final_param_name));

  xls::BValue arg_io_val;

  xls::BValue new_token;
  const ChannelBundle* bundle_ptr = nullptr;

  if (op.op != OpType::kTrace) {
    bundle_ptr = &prepared.xls_channel_by_function_channel.at(op.channel);
  }

  if (op.op == OpType::kRecv) {
    xls::Channel* xls_channel = bundle_ptr->regular;

    unused_xls_channel_ops_.remove({xls_channel, /*is_send=*/false});

    CHECK_NE(xls_channel, nullptr);

    xls::BValue condition = ret_io_value;
    CHECK_EQ(condition.GetType()->GetFlatBitCount(), 1);
    condition = ConditionWithExtra(pb, condition, invoke, op_loc);
    xls::BValue receive;
    if (op.is_blocking) {
      receive = pb.ReceiveIf(xls_channel, before_token, condition, op_loc,
                             /*name=*/Debug_OpName(op));
    } else {
      receive = pb.ReceiveIfNonBlocking(xls_channel, before_token, condition,
                                        op_loc, /*name=*/Debug_OpName(op));
    }
    new_token =
        pb.TupleIndex(receive, 0, op_loc,
                      /*name=*/absl::StrFormat("%s_token", Debug_OpName(op)));

    xls::BValue in_val;
    if (op.is_blocking) {
      in_val =
          pb.TupleIndex(receive, 1, op_loc,
                        /*name=*/absl::StrFormat("%s_value", Debug_OpName(op)));
    } else {
      in_val = pb.Tuple(
          {pb.TupleIndex(
               receive, 1, op_loc,
               /*name=*/absl::StrFormat("%s_value", Debug_OpName(op))),
           pb.TupleIndex(
               receive, 2, op_loc,
               /*name=*/absl::StrFormat("%s_valid", Debug_OpName(op)))});
    }
    arg_io_val = in_val;
  } else if (op.op == OpType::kSend) {
    xls::Channel* xls_channel = bundle_ptr->regular;

    unused_xls_channel_ops_.remove({xls_channel, /*is_send=*/true});

    CHECK_NE(xls_channel, nullptr);
    xls::BValue val =
        pb.TupleIndex(ret_io_value, 0, op_loc,
                      /*name=*/absl::StrFormat("%s_value", Debug_OpName(op)));
    xls::BValue condition = pb.TupleIndex(
        ret_io_value, 1, op_loc, absl::StrFormat("%s_pred", Debug_OpName(op)));
    CHECK_EQ(condition.GetType()->GetFlatBitCount(), 1);
    condition = ConditionWithExtra(pb, condition, invoke, op_loc);

    new_token = pb.SendIf(xls_channel, before_token, condition, val, op_loc,
                          /*name=*/absl::StrFormat("%s", Debug_OpName(op)));
  } else if (op.op == OpType::kRead) {
    CHECK_EQ(bundle_ptr->regular, nullptr);
    CHECK_NE(bundle_ptr->read_request, nullptr);
    CHECK_NE(bundle_ptr->read_response, nullptr);

    unused_xls_channel_ops_.remove(
        {bundle_ptr->read_request, /*is_send=*/true});
    unused_xls_channel_ops_.remove(
        {bundle_ptr->read_response, /*is_send=*/false});

    xls::BValue addr =
        pb.TupleIndex(ret_io_value, 0, op_loc,
                      /*name=*/absl::StrFormat("%s_addr", Debug_OpName(op)));
    xls::BValue condition =
        pb.TupleIndex(ret_io_value, 1, op_loc,
                      /*name=*/absl::StrFormat("%s_cond", Debug_OpName(op)));
    CHECK_EQ(condition.GetType()->GetFlatBitCount(), 1);
    condition = ConditionWithExtra(pb, condition, invoke, op_loc);

    // TODO(google/xls#861): supported masked memory operations.
    xls::BValue mask = pb.Literal(xls::Value::Tuple({}), op_loc);
    xls::BValue send_tuple_with_mask = pb.Tuple({addr, mask}, op_loc);
    new_token = pb.SendIf(bundle_ptr->read_request, before_token, condition,
                          send_tuple_with_mask, op_loc,
                          /*name=*/absl::StrFormat("%s", Debug_OpName(op)));

    xls::BValue receive =
        pb.ReceiveIf(bundle_ptr->read_response, new_token, condition, op_loc,
                     /*name=*/Debug_OpName(op));

    new_token =
        pb.TupleIndex(receive, 0, op_loc,
                      /*name=*/absl::StrFormat("%s_token", Debug_OpName(op)));
    xls::BValue response_tup = pb.TupleIndex(
        receive, 1, op_loc,
        /*name=*/absl::StrFormat("%s_response_tup", Debug_OpName(op)));
    xls::BValue response = pb.TupleIndex(
        response_tup, 0, op_loc,
        /*name=*/absl::StrFormat("%s_response", Debug_OpName(op)));

    arg_io_val = response;
  } else if (op.op == OpType::kWrite) {
    CHECK_EQ(bundle_ptr->regular, nullptr);
    CHECK_NE(bundle_ptr->write_request, nullptr);
    CHECK_NE(bundle_ptr->write_response, nullptr);

    unused_xls_channel_ops_.remove(
        {bundle_ptr->write_request, /*is_send=*/true});
    unused_xls_channel_ops_.remove(
        {bundle_ptr->write_response, /*is_send=*/false});

    // This has (addr, value)
    xls::BValue send_tuple = pb.TupleIndex(
        ret_io_value, 0, op_loc,
        /*name=*/absl::StrFormat("%s_send_tup", Debug_OpName(op)));
    xls::BValue condition = pb.TupleIndex(
        ret_io_value, 1, op_loc, absl::StrFormat("%s_pred", Debug_OpName(op)));
    CHECK_EQ(condition.GetType()->GetFlatBitCount(), 1);
    condition = ConditionWithExtra(pb, condition, invoke, op_loc);

    // This has (addr, value, mask)
    xls::BValue addr =
        pb.TupleIndex(send_tuple, 0, op_loc,
                      /*name=*/absl::StrFormat("%s_addr", Debug_OpName(op)));
    xls::BValue value =
        pb.TupleIndex(send_tuple, 1, op_loc,
                      /*name=*/absl::StrFormat("%s_value", Debug_OpName(op)));
    // TODO(google/xls#861): supported masked memory operations.
    xls::BValue mask = pb.Literal(xls::Value::Tuple({}), op_loc);
    xls::BValue send_tuple_with_mask = pb.Tuple({addr, value, mask}, op_loc);
    new_token = pb.SendIf(bundle_ptr->write_request, before_token, condition,
                          send_tuple_with_mask, op_loc,
                          /*name=*/absl::StrFormat("%s", Debug_OpName(op)));

    xls::BValue receive =
        pb.ReceiveIf(bundle_ptr->write_response, new_token, condition, op_loc,
                     /*name=*/absl::StrFormat("%s", Debug_OpName(op)));
    new_token =
        pb.TupleIndex(receive, 0, op_loc,
                      /*name=*/absl::StrFormat("%s_token", Debug_OpName(op)));
    // Ignore received value, should be an empty tuple
  } else if (op.op == OpType::kTrace) {
    XLS_ASSIGN_OR_RETURN(
        new_token, GenerateTrace(ret_io_value, before_token, op, pb, invoke));
  } else {
    CHECK_EQ("Unknown IOOp type", nullptr);
  }

  if (prepared.arg_index_for_op.contains(&op)) {
    const int64_t arg_index = prepared.arg_index_for_op.at(&op);
    CHECK(arg_index >= 0 && arg_index < prepared.args.size());
    prepared.args[arg_index] = arg_io_val;
  }

  // The function is invoked again with the value received from the channel
  //  for each read() Op. The final invocation will produce all complete
  //  outputs.
  last_ret_val =
      pb.Invoke(prepared.args, prepared.xls_func->xls_func, op_loc,
                /*name=*/absl::StrFormat("invoke_%s", Debug_OpName(invoke.op)));
  XLSCC_CHECK(last_ret_val.valid(), op_loc);
  return new_token;
}

absl::StatusOr<xls::BValue> Translator::GenerateIOInvokesWithAfterOps(
    IOSchedulingOption option, xls::BValue origin_token,
    const std::list<Translator::InvokeToGenerate>& invokes_to_generate,
    absl::flat_hash_map<const IOOp*, xls::BValue>& op_tokens,
    xls::BValue& last_ret_val, Translator::PreparedBlock& prepared,
    xls::ProcBuilder& pb, const xls::SourceInfo& body_loc) {
  std::vector<xls::BValue> fan_ins_tokens;
  for (const InvokeToGenerate& invoke : invokes_to_generate) {
    const IOOp& op = invoke.op;
    if (op.scheduling_option != option) {
      continue;
    }

    xls::SourceInfo op_loc = op.op_location;

    xls::BValue before_token = origin_token;
    if (!op.after_ops.empty()) {
      XLSCC_CHECK(op.scheduling_option == option, op_loc);

      std::vector<xls::BValue> after_tokens;
      after_tokens.reserve(op.after_ops.size());
      for (const IOOp* after_op : op.after_ops) {
        XLSCC_CHECK(after_op->scheduling_option == option, op_loc);
        XLSCC_CHECK_NE(&op, after_op, op_loc);
        xls::BValue after_token = op_tokens.at(after_op);
        after_tokens.push_back(after_token);
      }
      before_token = pb.AfterAll(
          after_tokens, body_loc,
          /*name=*/absl::StrFormat("after_tokens_%s", op.final_param_name));
    }

    XLS_ASSIGN_OR_RETURN(
        xls::BValue new_token,
        GenerateIOInvoke(invoke, before_token, prepared, last_ret_val, pb));

    CHECK(!op_tokens.contains(&op));
    op_tokens[&op] = new_token;

    fan_ins_tokens.push_back(new_token);
  }

  if (!fan_ins_tokens.empty()) {
    return pb.AfterAll(fan_ins_tokens, body_loc, /*name=*/"fan_ins_tokens");
  }

  return xls::BValue();
}

absl::StatusOr<xls::BValue> Translator::GenerateTrace(
    xls::BValue trace_out_value, xls::BValue before_token, const IOOp& op,
    xls::ProcBuilder& pb, const InvokeToGenerate& invoke) {
  switch (op.trace_type) {
    case TraceType::kNull:
      break;
    case TraceType::kTrace: {
      // Tuple is (condition, ... args ...)
      const uint64_t tuple_count =
          trace_out_value.GetType()->AsTupleOrDie()->size();
      CHECK_GE(tuple_count, 1);
      xls::BValue condition =
          pb.TupleIndex(trace_out_value, 0, op.op_location,
                        /*name=*/absl::StrFormat("%s_cond", Debug_OpName(op)));
      CHECK_EQ(condition.GetType()->GetFlatBitCount(), 1);
      condition = ConditionWithExtra(pb, condition, invoke, op.op_location);
      std::vector<xls::BValue> args;
      for (int tuple_idx = 1; tuple_idx < tuple_count; ++tuple_idx) {
        xls::BValue arg = pb.TupleIndex(
            trace_out_value, tuple_idx, op.op_location,
            /*name=*/absl::StrFormat("%s_arg_%i", Debug_OpName(op), tuple_idx));
        args.push_back(arg);
      }

      return pb.Trace(before_token, /*condition=*/condition, args,
                      /*format_string=*/op.trace_message_string,
                      /*verbosity=*/0, op.op_location);
    }
    case TraceType::kAssert: {
      xls::BValue condition = pb.Not(trace_out_value, op.op_location);
      condition = ConditionWithExtra(pb, condition, invoke, op.op_location);
      // Assert condition is !fire
      return pb.Assert(before_token,
                       /*condition=*/condition,
                       /*message=*/op.trace_message_string,
                       /*label=*/op.label_string.empty()
                           ? std::nullopt
                           : std::optional<std::string>(op.label_string),
                       op.op_location);
    }
  }
  return absl::InternalError(
      ErrorMessage(op.op_location, "Unknown trace type %i", op.trace_type));
}

absl::Status Translator::GenerateIRBlockCheck(
    const HLSBlock& block, const std::list<ExternalChannelInfo>& top_decls,
    const xls::SourceInfo& body_loc) {
  if (!block.has_name()) {
    return absl::InvalidArgumentError(absl::StrFormat("HLSBlock has no name"));
  }

  absl::flat_hash_set<string> channel_names_in_block;
  for (const HLSChannel& channel : block.channels()) {
    if (!channel.has_name() || !channel.has_type() ||
        !(channel.has_is_input() || channel.has_depth())) {
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

absl::StatusOr<CValue> Translator::GenerateTopClassInitValue(
    const std::shared_ptr<CType>& this_type,
    // Can be nullptr
    const clang::CXXRecordDecl* this_decl, const xls::SourceInfo& body_loc) {
  for (const clang::CXXConstructorDecl* ctor : this_decl->ctors()) {
    if (!(ctor->isTrivial() || ctor->isDefaultConstructor())) {
      ctor->dump();
      return absl::UnimplementedError(ErrorMessage(
          body_loc, "Non-trivial constructors in top class not yet supported"));
    }
  }

  XLS_ASSIGN_OR_RETURN(std::shared_ptr<CType> resolved_type,
                       ResolveTypeInstance(this_type));

  auto struct_type = std::dynamic_pointer_cast<CStructType>(resolved_type);
  CHECK_NE(struct_type, nullptr);

  PushContextGuard temporary_this_context(*this, body_loc);

  // Don't allow "this" to be propagated up: it's only temporary for use
  // within the initializer list
  context().propagate_up = false;
  context().override_this_decl_ = this_decl;
  context().ast_context = &this_decl->getASTContext();

  XLS_ASSIGN_OR_RETURN(CValue this_val,
                       CreateDefaultCValue(this_type, body_loc));

  XLS_RETURN_IF_ERROR(DeclareVariable(this_decl, this_val, body_loc));

  // Check for side-effects
  for (const clang::FieldDecl* field_decl : this_decl->fields()) {
    std::shared_ptr<CField> field = struct_type->get_field(field_decl);
    std::shared_ptr<CType> field_type = field->type();
    auto field_decl_loc = GetLoc(*field_decl);

    CValue field_val;
    if (!field_decl->hasInClassInitializer()) {
      XLS_ASSIGN_OR_RETURN(bool contains_lvalues,
                           field_type->ContainsLValues(*this));
      if (error_on_uninitialized_ && !contains_lvalues) {
        return absl::InvalidArgumentError(ErrorMessage(
            field_decl_loc,
            "Class member %s not initialized with error_on_uninitialized set",
            field_decl->getQualifiedNameAsString()));
      }
      XLS_ASSIGN_OR_RETURN(field_val,
                           CreateDefaultCValue(field_type, field_decl_loc));
    } else {
      PushContextGuard guard(*this, field_decl_loc);
      context().any_side_effects_requested = false;

      XLS_ASSIGN_OR_RETURN(
          field_val,
          GenerateIR_Expr(field_decl->getInClassInitializer(), field_decl_loc));

      if (context().any_side_effects_requested) {
        return absl::UnimplementedError(
            ErrorMessage(field_decl_loc,
                         "Side effects in initializer for top class field %s",
                         field_decl->getQualifiedNameAsString()));
      }
    }

    CHECK(*field_val.type() == *field_type);

    {
      UnmaskAndIgnoreSideEffectsGuard unmask_guard(*this);
      XLS_RETURN_IF_ERROR(
          AssignMember(this_decl, field_decl, field_val, field_decl_loc));
    }
  }

  XLS_ASSIGN_OR_RETURN(this_val, GetIdentifier(this_decl, body_loc));

  return this_val;
}

absl::StatusOr<std::unique_ptr<GeneratedFunction>>
Translator::GenerateIRBlockPrepare(
    PreparedBlock& prepared, xls::ProcBuilder& pb, int64_t next_return_index,
    const std::shared_ptr<CType>& this_type,
    const clang::CXXRecordDecl* this_decl,
    const std::list<ExternalChannelInfo>& top_decls,
    const GeneratedFunction* caller_sub_function,
    const xls::SourceInfo& body_loc) {
  // For defaults, updates, invokes
  auto temp_sf = std::make_unique<GeneratedFunction>();

  CHECK(!context_stack_.empty());
  context() = TranslationContext();
  context().propagate_up = false;
  context().sf = temp_sf.get();
  context().fb = dynamic_cast<xls::BuilderBase*>(&pb);

  // This state and argument
  if (this_decl != nullptr) {
    XLS_ASSIGN_OR_RETURN(CValue this_cval, GenerateTopClassInitValue(
                                               this_type, this_decl, body_loc));

    CHECK(this_cval.rvalue().valid());
    XLS_ASSIGN_OR_RETURN(xls::Value this_init_val,
                         EvaluateBVal(this_cval.rvalue(), body_loc));

    xls::BValue elem_bval = pb.StateElement("this", this_init_val, body_loc);
    xls::StateElement* state_elem =
        elem_bval.node()->As<xls::StateRead>()->state_element();

    // Don't need to worry about sharing for this, as it's only used at the top
    // level (block as class)
    prepared.state_element_for_variable[this_decl] = state_elem;
    prepared.args.push_back(elem_bval);
  }

  for (const clang::NamedDecl* namedecl :
       prepared.xls_func->GetDeterministicallyOrderedStaticValues()) {
    const ConstValue& initval = prepared.xls_func->static_values.at(namedecl);

    // Don't need to worry about sharing for this, as it's only used when a
    // static variable is declared in the loop body
    xls::BValue elem_bval = pb.StateElement(
        XLSNameMangle(clang::GlobalDecl(namedecl)), initval.rvalue(), body_loc);
    xls::StateElement* state_elem =
        elem_bval.node()->As<xls::StateRead>()->state_element();

    prepared.return_index_for_static[namedecl] = next_return_index++;
    prepared.state_element_for_variable[namedecl] = state_elem;
  }

  // This return
  if (this_decl != nullptr) {
    prepared.return_index_for_static[this_decl] = next_return_index++;
  }

  // Generate control receive (with direct-ins)
  if (caller_sub_function != nullptr) {
    CHECK_NE(caller_sub_function->control_in_channel, nullptr);
    xls::BValue control_tup = pb.Receive(
        caller_sub_function->control_in_channel, prepared.token, body_loc,
        /*name=*/"control_receive");
    prepared.token = pb.TupleIndex(control_tup, 0, body_loc,
                                   /*name=*/"control_receive_token");

    CHECK_NE(caller_sub_function->direct_ins_channel, nullptr);
    xls::BValue direct_ins_tup_recvd = pb.Receive(
        caller_sub_function->direct_ins_channel, prepared.token, body_loc,
        /*name=*/"direct_ins_receive");
    prepared.token = pb.TupleIndex(direct_ins_tup_recvd, 0, body_loc,
                                   /*name=*/"direct_ins_receive_token");
    xls::BValue direct_ins_tup =
        pb.TupleIndex(direct_ins_tup_recvd, 1, body_loc,
                      /*name=*/"direct_ins_tup");

    for (int64_t pi = 0; pi < prepared.xls_func->clang_decl->getNumParams();
         ++pi) {
      XLS_ASSIGN_OR_RETURN(
          bool is_direct_in,
          IsSubBlockDirectInParam(prepared.xls_func->clang_decl, pi));
      if (!is_direct_in) {
        continue;
      }
      prepared.args.push_back(pb.TupleIndex(direct_ins_tup, pi));
    }
  }

  // Prepare direct-ins
  for (const ExternalChannelInfo& top_decl : top_decls) {
    if (top_decl.interface_type != InterfaceType::kDirect) {
      continue;
    }

    const ChannelBundle& bundle = top_decl.external_channels;
    xls::Channel* xls_channel = bundle.regular;
    unused_xls_channel_ops_.remove({xls_channel, /*is_send=*/false});

    xls::BValue receive = pb.Receive(xls_channel, prepared.token);
    prepared.token = pb.TupleIndex(
        receive, 0, body_loc,
        /*name=*/absl::StrFormat("%s_token", xls_channel->name()));
    xls::BValue direct_in_value = pb.TupleIndex(
        receive, 1, body_loc,
        /*name=*/absl::StrFormat("%s_value", xls_channel->name()));

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

    if (!prepared.xls_channel_by_function_channel.contains(op.channel)) {
      std::optional<ChannelBundle> bundle = GetChannelBundleForOp(op, body_loc);
      if (bundle.has_value()) {
        prepared.xls_channel_by_function_channel[op.channel] = bundle.value();
      }
    }
  }

  // Params
  for (const xlscc::SideEffectingParameter& param :
       prepared.xls_func->side_effecting_parameters) {
    switch (param.type) {
      case xlscc::SideEffectingParameterType::kIOOp: {
        const IOOp& op = *param.io_op;
        if (op.op == OpType::kRead || op.op == OpType::kRecv) {
          xls::BValue val =
              pb.Literal(xls::ZeroOfType(param.xls_io_param_type), body_loc);
          prepared.arg_index_for_op[&op] = prepared.args.size();
          prepared.args.push_back(val);
        }
        break;
      }
      case xlscc::SideEffectingParameterType::kStatic: {
        xls::BValue bval(
            pb.proc()->GetStateRead(
                prepared.state_element_for_variable.at(param.static_value)),
            &pb);
        prepared.args.push_back(bval);
        break;
      }
      default: {
        return absl::InternalError(
            ErrorMessage(body_loc, "Unknown type of SideEffectingParameter"));
        break;
      }
    }
  }

  return std::move(temp_sf);
}

std::optional<ChannelBundle> Translator::GetChannelBundleForOp(
    const IOOp& op, const xls::SourceInfo& loc) {
  if (op.op == OpType::kTrace) {
    return std::nullopt;
  }

  XLSCC_CHECK_NE(op.channel, nullptr, loc);

  if (op.channel->generated.has_value()) {
    ChannelBundle generated_bundle = {.regular = op.channel->generated.value()};
    return generated_bundle;
  }

  XLSCC_CHECK(external_channels_by_internal_channel_.contains(op.channel), loc);
  XLSCC_CHECK_EQ(external_channels_by_internal_channel_.count(op.channel), 1,
                 loc);
  const ChannelBundle bundle =
      external_channels_by_internal_channel_.find(op.channel)->second;
  return bundle;
}

absl::StatusOr<xls::Proc*> Translator::BuildWithNextStateValueMap(
    xls::ProcBuilder& pb, xls::BValue token,
    const absl::btree_multimap<const xls::StateElement*, NextStateValue>&
        next_state_values,
    const xls::SourceInfo& loc) {
  const int64_t n_state_elems = pb.proc()->StateElements().size();

  std::vector<xls::BValue> next_state_values_list;
  next_state_values_list.reserve(n_state_elems);

  for (xls::StateElement* elem : pb.proc()->StateElements()) {
    const int64_t values_for_elem = next_state_values.count(elem);
    XLSCC_CHECK_GE(values_for_elem, 0, loc);
    if (values_for_elem == 0) {
      return absl::InternalError(
          absl::StrFormat("No next values for state element %s", elem->name()));
    }
    xls::StateRead* state_read = pb.proc()->GetStateRead(elem);
    xls::BValue read_bval(state_read, &pb);
    if (values_for_elem == 1) {
      const NextStateValue& next_state_value =
          next_state_values.find(elem)->second;
      xls::BValue next_state_value_bval;
      if (next_state_value.condition.valid()) {
        next_state_value_bval =
            pb.Select(next_state_value.condition,
                      /*on_true=*/next_state_value.value,
                      /*on_false=*/read_bval, loc, /*name=*/
                      absl::StrFormat("%s_next_state_value", elem->name()));
      } else {
        next_state_value_bval = next_state_value.value;
      }
      next_state_values_list.push_back(next_state_value_bval);
      continue;
    }
    // More than one next value

    // Sort by priority (all must have one)
    std::vector<NextStateValue> next_state_values_sorted_by_priority;
    next_state_values_sorted_by_priority.reserve(values_for_elem);
    for (auto it = next_state_values.lower_bound(elem);
         it != next_state_values.upper_bound(elem); ++it) {
      next_state_values_sorted_by_priority.push_back(it->second);
      if (it->second.priority < 0) {
        return absl::InternalError(
            absl::StrFormat("State element %s has multiple next values, but "
                            "without a priority specified (%s)",
                            elem->name(), it->second.extra_label));
      }
      if (!it->second.condition.valid()) {
        return absl::InternalError(
            absl::StrFormat("State element %s has multiple next values, but "
                            "without a valid condition (%s)",
                            elem->name(), it->second.extra_label));
      }
    }
    std::sort(next_state_values_sorted_by_priority.begin(),
              next_state_values_sorted_by_priority.end(),
              [](const NextStateValue& a, const NextStateValue& b) {
                return a.priority > b.priority;
              });

    // Construct the next state value
    xls::BValue next_state_value_bval;

    {
      // Separate lists for priority select
      std::vector<xls::BValue> conditions;
      conditions.reserve(next_state_values_sorted_by_priority.size() + 1);
      std::vector<xls::BValue> values;
      values.reserve(next_state_values_sorted_by_priority.size() + 1);

      for (const NextStateValue& next_value :
           next_state_values_sorted_by_priority) {
        conditions.push_back(next_value.condition);
        values.push_back(next_value.value);
      }

      // The value corresponding to the LSBit goes first here, but the MSBit
      // goes first in Concat
      std::reverse(values.begin(), values.end());

      xls::BValue selector = pb.Concat(
          conditions, loc,
          /*name=*/absl::StrFormat("%s_all_conditions", elem->name()));

      // Default to no change of state when all selectors are 0
      next_state_value_bval = pb.PrioritySelect(
          selector, values, /*default_value=*/read_bval, loc,
          /*name=*/absl::StrFormat("%s_select_next_value", elem->name()));
    }

    XLSCC_CHECK(next_state_value_bval.valid(), loc);
    next_state_values_list.push_back(next_state_value_bval);

    // Generate asserts for same priority conditions being mutually exclusive
    {
      absl::btree_multimap<int64_t, xls::BValue> all_conditions_by_priority;

      for (const NextStateValue& next_value :
           next_state_values_sorted_by_priority) {
        all_conditions_by_priority.insert(
            {next_value.priority, next_value.condition});
      }

      // Loop through the keys (priorities)
      for (auto priority_it = all_conditions_by_priority.begin();
           priority_it != all_conditions_by_priority.end();
           priority_it =
               all_conditions_by_priority.upper_bound(priority_it->first)) {
        // Loop through the next states in this priority
        std::vector<xls::BValue> conditions_this_priority;
        for (auto it =
                 all_conditions_by_priority.lower_bound(priority_it->first);
             it != all_conditions_by_priority.upper_bound(priority_it->first);
             ++it) {
          XLSCC_CHECK_EQ(priority_it->first, it->first, loc);
          conditions_this_priority.push_back(it->second);
        }

        // Assert that one_hot(x) == x
        xls::BValue conditions_bval =
            pb.Concat(conditions_this_priority, loc,
                      /*name=*/
                      absl::StrFormat("%s_all_conditions_priority_%i",
                                      elem->name(), priority_it->first));
        xls::BValue one_hot_wide =
            pb.OneHot(conditions_bval, xls::LsbOrMsb::kMsb, loc);
        xls::BValue one_hot =
            pb.BitSlice(one_hot_wide, /*start=*/0,
                        /*width=*/conditions_this_priority.size(), loc);
        token = pb.Assert(
            token, pb.Eq(one_hot, conditions_bval, loc),
            /*message=*/
            absl::StrFormat("Conditions for state element %s in %s are not "
                            "mutually exclusive for priority %i",
                            elem->name(), pb.name(), priority_it->first),
            /*label=*/std::nullopt, loc);
      }
    }
  }

  return pb.Build(next_state_values_list);
}

}  // namespace xlscc
