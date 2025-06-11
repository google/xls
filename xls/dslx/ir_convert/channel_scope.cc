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

#include "xls/dslx/ir_convert/channel_scope.h"

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "absl/types/variant.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/common/visitor.h"
#include "xls/dslx/channel_direction.h"
#include "xls/dslx/constexpr_evaluator.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/frontend/proc_id.h"
#include "xls/dslx/import_data.h"
#include "xls/dslx/interp_value.h"
#include "xls/dslx/interp_value_utils.h"
#include "xls/dslx/ir_convert/conversion_info.h"
#include "xls/dslx/ir_convert/ir_conversion_utils.h"
#include "xls/dslx/type_system/parametric_env.h"
#include "xls/dslx/type_system/type.h"
#include "xls/dslx/type_system/type_info.h"
#include "xls/ir/channel.h"
#include "xls/ir/channel_ops.h"
#include "xls/ir/package.h"
#include "xls/ir/type.h"
#include "xls/ir/value.h"
#include "xls/ir/xls_ir_interface.pb.h"

namespace xls::dslx {
namespace {

constexpr std::string_view kNameAndDimsSeparator = "__";
constexpr std::string_view kBetweenDimsSeparator = "_";

}  // namespace

ChannelScope::ChannelScope(PackageConversionData* conversion_info,
                           ImportData* import_data,
                           std::optional<FifoConfig> default_fifo_config)
    : conversion_info_(conversion_info),
      import_data_(import_data),
      channel_name_uniquer_(kNameAndDimsSeparator),
      default_fifo_config_(default_fifo_config) {
  // Populate channel name uniquer with pre-existing channel names.
  for (Channel* channel : conversion_info_->package->channels()) {
    channel_name_uniquer_.GetSanitizedUniqueName(channel->name());
  }
}

absl::StatusOr<ChannelOrArray> ChannelScope::DefineChannelOrArray(
    const ChannelDecl* decl) {
  VLOG(4) << "ChannelScope::DefineChannelOrArray: " << decl->ToString();
  XLS_RET_CHECK(function_context_.has_value());
  XLS_ASSIGN_OR_RETURN(
      InterpValue name_interp_value,
      ConstexprEvaluator::EvaluateToValue(
          import_data_, function_context_->type_info,
          /*warning_collector=*/nullptr, function_context_->bindings,
          &decl->channel_name_expr()));
  XLS_ASSIGN_OR_RETURN(std::string short_name,
                       InterpValueAsString(name_interp_value));
  XLS_ASSIGN_OR_RETURN(xls::Type * type, GetChannelType(decl));
  XLS_ASSIGN_OR_RETURN(std::optional<ChannelConfig> channel_config,
                       CreateChannelConfig(decl));
  XLS_ASSIGN_OR_RETURN(
      ChannelOrArray channel_or_array,
      DefineChannelOrArrayInternal(short_name, ChannelOps::kSendReceive, type,
                                   channel_config, decl->dims()));
  decl_to_channel_or_array_[decl] = channel_or_array;
  return channel_or_array;
}

absl::StatusOr<ChannelOrArray> ChannelScope::DefineChannelOrArrayInternal(
    std::string_view short_name, ChannelOps ops, xls::Type* type,
    std::optional<ChannelConfig> channel_config,
    const std::optional<std::vector<Expr*>>& dims) {
  XLS_ASSIGN_OR_RETURN(std::string base_channel_name,
                       CreateBaseChannelName(short_name));
  std::vector<std::string> channel_names;
  if (!dims.has_value()) {
    XLS_ASSIGN_OR_RETURN(
        Channel * channel,
        CreateChannel(base_channel_name, ops, type, channel_config));
    return channel;
  }
  ChannelArray* array = &arrays_.emplace_back(ChannelArray(base_channel_name));
  XLS_ASSIGN_OR_RETURN(std::vector<std::string> suffixes,
                       CreateAllArrayElementSuffixes(*dims));
  for (const std::string& suffix : suffixes) {
    std::string channel_name =
        absl::StrCat(base_channel_name, kNameAndDimsSeparator, suffix);
    XLS_ASSIGN_OR_RETURN(
        Channel * channel,
        CreateChannel(channel_name, ops, type, channel_config));
    array->AddChannel(channel_name, channel);
  }
  return array;
}

absl::StatusOr<ChannelOrArray> ChannelScope::DefineBoundaryChannelOrArray(
    const Param* param, TypeInfo* type_info) {
  VLOG(4) << "ChannelScope::DefineBoundaryChannelOrArray: "
          << param->ToString();
  auto* type_annot =
      dynamic_cast<ChannelTypeAnnotation*>(param->type_annotation());
  XLS_RET_CHECK(type_annot != nullptr);
  std::optional<Type*> type = type_info->GetItem(type_annot->payload());
  XLS_RET_CHECK(type.has_value());
  XLS_ASSIGN_OR_RETURN(
      xls::Type * ir_type,
      TypeToIr(conversion_info_->package.get(), **type, ParametricEnv()));
  ChannelOps op = type_annot->direction() == ChannelDirection::kIn
                      ? ChannelOps::kReceiveOnly
                      : ChannelOps::kSendOnly;
  XLS_ASSIGN_OR_RETURN(
      ChannelOrArray channel_or_array,
      DefineChannelOrArrayInternal(param->identifier(), op, ir_type,
                                   /*channel_config=*/std::nullopt,
                                   type_annot->dims()));
  XLS_RETURN_IF_ERROR(DefineProtoChannelOrArray(channel_or_array, type_annot,
                                                ir_type, type_info));
  return channel_or_array;
}

absl::Status ChannelScope::DefineProtoChannelOrArray(
    ChannelOrArray channel_or_array, dslx::ChannelTypeAnnotation* type_annot,
    xls::Type* ir_type, TypeInfo* type_info) {
  if (std::holds_alternative<ChannelArray*>(channel_or_array)) {
    auto* array = std::get<ChannelArray*>(channel_or_array);
    for (const std::string& name : array->flattened_names_in_order()) {
      std::optional<Channel*> channel = array->FindChannel(name);
      XLS_RET_CHECK(channel.has_value());
      XLS_RETURN_IF_ERROR(
          DefineProtoChannelOrArray(*channel, type_annot, ir_type, type_info));
    }
    return absl::OkStatus();
  }
  Channel* channel = std::get<Channel*>(channel_or_array);
  PackageInterfaceProto::Channel* proto_chan =
      conversion_info_->interface.add_channels();
  *proto_chan->mutable_name() = channel->name();
  *proto_chan->mutable_type() = ir_type->ToProto();
  // Channels at the boundary only have one direction, with the other direction
  // being used externally to the DSLX code.
  proto_chan->set_direction(type_annot->direction() == ChannelDirection::kIn
                                ? PackageInterfaceProto::Channel::IN
                                : PackageInterfaceProto::Channel::OUT);
  XLS_ASSIGN_OR_RETURN(std::optional<std::string> first_sv_type,
                       type_info->FindSvType(type_annot->payload()));
  if (first_sv_type) {
    *proto_chan->mutable_sv_type() = *first_sv_type;
  }
  return absl::OkStatus();
}

absl::StatusOr<ChannelOrArray>
ChannelScope::AssociateWithExistingChannelOrArray(const ProcId& proc_id,
                                                  const NameDef* name_def,
                                                  const ChannelDecl* decl) {
  VLOG(4) << "ChannelScope::AssociateWithExistingChannelOrArray : "
          << name_def->ToString() << " -> " << decl->ToString();
  if (!decl_to_channel_or_array_.contains(decl)) {
    return absl::NotFoundError(absl::StrCat(
        "Decl is not associated with a channel or array: ", decl->ToString()));
  }
  ChannelOrArray channel_or_array = decl_to_channel_or_array_.at(decl);
  XLS_RETURN_IF_ERROR(
      AssociateWithExistingChannelOrArray(proc_id, name_def, channel_or_array));
  return channel_or_array;
}

absl::Status ChannelScope::AssociateWithExistingChannelOrArray(
    const ProcId& proc_id, const NameDef* name_def,
    ChannelOrArray channel_or_array) {
  VLOG(4) << "ChannelScope::AssociateWithExistingChannelOrArray : "
          << name_def->ToString() << " -> "
          << GetBaseNameForChannelOrArray(channel_or_array) << " (array: "
          << std::holds_alternative<ChannelArray*>(channel_or_array) << ")";
  name_def_to_channel_or_array_[std::make_pair(proc_id, name_def)] =
      channel_or_array;
  return absl::OkStatus();
}

absl::StatusOr<Channel*> ChannelScope::GetChannelForArrayIndex(
    const ProcId& proc_id, const Index* index) {
  XLS_ASSIGN_OR_RETURN(
      ChannelOrArray result,
      EvaluateIndex(proc_id, index, /*allow_subarray_reference=*/false));
  XLS_RET_CHECK(std::holds_alternative<Channel*>(result));
  return std::get<Channel*>(result);
}

absl::StatusOr<ChannelOrArray> ChannelScope::GetChannelOrArrayForArrayIndex(
    const ProcId& proc_id, const Index* index) {
  return EvaluateIndex(proc_id, index, /*allow_subarray_reference=*/true);
}

absl::StatusOr<ChannelOrArray> ChannelScope::EvaluateIndex(
    const ProcId& proc_id, const Index* index, bool allow_subarray_reference) {
  VLOG(4) << "ChannelScope::GetChannelForArrayIndex : " << index->ToString();
  XLS_RET_CHECK(function_context_.has_value());
  std::string suffix;
  for (;;) {
    if (!std::holds_alternative<Expr*>(index->rhs())) {
      return absl::UnimplementedError(
          "Channel array elements must be accessed with a constexpr index.");
    }
    XLS_ASSIGN_OR_RETURN(
        InterpValue dim_interp_value,
        ConstexprEvaluator::EvaluateToValue(
            import_data_, function_context_->type_info,
            /*warning_collector=*/nullptr, function_context_->bindings,
            std::get<Expr*>(index->rhs())));
    XLS_ASSIGN_OR_RETURN(int64_t dim_value,
                         dim_interp_value.GetBitValueUnsigned());
    suffix = suffix.empty()
                 ? absl::StrCat(dim_value)
                 : absl::StrCat(dim_value, kBetweenDimsSeparator, suffix);
    if (const NameRef* name_ref = dynamic_cast<NameRef*>(index->lhs());
        name_ref) {
      return GetChannelArrayElement(proc_id, name_ref, suffix,
                                    allow_subarray_reference);
    }
    Index* new_index = dynamic_cast<Index*>(index->lhs());
    if (!new_index) {
      return absl::InvalidArgumentError(
          absl::StrCat("Expected Index or NameRef as left-hand side of: ",
                       index->ToString()));
    }
    index = new_index;
  }
}

std::string_view ChannelScope::GetBaseNameForChannelOrArray(
    ChannelOrArray channel_or_array) {
  return absl::visit(
      Visitor{[](Channel* channel) { return channel->name(); },
              [](ChannelArray* array) -> std::string_view {
                return array->base_channel_name();
              },
              [](ChannelInterface* channel) { return channel->name(); }},
      channel_or_array);
}

absl::StatusOr<std::vector<std::string>>
ChannelScope::CreateAllArrayElementSuffixes(const std::vector<Expr*>& dims) {
  std::vector<std::string> strings;
  // Note: dims are in the opposite of indexing order, and here we want to use
  // them to produce index strings in indexing order, hence the backwards loop.
  for (int64_t i = dims.size() - 1; i >= 0; --i) {
    Expr* dim = dims[i];
    XLS_ASSIGN_OR_RETURN(
        InterpValue dim_interp_value,
        ConstexprEvaluator::EvaluateToValue(
            import_data_, function_context_->type_info,
            /*warning_collector=*/nullptr, function_context_->bindings, dim));
    XLS_ASSIGN_OR_RETURN(int64_t dim_value,
                         dim_interp_value.GetBitValueUnsigned());
    std::vector<std::string> new_strings;
    new_strings.reserve(dim_value * strings.size());
    for (int64_t element_index = 0; element_index < dim_value;
         element_index++) {
      if (strings.empty()) {
        new_strings.push_back(absl::StrCat(element_index));
        continue;
      }
      for (const std::string& next : strings) {
        new_strings.push_back(
            absl::StrCat(next, kBetweenDimsSeparator, element_index));
      }
    }
    strings = std::move(new_strings);
  }
  return strings;
}

absl::StatusOr<std::string> ChannelScope::CreateBaseChannelName(
    std::string_view short_name) {
  return channel_name_uniquer_.GetSanitizedUniqueName(absl::StrCat(
      conversion_info_->package->name(), kNameAndDimsSeparator, short_name));
}

absl::StatusOr<xls::Type*> ChannelScope::GetChannelType(
    const ChannelDecl* decl) const {
  std::optional<Type*> type =
      function_context_->type_info->GetItem(decl->type());
  XLS_RET_CHECK(type.has_value());
  return TypeToIr(conversion_info_->package.get(), **type,
                  function_context_->bindings);
}

absl::StatusOr<std::optional<ChannelConfig>> ChannelScope::CreateChannelConfig(
    const ChannelDecl* decl) const {
  if (decl->channel_config().has_value()) {
    XLS_RET_CHECK(!decl->fifo_depth().has_value())
        << "Cannot specify both fifo_depth and channel_config.";
    return decl->channel_config();
  }

  std::optional<int64_t> fifo_depth;
  if (decl->fifo_depth().has_value()) {
    // Note: warning collect is nullptr since all warnings should have been
    // flagged in typechecking.
    XLS_ASSIGN_OR_RETURN(
        InterpValue iv,
        ConstexprEvaluator::EvaluateToValue(
            import_data_, function_context_->type_info,
            /*warning_collector=*/nullptr, function_context_->bindings,
            decl->fifo_depth().value()));
    XLS_ASSIGN_OR_RETURN(Value fifo_depth_value, iv.ConvertToIr());
    if (!fifo_depth_value.IsBits()) {
      return absl::InvalidArgumentError(
          absl::StrFormat("Expected fifo depth to be bits type, got %s.",
                          fifo_depth_value.ToHumanString()));
    }
    XLS_ASSIGN_OR_RETURN(fifo_depth, fifo_depth_value.bits().ToInt64());
    // We choose bypass=true FIFOs by default and register push outputs (ready).
    // The idea is to avoid combo loops introduced by pop->push ready
    // combinational paths. For depth zero FIFOs, we do not register push
    // outputs as for now we think of these FIFOs as direct connections.
    // TODO: google/xls#1391 - we should have a better way to specify fifo
    // configuration.
    return ChannelConfig().WithFifoConfig(FifoConfig(
        /*depth=*/*fifo_depth,
        /*bypass=*/true,
        /*register_push_outputs=*/*fifo_depth != 0,
        /*register_pop_outputs=*/false));
  }

  return ChannelConfig().WithFifoConfig(default_fifo_config_);
}

absl::StatusOr<Channel*> ChannelScope::CreateChannel(
    std::string_view name, ChannelOps ops, xls::Type* type,
    std::optional<ChannelConfig> channel_config) {
  if (channel_config.has_value()) {
    return conversion_info_->package->CreateStreamingChannel(
        name, ops, type,
        /*initial_values=*/{},
        /*channel_config=*/*channel_config);
  }
  return conversion_info_->package->CreateStreamingChannel(name, ops, type);
}

absl::StatusOr<ChannelOrArray> ChannelScope::GetChannelArrayElement(
    const ProcId& proc_id, const NameRef* name_ref,
    std::string_view flattened_name_suffix, bool allow_subarray_reference) {
  const auto* name_def = std::get<const NameDef*>(name_ref->name_def());
  const auto it =
      name_def_to_channel_or_array_.find(std::make_pair(proc_id, name_def));
  if (it == name_def_to_channel_or_array_.end()) {
    return absl::NotFoundError(
        absl::StrCat("Not a channel or channel array: ", name_def->ToString()));
  }
  ChannelOrArray channel_or_array = it->second;
  if (!std::holds_alternative<ChannelArray*>(channel_or_array)) {
    return absl::InvalidArgumentError(
        absl::StrCat("Attempted to index into an individual channel "
                     "instead of an array: ",
                     std::get<Channel*>(channel_or_array)->name()));
  }
  ChannelArray* array = std::get<ChannelArray*>(channel_or_array);
  std::string flattened_channel_name = absl::StrCat(
      array->base_channel_name(),
      array->is_subarray() ? kBetweenDimsSeparator : kNameAndDimsSeparator,
      flattened_name_suffix);
  std::optional<Channel*> channel = array->FindChannel(flattened_channel_name);
  if (channel.has_value()) {
    VLOG(4) << "Found channel array element: " << (*channel)->name();
    return *channel;
  }
  if (allow_subarray_reference) {
    return GetOrDefineSubarray(array, flattened_channel_name);
  }
  return absl::NotFoundError(absl::StrCat(
      "No array element with flattened name: ", flattened_channel_name));
}

absl::StatusOr<ChannelArray*> ChannelScope::GetOrDefineSubarray(
    ChannelArray* array, std::string_view subarray_name) {
  const auto it = subarrays_.find(subarray_name);
  if (it != subarrays_.end()) {
    VLOG(5) << "Found subarray " << subarray_name;
    return it->second;
  }
  ChannelArray* subarray =
      &arrays_.emplace_back(ChannelArray(subarray_name, /*subarray=*/true));
  subarrays_.emplace_hint(it, subarray_name, subarray);
  std::string subarray_prefix =
      absl::StrCat(subarray_name, kBetweenDimsSeparator);
  VLOG(5) << "Searching for subarray elements with prefix " << subarray_prefix;
  for (const std::string& name : array->flattened_names_in_order()) {
    if (absl::StartsWith(name, subarray_prefix)) {
      Channel* channel = *array->FindChannel(name);
      subarray->AddChannel(channel->name(), channel);
    }
  }
  // If type checking has been done right etc., there should never be a request
  // for a subarray prefix that matches zero channels, even when compiling
  // erroneous DSLX code.
  XLS_RET_CHECK(!subarray->flattened_names_in_order().empty());
  VLOG(5) << "Defined subarray " << subarray_name << " with "
          << subarray->flattened_names_in_order().size() << " elements.";
  return subarray;
}

}  // namespace xls::dslx
