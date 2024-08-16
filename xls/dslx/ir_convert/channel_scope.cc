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
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "absl/types/variant.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/common/visitor.h"
#include "xls/dslx/constexpr_evaluator.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/import_data.h"
#include "xls/dslx/interp_value.h"
#include "xls/dslx/interp_value_utils.h"
#include "xls/dslx/ir_convert/conversion_info.h"
#include "xls/dslx/ir_convert/ir_conversion_utils.h"
#include "xls/dslx/type_system/type_info.h"
#include "xls/ir/channel.h"
#include "xls/ir/channel_ops.h"
#include "xls/ir/package.h"
#include "xls/ir/value.h"

namespace xls::dslx {

ChannelScope::ChannelScope(PackageConversionData* conversion_info,
                           ImportData* import_data)
    : conversion_info_(conversion_info),
      import_data_(import_data),
      channel_name_uniquer_(/*separator=*/"__") {
  // Populate channel name uniquer with pre-existing channel names.
  for (Channel* channel : conversion_info_->package->channels()) {
    channel_name_uniquer_.GetSanitizedUniqueName(channel->name());
  }
}

absl::StatusOr<ChannelOrArray> ChannelScope::DefineChannelOrArray(
    const ChannelDecl* decl) {
  VLOG(4) << "ChannelScope::HandleChannelDecl: " << decl->ToString() << " : "
          << decl->span().ToString();
  CHECK(function_context_.has_value());
  XLS_ASSIGN_OR_RETURN(std::string base_channel_name,
                       CreateBaseChannelName(decl));
  XLS_ASSIGN_OR_RETURN(xls::Type * type, GetChannelType(decl));
  XLS_ASSIGN_OR_RETURN(std::optional<FifoConfig> fifo_config,
                       CreateFifoConfig(decl));
  std::vector<std::string> channel_names;
  if (!decl->dims().has_value()) {
    XLS_ASSIGN_OR_RETURN(Channel * channel,
                         CreateChannel(base_channel_name, type, fifo_config));
    decl_to_channel_or_array_[decl] = channel;
    return channel;
  }
  ChannelArray* array = &arrays_.emplace_back(ChannelArray(base_channel_name));
  XLS_ASSIGN_OR_RETURN(std::vector<std::string> suffixes,
                       CreateAllArrayElementSuffixes(*decl->dims()));
  for (const std::string& suffix : suffixes) {
    std::string channel_name = absl::StrCat(base_channel_name, "__", suffix);
    XLS_ASSIGN_OR_RETURN(Channel * channel,
                         CreateChannel(channel_name, type, fifo_config));
    array->AddChannel(channel_name, channel);
  }
  decl_to_channel_or_array_[decl] = array;
  return array;
}

absl::StatusOr<ChannelOrArray>
ChannelScope::AssociateWithExistingChannelOrArray(const NameDef* name_def,
                                                  const ChannelDecl* decl) {
  VLOG(4) << "ChannelScope::AssociateWithExistingChannelOrArray : "
          << name_def->ToString() << " -> " << decl->ToString();
  if (!decl_to_channel_or_array_.contains(decl)) {
    return absl::NotFoundError(absl::StrCat(
        "Decl is not associated with a channel or array: ", decl->ToString()));
  }
  ChannelOrArray channel_or_array = decl_to_channel_or_array_.at(decl);
  XLS_RETURN_IF_ERROR(
      AssociateWithExistingChannelOrArray(name_def, channel_or_array));
  return channel_or_array;
}

absl::Status ChannelScope::AssociateWithExistingChannelOrArray(
    const NameDef* name_def, ChannelOrArray channel_or_array) {
  VLOG(4) << "ChannelScope::AssociateWithExistingChannelOrArray : "
          << name_def->ToString() << " -> "
          << GetBaseNameForChannelOrArray(channel_or_array);
  name_def_to_channel_or_array_[name_def] = channel_or_array;
  return absl::OkStatus();
}

absl::StatusOr<Channel*> ChannelScope::GetChannelForArrayIndex(
    const Index* index) {
  VLOG(4) << "ChannelScope::GetChannelForArrayIndex : " << index->ToString();
  CHECK(function_context_.has_value());
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
    suffix = suffix.empty() ? absl::StrCat(dim_value)
                            : absl::StrCat(dim_value, "_", suffix);
    if (const NameRef* name_ref = dynamic_cast<NameRef*>(index->lhs());
        name_ref) {
      return GetChannelArrayElement(name_ref, suffix);
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

absl::StatusOr<std::string_view> ChannelScope::GetBaseNameForNameDef(
    const NameDef* name_def) {
  const auto it = name_def_to_channel_or_array_.find(name_def);
  if (it == name_def_to_channel_or_array_.end()) {
    return absl::NotFoundError(absl::StrCat(
        "No channel or array associated with NameDef: ", name_def->ToString()));
  }
  return GetBaseNameForChannelOrArray(it->second);
}

std::string_view ChannelScope::GetBaseNameForChannelOrArray(
    ChannelOrArray channel_or_array) {
  return absl::visit(Visitor{[](Channel* channel) { return channel->name(); },
                             [](ChannelArray* array) -> std::string_view {
                               return array->base_channel_name();
                             }},
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
        new_strings.push_back(absl::StrCat(next, "_", element_index));
      }
    }
    strings = std::move(new_strings);
  }
  return strings;
}

absl::StatusOr<std::string> ChannelScope::CreateBaseChannelName(
    const ChannelDecl* decl) {
  XLS_ASSIGN_OR_RETURN(
      InterpValue name_interp_value,
      ConstexprEvaluator::EvaluateToValue(
          import_data_, function_context_->type_info,
          /*warning_collector=*/nullptr, function_context_->bindings,
          &decl->channel_name_expr()));
  XLS_ASSIGN_OR_RETURN(std::string base_channel_name,
                       InterpValueAsString(name_interp_value));
  return channel_name_uniquer_.GetSanitizedUniqueName(
      absl::StrCat(conversion_info_->package->name(), "__", base_channel_name));
}

absl::StatusOr<xls::Type*> ChannelScope::GetChannelType(
    const ChannelDecl* decl) const {
  auto maybe_type = function_context_->type_info->GetItem(decl->type());
  XLS_RET_CHECK(maybe_type.has_value());
  return TypeToIr(conversion_info_->package.get(), *maybe_type.value(),
                  function_context_->bindings);
}

absl::StatusOr<std::optional<FifoConfig>> ChannelScope::CreateFifoConfig(
    const ChannelDecl* decl) const {
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
  }

  if (!fifo_depth.has_value()) {
    return std::nullopt;
  }
  // We choose bypass=true FIFOs by default and register push outputs (ready).
  // The idea is to avoid combo loops introduced by pop->push ready
  // combinational paths. For depth zero FIFOs, we do not register push outputs
  // as for now we think of these FIFOs as direct connections.
  // TODO: google/xls#1391 - we should have a better way to specify fifo
  // configuration.
  return FifoConfig(
      /*depth=*/*fifo_depth,
      /*bypass=*/true,
      /*register_push_outputs=*/*fifo_depth != 0,
      /*register_pop_outputs=*/false);
}

absl::StatusOr<Channel*> ChannelScope::CreateChannel(
    std::string_view name, xls::Type* type,
    std::optional<FifoConfig> fifo_config) {
  return conversion_info_->package->CreateStreamingChannel(
      name, ChannelOps::kSendReceive, type,
      /*initial_values=*/{},
      /*fifo_config=*/fifo_config);
}

absl::StatusOr<Channel*> ChannelScope::GetChannelArrayElement(
    const NameRef* name_ref, std::string_view flattened_name_suffix) {
  const auto* name_def = std::get<const NameDef*>(name_ref->name_def());
  const auto it = name_def_to_channel_or_array_.find(name_def);
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
  std::string flattened_channel_name =
      absl::StrCat(array->base_channel_name(), "__", flattened_name_suffix);
  std::optional<Channel*> channel = array->FindChannel(flattened_channel_name);
  if (channel.has_value()) {
    VLOG(4) << "Found channel array element: " << (*channel)->name();
    return *channel;
  }
  return absl::NotFoundError(absl::StrCat(
      "No array element with flattened name: ", flattened_channel_name));
}

}  // namespace xls::dslx
