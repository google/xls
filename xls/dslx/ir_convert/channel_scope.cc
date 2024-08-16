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

#include "absl/container/flat_hash_map.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/dslx/constexpr_evaluator.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/import_data.h"
#include "xls/dslx/interp_value.h"
#include "xls/dslx/interp_value_utils.h"
#include "xls/dslx/ir_convert/conversion_info.h"
#include "xls/dslx/ir_convert/ir_conversion_utils.h"
#include "xls/dslx/type_system/parametric_env.h"
#include "xls/dslx/type_system/type_info.h"
#include "xls/ir/channel.h"
#include "xls/ir/channel_ops.h"
#include "xls/ir/package.h"
#include "xls/ir/value.h"

namespace xls::dslx {

ChannelScope::ChannelScope(PackageConversionData* conversion_info,
                           TypeInfo* type_info, ImportData* import_data,
                           const ParametricEnv& bindings)
    : conversion_info_(conversion_info),
      type_info_(type_info),
      import_data_(import_data),
      channel_name_uniquer_(/*separator=*/"__"),
      bindings_(bindings) {
  // Populate channel name uniquer with pre-existing channel names.
  for (Channel* channel : conversion_info_->package->channels()) {
    channel_name_uniquer_.GetSanitizedUniqueName(channel->name());
  }
}

absl::StatusOr<ChannelOrArray> ChannelScope::DefineChannelOrArray(
    const ChannelDecl* decl) {
  VLOG(4) << "ChannelScope::HandleChannelDecl: " << decl->ToString() << " : "
          << decl->span().ToString();
  XLS_ASSIGN_OR_RETURN(
      InterpValue name_interp_value,
      ConstexprEvaluator::EvaluateToValue(
          import_data_, type_info_, /*warning_collector=*/nullptr, bindings_,
          &decl->channel_name_expr()));
  XLS_ASSIGN_OR_RETURN(std::string channel_name,
                       InterpValueAsString(name_interp_value));
  channel_name = channel_name_uniquer_.GetSanitizedUniqueName(
      absl::StrCat(conversion_info_->package->name(), "__", channel_name));
  auto maybe_type = type_info_->GetItem(decl->type());
  XLS_RET_CHECK(maybe_type.has_value());
  XLS_ASSIGN_OR_RETURN(xls::Type * type,
                       TypeToIr(conversion_info_->package.get(),
                                *maybe_type.value(), bindings_));

  std::optional<int64_t> fifo_depth;
  if (decl->fifo_depth().has_value()) {
    // Note: warning collect is nullptr since all warnings should have been
    // flagged in typechecking.
    XLS_ASSIGN_OR_RETURN(
        InterpValue iv,
        ConstexprEvaluator::EvaluateToValue(
            import_data_, type_info_, /*warning_collector=*/nullptr, bindings_,
            decl->fifo_depth().value()));
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
  // TODO: https://github.com/google/xls/issues/704 - If `decl` has dims, then
  // instead of this, create a channel per element and return something
  // representing the whole array.
  XLS_ASSIGN_OR_RETURN(StreamingChannel * channel,
                       conversion_info_->package->CreateStreamingChannel(
                           channel_name, ChannelOps::kSendReceive, type,
                           /*initial_values=*/{},
                           /*fifo_config=*/fifo_config));
  decl_to_channel_[decl] = channel;
  return channel;
}

absl::StatusOr<ChannelOrArray>
ChannelScope::AssociateWithExistingChannelOrArray(const NameDef* name_def,
                                                  const ChannelDecl* decl) {
  VLOG(4) << "ChannelScope::AssociateWithExistingChannelOrArray : "
          << name_def->ToString() << " -> " << decl->ToString();
  if (!decl_to_channel_.contains(decl)) {
    return absl::NotFoundError(absl::StrCat(
        "Decl is not associated with a channel: ", decl->ToString()));
  }
  Channel* channel = decl_to_channel_.at(decl);
  XLS_RETURN_IF_ERROR(AssociateWithExistingChannel(name_def, channel));
  return channel;
}

absl::Status ChannelScope::AssociateWithExistingChannel(const NameDef* name_def,
                                                        Channel* channel) {
  name_def_to_channel_[name_def] = channel;
  return absl::OkStatus();
}

}  // namespace xls::dslx
