// Copyright 2025 The XLS Authors
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

#include "xls/dslx/ir_convert/proc_scoped_channel_scope.h"

#include <optional>
#include <string_view>

#include "absl/status/statusor.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/channel.h"
#include "xls/ir/channel_ops.h"
#include "xls/ir/proc.h"
#include "xls/ir/type.h"

namespace xls::dslx {

absl::StatusOr<ChannelRef> ProcScopedChannelScope::CreateChannel(
    std::string_view name, ChannelOps ops, xls::Type* type,
    std::optional<ChannelConfig> channel_config, bool interface_channel) {
  if (interface_channel) {
    XLS_RET_CHECK_NE(ops, ChannelOps::kSendReceive)
        << "Cannot define interface channel as both send and receive";

    xls::Proc* ir_proc = proc_builder_->proc();

    // TOOD: davidplass - figure out how to get strictness, flow control,
    // kind instead of defaults.
    // ChannelStrictness seems to be only set from non-DSLX-generated IR.
    std::optional<ChannelStrictness> strictness = kDefaultChannelStrictness;
    // FlowControl never seems to be set to anything other than ready valid.
    FlowControl flow_control = FlowControl::kReadyValid;
    // ChannelKind is never set to anything but streaming by the FE.
    ChannelKind kind = ChannelKind::kStreaming;

    if (ops == ChannelOps::kReceiveOnly) {
      return ir_proc->AddInputChannel(name, type, kind, flow_control,
                                      strictness);
    }

    return ir_proc->AddOutputChannel(name, type, kind, flow_control,
                                     strictness);
  }

  // Create a proc-scoped channel on the proc.
  XLS_ASSIGN_OR_RETURN(auto channel_with_interfaces,
                       proc_builder_->AddChannel(name, type));
  return channel_with_interfaces.channel;
}

}  // namespace xls::dslx
