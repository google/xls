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
#include "xls/ir/function_builder.h"
#include "xls/ir/proc.h"
#include "xls/ir/type.h"

namespace xls::dslx {

absl::StatusOr<ChannelRef> ProcScopedChannelScope::CreateChannel(
    std::string_view name, ChannelOps ops, xls::Type* type,
    std::optional<ChannelConfig> channel_config, bool interface_channel,
    std::optional<ChannelStrictness> strictness,
    std::optional<FlowControl> flow_control) {
  if (interface_channel) {
    XLS_RET_CHECK_NE(ops, ChannelOps::kSendReceive)
        << "Cannot define interface channel as both send and receive";

    xls::Proc* ir_proc = proc_builder_->proc();

    // ChannelKind is never set to anything but streaming by the FE.
    ChannelKind kind = ChannelKind::kStreaming;
    flow_control =
        flow_control.has_value() ? flow_control : kDefaultChannelFlowControl;
    strictness =
        strictness.has_value() ? strictness : kDefaultChannelStrictness;

    if (ops == ChannelOps::kReceiveOnly) {
      return ir_proc->AddInputChannel(name, type, kind, *flow_control,
                                      strictness);
    }

    return ir_proc->AddOutputChannel(name, type, kind, *flow_control,
                                     strictness);
  }

  // Create a proc-scoped channel on the proc.
  BChannelWithInterfaces channel_with_interfaces =
      proc_builder_->AddChannel(name, type, ChannelKind::kStreaming,
                                /*initial_values=*/{}, channel_config);
  XLS_RETURN_IF_ERROR(proc_builder_->GetError());
  return channel_with_interfaces.channel.channel();
}

}  // namespace xls::dslx
