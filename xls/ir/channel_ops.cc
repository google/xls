// Copyright 2021 The XLS Authors
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

#include "xls/ir/channel_ops.h"

#include <cstdint>
#include <ostream>
#include <string>
#include <string_view>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"

namespace xls {

std::string ChannelOpsToString(ChannelOps ops) {
  switch (ops) {
    case ChannelOps::kSendOnly:
      return "send_only";
    case ChannelOps::kReceiveOnly:
      return "receive_only";
    case ChannelOps::kSendReceive:
      return "send_receive";
  }
  LOG(FATAL) << "Invalid channel kind: " << static_cast<int64_t>(ops);
}

absl::StatusOr<ChannelOps> StringToChannelOps(std::string_view str) {
  if (str == "send_only") {
    return ChannelOps::kSendOnly;
  }
  if (str == "receive_only") {
    return ChannelOps::kReceiveOnly;
  }
  if (str == "send_receive") {
    return ChannelOps::kSendReceive;
  }
  return absl::InvalidArgumentError(
      absl::StrCat("Unknown channel ops value: ", str));
}

std::ostream& operator<<(std::ostream& os, ChannelOps ops) {
  os << ChannelOpsToString(ops);
  return os;
}

}  // namespace xls
