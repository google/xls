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

#include "xls/ir/channel.h"

#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "xls/common/status/ret_check.h"

namespace xls {

std::string ChannelKindToString(ChannelKind kind) {
  switch (kind) {
    case kStreaming:
      return "streaming";
    case kPort:
      return "port";
    case kRegister:
      return "register";
    case kLogical:
      return "logical";
  }
  XLS_LOG(FATAL) << "Invalid channel kind: " << static_cast<int64>(kind);
}

absl::StatusOr<ChannelKind> StringToChannelKind(absl::string_view str) {
  if (str == "streaming") {
    return kStreaming;
  } else if (str == "port") {
    return kPort;
  } else if (str == "register") {
    return kRegister;
  } else if (str == "logical") {
    return kLogical;
  }
  return absl::InvalidArgumentError(
      absl::StrFormat("Invalid channel kind '%s'", str));
}

std::ostream& operator<<(std::ostream& os, ChannelKind kind) {
  os << ChannelKindToString(kind);
  return os;
}

std::string ChannelOpsToString(ChannelOps ops) {
  switch (ops) {
    case ChannelOps::kSendOnly:
      return "send_only";
    case ChannelOps::kReceiveOnly:
      return "receive_only";
    case ChannelOps::kSendReceive:
      return "send_receive";
  }
  XLS_LOG(FATAL) << "Invalid channel kind: " << static_cast<int64>(ops);
}

absl::StatusOr<ChannelOps> StringToChannelOps(absl::string_view str) {
  if (str == "send_only") {
    return ChannelOps::kSendOnly;
  } else if (str == "receive_only") {
    return ChannelOps::kReceiveOnly;
  } else if (str == "send_receive") {
    return ChannelOps::kSendReceive;
  }
  return absl::InvalidArgumentError(
      absl::StrCat("Unknown channel ops value: ", str));
}

std::ostream& operator<<(std::ostream& os, ChannelOps ops) {
  os << ChannelOpsToString(ops);
  return os;
}

std::string Channel::ToString() const {
  std::string result = absl::StrFormat("chan %s(", name());
  absl::StrAppendFormat(&result, "%s, ", type()->ToString());
  if (!initial_values().empty()) {
    absl::StrAppendFormat(&result, "initial_values={%s}, ",
                          absl::StrJoin(initial_values(), ", ",
                                        [](std::string* out, const Value& v) {
                                          absl::StrAppend(out,
                                                          v.ToHumanString());
                                        }));
  }
  absl::StrAppendFormat(
      &result, "id=%d, kind=%s, ops=%s, metadata=\"\"\"%s\"\"\")", id(),
      ChannelKindToString(kind_), ChannelOpsToString(supported_ops()),
      metadata().ShortDebugString());

  return result;
}

bool Channel::IsStreaming() const { return kind_ == ChannelKind::kStreaming; }

bool Channel::IsRegister() const { return kind_ == ChannelKind::kRegister; }

bool Channel::IsPort() const { return kind_ == ChannelKind::kPort; }

bool Channel::IsLogical() const { return kind_ == ChannelKind::kLogical; }

}  // namespace xls
