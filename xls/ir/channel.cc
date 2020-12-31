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

std::string SupportedOpsToString(Channel::SupportedOps supported_ops) {
  switch (supported_ops) {
    case Channel::SupportedOps::kSendOnly:
      return "send_only";
    case Channel::SupportedOps::kReceiveOnly:
      return "receive_only";
    case Channel::SupportedOps::kSendReceive:
      return "send_receive";
  }
  XLS_LOG(FATAL) << "Invalid channel kind: "
                 << static_cast<int64>(supported_ops);
}

absl::StatusOr<Channel::SupportedOps> StringToSupportedOps(
    absl::string_view str) {
  if (str == "send_only") {
    return Channel::SupportedOps::kSendOnly;
  } else if (str == "receive_only") {
    return Channel::SupportedOps::kReceiveOnly;
  } else if (str == "send_receive") {
    return Channel::SupportedOps::kSendReceive;
  }
  return absl::InvalidArgumentError(
      absl::StrCat("Unknown channel kind: ", str));
}

std::ostream& operator<<(std::ostream& os,
                         Channel::SupportedOps supported_ops) {
  os << SupportedOpsToString(supported_ops);
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
      ChannelKindToString(kind_), SupportedOpsToString(supported_ops()),
      metadata().ShortDebugString());

  return result;
}

bool Channel::IsStreaming() const { return kind_ == ChannelKind::kStreaming; }

bool Channel::IsRegister() const { return kind_ == ChannelKind::kRegister; }

bool Channel::IsPort() const { return kind_ == ChannelKind::kPort; }

bool Channel::IsLogical() const { return kind_ == ChannelKind::kLogical; }

}  // namespace xls
