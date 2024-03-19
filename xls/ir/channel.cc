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

#include <cstdint>
#include <optional>
#include <ostream>
#include <string>
#include <string_view>
#include <variant>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/types/variant.h"
#include "google/protobuf/text_format.h"
#include "xls/common/casts.h"
#include "xls/common/logging/logging.h"
#include "xls/ir/channel_ops.h"
#include "xls/ir/type.h"
#include "xls/ir/value_utils.h"

namespace xls {

std::string ChannelKindToString(ChannelKind kind) {
  switch (kind) {
    case ChannelKind::kStreaming:
      return "streaming";
    case ChannelKind::kSingleValue:
      return "single_value";
  }
  LOG(FATAL) << "Invalid channel kind: " << static_cast<int64_t>(kind);
}

absl::StatusOr<ChannelKind> StringToChannelKind(std::string_view str) {
  if (str == "streaming") {
    return ChannelKind::kStreaming;
  }
  if (str == "single_value") {
    return ChannelKind::kSingleValue;
  }
  return absl::InvalidArgumentError(
      absl::StrFormat("Invalid channel kind '%s'", str));
}

std::ostream& operator<<(std::ostream& os, ChannelKind kind) {
  os << ChannelKindToString(kind);
  return os;
}

std::string Channel::ToString() const {
  std::string result = absl::StrFormat("chan %s(", name());
  absl::StrAppendFormat(&result, "%s, ", type()->ToString());
  if (!initial_values().empty()) {
    absl::StrAppendFormat(
        &result, "initial_values={%s}, ",
        absl::StrJoin(initial_values(), ", ", UntypedValueFormatter));
  }
  absl::StrAppendFormat(&result, "id=%d, kind=%s, ops=%s, ", id(),
                        ChannelKindToString(kind_),
                        ChannelOpsToString(supported_ops()));

  if (kind() == ChannelKind::kStreaming) {
    const StreamingChannel* streaming_channel =
        down_cast<const StreamingChannel*>(this);
    absl::StrAppendFormat(
        &result, "flow_control=%s, strictness=%s, ",
        FlowControlToString(streaming_channel->GetFlowControl()),
        ChannelStrictnessToString(streaming_channel->GetStrictness()));
    const std::optional<FifoConfig>& fifo_config =
        streaming_channel->fifo_config();
    if (fifo_config.has_value()) {
      absl::StrAppendFormat(&result, "fifo_depth=%d, ", fifo_config->depth);
      if (!fifo_config->bypass) {
        absl::StrAppend(&result, "bypass=true, ");
      }
    }
  }

  std::string metadata_textproto;
  google::protobuf::TextFormat::Printer printer;
  printer.SetSingleLineMode(true);
  printer.PrintToString(metadata(), &metadata_textproto);
  if (!metadata_textproto.empty() && metadata_textproto.back() == ' ') {
    metadata_textproto.pop_back();
  }
  absl::StrAppendFormat(&result, "metadata=\"\"\"%s\"\"\")",
                        metadata_textproto);

  return result;
}

std::string FlowControlToString(FlowControl fc) {
  switch (fc) {
    case FlowControl::kNone:
      return "none";
    case FlowControl::kReadyValid:
      return "ready_valid";
  }
  LOG(FATAL) << "Invalid flow control value: " << static_cast<int64_t>(fc);
}

absl::StatusOr<FlowControl> StringToFlowControl(std::string_view str) {
  if (str == "none") {
    return FlowControl::kNone;
  }
  if (str == "ready_valid") {
    return FlowControl::kReadyValid;
  }
  return absl::InvalidArgumentError(
      absl::StrFormat("Invalid channel kind '%s'", str));
}

std::ostream& operator<<(std::ostream& os, FlowControl fc) {
  os << FlowControlToString(fc);
  return os;
}

absl::StatusOr<ChannelStrictness> ChannelStrictnessFromString(
    std::string_view text) {
  if (text == "proven_mutually_exclusive") {
    return ChannelStrictness::kProvenMutuallyExclusive;
  }
  if (text == "runtime_mutually_exclusive") {
    return ChannelStrictness::kRuntimeMutuallyExclusive;
  }
  if (text == "total_order") {
    return ChannelStrictness::kTotalOrder;
  }
  if (text == "runtime_ordered") {
    return ChannelStrictness::kRuntimeOrdered;
  }
  if (text == "arbitrary_static_order") {
    return ChannelStrictness::kArbitraryStaticOrder;
  }
  return absl::InvalidArgumentError(
      absl::StrFormat("Invalid strictness %s.", text));
}
std::string ChannelStrictnessToString(ChannelStrictness in) {
  if (in == ChannelStrictness::kProvenMutuallyExclusive) {
    return "proven_mutually_exclusive";
  }
  if (in == ChannelStrictness::kRuntimeMutuallyExclusive) {
    return "runtime_mutually_exclusive";
  }
  if (in == ChannelStrictness::kTotalOrder) {
    return "total_order";
  }
  if (in == ChannelStrictness::kRuntimeOrdered) {
    return "runtime_ordered";
  }
  if (in == ChannelStrictness::kArbitraryStaticOrder) {
    return "arbitrary_static_order";
  }
  return "unknown";
}

std::string DirectionToString(Direction direction) {
  switch (direction) {
    case Direction::kSend:
      return "send";
    case Direction::kReceive:
      return "receive";
  }
}

std::string_view ChannelRefName(ChannelRef ref) {
  return absl::visit([](const auto& ch) { return ch->name(); }, ref);
}

Type* ChannelRefType(ChannelRef ref) {
  if (std::holds_alternative<ChannelReference*>(ref)) {
    return std::get<ChannelReference*>(ref)->type();
  }
  return std::get<Channel*>(ref)->type();
}

ChannelKind ChannelRefKind(ChannelRef ref) {
  if (std::holds_alternative<ChannelReference*>(ref)) {
    return std::get<ChannelReference*>(ref)->kind();
  }
  return std::get<Channel*>(ref)->kind();
}

std::string ChannelReference::ToString() const {
  return absl::StrFormat("%s: %s %s %s", name(), type()->ToString(),
                         ChannelKindToString(kind()),
                         direction() == Direction::kSend ? "out" : "in");
}

}  // namespace xls
