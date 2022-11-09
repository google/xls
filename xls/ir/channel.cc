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
#include "google/protobuf/text_format.h"
#include "xls/common/status/ret_check.h"
#include "xls/ir/value_helpers.h"

namespace xls {

std::string ChannelKindToString(ChannelKind kind) {
  switch (kind) {
    case ChannelKind::kStreaming:
      return "streaming";
    case ChannelKind::kSingleValue:
      return "single_value";
  }
  XLS_LOG(FATAL) << "Invalid channel kind: " << static_cast<int64_t>(kind);
}

absl::StatusOr<ChannelKind> StringToChannelKind(std::string_view str) {
  if (str == "streaming") {
    return ChannelKind::kStreaming;
  } else if (str == "single_value") {
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
        &result, "flow_control=%s, ",
        FlowControlToString(streaming_channel->GetFlowControl()));
    if (streaming_channel->GetFifoDepth().has_value()) {
      absl::StrAppendFormat(&result, "fifo_depth=%d, ",
                            streaming_channel->GetFifoDepth().value());
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
  XLS_LOG(FATAL) << "Invalid flow control value: " << static_cast<int64_t>(fc);
}

absl::StatusOr<FlowControl> StringToFlowControl(std::string_view str) {
  if (str == "none") {
    return FlowControl::kNone;
  } else if (str == "ready_valid") {
    return FlowControl::kReadyValid;
  }
  return absl::InvalidArgumentError(
      absl::StrFormat("Invalid channel kind '%s'", str));
}

std::ostream& operator<<(std::ostream& os, FlowControl fc) {
  os << FlowControlToString(fc);
  return os;
}

}  // namespace xls
