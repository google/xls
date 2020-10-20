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

namespace xls {

std::string DataElement::ToString() const {
  return absl::StrFormat("%s: %s", name, type->ToString());
}

std::ostream& operator<<(std::ostream& os, const DataElement& data_element) {
  os << data_element.ToString();
  return os;
}

std::string ChannelKindToString(ChannelKind kind) {
  switch (kind) {
    case ChannelKind::kSendOnly:
      return "send_only";
    case ChannelKind::kReceiveOnly:
      return "receive_only";
    case ChannelKind::kSendReceive:
      return "send_receive";
  }
}

absl::StatusOr<ChannelKind> StringToChannelKind(absl::string_view str) {
  if (str == "send_only") {
    return ChannelKind::kSendOnly;
  } else if (str == "receive_only") {
    return ChannelKind::kReceiveOnly;
  } else if (str == "send_receive") {
    return ChannelKind::kSendReceive;
  }
  return absl::InvalidArgumentError(
      absl::StrCat("Unknown channel kind: ", str));
}

std::ostream& operator<<(std::ostream& os, ChannelKind kind) {
  os << ChannelKindToString(kind);
  return os;
}

std::string Channel::ToString() const {
  std::string result = absl::StrFormat("chan %s(", name());
  absl::StrAppend(&result,
                  absl::StrJoin(data_elements(), ", ",
                                [](std::string* out, const DataElement& e) {
                                  absl::StrAppend(out, e.ToString());
                                }));
  absl::StrAppendFormat(&result, ", id=%d, kind=%s, metadata=\"\"\"%s\"\"\")",
                        id(), ChannelKindToString(kind()),
                        metadata().ShortDebugString());

  return result;
}

}  // namespace xls
