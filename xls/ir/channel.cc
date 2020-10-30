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
  XLS_LOG(FATAL) << "Invalid channel kind: " << static_cast<int64>(kind);
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

Channel::Channel(absl::string_view name, int64 id, ChannelKind kind,
                 absl::Span<const DataElement> data_elements,
                 const ChannelMetadataProto& metadata)
    : name_(name),
      id_(id),
      kind_(kind),
      data_elements_(data_elements.begin(), data_elements.end()),
      metadata_(metadata) {
  XLS_CHECK(!data_elements_.empty());
  // Verify the number of initial values across data elements are all the same.
  for (int64 i = 1; i < data_elements_.size(); ++i) {
    XLS_CHECK_EQ(data_elements_[0].initial_values.size(),
                 data_elements_[i].initial_values.size());
  }
  // Transpose the initial values held in each DataElement to a vector
  // containing the initial values across data elements.
  // TODO(meheff): Consider passing in the initial values in this transposed
  // form.
  for (int64 i = 0; i < data_elements_[0].initial_values.size(); ++i) {
    std::vector<Value> values;
    for (const DataElement& data_element : data_elements_) {
      values.push_back(data_element.initial_values[i]);
    }
    initial_values_.push_back(std::move(values));
  }
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
