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

std::string DataElement::ToString() const {
  return absl::StrFormat("%s: %s", name, type->ToString());
}

std::ostream& operator<<(std::ostream& os, const DataElement& data_element) {
  os << data_element.ToString();
  return os;
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

namespace {

absl::StatusOr<std::vector<ChannelData>> ExtractInitialValues(
    absl::Span<const DataElement> data_elements) {
  // Verify the number of initial values across data elements are all the same.
  for (int64 i = 1; i < data_elements.size(); ++i) {
    XLS_RET_CHECK_EQ(data_elements[0].initial_values.size(),
                     data_elements[i].initial_values.size());
  }
  // Transpose the initial values held in each DataElement to a vector
  // containing the initial values across data elements.
  std::vector<ChannelData> initial_values;
  for (int64 i = 0; i < data_elements[0].initial_values.size(); ++i) {
    std::vector<Value> values;
    for (const DataElement& data_element : data_elements) {
      values.push_back(data_element.initial_values[i]);
    }
    initial_values.push_back(std::move(values));
  }

  return initial_values;
}

}  // namespace

std::string Channel::ToString() const {
  std::string result = absl::StrFormat("chan %s(", name());
  absl::StrAppend(&result,
                  absl::StrJoin(data_elements(), ", ",
                                [](std::string* out, const DataElement& e) {
                                  absl::StrAppend(out, e.ToString());
                                }));
  absl::StrAppendFormat(
      &result, ", id=%d, kind=%s, ops=%s, metadata=\"\"\"%s\"\"\")", id(),
      IsSingleValue() ? "single_value" : "streaming",
      SupportedOpsToString(supported_ops()), metadata().ShortDebugString());

  return result;
}

bool Channel::IsStreaming() const {
  return dynamic_cast<const StreamingChannel*>(this) != nullptr;
}

bool Channel::IsSingleValue() const {
  return dynamic_cast<const SingleValueChannel*>(this) != nullptr;
}

/* static */ absl::StatusOr<std::unique_ptr<StreamingChannel>>
StreamingChannel::Create(absl::string_view name, int64 id,
                         Channel::SupportedOps supported_ops,
                         absl::Span<const DataElement> data_elements,
                         const ChannelMetadataProto& metadata) {
  if (data_elements.empty()) {
    return absl::InvalidArgumentError(
        "Channel must have at least one data element.");
  }
  XLS_ASSIGN_OR_RETURN(std::vector<ChannelData> initial_values,
                       ExtractInitialValues(data_elements));

  return absl::WrapUnique(
      new StreamingChannel(name, id, supported_ops, data_elements,
                           std::move(initial_values), metadata));
}

/* static */ absl::StatusOr<std::unique_ptr<SingleValueChannel>>
SingleValueChannel::Create(absl::string_view name, int64 id,
                           Channel::SupportedOps supported_ops,
                           absl::Span<const DataElement> data_elements,
                           const ChannelMetadataProto& metadata) {
  if (data_elements.empty()) {
    return absl::InvalidArgumentError(
        "Channel must have at least one data element.");
  }
  XLS_ASSIGN_OR_RETURN(std::vector<ChannelData> initial_values,
                       ExtractInitialValues(data_elements));
  if (initial_values.size() > 1) {
    return absl::InvalidArgumentError(
        "A single value channel can not have more than one initial value.");
  }
  return absl::WrapUnique(
      new SingleValueChannel(name, id, supported_ops, data_elements,
                             std::move(initial_values), metadata));
}

}  // namespace xls
