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

#ifndef XLS_IR_CHANNEL_H_
#define XLS_IR_CHANNEL_H_

#include <iosfwd>

#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xls/common/integral_types.h"
#include "xls/ir/channel.pb.h"
#include "xls/ir/type.h"
#include "xls/ir/value.h"

namespace xls {

// The kinds indicates the type(s) of operations permitted on the
// channel. Send-only channels can only have send operations (not
// receive) associated with the channel as might be used for communicated
// to a component outside of XLS. Receive-only channels are similarly
// defined. Send-receive channels can have both send and receive operations and
// can be used for communicated between procs.
enum class ChannelKind { kSendOnly, kReceiveOnly, kSendReceive };

// Returns the string representation of the given channel kind.
std::string ChannelKindToString(ChannelKind kind);

// Converts the string representation of a channel to a ChannelKind. Returns an
// error if the string is not a valid channel kind.
absl::StatusOr<ChannelKind> StringToChannelKind(absl::string_view str);

std::ostream& operator<<(std::ostream& os, ChannelKind kind);

// Describes a single data element conveyed by a channel. A channel can have
// more than one data element.
struct DataElement {
  std::string name;
  Type* type;
  // The initial values (if any) in the channel for this data element. All
  // DataElements in a channel must have the same number of elements.
  std::vector<Value> initial_values;

  std::string ToString() const;
};

std::ostream& operator<<(std::ostream& os, const DataElement& data_element);

// Abstraction describing a channel in XLS IR. Channels are a mechanism for
// communicating between procs or between procs and components outside of
// XLS. Send and receive nodes in procs are associated with a particular
// channel. The channel data structure carries information about how
// communication occurs over the channel.
class Channel {
 public:
  Channel(absl::string_view name, int64 id, ChannelKind kind,
          absl::Span<const DataElement> data_elements,
          const ChannelMetadataProto& metadata);

  // Returns the name of the channel.
  const std::string& name() const { return name_; }

  // Returns the ID of the channel. The ID is unique within the scope of a
  // package.
  int64 id() const { return id_; }

  // Returns the channel kind: send-only, receive-only, or send-receive.
  ChannelKind kind() const { return kind_; }

  // Returns the data elements communicated by the channel with each
  // transaction.
  absl::Span<const DataElement> data_elements() const { return data_elements_; }

  // Returns the i-th data element.
  const DataElement& data_element(int64 i) const {
    return data_elements_.at(i);
  }

  // Returns the initial values held in the channel. The inner span holds the
  // values across data elements. The outer span holds the entries in the
  // channel FIFO.
  const std::vector<std::vector<Value>>& initial_values() const {
    return initial_values_;
  }

  // Returns the metadata associated with this channel.
  const ChannelMetadataProto& metadata() const { return metadata_; }

  // Returns whether this channel can be used to send (receive) data.
  bool CanSend() const {
    return kind() == ChannelKind::kSendOnly ||
           kind() == ChannelKind::kSendReceive;
  }
  bool CanReceive() const {
    return kind() == ChannelKind::kReceiveOnly ||
           kind() == ChannelKind::kSendReceive;
  }

  std::string ToString() const;

 private:
  std::string name_;
  int64 id_;
  ChannelKind kind_;
  std::vector<DataElement> data_elements_;
  ChannelMetadataProto metadata_;
  std::vector<std::vector<Value>> initial_values_;
};

}  // namespace xls

#endif  // XLS_IR_CHANNEL_H_
