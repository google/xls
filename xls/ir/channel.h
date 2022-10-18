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

#include <cstdint>
#include <iosfwd>

#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xls/ir/channel.pb.h"
#include "xls/ir/channel_ops.h"
#include "xls/ir/type.h"
#include "xls/ir/value.h"

namespace xls {

// Enum for the various kinds of channels supported in XLS.
enum class ChannelKind {
  //  A channel with FIFO semantics.
  kStreaming,

  // A channel which holds a single value. Values are written to the channel via
  // send operations which overwrites the previously sent values. Receives
  // nondestructively read the most-recently sent value.
  kSingleValue,
};

std::string ChannelKindToString(ChannelKind kind);
absl::StatusOr<ChannelKind> StringToChannelKind(std::string_view str);
std::ostream& operator<<(std::ostream& os, ChannelKind kind);

// Abstraction describing a channel in XLS IR. Channels are a mechanism for
// communicating between procs or between procs and components outside of
// XLS. Send and receive nodes in procs are associated with a particular
// channel. The channel data structure carries information about how
// communication occurs over the channel.
class Channel {
 public:
  Channel(std::string_view name, int64_t id, ChannelOps supported_ops,
          ChannelKind kind, Type* type, absl::Span<const Value> initial_values,
          const ChannelMetadataProto& metadata)
      : name_(name),
        id_(id),
        supported_ops_(supported_ops),
        kind_(kind),
        type_(type),
        initial_values_(initial_values.begin(), initial_values.end()),
        metadata_(metadata) {}

  virtual ~Channel() = default;

  // Returns the name of the channel.
  const std::string& name() const { return name_; }

  // Returns the ID of the channel. The ID is unique within the scope of a
  // package.
  int64_t id() const { return id_; }

  // Returns the suppored ops for the channel: send-only, receive-only, or
  // send-receive.
  ChannelOps supported_ops() const { return supported_ops_; }

  Type* type() const { return type_; }
  absl::Span<const Value> initial_values() const { return initial_values_; }

  // Returns the metadata associated with this channel.
  const ChannelMetadataProto& metadata() const { return metadata_; }

  // Returns whether this channel can be used to send (receive) data.
  bool CanSend() const {
    return supported_ops() == ChannelOps::kSendOnly ||
           supported_ops() == ChannelOps::kSendReceive;
  }
  bool CanReceive() const {
    return supported_ops() == ChannelOps::kReceiveOnly ||
           supported_ops() == ChannelOps::kSendReceive;
  }

  // Returns true if the metadata for block ports is complete.
  virtual bool HasCompletedBlockPortNames() const = 0;

  // Returns / sets name of block this channel is associated with.
  void SetBlockName(std::string_view name) {
    metadata_.mutable_block_ports()->set_block_name(std::string(name));
  }
  std::optional<std::string> GetBlockName() const {
    if (metadata_.block_ports().has_block_name()) {
      return metadata_.block_ports().block_name();
    }
    return absl::nullopt;
  }

  // Returns / sets name of data port this channel is associated with.
  void SetDataPortName(std::string_view name) {
    metadata_.mutable_block_ports()->set_data_port_name(std::string(name));
  }
  std::optional<std::string> GetDataPortName() const {
    if (metadata_.block_ports().has_data_port_name()) {
      return metadata_.block_ports().data_port_name();
    }
    return absl::nullopt;
  }

  // Returns / sets name of valid port this channel is associated with.
  void SetValidPortName(std::string_view name) {
    metadata_.mutable_block_ports()->set_valid_port_name(std::string(name));
  }
  std::optional<std::string> GetValidPortName() const {
    if (metadata_.block_ports().has_valid_port_name()) {
      return metadata_.block_ports().valid_port_name();
    }
    return absl::nullopt;
  }

  // Returns / sets name of ready port this channel is associated with.
  void SetReadyPortName(std::string_view name) {
    metadata_.mutable_block_ports()->set_ready_port_name(std::string(name));
  }
  std::optional<std::string> GetReadyPortName() const {
    if (metadata_.block_ports().has_valid_port_name()) {
      return metadata_.block_ports().ready_port_name();
    }
    return absl::nullopt;
  }

  ChannelKind kind() const { return kind_; }

  virtual std::string ToString() const;

 protected:
  std::string name_;
  int64_t id_;
  ChannelOps supported_ops_;
  ChannelKind kind_;
  Type* type_;
  std::vector<Value> initial_values_;
  ChannelMetadataProto metadata_;
};

// The flow control mechanism to use for streaming channels. This affects how
// the channels are lowered to verilog.
enum class FlowControl {
  // The channel has no flow control. Some external mechanism ensures data is
  // neither lost nor corrupted.
  kNone,

  // The channel uses a ready-valid handshake. A ready signal indicates the
  // receiver is ready to accept data, and a valid signal indicates the data
  // signal is valid. When both ready and valid are asserted a transaction
  // occurs.
  kReadyValid,
};

std::string FlowControlToString(FlowControl fc);
absl::StatusOr<FlowControl> StringToFlowControl(std::string_view str);
std::ostream& operator<<(std::ostream& os, FlowControl fc);

// A channel with FIFO semantics. Send operations add an data entry to the
// channel; receives remove an element from the channel with FIFO ordering.
class StreamingChannel : public Channel {
 public:
  StreamingChannel(std::string_view name, int64_t id, ChannelOps supported_ops,
                   Type* type, absl::Span<const Value> initial_values,
                   std::optional<int64_t> fifo_depth, FlowControl flow_control,
                   const ChannelMetadataProto& metadata)
      : Channel(name, id, supported_ops, ChannelKind::kStreaming, type,
                initial_values, metadata),
        fifo_depth_(fifo_depth),
        flow_control_(flow_control) {}

  virtual bool HasCompletedBlockPortNames() const override {
    if (GetFlowControl() == FlowControl::kReadyValid) {
      return GetBlockName().has_value() && GetDataPortName().has_value() &&
             GetReadyPortName().has_value() && GetValidPortName().has_value();
    }

    return GetBlockName().has_value() && GetDataPortName().has_value();
  }

  std::optional<int64_t> GetFifoDepth() const { return fifo_depth_; }
  void SetFifoDepth(std::optional<int64_t> value) { fifo_depth_ = value; }

  FlowControl GetFlowControl() const { return flow_control_; }
  void SetFlowControl(FlowControl value) { flow_control_ = value; }

 public:
  std::optional<int64_t> fifo_depth_;
  FlowControl flow_control_;
};

// A channel which holds a single value. Values are written to the channel via
// send operations, and receives nondestructively read the most-recently sent
// value. SingleValueChannels are stateless and do not support initial values.
class SingleValueChannel : public Channel {
 public:
  SingleValueChannel(std::string_view name, int64_t id,
                     ChannelOps supported_ops, Type* type,
                     const ChannelMetadataProto& metadata)
      : Channel(name, id, supported_ops, ChannelKind::kSingleValue, type,
                /*initial_values=*/{}, metadata) {}

  virtual bool HasCompletedBlockPortNames() const override {
    return GetBlockName().has_value() && GetDataPortName().has_value();
  }
};

// For use in e.g. absl::StrJoin.
inline void ChannelFormatter(std::string* out, Channel* channel) {
  absl::StrAppend(out, channel->name());
}

inline std::ostream& operator<<(std::ostream& os, const Channel* channel) {
  os << (channel == nullptr ? std::string("<nullptr Channel*>")
                            : channel->name());
  return os;
}

}  // namespace xls

#endif  // XLS_IR_CHANNEL_H_
