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

  // A channel corresponding to a port in code-generated block.
  // TODO(https://github.com/google/xls/issues/410): 2021/05/07 Remove when
  // blocks are supported.
  kPort,

  // A channel corresponding to a hardware register in codegen.
  // TODO(https://github.com/google/xls/issues/410): 2021/05/07 Remove when
  // blocks are supported.
  kRegister,

  // An abstract channel gathering together a data port channel
  // with flow control ports (e.g., ready and valid).
  // TODO(meheff): 2021/06/24 Remove until there is a motivating use case.
  kLogical,

  // A channel which holds a single value. Values are written to the channel via
  // send operations which overwrites the previously sent values. Receives
  // nondestructively read the most-recently sent value.
  kSingleValue,
};

std::string ChannelKindToString(ChannelKind kind);
absl::StatusOr<ChannelKind> StringToChannelKind(absl::string_view str);
std::ostream& operator<<(std::ostream& os, ChannelKind kind);

// Abstraction describing a channel in XLS IR. Channels are a mechanism for
// communicating between procs or between procs and components outside of
// XLS. Send and receive nodes in procs are associated with a particular
// channel. The channel data structure carries information about how
// communication occurs over the channel.
class Channel {
 public:
  Channel(absl::string_view name, int64_t id, ChannelOps supported_ops,
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
absl::StatusOr<FlowControl> StringToFlowControl(absl::string_view str);
std::ostream& operator<<(std::ostream& os, FlowControl fc);

// A channel with FIFO semantics. Send operations add an data entry to the
// channel; receives remove an element from the channel with FIFO ordering.
class StreamingChannel : public Channel {
 public:
  StreamingChannel(absl::string_view name, int64_t id, ChannelOps supported_ops,
                   Type* type, absl::Span<const Value> initial_values,
                   FlowControl flow_control,
                   const ChannelMetadataProto& metadata)
      : Channel(name, id, supported_ops, ChannelKind::kStreaming, type,
                initial_values, metadata),
        flow_control_(flow_control) {}

  FlowControl flow_control() const { return flow_control_; }

 public:
  FlowControl flow_control_;
};

// A channel representing a port in a generated block. PortChannels do not
// support initial values.
// TODO(https://github.com/google/xls/issues/410): 2021/05/07 Remove when
// blocks are supported.
class PortChannel : public Channel {
 public:
  PortChannel(absl::string_view name, int64_t id, ChannelOps supported_ops,
              Type* type, const ChannelMetadataProto& metadata)
      : Channel(name, id, supported_ops, ChannelKind::kPort, type,
                /*initial_values=*/{}, metadata) {}

  // The ordinal position of the port among all ports in the module. The first
  // port has position zero. If not specified, ports are ordered by channel ID.
  void SetPosition(int64_t position) { position_ = position; }
  absl::optional<int64_t> GetPosition() const { return position_; }

 private:
  absl::optional<int64_t> position_;
};

// A channel representing a register within a block. All register channels are
// send/receive (ChannelOps::kSendReceive) and may have at most one initial
// value (the reset value).
class RegisterChannel : public Channel {
 public:
  RegisterChannel(absl::string_view name, int64_t id, Type* type,
                  absl::optional<Value> reset_value,
                  const ChannelMetadataProto& metadata)
      : Channel(
            name, id, ChannelOps::kSendReceive, ChannelKind::kRegister, type,
            reset_value.has_value() ? std::vector<Value>({reset_value.value()})
                                    : std::vector<Value>(),
            metadata) {}

  absl::optional<Value> reset_value() const {
    if (initial_values().empty()) {
      return absl::nullopt;
    } else {
      return initial_values().front();
    }
  }
};

// A channel representing a data port with ready/valid ports for flow control. A
// logical channel is simply a logical grouping of these ports and does not have
// directly associated send/receive nodes.
class LogicalChannel : public Channel {
 public:
  LogicalChannel(absl::string_view name, int64_t id, PortChannel* ready_channel,
                 PortChannel* valid_channel, PortChannel* data_channel,
                 const ChannelMetadataProto& metadata)
      : Channel(name, id, data_channel->supported_ops(), ChannelKind::kLogical,
                data_channel->type(), data_channel->initial_values(),
                metadata) {}

  PortChannel* ready_channel() const { return ready_channel_; }
  PortChannel* valid_channel() const { return valid_channel_; }
  PortChannel* data_channel() const { return data_channel_; }

  std::string ToString() const override {
    XLS_LOG(FATAL) << "LogicalChannel::ToString() not implemented.";
  }

 private:
  PortChannel* ready_channel_;
  PortChannel* valid_channel_;
  PortChannel* data_channel_;
};

// A channel which holds a single value. Values are written to the channel via
// send operations, and receives nondestructively read the most-recently sent
// value. SingleValueChannels are stateless and do not support initial values.
class SingleValueChannel : public Channel {
 public:
  SingleValueChannel(absl::string_view name, int64_t id,
                     ChannelOps supported_ops, Type* type,
                     const ChannelMetadataProto& metadata)
      : Channel(name, id, supported_ops, ChannelKind::kSingleValue, type,
                /*initial_values=*/{}, metadata) {}
};

}  // namespace xls

#endif  // XLS_IR_CHANNEL_H_
