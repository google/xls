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
#include <optional>
#include <ostream>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "xls/ir/channel.pb.h"
#include "xls/ir/channel_ops.h"
#include "xls/ir/type.h"
#include "xls/ir/value.h"

namespace xls {

// Enum for the various kinds of channels supported in XLS.
enum class ChannelKind : uint8_t {
  // A channel with FIFO semantics.
  kStreaming,

  // A channel which holds a single value. Values are written to the channel via
  // send operations which overwrites the previously sent values. Receives
  // nondestructively read the most-recently sent value.
  kSingleValue,
};

class FifoConfig {
 public:
  FifoConfig(int64_t depth, bool bypass, bool register_push_outputs,
             bool register_pop_outputs);

  int64_t depth() const { return depth_; }
  bool bypass() const { return bypass_; }
  bool register_push_outputs() const { return register_push_outputs_; }
  bool register_pop_outputs() const { return register_pop_outputs_; }

  bool operator==(const FifoConfig& other) const = default;
  bool operator<=>(const FifoConfig& other) const = default;

  FifoConfigProto ToProto(int64_t width) const;

  std::string ToString() const;

  template <typename H>
  friend H AbslHashValue(H h, const FifoConfig& config) {
    return H::combine(std::move(h), config.depth_, config.bypass_,
                      config.register_push_outputs_,
                      config.register_pop_outputs_);
  }

 private:
  int64_t depth_;
  bool bypass_;
  bool register_push_outputs_;
  bool register_pop_outputs_;
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
  virtual ~Channel() = default;

  // Returns the name of the channel.
  std::string_view name() const { return name_; }

  // Returns the ID of the channel. The ID is unique within the scope of a
  // package.
  int64_t id() const { return id_; }

  // Returns the supported ops for the channel: send-only, receive-only, or
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
    return std::nullopt;
  }

  // Returns / sets name of data port this channel is associated with.
  void SetDataPortName(std::string_view name) {
    metadata_.mutable_block_ports()->set_data_port_name(std::string(name));
  }
  std::optional<std::string> GetDataPortName() const {
    if (metadata_.block_ports().has_data_port_name()) {
      return metadata_.block_ports().data_port_name();
    }
    return std::nullopt;
  }

  // Returns / sets name of valid port this channel is associated with.
  void SetValidPortName(std::string_view name) {
    metadata_.mutable_block_ports()->set_valid_port_name(std::string(name));
  }
  std::optional<std::string> GetValidPortName() const {
    if (metadata_.block_ports().has_valid_port_name()) {
      return metadata_.block_ports().valid_port_name();
    }
    return std::nullopt;
  }

  // Returns / sets name of ready port this channel is associated with.
  void SetReadyPortName(std::string_view name) {
    metadata_.mutable_block_ports()->set_ready_port_name(std::string(name));
  }
  std::optional<std::string> GetReadyPortName() const {
    if (metadata_.block_ports().has_valid_port_name()) {
      return metadata_.block_ports().ready_port_name();
    }
    return std::nullopt;
  }

  ChannelKind kind() const { return kind_; }

  virtual std::string ToString() const;

  template <typename Sink>
  friend void AbslStringify(Sink& sink, const Channel& p) {
    absl::Format(&sink, "%s", p.name());
  }

  // Comparator used for sorting by name.
  struct NameLessThan {
    bool operator()(const Channel* a, const Channel* b) const {
      return a->name() < b->name();
    }
  };

 protected:
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
enum class FlowControl : uint8_t {
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

// When multiple of the same channel operations happen on the same channel,
// scheduling legalizes them through a combination of:
//  1. Requiring proven properties of the channel operations.
//  2. Runtime checks (assertions) that properties of the channel are true.
//  3. Arbitrary selection of priority between operations.
//
// Note that this does not apply to e.g. a send and receive on an internal
// SendReceive channel. This only applies when multiples of the same channel
// operation are being performed on the same channel.
enum class ChannelStrictness : uint8_t {
  // Requires that channel operations be formally proven to be mutually
  // exclusive by Z3.
  kProvenMutuallyExclusive,
  // Requires that channel operations be mutually exclusive- enforced during
  // simulation via assertions.
  kRuntimeMutuallyExclusive,
  // For each proc, requires a total order on all operations on a channel. Note:
  // operations from different procs will not be ordered with respect to each
  // other.
  kTotalOrder,
  // Requires that a total order exists on every subset of channel operations
  // that fires at runtime. Adds assertions.
  kRuntimeOrdered,
  // For each proc, an arbitrary (respecting existing token relationships)
  // static priority is chosen for multiple channel operations. Operations
  // coming from different procs must be mutually exclusive (enforced via
  // assertions).
  kArbitraryStaticOrder,
};

absl::StatusOr<ChannelStrictness> ChannelStrictnessFromString(
    std::string_view text);
std::string ChannelStrictnessToString(ChannelStrictness in);

inline bool AbslParseFlag(std::string_view text, ChannelStrictness* result,
                          std::string* error) {
  absl::StatusOr<ChannelStrictness> channel_strictness =
      ChannelStrictnessFromString(text);
  if (channel_strictness.ok()) {
    *result = *std::move(channel_strictness);
    return true;
  }
  *error = channel_strictness.status().ToString();
  return false;
}

inline std::string AbslUnparseFlag(
    const ChannelStrictness& channel_strictness) {
  return ChannelStrictnessToString(channel_strictness);
}

// A channel with FIFO semantics. Send operations add an data entry to the
// channel; receives remove an element from the channel with FIFO ordering.
class StreamingChannel final : public Channel {
 public:
  StreamingChannel(std::string_view name, int64_t id, ChannelOps supported_ops,
                   Type* type, absl::Span<const Value> initial_values,
                   std::optional<FifoConfig> fifo_config,
                   FlowControl flow_control, ChannelStrictness strictness,
                   const ChannelMetadataProto& metadata)
      : Channel(name, id, supported_ops, ChannelKind::kStreaming, type,
                initial_values, metadata),
        fifo_config_(fifo_config),
        flow_control_(flow_control),
        strictness_(strictness) {}

  bool HasCompletedBlockPortNames() const final {
    if (GetFlowControl() == FlowControl::kReadyValid) {
      return GetBlockName().has_value() && GetDataPortName().has_value() &&
             GetReadyPortName().has_value() && GetValidPortName().has_value();
    }

    return GetBlockName().has_value() && GetDataPortName().has_value();
  }

  std::optional<int64_t> GetFifoDepth() const {
    if (fifo_config_.has_value()) {
      return fifo_config_->depth();
    }
    return std::nullopt;
  }

  const std::optional<FifoConfig>& fifo_config() const { return fifo_config_; }
  void fifo_config(FifoConfig value) { fifo_config_ = value; }

  FlowControl GetFlowControl() const { return flow_control_; }
  void SetFlowControl(FlowControl value) { flow_control_ = value; }

  ChannelStrictness GetStrictness() const { return strictness_; }
  void SetStrictness(ChannelStrictness value) { strictness_ = value; }

 private:
  std::optional<FifoConfig> fifo_config_;
  FlowControl flow_control_;
  ChannelStrictness strictness_;
};

// A channel which holds a single value. Values are written to the channel via
// send operations, and receives nondestructively read the most-recently sent
// value. SingleValueChannels are stateless and do not support initial values.
class SingleValueChannel final : public Channel {
 public:
  SingleValueChannel(std::string_view name, int64_t id,
                     ChannelOps supported_ops, Type* type,
                     const ChannelMetadataProto& metadata)
      : Channel(name, id, supported_ops, ChannelKind::kSingleValue, type,
                /*initial_values=*/{}, metadata) {}

  bool HasCompletedBlockPortNames() const final {
    return GetBlockName().has_value() && GetDataPortName().has_value();
  }
};

inline std::ostream& operator<<(std::ostream& os, const Channel* channel) {
  os << (channel == nullptr ? std::string("<nullptr Channel*>")
                            : channel->name());
  return os;
}

enum class Direction : int8_t { kSend, kReceive };

std::string DirectionToString(Direction direction);

// Abstraction representing a reference to a channel. The reference can be
// typed to refer to the send or receive side. With proc-scoped channels (new
// style procs), channel-using operations such as send/receive refer to
// channel references rather than channel objects. In elaboration these
// channel references are bound to channel objects.
class ChannelReference {
 public:
  ChannelReference(std::string_view name, Type* type, ChannelKind kind)
      : name_(name), type_(type), kind_(kind) {}
  virtual ~ChannelReference() {}

  // Like most IR constructs, ChannelReferences are passed around by pointer
  // and are not copyable.
  ChannelReference(const ChannelReference&) = delete;
  ChannelReference& operator=(const ChannelReference&) = delete;

  std::string_view name() const { return name_; }
  Type* type() const { return type_; }
  ChannelKind kind() const { return kind_; }
  virtual Direction direction() const = 0;

  std::string ToString() const;

 private:
  std::string name_;
  Type* type_;
  ChannelKind kind_;
};

class SendChannelReference : public ChannelReference {
 public:
  SendChannelReference(std::string_view name, Type* type, ChannelKind kind)
      : ChannelReference(name, type, kind) {}
  ~SendChannelReference() override {}
  Direction direction() const override { return Direction::kSend; }
};

class ReceiveChannelReference : public ChannelReference {
 public:
  ReceiveChannelReference(std::string_view name, Type* type, ChannelKind kind)
      : ChannelReference(name, type, kind) {}
  ~ReceiveChannelReference() override {}
  Direction direction() const override { return Direction::kReceive; }
};

// Abstraction holding pointers hold both ends of a particular channel.
struct ChannelReferences {
  SendChannelReference* send_ref;
  ReceiveChannelReference* receive_ref;
};

// Type which holds a channel or channel reference. This is a type used to
// transition to proc-scoped channels. In the proc-scoped channel universe all
// uses of channels use ChannelReferences rather than Channel objects.
// TODO(https://github.com/google/xls/issues/869): Remove these and replace
// with ChannelReference* when all procs are new style.
using ChannelRef = std::variant<Channel*, ChannelReference*>;
using SendChannelRef = std::variant<Channel*, SendChannelReference*>;
using ReceiveChannelRef = std::variant<Channel*, ReceiveChannelReference*>;

// Return the name/type/kind of a channel reference.
std::string_view ChannelRefName(ChannelRef ref);
Type* ChannelRefType(ChannelRef ref);
ChannelKind ChannelRefKind(ChannelRef ref);

}  // namespace xls

#endif  // XLS_IR_CHANNEL_H_
