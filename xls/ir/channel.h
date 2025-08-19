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
#include <memory>
#include <optional>
#include <ostream>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "xls/ir/channel.pb.h"
#include "xls/ir/channel_ops.h"
#include "xls/ir/type.h"
#include "xls/ir/value.h"

namespace xls {

class Package;

// Enum for the various kinds of channels supported in XLS.
enum class ChannelKind : uint8_t {
  // A channel with FIFO semantics.
  kStreaming,

  // A channel which holds a single value. Values are written to the channel via
  // send operations which overwrites the previously sent values. Receives
  // nondestructively read the most-recently sent value.
  kSingleValue,
};

// Configuration of the actual fifo underlying a streaming channel.
class FifoConfig {
 public:
  constexpr FifoConfig(int64_t depth, bool bypass, bool register_push_outputs,
                       bool register_pop_outputs)
      : depth_(depth),
        bypass_(bypass),
        register_push_outputs_(register_push_outputs),
        register_pop_outputs_(register_pop_outputs) {}

  int64_t depth() const { return depth_; }
  bool bypass() const { return bypass_; }
  bool register_push_outputs() const { return register_push_outputs_; }
  bool register_pop_outputs() const { return register_pop_outputs_; }
  absl::Status Validate() const;

  bool operator==(const FifoConfig& other) const = default;
  bool operator<=>(const FifoConfig& other) const = default;

  static absl::StatusOr<FifoConfig> FromProto(const FifoConfigProto& proto);
  // Serialize this config as a proto. Width is the actual bit-size of the
  // values held in this fifo.
  FifoConfigProto ToProto(int64_t width) const;

  std::string ToString() const;
  // Returns a vector of key-value pairs for the DSLX kwargs for this config,
  // e.g. {{"fifo_depth", "10"}, {"bypass", "true"}, {"register_push_outputs",
  // "true"}, {"register_pop_outputs", "true"}}.
  std::vector<std::pair<std::string, std::string>> GetDslxKwargs() const;

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

enum class FlopKind : int8_t {
  // The input/output is not flopped and is directly connected by wires.
  kNone,
  // Adds a pipeline stage at the beginning or end of the channel to hold
  // inputs or outputs. This is essentially a single-element FIFO.
  kFlop,
  // Adds a skid buffer at the inputs or outputs of the channel. The skid
  // buffer can hold 2 entries
  kSkid,
  // Adds a zero-latency buffer at the beginning or end of the block. This is
  // essentially a single-element FIFO with bypass
  kZeroLatency,
};

template <typename Sink>
void AbslStringify(Sink& sink, FlopKind value) {
  switch (value) {
    case FlopKind::kNone:
      absl::Format(&sink, "none");
      break;
    case FlopKind::kFlop:
      absl::Format(&sink, "flop");
      break;
    case FlopKind::kSkid:
      absl::Format(&sink, "skid");
      break;
    case FlopKind::kZeroLatency:
      absl::Format(&sink, "zero_latency");
      break;
  }
}

absl::StatusOr<std::optional<FlopKind>> FlopKindFromProto(FlopKindProto f);

inline std::string FlopKindToString(FlopKind kind) {
  switch (kind) {
    case FlopKind::kNone:
      return "none";
    case FlopKind::kFlop:
      return "flop";
    case FlopKind::kSkid:
      return "skid";
    case FlopKind::kZeroLatency:
      return "zero_latency";
  }
}
absl::StatusOr<FlopKind> StringToFlopKind(std::string_view str);

// A configuration set for a single channel.
class ChannelConfig {
 public:
  explicit constexpr ChannelConfig(
      std::optional<FifoConfig> fifo_config = std::nullopt,
      std::optional<FlopKind> input_flop_kind = std::nullopt,
      std::optional<FlopKind> output_flop_kind = std::nullopt)
      : fifo_config_(fifo_config),
        input_flop_kind_(input_flop_kind),
        output_flop_kind_(output_flop_kind) {}

  ChannelConfig WithFifoConfig(std::optional<FifoConfig> f) const {
    return ChannelConfig(f, input_flop_kind_, output_flop_kind_);
  }
  const std::optional<FifoConfig>& fifo_config() const { return fifo_config_; }

  ChannelConfig WithInputFlopKind(std::optional<FlopKind> f) const {
    return ChannelConfig(fifo_config_, f, output_flop_kind_);
  }
  std::optional<FlopKind> input_flop_kind() const { return input_flop_kind_; }

  ChannelConfig WithOutputFlopKind(std::optional<FlopKind> f) const {
    return ChannelConfig(fifo_config_, input_flop_kind_, f);
  }
  std::optional<FlopKind> output_flop_kind() const { return output_flop_kind_; }

  bool operator==(const ChannelConfig& other) const = default;

  static absl::StatusOr<ChannelConfig> FromProto(
      const ChannelConfigProto& proto);
  // Serialize this config as a proto. Width is the actual bit-size of the
  // values held in this channel.
  ChannelConfigProto ToProto(int64_t width) const;

  std::string ToString() const;

  // Returns a vector of key-value pairs for the DSLX kwargs for this config,
  // e.g. {{"fifo_depth", "10"}, {"bypass", "true"}, {"input_flop_kind",
  // "none"}}.
  std::vector<std::pair<std::string, std::string>> GetDslxKwargs() const;

  template <typename Sink>
  friend void AbslStringify(Sink& sink, const ChannelConfig& value) {
    absl::Format(&sink, "%s", value.ToString());
  }

  template <typename H>
  friend H AbslHashValue(H h, const ChannelConfig& config) {
    return H::combine(std::move(h), config.fifo_config_,
                      config.input_flop_kind_, config.output_flop_kind_);
  }

 private:
  std::optional<FifoConfig> fifo_config_;
  std::optional<FlopKind> input_flop_kind_;
  std::optional<FlopKind> output_flop_kind_;
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

  template <typename Sink>
  friend void AbslStringify(Sink& sink, const Channel& p) {
    absl::Format(&sink, "%s", p.name());
  }

  // Comparators used for sorting by name/id.
  static bool NameLessThan(const Channel* a, const Channel* b) {
    return a->name() < b->name();
  }
  static bool IdLessThan(const Channel* a, const Channel* b) {
    return a->id() < b->id();
  }

  // Struct form for passing comparators as template arguments.
  struct NameLessThan {
    bool operator()(const Channel* a, const Channel* b) const {
      return Channel::NameLessThan(a, b);
    }
  };
  struct IdLessThan {
    bool operator()(const Channel* a, const Channel* b) const {
      return Channel::IdLessThan(a, b);
    }
  };

 protected:
  Channel(std::string_view name, int64_t id, ChannelOps supported_ops,
          ChannelKind kind, Type* type, absl::Span<const Value> initial_values)
      : name_(name),
        id_(id),
        supported_ops_(supported_ops),
        kind_(kind),
        type_(type),
        initial_values_(initial_values.begin(), initial_values.end()) {}

  void SetName(std::string_view name) { name_ = name; }

  std::string name_;
  int64_t id_;
  ChannelOps supported_ops_;
  ChannelKind kind_;
  Type* type_;
  std::vector<Value> initial_values_;

  // For SetName;
  friend class Package;
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

inline constexpr ChannelStrictness kDefaultChannelStrictness =
    ChannelStrictness::kProvenMutuallyExclusive;

absl::StatusOr<ChannelStrictness> ChannelStrictnessFromString(
    std::string_view text);
std::string ChannelStrictnessToString(ChannelStrictness in);
std::ostream& operator<<(std::ostream& os, ChannelStrictness in);

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
                   ChannelConfig channel_config, FlowControl flow_control,
                   ChannelStrictness strictness)
      : Channel(name, id, supported_ops, ChannelKind::kStreaming, type,
                initial_values),
        channel_config_(std::move(channel_config)),
        flow_control_(flow_control),
        strictness_(strictness) {}

  std::optional<int64_t> GetFifoDepth() const {
    if (channel_config_.fifo_config()) {
      return channel_config_.fifo_config()->depth();
    }
    return std::nullopt;
  }

  const ChannelConfig& channel_config() const { return channel_config_; }
  void channel_config(ChannelConfig value) { channel_config_ = value; }

  FlowControl GetFlowControl() const { return flow_control_; }
  void SetFlowControl(FlowControl value) { flow_control_ = value; }

  ChannelStrictness GetStrictness() const { return strictness_; }
  void SetStrictness(ChannelStrictness value) { strictness_ = value; }

 private:
  ChannelConfig channel_config_;
  FlowControl flow_control_;
  ChannelStrictness strictness_;
};

// A channel which holds a single value. Values are written to the channel via
// send operations, and receives nondestructively read the most-recently sent
// value. SingleValueChannels are stateless and do not support initial values.
class SingleValueChannel final : public Channel {
 public:
  SingleValueChannel(std::string_view name, int64_t id,
                     ChannelOps supported_ops, Type* type)
      : Channel(name, id, supported_ops, ChannelKind::kSingleValue, type,
                /*initial_values=*/{}) {}
};

inline std::ostream& operator<<(std::ostream& os, const Channel* channel) {
  os << (channel == nullptr ? std::string("<nullptr Channel*>")
                            : channel->name());
  return os;
}

enum class ChannelDirection : int8_t { kSend, kReceive };

std::string ChannelDirectionToString(ChannelDirection direction);
absl::StatusOr<ChannelDirection> ChannelDirectionFromString(
    std::string_view str);
std::ostream& operator<<(std::ostream& os, ChannelDirection direction);
inline ChannelDirection InvertChannelDirection(ChannelDirection d) {
  switch (d) {
    case ChannelDirection::kSend:
      return ChannelDirection::kReceive;
    case ChannelDirection::kReceive:
      return ChannelDirection::kSend;
  }
}

// Abstraction representing an interface to a channel. The interface can be
// typed to refer to the send or receive side. With proc-scoped channels (new
// style procs), channel-using operations such as send/receive refer to
// channel interfaces rather than channel objects. In elaboration these
// channel interfaces are bound to channel objects.
//
// TODO(https://github.com/google/xls/issues/869): Reconsider whether channel
// kind and strictness should be held by the channel interface. This is required
// for storing these properties on the interface of new-style procs. An
// alternative would be to have a separate data structure for interface
// channels.
class ChannelInterface {
 public:
  ChannelInterface(std::string_view name, Type* type, ChannelKind kind)
      : name_(name),
        type_(type),
        kind_(kind),
        strictness_(std::nullopt),
        flow_control_(FlowControl::kReadyValid),
        flop_kind_(FlopKind::kNone) {}
  virtual ~ChannelInterface() = default;

  // Like most IR constructs, ChannelInterfaces are passed around by pointer
  // and are not copyable.
  ChannelInterface(const ChannelInterface&) = delete;
  ChannelInterface& operator=(const ChannelInterface&) = delete;

  std::string_view name() const { return name_; }
  Type* type() const { return type_; }

  // TODO(meheff): Remove kind from ChannelInterface. This is a property of the
  // channel not the interface.
  void SetKind(ChannelKind value) { kind_ = value; }
  ChannelKind kind() const { return kind_; }

  void SetStrictness(std::optional<ChannelStrictness> value) {
    strictness_ = value;
  }
  std::optional<ChannelStrictness> strictness() const { return strictness_; }

  void SetFlowControl(FlowControl value) { flow_control_ = value; }
  FlowControl flow_control() const { return flow_control_; }

  void SetFlopKind(FlopKind value) { flop_kind_ = value; }
  FlopKind flop_kind() const { return flop_kind_; }

  virtual ChannelDirection direction() const = 0;

  std::string ToString() const;

 private:
  std::string name_;
  Type* type_;
  ChannelKind kind_;
  std::optional<ChannelStrictness> strictness_;
  FlowControl flow_control_;
  FlopKind flop_kind_;
};

class SendChannelInterface : public ChannelInterface {
 public:
  SendChannelInterface(std::string_view name, Type* type,
                       ChannelKind kind = ChannelKind::kStreaming)
      : ChannelInterface(name, type, kind) {}
  ~SendChannelInterface() override = default;
  std::unique_ptr<SendChannelInterface> Clone(
      std::optional<std::string_view> new_name = std::nullopt) const;
  ChannelDirection direction() const override {
    return ChannelDirection::kSend;
  }
};

class ReceiveChannelInterface : public ChannelInterface {
 public:
  ReceiveChannelInterface(std::string_view name, Type* type,
                          ChannelKind kind = ChannelKind::kStreaming)
      : ChannelInterface(name, type, kind) {}
  ~ReceiveChannelInterface() override = default;
  std::unique_ptr<ReceiveChannelInterface> Clone(
      std::optional<std::string_view> new_name = std::nullopt) const;
  ChannelDirection direction() const override {
    return ChannelDirection::kReceive;
  }
};

// Abstraction gathering a channel with send and receive interfaces.
struct ChannelWithInterfaces {
  Channel* channel;
  SendChannelInterface* send_interface;
  ReceiveChannelInterface* receive_interface;
};

// Type which holds a channel or channel reference. This is a type used to
// transition to proc-scoped channels. In the proc-scoped channel universe all
// uses of channels use ChannelInterfaces rather than Channel objects.
// TODO(https://github.com/google/xls/issues/869): Remove these and replace
// with ChannelInterface* when all procs are new style.
using ChannelRef = std::variant<Channel*, ChannelInterface*>;
using SendChannelRef = std::variant<Channel*, SendChannelInterface*>;
using ReceiveChannelRef = std::variant<Channel*, ReceiveChannelInterface*>;

using AnyChannelRef =
    std::variant<ChannelRef, SendChannelRef, ReceiveChannelRef>;

// Return the name/type/kind/etc of a channel reference.
std::string_view ChannelRefName(ChannelRef ref);
Type* ChannelRefType(ChannelRef ref);
ChannelKind ChannelRefKind(ChannelRef ref);
std::optional<ChannelStrictness> ChannelRefStrictness(ChannelRef ref);
FlowControl ChannelRefFlowControl(ChannelRef ref);

std::string ChannelRefToString(ChannelRef ref);

ChannelRef ToChannelRef(AnyChannelRef ref);

inline std::string_view ChannelRefName(AnyChannelRef ref) {
  return ChannelRefName(ToChannelRef(ref));
}
inline Type* ChannelRefType(AnyChannelRef ref) {
  return ChannelRefType(ToChannelRef(ref));
}
inline ChannelKind ChannelRefKind(AnyChannelRef ref) {
  return ChannelRefKind(ToChannelRef(ref));
}
inline std::optional<ChannelStrictness> ChannelRefStrictness(
    AnyChannelRef ref) {
  return ChannelRefStrictness(ToChannelRef(ref));
}
inline FlowControl ChannelRefFlowControl(AnyChannelRef ref) {
  return ChannelRefFlowControl(ToChannelRef(ref));
}

inline std::string ChannelRefToString(AnyChannelRef ref) {
  return ChannelRefToString(ToChannelRef(ref));
}

}  // namespace xls

#endif  // XLS_IR_CHANNEL_H_
