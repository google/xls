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

#ifndef XLS_NOC_SIMULATION_COMMON_H_
#define XLS_NOC_SIMULATION_COMMON_H_

#include <cstdint>
#include <limits>
#include <utility>

#include "absl/status/statusor.h"

namespace xls {
namespace noc {

// Used for all id components to denote an invalid id.
constexpr int kNullIdValue = -1;

// Each simulation network component has a kind defined here.
//
// TODO(tedhong): 2021-02-22 Consider removing kNone.
enum class NetworkComponentKind {
  kNone,    // A network component that is invalid or has not been configured.
  kRouter,  // A network router.
  kLink,    // A bundle of wires/registers between network components.
  kNISrc,   // A network interfrace (src) for ingress into the network.
  kNISink,  // A network interface (sink) for egress from the network.
};

// Simulation objects have ports and those ports can be either
// an input or output port.
enum class PortDirection {
  kInput,
  kOutput,
};

// Stores id of a single network.
class NetworkId {
 public:
  // Default constructor initializing this to an invalid id.
  constexpr NetworkId() : id_(kNullIdValue) {}

  // Constructor given an expanded id.
  constexpr explicit NetworkId(uint16_t id) : id_(id) {}

  // Returns true if id is not kInvalid.
  bool IsValid() const;

  // Returns the local id.
  uint16_t id() const { return id_; }

  // Represent this as a 64-bit integer (used for logging/debug).
  uint64_t AsUInt64() const { return static_cast<uint64_t>(id_) << 48; }

  // Validate id is within range and return id object given an unpacked id.
  static absl::StatusOr<NetworkId> ValidateAndReturnId(int64_t id);

  // TODO(tedhong): 2021-06-27 Possible to have this be kMaxIndex instead.
  // Returns maximum id/index that can be encoded.
  static constexpr int64_t MaxIndex() {
    return std::numeric_limits<decltype(id_)>::max() - 1;
  }

  // Hash.
  template <typename H>
  friend H AbslHashValue(H h, NetworkId id) {
    return H::combine(std::move(h), id.id_);
  }

  // Denotes an invalid network id.
  static const NetworkId kInvalid;

 private:
  uint16_t id_;
};

// NetworkId equality.
inline bool operator==(NetworkId lhs, NetworkId rhs) {
  return lhs.id() == rhs.id();
}

// NetworkId inequality.
inline bool operator!=(NetworkId lhs, NetworkId rhs) {
  return lhs.id() != rhs.id();
}

inline bool NetworkId::IsValid() const { return *this != kInvalid; }

// Default constructor is setup to initialize the id to invalid.
constexpr NetworkId NetworkId::kInvalid = NetworkId();

// Asserts that ids can fit within a 64-bit integer
// that way in terms of storage, an id is no worse than
// a 64-bit pointer.
static_assert(sizeof(NetworkId) <= 8 * sizeof(char));

// Id of a network component.
//
// An id is composed of the id of the network this component belongs to
// along with the component's local id.
class NetworkComponentId {
 public:
  // Default constructor initializing this to an invalid id.
  constexpr NetworkComponentId() : network_(kNullIdValue), id_(kNullIdValue) {}

  // Constructor given an non-expanded NetworkId.
  NetworkComponentId(NetworkId network, uint32_t id)
      : network_(network.id()), id_(id) {}

  // Constructor given an expanded id.
  constexpr NetworkComponentId(uint16_t network, uint32_t id)
      : network_(network), id_(id) {}

  // Validate id is within range and return id object given an unpacked id.
  static absl::StatusOr<NetworkComponentId> ValidateAndReturnId(int64_t network,
                                                                int64_t id);

  // Returns true if id is not kInvalid.
  bool IsValid() const;

  // Returns the local id.
  uint32_t id() const { return id_; }

  // Returns the local network id.
  uint16_t network() const { return network_; }

  // Returns the NetworkId of this component.
  NetworkId GetNetworkId() const { return NetworkId(network_); }

  // Represent this as a 64-bit integer (used for logging/debug).
  uint64_t AsUInt64() const {
    return (static_cast<uint64_t>(network_) << 48) |
           (static_cast<uint64_t>(id_) << 16);
  }

  // Returns maximum id/index that can be encoded.
  static constexpr int64_t MaxIndex() {
    return std::numeric_limits<decltype(id_)>::max() - 1;
  }

  // Hash.
  template <typename H>
  friend H AbslHashValue(H h, NetworkComponentId id) {
    return H::combine(std::move(h), id.network_, id.id_);
  }

  // Denotes an invalid component id;
  static const NetworkComponentId kInvalid;

 private:
  // Store the raw unpacked network id so that these parent-id's can
  // be packed per efficiently (see PortId).
  uint16_t network_;
  uint32_t id_;
};

// NetworkComponentId equality.
inline bool operator==(NetworkComponentId lhs, NetworkComponentId rhs) {
  return (lhs.network() == rhs.network()) && (lhs.id() == rhs.id());
}

// NetworkComponentId inequality.
inline bool operator!=(NetworkComponentId lhs, NetworkComponentId rhs) {
  return !(lhs == rhs);
}

inline bool NetworkComponentId::IsValid() const { return *this != kInvalid; }

// Default constructor is setup to initialize the id to invalid.
constexpr NetworkComponentId NetworkComponentId::kInvalid =
    NetworkComponentId();

// Asserts that ids can fit within a 64-bit integer
// that way in terms of storage, an id is no worse than
// a 64-bit pointer.
static_assert(sizeof(NetworkComponentId) <= 8 * sizeof(char));

// Id of a connection -- an edge between two ports.
//
// An id is composed of the id of the network this connectionbelongs to
// along with the connection's local id.
class ConnectionId {
 public:
  // Default constructor initializing this to an invalid id.
  constexpr ConnectionId() : network_(kNullIdValue), id_(kNullIdValue) {}

  // Constructor given an non-expanded NetworkId .
  ConnectionId(NetworkId network, uint32_t id)
      : network_(network.id()), id_(id) {}

  // Constructor given an expanded id.
  constexpr ConnectionId(uint16_t network, uint32_t id)
      : network_(network), id_(id) {}

  // Validate ids are within range and return id object given an unpacked id.
  static absl::StatusOr<ConnectionId> ValidateAndReturnId(int64_t network,
                                                          int64_t id);

  // Returns true if id is not kInvalid.
  bool IsValid() const;

  // Returns the local id.
  uint32_t id() const { return id_; }

  // Returns the local network id.
  uint16_t network() const { return network_; }

  // Returns the NetworkId of this component.
  NetworkId GetNetworkId() const { return NetworkId(network_); }

  // Represent this as a 64-bit integer (used for logging/debug).
  uint64_t AsUInt64() const {
    return (static_cast<uint64_t>(network_) << 48) |
           (static_cast<uint64_t>(id_) << 16);
  }

  // Returns maximum id/index that can be encoded.
  static constexpr int64_t MaxIndex() {
    return std::numeric_limits<decltype(id_)>::max() - 1;
  }

  // Hash.
  template <typename H>
  friend H AbslHashValue(H h, ConnectionId id) {
    return H::combine(std::move(h), id.network_, id.id_);
  }

  // Denotes an invalid connection id.
  static const ConnectionId kInvalid;

 private:
  uint16_t network_;
  uint32_t id_;
};

// ConnectionId equality.
inline bool operator==(ConnectionId lhs, ConnectionId rhs) {
  return (lhs.network() == rhs.network()) && (lhs.id() == rhs.id());
}

// ConnectionId inequality.
inline bool operator!=(ConnectionId lhs, ConnectionId rhs) {
  return !(lhs == rhs);
}

inline bool ConnectionId::IsValid() const { return *this != kInvalid; }

// Default constructor is setup to initialize the id to invalid.
constexpr ConnectionId ConnectionId::kInvalid = ConnectionId();

// Asserts that ids can fit within a 64-bit integer
// that way in terms of storage, an id is no worse than
// a 64-bit pointer.
static_assert(sizeof(ConnectionId) <= 8 * sizeof(char));

// Stores id of a port.
//
// An id is composed of the id of the network, the component
// this port belongs to, along with this port's local id.
struct PortId {
  // Default constructor initializing this to an invalid id.
  constexpr PortId()
      : id_(kNullIdValue), network_(kNullIdValue), component_(kNullIdValue) {}

  // Constructor given an non-expanded NetworkComponentId.
  PortId(NetworkComponentId component, uint16_t id)
      : id_(id),
        network_(component.GetNetworkId().id()),
        component_(component.id()) {}

  // Constructor given an expanded id.
  constexpr PortId(uint16_t network, uint32_t component, uint16_t id)
      : id_(id), network_(network), component_(component) {}

  // Returns true if id is not kInvalid.
  bool IsValid() const;

  // Validate ids are within range and return id object given an unpacked id.
  static absl::StatusOr<PortId> ValidateAndReturnId(int64_t network,
                                                    int64_t component,
                                                    int64_t id);

  // Returns the local id.
  uint16_t id() const { return id_; }

  // Returns the local network id.
  uint16_t network() const { return network_; }

  // Returns the local component id.
  uint32_t component() const { return component_; }

  // Returns the NetworkId of this port.
  NetworkId GetNetworkId() const { return NetworkId(network_); }

  // Returns the NetworkComponentId of this port.
  NetworkComponentId GetNetworkComponentId() const {
    return NetworkComponentId(network_, component_);
  }

  // Represent this as a 64-bit integer (used for logging/debug).
  uint64_t AsUInt64() const {
    return (static_cast<uint64_t>(network_) << 48) |
           (static_cast<uint64_t>(component_) << 16) |
           (static_cast<uint64_t>(id_));
  }

  // Returns maximum id/index that can be encoded.
  static constexpr int64_t MaxIndex() {
    return std::numeric_limits<decltype(id_)>::max() - 1;
  }

  // Hash.
  template <typename H>
  friend H AbslHashValue(H h, PortId id) {
    return H::combine(std::move(h), id.network_, id.component_, id.id_);
  }

  // An invalid port id.
  static const PortId kInvalid;

 private:
  uint16_t id_;
  uint16_t network_;
  uint32_t component_;
};

// PortId equality.
inline bool operator==(PortId lhs, PortId rhs) {
  return (lhs.id() == rhs.id()) && (lhs.network() == rhs.network()) &&
         (lhs.component() == rhs.component());
}

// PortId inequality.
inline bool operator!=(PortId lhs, PortId rhs) { return !(lhs == rhs); }

inline bool PortId::IsValid() const { return *this != kInvalid; }

// Default constructor is setup to initialize the id to invalid.
constexpr PortId PortId::kInvalid = PortId();

// Asserts that ids can fit within a 64-bit integer
// that way in terms of storage, an id is no worse than
// a 64-bit pointer.
static_assert(sizeof(PortId) <= 8 * sizeof(char));

// Stores id of a single traffic flow.
class TrafficFlowId {
 public:
  // Default constructor initializing this to an invalid id.
  constexpr TrafficFlowId() : id_(kNullIdValue) {}

  // Constructor given an expanded id.
  constexpr explicit TrafficFlowId(uint32_t id) : id_(id) {}

  // Returns true if id is not kInvalid.
  bool IsValid() const;

  // Returns the local id.
  uint32_t id() const { return id_; }

  // Represent this as a 64-bit integer (used for logging/debug).
  uint64_t AsUInt64() const { return id_; }

  // Validate id is within range and return id object given an unpacked id.
  static absl::StatusOr<TrafficFlowId> ValidateAndReturnId(int64_t id);

  // Returns maximum id/index that can be encoded.
  static constexpr int64_t MaxIndex() {
    return std::numeric_limits<decltype(id_)>::max() - 1;
  }

  // Hash.
  template <typename H>
  friend H AbslHashValue(H h, TrafficFlowId id) {
    return H::combine(std::move(h), id.id_);
  }

  // Denotes an invalid network id.
  static const TrafficFlowId kInvalid;

 private:
  uint32_t id_;
};

// TrafficFlowId equality.
inline bool operator==(TrafficFlowId lhs, TrafficFlowId rhs) {
  return lhs.id() == rhs.id();
}

// TrafficFlowId inequality.
inline bool operator!=(TrafficFlowId lhs, TrafficFlowId rhs) {
  return lhs.id() != rhs.id();
}

inline bool TrafficFlowId::IsValid() const { return *this != kInvalid; }

// Default constructor is setup to initialize the id to invalid.
constexpr TrafficFlowId TrafficFlowId::kInvalid = TrafficFlowId();

// Asserts that ids can fit within a 64-bit integer
// that way in terms of storage, an id is no worse than
// a 64-bit pointer.
static_assert(sizeof(TrafficFlowId) <= 8 * sizeof(char));

// Stores id of a single traffic mode (group of flows).
class TrafficModeId {
 public:
  // Default constructor initializing this to an invalid id.
  constexpr TrafficModeId() : id_(kNullIdValue) {}

  // Constructor given an expanded id.
  constexpr explicit TrafficModeId(uint32_t id) : id_(id) {}

  // Returns true if id is not kInvalid.
  bool IsValid() const;

  // Returns the local id.
  uint32_t id() const { return id_; }

  // Represent this as a 64-bit integer (used for logging/debug).
  uint64_t AsUInt64() const { return id_; }

  // Validate id is within range and return id object given an unpacked id.
  static absl::StatusOr<TrafficModeId> ValidateAndReturnId(int64_t id);

  // Returns maximum id/index that can be encoded.
  static constexpr int64_t MaxIndex() {
    return std::numeric_limits<decltype(id_)>::max() - 1;
  }

  // Hash.
  template <typename H>
  friend H AbslHashValue(H h, TrafficModeId id) {
    return H::combine(std::move(h), id.id_);
  }

  // Denotes an invalid network id.
  static const TrafficModeId kInvalid;

 private:
  uint32_t id_;
};

// TrafficModeId equality.
inline bool operator==(TrafficModeId lhs, TrafficModeId rhs) {
  return lhs.id() == rhs.id();
}

// TrafficModeId inequality.
inline bool operator!=(TrafficModeId lhs, TrafficModeId rhs) {
  return lhs.id() != rhs.id();
}

inline bool TrafficModeId::IsValid() const { return *this != kInvalid; }

// Default constructor is setup to initialize the id to invalid.
constexpr TrafficModeId TrafficModeId::kInvalid = TrafficModeId();

// Asserts that ids can fit within a 64-bit integer
// that way in terms of storage, an id is no worse than
// a 64-bit pointer.
static_assert(sizeof(TrafficModeId) <= 8 * sizeof(char));

}  // namespace noc
}  // namespace xls

#endif  // XLS_NOC_SIMULATION_COMMON_H_
