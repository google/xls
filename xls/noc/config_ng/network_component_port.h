// Copyright 2021 The XLS Authors
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

#ifndef XLS_NOC_CONFIG_NG_NETWORK_COMPONENT_PORT_H_
#define XLS_NOC_CONFIG_NG_NETWORK_COMPONENT_PORT_H_

#include "absl/container/flat_hash_set.h"

namespace xls::noc {

// Forward declaration: a port has a reference to a network component and a set
// of pointers to connections.
class NetworkConnection;
class NetworkComponent;

// Options for a port direction. The direction of the port is from the
// perspective of an external entity communicating with the port.
enum class PortDirection {
  kInput,
  kOutput,
};

// Options for a port type. The port type dictates the type of signal that the
// port transfers.
// TODO (vmirian) 02-05-21 add support for a dual type port (data&control).
// An example scenario for a dual type port is the transmission of flow
// control information over a data transmission line between routers. Add a
// separate flag for data and control that are modified independently.
enum class PortType {
  kData,
  kControl,
};

// The representation of a port for a network component. Instances of this class
// are used in the network component and network connection classes.
class NetworkComponentPort final {
 public:
  NetworkComponentPort(const NetworkComponentPort&) = delete;
  NetworkComponentPort& operator=(const NetworkComponentPort&) = delete;

  NetworkComponentPort(const NetworkComponentPort&&) = delete;
  NetworkComponentPort& operator=(const NetworkComponentPort&&) = delete;

  // Gets the network component.
  NetworkComponent& GetComponent() const;

  // Return true if the direction is input. Otherwise, returns false.
  bool IsInput() const;

  // Return true if the direction is output. Otherwise, returns false.
  bool IsOutput() const;

  // Return true if the type is data. Otherwise, returns false.
  bool IsData() const;

  // Return true if the type is control. Otherwise, returns false.
  bool IsControl() const;

  // Gets the connections. The objects are guaranteed to be non-null.
  const absl::flat_hash_set<const NetworkConnection*>& GetConnections() const;

 private:
  // The private constructor is accessed by the network component.
  friend class NetworkComponent;
  // network_component: the network component that the port belongs to. Does not
  // take ownership of the network component. The network component must refer
  // to a valid object that outlives this object.
  NetworkComponentPort(NetworkComponent* network_component, PortType port_type,
                       PortDirection direction);

  // Gets the direction of the network component port.
  PortDirection GetDirection() const;
  // Gets the type of the network component port.
  PortType GetType() const;

  // The add connection and remove connection functions are accessed by the
  // network connection class.
  friend class NetworkConnection;
  // There is a single entry for a connection.
  absl::flat_hash_set<const NetworkConnection*> connections_;
  // Adding the same connection instance several times results in a single
  // entry.
  void AddConnection(NetworkConnection* connection);
  void RemoveConnection(NetworkConnection* connection);

  // member attributes
  NetworkComponent& component_;
  // TODO(vmirian) 02-05-21 make data and control port
  PortType type_;
  PortDirection direction_;
};

}  // namespace xls::noc

#endif  // XLS_NOC_CONFIG_NG_NETWORK_COMPONENT_PORT_H_
