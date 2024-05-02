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

#ifndef XLS_NOC_SIMULATION_NETWORK_GRAPH_H_
#define XLS_NOC_SIMULATION_NETWORK_GRAPH_H_

#include <cstdint>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xls/noc/simulation/common.h"

namespace xls {
namespace noc {

// This file defines simulation objects used to describe the
// main graph of the simulation containing
//   Networks, Components (w/ Ports), and Connections between them.
//
// APIs are provided to traverse the graph:
//  From a network
//    Iterate through components and connections
//  From a connection
//    Find the ports and hence component it is attached to
//  From a components
//    Find ports
//  From a port
//    Find a connection
//
// Dangling ports/connections are possible,  in that case, the corresponding
// connection/port has an invalid ID.
//
class NetworkManager;

// A port is how a NetworkComponent connects to other components.
class Port {
 public:
  // Construct a Port object.
  Port(PortId id, PortDirection dir) : id_(id), dir_(dir) {}

  // Returns id of this Port.
  PortId id() const { return id_; }

  // Returns id of the connection attached to this Port.
  ConnectionId connection() const { return connection_; }

  // Returns direction of this Port.
  PortDirection direction() const { return dir_; }

  // Print to stdout information about this object.
  void Dump(int indent_level = 0) const;

  // Set connection of this Port.
  //  - used by Connection::Attach()
  void SetAttached(ConnectionId conn) { connection_ = conn; }

  // Set connection of this Port to an invalid Connection.
  //  - used by Connection::DetachSrc()
  //  - used by Connection::DetachSink()
  //  - used by Connection::Attach()
  void SetDetached() { connection_ = ConnectionId::kInvalid; }

 private:
  PortId id_;
  PortDirection dir_;  // Input or output.
  ConnectionId connection_;
};

// A connection is a relationship between two ports.
class Connection {
 public:
  // Construct a Connection object.
  Connection(NetworkManager* mgr, ConnectionId id) : mgr_(mgr), id_(id) {}

  // Returns id of this Connection.
  ConnectionId id() const { return id_; }

  // Returns src port id.
  PortId src() const { return src_; }

  // Returns sink port id.
  PortId sink() const { return sink_; }

  // Detaches this connection from its src port.
  void DetachSrc();

  // Detaches this connection from its sink port.
  void DetachSink();

  // Attaches this connection to the port, detaching it from the prior
  // attached port if needed.
  PortId Attach(PortId port);

  // Print to stdout information about this object.
  void Dump(int indent_level = 0) const;

 private:
  NetworkManager* mgr_;
  ConnectionId id_;
  PortId src_;
  PortId sink_;
};

// A NetworkComponent is a specific network block and has ports.
class NetworkComponent {
 public:
  // Construct a NetworkComponent object.
  NetworkComponent(NetworkComponentId id, NetworkComponentKind kind)
      : id_(id), kind_(kind) {}

  // Creates/adds a port to a network component.
  absl::StatusOr<PortId> CreatePort(PortDirection dir);

  // Get Port object given an id.
  Port& GetPort(PortId id);

  // Get Port object given an id.
  const Port& GetPort(PortId id) const;

  // Returns id of this NetworkComponent.
  NetworkComponentId id() const { return id_; }

  // Returns kind (switch, link, etc...) of this NetworkCompnent.
  NetworkComponentKind kind() const { return kind_; }

  // Get the id of the i-th Port.
  PortId GetPortIdByIndex(int64_t i) const {
    return PortId(id_.GetNetworkId().id(),  // network
                  id_.id(),                 // component
                  i);
  }

  // Get the object of the i-th Port.
  Port& GetPortByIndex(int64_t i) { return ports_[i]; }

  // Get the object of the i-th Port.
  const Port& GetPortByIndex(int64_t i) const { return ports_[i]; }

  // Returns a vector of all managed port ids.
  std::vector<PortId> GetPortIds() const;

  // Returns a vector of all managed output port ids.
  std::vector<PortId> GetOutputPortIds() const;

  // Returns a vector of all managed output port ids.
  std::vector<PortId> GetInputPortIds() const;

  // Returns a span of all Port objects.
  absl::Span<Port> GetPorts() { return absl::MakeSpan(ports_); }

  // Returns a span of all Port objects.
  absl::Span<const Port> GetPorts() const { return absl::MakeSpan(ports_); }

  // Count of ports.
  int64_t GetPortCount() const { return ports_.size(); }

  // Print to stdout information about this object.
  void Dump(int indent_level = 0) const;

 private:
  NetworkComponentId id_;
  NetworkComponentKind kind_;

  std::vector<Port> ports_;
};

// A Network is responsible for storing connections and components.
class Network {
 public:
  // Construct a Network object.
  //  mgr is a pointer to the owner of this Network.
  Network(NetworkManager* mgr, NetworkId id) : mgr_(mgr), id_(id) {}

  // Create a new network component under a network.
  absl::StatusOr<NetworkComponentId> CreateNetworkComponent(
      NetworkComponentKind kind);

  // Creates/adds a port to a network component.
  absl::StatusOr<PortId> CreatePort(NetworkComponentId component,
                                    PortDirection dir);

  // Create a floating connection associated with a network.
  // Afterwards, Attach() may be used to associate the connection
  // with a port.
  absl::StatusOr<ConnectionId> CreateConnection();

  // Create a connection between two ports.
  // Either from or to PortIds may be invalid to create a dangling connection.
  absl::StatusOr<ConnectionId> CreateConnection(PortId src, PortId sink);

  // Associate a connection with a specific port.
  // If necessary, this will dissociate the connection with its exiting
  // port.
  void Attach(ConnectionId conn, PortId port);

  // Get NetworkComponent object given an id.
  NetworkComponent& GetNetworkComponent(NetworkComponentId id);

  // Get NetworkComponent object given an id.
  const NetworkComponent& GetNetworkComponent(NetworkComponentId id) const;

  // Get Connection object given an id.
  Connection& GetConnection(ConnectionId id);

  // Get Connection object given an id.
  const Connection& GetConnection(ConnectionId id) const;

  // Get Port object given an id.
  Port& GetPort(PortId id);

  // Get Port object given an id.
  const Port& GetPort(PortId id) const;

  // Returns id of this Network.
  NetworkId id() const { return id_; }

  // Get the id of the i-th NetworkComponent.
  NetworkComponentId GetNetworkComponentIdByIndex(int64_t i) const {
    return NetworkComponentId(id_.id(), i);
  }

  // Get the object of the i-th NetworkComponent.
  NetworkComponent& GetNetworkComponentByIndex(int64_t i) {
    return components_[i];
  }

  // Get the object of the i-th NetworkComponent.
  const NetworkComponent& GetNetworkComponentByIndex(int64_t i) const {
    return components_[i];
  }

  // Return a vector of all managed NetworkComponent ids.
  std::vector<NetworkComponentId> GetNetworkComponentIds() const {
    std::vector<NetworkComponentId> ret(components_.size());
    for (int64_t i = 0; i < components_.size(); ++i) {
      ret[i] = NetworkComponentId(id_.id(), i);
    }
    return ret;
  }

  // Return a span of all NetworkComponents.
  absl::Span<NetworkComponent> GetNetworkComponents() {
    return absl::MakeSpan(components_);
  }

  // Return a span of all NetworkComponents.
  absl::Span<const NetworkComponent> GetNetworkComponents() const {
    return absl::MakeSpan(components_);
  }

  // Get the id of the i-th Connection.
  ConnectionId GetConnectionIdByIndex(int64_t i) const {
    return ConnectionId(id_.id(), i);
  }

  // Get the object of the i-th Connection.
  Connection& GetConnectionByIndex(int64_t i) { return connections_[i]; }

  // Get the object of the i-th Connection.
  const Connection& GetConnectionByIndex(int64_t i) const {
    return connections_[i];
  }

  // Return a vector of all managed Connection ids.
  std::vector<ConnectionId> GetConnectionIds() const {
    std::vector<ConnectionId> ret(components_.size());
    for (int64_t i = 0; i < components_.size(); ++i) {
      ret[i] = ConnectionId(id_.id(), i);
    }
    return ret;
  }

  // Return a span of all Connections.
  absl::Span<Connection> GetConnections() {
    return absl::MakeSpan(connections_);
  }

  // Return a span of all Connections.
  absl::Span<const Connection> GetConnections() const {
    return absl::MakeSpan(connections_);
  }

  // Count of networks components managed by this object.
  int64_t GetNetworkComponentCount() const { return components_.size(); }

  // Count of connections managed by this object.
  int64_t GetConnectionCount() const { return connections_.size(); }

  // Print to stdout information about this object.
  void Dump(int indent_level = 0) const;

 private:
  NetworkManager* mgr_;
  NetworkId id_;

  std::vector<NetworkComponent> components_;
  std::vector<Connection> connections_;
};

// A NetworkManager is responsible for storing all networks and serves
// as the entry port for accessing and creating network graph objects.
class NetworkManager {
 public:
  // Create a new network.
  absl::StatusOr<NetworkId> CreateNetwork();

  // Create a new network component under a network.
  absl::StatusOr<NetworkComponentId> CreateNetworkComponent(
      NetworkId network, NetworkComponentKind kind);

  // Creates/adds a port to a network component.
  absl::StatusOr<PortId> CreatePort(NetworkComponentId component,
                                    PortDirection dir);

  // Create a connection between two ports.
  // Either from or to PortIds may be invalid to create a dangling connection.
  absl::StatusOr<ConnectionId> CreateConnection(NetworkId network, PortId src,
                                                PortId sink);

  // Create a floating connection associated with a network.
  // Afterwards, Attach() may be used to associate the connection
  // with a port.
  absl::StatusOr<ConnectionId> CreateConnection(NetworkId network);

  // Associate a connection with a specific port.
  // If necessary, this will dissociate the connection with its exiting
  // port.
  void Attach(ConnectionId conn, PortId port);

  // Get Network object given an id.
  Network& GetNetwork(NetworkId id);

  // Get Network object given an id.
  const Network& GetNetwork(NetworkId id) const;

  // Get NetworkComponent object given an id.
  NetworkComponent& GetNetworkComponent(NetworkComponentId id);

  // Get NetworkComponent object given an id.
  const NetworkComponent& GetNetworkComponent(NetworkComponentId id) const;

  // Get Connection object given an id.
  Connection& GetConnection(ConnectionId id);

  // Get Connection object given an id.
  const Connection& GetConnection(ConnectionId id) const;

  // Get Port object given an id.
  Port& GetPort(PortId id);

  // Get Port object given an id.
  const Port& GetPort(PortId id) const;

  // Get the id of the i-th network.
  NetworkId GetNetworkIdByIndex(int64_t i) const { return NetworkId(i); }

  // Get the object of the i-th network.
  Network& GetNetworkByIndex(int64_t i) { return networks_[i]; }

  // Get the object of the i-th network.
  const Network& GetNetworkByIndex(int64_t i) const { return networks_[i]; }

  // Returns a vector of all managed network ids.
  std::vector<NetworkId> GetNetworkIds() const {
    std::vector<NetworkId> ret(networks_.size());
    for (int64_t i = 0; i < networks_.size(); ++i) {
      ret[i] = NetworkId(i);
    }
    return ret;
  }

  // Returns a span of all network objects.
  absl::Span<Network> GetNetworks() { return absl::MakeSpan(networks_); }

  // Returns a span of all network objects.
  absl::Span<const Network> GetNetworks() const {
    return absl::MakeSpan(networks_);
  }

  // Count of networks managed by this object.
  int64_t GetNetworkCount() const { return networks_.size(); }

  // Print to stdout information about the network graph.
  void Dump(int indent_level = 0) const;

 private:
  std::vector<Network> networks_;
};

}  // namespace noc
}  // namespace xls

#endif  // XLS_NOC_SIMULATION_NETWORK_GRAPH_H_
