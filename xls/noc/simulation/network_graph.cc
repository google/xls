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

#include "xls/noc/simulation/network_graph.h"

#include <cstdint>
#include <string>
#include <vector>

#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"

namespace {

// Creates a string with 4 characters per indent.
std::string IndentLevel(int indent_level, char c = ' ') {
  return std::string(indent_level * 4, c);
}

}  // namespace

namespace xls {
namespace noc {

// - Network Manager
absl::StatusOr<NetworkId> NetworkManager::CreateNetwork() {
  int64_t n_networks = networks_.size();

  XLS_ASSIGN_OR_RETURN(NetworkId next_id,
                       NetworkId::ValidateAndReturnId(n_networks));
  networks_.emplace_back(this, next_id);
  return next_id;
}

absl::StatusOr<NetworkComponentId> NetworkManager::CreateNetworkComponent(
    NetworkId network, NetworkComponentKind kind) {
  XLS_RET_CHECK(network.id() < networks_.size());

  return networks_[network.id()].CreateNetworkComponent(kind);
}

absl::StatusOr<PortId> NetworkManager::CreatePort(NetworkComponentId component,
                                                  PortDirection dir) {
  XLS_RET_CHECK(component.GetNetworkId().id() < networks_.size());

  return networks_[component.GetNetworkId().id()].CreatePort(component, dir);
}

absl::StatusOr<ConnectionId> NetworkManager::CreateConnection(
    NetworkId network) {
  XLS_RET_CHECK(network.id() < networks_.size());

  return networks_.at(network.id()).CreateConnection();
}

absl::StatusOr<ConnectionId> NetworkManager::CreateConnection(NetworkId network,
                                                              PortId src,
                                                              PortId sink) {
  XLS_RET_CHECK(network.id() < networks_.size());

  return networks_.at(network.id()).CreateConnection(src, sink);
}

void NetworkManager::Attach(ConnectionId conn, PortId p) {
  return networks_.at(conn.GetNetworkId().id()).Attach(conn, p);
}

Network& NetworkManager::GetNetwork(NetworkId id) {
  return networks_.at(id.id());
}

const Network& NetworkManager::GetNetwork(NetworkId id) const {
  return networks_.at(id.id());
}

NetworkComponent& NetworkManager::GetNetworkComponent(NetworkComponentId id) {
  return networks_.at(id.GetNetworkId().id()).GetNetworkComponent(id);
}

const NetworkComponent& NetworkManager::GetNetworkComponent(
    NetworkComponentId id) const {
  return networks_.at(id.GetNetworkId().id()).GetNetworkComponent(id);
}

Connection& NetworkManager::GetConnection(ConnectionId id) {
  return networks_.at(id.GetNetworkId().id()).GetConnection(id);
}

const Connection& NetworkManager::GetConnection(ConnectionId id) const {
  return networks_.at(id.GetNetworkId().id()).GetConnection(id);
}

Port& NetworkManager::GetPort(PortId id) {
  return networks_.at(id.GetNetworkId().id()).GetPort(id);
}

const Port& NetworkManager::GetPort(PortId id) const {
  return networks_.at(id.GetNetworkId().id()).GetPort(id);
}

void NetworkManager::Dump(int indent_level) const {
  VLOG(1) << IndentLevel(indent_level) << "Network Manager:";

  int64_t i = 0;
  for (const auto& network : networks_) {
    VLOG(1) << IndentLevel(indent_level) << "- #" << i << " Network:";

    network.Dump(indent_level + 1);

    ++i;
  }
}

// - Network

absl::StatusOr<NetworkComponentId> Network::CreateNetworkComponent(
    NetworkComponentKind kind) {
  int64_t n_network_components = components_.size();

  XLS_ASSIGN_OR_RETURN(
      NetworkComponentId next_id,
      NetworkComponentId::ValidateAndReturnId(id_.id(),  // NetworkId
                                              n_network_components));
  components_.emplace_back(next_id, kind);
  return next_id;
}

absl::StatusOr<PortId> Network::CreatePort(NetworkComponentId component,
                                           PortDirection dir) {
  if (component.GetNetworkId() != id_) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Cannot CreatePort on component %x, "
                        "expected network %x",
                        component.AsUInt64(), id_.id()));
  }

  XLS_RET_CHECK(component.id() < components_.size());

  return components_[component.id()].CreatePort(dir);
}

absl::StatusOr<ConnectionId> Network::CreateConnection(PortId src,
                                                       PortId sink) {
  XLS_ASSIGN_OR_RETURN(ConnectionId conn, CreateConnection());
  XLS_RET_CHECK(conn.id() < connections_.size());

  if (src.GetNetworkId() != id_) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Cannot CreateConnectio from src port %x, "
                        "expected network %x",
                        src.AsUInt64(), id_.id()));
  }

  if (sink.GetNetworkId() != id_) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Cannot CreateConnection to sink port %x, "
                        "expected network %x",
                        sink.AsUInt64(), id_.id()));
  }

  if (conn.IsValid()) {
    connections_[conn.id()].Attach(src);
    connections_[conn.id()].Attach(sink);
  }

  return conn;
}

absl::StatusOr<ConnectionId> Network::CreateConnection() {
  uint64_t n_connections = connections_.size();

  XLS_ASSIGN_OR_RETURN(ConnectionId next_id,
                       ConnectionId::ValidateAndReturnId(id_.id(),  // NetworkId
                                                         n_connections));
  connections_.emplace_back(mgr_, next_id);
  return next_id;
}

void Network::Attach(ConnectionId conn, PortId p) {
  connections_.at(conn.id()).Attach(p);
}

NetworkComponent& Network::GetNetworkComponent(NetworkComponentId id) {
  return components_.at(id.id());
}

const NetworkComponent& Network::GetNetworkComponent(
    NetworkComponentId id) const {
  return components_.at(id.id());
}

Connection& Network::GetConnection(ConnectionId id) {
  return connections_.at(id.id());
}

const Connection& Network::GetConnection(ConnectionId id) const {
  return connections_.at(id.id());
}

Port& Network::GetPort(PortId id) {
  return components_.at(id.GetNetworkComponentId().id()).GetPort(id);
}

const Port& Network::GetPort(PortId id) const {
  return components_.at(id.GetNetworkComponentId().id()).GetPort(id);
}

void Network::Dump(int indent_level) const {
  VLOG(1) << IndentLevel(indent_level)
          << absl::StreamFormat("Network id %x", id_.AsUInt64());

  int64_t i = 0;
  for (const auto& component : components_) {
    VLOG(1) << IndentLevel(indent_level) << "- #" << i << " Component:";
    component.Dump(indent_level + 1);
    ++i;
  }

  i = 0;
  for (auto connection : connections_) {
    VLOG(1) << IndentLevel(indent_level) << "- #" << i << " Connection:";
    connection.Dump(indent_level + 1);
    ++i;
  }
}

// - Network Component

absl::StatusOr<PortId> NetworkComponent::CreatePort(PortDirection dir) {
  int64_t n_ports = ports_.size();

  XLS_ASSIGN_OR_RETURN(
      PortId next_id,
      PortId::ValidateAndReturnId(id_.GetNetworkId().id(),  // Network
                                  id_.id(),                 // Component
                                  n_ports));
  ports_.emplace_back(next_id, dir);
  return next_id;
}

Port& NetworkComponent::GetPort(PortId id) { return ports_.at(id.id()); }
const Port& NetworkComponent::GetPort(PortId id) const {
  return ports_.at(id.id());
}

std::vector<PortId> NetworkComponent::GetPortIds() const {
  std::vector<PortId> ret(ports_.size());
  for (int64_t i = 0; i < ports_.size(); ++i) {
    ret[i] = GetPortIdByIndex(i);
  }
  return ret;
}

std::vector<PortId> NetworkComponent::GetOutputPortIds() const {
  std::vector<PortId> ret;
  ret.reserve(ports_.size());

  for (int64_t i = 0; i < ports_.size(); ++i) {
    if (ports_[i].direction() == PortDirection::kOutput) {
      ret.emplace_back(GetPortIdByIndex(i));
    }
  }
  return ret;
}

std::vector<PortId> NetworkComponent::GetInputPortIds() const {
  std::vector<PortId> ret;
  ret.reserve(ports_.size());

  for (int64_t i = 0; i < ports_.size(); ++i) {
    if (ports_[i].direction() == PortDirection::kInput) {
      ret.emplace_back(GetPortIdByIndex(i));
    }
  }
  return ret;
}

void NetworkComponent::Dump(int indent_level) const {
  VLOG(1) << IndentLevel(indent_level)
          << absl::StreamFormat("NetworkComponent id %x kind %d",
                                id_.AsUInt64(), kind_);

  int64_t i = 0;
  for (auto port : ports_) {
    VLOG(1) << IndentLevel(indent_level) << "- #" << i << " Port:";
    port.Dump(indent_level + 1);
    ++i;
  }
}

// - Port
void Port::Dump(int indent_level) const {
  VLOG(1) << IndentLevel(indent_level)
          << absl::StreamFormat("Port id %x Dir %d Connection %x",
                                id_.AsUInt64(), dir_, connection_.AsUInt64());
}

// -  Connection

void Connection::DetachSink() {
  if (sink_.IsValid()) {
    mgr_->GetPort(sink_).SetDetached();
  }

  sink_ = PortId::kInvalid;
}

void Connection::DetachSrc() {
  if (src_.IsValid()) {
    mgr_->GetPort(src_).SetDetached();
  }

  src_ = PortId::kInvalid;
}

PortId Connection::Attach(PortId port) {
  if (port.IsValid()) {
    auto& p = mgr_->GetPort(port);

    if (p.direction() == PortDirection::kOutput) {  // Output --> From
      // Detach this connection from the existing src port (if any)
      DetachSrc();

      // Attach this connection to the new port
      src_ = port;
      p.SetAttached(id_);

      return port;
    }
    if (p.direction() == PortDirection::kInput) {  // To --> Input
      // Detach this connection from the existing to port (if any)
      DetachSink();

      // Attach this connection to the new port
      sink_ = port;
      p.SetAttached(id_);

      return port;
    }
  }

  return PortId::kInvalid;
}

void Connection::Dump(int indent_level) const {
  VLOG(1) << IndentLevel(indent_level)
          << absl::StreamFormat("Connection id %x src %x sink %x",
                                id_.AsUInt64(), src_.AsUInt64(),
                                sink_.AsUInt64());
}

}  // namespace noc
}  // namespace xls
