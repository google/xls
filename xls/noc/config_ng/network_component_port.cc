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

#include "xls/noc/config_ng/network_component_port.h"

#include "absl/container/flat_hash_set.h"
#include "absl/log/die_if_null.h"

namespace xls::noc {

NetworkComponentPort::NetworkComponentPort(NetworkComponent* network_component,
                                           const PortType port_type,
                                           const PortDirection direction)
    : component_(*ABSL_DIE_IF_NULL(network_component)),
      type_(port_type),
      direction_(direction) {}

NetworkComponent& NetworkComponentPort::GetComponent() const {
  return component_;
}

PortDirection NetworkComponentPort::GetDirection() const { return direction_; }

bool NetworkComponentPort::IsInput() const {
  return GetDirection() == PortDirection::kInput;
}

bool NetworkComponentPort::IsOutput() const {
  return GetDirection() == PortDirection::kOutput;
}

PortType NetworkComponentPort::GetType() const { return type_; }

bool NetworkComponentPort::IsData() const {
  return GetType() == PortType::kData;
}

bool NetworkComponentPort::IsControl() const {
  return GetType() == PortType::kControl;
}

void NetworkComponentPort::AddConnection(NetworkConnection* connection) {
  connections_.insert(connection);
}

void NetworkComponentPort::RemoveConnection(NetworkConnection* connection) {
  connections_.erase(connection);
}

const absl::flat_hash_set<const NetworkConnection*>&
NetworkComponentPort::GetConnections() const {
  return connections_;
}

}  // namespace xls::noc
