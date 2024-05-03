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

#include "xls/noc/config_ng/network_component_port_utils.h"

#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/types/span.h"
#include "xls/common/status/ret_check.h"
#include "xls/noc/config_ng/network_component_port.h"
#include "xls/noc/config_ng/network_connection.h"

namespace xls::noc {

std::vector<const NetworkComponentPort*> GetInputPortsFrom(
    absl::Span<NetworkComponentPort* const> ports) {
  std::vector<const NetworkComponentPort*> input_ports;
  for (const NetworkComponentPort* port : ports) {
    if (port->IsInput()) {
      input_ports.push_back(port);
    }
  }
  return input_ports;
}

std::vector<const NetworkComponentPort*> GetOutputPortsFrom(
    absl::Span<NetworkComponentPort* const> ports) {
  std::vector<const NetworkComponentPort*> output_ports;
  for (const NetworkComponentPort* port : ports) {
    if (port->IsOutput()) {
      output_ports.emplace_back(port);
    }
  }
  return output_ports;
}

std::vector<const NetworkComponentPort*> GetDataPortsFrom(
    absl::Span<NetworkComponentPort* const> ports) {
  std::vector<const NetworkComponentPort*> data_ports;
  for (const NetworkComponentPort* port : ports) {
    if (port->IsData()) {
      data_ports.emplace_back(port);
    }
  }
  return data_ports;
}

std::vector<const NetworkComponentPort*> GetControlPortsFrom(
    absl::Span<NetworkComponentPort* const> ports) {
  std::vector<const NetworkComponentPort*> control_ports;
  for (const NetworkComponentPort* const port : ports) {
    if (port->IsControl()) {
      control_ports.emplace_back(port);
    }
  }
  return control_ports;
}

absl::Status ValidateNetworkComponentPort(const NetworkComponentPort& port) {
  absl::flat_hash_set<const NetworkConnection*> connections;
  for (const NetworkConnection* connection : port.GetConnections()) {
    XLS_RET_CHECK(connection != nullptr)
        << "The port must have a valid connection.";
    XLS_RET_CHECK(connections.insert(connection).second)
        << "The port's connections must be unique.";
  }
  return absl::Status();
}

std::vector<const NetworkComponent*> GetComponentsConnectedTo(
    const NetworkComponentPort& port) {
  std::vector<const NetworkComponent*> components;
  if (port.IsOutput()) {
    for (const NetworkConnection* connection : port.GetConnections()) {
      CHECK(connection != nullptr) << "The port must have a valid connection.";
      if (connection->GetSinkPort() != nullptr) {
        components.push_back(&connection->GetSinkPort()->GetComponent());
      }
    }
  }
  if (port.IsInput()) {
    for (const NetworkConnection* connection : port.GetConnections()) {
      CHECK(connection != nullptr) << "The port must have a valid connection.";
      if (connection->GetSinkPort() != nullptr) {
        components.push_back(&connection->GetSourcePort()->GetComponent());
      }
    }
  }
  return components;
}

}  // namespace xls::noc
