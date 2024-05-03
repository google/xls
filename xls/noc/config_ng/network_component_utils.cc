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

#include "xls/noc/config_ng/network_component_utils.h"

#include <string>
#include <string_view>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/noc/config_ng/network_component.h"
#include "xls/noc/config_ng/network_component_port_utils.h"
#include "xls/noc/config_ng/network_connection.h"

namespace xls::noc {

absl::Status ValidateNetworkComponent(const NetworkComponent& component) {
  for (NetworkComponentPort* port : component.ports()) {
    XLS_RETURN_IF_ERROR(ValidateNetworkComponentPort(*port));
    if (&port->GetComponent() != &component) {
      return absl::FailedPreconditionError(
          "Component contains port that it does not own.");
    }
  }
  return absl::Status();
}

std::vector<const NetworkComponentPort*> GetInputPortsFrom(
    const NetworkComponent& component) {
  std::vector<const NetworkComponentPort*> input_ports;
  for (const NetworkComponentPort* port : component.ports()) {
    if (port->IsInput()) {
      input_ports.push_back(port);
    }
  }
  return input_ports;
}

std::vector<const NetworkComponentPort*> GetOutputPortsFrom(
    const NetworkComponent& component) {
  std::vector<const NetworkComponentPort*> output_ports;
  for (const NetworkComponentPort* port : component.ports()) {
    if (port->IsOutput()) {
      output_ports.push_back(port);
    }
  }
  return output_ports;
}

std::vector<const NetworkComponentPort*> GetDataPortsFrom(
    const NetworkComponent& component) {
  std::vector<const NetworkComponentPort*> data_ports;
  for (const NetworkComponentPort* port : component.ports()) {
    if (port->IsData()) {
      data_ports.push_back(port);
    }
  }
  return data_ports;
}

std::vector<const NetworkComponentPort*> GetControlPortsFrom(
    const NetworkComponent& component) {
  std::vector<const NetworkComponentPort*> control_ports;
  for (const NetworkComponentPort* port : component.ports()) {
    if (port->IsControl()) {
      control_ports.push_back(port);
    }
  }
  return control_ports;
}

std::vector<const NetworkComponent*> GetComponentsConnectedToOutputPortsFrom(
    const NetworkComponent& component) {
  std::vector<const NetworkComponent*> components;
  for (const NetworkComponentPort* port : GetOutputPortsFrom(component)) {
    std::vector<const NetworkComponent*> connected_components =
        GetComponentsConnectedTo(*port);
    components.insert(components.end(), connected_components.begin(),
                      connected_components.end());
  }
  return components;
}

absl::Status AddNameToComponent(
    const absl::flat_hash_map<std::string, NetworkComponent*>& components) {
  // Set names for routers
  for (auto& [name, component] : components) {
    XLS_RET_CHECK(component != nullptr)
        << std::string_view("Component is not valid.");
    component->SetName(name);
  }
  return absl::Status();
}

}  // namespace xls::noc
