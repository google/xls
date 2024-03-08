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

#ifndef XLS_NOC_CONFIG_NG_NETWORK_COMPONENT_UTILS_H_
#define XLS_NOC_CONFIG_NG_NETWORK_COMPONENT_UTILS_H_

#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"

namespace xls::noc {

class NetworkComponentPort;
class NetworkComponent;

// Returns OK if the component is valid. Otherwise, returns an error.
// A valid component contains valid ports and is the owner of its ports.
absl::Status ValidateNetworkComponent(const NetworkComponent& component);

// Returns a list of input ports of the component. The lifetime of the ports is
// dictated by the lifetime of the component.
std::vector<const NetworkComponentPort*> GetInputPortsFrom(
    const NetworkComponent& component);

// Returns a list of output ports of the component. The lifetime of the ports is
// dictated by the lifetime of the component.
std::vector<const NetworkComponentPort*> GetOutputPortsFrom(
    const NetworkComponent& component);

// Returns a list of data ports of the component. The lifetime of the ports is
// dictated by the lifetime of the component.
std::vector<const NetworkComponentPort*> GetDataPortsFrom(
    const NetworkComponent& component);

// Returns a list of control ports of the component. The lifetime of the ports
// is dictated by the lifetime of the component.
std::vector<const NetworkComponentPort*> GetControlPortsFrom(
    const NetworkComponent& component);

// Returns a list of components connected to the output ports. Preserves
// duplicates. The lifetime of the returned components is dictated by the
// lifetime of the view holding the component.
std::vector<const NetworkComponent*> GetComponentsConnectedToOutputPortsFrom(
    const NetworkComponent& component);

// Sets the name of the component to the name that it is associated with.
// Side-effect: replaces the existing name in the component. Returns OK if all
// components are valid (e.g. not nullptr). Otherwise, returns an error.
absl::Status AddNameToComponent(
    const absl::flat_hash_map<std::string, NetworkComponent*>& components);

}  // namespace xls::noc

#endif  // XLS_NOC_CONFIG_NG_NETWORK_COMPONENT_UTILS_H_
