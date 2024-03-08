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

#ifndef XLS_NOC_CONFIG_NG_NETWORK_COMPONENT_PORT_UTILS_H_
#define XLS_NOC_CONFIG_NG_NETWORK_COMPONENT_PORT_UTILS_H_

#include <vector>

#include "absl/status/status.h"
#include "absl/types/span.h"

namespace xls::noc {

class NetworkComponentPort;
class NetworkComponent;

// Filters the input ports from the list provided. Returns the filtered list.
// The lifetime of a pointer in the filtered list is the same as the lifetime of
// its corresponding pointer in the list provided.
std::vector<const NetworkComponentPort*> GetInputPortsFrom(
    absl::Span<NetworkComponentPort* const> ports);

// Filters the output ports from the list provided. Returns the filtered list.
// The lifetime of a pointer in the filtered list is the same as the lifetime of
// its corresponding pointer in the list provided.
std::vector<const NetworkComponentPort*> GetOutputPortsFrom(
    absl::Span<NetworkComponentPort* const> ports);

// Filters the data ports from the list provided. Returns the filtered list. The
// lifetime of a pointer in the filtered list is the same as the lifetime of its
// corresponding pointer in the list provided.
std::vector<const NetworkComponentPort*> GetDataPortsFrom(
    absl::Span<NetworkComponentPort* const> ports);

// Filters the control ports from the list provided. Returns the filtered list.
// The lifetime of a pointer in the filtered list is the same as the lifetime of
// its corresponding pointer in the list provided.
std::vector<const NetworkComponentPort*> GetControlPortsFrom(
    absl::Span<NetworkComponentPort* const> ports);

// Returns OK if the port is valid. Otherwise, returns an error.
// A valid port is connected to a valid connection and all its connections are
// unique.
absl::Status ValidateNetworkComponentPort(const NetworkComponentPort& port);

// Returns a list of components connected to the port. Preserves duplicates. The
// lifetime of the returned components is dictated by the lifetime of the view
// holding the component.
std::vector<const NetworkComponent*> GetComponentsConnectedTo(
    const NetworkComponentPort& port);

}  // namespace xls::noc

#endif  // XLS_NOC_CONFIG_NG_NETWORK_COMPONENT_PORT_UTILS_H_
