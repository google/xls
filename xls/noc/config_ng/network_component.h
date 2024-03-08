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

#ifndef XLS_NOC_CONFIG_NG_NETWORK_COMPONENT_H_
#define XLS_NOC_CONFIG_NG_NETWORK_COMPONENT_H_

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "xls/common/iterator_range.h"
#include "xls/ir/unwrapping_iterator.h"
#include "xls/noc/config_ng/network_component_port.h"

namespace xls::noc {

// Forward declaration: a network component has a reference to a network view.
class NetworkView;
class NetworkComponentVisitor;

// The network component contains ports and a reference to the view that it
// belongs to. The component owns the ports that it contains.
// TODO (vmirian) 02-05-21 enable a view for a component:
// - make view that the component belongs to the parent view (GetParentView)
// - enable the creation of a view from a component.
// - port mappings from component to the port of the component in the view.
//     -  when a port is added to the component, its mirror port is reproduced
//        in its view
class NetworkComponent {
 public:
  NetworkComponent(const NetworkComponent&) = delete;
  NetworkComponent& operator=(const NetworkComponent&) = delete;

  NetworkComponent(const NetworkComponent&&) = delete;
  NetworkComponent& operator=(const NetworkComponent&&) = delete;

  // Add a port to the component. Returns a reference to newly added port.
  // The port is owned by the component, the lifetime of the port is equivalent
  // to the lifetime of the component.
  NetworkComponentPort& AddPort(PortType port_type, PortDirection direction);

  // Returns an iterator range for the ports. The objects are guaranteed to be
  // non-null. Note that, when using the result of this function, if the
  // component is modified (e.g. a port is added), the returned result may
  // become invalid.
  xabsl::iterator_range<UnwrappingIterator<
      std::vector<std::unique_ptr<NetworkComponentPort>>::iterator>>
  ports();
  xabsl::iterator_range<UnwrappingIterator<
      std::vector<std::unique_ptr<NetworkComponentPort>>::const_iterator>>
  ports() const;

  // Visit the component.
  virtual absl::Status Visit(NetworkComponentVisitor& v) = 0;

  // Accepts a network component visitor to visit the component.
  absl::Status Accept(NetworkComponentVisitor& v);

  // Returns the number of ports.
  int64_t GetPortCount() const;

  // Sets the name of the component.
  void SetName(std::string name);

  // Gets the name of the component.
  const std::string& GetName() const;

  // Get network view of the component.
  NetworkView& GetNetworkView() const;

  virtual ~NetworkComponent() = default;

 protected:
  // network_view: the network view that the component belongs to. Does not take
  // ownership of the network view. The network view must refer to a valid
  // object that outlives this object.
  explicit NetworkComponent(NetworkView* network_view);

 private:
  std::vector<std::unique_ptr<NetworkComponentPort>> ports_;
  NetworkView& network_view_;
  std::string name_;
};

}  // namespace xls::noc

#endif  // XLS_NOC_CONFIG_NG_NETWORK_COMPONENT_H_
