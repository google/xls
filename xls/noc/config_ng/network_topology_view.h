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

#ifndef XLS_NOC_CONFIG_NG_NETWORK_TOPOLOGY_VIEW_H_
#define XLS_NOC_CONFIG_NG_NETWORK_TOPOLOGY_VIEW_H_

#include <cstdint>
#include <vector>

#include "absl/status/statusor.h"
#include "xls/noc/config_ng/network_topology_component.h"
#include "xls/noc/config_ng/network_view.h"

namespace xls::noc {

// A network view representing the topology. The class provides helper functions
// to create topology components and connect them to create a topology view of
// the network.
class NetworkTopologyView final : public NetworkView {
 public:
  NetworkTopologyView() = default;

  // See xls::noc::NetworkView::AddComponent.
  SendPortTopologyComponent& AddSendPort();

  // See xls::noc::NetworkView::AddComponent.
  ReceivePortTopologyComponent& AddReceivePort();

  // See xls::noc::NetworkView::AddComponent.
  RouterTopologyComponent& AddRouter();

  // See xls::noc::NetworkView::AddComponent.
  ChannelTopologyComponent& AddChannel();

  int64_t GetSendPortCount() const;

  int64_t GetReceivePortCount() const;

  int64_t GetRouterCount() const;

  int64_t GetChannelCount() const;

  // Connects the source network component to the sink network component using a
  // channel. The channel is a new instance. The source component and sink
  // component must be from the same view instance.
  // Connection: source component -> channel (created) -> sink component
  //
  // Returns the newly created channel on success. Otherwise, returns an
  // absl::FailedPreconditionError.
  absl::StatusOr<ChannelTopologyComponent*> ConnectThroughChannel(
      NetworkComponent& source, NetworkComponent& sink);

  // TODO(vmirian) 02-05-2021 return send ports, receive ports and channels
  // Adds a router to the view and connects the router to send ports
  // and receive ports through channels.
  // send_port_count: the number of send ports to connect to the router.
  // recv_port_count: the number of receive ports to connect to the router.
  //
  // Example:
  //     AddRouter(2, 1);
  //     yields the following component and connections:
  //     send port [A] -> channel [0] -> router [R]
  //     send port [B] -> channel [1] -> router [R]
  //     router [R] -> channel [2] -> receive port [A]
  //     send port [A], send port [B], channel [0], channel [1], channel [2],
  //     router [R] and receive port [A] are new components added to the view.
  //
  // Returns the new router on success. Otherwise, returns an
  // absl::FailedPreconditionError. Implementation details: calls
  // ConnectThroughChannel.
  absl::StatusOr<RouterTopologyComponent*> AddRouter(int64_t send_port_count,
                                                     int64_t recv_port_count);

  // TODO(vmirian) 02-05-2021 return send ports and channels
  // Connects send ports to the component through channels.
  // send_port_count: the number of send ports to connect to the component.
  // component: a component instance from the view
  //
  // Example:
  //     ConnectSendPortsToComponent(2, component);
  //     yields the following component and connections:
  //     send port [A] -> channel [0] -> component [C]
  //     send port [B] -> channel [1] -> component [C]
  //     send port [A], send port [B], channel [0], channel [1] are new
  //     components added to the view.
  //
  //  Returns absl::OkStatus on success. Otherwise, returns an
  //  absl::FailedPreconditionError. Implementation details: calls
  //  ConnectThroughChannel.
  absl::Status ConnectSendPortsToComponent(int64_t send_port_count,
                                           NetworkComponent& component);

  // TODO(vmirian) 02-05-2021 return receive ports and channels
  // Connects the component to receive ports through channels.
  // component: a component instance from the view
  // recv_port_count: the number of receive ports to connect to the component.
  //
  // Example:
  //     ConnectComponentToReceivePort(component, 1);
  //     yields the following component and connections:
  //     component [C] -> channel [0] -> receive port [A]
  //     receive port [A], channel [0], are new components added to the view.
  //
  //  Returns absl::OkStatus on success. Otherwise, returns an
  //  absl::FailedPreconditionError. Implementation details: calls
  //  ConnectThroughChannel.
  absl::Status ConnectComponentToReceivePort(NetworkComponent& component,
                                             int64_t recv_port_count);

  ~NetworkTopologyView() override = default;
};

}  // namespace xls::noc

#endif  // XLS_NOC_CONFIG_NG_NETWORK_TOPOLOGY_VIEW_H_
