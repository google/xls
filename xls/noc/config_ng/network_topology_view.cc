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

#include "xls/noc/config_ng/network_topology_view.h"

#include <cstdint>

#include "xls/common/status/status_macros.h"
#include "xls/noc/config_ng/network_view_utils.h"

namespace xls::noc {

SendPortTopologyComponent& NetworkTopologyView::AddSendPort() {
  return AddComponent<SendPortTopologyComponent>();
}

ReceivePortTopologyComponent& NetworkTopologyView::AddReceivePort() {
  return AddComponent<ReceivePortTopologyComponent>();
}

RouterTopologyComponent& NetworkTopologyView::AddRouter() {
  return AddComponent<RouterTopologyComponent>();
}

ChannelTopologyComponent& NetworkTopologyView::AddChannel() {
  return AddComponent<ChannelTopologyComponent>();
}

int64_t NetworkTopologyView::GetSendPortCount() const {
  return GetCount<SendPortTopologyComponent>();
}

int64_t NetworkTopologyView::GetReceivePortCount() const {
  return GetCount<ReceivePortTopologyComponent>();
}

int64_t NetworkTopologyView::GetRouterCount() const {
  return GetCount<RouterTopologyComponent>();
}

int64_t NetworkTopologyView::GetChannelCount() const {
  return GetCount<ChannelTopologyComponent>();
}

absl::StatusOr<ChannelTopologyComponent*>
NetworkTopologyView::ConnectThroughChannel(NetworkComponent& source,
                                           NetworkComponent& sink) {
  if (&source.GetNetworkView() != &sink.GetNetworkView()) {
    return absl::FailedPreconditionError(
        "source and sink components are from different views.");
  }
  if (&source.GetNetworkView() != this) {
    return absl::FailedPreconditionError(
        "source component is from a different view.");
  }
  ChannelTopologyComponent& channel = this->AddChannel();
  // connect source component to channel component
  this->AddConnection()
      .ConnectToSourcePort(
          &source.AddPort(PortType::kData, PortDirection::kOutput))
      .ConnectToSinkPort(
          &channel.AddPort(PortType::kData, PortDirection::kInput));
  // connect channel component to sink component
  this->AddConnection()
      .ConnectToSourcePort(
          &channel.AddPort(PortType::kData, PortDirection::kOutput))
      .ConnectToSinkPort(&sink.AddPort(PortType::kData, PortDirection::kInput));
  return &channel;
}

absl::StatusOr<RouterTopologyComponent*> NetworkTopologyView::AddRouter(
    int64_t send_port_count, int64_t recv_port_count) {
  RouterTopologyComponent& router = this->AddRouter();
  for (int64_t count = 0; count < send_port_count; count++) {
    XLS_RETURN_IF_ERROR(
        this->ConnectThroughChannel(this->AddSendPort(), router).status());
  }
  for (int64_t count = 0; count < recv_port_count; count++) {
    XLS_RETURN_IF_ERROR(
        this->ConnectThroughChannel(router, this->AddReceivePort()).status());
  }
  return &router;
}

absl::Status NetworkTopologyView::ConnectSendPortsToComponent(
    const int64_t send_port_count, NetworkComponent& component) {
  if (&component.GetNetworkView() != this) {
    return absl::FailedPreconditionError("component is from a different view.");
  }
  for (int64_t count = 0; count < send_port_count; count++) {
    XLS_RETURN_IF_ERROR(
        this->ConnectThroughChannel(this->AddSendPort(), component).status());
  }
  return absl::OkStatus();
}

absl::Status NetworkTopologyView::ConnectComponentToReceivePort(
    NetworkComponent& component, const int64_t recv_port_count) {
  if (&component.GetNetworkView() != this) {
    return absl::FailedPreconditionError("component is from a different view.");
  }
  for (int64_t count = 0; count < recv_port_count; count++) {
    XLS_RETURN_IF_ERROR(
        this->ConnectThroughChannel(component, this->AddReceivePort())
            .status());
  }
  return absl::OkStatus();
}

}  // namespace xls::noc
