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

#include "xls/noc/config_ng/network_topology_component.h"

#include "absl/status/status.h"
#include "xls/noc/config_ng/network_component_visitor_abstract.h"

namespace xls::noc {

ChannelTopologyComponent::ChannelTopologyComponent(NetworkView* view)
    : NetworkComponent(view) {}

absl::Status Visit(NetworkComponentVisitor& v);

absl::Status ChannelTopologyComponent::Visit(NetworkComponentVisitor& v) {
  return v.Handle(*this);
}

ChannelTopologyComponent::~ChannelTopologyComponent() = default;

ReceivePortTopologyComponent::ReceivePortTopologyComponent(NetworkView* view)
    : NetworkComponent(view) {}

absl::Status ReceivePortTopologyComponent::Visit(NetworkComponentVisitor& v) {
  return v.Handle(*this);
}

ReceivePortTopologyComponent::~ReceivePortTopologyComponent() = default;

RouterTopologyComponent::RouterTopologyComponent(NetworkView* view)
    : NetworkComponent(view), coordinate_({}) {}

const Coordinate& RouterTopologyComponent::GetCoordinate() const {
  return coordinate_;
}

RouterTopologyComponent& RouterTopologyComponent::SetCoordinate(
    const Coordinate& coordinate) {
  coordinate_ = coordinate;
  return *this;
}

absl::Status RouterTopologyComponent::Visit(NetworkComponentVisitor& v) {
  return v.Handle(*this);
}

RouterTopologyComponent::~RouterTopologyComponent() = default;

SendPortTopologyComponent::SendPortTopologyComponent(NetworkView* view)
    : NetworkComponent(view) {}

absl::Status SendPortTopologyComponent::Visit(NetworkComponentVisitor& v) {
  return v.Handle(*this);
}

SendPortTopologyComponent::~SendPortTopologyComponent() = default;

}  // namespace xls::noc
