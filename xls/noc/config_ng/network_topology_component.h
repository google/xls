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

#ifndef XLS_NOC_CONFIG_NG_NETWORK_TOPOLOGY_COMPONENT_H_
#define XLS_NOC_CONFIG_NG_NETWORK_TOPOLOGY_COMPONENT_H_

#include "absl/status/status.h"
#include "xls/noc/config_ng/coordinate.h"
#include "xls/noc/config_ng/network_component.h"

namespace xls::noc {

// A channel used in the network topology view.
class ChannelTopologyComponent final : public NetworkComponent {
 public:
  // See xls::noc::NetworkComponent::NetworkComponent.
  explicit ChannelTopologyComponent(NetworkView* view);

  // See xls::noc::NetworkComponent::Visit.
  absl::Status Visit(NetworkComponentVisitor& v) override;

  ~ChannelTopologyComponent() override;
};

// A receive port used in the network topology view.
class ReceivePortTopologyComponent final : public NetworkComponent {
 public:
  // See xls::noc::NetworkComponent::NetworkComponent.
  explicit ReceivePortTopologyComponent(NetworkView* view);

  // See xls::noc::NetworkComponent::Visit.
  absl::Status Visit(NetworkComponentVisitor& v) override;

  ~ReceivePortTopologyComponent() override;
};

// A router used in the network topology view.
class RouterTopologyComponent final : public NetworkComponent {
 public:
  // See xls::noc::NetworkComponent::NetworkComponent.
  explicit RouterTopologyComponent(NetworkView* view);

  const Coordinate& GetCoordinate() const;

  RouterTopologyComponent& SetCoordinate(const Coordinate& coordinate);

  // See xls::noc::NetworkComponent::Visit.
  absl::Status Visit(NetworkComponentVisitor& v) override;

  ~RouterTopologyComponent() override;

 private:
  Coordinate coordinate_;
};

// A send port used in the network topology view.
class SendPortTopologyComponent final : public NetworkComponent {
 public:
  // See xls::noc::NetworkComponent::NetworkComponent.
  explicit SendPortTopologyComponent(NetworkView* view);

  // See xls::noc::NetworkComponent::Visit.
  absl::Status Visit(NetworkComponentVisitor& v) override;

  ~SendPortTopologyComponent() override;
};

}  // namespace xls::noc

#endif  // XLS_NOC_CONFIG_NG_NETWORK_TOPOLOGY_COMPONENT_H_
