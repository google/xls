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

#ifndef XLS_NOC_CONFIG_NG_NETWORK_COMPONENT_VISITOR_ABSTRACT_H_
#define XLS_NOC_CONFIG_NG_NETWORK_COMPONENT_VISITOR_ABSTRACT_H_

#include "absl/status/status.h"

namespace xls::noc {

// Forward declaration to avoid circular dependency.
class SendPortTopologyComponent;
class ReceivePortTopologyComponent;
class RouterTopologyComponent;
class ChannelTopologyComponent;

// A network component visitor is an abstract class. It contains the handlers
// (callback functions) for the visitor pattern on a network component. See
// xls::noc::NetworkComponent::Accept.
class NetworkComponentVisitor {
 public:
  virtual absl::Status Handle(SendPortTopologyComponent& component) = 0;

  virtual absl::Status Handle(ReceivePortTopologyComponent& component) = 0;

  virtual absl::Status Handle(RouterTopologyComponent& component) = 0;

  virtual absl::Status Handle(ChannelTopologyComponent& component) = 0;

  virtual ~NetworkComponentVisitor() = default;
};

}  // namespace xls::noc

#endif  // XLS_NOC_CONFIG_NG_NETWORK_COMPONENT_VISITOR_ABSTRACT_H_
