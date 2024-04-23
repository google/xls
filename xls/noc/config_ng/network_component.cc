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

#include "xls/noc/config_ng/network_component.h"

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/die_if_null.h"

namespace xls::noc {

int64_t NetworkComponent::GetPortCount() const { return ports_.size(); }

void NetworkComponent::SetName(std::string name) { name_ = std::move(name); }

const std::string& NetworkComponent::GetName() const { return name_; }

NetworkView& NetworkComponent::GetNetworkView() const { return network_view_; }

NetworkComponentPort& NetworkComponent::AddPort(const PortType port_type,
                                                const PortDirection direction) {
  // Using `new` to access a non-public constructor.
  ports_.push_back(
      absl::WrapUnique(new NetworkComponentPort(this, port_type, direction)));
  return (*ports_.back());
}

xabsl::iterator_range<UnwrappingIterator<
    std::vector<std::unique_ptr<NetworkComponentPort>>::iterator>>
NetworkComponent::ports() {
  return xabsl::make_range(MakeUnwrappingIterator(ports_.begin()),
                           MakeUnwrappingIterator(ports_.end()));
}

xabsl::iterator_range<UnwrappingIterator<
    std::vector<std::unique_ptr<NetworkComponentPort>>::const_iterator>>
NetworkComponent::ports() const {
  return xabsl::make_range(MakeUnwrappingIterator(ports_.begin()),
                           MakeUnwrappingIterator(ports_.end()));
}

NetworkComponent::NetworkComponent(NetworkView* network_view)
    : network_view_(*ABSL_DIE_IF_NULL(network_view)) {}

absl::Status NetworkComponent::Accept(NetworkComponentVisitor& v) {
  return Visit(v);
}

}  // namespace xls::noc
