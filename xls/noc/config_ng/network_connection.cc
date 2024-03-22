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

#include "xls/noc/config_ng/network_connection.h"

#include "absl/log/die_if_null.h"
#include "xls/noc/config_ng/network_component_port.h"

namespace xls::noc {

NetworkConnection::NetworkConnection(NetworkView* network_view)
    : source_port_(nullptr),
      sink_port_(nullptr),
      network_view_(*ABSL_DIE_IF_NULL(network_view)) {}

NetworkConnection& NetworkConnection::ConnectToSourcePort(
    NetworkComponentPort* source_port) {
  if (source_port_ != nullptr) {
    source_port_->RemoveConnection(this);
  }
  source_port_ = source_port;
  source_port_->AddConnection(this);
  return *this;
}

NetworkComponentPort* NetworkConnection::GetSourcePort() const {
  return source_port_;
}

NetworkConnection& NetworkConnection::ConnectToSinkPort(
    NetworkComponentPort* sink_port) {
  if (sink_port_ != nullptr) {
    sink_port_->RemoveConnection(this);
  }
  sink_port_ = sink_port;
  sink_port_->AddConnection(this);
  return *this;
}

NetworkComponentPort* NetworkConnection::GetSinkPort() const {
  return sink_port_;
}

NetworkView& NetworkConnection::GetNetworkView() const { return network_view_; }

}  // namespace xls::noc
