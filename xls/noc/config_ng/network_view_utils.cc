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

#include "xls/noc/config_ng/network_view_utils.h"

#include "absl/status/status.h"
#include "xls/common/status/status_macros.h"
#include "xls/noc/config_ng/network_component_utils.h"
#include "xls/noc/config_ng/network_connection_utils.h"
#include "xls/noc/config_ng/network_view.h"

namespace xls::noc {

absl::Status ValidateNetworkView(const NetworkView& view) {
  for (const NetworkComponent* component : view.components()) {
    XLS_RETURN_IF_ERROR(ValidateNetworkComponent(*component));
    if (&component->GetNetworkView() != &view) {
      return absl::FailedPreconditionError(
          "Network view contains component that it does not own.");
    }
  }
  for (const NetworkConnection* connection : view.connections()) {
    XLS_RETURN_IF_ERROR(ValidateNetworkConnection(*connection));
    if (&connection->GetNetworkView() != &view) {
      return absl::FailedPreconditionError(
          "Network view contains connection that it does not own.");
    }
  }
  return absl::Status();
}

}  // namespace xls::noc
