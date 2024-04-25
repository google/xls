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

#include "xls/noc/config_ng/network_connection_utils.h"

#include "absl/status/status.h"
#include "xls/noc/config_ng/network_component_port.h"
#include "xls/noc/config_ng/network_connection.h"

namespace xls::noc {

absl::Status ValidateNetworkConnection(const NetworkConnection& connection) {
  const NetworkComponentPort* source_port = connection.GetSourcePort();
  const NetworkComponentPort* sink_port = connection.GetSinkPort();
  if (source_port == nullptr) {
    return absl::FailedPreconditionError("Source port not set.");
  }
  if (sink_port == nullptr) {
    return absl::FailedPreconditionError("Sink port not set.");
  }
  if (!source_port->IsOutput()) {
    return absl::FailedPreconditionError(
        "Source port does not have its direction set to output.");
  }
  if (!sink_port->IsInput()) {
    return absl::FailedPreconditionError(
        "Sink port does not have its direction set to input.");
  }
  if (source_port->IsData() != sink_port->IsData() ||
      sink_port->IsControl() != sink_port->IsControl()) {
    return absl::FailedPreconditionError(
        "Source port and sink port types do not match.");
  }
  return absl::Status();
}

}  // namespace xls::noc
