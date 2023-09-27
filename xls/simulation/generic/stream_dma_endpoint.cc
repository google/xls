// Copyright 2023 The XLS Authors
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

#include "xls/simulation/generic/stream_dma_endpoint.h"

#include "absl/strings/str_format.h"
#include "xls/common/logging/logging.h"
#include "xls/common/status/status_macros.h"

namespace xls::simulation::generic {

absl::Status StreamDmaEndpoint::Write(Payload payload) {
  XLS_CHECK(!IsReadStream());
  if (payload.data.empty()) {
    // Empty writes are ignored
    return absl::OkStatus();
  }
  if (payload.data.size() != transfer_size_) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Invalid transfer size: %u bytes, must be %u",
                        payload.data.size(), transfer_size_));
  }
  for (uint64_t i = 0; i < transfer_size_; i++) {
    XLS_RETURN_IF_ERROR(stream_->SetPayloadData8(i, payload.data[i]));
  }
  XLS_RETURN_IF_ERROR(stream_->Transfer());
  return absl::OkStatus();
}

absl::StatusOr<StreamDmaEndpoint::Payload> StreamDmaEndpoint::Read() {
  XLS_CHECK(IsReadStream());
  XLS_RETURN_IF_ERROR(stream_->Transfer());
  Payload payload{};
  payload.data.resize(transfer_size_);
  for (uint64_t i = 0; i < transfer_size_; i++) {
    XLS_ASSIGN_OR_RETURN(payload.data[i], stream_->GetPayloadData8(i));
  }
  return payload;
}

}  // namespace xls::simulation::generic
