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

#include "xls/simulation/generic/axi_stream_like_dma_endpoint.h"

#include <memory>
#include <utility>
#include <vector>

#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "xls/common/logging/logging.h"
#include "xls/common/status/status_macros.h"
#include "xls/simulation/generic/idmaendpoint.h"

namespace xls::simulation::generic {

AxiStreamLikeDmaEndpoint::AxiStreamLikeDmaEndpoint(
    std::unique_ptr<IAxiStreamLike> stream)
    : stream_(std::move(stream)) {
  num_symbols_ = stream_->GetNumSymbols();
  symbol_size_ = stream_->GetSymbolSize();
  XLS_CHECK(symbol_size_ > 0);
  XLS_CHECK(num_symbols_ > 0);
}

absl::Status AxiStreamLikeDmaEndpoint::Write(Payload payload) {
  XLS_CHECK(!IsReadStream());
  auto num_symbols_to_write = payload.data.size() / symbol_size_;
  if (payload.data.size() % symbol_size_) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Invalid transfer size: %u bytes, must be multiple of %u",
        payload.data.size(), symbol_size_));
  }
  if (num_symbols_to_write > num_symbols_) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Transfer too big: %u bytes, max is %u",
                        payload.data.size(), symbol_size_ * num_symbols_));
  }
  // Set TLAST
  stream_->SetLast(payload.last);
  // Set TKEEP
  std::vector<bool> tkeep(num_symbols_, false);
  for (uint64_t i = 0; i < num_symbols_to_write; i++) {
    tkeep[i] = true;
  }
  stream_->SetDataValid(std::move(tkeep));
  // Set data
  for (uint64_t i = 0; i < (num_symbols_ * symbol_size_); i++) {
    if (i < payload.data.size()) {
      XLS_RETURN_IF_ERROR(stream_->SetPayloadData8(i, payload.data[i]));
    } else {
      XLS_RETURN_IF_ERROR(stream_->SetPayloadData8(i, 0));
    }
  }
  XLS_RETURN_IF_ERROR(stream_->Transfer());
  return absl::OkStatus();
}

absl::StatusOr<AxiStreamLikeDmaEndpoint::Payload>
AxiStreamLikeDmaEndpoint::Read() {
  XLS_CHECK(IsReadStream());
  XLS_RETURN_IF_ERROR(stream_->Transfer());
  Payload payload{};
  // Get TLAST
  payload.last = stream_->GetLast();
  // Get TKEEP
  auto tkeep = stream_->GetDataValid();
  XLS_CHECK(tkeep.size() == num_symbols_);
  // Find the last valid byte
  int64_t last_valid = -1;
  bool continuous = true;
  for (uint64_t i = 0; i < num_symbols_; i++) {
    if (tkeep[i]) {
      // Detect holes in TKEEP
      if (static_cast<uint64_t>(last_valid + 1) != i) {
        continuous = false;
      }
      last_valid = i;
    }
  }
  uint64_t valid_symbols = last_valid + 1;
  if (!continuous) {
    XLS_LOG(WARNING) << absl::StreamFormat(
        "TKEEP returned by the stream is discontinuous: [%s]. Holes are "
        "ignored, assuming %u valid symbols",
        absl::StrJoin(tkeep, ", "), valid_symbols);
  }
  uint64_t xfer_nbytes = valid_symbols * symbol_size_;
  payload.data.resize(xfer_nbytes);
  for (uint64_t i = 0; i < xfer_nbytes; i++) {
    XLS_ASSIGN_OR_RETURN(payload.data[i], stream_->GetPayloadData8(i));
  }
  return payload;
}

}  // namespace xls::simulation::generic
