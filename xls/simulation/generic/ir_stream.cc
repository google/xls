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

#include "xls/simulation/generic/ir_stream.h"

#include <optional>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/value.h"
#include "xls/ir/value_helpers.h"
#include "xls/simulation/generic/ir_value_access_methods.h"

namespace xls::simulation::generic {

/* static */ absl::StatusOr<IRStream> IRStream::MakeIRStream(
    xls::ChannelQueue* queue) {
  if (queue->channel()->kind() != xls::ChannelKind::kStreaming)
    return absl::InvalidArgumentError(
        "MakeIRStream expects queue with Streaming channel.");
  if (queue->channel()->CanSend() && queue->channel()->CanReceive())
    return absl::InvalidArgumentError(
        "MakeIRStream expects unidirectional Streaming channel.");
  return IRStream(queue);
}

IRStream::IRStream(xls::ChannelQueue* queue)
    : read_channel_(queue->channel()->CanSend()),
      holding_reg_(ZeroOfType(queue->channel()->type())),
      ir_stream_queue_(queue) {}

absl::StatusOr<uint64_t> IRStream::InternalRead(uint64_t offset,
                                                int count) const {
  std::vector<Value> input;
  input.push_back(this->holding_reg_);
  XLS_ASSIGN_OR_RETURN(
      uint64_t ans,
      ValueArrayReadUInt64(input, this->ir_stream_queue_->channel()->name(),
                           offset, count));
  return ans;
}

absl::StatusOr<uint8_t> IRStream::GetPayloadData8(uint64_t offset) const {
  return InternalRead(offset, 1);
}

absl::StatusOr<uint16_t> IRStream::GetPayloadData16(uint64_t offset) const {
  return InternalRead(offset, 2);
}

absl::StatusOr<uint32_t> IRStream::GetPayloadData32(uint64_t offset) const {
  return InternalRead(offset, 4);
}

absl::StatusOr<uint64_t> IRStream::GetPayloadData64(uint64_t offset) const {
  return InternalRead(offset, 8);
}

absl::Status IRStream::InternalWrite(uint64_t offset, uint64_t data,
                                     int count) {
  std::vector<Value> input;
  input.push_back(this->holding_reg_);
  XLS_ASSIGN_OR_RETURN(
      auto holding_reg_,
      ValueArrayWriteUInt64(input, this->ir_stream_queue_->channel()->name(),
                            offset, count, data));

  // IRStream holding register consists of a single xls::Value
  // Output vector from ValueArrayWriteUInt64 only has single element
  // at index 0
  this->holding_reg_ = holding_reg_[0];
  return absl::OkStatus();
}

absl::Status IRStream::SetPayloadData8(uint64_t offset, uint8_t data) {
  return InternalWrite(offset, data, 1);
}

absl::Status IRStream::SetPayloadData16(uint64_t offset, uint16_t data) {
  return InternalWrite(offset, data, 2);
}

absl::Status IRStream::SetPayloadData32(uint64_t offset, uint32_t data) {
  return InternalWrite(offset, data, 4);
}

absl::Status IRStream::SetPayloadData64(uint64_t offset, uint64_t data) {
  return InternalWrite(offset, data, 8);
}

uint64_t IRStream::GetChannelWidth() const {
  return this->holding_reg_.GetFlatBitCount();
}

bool IRStream::IsReadStream() const { return this->read_channel_; }

bool IRStream::IsReady() const {
  if (this->read_channel_) {
    // Read FIFO
    return !this->ir_stream_queue_->IsEmpty();
  }
  // Write FIFO
  return this->ir_stream_queue_->GetSize() < this->kWriteFifoMaxDepth;
}

absl::Status IRStream::Transfer() {
  if (this->read_channel_) {
    std::optional<Value> read_value = this->ir_stream_queue_->Read();
    if (!read_value.has_value())
      return absl::InternalError(
          "Streaming queue: " + this->ir_stream_queue_->channel()->name() +
          " was empty during read");
    this->holding_reg_ = read_value.value();
    return absl::OkStatus();
  }

  if (this->ir_stream_queue_->GetSize() >= this->kWriteFifoMaxDepth)
    return absl::InternalError(
        "Streaming queue: " + this->ir_stream_queue_->channel()->name() +
        " overfill");

  XLS_CHECK_OK(this->ir_stream_queue_->Write(this->holding_reg_));
  this->holding_reg_ = ZeroOfType(this->ir_stream_queue_->channel()->type());
  return absl::OkStatus();
}

}  // namespace xls::simulation::generic
