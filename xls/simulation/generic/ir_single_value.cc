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

#include "xls/simulation/generic/ir_single_value.h"

#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/value.h"
#include "xls/ir/value_helpers.h"
#include "xls/simulation/generic/ir_value_access_methods.h"

namespace xls::simulation::generic {

/* static */ absl::StatusOr<IRSingleValue> IRSingleValue::MakeIRSingleValue(
    xls::ChannelQueue* queue) {
  if (queue->channel()->kind() != xls::ChannelKind::kSingleValue)
    return absl::InvalidArgumentError(
        "MakeIRSingleValue expects queue with SingleValueChannel channel.");
  return IRSingleValue(queue);
}

IRSingleValue::IRSingleValue(xls::ChannelQueue* queue)
    : ir_single_value_queue_(queue) {}

absl::StatusOr<uint64_t> IRSingleValue::InternalRead(uint64_t offset,
                                                     int count) const {
  Value value_ = ZeroOfType(this->ir_single_value_queue_->channel()->type());
  std::optional<Value> read_value = this->ir_single_value_queue_->Read();
  if (read_value.has_value()) value_ = read_value.value();
  std::vector<Value> input;
  input.push_back(value_);
  XLS_ASSIGN_OR_RETURN(
      uint64_t ans, ValueArrayReadUInt64(
                        input, this->ir_single_value_queue_->channel()->name(),
                        offset, count));
  return ans;
}

absl::StatusOr<uint8_t> IRSingleValue::GetPayloadData8(uint64_t offset) const {
  return InternalRead(offset, 1);
}

absl::StatusOr<uint16_t> IRSingleValue::GetPayloadData16(
    uint64_t offset) const {
  return InternalRead(offset, 2);
}

absl::StatusOr<uint32_t> IRSingleValue::GetPayloadData32(
    uint64_t offset) const {
  return InternalRead(offset, 4);
}

absl::StatusOr<uint64_t> IRSingleValue::GetPayloadData64(
    uint64_t offset) const {
  return InternalRead(offset, 8);
}

absl::Status IRSingleValue::InternalWrite(uint64_t offset, uint64_t data,
                                          int count) {
  Value value_ = ZeroOfType(this->ir_single_value_queue_->channel()->type());
  std::optional<Value> read_value = this->ir_single_value_queue_->Read();
  if (read_value.has_value()) value_ = read_value.value();
  std::vector<Value> input;
  input.push_back(value_);
  XLS_ASSIGN_OR_RETURN(
      std::vector<Value> output_,
      ValueArrayWriteUInt64(input,
                            this->ir_single_value_queue_->channel()->name(),
                            offset, count, data));
  XLS_CHECK_OK(this->ir_single_value_queue_->Write(output_[0]));
  return absl::OkStatus();
}

absl::Status IRSingleValue::SetPayloadData8(uint64_t offset, uint8_t data) {
  return InternalWrite(offset, data, 1);
}

absl::Status IRSingleValue::SetPayloadData16(uint64_t offset, uint16_t data) {
  return InternalWrite(offset, data, 2);
}

absl::Status IRSingleValue::SetPayloadData32(uint64_t offset, uint32_t data) {
  return InternalWrite(offset, data, 4);
}

absl::Status IRSingleValue::SetPayloadData64(uint64_t offset, uint64_t data) {
  return InternalWrite(offset, data, 8);
}

uint64_t IRSingleValue::GetChannelWidth() const {
  return this->ir_single_value_queue_->channel()->type()->GetFlatBitCount();
}

}  // namespace xls::simulation::generic
