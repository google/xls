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

#include "xls/simulation/generic/ir_axistreamlike.h"

#include <algorithm>
#include <vector>

#include "absl/strings/str_format.h"
#include "xls/ir/value.h"
#include "xls/ir/value_helpers.h"
#include "xls/simulation/generic/ir_value_access_methods.h"

namespace xls::simulation::generic {

/* static */ absl::StatusOr<IrAxiStreamLike> IrAxiStreamLike::Make(
    xls::ChannelQueue* queue, bool multisymbol, uint64_t data_value_index,
    std::optional<uint64_t> tlast_value_index,
    std::optional<uint64_t> tkeep_value_index) {
  if (queue->channel()->kind() != xls::ChannelKind::kStreaming) {
    return absl::InvalidArgumentError(
        "Make expects queue from a Streaming channel.");
  }
  if (queue->channel()->supported_ops() == ChannelOps::kSendReceive) {
    return absl::InvalidArgumentError(
        "Given a bidirectional channel. These are not supported.");
  }
  // Check the type of the queue
  const auto type = queue->channel()->type();
  if (!type->IsTuple()) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Channel type must be a Tuple, not '%s'", type->ToString()));
  }
  auto tuple = type->AsTupleOrDie();
  // Check index specifications
  if (data_value_index >= tuple->size()) {
    return absl::InvalidArgumentError(
        absl::StrFormat("DATA index %u is out of bounds of the "
                        "specified channel type %s",
                        data_value_index, type->ToString()));
  }
  if (multisymbol) {
    // Type of the DATA element must be an array
    auto eltype = tuple->element_type(data_value_index);
    if (!eltype->IsArray()) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "DATA value idx=%u should be of array type, but it is '%s'",
          data_value_index, eltype->ToString()));
    }
  }
  if (tlast_value_index) {
    auto idx = *tlast_value_index;
    if (idx >= tuple->size()) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "TLAST index %u is out of bounds of the specified channel type %s",
          idx, type->ToString()));
    }
    // TLAST must point to bits[1] type
    auto eltype = tuple->element_type(idx);
    if (!eltype->IsBits() || eltype->AsBitsOrDie()->bit_count() != 1) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "TLAST value idx=%u should be of type 'bits[1]', but it is '%s'", idx,
          eltype->ToString()));
    }
  }
  if (tkeep_value_index) {
    auto idx = *tkeep_value_index;
    if (idx >= tuple->size()) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "TKEEP index %u is out of bounds of the specified channel type %s",
          idx, type->ToString()));
    }
    // TKEEP must have correct size
    uint64_t expected_size =
        multisymbol
            ? tuple->element_type(data_value_index)->AsArrayOrDie()->size()
            : 1;
    auto eltype = tuple->element_type(idx);
    if (!eltype->IsBits() ||
        eltype->AsBitsOrDie()->bit_count() != expected_size) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "TKEEP value idx=%u should be of type 'bits[%u]', but it is '%s'",
          idx, expected_size, eltype->ToString()));
    }
  }
  return IrAxiStreamLike(queue, multisymbol, data_value_index,
                         tlast_value_index, tkeep_value_index);
}

IrAxiStreamLike::IrAxiStreamLike(xls::ChannelQueue* queue, bool multisymbol,
                                 uint64_t data_index,
                                 std::optional<uint64_t> tlast_index,
                                 std::optional<uint64_t> tkeep_index)
    : queue_(queue),
      multisymbol_(multisymbol),
      data_index_(data_index),
      tlast_index_(tlast_index),
      tkeep_index_(tkeep_index) {
  auto chan_type = queue->channel()->type()->AsTupleOrDie();
  auto data_type = chan_type->element_type(data_index);
  auto sym_type =
      multisymbol ? data_type->AsArrayOrDie()->element_type() : data_type;
  num_symbols_ = multisymbol ? data_type->AsArrayOrDie()->size() : 1;
  symbol_bits_ = sym_type->GetFlatBitCount();
  symbol_bytes_padded_ = (symbol_bits_ + 7) / 8;
  zero_padding_ = Value(Bits(symbol_bytes_padded_ * 8 - symbol_bits_));
  zero_payload_ = ZeroOfType(chan_type);
  read_channel_ = queue->channel()->CanSend();
  // Initialize all CPU-accessible registers including data holding register
  // which will include padding
  UnpackChannelPayload(zero_payload_);
}

Value IrAxiStreamLike::PackChannelPayload() {
  // Start with a zero-filled payload
  std::vector<Value> payload{zero_payload_.elements().begin(),
                             zero_payload_.elements().end()};
  // Fill data item
  if (multisymbol_) {
    std::vector<Value> symbols;
    for (size_t i = 0; i < num_symbols_; i++) {
      // odd-indexed values are padding, skip them
      symbols.push_back(data_reg_[2 * i]);
    }
    payload[data_index_] = Value::ArrayOrDie(symbols);
  } else {
    payload[data_index_] = data_reg_[0];
  }
  // Fill TLAST
  if (tlast_index_) {
    payload[*tlast_index_] =
        Value(Bits(absl::InlinedVector<bool, 1>({tlast_reg_})));
  }
  // Fill TKEEP
  if (tkeep_index_) {
    payload[*tkeep_index_] = Value(Bits(tkeep_reg_));
  }
  return Value::Tuple(payload);
}

void IrAxiStreamLike::UnpackChannelPayload(const Value& value) {
  XLS_CHECK(value.SameTypeAs(zero_payload_));
  // unpack data payload + add padding for CPU
  const auto& data_value = value.element(data_index_);
  data_reg_.clear();
  if (multisymbol_) {
    // data_value is an array
    for (size_t i = 0; i < data_value.size(); i++) {
      data_reg_.push_back(data_value.element(i));
      data_reg_.push_back(zero_padding_);
    }
  } else {
    XLS_CHECK_EQ(data_value.GetFlatBitCount(), symbol_bits_);
    data_reg_.push_back(data_value);
  }
  // unpack tlast & tkeep
  if (tlast_index_) {
    const auto& tlast_value = value.element(*tlast_index_);
    tlast_reg_ = tlast_value.bits().Get(0);
  } else {
    tlast_reg_ = false;
  }
  if (tkeep_index_) {
    const auto& tkeep_value = value.element(*tkeep_index_);
    tkeep_reg_ = tkeep_value.bits().ToBitVector();
  } else {
    tkeep_reg_.clear();
    tkeep_reg_.resize(num_symbols_, true);
  }
}

absl::StatusOr<uint64_t> IrAxiStreamLike::DataRead(uint64_t offset,
                                                   uint64_t byte_count) const {
  // Note: data_reg_ already includes inter-symbol padding
  XLS_ASSIGN_OR_RETURN(
      uint64_t ans,
      ValueArrayReadUInt64(data_reg_, this->queue_->channel()->name(), offset,
                           byte_count));
  return ans;
}

absl::Status IrAxiStreamLike::DataWrite(uint64_t offset, uint64_t data,
                                        uint64_t byte_count) {
  // Note: data_reg_ already includes inter-symbol padding
  XLS_ASSIGN_OR_RETURN(
      auto data_reg,
      ValueArrayWriteUInt64(data_reg_, this->queue_->channel()->name(), offset,
                            byte_count, data));
  data_reg_ = data_reg;
  return absl::OkStatus();
}

absl::StatusOr<uint8_t> IrAxiStreamLike::GetPayloadData8(
    uint64_t offset) const {
  return DataRead(offset, 1);
}

absl::StatusOr<uint16_t> IrAxiStreamLike::GetPayloadData16(
    uint64_t offset) const {
  return DataRead(offset, 2);
}

absl::StatusOr<uint32_t> IrAxiStreamLike::GetPayloadData32(
    uint64_t offset) const {
  return DataRead(offset, 4);
}

absl::StatusOr<uint64_t> IrAxiStreamLike::GetPayloadData64(
    uint64_t offset) const {
  return DataRead(offset, 8);
}

absl::Status IrAxiStreamLike::SetPayloadData8(uint64_t offset, uint8_t data) {
  return DataWrite(offset, data, 1);
}

absl::Status IrAxiStreamLike::SetPayloadData16(uint64_t offset, uint16_t data) {
  return DataWrite(offset, data, 2);
}

absl::Status IrAxiStreamLike::SetPayloadData32(uint64_t offset, uint32_t data) {
  return DataWrite(offset, data, 4);
}

absl::Status IrAxiStreamLike::SetPayloadData64(uint64_t offset, uint64_t data) {
  return DataWrite(offset, data, 8);
}

bool IrAxiStreamLike::IsReady() const {
  if (read_channel_) {
    // Read FIFO
    return !queue_->IsEmpty();
  }
  // Write FIFO
  return queue_->GetSize() < kWriteFifoMaxDepth;
}

absl::Status IrAxiStreamLike::Transfer() {
  if (read_channel_) {
    std::optional<Value> read_value = queue_->Read();
    if (!read_value) {
      return absl::InternalError(absl::StrFormat(
          "Streaming queue '%s' is empty", queue_->channel()->name()));
    }
    UnpackChannelPayload(*read_value);
    return absl::OkStatus();
  }

  if (queue_->GetSize() >= kWriteFifoMaxDepth) {
    return absl::InternalError(
        absl::StrFormat("Streaming queue '%s' overflown, data is lost",
                        queue_->channel()->name()));
  }

  XLS_RETURN_IF_ERROR(queue_->Write(PackChannelPayload()));
  return absl::OkStatus();
}

void IrAxiStreamLike::SetDataValid(std::vector<bool> dataValid) {
  XLS_CHECK_EQ(dataValid.size(), num_symbols_);
  for (size_t i = 0; i < num_symbols_; i++) {
    tkeep_reg_[i] = dataValid[i];
  }
}

std::vector<bool> IrAxiStreamLike::GetDataValid() const {
  std::vector<bool> ret(num_symbols_);
  std::copy(tkeep_reg_.begin(), tkeep_reg_.end(), ret.begin());
  return ret;
}

void IrAxiStreamLike::SetLast(bool last) { tlast_reg_ = last; }

bool IrAxiStreamLike::GetLast() const { return tlast_reg_; }

}  // namespace xls::simulation::generic
