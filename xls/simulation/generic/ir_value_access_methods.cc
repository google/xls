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

#include "xls/simulation/generic/ir_value_access_methods.h"

#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/bits.h"
#include "xls/ir/value.h"

namespace xls::simulation::generic {

static absl::Status ReadBits(const Value value_node,
                             uint64_t& processed_value_bits,
                             uint64_t const start_bit,
                             uint64_t& used_payload_bits,
                             uint64_t const bits_to_read, uint64_t& payload_) {
  if (value_node.IsToken() || value_node.kind() == ValueKind::kInvalid)
    return absl::InternalError("Got token or invalid XLS::Value");
  if (used_payload_bits >= bits_to_read) {
    return absl::OkStatus();
  }

  uint64_t bits_size = value_node.GetFlatBitCount();
  if (processed_value_bits + bits_size < start_bit) {
    processed_value_bits += bits_size;
    return absl::OkStatus();
  }

  if (value_node.IsBits()) {
    for (int i = 0; i < value_node.bits().bit_count();
         ++i, ++processed_value_bits) {
      if (used_payload_bits >= bits_to_read) {
        return absl::OkStatus();
      }
      if (start_bit <= processed_value_bits) {
        payload_ |= ((uint64_t)value_node.bits().bitmap().Get(i))
                    << used_payload_bits;
        used_payload_bits++;
      }
    }
    return absl::OkStatus();
  }

  for (auto& node : value_node.elements()) {
    XLS_RETURN_IF_ERROR(ReadBits(node, processed_value_bits, start_bit,
                                 used_payload_bits, bits_to_read, payload_));
    if (used_payload_bits >= bits_to_read) {
      return absl::OkStatus();
    }
  }
  return absl::OkStatus();
}

absl::StatusOr<uint64_t> ValueArrayReadUInt64(
    absl::Span<const Value> data_array, std::string_view channel_name,
    uint64_t byte_offset, uint64_t byte_count) {
  uint64_t size_ = 0;
  for (auto& it : data_array) {
    size_ += it.GetFlatBitCount();
  }
  int byte_size_ = (size_ + 7) / 8;

  // Outside of the address range
  if (byte_offset >= byte_size_)
    return absl::InvalidArgumentError("Offset: " + std::to_string(byte_offset) +
                                      std::string(" is outside ") +
                                      std::string(channel_name) + " range");

  uint64_t ans = 0;
  uint64_t processed_value_bits = 0;
  uint64_t used_payload_bits = 0;
  for (auto& it : data_array) {
    XLS_RETURN_IF_ERROR(ReadBits(it, processed_value_bits, byte_offset * 8,
                                 used_payload_bits, byte_count * 8, ans));
  }
  return ans;
}

static absl::StatusOr<Value> UpdateBits(const Value& value_node,
                                        uint64_t& processed_value_bits,
                                        uint64_t const start_bit,
                                        uint64_t& used_payload_bits,
                                        uint64_t const bits_to_update,
                                        uint64_t const payload) {
  if (value_node.IsToken() || value_node.kind() == ValueKind::kInvalid)
    return absl::InternalError("Got Token or invalid XLS::Value");
  if (used_payload_bits >= bits_to_update) {
    return value_node;
  }

  uint64_t bits_size = value_node.GetFlatBitCount();
  if (processed_value_bits + bits_size < start_bit) {
    processed_value_bits += bits_size;
    return value_node;
  }

  if (value_node.IsBits()) {
    Bits bit_representation = value_node.bits();
    for (int i = 0; i < bit_representation.bit_count();
         ++i, ++processed_value_bits) {
      if (used_payload_bits >= bits_to_update) {
        return Value(bit_representation);
      }
      if (start_bit <= processed_value_bits) {
        bit_representation = bit_representation.UpdateWithSet(
            i, !!(payload & (1LL << used_payload_bits)));
        used_payload_bits++;
      }
    }
    return Value(bit_representation);
  }

  std::vector<Value> new_elements;
  new_elements.reserve(value_node.size());
  for (auto& node : value_node.elements()) {
    XLS_ASSIGN_OR_RETURN(
        Value new_node, UpdateBits(node, processed_value_bits, start_bit,
                                   used_payload_bits, bits_to_update, payload));
    new_elements.push_back(new_node);
  }
  if (value_node.IsTuple()) return Value::Tuple(new_elements);
  return Value::Array(new_elements);
}

absl::StatusOr<std::vector<Value>> ValueArrayWriteUInt64(
    absl::Span<const Value> data_array, std::string_view channel_name,
    uint64_t byte_offset, uint64_t byte_count, uint64_t const payload_) {
  uint64_t size_ = 0;
  for (auto& it : data_array) {
    size_ += it.GetFlatBitCount();
  }
  uint64_t byte_size_ = (size_ + 7) / 8;

  // Outside of the address range
  if (byte_offset >= byte_size_)
    return absl::InvalidArgumentError("Offset: " + std::to_string(byte_offset) +
                                      std::string(" is outside ") +
                                      std::string(channel_name) + " range");
  uint64_t processed_value_bits = 0;
  uint64_t used_payload_bits = 0;
  std::vector<Value> output;
  output.reserve(data_array.size());
  for (auto& it : data_array) {
    XLS_ASSIGN_OR_RETURN(
        Value updated_value,
        UpdateBits(it, processed_value_bits, byte_offset * 8, used_payload_bits,
                   byte_count * 8, payload_));
    output.push_back(updated_value);
  }
  return output;
}

}  // namespace xls::simulation::generic
