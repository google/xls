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

#include "xls/simulation/generic/singlevaluemanager.h"

#include <cstdint>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "xls/common/logging/logging.h"
#include "xls/common/status/status_macros.h"
#include "xls/simulation/generic/common.h"
#include "xls/simulation/generic/ichannelmanager.h"

namespace xls::simulation::generic {

SingleValueManager::SingleValueManager(uint64_t base_address)
    : IChannelManager(base_address) {}

absl::Status SingleValueManager::RegisterIRegister(
    std::unique_ptr<IRegister> reg, uint64_t offset) {
  if (reg == nullptr)
    return absl::InvalidArgumentError("Invalid unique pointer!");
  if (offset % kChannelAlignment != 0)
    return absl::InvalidArgumentError("Unaligned channel offset: " +
                                      std::to_string(offset));

  if (!IsOffsetFree(offset, reg->GetChannelWidth()))
    return absl::InvalidArgumentError("Channel to big or bad offset " +
                                      std::to_string(offset));

  // Check if registered channel moves address range upperbound
  if (offset + ((reg->GetChannelWidth() + 63) / 64) * kChannelAlignment >
      this->address_range_end_)
    this->address_range_end_ =
        offset + ((reg->GetChannelWidth() + 63) / 64) * kChannelAlignment;
  this->channel_to_offset_.insert({reg.get(), offset});
  this->offset_to_channel_.insert({offset, std::move(reg)});
  return absl::OkStatus();
}

bool SingleValueManager::IsOffsetFree(uint64_t offset, uint64_t channel_width) {
  if (this->offset_to_channel_.empty()) {
    // No channels registered, any valid offset is good
    return true;
  }

  auto upper = this->offset_to_channel_.upper_bound(offset);

  // Insert before first mapping
  if (upper == this->offset_to_channel_.begin()) {
    auto before_first_element =
        (offset + BitsToBytes(channel_width)) <= upper->first;
    XLS_LOG_IF(WARNING, !before_first_element)
        << "Offset: " + std::to_string(offset) +
               " is already ocupied by channel at offset: " +
               std::to_string(upper->first);
    return before_first_element;
  }

  // Insert after last mapping
  auto lower = std::prev(upper);
  if (upper == this->offset_to_channel_.end()) {
    auto after_last_element =
        (lower->first + BitsToBytes(lower->second->GetChannelWidth())) <=
        offset;
    XLS_LOG_IF(WARNING, !after_last_element)
        << "Offset: " + std::to_string(offset) +
               " is already ocupied by channel at offset: " +
               std::to_string(lower->first);
    return after_last_element;
  }

  // Insert between existing mappings
  auto before_upper_bound_element =
      (offset + BitsToBytes(channel_width)) <= upper->first;
  auto after_lower_bound_element =
      (lower->first + BitsToBytes(lower->second->GetChannelWidth())) <= offset;
  return before_upper_bound_element && after_lower_bound_element;
}

static std::string ErrorMessageWrite(uint64_t address, uint64_t value) {
  return "Write to address: " + std::to_string(address) +
         " data: " + std::to_string(value) +
         " failed! Address doesn't match to channel.";
}

static std::string ErrorMessageRead(uint64_t address) {
  return "Read from address: " + std::to_string(address) +
         " failed! Address doesn't match to channel.";
}

absl::StatusOr<IRegister*> SingleValueManager::GetMappingOrStatus(
    uint64_t address, std::string&& err_msg) {
  uint64_t internal_address = address - this->base_address_;
  auto upper = this->offset_to_channel_.upper_bound(internal_address);
  if (upper == this->offset_to_channel_.begin())
    return absl::InternalError(err_msg);

  auto possible_mapping = std::prev(upper);
  uint64_t in_channel_offset = internal_address - possible_mapping->first;
  if (BitsToBytes(possible_mapping->second->GetChannelWidth()) <=
      in_channel_offset)
    return absl::InternalError(err_msg);
  return possible_mapping->second.get();
}

static constexpr absl::Status (*SetPayloadWPow[4])(IRegister*, uint64_t,
                                                   uint64_t) = {
    [](IRegister* stream, uint64_t addr, uint64_t data) -> absl::Status {
      return stream->SetPayloadData8(addr, data);
    },
    [](IRegister* stream, uint64_t addr, uint64_t data) -> absl::Status {
      return stream->SetPayloadData16(addr, data);
    },
    [](IRegister* stream, uint64_t addr, uint64_t data) -> absl::Status {
      return stream->SetPayloadData32(addr, data);
    },
    [](IRegister* stream, uint64_t addr, uint64_t data) -> absl::Status {
      return stream->SetPayloadData64(addr, data);
    },
};

absl::Status SingleValueManager::WriteInternal(channel_addr_t address,
                                               uint64_t data,
                                               uint64_t log_byte_count) {
  XLS_ASSIGN_OR_RETURN(
      IRegister * reg,
      GetMappingOrStatus(address, ErrorMessageWrite(address, data)));
  uint64_t internal_address = address - this->base_address_;
  uint64_t in_channel_offset =
      internal_address - this->channel_to_offset_.at(reg);

  XLS_RETURN_IF_ERROR(
      SetPayloadWPow[log_byte_count](reg, in_channel_offset, data));

  uint64_t byte_count = 1 << log_byte_count;
  if ((address + byte_count - 1) % kChannelAlignment >
      address % kChannelAlignment)
    return absl::OkStatus();

  // Write may span multiple channels, check if true
  uint64_t overlap = (address + byte_count) % kChannelAlignment;
  uint64_t used_bytes = byte_count - overlap;
  data >>= BytesToBits(used_bytes);

  auto overlapping_channel =
      GetMappingOrStatus(address + used_bytes, std::string());
  if (!overlapping_channel.ok()) return absl::OkStatus();
  int address_in_overlapping_channel = 0;
  for (int i = 0; overlap > 0; ++i) {
    if (overlap & 1) {
      XLS_RETURN_IF_ERROR(SetPayloadWPow[i](
          overlapping_channel.value(), address_in_overlapping_channel, data));
      address_in_overlapping_channel += 1 << i;
      data >>= (8 << i);
    }
    overlap >>= 1;
  }
  return absl::OkStatus();
}

absl::Status SingleValueManager::WriteU8AtAddress(channel_addr_t address,
                                                  uint8_t value) {
  return this->WriteInternal(address, value, 0);
}

absl::Status SingleValueManager::WriteU16AtAddress(channel_addr_t address,
                                                   uint16_t value) {
  return this->WriteInternal(address, value, 1);
}

absl::Status SingleValueManager::WriteU32AtAddress(channel_addr_t address,
                                                   uint32_t value) {
  return this->WriteInternal(address, value, 2);
}

absl::Status SingleValueManager::WriteU64AtAddress(channel_addr_t address,
                                                   uint64_t value) {
  return this->WriteInternal(address, value, 3);
}

static constexpr absl::StatusOr<uint64_t> (*GetPayloadWPow[4])(IRegister*,
                                                               uint64_t) = {
    [](IRegister* stream, uint64_t addr) -> absl::StatusOr<uint64_t> {
      return stream->GetPayloadData8(addr);
    },
    [](IRegister* stream, uint64_t addr) -> absl::StatusOr<uint64_t> {
      return stream->GetPayloadData16(addr);
    },
    [](IRegister* stream, uint64_t addr) -> absl::StatusOr<uint64_t> {
      return stream->GetPayloadData32(addr);
    },
    [](IRegister* stream, uint64_t addr) -> absl::StatusOr<uint64_t> {
      return stream->GetPayloadData64(addr);
    },
};

absl::StatusOr<uint64_t> SingleValueManager::ReadInternal(
    channel_addr_t address, uint64_t log_byte_count) {
  XLS_ASSIGN_OR_RETURN(IRegister * reg,
                       GetMappingOrStatus(address, ErrorMessageRead(address)));
  uint64_t internal_address = address - this->base_address_;
  uint64_t in_channel_offset =
      internal_address - this->channel_to_offset_.at(reg);
  XLS_ASSIGN_OR_RETURN(uint64_t data,
                       GetPayloadWPow[log_byte_count](reg, in_channel_offset));

  uint64_t byte_count = 1 << log_byte_count;
  if ((address + byte_count - 1) % kChannelAlignment >=
      address % kChannelAlignment) {
    return data;
  }

  // Read may span multiple channels, check if true
  uint64_t overlap = (address + byte_count) % kChannelAlignment;
  uint64_t used_bytes = byte_count - overlap;

  auto overlapping_channel =
      GetMappingOrStatus(address + used_bytes, std::string());
  if (!overlapping_channel.ok()) return data;
  int address_in_overlapping_channel = 0;
  for (int i = 0; overlap > 0; ++i, overlap >>= 1) {
    if (overlap & 1) {
      auto next_channel_bytes = GetPayloadWPow[i](
          overlapping_channel.value(), address_in_overlapping_channel);
      if (!next_channel_bytes.ok()) return data;
      data |= next_channel_bytes.value()
              << BytesToBits(used_bytes + address_in_overlapping_channel);
      address_in_overlapping_channel |= 1 << i;
    }
  }
  return data;
}

absl::StatusOr<uint8_t> SingleValueManager::ReadU8AtAddress(
    channel_addr_t address) {
  return this->ReadInternal(address, 0);
}

absl::StatusOr<uint16_t> SingleValueManager::ReadU16AtAddress(
    channel_addr_t address) {
  return this->ReadInternal(address, 1);
}

absl::StatusOr<uint32_t> SingleValueManager::ReadU32AtAddress(
    channel_addr_t address) {
  return this->ReadInternal(address, 2);
}

absl::StatusOr<uint64_t> SingleValueManager::ReadU64AtAddress(
    channel_addr_t address) {
  return this->ReadInternal(address, 3);
}

}  // namespace xls::simulation::generic
