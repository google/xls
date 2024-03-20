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

#include "xls/simulation/generic/streammanager.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "xls/common/bits_util.h"
#include "xls/simulation/generic/common.h"
#include "xls/simulation/generic/istream.h"

namespace xls::simulation::generic {

using channel_addr_t = IChannelManager::channel_addr_t;

std::string ToHex(uint64_t value) {
  return "0x" + absl::StrCat(absl::Hex(value, absl::kZeroPad16));
}

absl::Status ChannelNotFound() {
  return absl::InternalError("Channel not found");
}

channel_addr_t StreamManager::StreamEnd(channel_addr_t offset_bytes,
                                        const IStream& stream) {
  return offset_bytes + StreamManager::ctrl_reg_width_bytes_ +
         BitsToBytes(stream.GetChannelWidth());
}

StreamManager::StreamManager(uint64_t base_address,
                             std::vector<StreamRange>&& sorted_channels)
    : IChannelManager(base_address), channels_(std::move(sorted_channels)) {
  if (!channels_.empty()) {
    StreamRange& stream_range = channels_[channels_.size() - 1];
    address_range_end_ =
        StreamManager::StreamEnd(stream_range.first, *stream_range.second);
  } else {
    address_range_end_ = 0x0;
  }
}

absl::StatusOr<StreamManager> StreamManager::Build(
    uint64_t base_address, const StreamCreator& BuildStreams) {
  std::vector<StreamRange> channels;
  StreamRegistrator RegisterStream = [&channels](channel_addr_t offset,
                                                 IStream* stream) {
    channels.push_back(StreamRange(offset, std::unique_ptr<IStream>(stream)));
  };
  absl::Status build_status = BuildStreams(RegisterStream);
  if (!build_status.ok()) {
    return build_status;
  }
  std::sort(channels.begin(), channels.end(),
            [](const StreamRange& a, const StreamRange& b) {
              return a.first < b.first;
            });
  for (int i = 1; i < channels.size(); ++i) {
    channel_addr_t end_of_stream = StreamManager::StreamEnd(
        channels[i - 1].first, *channels[i - 1].second);
    if (channels[i].first < end_of_stream) {
      return absl::InternalError(
          "Overlapping streams. Channel to be added at " +
          ToHex(channels[i].first) + " overlaps with a channel added at " +
          ToHex(channels[i - 1].first) + " (width: " +
          std::to_string(channels[i - 1].second->GetChannelWidth()) + ").");
    }
  }

  return StreamManager(base_address, std::move(channels));
}

// Assumption regarding R/W ops: The emulated architecture is little-endian
// Logic for truncatting values and splitting them between ctrl register area
// stream payload would differ for big-endian architectures.
//
// Another assumption: The memory outside of the channel is unmapped. Write
// should fail, even if it partially overlaps with the channel.

template <>
class StreamManager::TypedAccessToStream<uint8_t>
    : public StreamManager::AccessToStream {
 public:
  absl::StatusOr<uint64_t> GetData() const {
    return this->ShiftResult(
        this->stream_->GetPayloadData8(this->OffsetRelativeToPayload()));
  }
  absl::Status SetData(uint64_t data) const {
    return this->stream_->SetPayloadData8(this->OffsetRelativeToPayload(),
                                          data);
  }
};

template <>
class StreamManager::TypedAccessToStream<uint16_t>
    : public StreamManager::AccessToStream {
 public:
  absl::StatusOr<uint64_t> GetData() const {
    return this->ShiftResult(
        this->stream_->GetPayloadData16(this->OffsetRelativeToPayload()));
  }
  absl::Status SetData(uint64_t data) const {
    return this->stream_->SetPayloadData16(this->OffsetRelativeToPayload(),
                                           data);
  }
};

template <>
class StreamManager::TypedAccessToStream<uint32_t>
    : public StreamManager::AccessToStream {
 public:
  absl::StatusOr<uint64_t> GetData() const {
    return this->ShiftResult(
        this->stream_->GetPayloadData32(this->OffsetRelativeToPayload()));
  }
  absl::Status SetData(uint64_t data) const {
    return this->stream_->SetPayloadData32(this->OffsetRelativeToPayload(),
                                           data);
  }
};

template <>
class StreamManager::TypedAccessToStream<uint64_t>
    : public StreamManager::AccessToStream {
 public:
  absl::StatusOr<uint64_t> GetData() const {
    return this->ShiftResult(
        this->stream_->GetPayloadData64(this->OffsetRelativeToPayload()));
  }
  absl::Status SetData(uint64_t data) const {
    return this->stream_->SetPayloadData64(this->OffsetRelativeToPayload(),
                                           data);
  }
};

template <typename AccessType>
absl::StatusOr<AccessType> StreamManager::ReadAccess(
    TypedAccessToStream<AccessType> access) {
  // Ctrl register sits at the beginning of this stream (0x0)
  if (access.offset_bytes_ == StreamManager::ctrl_reg_offset_bytes_) {
    return this->ReadCtrl(access);
  }

  if (StreamManager::AccessExceedsCtrlArea(access)) {
    return this->ReadTail(access);
  }
  if (access.OffsetRelativeToPayload() >= 0) {
    return access.GetData();
  }
  return 0;
}

template <typename AccessType>
absl::Status StreamManager::WriteAccess(TypedAccessToStream<AccessType> access,
                                        AccessType value) {
  // Ctrl register sits at the beginning of this stream (0x0)
  if (access.offset_bytes_ == StreamManager::ctrl_reg_offset_bytes_) {
    this->WriteCtrl(access, value);
    return absl::OkStatus();
  }

  if (StreamManager::AccessExceedsCtrlArea(access)) {
    return this->WriteTail(access, value);
  }

  if (access.OffsetRelativeToPayload() >= 0) {
    return access.SetData(value);
  }
  return absl::OkStatus();
}

// Read from 0x0 address of a stream into a portion of returned value.
template <typename AccessType>
absl::StatusOr<uint64_t> StreamManager::ReadTail(
    const TypedAccessToStream<AccessType>& access) const {
  auto read = access.stream_->GetPayloadData64(0x0);
  if (!read.ok()) {
    return read.status();
  }

  uint64_t stream_head = read.value();
  // Assumption: litle-endian arch
  uint64_t tail_width_bytes =
      sizeof(AccessType) + access.OffsetRelativeToPayload();
  uint64_t keep_mask = xls::Mask(BytesToBits(tail_width_bytes));
  return (stream_head & keep_mask)
         << (access.shift_ - BytesToBits(access.OffsetRelativeToPayload()));
}

template <typename AccessType>
absl::Status StreamManager::WriteTail(
    const TypedAccessToStream<AccessType> access, AccessType value) const {
  // Read 8 bytes, modify them and then write them back
  auto read = access.stream_->GetPayloadData64(0x0);
  if (!read.ok()) {
    return read.status();
  }
  uint64_t stream_head = read.value();
  // Assumption: litle-endian arch
  uint64_t keep_mask = ~xls::Mask(
      BytesToBits(sizeof(AccessType) + access.OffsetRelativeToPayload()));
  stream_head =
      (stream_head & keep_mask) | (value >> -access.OffsetRelativeToPayload());
  // In case the stream is shorter, the value should be truncated
  return access.stream_->SetPayloadData64(0x0, stream_head);
}

template <typename AccessType>
bool StreamManager::AccessExceedsCtrlArea(
    const TypedAccessToStream<AccessType>& access) {
  return (access.offset_bytes_ < StreamManager::ctrl_reg_end_) &&
         (access.offset_bytes_ + sizeof(AccessType) >
          StreamManager::ctrl_reg_end_);
}

template <typename AccessType, typename F>
absl::Status StreamManager::forEachStreamInRange(
    channel_addr_t global_offset_bytes, F action) {
  static_assert(std::is_invocable<F, TypedAccessToStream<AccessType>>::value,
                "action is not callable");
  static_assert(std::numeric_limits<AccessType>::is_integer,
                "AccessType must be an integer");

  absl::StatusOr<std::pair<channel_addr_t, IStream*>> get_res;
  absl::Status action_status;

  channel_addr_t local_offset_bytes =
      global_offset_bytes - this->GetBaseAddress();

  get_res = this->GetOffsetChannelPair(local_offset_bytes);
  if (!get_res.ok()) {
    return get_res.status();
  }
  channel_addr_t rel_addr = get_res->first;
  IStream* ch1 = get_res->second;

  // Assumption: litle-endian arch
  action_status = action(TypedAccessToStream<AccessType>{ch1, rel_addr, 0});
  if (!action_status.ok()) {
    return action_status;
  }

  if (rel_addr == 0) {
    // We can skip further checks if this was an access to Ctrl register as its
    // 64-bit wide
    return absl::OkStatus();
  }

  channel_addr_t new_beginning = local_offset_bytes - rel_addr +
                                 BitsToBytes(ch1->GetChannelWidth()) +
                                 StreamManager::ctrl_reg_width_bytes_;
  if (local_offset_bytes + sizeof(AccessType) <= new_beginning) {
    return absl::OkStatus();
  }

  get_res = this->GetOffsetChannelPair(local_offset_bytes + sizeof(AccessType));
  if (!get_res.ok()) {
    return absl::OkStatus();
  }  // There was no second channel

  // Assumption: litle-endian arch
  // Read from a beginning of an adjacent stream. A shift paramenter is required
  // to adjust the value.
  uint8_t shift = BytesToBits(sizeof(AccessType) - get_res->first);
  return action(TypedAccessToStream<AccessType>{get_res->second, 0x0, shift});
}

template <typename AccessType>
absl::StatusOr<AccessType> StreamManager::ReadUAtAddress(channel_addr_t addr) {
  static_assert(std::numeric_limits<AccessType>::is_integer,
                "AccessType must be an integer");

  if (addr < this->base_address_) {
    return absl::InternalError("Address not in range");
  }

  uint64_t value = 0;
  absl::Status stream_access_status = this->forEachStreamInRange<AccessType>(
      addr, [&](const TypedAccessToStream<AccessType>& access) {
        absl::StatusOr<AccessType> read = this->ReadAccess(access);
        if (!read.ok()) {
          return read.status();
        }
        value |= *read;
        return absl::OkStatus();
      });

  if (stream_access_status.ok()) {
    return value;
  }
  return stream_access_status;
}

template <typename AccessType>
absl::Status StreamManager::WriteUAtAddress(channel_addr_t addr,
                                            AccessType value) {
  static_assert(std::numeric_limits<AccessType>::is_integer,
                "T must be an integer");

  if (addr < this->base_address_) {
    return absl::InternalError("Address not in range");
  }

  return this->forEachStreamInRange<AccessType>(
      addr, [&](const TypedAccessToStream<AccessType>& access) {
        return this->WriteAccess(access, value);
      });
}

absl::Status StreamManager::WriteU8AtAddress(channel_addr_t addr,
                                             uint8_t value) {
  return this->WriteUAtAddress(addr, value);
}

absl::Status StreamManager::WriteU16AtAddress(channel_addr_t addr,
                                              uint16_t value) {
  return this->WriteUAtAddress(addr, value);
}

absl::Status StreamManager::WriteU32AtAddress(channel_addr_t addr,
                                              uint32_t value) {
  return this->WriteUAtAddress(addr, value);
}

absl::Status StreamManager::WriteU64AtAddress(channel_addr_t addr,
                                              uint64_t value) {
  return this->WriteUAtAddress(addr, value);
}

absl::StatusOr<uint8_t> StreamManager::ReadU8AtAddress(channel_addr_t addr) {
  return this->ReadUAtAddress<uint8_t>(addr);
}

absl::StatusOr<uint16_t> StreamManager::ReadU16AtAddress(channel_addr_t addr) {
  return this->ReadUAtAddress<uint16_t>(addr);
}

absl::StatusOr<uint32_t> StreamManager::ReadU32AtAddress(channel_addr_t addr) {
  return this->ReadUAtAddress<uint32_t>(addr);
}

absl::StatusOr<uint64_t> StreamManager::ReadU64AtAddress(channel_addr_t addr) {
  return this->ReadUAtAddress<uint64_t>(addr);
}

void StreamManager::WriteCtrl(const AccessToStream& access, uint64_t value) {
  value >>= access.shift_;
  if (value & StreamManager::ctrl_reg_DOXFER_) {
    this->tsfr_status_ = access.stream_->Transfer();
  }
  if (value & StreamManager::ctrl_reg_ERRXFER_) {
    this->tsfr_status_ = absl::OkStatus();
  }
}

uint64_t StreamManager::ReadCtrl(const AccessToStream& access) {
  uint64_t value = 0x0;
  value |= access.stream_->IsReady() ? StreamManager::ctrl_reg_READY_ : 0x0;
  value |= access.stream_->IsReadStream() ? StreamManager::ctrl_reg_DIR_ : 0x0;
  value |= this->tsfr_status_.ok() ? 0x0 : StreamManager::ctrl_reg_ERRXFER_;
  return value << access.shift_;
}

absl::StatusOr<std::pair<channel_addr_t, IStream*>>
StreamManager::GetOffsetChannelPair(channel_addr_t offset_bytes) {
  // Binary-search for a channel.
  auto it = std::upper_bound(
      this->channels_.begin(), this->channels_.end(), offset_bytes,
      [](uint64_t value, const StreamManager::StreamRange& range) -> bool {
        return value < range.first;
      });
  if (it == this->channels_.begin()) {
    return ChannelNotFound();
  }
  --it;
  channel_addr_t end_of_strem =
      StreamManager::StreamEnd(it->first, *it->second);
  if ((offset_bytes >= it->first) && (offset_bytes < end_of_strem)) {
    return std::make_pair(offset_bytes - it->first, it->second.get());
  }
  return ChannelNotFound();
}

}  // namespace xls::simulation::generic
