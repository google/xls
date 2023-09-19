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

#include "xls/simulation/generic/dmastreammanager.h"

#include <climits>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xls/common/bits_util.h"
#include "xls/common/logging/logging.h"
#include "xls/common/status/status_macros.h"
#include "xls/simulation/generic/streamdmachannel.h"

namespace xls::simulation::generic {

DmaStreamManager::DmaStreamManager(uint64_t base_address)
    : IChannelManager(base_address),
      highest_channel_id_(-1),
      dma_channel_irq_mask_reg_size_(sizeof(uint64_t)),
      dma_channel_irq_reg_size_(sizeof(uint64_t)) {
  offset_to_first_dma_channel_ = CalculateOffsetToFirstDMAChannel();
  // No channels registered yet, assign offset to first addresable byte in the
  // space reserved for the first DMA Channel
  address_range_end_ = offset_to_first_dma_channel_;
}

absl::Status DmaStreamManager::RegisterEndpoint(
    std::unique_ptr<IDmaEndpoint> endpoint, int64_t id,
    IMasterPort* bus_master_port) {
  if (id_to_channel_.find(id) != id_to_channel_.end()) {
    return absl::InvalidArgumentError("Id: " + std::to_string(id) +
                                      " already in use!");
  }
  if (id > highest_channel_id_) {
    highest_channel_id_ = id;
    channel_irq_mask_.resize(id + 1, false);
    active_irq_mask_.resize(id + 1, false);
    dma_channel_irq_mask_reg_size_ =
        (1 + (id / (sizeof(uint64_t) * CHAR_BIT))) * sizeof(uint64_t);
    dma_channel_irq_reg_size_ = dma_channel_irq_mask_reg_size_;
    offset_to_first_dma_channel_ = CalculateOffsetToFirstDMAChannel();
    address_range_end_ += kDMAChannelAddressSpace;
  }
  id_to_channel_.insert({id, std::make_unique<StreamDmaChannel>(
                                 std::move(endpoint), bus_master_port)});
  return absl::OkStatus();
}

absl::Status DmaStreamManager::CheckAddressAlignment(channel_addr_t address,
                                                     uint64_t bytes_count) {
  if ((address & (bytes_count - 1)) != 0) {
    return absl::InvalidArgumentError(
        "Unaligned access, address: " + std::to_string(address) +
        " access width: " + std::to_string(bytes_count));
  }
  return absl::OkStatus();
}

uint64_t DmaStreamManager::IRQBitAccess(const std::vector<bool>& bits,
                                        uint64_t bytes_count,
                                        uint64_t byte_offset,
                                        uint64_t mask) const {
  uint64_t return_payload = 0;
  uint64_t current_byte = byte_offset;
  uint64_t current_first_bit = current_byte * CHAR_BIT;
  // Access IRQ mask
  for (int byte = 0; (byte < bytes_count) && (current_first_bit < bits.size());
       ++byte) {
    for (int bit = 0;
         (bit < CHAR_BIT) && ((current_first_bit + bit) < bits.size()); ++bit) {
      uint64_t bits_to_shift = CHAR_BIT * byte + bit;
      return_payload |= static_cast<int>(bits[current_first_bit + bit])
                        << bits_to_shift;
    }
    current_byte++;
    current_first_bit += CHAR_BIT;
  }
  return return_payload & mask;
}

uint64_t DmaStreamManager::CalculateOffsetToFirstDMAChannel() const {
  uint64_t minimal_offset =
      kNumberOfDMAChannelsRegSize + kOffsetToFirstDMAChannelRegSize +
      dma_channel_irq_mask_reg_size_ + dma_channel_irq_reg_size_;
  // Align to 64 bytes
  return ((minimal_offset + kChannelAlignment - 1) / kChannelAlignment) *
         kChannelAlignment;
}
absl::StatusOr<uint64_t> DmaStreamManager::ReadNumberOfDMAChannels(
    uint64_t byte_offset, uint64_t mask) const {
  return ((highest_channel_id_ + 1) >> (CHAR_BIT * byte_offset)) & mask;
}

absl::StatusOr<uint64_t> DmaStreamManager::ReadOffsetToFirstDMAChannel(
    uint64_t byte_offset, uint64_t mask) const {
  return (offset_to_first_dma_channel_ >> (CHAR_BIT * byte_offset)) & mask;
}

absl::StatusOr<uint64_t> DmaStreamManager::ReadDMAChannelIRQMask(
    uint64_t bytes_count, uint64_t byte_offset, uint64_t mask) const {
  return IRQBitAccess(channel_irq_mask_, bytes_count, byte_offset, mask);
}

absl::StatusOr<uint64_t> DmaStreamManager::ReadDMAChannelIRQ(
    uint64_t bytes_count, uint64_t byte_offset, uint64_t mask) const {
  return IRQBitAccess(active_irq_mask_, bytes_count, byte_offset, mask);
}

absl::StatusOr<uint64_t> DmaStreamManager::ReadDMAChannel(
    channel_addr_t address, uint64_t bytes_count) {
  uint64_t byte_offset = address - base_address_ - offset_to_first_dma_channel_;
  uint64_t id = byte_offset / kChannelAlignment;
  if (id_to_channel_.find(id) == id_to_channel_.end()) {
    XLS_LOG(WARNING) << "DMA channel with id:" + std::to_string(id) +
                            " does not exist.";
    return 0;
  }

  const auto& channel = id_to_channel_[id];
  uint64_t in_channel_address = byte_offset % kChannelAlignment;
  uint64_t channel_register = in_channel_address / kDMAChannelRegisterSize;
  uint64_t return_payload;

  switch (channel_register) {
    case kDMAChannelTransferBaseAddressIdx:
      return_payload = channel->GetTransferBaseAddress();
      break;
    case kDMAChannelMaxTransferLengthIdx:
      return_payload = channel->GetMaxTransferLength();
      break;
    case kDMAChannelTransferredLengthIdx:
      return_payload = channel->GetTransferredLength();
      break;
    case kDMAChannelControlRegIdx:
      return_payload = channel->GetControlRegister();
      break;
    case kDMAChannelIRQIdx:
      return_payload = channel->GetIRQReg();
      break;
    default:
      return_payload = 0;
      XLS_LOG(WARNING) << "Offset:" + std::to_string(in_channel_address) +
                              " doesn't map to any DMA register!";
  }

  uint64_t byte_shift = in_channel_address % kDMAChannelRegisterSize;
  uint64_t mask = Mask(CHAR_BIT * bytes_count);

  return (return_payload >> (CHAR_BIT * byte_shift)) & mask;
}

absl::StatusOr<uint64_t> DmaStreamManager::ReadInternal(
    channel_addr_t address, uint64_t log_byte_count) {
  uint64_t bytes_count = 1 << log_byte_count;
  XLS_RETURN_IF_ERROR(CheckAddressAlignment(address, bytes_count));
  uint64_t byte_offset = address - base_address_;

  if (offset_to_first_dma_channel_ <= byte_offset)
    return ReadDMAChannel(address, bytes_count);

  uint64_t mask = Mask(CHAR_BIT * bytes_count);

  if (byte_offset < kNumberOfDMAChannelsRegSize)
    return ReadNumberOfDMAChannels(byte_offset, mask);
  byte_offset -= kNumberOfDMAChannelsRegSize;

  if (byte_offset < kOffsetToFirstDMAChannelRegSize)
    return ReadOffsetToFirstDMAChannel(byte_offset, mask);
  byte_offset -= kOffsetToFirstDMAChannelRegSize;

  if (byte_offset < dma_channel_irq_mask_reg_size_)
    return ReadDMAChannelIRQMask(bytes_count, byte_offset, mask);
  byte_offset -= dma_channel_irq_mask_reg_size_;

  if (byte_offset < dma_channel_irq_reg_size_)
    return ReadDMAChannelIRQ(bytes_count, byte_offset, mask);

  XLS_LOG(WARNING) << "Read access to not existing register";
  return 0;
}

absl::Status DmaStreamManager::WriteNumberOfDMAChannels() {
  XLS_LOG(WARNING) << "Write access to RO register: NumberOfDMAChannels!";
  return absl::OkStatus();
}

absl::Status DmaStreamManager::WriteOffsetToFirstDMAChannel() {
  XLS_LOG(WARNING) << "Write access to RO register: OffsetToFirstDMAChannel!";
  return absl::OkStatus();
}

absl::Status DmaStreamManager::WriteDMAChannelIRQMask(uint64_t byte_offset,
                                                      uint64_t bytes_count,
                                                      uint64_t payload) {
  uint64_t current_byte = byte_offset;
  uint64_t current_first_bit = current_byte * CHAR_BIT;
  for (int byte = 0;
       (byte < bytes_count) && (current_first_bit < channel_irq_mask_.size());
       ++byte) {
    for (int bit = 0; (bit < CHAR_BIT) &&
                      (current_first_bit + bit < channel_irq_mask_.size());
         ++bit) {
      channel_irq_mask_[current_first_bit + bit] = ((payload & 1) != 0u);
      payload >>= 1;
    }
    current_byte++;
    current_first_bit += CHAR_BIT;
  }
  return absl::OkStatus();
}

absl::Status DmaStreamManager::WriteDMAChannelIRQ() {
  XLS_LOG(WARNING) << "Write access to RO register: ActiveIRQs!";
  return absl::OkStatus();
}

absl::Status DmaStreamManager::WriteDMAChannel(channel_addr_t address,
                                               uint64_t payload,
                                               uint64_t bytes_count) {
  uint64_t byte_offset = address - base_address_ - offset_to_first_dma_channel_;
  uint64_t id = byte_offset / kChannelAlignment;
  if (id_to_channel_.find(id) == id_to_channel_.end()) {
    XLS_LOG(WARNING) << "DMA channel with id:" + std::to_string(id) +
                            " does not exist.";
    return absl::OkStatus();
  }

  uint64_t in_channel_address = byte_offset % kChannelAlignment;
  uint64_t channel_register = in_channel_address / kDMAChannelRegisterSize;

  auto HandleShiftedWrite = [=](uint64_t (StreamDmaChannel::*Getter)(),
                                void (StreamDmaChannel::*Setter)(uint64_t)) {
    const auto& channel = id_to_channel_[id];
    uint64_t byte_shift = in_channel_address % kDMAChannelRegisterSize;
    uint64_t mask = Mask(BytesToBits(bytes_count));

    uint64_t updated_value = (channel.get()->*Getter)();
    updated_value &= ~(mask << BytesToBits(byte_shift));
    updated_value |= (payload & mask) << BytesToBits(byte_shift);
    (channel.get()->*Setter)(updated_value);
  };

  switch (channel_register) {
    case kDMAChannelTransferBaseAddressIdx:
      HandleShiftedWrite(&StreamDmaChannel::GetTransferBaseAddress,
                         &StreamDmaChannel::SetTransferBaseAddress);
      break;
    case kDMAChannelMaxTransferLengthIdx:
      HandleShiftedWrite(&StreamDmaChannel::GetMaxTransferLength,
                         &StreamDmaChannel::SetMaxTransferLength);
      break;
    case kDMAChannelTransferredLengthIdx:
      XLS_LOG(WARNING)
          << "Write access to DMA channel RO register: TransferredLength!";
      break;
    case kDMAChannelControlRegIdx:
      HandleShiftedWrite(&StreamDmaChannel::GetControlRegister,
                         &StreamDmaChannel::SetControlRegister);
      break;
    case kDMAChannelIRQIdx:
      HandleShiftedWrite(&StreamDmaChannel::GetIRQReg,
                         &StreamDmaChannel::ClearIRQReg);
      break;
    default:
      XLS_LOG(WARNING) << "Offset:" + std::to_string(in_channel_address) +
                              " doesn't map to any DMA register!";
  }

  return absl::OkStatus();
}

absl::Status DmaStreamManager::WriteInternal(channel_addr_t address,
                                             uint64_t payload,
                                             uint64_t log_byte_count) {
  uint64_t bytes_count = 1 << log_byte_count;
  XLS_RETURN_IF_ERROR(CheckAddressAlignment(address, bytes_count));
  uint64_t byte_offset = address - base_address_;

  if (offset_to_first_dma_channel_ <= byte_offset)
    return WriteDMAChannel(address, payload, bytes_count);

  if (byte_offset < kNumberOfDMAChannelsRegSize)
    return WriteNumberOfDMAChannels();
  byte_offset -= kNumberOfDMAChannelsRegSize;

  if (byte_offset < kOffsetToFirstDMAChannelRegSize)
    return WriteOffsetToFirstDMAChannel();
  byte_offset -= kOffsetToFirstDMAChannelRegSize;

  if (byte_offset < dma_channel_irq_mask_reg_size_)
    return WriteDMAChannelIRQMask(byte_offset, bytes_count, payload);
  byte_offset -= dma_channel_irq_mask_reg_size_;

  if (byte_offset < dma_channel_irq_reg_size_) return WriteDMAChannelIRQ();

  XLS_LOG(WARNING) << "Write access to not existing register";
  return absl::OkStatus();
}

absl::Status DmaStreamManager::Update() {
  for (const auto& it : id_to_channel_) {
    XLS_RETURN_IF_ERROR(it.second->Update());
  }
  return absl::OkStatus();
}

absl::Status DmaStreamManager::UpdateIRQ() {
  for (const auto& it : id_to_channel_) {
    XLS_RETURN_IF_ERROR(it.second->UpdateIRQ());
    active_irq_mask_[it.first] =
        channel_irq_mask_[it.first] && it.second->GetIRQ();
  }
  return absl::OkStatus();
}

bool DmaStreamManager::GetIRQ() {
  for (int i = 0; i < active_irq_mask_.size(); ++i) {
    if (active_irq_mask_[i]) return true;
  }

  return false;
}

}  // namespace xls::simulation::generic
