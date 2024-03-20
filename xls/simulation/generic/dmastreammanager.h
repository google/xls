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

#ifndef XLS_SIMULATION_GENERIC_DMASTREAMMANAGER_H_
#define XLS_SIMULATION_GENERIC_DMASTREAMMANAGER_H_

#include <map>
#include <memory>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xls/simulation/generic/iactive.h"
#include "xls/simulation/generic/ichannelmanager.h"
#include "xls/simulation/generic/idmaendpoint.h"
#include "xls/simulation/generic/iirq.h"
#include "xls/simulation/generic/imasterport.h"
#include "xls/simulation/generic/streamdmachannel.h"

namespace xls::simulation::generic {

class DmaStreamManager : public IActive, public IIRQ, public IChannelManager {
 public:
  using channel_addr_t = uint64_t;

  explicit DmaStreamManager(uint64_t base_address);
  DmaStreamManager(DmaStreamManager&& other) = default;
  DmaStreamManager& operator=(DmaStreamManager&& other) = default;

  ~DmaStreamManager() override = default;

  absl::Status RegisterEndpoint(std::unique_ptr<IDmaEndpoint> endpoint,
                                int64_t id, IMasterPort* bus_master_port);

  // IChannelManager
  absl::Status WriteU8AtAddress(channel_addr_t address,
                                uint8_t value) override {
    return WriteInternal(address, value, 0);
  }
  absl::Status WriteU16AtAddress(channel_addr_t address,
                                 uint16_t value) override {
    return WriteInternal(address, value, 1);
  }
  absl::Status WriteU32AtAddress(channel_addr_t address,
                                 uint32_t value) override {
    return WriteInternal(address, value, 2);
  }
  absl::Status WriteU64AtAddress(channel_addr_t address,
                                 uint64_t value) override {
    return WriteInternal(address, value, 3);
  }
  absl::StatusOr<uint8_t> ReadU8AtAddress(channel_addr_t address) override {
    return ReadInternal(address, 0);
  }
  absl::StatusOr<uint16_t> ReadU16AtAddress(channel_addr_t address) override {
    return ReadInternal(address, 1);
  }
  absl::StatusOr<uint32_t> ReadU32AtAddress(channel_addr_t address) override {
    return ReadInternal(address, 2);
  }
  absl::StatusOr<uint64_t> ReadU64AtAddress(channel_addr_t address) override {
    return ReadInternal(address, 3);
  }

  // IActive
  absl::Status Update() override;

  // IIRQ
  bool GetIRQ() override;
  absl::Status UpdateIRQ() override;

 protected:
  virtual absl::StatusOr<uint64_t> ReadDMAChannel(channel_addr_t address,
                                                  uint64_t bytes_count);
  virtual absl::Status WriteDMAChannel(channel_addr_t address, uint64_t payload,
                                       uint64_t bytes_count);

 private:
  static const uint64_t kChannelAlignment = sizeof(uint64_t) * CHAR_BIT;
  static const uint64_t kNumberOfDMAChannelsRegSize = sizeof(uint64_t);
  static const uint64_t kOffsetToFirstDMAChannelRegSize = sizeof(uint64_t);
  // DMA Channel mapping constants
  // Reserve 64 bytes of space for each DMA Channel
  static const uint64_t kDMAChannelAddressSpace = 64;
  static const uint64_t kDMAChannelRegisterSize = sizeof(uint64_t);
  static const uint64_t kDMAChannelTransferBaseAddressIdx = 0;  // 0x00 - 0x07
  static const uint64_t kDMAChannelMaxTransferLengthIdx = 1;    // 0x08 - 0x0F
  static const uint64_t kDMAChannelTransferredLengthIdx = 2;    // 0x10 - 0x17
  static const uint64_t kDMAChannelControlRegIdx = 3;           // 0x18 - 0x1F
  static const uint64_t kDMAChannelIRQIdx = 4;                  // 0x20 - 0x27
  static const uint64_t kDMAChannelStartOfUnusedIdx = 5;        // 0x28 - 0x3F

  DmaStreamManager();

  absl::Status CheckAddressAlignment(channel_addr_t address,
                                     uint64_t bytes_count);
  uint64_t IRQBitAccess(const std::vector<bool>& bits, uint64_t bytes_count,
                        uint64_t byte_offset, uint64_t mask) const;
  uint64_t CalculateOffsetToFirstDMAChannel() const;

  absl::StatusOr<uint64_t> ReadNumberOfDMAChannels(uint64_t byte_offset,
                                                   uint64_t mask) const;
  absl::StatusOr<uint64_t> ReadOffsetToFirstDMAChannel(uint64_t byte_offset,
                                                       uint64_t mask) const;
  absl::StatusOr<uint64_t> ReadDMAChannelIRQMask(uint64_t bytes_count,
                                                 uint64_t byte_offset,
                                                 uint64_t mask) const;
  absl::StatusOr<uint64_t> ReadDMAChannelIRQ(uint64_t bytes_count,
                                             uint64_t byte_offset,
                                             uint64_t mask) const;
  absl::StatusOr<uint64_t> ReadInternal(channel_addr_t address,
                                        uint64_t log_byte_count);

  absl::Status WriteNumberOfDMAChannels();
  absl::Status WriteOffsetToFirstDMAChannel();
  absl::Status WriteDMAChannelIRQMask(uint64_t byte_offset,
                                      uint64_t bytes_count, uint64_t payload);
  absl::Status WriteDMAChannelIRQ();
  absl::Status WriteInternal(channel_addr_t address, uint64_t payload,
                             uint64_t log_byte_count);

  int64_t highest_channel_id_;
  // size in bytes but aligned to quad words (8 bytes)
  uint64_t offset_to_first_dma_channel_;
  uint64_t dma_channel_irq_mask_reg_size_;
  uint64_t dma_channel_irq_reg_size_;
  std::map<uint64_t, std::unique_ptr<StreamDmaChannel>> id_to_channel_;
  std::vector<bool> channel_irq_mask_;
  std::vector<bool> active_irq_mask_;
};

}  // namespace xls::simulation::generic

#endif  // XLS_SIMULATION_GENERIC_DMASTREAMMANAGER_H_
