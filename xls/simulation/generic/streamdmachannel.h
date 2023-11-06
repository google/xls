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

#ifndef XLS_SIMULATION_GENERIC_STREAMDMACHANNEL_H_
#define XLS_SIMULATION_GENERIC_STREAMDMACHANNEL_H_

#include <memory>
#include <utility>

#include "absl/status/status.h"
#include "xls/simulation/generic/common.h"
#include "xls/simulation/generic/iactive.h"
#include "xls/simulation/generic/idmaendpoint.h"
#include "xls/simulation/generic/iirq.h"
#include "xls/simulation/generic/imasterport.h"

namespace xls::simulation::generic {

class StreamDmaChannel : public IActive, public IIRQ {
 public:
  using channel_addr_t = uint64_t;

  // Control and status register
  // 63       7       6             5            4            3       2        0
  // ┌────────┬───────┬─────────────┬────────────┬────────────┬───────┬────────┐
  // │(unused)│IsReady│DMA discard N│IsReadStream│DMA finished│DMA run│IRQ mask│
  // └────────┴───────┴─────────────┴────────────┴────────────┴───────┴────────┘
  //
  // IRQ mask - (control) masks IRQ sources
  // DMA run - (control) enables/resets DMA
  // DMA finished - (status) set to 1 when transfered all data
  // IsReadStream - (status) informs whether underlying stream can be read
  // DMA discard - (control) when 0, performs the transfer through channel but
  // doesn't read the transfered data
  // IsReady - (status) set to 1 when the stream is ready to perform a single
  // transfer
  static const uint64_t kTransferFinishedIrq = 1;
  static const uint64_t kReceivedLastIrq = 2;
  static const uint64_t kCsrIrqMaskMask = 0b11;
  static const uint64_t kCsrDmaRunShift = 2;
  static const uint64_t kCsrDmaRunMask = 0b1;
  static const uint64_t kCsrDmaFinishedShift = 3;
  static const uint64_t kCsrIsReadStreamShift = 4;
  static const uint64_t kCsrDmaDiscardNShift = 5;
  static const uint64_t kCsrDmaDiscardNMask = 0b1;
  static const uint64_t kCsrIsReadyShift = 6;

  StreamDmaChannel(std::unique_ptr<IDmaEndpoint> endpoint,
                   IMasterPort* bus_master_port)
      : endpoint_(std::move(endpoint)),
        transfer_base_address_(0),
        max_transfer_length_(0),
        transferred_length_(0),
        irq_mask_(0),
        dma_run_(false),
        dma_finished_(false),
        dma_discard_(true),
        irq_(0),
        bus_master_port_(bus_master_port),
        bytes_in_current_xfer_(0) {
    this->element_size_ = endpoint_->GetElementSize();
  }

  StreamDmaChannel(StreamDmaChannel&& other) = default;
  StreamDmaChannel& operator=(StreamDmaChannel&& other_) = default;

  ~StreamDmaChannel() override = default;

  void SetTransferBaseAddress(channel_addr_t address);
  channel_addr_t GetTransferBaseAddress() {
    return this->transfer_base_address_;
  }

  void SetMaxTransferLength(uint64_t length);
  uint64_t GetMaxTransferLength() { return this->max_transfer_length_; }

  uint64_t GetTransferredLength() { return this->transferred_length_; }

  void ClearIRQReg(uint64_t reset_vector) { this->irq_ &= ~reset_vector; }
  uint64_t GetIRQReg() { return this->irq_; }

  void SetControlRegister(uint64_t update_state);
  uint64_t GetControlRegister();

  // IActive
  absl::Status Update() override;

  // IIRQ
  bool GetIRQ() override;
  absl::Status UpdateIRQ() override { return absl::OkStatus(); }

 protected:
  std::unique_ptr<IDmaEndpoint> endpoint_;
  virtual absl::Status UpdateReadFromEmulator();
  template <typename T>
  absl::StatusOr<T> UpdateReadFromEmulatorHelper();
  virtual absl::Status UpdateWriteToEmulator();
  template <typename T>
  absl::Status UpdateWriteToEmulatorHelper(T payload);

 private:
  friend class StreamDmaChannelPrivateAccess;

  channel_addr_t transfer_base_address_;
  uint64_t max_transfer_length_;
  uint64_t transferred_length_;

  // Control and status register fields
  uint64_t irq_mask_;
  bool dma_run_;
  bool dma_finished_;
  bool dma_discard_;

  uint64_t irq_;
  uint64_t element_size_;
  IMasterPort* bus_master_port_;

  uint64_t bytes_in_current_xfer_;
  uint64_t bytes_transferred_in_current_xfer_;
  IDmaEndpoint::Payload current_xfer_;
};

}  // namespace xls::simulation::generic

#endif  // XLS_SIMULATION_GENERIC_STREAMDMACHANNEL_H_
