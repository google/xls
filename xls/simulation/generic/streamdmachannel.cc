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

#include "xls/simulation/generic/streamdmachannel.h"

#include <algorithm>

#include "absl/strings/str_format.h"
#include "xls/common/bits_util.h"
#include "xls/common/logging/logging.h"
#include "xls/common/status/status_macros.h"

namespace xls::simulation::generic {

void StreamDmaChannel::SetTransferBaseAddress(channel_addr_t address) {
  if (dma_run_) {
    XLS_LOG(WARNING) << "Base address can't be modified while DMA is running";
    return;
  }
  transfer_base_address_ = address;
}

void StreamDmaChannel::SetMaxTransferLength(uint64_t length) {
  if (dma_run_) {
    XLS_LOG(WARNING)
        << "Max transfer length can't be modified while DMA is running";
    return;
  }
  if (length % element_size_ != 0) {
    XLS_LOG(WARNING) << "Max transfer length will be clipped to: "
                     << length - (length % element_size_);
  }
  length -= length % element_size_;
  max_transfer_length_ = length;
}

void StreamDmaChannel::SetControlRegister(uint64_t update_state) {
  // First 2 bits are used for IRQ masking
  irq_mask_ = update_state & kCsrIrqMaskMask;
  // Third bit enables/resets DMA
  bool new_dma_run =
      static_cast<bool>((update_state >> kCsrDmaRunShift) & kCsrDmaRunMask);
  if (!new_dma_run) {
    transferred_length_ = 0;
    dma_finished_ = false;
  }
  dma_run_ = new_dma_run;
  // Sixth bit controls if DMA should discard read data
  bool new_dma_discard_n = static_cast<bool>(
      (update_state >> kCsrDmaDiscardNShift) & kCsrDmaDiscardNMask);
  dma_discard_ = !new_dma_discard_n;
}

uint64_t StreamDmaChannel::GetControlRegister() {
  uint64_t intermediate_cr_ = irq_mask_ & kCsrIrqMaskMask;
  intermediate_cr_ |= static_cast<uint64_t>(dma_run_) << kCsrDmaRunShift;
  intermediate_cr_ |= static_cast<uint64_t>(dma_finished_)
                      << kCsrDmaFinishedShift;
  intermediate_cr_ |= static_cast<uint64_t>(endpoint_->IsReadStream())
                      << kCsrIsReadStreamShift;
  intermediate_cr_ |= static_cast<uint64_t>(!dma_discard_)
                      << kCsrDmaDiscardNShift;
  intermediate_cr_ |= static_cast<uint64_t>(endpoint_->IsReady())
                      << kCsrIsReadyShift;
  return intermediate_cr_;
}

bool StreamDmaChannel::GetIRQ() { return (irq_mask_ & irq_) != 0u; }

static AccessWidth AccessOfWidthPow(uint64_t pow2) {
  switch (pow2) {
    case 0:
      return AccessWidth::BYTE;
    case 1:
      return AccessWidth::WORD;
    case 2:
      return AccessWidth::DWORD;
    case 3:
      return AccessWidth::QWORD;
  }
  ABSL_ASSERT(false);
  // Unreachable
  return AccessWidth::BYTE;
}

absl::Status StreamDmaChannel::UpdateWriteToEmulator() {
  while (endpoint_->IsReady() && transferred_length_ < max_transfer_length_) {
    XLS_ASSIGN_OR_RETURN(auto xfer, endpoint_->Read());
    auto n_received = xfer.data.size();
    XLS_CHECK_EQ(n_received % element_size_, 0);
    uint64_t n_bytes_to_transfer =
        std::min(n_received, max_transfer_length_ - transferred_length_);
    if (dma_discard_) {
      // Discard mode
      transferred_length_ += n_bytes_to_transfer;
    } else {
      if (n_bytes_to_transfer != n_received) {
        XLS_LOG(WARNING) << absl::StreamFormat(
            "Endpoint returned %u bytes, but only %u will be transferred to "
            "the memory to obey the max DMA transfer size set by the user. "
            "Data will be lost.",
            n_received, n_bytes_to_transfer);
      }
      // Where possible, use 64-bit DMA accesses
      uint64_t quotient = n_bytes_to_transfer / sizeof(uint64_t);
      for (uint64_t i = 0; i < quotient; ++i) {
        // u8[8]<->u64 cast has to be done in native endianness
        uint64_t payload;
        memcpy(&payload, xfer.data.data() + i * sizeof(uint64_t),
               sizeof(uint64_t));
        bus_master_port_->RequestWrite(
            transfer_base_address_ + transferred_length_, payload,
            AccessWidth::QWORD);
        transferred_length_ += sizeof(uint64_t);
      }

      // Handle remaining bytes using 8-bit accesses
      for (uint64_t i = sizeof(uint64_t) * quotient; i < n_bytes_to_transfer;
           ++i) {
        bus_master_port_->RequestWrite(
            transfer_base_address_ + transferred_length_, xfer.data[i],
            AccessWidth::BYTE);
        transferred_length_ += sizeof(uint8_t);
      }
    }
    // If last element was signalled by the endpoint, signal the end of the DMA
    // transfer
    if (xfer.last) {
      dma_finished_ = true;
      irq_ |= kReceivedLastIrq;
      break;
    }
  }
  return absl::OkStatus();
}

absl::Status StreamDmaChannel::UpdateReadFromEmulator() {
  while (endpoint_->IsReady() && transferred_length_ < max_transfer_length_) {
    uint64_t max_bytes_per_transfer =
        element_size_ * endpoint_->GetMaxElementsPerTransfer();
    uint64_t n_bytes_to_transfer = std::min(
        max_bytes_per_transfer, max_transfer_length_ - transferred_length_);

    IDmaEndpoint::Payload xfer{};

    // Where possible, use 64-bit DMA accesses
    uint64_t quotient = n_bytes_to_transfer / sizeof(uint64_t);
    for (uint64_t i = 0; i < quotient; ++i) {
      auto resp = bus_master_port_->RequestRead(
          transfer_base_address_ + transferred_length_, AccessWidth::QWORD);
      uint64_t payload = resp.value();
      // u8[8]<->u64 cast has to be done in native endianness
      xfer.data.resize(xfer.data.size() + sizeof(uint64_t));
      memcpy(xfer.data.data() + sizeof(uint64_t) * i, &payload,
             sizeof(uint64_t));
      transferred_length_ += sizeof(uint64_t);
    }
    // Handle remaining bytes using 8-bit accesses
    for (uint64_t i = sizeof(uint64_t) * quotient; i < n_bytes_to_transfer;
         ++i) {
      auto resp = bus_master_port_->RequestRead(
          transfer_base_address_ + transferred_length_, AccessWidth::BYTE);
      uint8_t payload = resp.value();
      xfer.data.push_back(payload);
      transferred_length_ += sizeof(uint8_t);
    }
    // Set 'last' flag
    if (transferred_length_ >= max_transfer_length_) {
      xfer.last = true;
    }
    XLS_RETURN_IF_ERROR(endpoint_->Write(std::move(xfer)));
  }
  return absl::OkStatus();
}

absl::Status StreamDmaChannel::Update() {
  if (dma_run_ && !dma_finished_) {
    if (endpoint_->IsReadStream()) {
      XLS_RETURN_IF_ERROR(UpdateWriteToEmulator());
    } else {
      XLS_RETURN_IF_ERROR(UpdateReadFromEmulator());
    }
    // If all the requested data has been transferred, or the transfer has been
    // prematurely marked as complete by the Update function, signal the end of
    // the transfer
    if (transferred_length_ >= max_transfer_length_ || dma_finished_) {
      dma_finished_ = true;
      irq_ |= kTransferFinishedIrq;
    }
  }
  return absl::OkStatus();
}

}  // namespace xls::simulation::generic
