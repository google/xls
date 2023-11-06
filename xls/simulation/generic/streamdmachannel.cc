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

template <typename T>
absl::Status StreamDmaChannel::UpdateWriteToEmulatorHelper(T payload) {
  constexpr uint64_t access_width_log = []() -> uint64_t {
    uint64_t log_width = 0;
    while ((sizeof(T) & (1ull << log_width)) == 0) {
      ++log_width;
    }
    return log_width;
  }();
  absl::Status data_sent_successfully = bus_master_port_->RequestWrite(
      transfer_base_address_ + transferred_length_, payload,
      AccessOfWidthPow(access_width_log));
  XLS_LOG(WARNING) << data_sent_successfully;
  if (!data_sent_successfully.ok()) {
    return data_sent_successfully;
  } else {
    transferred_length_ += sizeof(T);
    bytes_transferred_in_current_xfer_ += sizeof(T);
    return absl::OkStatus();
  }
}

absl::Status StreamDmaChannel::UpdateWriteToEmulator() {
  while ((endpoint_->IsReady() || bytes_in_current_xfer_ != 0) &&
         transferred_length_ < max_transfer_length_) {
    if (bytes_in_current_xfer_ == 0) {
      XLS_ASSIGN_OR_RETURN(current_xfer_, endpoint_->Read());
      bytes_transferred_in_current_xfer_ = 0;
      auto n_received = current_xfer_.data.size();
      XLS_CHECK_EQ(n_received % element_size_, 0);
      bytes_in_current_xfer_ =
          std::min(n_received, max_transfer_length_ - transferred_length_);
      if (bytes_in_current_xfer_ != n_received) {
        XLS_LOG(WARNING) << absl::StreamFormat(
            "Endpoint returned %u bytes, but only %u will be transferred to "
            "the memory to obey the max DMA transfer size set by the user. "
            "Data will be lost.",
            n_received, bytes_in_current_xfer_);
      }
    }
    if (dma_discard_) {
      // Discard mode
      transferred_length_ += bytes_in_current_xfer_;
      bytes_in_current_xfer_ = 0;
    } else {
      // Where possible, use 64-bit DMA accesses
      uint64_t start_position =
          bytes_transferred_in_current_xfer_ / sizeof(uint64_t);
      uint64_t quotient = bytes_in_current_xfer_ / sizeof(uint64_t);
      for (uint64_t i = start_position; i < quotient; ++i) {
        // u8[8]<->u64 cast has to be done in native endianness
        uint64_t payload;
        memcpy(&payload, current_xfer_.data.data() + i * sizeof(uint64_t),
               sizeof(uint64_t));
        XLS_RETURN_IF_ERROR(UpdateWriteToEmulatorHelper(payload));
      }

      // Handle remaining bytes using 8-bit accesses
      for (uint64_t i = sizeof(uint64_t) * quotient; i < bytes_in_current_xfer_;
           ++i) {
        XLS_RETURN_IF_ERROR(UpdateWriteToEmulatorHelper(current_xfer_.data[i]));
      }
    }

    // Transfer finished
    bytes_in_current_xfer_ = 0;
    // If last element was signalled by the endpoint, signal the end of the DMA
    // transfer
    if (current_xfer_.last) {
      dma_finished_ = true;
      irq_ |= kReceivedLastIrq;
      break;
    }
  }
  return absl::OkStatus();
}

template <typename T>
absl::StatusOr<T> StreamDmaChannel::UpdateReadFromEmulatorHelper() {
  constexpr uint64_t access_width_log = []() -> uint64_t {
    uint64_t log_width = 0;
    while ((sizeof(T) & (1ull << log_width)) == 0) {
      ++log_width;
    }
    return log_width;
  }();
  absl::StatusOr<T> data_read = bus_master_port_->RequestRead(
      transfer_base_address_ + transferred_length_,
      AccessOfWidthPow(access_width_log));
  if (data_read.ok()) {
    transferred_length_ += sizeof(T);
    bytes_transferred_in_current_xfer_ += sizeof(T);
  }
  return data_read;
}

absl::Status StreamDmaChannel::UpdateReadFromEmulator() {
  while (endpoint_->IsReady() && transferred_length_ < max_transfer_length_) {
    if (bytes_in_current_xfer_ == 0) {
      bytes_transferred_in_current_xfer_ = 0;
      uint64_t max_bytes_per_transfer =
          element_size_ * endpoint_->GetMaxElementsPerTransfer();
      bytes_in_current_xfer_ = std::min(
          max_bytes_per_transfer, max_transfer_length_ - transferred_length_);
    }

    // Where possible, use 64-bit DMA accesses
    uint64_t start_position =
        bytes_transferred_in_current_xfer_ / sizeof(uint64_t);
    uint64_t quotient = bytes_in_current_xfer_ / sizeof(uint64_t);
    for (uint64_t i = start_position; i < quotient; ++i) {
      XLS_ASSIGN_OR_RETURN(uint64_t payload,
                           UpdateReadFromEmulatorHelper<uint64_t>());
      // u8[8]<->u64 cast has to be done in native endianness
      current_xfer_.data.resize(current_xfer_.data.size() + sizeof(uint64_t));
      memcpy(current_xfer_.data.data() + sizeof(uint64_t) * i, &payload,
             sizeof(uint64_t));
    }
    // Handle remaining bytes using 8-bit accesses
    for (uint64_t i = sizeof(uint64_t) * quotient; i < bytes_in_current_xfer_;
         ++i) {
      XLS_ASSIGN_OR_RETURN(uint8_t payload,
                           UpdateReadFromEmulatorHelper<uint8_t>());
      current_xfer_.data.push_back(payload);
    }
    // Set 'last' flag
    if (transferred_length_ >= max_transfer_length_) {
      current_xfer_.last = true;
    }
    if (bytes_in_current_xfer_ != 0 &&
        bytes_in_current_xfer_ == bytes_transferred_in_current_xfer_) {
      XLS_RETURN_IF_ERROR(endpoint_->Write(std::move(current_xfer_)));
      bytes_in_current_xfer_ = 0;
    }
  }
  return absl::OkStatus();
}

absl::Status StreamDmaChannel::Update() {
  if (dma_run_ && !dma_finished_) {
    absl::Status res;
    if (endpoint_->IsReadStream()) {
      res = UpdateWriteToEmulator();
    } else {
      res = UpdateReadFromEmulator();
    }
    // Check Update's result.
    // Update should return either Ok, meaning that all data was transferred,
    // or Unavailable, when master port is stuck and can't ingest more data.
    // All other return codes are treated as errors.
    if (!(res.ok() || IsUnavailable(res))) {
      return res;
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
