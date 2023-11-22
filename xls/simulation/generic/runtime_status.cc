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

#include "xls/simulation/generic/runtime_status.h"

#include <cstdint>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xls/common/logging/logging.h"

namespace xls::simulation::generic {
absl::StatusOr<uint64_t> RuntimeStatus::GetPayloadData64(
    uint64_t offset) const {
  switch (offset) {
    case 0x0:
      return this->deadlock_;
    default:
      XLS_LOG(WARNING) << "RuntimeStatus::GetPayloadData64: "
                          "Selected register does not exist";
      return 0;
  }
}

absl::Status RuntimeStatus::SetPayloadData64(uint64_t offset, uint64_t data) {
  return absl::UnavailableError("All registers are read-only");
}

absl::StatusOr<uint8_t> RuntimeStatus::GetPayloadData8(uint64_t offset) const {
  return GetPayloadData64(offset);
}

absl::StatusOr<uint16_t> RuntimeStatus::GetPayloadData16(
    uint64_t offset) const {
  return GetPayloadData64(offset);
}

absl::StatusOr<uint32_t> RuntimeStatus::GetPayloadData32(
    uint64_t offset) const {
  return GetPayloadData64(offset);
}

absl::Status RuntimeStatus::SetPayloadData8(uint64_t offset, uint8_t data) {
  return this->SetPayloadData64(offset, data);
}
absl::Status RuntimeStatus::SetPayloadData16(uint64_t offset, uint16_t data) {
  return this->SetPayloadData64(offset, data);
}
absl::Status RuntimeStatus::SetPayloadData32(uint64_t offset, uint32_t data) {
  return this->SetPayloadData64(offset, data);
}

uint64_t RuntimeStatus::GetChannelWidth() const { return 1; }

}  // namespace xls::simulation::generic
