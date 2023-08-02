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

#include "xls/simulation/generic/iregister_stub.h"

#include <cstdint>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xls/common/logging/logging.h"

namespace xls::simulation::generic {

absl::StatusOr<uint8_t> IRegisterStub::GetPayloadData8(uint64_t offset) const {
  XLS_LOG(INFO) << "IRegisterStub::GetPayloadData8()";
  return 0x12;
}

absl::StatusOr<uint16_t> IRegisterStub::GetPayloadData16(
    uint64_t offset) const {
  XLS_LOG(INFO) << "IRegisterStub::GetPayloadData16()";
  return 0x1234;
}

absl::StatusOr<uint32_t> IRegisterStub::GetPayloadData32(
    uint64_t offset) const {
  XLS_LOG(INFO) << "IRegisterStub::GetPayloadData32()";
  return 0x12345678;
}

absl::StatusOr<uint64_t> IRegisterStub::GetPayloadData64(
    uint64_t offset) const {
  XLS_LOG(INFO) << "IRegisterStub::GetPayloadData64()";
  return 0x1234567890ABCDEFll;
}

absl::Status IRegisterStub::SetPayloadData8(uint64_t offset, uint8_t data) {
  XLS_LOG(INFO) << "IRegisterStub::SetPayloadData8()";
  return absl::OkStatus();
}

absl::Status IRegisterStub::SetPayloadData16(uint64_t offset, uint16_t data) {
  XLS_LOG(INFO) << "IRegisterStub::SetPayloadData16()";
  return absl::OkStatus();
}

absl::Status IRegisterStub::SetPayloadData32(uint64_t offset, uint32_t data) {
  XLS_LOG(INFO) << "IRegisterStub::SetPayloadData32()";
  return absl::OkStatus();
}

absl::Status IRegisterStub::SetPayloadData64(uint64_t offset, uint64_t data) {
  XLS_LOG(INFO) << "IRegisterStub::SetPayloadData64()";
  return absl::OkStatus();
}

uint64_t IRegisterStub::GetChannelWidth() const {
  XLS_LOG(INFO) << "IRegisterStub::GetChannelWidth()";
  return 17;
}

}  // namespace xls::simulation::generic
