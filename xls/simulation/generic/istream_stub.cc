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

#include "xls/simulation/generic/istream_stub.h"

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xls/common/logging/logging.h"

namespace xls::simulation::generic {

// IStreamStub

bool IStreamStub::IsReadStream() const {
  XLS_LOG(INFO) << "IStreamStub::IsReadStream()";
  return true;
}

bool IStreamStub::IsReady() const {
  XLS_LOG(INFO) << "IStreamStub::IsReady()";
  return true;
}

absl::Status IStreamStub::Transfer() {
  XLS_LOG(INFO) << "IStreamStub::Transfer()";
  return absl::OkStatus();
}

absl::StatusOr<uint8_t> IStreamStub::GetPayloadData8(uint64_t offset) const {
  XLS_LOG(INFO) << "IStreamStub::GetPayloadData8()";
  return 0x12;
}

absl::StatusOr<uint16_t> IStreamStub::GetPayloadData16(uint64_t offset) const {
  XLS_LOG(INFO) << "IStreamStub::GetPayloadData16()";
  return 0x1234;
}

absl::StatusOr<uint32_t> IStreamStub::GetPayloadData32(uint64_t offset) const {
  XLS_LOG(INFO) << "IStreamStub::GetPayloadData32()";
  return 0x12345678;
}

absl::StatusOr<uint64_t> IStreamStub::GetPayloadData64(uint64_t offset) const {
  XLS_LOG(INFO) << "IStreamStub::GetPayloadData64()";
  return 0x1234567890ABCDEFll;
}

absl::Status IStreamStub::SetPayloadData8(uint64_t offset, uint8_t data) {
  XLS_LOG(INFO) << "IStreamStub::SetPayloadData8()";
  return absl::OkStatus();
}

absl::Status IStreamStub::SetPayloadData16(uint64_t offset, uint16_t data) {
  XLS_LOG(INFO) << "IStreamStub::SetPayloadData16()";
  return absl::OkStatus();
}

absl::Status IStreamStub::SetPayloadData32(uint64_t offset, uint32_t data) {
  XLS_LOG(INFO) << "IStreamStub::SetPayloadData32()";
  return absl::OkStatus();
}

absl::Status IStreamStub::SetPayloadData64(uint64_t offset, uint64_t data) {
  XLS_LOG(INFO) << "IStreamStub::SetPayloadData64()";
  return absl::OkStatus();
}

uint64_t IStreamStub::GetChannelWidth() const {
  XLS_LOG(INFO) << "IStreamStub::GetChannelWidth()";
  return 17;
}

}  // namespace xls::simulation::generic
