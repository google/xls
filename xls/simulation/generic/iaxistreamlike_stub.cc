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

#include "xls/simulation/generic/iaxistreamlike_stub.h"

#include <cstdint>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xls/common/logging/logging.h"

namespace xls::simulation::generic {

uint64_t IAxiStreamLikeStub::GetNumSymbols() const {
  XLS_LOG(INFO) << "IAxiStreamLikeStub::GetNumSymbols()";
  return 2;
}

uint64_t IAxiStreamLikeStub::GetSymbolWidth() const {
  XLS_LOG(INFO) << "IAxiStreamLikeStub::GetSymbolWidth()";
  return 14;
}

uint64_t IAxiStreamLikeStub::GetSymbolSize() const {
  XLS_LOG(INFO) << "IAxiStreamLikeStub::GetSymbolSize()";
  return 2;
}

uint64_t IAxiStreamLikeStub::GetChannelWidth() const {
  XLS_LOG(INFO) << "IAxiStreamLikeStub::GetChannelWidth()";
  return 32;
}

void IAxiStreamLikeStub::SetDataValid(std::vector<bool> dataValid) {
  XLS_LOG(INFO) << "IAxiStreamLikeStub::SetDataValid()";
}

std::vector<bool> IAxiStreamLikeStub::GetDataValid() const {
  XLS_LOG(INFO) << "IAxiStreamLikeStub::GetDataValid()";
  return {true, true};
}

void IAxiStreamLikeStub::SetLast(bool last) {
  XLS_LOG(INFO) << "IAxiStreamLikeStub::SetLast()";
}

bool IAxiStreamLikeStub::GetLast() const {
  XLS_LOG(INFO) << "IAxiStreamLikeStub::GetLast()";
  return true;
}

bool IAxiStreamLikeStub::IsReadStream() const {
  XLS_LOG(INFO) << "IAxiStreamLikeStub::IsReadStream()";
  return true;
}

bool IAxiStreamLikeStub::IsReady() const {
  XLS_LOG(INFO) << "IAxiStreamLikeStub::IsReady()";
  return true;
}

absl::Status IAxiStreamLikeStub::Transfer() {
  XLS_LOG(INFO) << "IAxiStreamLikeStub::Transfer()";
  return absl::OkStatus();
}

absl::StatusOr<uint8_t> IAxiStreamLikeStub::GetPayloadData8(
    uint64_t offset) const {
  XLS_LOG(INFO) << "IAxiStreamLikeStub::GetPayloadData8()";
  return 0x12;
}

absl::StatusOr<uint16_t> IAxiStreamLikeStub::GetPayloadData16(
    uint64_t offset) const {
  XLS_LOG(INFO) << "IAxiStreamLikeStub::GetPayloadData16()";
  return 0x1234;
}

absl::StatusOr<uint32_t> IAxiStreamLikeStub::GetPayloadData32(
    uint64_t offset) const {
  XLS_LOG(INFO) << "IAxiStreamLikeStub::GetPayloadData32()";
  return 0x12345678;
}

absl::StatusOr<uint64_t> IAxiStreamLikeStub::GetPayloadData64(
    uint64_t offset) const {
  XLS_LOG(INFO) << "IAxiStreamLikeStub::GetPayloadData64()";
  return 0x1234567890ABCDEFll;
}

absl::Status IAxiStreamLikeStub::SetPayloadData8(uint64_t offset,
                                                 uint8_t data) {
  XLS_LOG(INFO) << "IAxiStreamLikeStub::SetPayloadData8()";
  return absl::OkStatus();
}

absl::Status IAxiStreamLikeStub::SetPayloadData16(uint64_t offset,
                                                  uint16_t data) {
  XLS_LOG(INFO) << "IAxiStreamLikeStub::SetPayloadData16()";
  return absl::OkStatus();
}

absl::Status IAxiStreamLikeStub::SetPayloadData32(uint64_t offset,
                                                  uint32_t data) {
  XLS_LOG(INFO) << "IAxiStreamLikeStub::SetPayloadData32()";
  return absl::OkStatus();
}

absl::Status IAxiStreamLikeStub::SetPayloadData64(uint64_t offset,
                                                  uint64_t data) {
  XLS_LOG(INFO) << "IAxiStreamLikeStub::SetPayloadData64()";
  return absl::OkStatus();
}

}  // namespace xls::simulation::generic
