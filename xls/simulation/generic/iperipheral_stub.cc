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

#include "xls/simulation/generic/iperipheral_stub.h"

#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "xls/common/logging/logging.h"

namespace xls::simulation::generic {

IPeripheralStub::IPeripheralStub() {
  XLS_LOG(INFO) << "IPeripheralStub::IPeripheralStub()";
}

absl::Status IPeripheralStub::CheckRequest(uint64_t addr, AccessWidth width) {
  XLS_LOG(INFO) << absl::StreamFormat(
      "IPeripheralStub::CheckRequest(addr=%lu, width=%d)", addr, width);
  return absl::OkStatus();
}

absl::StatusOr<uint64_t> IPeripheralStub::HandleRead(uint64_t addr,
                                                     AccessWidth width) {
  XLS_LOG(INFO) << absl::StreamFormat(
      "IPeripheralStub::HandleRead(addr=%lu, width=%d), returns 0", addr,
      width);
  return 0;
}

absl::Status IPeripheralStub::HandleWrite(uint64_t addr, AccessWidth width,
                                          uint64_t payload) {
  XLS_LOG(INFO) << absl::StreamFormat(
      "IPeripheralStub::HandleWrite(addr=%lu, width=%d, payload=%lu)", addr,
      width, payload);
  return absl::OkStatus();
}

absl::StatusOr<IRQEnum> IPeripheralStub::HandleIRQ() {
  XLS_LOG(INFO) << absl::StreamFormat("IPeripheralStub::HandleIRQ()");
  return IRQEnum::NoChange;
}

absl::Status IPeripheralStub::HandleTick() {
  XLS_LOG(INFO) << absl::StreamFormat("IPeripheralStub::HandleTick()");
  return absl::OkStatus();
}

absl::Status IPeripheralStub::Reset() {
  XLS_LOG(INFO) << "IPeripheralStub::Reset()";
  return absl::OkStatus();
}

}  // namespace xls::simulation::generic
