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

#include "xls/simulation/renode/renode_masterport.h"

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "xls/simulation/generic/common.h"
#include "xls/simulation/renode/renode_protocol.h"
#include "xls/simulation/renode/sharedlibconnection.h"

namespace xls::simulation::renode {

static ::renode::ProtocolMessage FormWriteRequest(uint64_t address,
                                                  uint64_t payload,
                                                  AccessWidth width) {
  int actionId;
  switch (width) {
    case AccessWidth::BYTE:
      actionId = ::renode::Action::pushByte;
      break;
    case AccessWidth::WORD:
      actionId = ::renode::Action::pushWord;
      break;
    case AccessWidth::DWORD:
      actionId = ::renode::Action::pushDoubleWord;
      break;
    case AccessWidth::QWORD:
      actionId = ::renode::Action::pushQuadWord;
      break;
    default:
      XLS_LOG(FATAL) << absl::StreamFormat("Unhandled AccessWidth: %d", width);
      break;
  }

  return ::renode::ProtocolMessage(actionId, address, payload);
}

absl::Status RenodeMasterPort::RequestWrite(uint64_t address, uint64_t value,
                                            AccessWidth type) {
  absl::Status op_status = absl::OkStatus();
  auto req = FormWriteRequest(address, value, type);

  XLS_CHECK_OK(SharedLibConnection::Instance().SendRequest(req));
  return absl::OkStatus();
  // Renode doesn't send respons to write request
}

static ::renode::ProtocolMessage FormReadRequest(uint64_t address,
                                                 AccessWidth width) {
  int actionId;

  switch (width) {
    case AccessWidth::BYTE:
      actionId = ::renode::Action::getByte;
      break;
    case AccessWidth::WORD:
      actionId = ::renode::Action::getWord;
      break;
    case AccessWidth::DWORD:
      actionId = ::renode::Action::getDoubleWord;
      break;
    case AccessWidth::QWORD:
      actionId = ::renode::Action::getQuadWord;
      break;
    default:
      XLS_LOG(FATAL) << absl::StreamFormat("Unhandled AccessWidth: %d", width);
      break;
  }

  return ::renode::ProtocolMessage(actionId, address, 0);
}

static absl::Status CheckResponse(::renode::ProtocolMessage resp,
                                  ::renode::Action expected) {
  if (resp.actionId == ::renode::Action::error)
    return absl::InternalError("Read request unsuccessful");

  if (resp.actionId != expected)
    return absl::UnimplementedError(
        absl::StrFormat("Unexpected response: %d", resp.actionId));

  return absl::OkStatus();
}

absl::StatusOr<uint64_t> RenodeMasterPort::RequestRead(uint64_t address,
                                                       AccessWidth width) {
  absl::Status op_status = absl::OkStatus();
  auto req = FormReadRequest(address, width);

  XLS_CHECK_OK(SharedLibConnection::Instance().SendRequest(req));
  auto resp = SharedLibConnection::Instance().ReceiveResponse();
  XLS_CHECK_OK(resp.status());
  // Renode shouldn't send error
  XLS_CHECK_OK(CheckResponse(resp.value(), ::renode::Action::writeRequest));
  return resp.value().value;
}

}  // namespace xls::simulation::renode
