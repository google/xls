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

#include "xls/simulation/renode/sharedlibconnection.h"

#include <cstdint>
#include <string>
#include <string_view>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xls/common/logging/log_flags.h"
#include "xls/common/logging/logging.h"
#include "xls/common/status/ret_check.h"
#include "xls/simulation/generic/iperipheral.h"
#include "xls/simulation/generic/peripheral_factory.h"
#include "xls/simulation/renode/renode_logger.h"
#include "xls/simulation/renode/renode_protocol.h"
#include "xls/simulation/renode/renode_protocol_native.h"

namespace xls::simulation::renode {
namespace generic = xls::simulation::generic;

static void retarget_xls_logging(generic::IConnection& renode) {
  // Set and configure XLS logging through Renode

  // Renode logger registration can create logs,
  // but they aren't yet passes to Renode.
  // Redirect them to stderr during registration,
  // and disable redirect after logger is registered.
  absl::SetFlag(&FLAGS_logtostderr, true);
  XLS_CHECK_OK(RenodeLogger::RegisterRenodeLogger(renode));
  absl::SetFlag(&FLAGS_logtostderr, false);
  absl::SetFlag(&FLAGS_stderrthreshold,
                static_cast<int>(absl::LogSeverityAtLeast::kInfinity));
}

static void restore_xls_logging() {
  // Renode logger removal can create logs,
  // but they are no longer passed to Renode.
  // Redirect them to stderr during removal,
  // and disable redirect after logger is removed.
  absl::SetFlag(&FLAGS_logtostderr, true);
  XLS_CHECK_OK(RenodeLogger::UnRegisterRenodeLogger());
  absl::SetFlag(&FLAGS_logtostderr, false);
}

static bool retarget_logs_to_renode_ = true;

/* static */ void SharedLibConnection::RetargetLogsToRenode(bool on) {
  retarget_logs_to_renode_ = on;
}

SharedLibConnection::SharedLibConnection() {
  if (retarget_logs_to_renode_) {
    retarget_xls_logging(*this);
  }
}

SharedLibConnection::~SharedLibConnection() {
  if (retarget_logs_to_renode_) {
    restore_xls_logging();
  }
}

SharedLibConnection& SharedLibConnection::Instance() {
  static SharedLibConnection inst{};
  return inst;
}

absl::StatusOr<::renode::ProtocolMessage>
SharedLibConnection::ReceiveResponse() {
  ::renode::ProtocolMessage msg{};
  XLS_RETURN_IF_ERROR(RenodeHandleReceive(&msg));
  return msg;
}

absl::Status SharedLibConnection::SendResponse(::renode::ProtocolMessage resp) {
  XLS_RETURN_IF_ERROR(RenodeHandleMainMessage(&resp));
  return absl::OkStatus();
}

absl::Status SharedLibConnection::SendRequest(::renode::ProtocolMessage req) {
  XLS_RETURN_IF_ERROR(RenodeHandleSenderMessage(&req));
  return absl::OkStatus();
}

absl::Status SharedLibConnection::Log(absl::LogSeverity level,
                                      std::string_view msg) {
  // Map from XLS log level onto Renode log level
  ::renode::LogLevel renode_lvl;

  switch (level) {
    case absl::LogSeverity::kInfo:
      renode_lvl = ::renode::LOG_LEVEL_INFO;
      break;
    case absl::LogSeverity::kWarning:
      renode_lvl = ::renode::LOG_LEVEL_WARNING;
      break;
    case absl::LogSeverity::kError:
    case absl::LogSeverity::kFatal:
    default:
      renode_lvl = ::renode::LOG_LEVEL_ERROR;
      break;
  }

  std::string str{msg};
  // Renode logMessage protocol specifies two messages:
  // 1st - send cstring message (addr=str.size(), data=str.c_str())
  // 2nd - send log level (addr=0, data=renode_lvl)
  ::renode::ProtocolMessage packet;
  packet = ::renode::ProtocolMessage{::renode::logMessage, str.size(),
                                     reinterpret_cast<uint64_t>(str.c_str())};
  XLS_RETURN_IF_ERROR(RenodeHandleSenderMessage(&packet));
  packet = ::renode::ProtocolMessage{::renode::logMessage, 0,
                                     static_cast<uint64_t>(renode_lvl)};
  XLS_RETURN_IF_ERROR(RenodeHandleSenderMessage(&packet));
  return absl::OkStatus();
}

generic::IMasterPort* SharedLibConnection::GetMasterPort() {
  auto new_port = std::make_unique<RenodeMasterPort>();
  ports_.push_back(std::move(new_port));
  return ports_.back().get();
}

absl::Status SharedLibConnection::RenodeHandleMainMessage(
    ::renode::ProtocolMessage* msg) {
  if (!renode_handle_main_message_fn_) {
    return absl::InternalError(
        "renode_handle_main_message_fn_ is null. Renode didn't call attacher?");
  }
  renode_handle_main_message_fn_(msg);
  return absl::OkStatus();
}

absl::Status SharedLibConnection::RenodeHandleSenderMessage(
    ::renode::ProtocolMessage* msg) {
  if (!renode_handle_sender_message_fn_) {
    return absl::InternalError(
        "renode_handle_sender_message_fn_ is null. Renode didn't call "
        "attacher?");
  }
  renode_handle_sender_message_fn_(msg);
  return absl::OkStatus();
}

absl::Status SharedLibConnection::RenodeHandleReceive(
    ::renode::ProtocolMessage* msg) {
  if (!renode_receive_fn_) {
    return absl::InternalError(
        "renode_receive_fn_ is null. Renode didn't call attacher?");
  }
  renode_receive_fn_(msg);
  return absl::OkStatus();
}

void SharedLibConnection::CApiInitializeContext(const char* context) {
  context_ = std::string{context};
}

void SharedLibConnection::CApiInitializeNative() {
  // Instantiate peripheral using factory
  peripheral_ = generic::PeripheralFactory::Instance().Make(*this, context_);
}

absl::Status SharedLibConnection::HandleTick() {
  XLS_RETURN_IF_ERROR(peripheral_->HandleTick());
  XLS_ASSIGN_OR_RETURN(auto irq, peripheral_->HandleIRQ());

  if (irq == IRQEnum::SetIRQ)
    XLS_CHECK_OK(SendRequest(
        ::renode::ProtocolMessage{::renode::Action::interrupt, 0, 1}));
  else if (irq == IRQEnum::UnsetIRQ)
    XLS_CHECK_OK(SendRequest(
        ::renode::ProtocolMessage{::renode::Action::interrupt, 0, 0}));
  XLS_CHECK_OK(SendRequest(
      ::renode::ProtocolMessage{::renode::Action::tickClock, 0, 0}));
  return absl::OkStatus();
}

static AccessWidth RenodeActionToAccessWidth(::renode::Action access_action) {
  // All read and write actions have continuous IDs and are sorted by type
  // (r/w) and then by width. So, we can take ID's their offset relative to
  // the first action and do a mod 4 (equivalent to & 0x3) to get the width
  // identifier.
  switch (access_action) {
    case ::renode::Action::readRequestByte:
    case ::renode::Action::writeRequestByte:
      return AccessWidth::BYTE;
    case ::renode::Action::readRequestWord:
    case ::renode::Action::writeRequestWord:
      return AccessWidth::WORD;
    case ::renode::Action::readRequestDoubleWord:
    case ::renode::Action::writeRequestDoubleWord:
      return AccessWidth::DWORD;
    case ::renode::Action::readRequestQuadWord:
    case ::renode::Action::writeRequestQuadWord:
      return AccessWidth::QWORD;
    default:
      XLS_LOG(ERROR) << "Unhandled Renode access action!";
      XLS_DCHECK(false);
  }
  return AccessWidth::BYTE;
}

absl::Status SharedLibConnection::HandleRead(::renode::ProtocolMessage* req) {
  AccessWidth access =
      RenodeActionToAccessWidth(static_cast<::renode::Action>(req->actionId));
  XLS_RETURN_IF_ERROR(peripheral_->CheckRequest(req->addr, access));
  auto payload = peripheral_->HandleRead(req->addr, access);

  XLS_ASSIGN_OR_RETURN(auto irq, peripheral_->HandleIRQ());
  if (irq == IRQEnum::SetIRQ)
    XLS_CHECK_OK(SendRequest(
        ::renode::ProtocolMessage{::renode::Action::interrupt, 0, 1}));
  else if (irq == IRQEnum::UnsetIRQ)
    XLS_CHECK_OK(SendRequest(
        ::renode::ProtocolMessage{::renode::Action::interrupt, 0, 0}));

  if (payload.ok()) {
    XLS_CHECK_OK(SendResponse(
        ::renode::ProtocolMessage{::renode::Action::ok, 0, *payload}));
  }

  return payload.status();
}

absl::Status SharedLibConnection::HandleWrite(::renode::ProtocolMessage* req) {
  AccessWidth access =
      RenodeActionToAccessWidth(static_cast<::renode::Action>(req->actionId));
  XLS_RETURN_IF_ERROR(peripheral_->CheckRequest(req->addr, access));
  auto status = peripheral_->HandleWrite(req->addr, access, req->value);

  XLS_ASSIGN_OR_RETURN(auto irq, peripheral_->HandleIRQ());
  if (irq == IRQEnum::SetIRQ)
    XLS_CHECK_OK(SendRequest(
        ::renode::ProtocolMessage{::renode::Action::interrupt, 0, 1}));
  else if (irq == IRQEnum::UnsetIRQ)
    XLS_CHECK_OK(SendRequest(
        ::renode::ProtocolMessage{::renode::Action::interrupt, 0, 0}));

  if (status.ok()) {
    XLS_CHECK_OK(
        SendResponse(::renode::ProtocolMessage{::renode::Action::ok, 0, 0}));
  }

  return status;
}

void SharedLibConnection::CApiHandleRequest(
    ::renode::ProtocolMessage* request) {
  XLS_CHECK(peripheral_)
      << "HandleRequest() called on uninitialized peripheral";
  absl::Status op_status = absl::OkStatus();

  switch (request->actionId) {
    case ::renode::Action::tickClock:
      op_status = HandleTick();
      break;
    case ::renode::Action::resetPeripheral:
      op_status = peripheral_->Reset();
      break;
    case ::renode::Action::writeRequestByte:
    case ::renode::Action::writeRequestWord:
    case ::renode::Action::writeRequestDoubleWord:
    case ::renode::Action::writeRequestQuadWord:
      op_status = HandleWrite(request);
      break;
    case ::renode::Action::readRequestByte:
    case ::renode::Action::readRequestWord:
    case ::renode::Action::readRequestDoubleWord:
    case ::renode::Action::readRequestQuadWord:
      op_status = HandleRead(request);
      break;
    default:
      op_status =
          Log(absl::LogSeverity::kError,
              absl::StrFormat("Unsupported action: %d", request->actionId));
      if (!op_status.ok()) {
        break;
      }
      op_status = SendResponse(
          ::renode::ProtocolMessage{::renode::Action::error, 0, 0});
      break;
  }

  if (!op_status.ok()) {
    XLS_LOG(ERROR) << "Operation failed: " << op_status.ToString();
    XLS_CHECK_OK(
        SendResponse(::renode::ProtocolMessage{::renode::Action::error, 0, 0}));
  }
}

void SharedLibConnection::CApiResetPeripheral() {
  XLS_CHECK(peripheral_)
      << "ResetPeripheral() called on uninitialized peripheral";
  XLS_CHECK_OK(peripheral_->Reset());
}

void SharedLibConnection::CApiAttachHandleMainMessage(
    ::renode::ProtocolMessageHandler* handle_main_message_fn) {
  renode_handle_main_message_fn_ = handle_main_message_fn;
}

void SharedLibConnection::CApiAttachHandleSenderMessage(
    ::renode::ProtocolMessageHandler* handle_sender_message_fn) {
  renode_handle_sender_message_fn_ = handle_sender_message_fn;
}

void SharedLibConnection::CApiAttachReceive(
    ::renode::ProtocolMessageHandler* receive_fn) {
  renode_receive_fn_ = receive_fn;
}

}  // namespace xls::simulation::renode

// Implementation of 'extern "C"' DLL entry points called directly by Renode
namespace renode {

using xls::simulation::renode::SharedLibConnection;

// Implementation of Renode C API
extern "C" void initialize_native() {
  SharedLibConnection::Instance().CApiInitializeNative();
}

extern "C" void initialize_context(const char* context) {
  SharedLibConnection::Instance().CApiInitializeContext(context);
}

extern "C" void handle_request(ProtocolMessage* request) {
  SharedLibConnection::Instance().CApiHandleRequest(request);
}

extern "C" void reset_peripheral() {
  SharedLibConnection::Instance().CApiResetPeripheral();
}

extern "C" void renode_external_attach__ActionIntPtr__HandleMainMessage(
    ProtocolMessageHandler fn) {
  SharedLibConnection::Instance().CApiAttachHandleMainMessage(fn);
}

extern "C" void renode_external_attach__ActionIntPtr__HandleSenderMessage(
    ProtocolMessageHandler fn) {
  SharedLibConnection::Instance().CApiAttachHandleSenderMessage(fn);
}

extern "C" void renode_external_attach__ActionIntPtr__Receive(
    ProtocolMessageHandler fn) {
  SharedLibConnection::Instance().CApiAttachReceive(fn);
}

}  // namespace renode
