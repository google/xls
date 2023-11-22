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

#include <functional>
#include <memory>
#include <string>
#include <string_view>

#include "absl/strings/str_format.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "xls/common/logging/log_flags.h"
#include "xls/common/logging/logging.h"
#include "xls/common/status/matchers.h"
#include "xls/simulation/generic/iconnection.h"
#include "xls/simulation/generic/iperipheral_stub.h"
#include "xls/simulation/generic/peripheral_factory.h"
#include "xls/simulation/renode/renode_protocol.h"
#include "xls/simulation/renode/renode_protocol_native.h"

namespace xls::simulation::renode {

namespace {
using namespace xls::simulation::generic;

// Factory method that creates IPeripheralStub instead of what PeripheralFactory
// is normally supposed to create
std::unique_ptr<IPeripheral> make_peripheral_stub(IConnection& conn,
                                                  std::string_view context) {
  XLS_LOG(INFO) << absl::StreamFormat("make_peripheral_stub(%s)", context);
  return std::make_unique<IPeripheralStub>();
}

// Native->Renode callback stubs
void renode_stub_handlemainmessage(::renode::ProtocolMessage* msg) {
  XLS_LOG(INFO) << absl::StreamFormat(
      "renode_stub_handlemainmessage(action=%u, addr=%u, value=%u)",
      msg->actionId, msg->addr, msg->value);
}

void renode_stub_handlesendermessage(::renode::ProtocolMessage* msg) {
  XLS_LOG(INFO) << absl::StreamFormat(
      "renode_stub_handlesendermessage(action=%u, addr=%u, value=%u)",
      msg->actionId, msg->addr, msg->value);

  if (msg->actionId == ::renode::Action::logMessage) {
    if (!msg->addr) {
      // Contains log level
      XLS_LOG(INFO) << absl::StreamFormat(
          "renode_stub_handlesendermessage: log(level=%d)", msg->value);
    } else {
      // Contains pointer and length of log message
      std::string str{reinterpret_cast<const char*>(msg->value), msg->addr};
      XLS_LOG(INFO) << absl::StreamFormat(
          "renode_stub_handlesendermessage: log(msg=%s)", str);
    }
  }
}

void renode_stub_receive(::renode::ProtocolMessage* msg) {
  XLS_LOG(INFO) << "renode_stub_receive()";
  msg->actionId = ::renode::Action::handshake;
  msg->addr = 0x1234;
  msg->value = 0x5678;
}

class SharedLibConnectionTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    absl::SetFlag(&FLAGS_logtostderr, true);
    PeripheralFactory::Instance().OverrideFactoryMethod(make_peripheral_stub);
    SharedLibConnection::RetargetLogsToRenode(false);
  }
  void SetUp() override {
    ::testing::internal::CaptureStderr();

    ::renode::initialize_context("testctx");
    ::renode::initialize_native();
  }
};

TEST_F(SharedLibConnectionTest, Init) {
  auto output = ::testing::internal::GetCapturedStderr();
  EXPECT_THAT(output, ::testing::HasSubstr("make_peripheral_stub(testctx)"));
  EXPECT_THAT(output,
              ::testing::HasSubstr("IPeripheralStub::IPeripheralStub()"));
}

TEST_F(SharedLibConnectionTest, HandleReset) {
  ::renode::reset_peripheral();

  auto output = ::testing::internal::GetCapturedStderr();
  EXPECT_THAT(output, ::testing::HasSubstr("IPeripheralStub::Reset()"));
}

TEST_F(SharedLibConnectionTest, HandleRequest) {
  ::renode::renode_external_attach__ActionIntPtr__HandleMainMessage(
      renode_stub_handlemainmessage);

  ::renode::ProtocolMessage msg{::renode::Action::writeRequestWord, 0x1234,
                                0x5678};
  ::renode::handle_request(&msg);

  auto output = ::testing::internal::GetCapturedStderr();
  EXPECT_THAT(output, ::testing::HasSubstr("IPeripheralStub::HandleWrite"));
  EXPECT_THAT(output, ::testing::HasSubstr(absl::StrFormat("width=%u", 1)));
  EXPECT_THAT(output, ::testing::HasSubstr(absl::StrFormat("addr=%u", 0x1234)));
  EXPECT_THAT(output,
              ::testing::HasSubstr(absl::StrFormat("payload=%u", 0x5678)));
}

TEST_F(SharedLibConnectionTest, SendResponse) {
  ::renode::renode_external_attach__ActionIntPtr__HandleMainMessage(
      renode_stub_handlemainmessage);

  XLS_EXPECT_OK(SharedLibConnection::Instance().SendResponse(
      ::renode::ProtocolMessage{::renode::Action::getByte, 0x3456, 0x7890}));
  auto output = ::testing::internal::GetCapturedStderr();

  EXPECT_THAT(
      output,
      ::testing::HasSubstr(absl::StrFormat(
          "renode_stub_handlemainmessage(action=%u, addr=%u, value=%u)",
          static_cast<uint64_t>(::renode::Action::getByte), 0x3456, 0x7890)));
}

TEST_F(SharedLibConnectionTest, SendRequest) {
  ::renode::renode_external_attach__ActionIntPtr__HandleSenderMessage(
      renode_stub_handlesendermessage);
  XLS_EXPECT_OK(SharedLibConnection::Instance().SendRequest(
      ::renode::ProtocolMessage{::renode::Action::getByte, 0x3456, 0x7890}));
  auto output = ::testing::internal::GetCapturedStderr();

  EXPECT_THAT(
      output,
      ::testing::HasSubstr(absl::StrFormat(
          "renode_stub_handlesendermessage(action=%u, addr=%u, value=%u)",
          static_cast<uint64_t>(::renode::Action::getByte), 0x3456, 0x7890)));
}

TEST_F(SharedLibConnectionTest, ReceiveResponse) {
  ::renode::renode_external_attach__ActionIntPtr__Receive(renode_stub_receive);

  XLS_ASSERT_OK_AND_ASSIGN(auto resp,
                           SharedLibConnection::Instance().ReceiveResponse());
  auto output = ::testing::internal::GetCapturedStderr();

  EXPECT_THAT(output, ::testing::HasSubstr("renode_stub_receive()"));
  EXPECT_EQ(resp.actionId, ::renode::Action::handshake);
  EXPECT_EQ(resp.addr, 0x1234);
  EXPECT_EQ(resp.value, 0x5678);
}

TEST_F(SharedLibConnectionTest, Log) {
  ::renode::renode_external_attach__ActionIntPtr__HandleSenderMessage(
      renode_stub_handlesendermessage);
  auto ret =
      SharedLibConnection::Instance().Log(absl::LogSeverity::kInfo, "testmsg");
  auto output = ::testing::internal::GetCapturedStderr();

  EXPECT_THAT(output, ::testing::HasSubstr(
                          "renode_stub_handlesendermessage: log(msg=testmsg)"));
  EXPECT_THAT(output, ::testing::HasSubstr(
                          "renode_stub_handlesendermessage: log(level=1)"));
}

}  // namespace
}  // namespace xls::simulation::renode
