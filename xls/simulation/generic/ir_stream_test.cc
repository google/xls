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

#include "xls/simulation/generic/ir_stream.h"

#include <memory>

#include "absl/status/status.h"
#include "gtest/gtest.h"
#include "xls/common/status/matchers.h"
#include "xls/interpreter/channel_queue.h"
#include "xls/ir/channel.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/package.h"

namespace xls::simulation::generic {
namespace {

using ::testing::Optional;

class MakeIrStreamQueueTest : public IrTestBase {
 protected:
  MakeIrStreamQueueTest() : package_(TestName()) {}
  Package package_;
};

TEST_F(MakeIrStreamQueueTest, MakeIRStreamRead) {
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * channel,
      this->package_.CreateStreamingChannel("my_channel", ChannelOps::kSendOnly,
                                            this->package_.GetBitsType(64)));
  ChannelQueue single_value_queue = ChannelQueue(channel);
  EXPECT_THAT(IRStream::MakeIRStream(&single_value_queue),
              xls::status_testing::IsOk());
}

TEST_F(MakeIrStreamQueueTest, MakeIRStreamWrite) {
  XLS_ASSERT_OK_AND_ASSIGN(Channel * channel,
                           this->package_.CreateStreamingChannel(
                               "my_channel", ChannelOps::kReceiveOnly,
                               this->package_.GetBitsType(64)));
  ChannelQueue single_value_queue = ChannelQueue(channel);
  EXPECT_THAT(IRStream::MakeIRStream(&single_value_queue),
              xls::status_testing::IsOk());
}

TEST_F(MakeIrStreamQueueTest, MakeIRStreamReadWrite) {
  XLS_ASSERT_OK_AND_ASSIGN(Channel * channel,
                           this->package_.CreateStreamingChannel(
                               "my_channel", ChannelOps::kSendReceive,
                               this->package_.GetBitsType(64)));
  std::unique_ptr<ChannelQueue> single_value_queue =
      std::make_unique<ChannelQueue>(channel);
  EXPECT_THAT(
      IRStream::MakeIRStream(single_value_queue.get()),
      xls::status_testing::StatusIs(
          absl::StatusCode::kInvalidArgument,
          testing::HasSubstr(
              "MakeIRStream expects unidirectional Streaming channel.")));
}

TEST_F(MakeIrStreamQueueTest, MakeIRStreamWithSingleValueChannel) {
  XLS_ASSERT_OK_AND_ASSIGN(Channel * channel,
                           this->package_.CreateSingleValueChannel(
                               "my_channel", ChannelOps::kSendReceive,
                               this->package_.GetBitsType(64)));
  std::unique_ptr<ChannelQueue> single_value_queue =
      std::make_unique<ChannelQueue>(channel);
  EXPECT_THAT(IRStream::MakeIRStream(single_value_queue.get()),
              xls::status_testing::StatusIs(
                  absl::StatusCode::kInvalidArgument,
                  testing::HasSubstr(
                      "MakeIRStream expects queue with Streaming channel.")));
}

class IrStreamReadQueueTest : public IrTestBase {
 protected:
  IrStreamReadQueueTest() : pkg_(Package(TestName())) {}
  void SetUp() {
    auto new_channel_ = this->pkg_.CreateStreamingChannel(
        "my_channel", ChannelOps::kSendOnly, this->pkg_.GetBitsType(64));
    XLS_ASSERT_OK(new_channel_);
    Channel* channel_ = new_channel_.value();
    this->stream_queue_ = std::make_unique<ChannelQueue>(channel_);
    this->test_obj_ = std::make_unique<IRStream>(
        IRStream::MakeIRStream(this->stream_queue_.get()).value());
  }

  Package pkg_;
  std::unique_ptr<ChannelQueue> stream_queue_;
  std::unique_ptr<IRStream> test_obj_;
};

TEST_F(IrStreamReadQueueTest, GetChannelWidth) {
  EXPECT_EQ(this->test_obj_->GetChannelWidth(), 64);
}

TEST_F(IrStreamReadQueueTest, IsReadStream) {
  EXPECT_EQ(this->test_obj_->IsReadStream(), true);
}

TEST_F(IrStreamReadQueueTest, IsReady) {
  EXPECT_FALSE(this->test_obj_->IsReady());
  XLS_ASSERT_OK(
      this->stream_queue_->Write(Value(UBits(0xFEDCBA9876543210, 64))));
  EXPECT_TRUE(this->test_obj_->IsReady());

  this->stream_queue_->Read();
  EXPECT_TRUE(this->stream_queue_->IsEmpty());
  EXPECT_FALSE(this->test_obj_->IsReady());
}

TEST_F(IrStreamReadQueueTest, Transfer) {
  EXPECT_THAT(this->test_obj_->GetPayloadData64(0),
              xls::status_testing::IsOkAndHolds(0x0));
  EXPECT_FALSE(this->test_obj_->IsReady());
  EXPECT_THAT(this->stream_queue_->Write(Value(UBits(0xFEDCBA9876543210, 64))),
              xls::status_testing::IsOk());
  EXPECT_EQ(this->test_obj_->IsReady(), !this->stream_queue_->IsEmpty());
  EXPECT_THAT(this->test_obj_->Transfer(), xls::status_testing::IsOk());

  EXPECT_EQ(this->stream_queue_->IsEmpty(), true);
  EXPECT_EQ(this->test_obj_->IsReady(), false);

  EXPECT_THAT(this->test_obj_->GetPayloadData64(0),
              xls::status_testing::IsOkAndHolds(0xFEDCBA9876543210));

  EXPECT_THAT(this->test_obj_->Transfer(),
              xls::status_testing::StatusIs(
                  absl::StatusCode::kInternal,
                  testing::HasSubstr("was empty during read")));
  EXPECT_THAT(this->test_obj_->GetPayloadData64(0),
              xls::status_testing::IsOkAndHolds(0xFEDCBA9876543210));
}

class IrStreamWriteQueueTest : public IrTestBase {
 protected:
  IrStreamWriteQueueTest() : pkg_(Package(TestName())) {}

  void SetUp() {
    auto new_channel_ = this->pkg_.CreateStreamingChannel(
        "my_channel", ChannelOps::kReceiveOnly, this->pkg_.GetBitsType(64));
    XLS_ASSERT_OK(new_channel_);
    Channel* channel_ = new_channel_.value();
    this->stream_queue_ = std::make_unique<ChannelQueue>(channel_);
    this->test_obj_ = std::make_unique<IRStream>(
        IRStream::MakeIRStream(this->stream_queue_.get()).value());
  }

  Package pkg_;
  std::unique_ptr<ChannelQueue> stream_queue_;
  std::unique_ptr<IRStream> test_obj_;
};

TEST_F(IrStreamWriteQueueTest, GetChannelWidth) {
  EXPECT_EQ(this->test_obj_->GetChannelWidth(), 64);
}

TEST_F(IrStreamWriteQueueTest, IsReadStream) {
  EXPECT_EQ(this->test_obj_->IsReadStream(), false);
}

TEST_F(IrStreamWriteQueueTest, IsReady) {
  EXPECT_TRUE(this->test_obj_->IsReady());
  XLS_ASSERT_OK(
      this->stream_queue_->Write(Value(UBits(0xFEDCBA9876543210, 64))));
  EXPECT_TRUE(this->test_obj_->IsReady());
  XLS_ASSERT_OK(
      this->stream_queue_->Write(Value(UBits(0xFEDCBA9876543210, 64))));
  EXPECT_TRUE(this->test_obj_->IsReady());
  XLS_ASSERT_OK(
      this->stream_queue_->Write(Value(UBits(0xFEDCBA9876543210, 64))));
  EXPECT_TRUE(this->test_obj_->IsReady());
  XLS_ASSERT_OK(
      this->stream_queue_->Write(Value(UBits(0xFEDCBA9876543210, 64))));
  EXPECT_FALSE(this->test_obj_->IsReady());
}

TEST_F(IrStreamWriteQueueTest, Transfer) {
  EXPECT_THAT(this->test_obj_->SetPayloadData64(0, 0xFEDCBA9876543210),
              xls::status_testing::IsOk());
  EXPECT_TRUE(this->test_obj_->IsReady());
  EXPECT_THAT(this->test_obj_->Transfer(), xls::status_testing::IsOk());
  EXPECT_TRUE(this->test_obj_->IsReady());
  EXPECT_THAT(this->stream_queue_->Read(),
              Optional(Value(UBits(0xFEDCBA9876543210, 64))));

  ASSERT_TRUE(this->stream_queue_->IsEmpty());
  EXPECT_THAT(this->test_obj_->Transfer(), xls::status_testing::IsOk());
  EXPECT_THAT(this->test_obj_->Transfer(), xls::status_testing::IsOk());
  EXPECT_THAT(this->test_obj_->Transfer(), xls::status_testing::IsOk());
  EXPECT_THAT(this->test_obj_->Transfer(), xls::status_testing::IsOk());
  EXPECT_THAT(this->test_obj_->Transfer(),
              xls::status_testing::StatusIs(absl::StatusCode::kInternal,
                                            testing::HasSubstr("overfill")));
}

class IRStreamInternalBufferTestBase
    : public IrTestBase,
      public testing::WithParamInterface<xls::ChannelOps> {
 protected:
  IRStreamInternalBufferTestBase() : pkg_(Package(TestName())) {}

  void SetUp() {
    auto new_channel_ = this->pkg_.CreateStreamingChannel(
        "my_channel", GetParam(), this->pkg_.GetBitsType(64));
    XLS_ASSERT_OK(new_channel_);
    Channel* channel_ = new_channel_.value();
    this->stream_queue_ = std::make_unique<ChannelQueue>(channel_);
    this->test_obj_ = std::make_unique<IRStream>(
        IRStream::MakeIRStream(this->stream_queue_.get()).value());
  }

  Package pkg_;
  std::unique_ptr<ChannelQueue> stream_queue_;
  std::unique_ptr<IRStream> test_obj_;
};

// 2 following tests check that reads/writes to the IRStream only modify
// internal register and keep queue state instact.
TEST_P(IRStreamInternalBufferTestBase,
       TestReadAccessMethodNoQueueModification) {
  EXPECT_TRUE(this->stream_queue_->IsEmpty());

  xls::Value queue_payload = Value(UBits(0xFEDCBA9876543210, 64));
  auto queue_check = [&]() {
    EXPECT_EQ(this->stream_queue_->GetSize(), 1);
    EXPECT_THAT(this->stream_queue_->Read(), Optional(queue_payload));
    XLS_ASSERT_OK(this->stream_queue_->Write(queue_payload));
  };

  XLS_ASSERT_OK(this->stream_queue_->Write(queue_payload));
  EXPECT_EQ(this->stream_queue_->GetSize(), 1);

  EXPECT_THAT(this->test_obj_->GetPayloadData8(0),
              xls::status_testing::IsOkAndHolds(0x0));
  queue_check();

  EXPECT_THAT(this->test_obj_->GetPayloadData16(0),
              xls::status_testing::IsOkAndHolds(0x0));
  queue_check();

  EXPECT_THAT(this->test_obj_->GetPayloadData32(0),
              xls::status_testing::IsOkAndHolds(0x0));
  queue_check();

  EXPECT_THAT(this->test_obj_->GetPayloadData64(0),
              xls::status_testing::IsOkAndHolds(0x0));
  queue_check();
}

TEST_P(IRStreamInternalBufferTestBase,
       TestWriteAccessMethodNoQueueModification) {
  EXPECT_TRUE(this->stream_queue_->IsEmpty());

  xls::Value queue_payload = Value(UBits(0xFEDCBA9876543210, 64));
  auto queue_check = [&]() {
    EXPECT_EQ(this->stream_queue_->GetSize(), 1);
    EXPECT_THAT(this->stream_queue_->Read(), Optional(queue_payload));
    XLS_ASSERT_OK(this->stream_queue_->Write(queue_payload));
  };

  XLS_ASSERT_OK(this->stream_queue_->Write(queue_payload));
  EXPECT_EQ(this->stream_queue_->GetSize(), 1);

  EXPECT_THAT(this->test_obj_->SetPayloadData8(0, 0x0),
              xls::status_testing::IsOk());
  queue_check();

  EXPECT_THAT(this->test_obj_->SetPayloadData16(0, 0x0),
              xls::status_testing::IsOk());
  queue_check();

  EXPECT_THAT(this->test_obj_->SetPayloadData32(0, 0x0),
              xls::status_testing::IsOk());
  queue_check();

  EXPECT_THAT(this->test_obj_->SetPayloadData64(0, 0x0),
              xls::status_testing::IsOk());
  queue_check();
}

TEST_P(IRStreamInternalBufferTestBase, TestInternalBuffer) {
  EXPECT_THAT(this->test_obj_->SetPayloadData64(0, 0xFEDCBA9876543210),
              xls::status_testing::IsOk());
  EXPECT_THAT(this->test_obj_->GetPayloadData8(0),
              xls::status_testing::IsOkAndHolds(0x10));
  EXPECT_THAT(this->test_obj_->GetPayloadData16(0),
              xls::status_testing::IsOkAndHolds(0x3210));
  EXPECT_THAT(this->test_obj_->GetPayloadData32(0),
              xls::status_testing::IsOkAndHolds(0x76543210));
  EXPECT_THAT(this->test_obj_->GetPayloadData64(0),
              xls::status_testing::IsOkAndHolds(0xFEDCBA9876543210));

  EXPECT_THAT(this->test_obj_->SetPayloadData8(5, 0x0),
              xls::status_testing::IsOk());
  EXPECT_THAT(this->test_obj_->GetPayloadData64(0),
              xls::status_testing::IsOkAndHolds(0xFEDC009876543210));

  EXPECT_THAT(this->test_obj_->SetPayloadData16(4, 0xFF00),
              xls::status_testing::IsOk());
  EXPECT_THAT(this->test_obj_->GetPayloadData64(0),
              xls::status_testing::IsOkAndHolds(0xFEDCFF0076543210));

  EXPECT_THAT(this->test_obj_->SetPayloadData32(0, 0xDEADBEEF),
              xls::status_testing::IsOk());
  EXPECT_THAT(this->test_obj_->GetPayloadData64(0),
              xls::status_testing::IsOkAndHolds(0xFEDCFF00DEADBEEF));
}

INSTANTIATE_TEST_SUITE_P(IRStreamCommon, IRStreamInternalBufferTestBase,
                         testing::Values(ChannelOps::kReceiveOnly,
                                         ChannelOps::kSendOnly));

}  // namespace
}  // namespace xls::simulation::generic
