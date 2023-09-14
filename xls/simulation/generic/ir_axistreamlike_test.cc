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

#include "xls/simulation/generic/ir_axistreamlike.h"

#include <exception>
#include <memory>
#include <new>
#include <optional>
#include <stdexcept>

#include "absl/status/status.h"
#include "gtest/gtest.h"
#include "xls/common/status/matchers.h"
#include "xls/interpreter/channel_queue.h"
#include "xls/ir/channel.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/package.h"
#include "xls/ir/value_helpers.h"

namespace xls::simulation::generic {
namespace {

using ::testing::Optional;

class MakeTest : public IrTestBase {
 protected:
  MakeTest() : package_(TestName()) {}
  void SetUp() final {
    // Single-symbol channel type
    std::vector<Type*> single_members = {
        package_.GetBitsType(2),   // padding
        package_.GetBitsType(11),  // data
        package_.GetBitsType(1),   // tlast
        package_.GetBitsType(1),   // tkeep
        package_.GetBitsType(5),   // padding
    };
    singlesymbol_type_ = package_.GetTupleType(single_members);
    // Multi-symbol channel type
    std::vector<Type*> multi_members = {
        package_.GetBitsType(2),                             // padding
        package_.GetArrayType(3, package_.GetBitsType(11)),  // data
        package_.GetBitsType(1),                             // tlast
        package_.GetBitsType(3),                             // tkeep
        package_.GetBitsType(5),                             // padding
    };
    multisymbol_type_ = package_.GetTupleType(multi_members);
  }

  Package package_;
  TupleType* singlesymbol_type_;
  TupleType* multisymbol_type_;
};

TEST_F(MakeTest, RecvSingle) {
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * channel,
      this->package_.CreateStreamingChannel(
          "my_channel", ChannelOps::kReceiveOnly, singlesymbol_type_));
  ChannelQueue queue = ChannelQueue(channel);
  EXPECT_THAT(IrAxiStreamLike::Make(&queue, false, 1, 2, 3),
              xls::status_testing::IsOk());
}

TEST_F(MakeTest, RecvMulti) {
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * channel,
      this->package_.CreateStreamingChannel(
          "my_channel", ChannelOps::kReceiveOnly, multisymbol_type_));
  ChannelQueue queue = ChannelQueue(channel);
  EXPECT_THAT(IrAxiStreamLike::Make(&queue, true, 1, 2, 3),
              xls::status_testing::IsOk());
}

TEST_F(MakeTest, SendSingle) {
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * channel,
      this->package_.CreateStreamingChannel("my_channel", ChannelOps::kSendOnly,
                                            singlesymbol_type_));
  ChannelQueue queue = ChannelQueue(channel);
  EXPECT_THAT(IrAxiStreamLike::Make(&queue, false, 1, 2, 3),
              xls::status_testing::IsOk());
}

TEST_F(MakeTest, SendMulti) {
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * channel,
      this->package_.CreateStreamingChannel("my_channel", ChannelOps::kSendOnly,
                                            multisymbol_type_));
  ChannelQueue queue = ChannelQueue(channel);
  EXPECT_THAT(IrAxiStreamLike::Make(&queue, true, 1, 2, 3),
              xls::status_testing::IsOk());
}

TEST_F(MakeTest, SingleForMulti) {
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * channel,
      this->package_.CreateStreamingChannel(
          "my_channel", ChannelOps::kReceiveOnly, multisymbol_type_));
  ChannelQueue queue = ChannelQueue(channel);
  EXPECT_THAT(IrAxiStreamLike::Make(&queue, false, 1, 2, 3),
              xls::status_testing::StatusIs(
                  absl::StatusCode::kInvalidArgument,
                  testing::HasSubstr("TKEEP value idx=3 should be of type")));
}

TEST_F(MakeTest, MultiForSingle) {
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * channel,
      this->package_.CreateStreamingChannel(
          "my_channel", ChannelOps::kReceiveOnly, singlesymbol_type_));
  ChannelQueue queue = ChannelQueue(channel);
  EXPECT_THAT(
      IrAxiStreamLike::Make(&queue, true, 1, 2, 3),
      xls::status_testing::StatusIs(
          absl::StatusCode::kInvalidArgument,
          testing::HasSubstr("DATA value idx=1 should be of array type")));
}

TEST_F(MakeTest, Bidirectional) {
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * channel,
      this->package_.CreateStreamingChannel(
          "my_channel", ChannelOps::kSendReceive, singlesymbol_type_));
  ChannelQueue queue = ChannelQueue(channel);
  EXPECT_THAT(
      IrAxiStreamLike::Make(&queue, false, 1, 2, 3),
      xls::status_testing::StatusIs(
          absl::StatusCode::kInvalidArgument,
          testing::HasSubstr(
              "Given a bidirectional channel. These are not supported.")));
}

class SingleSymbolReadTest : public IrTestBase {
 protected:
  SingleSymbolReadTest() : package_(TestName()) {}
  void SetUp() override {
    std::vector<Type*> members = {
        package_.GetBitsType(2),   // padding
        package_.GetBitsType(11),  // data
        package_.GetBitsType(1),   // tlast
        package_.GetBitsType(5),   // padding
    };
    type_ = package_.GetTupleType(members);
    auto chn = this->package_.CreateStreamingChannel(
        "my_channel", ChannelOps::kSendOnly, type_);
    XLS_ASSERT_OK(chn);
    channel_ = chn.value();
    queue_ = std::make_unique<ChannelQueue>(channel_);
    dut_ = std::make_unique<IrAxiStreamLike>(
        IrAxiStreamLike::Make(queue_.get(), false, 1, 2, std::nullopt).value());
  }
  Package package_;
  TupleType* type_;
  Channel* channel_;
  std::unique_ptr<ChannelQueue> queue_;
  std::unique_ptr<IrAxiStreamLike> dut_;
};

TEST_F(SingleSymbolReadTest, IsReadStream) {
  EXPECT_EQ(dut_->IsReadStream(), true);
}

TEST_F(SingleSymbolReadTest, GetNumSymbols) {
  EXPECT_EQ(dut_->GetNumSymbols(), 1);
}

TEST_F(SingleSymbolReadTest, GetSymbolWidth) {
  // 11 bits symbol payload
  EXPECT_EQ(dut_->GetSymbolWidth(), 11);
}

TEST_F(SingleSymbolReadTest, GetSymbolSize) {
  // 11 bits padded to 16=2x8 bits, so 2 bytes
  EXPECT_EQ(dut_->GetSymbolSize(), 2);
}

TEST_F(SingleSymbolReadTest, GetChannelWidth) {
  // Width of all padded symbols combined, in bits
  EXPECT_EQ(dut_->GetChannelWidth(), 16);
}

TEST_F(SingleSymbolReadTest, Transfer) {
  EXPECT_FALSE(dut_->IsReady());
  // fill the queue
  XLS_ASSERT_OK(queue_->Write(ZeroOfType(type_)));
  EXPECT_TRUE(this->dut_->IsReady());
  XLS_ASSERT_OK(queue_->Write(ZeroOfType(type_)));
  EXPECT_TRUE(this->dut_->IsReady());
  // read out
  XLS_ASSERT_OK(dut_->Transfer());
  EXPECT_TRUE(this->dut_->IsReady());
  XLS_ASSERT_OK(dut_->Transfer());
  EXPECT_FALSE(this->dut_->IsReady());
  // check handling of underflow
  EXPECT_THAT(dut_->Transfer(),
              xls::status_testing::StatusIs(absl::StatusCode::kInternal,
                                            testing::HasSubstr("is empty")));
  EXPECT_FALSE(this->dut_->IsReady());
}

TEST_F(SingleSymbolReadTest, ReadDefault) {
  EXPECT_EQ(dut_->GetLast(), false);
  EXPECT_EQ(dut_->GetDataValid(), std::vector<bool>{true});
  EXPECT_THAT(dut_->GetPayloadData8(0), xls::status_testing::IsOkAndHolds(0));
  EXPECT_THAT(dut_->GetPayloadData16(0), xls::status_testing::IsOkAndHolds(0));
  EXPECT_THAT(dut_->GetPayloadData32(0), xls::status_testing::IsOkAndHolds(0));
  EXPECT_THAT(dut_->GetPayloadData64(0), xls::status_testing::IsOkAndHolds(0));
}

TEST_F(SingleSymbolReadTest, ReadPayload) {
  auto payload = Value::Tuple({
      Value(UBits(0x3, 2)),     // pad
      Value(UBits(0x7AB, 11)),  // data
      Value(UBits(1, 1)),       // tlast
      Value(UBits(0x11, 5)),    // pad
  });
  XLS_ASSERT_OK(queue_->Write(payload));
  XLS_ASSERT_OK(dut_->Transfer());
  EXPECT_EQ(dut_->GetLast(), true);
  EXPECT_EQ(dut_->GetDataValid(), std::vector<bool>{true});
  EXPECT_THAT(dut_->GetPayloadData8(0),
              xls::status_testing::IsOkAndHolds(0xAB));
  EXPECT_THAT(dut_->GetPayloadData8(1),
              xls::status_testing::IsOkAndHolds(0x07));
  EXPECT_THAT(dut_->GetPayloadData16(0),
              xls::status_testing::IsOkAndHolds(0x7AB));
  EXPECT_THAT(dut_->GetPayloadData32(0),
              xls::status_testing::IsOkAndHolds(0x7AB));
  EXPECT_THAT(dut_->GetPayloadData64(0),
              xls::status_testing::IsOkAndHolds(0x7AB));
  EXPECT_THAT(dut_->GetPayloadData8(2),
              xls::status_testing::StatusIs(absl::StatusCode::kInvalidArgument,
                                            testing::HasSubstr("is outside")));
}

class SingleSymbolWriteTest : public IrTestBase {
 protected:
  SingleSymbolWriteTest() : package_(TestName()) {}
  void SetUp() override {
    std::vector<Type*> members = {
        package_.GetBitsType(2),   // padding
        package_.GetBitsType(11),  // data
        package_.GetBitsType(1),   // tlast
        package_.GetBitsType(5),   // padding
    };
    type_ = package_.GetTupleType(members);
    auto chn = this->package_.CreateStreamingChannel(
        "my_channel", ChannelOps::kReceiveOnly, type_);
    XLS_ASSERT_OK(chn);
    channel_ = chn.value();
    queue_ = std::make_unique<ChannelQueue>(channel_);
    dut_ = std::make_unique<IrAxiStreamLike>(
        IrAxiStreamLike::Make(queue_.get(), false, 1, 2, std::nullopt).value());
  }
  Package package_;
  TupleType* type_;
  Channel* channel_;
  std::unique_ptr<ChannelQueue> queue_;
  std::unique_ptr<IrAxiStreamLike> dut_;
};

TEST_F(SingleSymbolWriteTest, IsReadStream) {
  EXPECT_EQ(dut_->IsReadStream(), false);
}

TEST_F(SingleSymbolWriteTest, GetNumSymbols) {
  EXPECT_EQ(dut_->GetNumSymbols(), 1);
}

TEST_F(SingleSymbolWriteTest, GetSymbolWidth) {
  // 11 bits symbol payload
  EXPECT_EQ(dut_->GetSymbolWidth(), 11);
}

TEST_F(SingleSymbolWriteTest, GetSymbolSize) {
  // 11 bits padded to 16=2x8 bits, so 2 bytes
  EXPECT_EQ(dut_->GetSymbolSize(), 2);
}

TEST_F(SingleSymbolWriteTest, GetChannelWidth) {
  // Width of all padded symbols combined, in bits
  EXPECT_EQ(dut_->GetChannelWidth(), 16);
}

TEST_F(SingleSymbolWriteTest, Transfer) {
  EXPECT_TRUE(dut_->IsReady());
  for (uint32_t i = 0; i < IrAxiStreamLike::kWriteFifoMaxDepth; i++) {
    XLS_ASSERT_OK(dut_->Transfer());
  }
  EXPECT_FALSE(dut_->IsReady());
  // check handling of overflow
  EXPECT_THAT(dut_->Transfer(),
              xls::status_testing::StatusIs(
                  absl::StatusCode::kInternal,
                  testing::HasSubstr("overflown, data is lost")));
  for (uint32_t i = 0; i < IrAxiStreamLike::kWriteFifoMaxDepth; i++) {
    EXPECT_EQ(queue_->Read()->empty(), false);
    EXPECT_TRUE(dut_->IsReady());
  }
  EXPECT_EQ(queue_->IsEmpty(), true);
}

TEST_F(SingleSymbolWriteTest, ReadDefault) {
  EXPECT_EQ(dut_->GetLast(), false);
  EXPECT_EQ(dut_->GetDataValid(), std::vector<bool>{{true}});
  EXPECT_THAT(dut_->GetPayloadData8(0), xls::status_testing::IsOkAndHolds(0));
  EXPECT_THAT(dut_->GetPayloadData16(0), xls::status_testing::IsOkAndHolds(0));
  EXPECT_THAT(dut_->GetPayloadData32(0), xls::status_testing::IsOkAndHolds(0));
  EXPECT_THAT(dut_->GetPayloadData64(0), xls::status_testing::IsOkAndHolds(0));
}

TEST_F(SingleSymbolWriteTest, WritePayload) {
  dut_->SetLast(true);
  XLS_ASSERT_OK(dut_->SetPayloadData64(0, 0x7AB));
  EXPECT_THAT(dut_->SetPayloadData64(2, 0),
              xls::status_testing::StatusIs(absl::StatusCode::kInvalidArgument,
                                            testing::HasSubstr("is outside")));
  XLS_ASSERT_OK(dut_->Transfer());
  auto popt = queue_->Read();
  EXPECT_EQ(popt->empty(), false);
  auto payload = *popt;
  auto payload_ref = Value::Tuple({
      Value(UBits(0, 2)),       // pad
      Value(UBits(0x7AB, 11)),  // data
      Value(UBits(1, 1)),       // tlast
      Value(UBits(0, 5)),       // pad
  });

  EXPECT_EQ(payload, payload_ref);
}

class MultiSymbolReadTest : public IrTestBase {
 protected:
  MultiSymbolReadTest() : package_(TestName()) {}
  void SetUp() override {
    std::vector<Type*> members = {
        package_.GetBitsType(2),                             // padding
        package_.GetArrayType(3, package_.GetBitsType(11)),  // data
        package_.GetBitsType(3),                             // tkeep
        package_.GetBitsType(1),                             // tlast
        package_.GetBitsType(5),                             // padding
    };
    type_ = package_.GetTupleType(members);
    auto chn = this->package_.CreateStreamingChannel(
        "my_channel", ChannelOps::kSendOnly, type_);
    XLS_ASSERT_OK(chn);
    channel_ = chn.value();
    queue_ = std::make_unique<ChannelQueue>(channel_);
    dut_ = std::make_unique<IrAxiStreamLike>(
        IrAxiStreamLike::Make(queue_.get(), true, 1, 3, 2).value());
  }
  Package package_;
  TupleType* type_;
  Channel* channel_;
  std::unique_ptr<ChannelQueue> queue_;
  std::unique_ptr<IrAxiStreamLike> dut_;
};

TEST_F(MultiSymbolReadTest, IsReadStream) {
  EXPECT_EQ(dut_->IsReadStream(), true);
}

TEST_F(MultiSymbolReadTest, GetNumSymbols) {
  EXPECT_EQ(dut_->GetNumSymbols(), 3);
}

TEST_F(MultiSymbolReadTest, GetSymbolWidth) {
  // 11 bits symbol payload
  EXPECT_EQ(dut_->GetSymbolWidth(), 11);
}

TEST_F(MultiSymbolReadTest, GetSymbolSize) {
  // 11 bits padded to 16=2x8 bits, so 2 bytes
  EXPECT_EQ(dut_->GetSymbolSize(), 2);
}

TEST_F(MultiSymbolReadTest, GetChannelWidth) {
  // Width of all padded symbols combined, in bits
  EXPECT_EQ(dut_->GetChannelWidth(), 48);
}

TEST_F(MultiSymbolReadTest, Transfer) {
  EXPECT_FALSE(dut_->IsReady());
  // fill the queue
  XLS_ASSERT_OK(queue_->Write(ZeroOfType(type_)));
  EXPECT_TRUE(this->dut_->IsReady());
  XLS_ASSERT_OK(queue_->Write(ZeroOfType(type_)));
  EXPECT_TRUE(this->dut_->IsReady());
  // read out
  XLS_ASSERT_OK(dut_->Transfer());
  EXPECT_TRUE(this->dut_->IsReady());
  XLS_ASSERT_OK(dut_->Transfer());
  EXPECT_FALSE(this->dut_->IsReady());
  // check handling of underflow
  EXPECT_THAT(dut_->Transfer(),
              xls::status_testing::StatusIs(absl::StatusCode::kInternal,
                                            testing::HasSubstr("is empty")));
  EXPECT_FALSE(this->dut_->IsReady());
}

TEST_F(MultiSymbolReadTest, ReadDefault) {
  EXPECT_EQ(dut_->GetLast(), false);
  auto valid_ref = std::vector<bool>{{false, false, false}};
  EXPECT_EQ(dut_->GetDataValid(), valid_ref);
  EXPECT_THAT(dut_->GetPayloadData8(0), xls::status_testing::IsOkAndHolds(0));
  EXPECT_THAT(dut_->GetPayloadData16(0), xls::status_testing::IsOkAndHolds(0));
  EXPECT_THAT(dut_->GetPayloadData32(0), xls::status_testing::IsOkAndHolds(0));
  EXPECT_THAT(dut_->GetPayloadData64(0), xls::status_testing::IsOkAndHolds(0));
}

TEST_F(MultiSymbolReadTest, ReadPayload) {
  auto payload = Value::Tuple({
      Value(UBits(0x3, 2)),  // pad
      Value::Array({
                       Value(UBits(0x4CD, 11)),
                       Value(UBits(0x7AB, 11)),
                       Value(UBits(0x511, 11)),
                   })
          .value(),           // data
      Value(UBits(0x6, 3)),   // tkeep
      Value(UBits(1, 1)),     // tlast
      Value(UBits(0x11, 5)),  // pad
  });
  XLS_ASSERT_OK(queue_->Write(payload));
  XLS_ASSERT_OK(dut_->Transfer());
  EXPECT_EQ(dut_->GetLast(), true);
  auto valid_ref = std::vector<bool>{{false, true, true}};
  EXPECT_EQ(dut_->GetDataValid(), valid_ref);
  EXPECT_THAT(dut_->GetPayloadData8(0),
              xls::status_testing::IsOkAndHolds(0xCD));
  EXPECT_THAT(dut_->GetPayloadData8(1),
              xls::status_testing::IsOkAndHolds(0x04));
  EXPECT_THAT(dut_->GetPayloadData8(2),
              xls::status_testing::IsOkAndHolds(0xAB));
  EXPECT_THAT(dut_->GetPayloadData8(3),
              xls::status_testing::IsOkAndHolds(0x07));
  EXPECT_THAT(dut_->GetPayloadData8(4),
              xls::status_testing::IsOkAndHolds(0x11));
  EXPECT_THAT(dut_->GetPayloadData8(5),
              xls::status_testing::IsOkAndHolds(0x05));
  EXPECT_THAT(dut_->GetPayloadData16(0),
              xls::status_testing::IsOkAndHolds(0x4CD));
  EXPECT_THAT(dut_->GetPayloadData16(1),
              xls::status_testing::IsOkAndHolds(0xAB04));
  EXPECT_THAT(dut_->GetPayloadData16(2),
              xls::status_testing::IsOkAndHolds(0x07AB));
  EXPECT_THAT(dut_->GetPayloadData16(4),
              xls::status_testing::IsOkAndHolds(0x511));
  EXPECT_THAT(dut_->GetPayloadData32(0),
              xls::status_testing::IsOkAndHolds(0x07AB04CD));
  EXPECT_THAT(dut_->GetPayloadData32(4),
              xls::status_testing::IsOkAndHolds(0x511));
  EXPECT_THAT(dut_->GetPayloadData64(0),
              xls::status_testing::IsOkAndHolds(0x051107AB04CD));
  EXPECT_THAT(dut_->GetPayloadData8(6),
              xls::status_testing::StatusIs(absl::StatusCode::kInvalidArgument,
                                            testing::HasSubstr("is outside")));
}

class MultiSymbolWriteTest : public IrTestBase {
 protected:
  MultiSymbolWriteTest() : package_(TestName()) {}
  void SetUp() override {
    std::vector<Type*> members = {
        package_.GetBitsType(2),                             // padding
        package_.GetArrayType(3, package_.GetBitsType(11)),  // data
        package_.GetBitsType(3),                             // tkeep
        package_.GetBitsType(1),                             // tlast
        package_.GetBitsType(5),                             // padding
    };
    type_ = package_.GetTupleType(members);
    auto chn = this->package_.CreateStreamingChannel(
        "my_channel", ChannelOps::kReceiveOnly, type_);
    XLS_ASSERT_OK(chn);
    channel_ = chn.value();
    queue_ = std::make_unique<ChannelQueue>(channel_);
    dut_ = std::make_unique<IrAxiStreamLike>(
        IrAxiStreamLike::Make(queue_.get(), true, 1, 3, 2).value());
  }
  Package package_;
  TupleType* type_;
  Channel* channel_;
  std::unique_ptr<ChannelQueue> queue_;
  std::unique_ptr<IrAxiStreamLike> dut_;
};

TEST_F(MultiSymbolWriteTest, IsReadStream) {
  EXPECT_EQ(dut_->IsReadStream(), false);
}

TEST_F(MultiSymbolWriteTest, GetNumSymbols) {
  EXPECT_EQ(dut_->GetNumSymbols(), 3);
}

TEST_F(MultiSymbolWriteTest, GetSymbolWidth) {
  // 11 bits symbol payload
  EXPECT_EQ(dut_->GetSymbolWidth(), 11);
}

TEST_F(MultiSymbolWriteTest, GetSymbolSize) {
  // 11 bits padded to 16=2x8 bits, so 2 bytes
  EXPECT_EQ(dut_->GetSymbolSize(), 2);
}

TEST_F(MultiSymbolWriteTest, GetChannelWidth) {
  // Width of all padded symbols combined, in bits
  EXPECT_EQ(dut_->GetChannelWidth(), 48);
}

TEST_F(MultiSymbolWriteTest, Transfer) {
  EXPECT_TRUE(dut_->IsReady());
  for (uint32_t i = 0; i < IrAxiStreamLike::kWriteFifoMaxDepth; i++) {
    XLS_ASSERT_OK(dut_->Transfer());
  }
  EXPECT_FALSE(dut_->IsReady());
  // check handling of overflow
  EXPECT_THAT(dut_->Transfer(),
              xls::status_testing::StatusIs(
                  absl::StatusCode::kInternal,
                  testing::HasSubstr("overflown, data is lost")));
  for (uint32_t i = 0; i < IrAxiStreamLike::kWriteFifoMaxDepth; i++) {
    EXPECT_EQ(queue_->Read()->empty(), false);
    EXPECT_TRUE(dut_->IsReady());
  }
  EXPECT_EQ(queue_->IsEmpty(), true);
}

TEST_F(MultiSymbolWriteTest, ReadDefault) {
  EXPECT_EQ(dut_->GetLast(), false);
  auto valid_ref = std::vector<bool>{{false, false, false}};
  EXPECT_EQ(dut_->GetDataValid(), valid_ref);
  EXPECT_THAT(dut_->GetPayloadData8(0), xls::status_testing::IsOkAndHolds(0));
  EXPECT_THAT(dut_->GetPayloadData16(0), xls::status_testing::IsOkAndHolds(0));
  EXPECT_THAT(dut_->GetPayloadData32(0), xls::status_testing::IsOkAndHolds(0));
  EXPECT_THAT(dut_->GetPayloadData64(0), xls::status_testing::IsOkAndHolds(0));
}

TEST_F(MultiSymbolWriteTest, WritePayload) {
  dut_->SetLast(true);
  dut_->SetDataValid(std::vector<bool>{{true, true, false}});
  XLS_ASSERT_OK(dut_->SetPayloadData64(0, 0x01AC07BA02DD));
  EXPECT_THAT(dut_->SetPayloadData64(6, 0),
              xls::status_testing::StatusIs(absl::StatusCode::kInvalidArgument,
                                            testing::HasSubstr("is outside")));
  XLS_ASSERT_OK(dut_->Transfer());
  auto popt = queue_->Read();
  EXPECT_EQ(popt->empty(), false);
  auto payload = *popt;
  auto payload_ref = Value::Tuple({
      Value(UBits(0, 2)),  // pad
      Value::Array({
                       Value(UBits(0x2DD, 11)),
                       Value(UBits(0x7BA, 11)),
                       Value(UBits(0x1AC, 11)),
                   })
          .value(),          // data
      Value(UBits(0x3, 3)),  // tkeep
      Value(UBits(1, 1)),    // tlast
      Value(UBits(0, 5)),    // pad
  });

  EXPECT_EQ(payload, payload_ref);
}

}  // namespace
}  // namespace xls::simulation::generic
