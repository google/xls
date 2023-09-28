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

#include "xls/simulation/generic/axi_stream_like_dma_endpoint.h"

#include <exception>
#include <memory>
#include <new>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "gtest/gtest.h"
#include "xls/common/status/matchers.h"
#include "xls/interpreter/channel_queue.h"
#include "xls/ir/channel.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/package.h"
#include "xls/ir/value_helpers.h"
#include "xls/simulation/generic/idmaendpoint.h"
#include "xls/simulation/generic/ir_axistreamlike.h"

namespace xls::simulation::generic {
namespace {

using ::testing::ElementsAre;

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
    auto stream = std::make_unique<IrAxiStreamLike>(
        IrAxiStreamLike::Make(queue_.get(), false, 1, 2, std::nullopt).value());
    dut_ = std::make_unique<AxiStreamLikeDmaEndpoint>(std::move(stream));
  }
  Package package_;
  TupleType* type_;
  Channel* channel_;
  std::unique_ptr<ChannelQueue> queue_;
  std::unique_ptr<AxiStreamLikeDmaEndpoint> dut_;
};

TEST_F(SingleSymbolReadTest, IsReadStream) {
  EXPECT_EQ(dut_->IsReadStream(), true);
}

TEST_F(SingleSymbolReadTest, GetMaxElementsPerTransfer) {
  EXPECT_EQ(dut_->GetMaxElementsPerTransfer(), 1);
}

TEST_F(SingleSymbolReadTest, GetElementSize) {
  // 11 bits padded to 16=2x8 bits, so 2 bytes
  EXPECT_EQ(dut_->GetElementSize(), 2);
}

TEST_F(SingleSymbolReadTest, Read) {
  EXPECT_FALSE(dut_->IsReady());
  // fill the queue
  XLS_ASSERT_OK(queue_->Write(Value::Tuple({
      Value(UBits(0, 2)),       // pad
      Value(UBits(0x567, 11)),  // data
      Value(UBits(0, 1)),       // tlast
      Value(UBits(0, 5)),       // pad
  })));
  EXPECT_TRUE(this->dut_->IsReady());
  XLS_ASSERT_OK(queue_->Write(Value::Tuple({
      Value(UBits(0, 2)),       // pad
      Value(UBits(0x765, 11)),  // data
      Value(UBits(1, 1)),       // tlast
      Value(UBits(0, 5)),       // pad
  })));
  EXPECT_TRUE(this->dut_->IsReady());
  // read out
  XLS_ASSERT_OK_AND_ASSIGN(auto payload, dut_->Read());
  ASSERT_THAT(payload.data, ElementsAre(0x67, 0x05));
  EXPECT_EQ(payload.last, false);
  EXPECT_TRUE(this->dut_->IsReady());
  XLS_ASSERT_OK_AND_ASSIGN(payload, dut_->Read());
  ASSERT_THAT(payload.data, ElementsAre(0x65, 0x07));
  EXPECT_EQ(payload.last, true);
  // check handling of underflow
  EXPECT_FALSE(this->dut_->IsReady());
  EXPECT_FALSE(dut_->Read().ok());
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
    auto stream = std::make_unique<IrAxiStreamLike>(
        IrAxiStreamLike::Make(queue_.get(), false, 1, 2, std::nullopt).value());
    dut_ = std::make_unique<AxiStreamLikeDmaEndpoint>(std::move(stream));
  }
  Package package_;
  TupleType* type_;
  Channel* channel_;
  std::unique_ptr<ChannelQueue> queue_;
  std::unique_ptr<AxiStreamLikeDmaEndpoint> dut_;
};

TEST_F(SingleSymbolWriteTest, IsReadStream) {
  EXPECT_EQ(dut_->IsReadStream(), false);
}

TEST_F(SingleSymbolWriteTest, GetMaxElementsPerTransfer) {
  EXPECT_EQ(dut_->GetMaxElementsPerTransfer(), 1);
}

TEST_F(SingleSymbolWriteTest, GetElementSize) {
  // 11 bits padded to 16=2x8 bits, so 2 bytes
  EXPECT_EQ(dut_->GetElementSize(), 2);
}

TEST_F(SingleSymbolWriteTest, Write) {
  XLS_ASSERT_OK(dut_->Write(IDmaEndpoint::Payload{{0x01, 0x2}, false}));
  XLS_ASSERT_OK(dut_->Write(IDmaEndpoint::Payload{{0xAB, 0x7}, true}));
  auto popt = queue_->Read();
  EXPECT_FALSE(popt->empty());
  auto payload = *popt;
  auto payload_ref = Value::Tuple({
      Value(UBits(0, 2)),       // pad
      Value(UBits(0x201, 11)),  // data
      Value(UBits(0, 1)),       // tlast
      Value(UBits(0, 5)),       // pad
  });
  EXPECT_EQ(payload, payload_ref);
  popt = queue_->Read();
  EXPECT_FALSE(popt->empty());
  payload = *popt;
  payload_ref = Value::Tuple({
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
    auto stream = std::make_unique<IrAxiStreamLike>(
        IrAxiStreamLike::Make(queue_.get(), true, 1, 3, 2).value());
    dut_ = std::make_unique<AxiStreamLikeDmaEndpoint>(std::move(stream));
  }
  Package package_;
  TupleType* type_;
  Channel* channel_;
  std::unique_ptr<ChannelQueue> queue_;
  std::unique_ptr<AxiStreamLikeDmaEndpoint> dut_;
};

TEST_F(MultiSymbolReadTest, IsReadStream) {
  EXPECT_EQ(dut_->IsReadStream(), true);
}

TEST_F(MultiSymbolReadTest, GetMaxElementsPerTransfer) {
  EXPECT_EQ(dut_->GetMaxElementsPerTransfer(), 3);
}

TEST_F(MultiSymbolReadTest, GetElementSize) {
  // 11 bits padded to 16=2x8 bits, so 2 bytes
  EXPECT_EQ(dut_->GetElementSize(), 2);
}

TEST_F(MultiSymbolReadTest, Read) {
  EXPECT_FALSE(dut_->IsReady());
  // fill the queue
  XLS_ASSERT_OK(queue_->Write(Value::Tuple({
      Value(UBits(0, 2)),  // pad
      Value::Array({
                       Value(UBits(0x4CD, 11)),
                       Value(UBits(0x7AB, 11)),
                       Value(UBits(0x511, 11)),
                   })
          .value(),          // data
      Value(UBits(0x3, 3)),  // tkeep
      Value(UBits(0, 1)),    // tlast
      Value(UBits(0, 5)),    // pad
  })));
  EXPECT_TRUE(this->dut_->IsReady());
  XLS_ASSERT_OK(queue_->Write(Value::Tuple({
      Value(UBits(0, 2)),  // pad
      Value::Array({
                       Value(UBits(0x3AE, 11)),
                       Value(UBits(0x6BA, 11)),
                       Value(UBits(0x7CC, 11)),
                   })
          .value(),          // data
      Value(UBits(0x7, 3)),  // tkeep
      Value(UBits(1, 1)),    // tlast
      Value(UBits(0, 5)),    // pad
  })));
  EXPECT_TRUE(this->dut_->IsReady());
  // read out
  XLS_ASSERT_OK_AND_ASSIGN(auto payload, dut_->Read());
  ASSERT_THAT(payload.data,
              ElementsAre(0xCD, 0x04, 0xAB, 0x07));  // elements 0 and 1
  EXPECT_EQ(payload.last, false);
  EXPECT_TRUE(this->dut_->IsReady());
  XLS_ASSERT_OK_AND_ASSIGN(payload, dut_->Read());
  ASSERT_THAT(payload.data, ElementsAre(0xAE, 0x03, 0xBA, 0x06, 0xCC,
                                        0x07));  // elements 0,1,2
  EXPECT_EQ(payload.last, true);
  // check handling of underflow
  EXPECT_FALSE(this->dut_->IsReady());
  EXPECT_FALSE(dut_->Read().ok());
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
    auto stream = std::make_unique<IrAxiStreamLike>(
        IrAxiStreamLike::Make(queue_.get(), true, 1, 3, 2).value());
    dut_ = std::make_unique<AxiStreamLikeDmaEndpoint>(std::move(stream));
  }
  Package package_;
  TupleType* type_;
  Channel* channel_;
  std::unique_ptr<ChannelQueue> queue_;
  std::unique_ptr<AxiStreamLikeDmaEndpoint> dut_;
};

TEST_F(MultiSymbolWriteTest, IsReadStream) {
  EXPECT_EQ(dut_->IsReadStream(), false);
}

TEST_F(MultiSymbolWriteTest, GetMaxElementsPerTransfer) {
  EXPECT_EQ(dut_->GetMaxElementsPerTransfer(), 3);
}

TEST_F(MultiSymbolWriteTest, GetElementSize) {
  // 11 bits padded to 16=2x8 bits, so 2 bytes
  EXPECT_EQ(dut_->GetElementSize(), 2);
}

TEST_F(MultiSymbolWriteTest, Write) {
  XLS_ASSERT_OK(
      dut_->Write(IDmaEndpoint::Payload{{0xCD, 0x04, 0xAB, 0x07}, false}));
  XLS_ASSERT_OK(dut_->Write(
      IDmaEndpoint::Payload{{0xAE, 0x03, 0xBA, 0x06, 0xCC, 0x07}, true}));
  auto popt = queue_->Read();
  EXPECT_FALSE(popt->empty());
  auto payload = *popt;
  auto payload_ref = Value::Tuple({
      Value(UBits(0, 2)),  // pad
      Value::Array({
                       Value(UBits(0x4CD, 11)),
                       Value(UBits(0x7AB, 11)),
                       Value(UBits(0, 11)),
                   })
          .value(),          // data
      Value(UBits(0x3, 3)),  // tkeep
      Value(UBits(0, 1)),    // tlast
      Value(UBits(0, 5)),    // pad
  });
  EXPECT_EQ(payload, payload_ref);
  popt = queue_->Read();
  EXPECT_FALSE(popt->empty());
  payload = *popt;
  payload_ref = Value::Tuple({
      Value(UBits(0, 2)),  // pad
      Value::Array({
                       Value(UBits(0x3AE, 11)),
                       Value(UBits(0x6BA, 11)),
                       Value(UBits(0x7CC, 11)),
                   })
          .value(),          // data
      Value(UBits(0x7, 3)),  // tkeep
      Value(UBits(1, 1)),    // tlast
      Value(UBits(0, 5)),    // pad
  });
  EXPECT_EQ(payload, payload_ref);
}

}  // namespace
}  // namespace xls::simulation::generic
