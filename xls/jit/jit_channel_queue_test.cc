// Copyright 2020 Google LLC
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
#include "xls/jit/jit_channel_queue.h"

#include <algorithm>
#include <cstdint>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace xls {
namespace {

using testing::HasSubstr;

template <typename T>
class FifoJitChannelQueueTypedTest : public ::testing::Test {};

TYPED_TEST_SUITE_P(FifoJitChannelQueueTypedTest);

TYPED_TEST_P(FifoJitChannelQueueTypedTest, BasicAccess) {
  TypeParam fifo(0);
  EXPECT_EQ(fifo.channel_id(), 0);
  EXPECT_TRUE(fifo.Empty());
  std::vector<uint8_t> send_buffer(1);
  std::vector<uint8_t> recv_buffer(1);
  // Send and receive immediately.
  for (int64_t i = 0; i < 10; i++) {
    send_buffer[0] = i;
    fifo.Send(send_buffer.data(), send_buffer.size());
    EXPECT_FALSE(fifo.Empty());
    EXPECT_TRUE(fifo.Recv(recv_buffer.data(), recv_buffer.size()));
    EXPECT_THAT(recv_buffer[0], i);
    EXPECT_TRUE(fifo.Empty());
  }
  // Send then receive.
  for (int64_t i = 0; i < 10; i++) {
    send_buffer[0] = i;
    fifo.Send(send_buffer.data(), send_buffer.size());
  }
  for (int64_t i = 0; i < 10; i++) {
    EXPECT_TRUE(fifo.Recv(recv_buffer.data(), recv_buffer.size()));
    EXPECT_THAT(recv_buffer[0], i);
  }
  EXPECT_TRUE(fifo.Empty());
}

REGISTER_TYPED_TEST_SUITE_P(FifoJitChannelQueueTypedTest, BasicAccess);

using FifoTypes =
    ::testing::Types<FifoJitChannelQueue, LocklessFifoJitChannelQueue>;
INSTANTIATE_TYPED_TEST_SUITE_P(FifoJitChannelQueue,
                               FifoJitChannelQueueTypedTest, FifoTypes);

TEST(JitChannelQueue, SingleValueJitChannelQueue) {
  SingleValueJitChannelQueue channel(0);
  EXPECT_EQ(channel.channel_id(), 0);
  EXPECT_TRUE(channel.Empty());
  std::vector<uint8_t> send_buffer(1);
  std::vector<uint8_t> recv_buffer(1);
  // Send once and receive twice.
  send_buffer[0] = 42;
  channel.Send(send_buffer.data(), send_buffer.size());
  EXPECT_FALSE(channel.Empty());
  EXPECT_TRUE(channel.Recv(recv_buffer.data(), recv_buffer.size()));
  EXPECT_THAT(recv_buffer[0], 42);
  EXPECT_FALSE(channel.Empty());
  EXPECT_TRUE(channel.Recv(recv_buffer.data(), recv_buffer.size()));
  EXPECT_THAT(recv_buffer[0], 42);
  EXPECT_FALSE(channel.Empty());
  // Send a new value and receive the updated value.
  send_buffer[0] = 64;
  channel.Send(send_buffer.data(), send_buffer.size());
  EXPECT_FALSE(channel.Empty());
  EXPECT_TRUE(channel.Recv(recv_buffer.data(), recv_buffer.size()));
  EXPECT_THAT(recv_buffer[0], 64);
  EXPECT_FALSE(channel.Empty());
}

}  // namespace
}  // namespace xls
