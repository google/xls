// Copyright 2022 The XLS Authors
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

#include "xls/interpreter/channel_queue.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "xls/common/status/matchers.h"
#include "xls/interpreter/channel_queue_test_base.h"
#include "xls/ir/channel.h"
#include "xls/ir/package.h"

namespace xls {
namespace {

using status_testing::StatusIs;
using ::testing::HasSubstr;

// Instantiate and run all the tests in channel_queue_test_base.cc.
INSTANTIATE_TEST_SUITE_P(
    ChannelQueueTest, ChannelQueueTestBase,
    testing::Values(ChannelQueueTestParam([](Channel* channel) {
      return std::make_unique<ChannelQueue>(channel);
    })));

// Separate tests for queue managers.
class ChannelQueueManagerTest : public IrTestBase {};

TEST_F(ChannelQueueManagerTest, SimpleChannelQueueManager) {
  Package package(TestName());
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * channel_a,
      package.CreateStreamingChannel("a", ChannelOps::kReceiveOnly,
                                     package.GetBitsType(32)));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * channel_b,
      package.CreateStreamingChannel("b", ChannelOps::kReceiveOnly,
                                     package.GetBitsType(32)));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * channel_c,
      package.CreateStreamingChannel("c", ChannelOps::kSendReceive,
                                     package.GetBitsType(32)));

  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<ChannelQueueManager> manager,
                           ChannelQueueManager::Create(&package));
  EXPECT_EQ(manager->queues().size(), 3);
  EXPECT_EQ(manager->GetQueue(channel_a).channel(), channel_a);
  EXPECT_EQ(manager->GetQueue(channel_b).channel(), channel_b);
  EXPECT_EQ(manager->GetQueue(channel_c).channel(), channel_c);
}

TEST_F(ChannelQueueManagerTest, ChannelQueueManagerNoChannels) {
  Package package(TestName());
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<ChannelQueueManager> manager,
                           ChannelQueueManager::Create(&package));
  EXPECT_EQ(manager->queues().size(), 0);
}

TEST_F(ChannelQueueManagerTest, ChannelQueueManagerCustomQueues) {
  Package package(TestName());
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * channel_a,
      package.CreateStreamingChannel("a", ChannelOps::kReceiveOnly,
                                     package.GetBitsType(32)));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * channel_b,
      package.CreateStreamingChannel("b", ChannelOps::kReceiveOnly,
                                     package.GetBitsType(32)));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * channel_c,
      package.CreateStreamingChannel("c", ChannelOps::kSendReceive,
                                     package.GetBitsType(32)));

  std::vector<std::unique_ptr<ChannelQueue>> queues;
  queues.push_back(std::make_unique<ChannelQueue>(channel_a));
  ChannelQueue* queue_a = queues.back().get();
  queues.push_back(std::make_unique<ChannelQueue>(channel_b));
  ChannelQueue* queue_b = queues.back().get();
  queues.push_back(std::make_unique<ChannelQueue>(channel_c));
  ChannelQueue* queue_c = queues.back().get();

  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<ChannelQueueManager> manager,
      ChannelQueueManager::Create(std::move(queues), &package));

  EXPECT_EQ(manager->queues().size(), 3);
  EXPECT_EQ(&manager->GetQueue(channel_a), queue_a);
  EXPECT_EQ(&manager->GetQueue(channel_b), queue_b);
  EXPECT_EQ(&manager->GetQueue(channel_c), queue_c);
}

TEST_F(ChannelQueueManagerTest, ChannelQueueManagerErrorConditions) {
  Package package(TestName());
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * channel_a,
      package.CreateStreamingChannel("a", ChannelOps::kReceiveOnly,
                                     package.GetBitsType(32)));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * channel_b,
      package.CreateStreamingChannel("b", ChannelOps::kReceiveOnly,
                                     package.GetBitsType(32)));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * channel_c,
      package.CreateStreamingChannel("c", ChannelOps::kSendReceive,
                                     package.GetBitsType(32)));

  {
    std::vector<std::unique_ptr<ChannelQueue>> queues;
    queues.push_back(std::make_unique<ChannelQueue>(channel_a));
    queues.push_back(std::make_unique<ChannelQueue>(channel_b));
    queues.push_back(std::make_unique<ChannelQueue>(channel_c));
    queues.push_back(std::make_unique<ChannelQueue>(channel_a));
    EXPECT_THAT(
        ChannelQueueManager::Create(std::move(queues), &package).status(),
        StatusIs(absl::StatusCode::kInvalidArgument,
                 HasSubstr("Multiple queues specified for channel `a`")));
  }

  {
    std::vector<std::unique_ptr<ChannelQueue>> queues;
    queues.push_back(std::make_unique<ChannelQueue>(channel_a));
    queues.push_back(std::make_unique<ChannelQueue>(channel_b));
    EXPECT_THAT(
        ChannelQueueManager::Create(std::move(queues), &package).status(),
        StatusIs(absl::StatusCode::kInvalidArgument,
                 HasSubstr("No queue specified for channel `c`")));
  }
}

}  // namespace
}  // namespace xls
