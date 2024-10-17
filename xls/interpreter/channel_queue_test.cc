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

#include <memory>
#include <utility>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xls/common/status/matchers.h"
#include "xls/common/status/status_macros.h"
#include "xls/interpreter/channel_queue_test_base.h"
#include "xls/ir/channel.h"
#include "xls/ir/channel_ops.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/package.h"
#include "xls/ir/proc_elaboration.h"

namespace xls {
namespace {

using ::absl_testing::StatusIs;
using ::testing::HasSubstr;

// Instantiate and run all the tests in channel_queue_test_base.cc.
INSTANTIATE_TEST_SUITE_P(ChannelQueueTest, ChannelQueueTestBase,
                         testing::Values(ChannelQueueTestParam(
                             [](ChannelInstance* channel_instance) {
                               return std::make_unique<ChannelQueue>(
                                   channel_instance);
                             })));

// Separate tests for queue managers.
class ChannelQueueManagerTest : public IrTestBase {
 protected:
  absl::StatusOr<std::unique_ptr<ChannelQueueManager>> MakeChannelQueueManager(
      Package* package, absl::Span<Channel* const> channels) {
    XLS_ASSIGN_OR_RETURN(ProcElaboration elaboration,
                         ProcElaboration::ElaborateOldStylePackage(package));
    std::vector<std::unique_ptr<ChannelQueue>> queues;
    for (Channel* channel : channels) {
      XLS_ASSIGN_OR_RETURN(ChannelInstance * instance,
                           elaboration.GetUniqueInstance(channel));
      queues.push_back(std::make_unique<ChannelQueue>(instance));
    }
    return ChannelQueueManager::Create(std::move(queues),
                                       std::move(elaboration));
  }
};

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

  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<ChannelQueueManager> manager,
                           ChannelQueueManager::Create(&package));

  EXPECT_EQ(manager->queues().size(), 3);
  EXPECT_EQ(manager
                ->GetQueue(
                    manager->elaboration().GetUniqueInstance(channel_a).value())
                .channel(),
            channel_a);
  EXPECT_EQ(manager
                ->GetQueue(
                    manager->elaboration().GetUniqueInstance(channel_b).value())
                .channel(),
            channel_b);
  EXPECT_EQ(manager
                ->GetQueue(
                    manager->elaboration().GetUniqueInstance(channel_c).value())
                .channel(),
            channel_c);
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
    XLS_ASSERT_OK_AND_ASSIGN(
        ProcElaboration elaboration,
        ProcElaboration::ElaborateOldStylePackage(&package));
    std::vector<std::unique_ptr<ChannelQueue>> queues;
    queues.push_back(std::make_unique<ChannelQueue>(
        elaboration.GetUniqueInstance(channel_a).value()));
    queues.push_back(std::make_unique<ChannelQueue>(
        elaboration.GetUniqueInstance(channel_b).value()));
    queues.push_back(std::make_unique<ChannelQueue>(
        elaboration.GetUniqueInstance(channel_c).value()));
    queues.push_back(std::make_unique<ChannelQueue>(
        elaboration.GetUniqueInstance(channel_a).value()));
    EXPECT_THAT(
        ChannelQueueManager::Create(std::move(queues), std::move(elaboration))
            .status(),
        StatusIs(
            absl::StatusCode::kInvalidArgument,
            HasSubstr("Multiple queues specified for channel instance `a`")));
  }

  {
    XLS_ASSERT_OK_AND_ASSIGN(
        ProcElaboration elaboration,
        ProcElaboration::ElaborateOldStylePackage(&package));
    std::vector<std::unique_ptr<ChannelQueue>> queues;
    queues.push_back(std::make_unique<ChannelQueue>(
        elaboration.GetUniqueInstance(channel_a).value()));
    queues.push_back(std::make_unique<ChannelQueue>(
        elaboration.GetUniqueInstance(channel_b).value()));
    EXPECT_THAT(
        ChannelQueueManager::Create(std::move(queues), std::move(elaboration))
            .status(),
        StatusIs(absl::StatusCode::kInvalidArgument,
                 HasSubstr("No queue specified for channel instance `c`")));
  }
}

}  // namespace
}  // namespace xls
