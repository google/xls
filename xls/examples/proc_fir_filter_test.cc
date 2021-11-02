// Copyright 2021 The XLS Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// See the License for the specific language governing permissions and
// limitations under the License.

#include "xls/examples/proc_fir_filter.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xls/common/status/matchers.h"
#include "xls/common/status/status_macros.h"
#include "xls/common/logging/logging.h"
#include "xls/ir/bits.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/ir_matcher.h"
#include "xls/ir/ir_scanner.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/package.h"
#include "xls/interpreter/proc_interpreter.h"
#include "xls/interpreter/channel_queue.h"

namespace xls {

namespace {

using status_testing::IsOkAndHolds;

class ProcFirFilterTest : public IrTestBase {
 protected:
  ProcFirFilterTest() = default;
};

// Test a FIR filter with kernel = {1, 2}.
TEST_F(ProcFirFilterTest, FIRSimpleTest) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Value kernel_value, Value::UBitsArray({1, 2}, 32));
  // ProcFirFilter pff;
  absl::string_view name = "fir_proc";
  Type* kernel_type = p->GetTypeForValue(kernel_value.element(0));

  XLS_ASSERT_OK_AND_ASSIGN(StreamingChannel* x_in,
                           p->CreateStreamingChannel(
                           absl::StrFormat("%s_x_in", name),
                           ChannelOps::kReceiveOnly, kernel_type));

  XLS_ASSERT_OK_AND_ASSIGN(StreamingChannel* filter_out,
                           p->CreateStreamingChannel(
                           absl::StrFormat("%s_out", name),
                           ChannelOps::kSendOnly, kernel_type));

  XLS_ASSERT_OK_AND_ASSIGN(Proc* fir_proc, CreateFirFilter(name,
                                                               kernel_value,
                                                               x_in,
                                                               filter_out,
                                                               p.get()));

  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<ChannelQueueManager> queue_manager,
      ChannelQueueManager::Create(/*user_defined_queues=*/{}, p.get()));
  ProcInterpreter pi(fir_proc, queue_manager.get());

  XLS_ASSERT_OK_AND_ASSIGN(Channel* send,
                           p.get()->GetChannel(
                               absl::StrFormat("%s_out", name)));
  XLS_ASSERT_OK_AND_ASSIGN(Channel* recv,
                           p.get()->GetChannel(
                               absl::StrFormat("%s_x_in", name)));

  ChannelQueue& send_queue = queue_manager->GetQueue(send);
  ChannelQueue& recv_queue = queue_manager->GetQueue(recv);

  ASSERT_TRUE(send_queue.empty());
  ASSERT_TRUE(recv_queue.empty());

  XLS_ASSERT_OK(recv_queue.Enqueue({Value(UBits(0, 32))}));

  ASSERT_THAT(
      pi.RunIterationUntilCompleteOrBlocked(),
      IsOkAndHolds(ProcInterpreter::RunResult{.iteration_complete = true,
                                              .progress_made = true,
                                              .blocked_channels = {}}));


  EXPECT_TRUE(pi.IsIterationComplete());
  EXPECT_EQ(send_queue.size(), 1);
  EXPECT_FALSE(send_queue.empty());
  EXPECT_THAT(send_queue.Dequeue(), IsOkAndHolds(Value(UBits(0, 32))));
  EXPECT_EQ(send_queue.size(), 0);
  EXPECT_TRUE(send_queue.empty());

  XLS_ASSERT_OK(recv_queue.Enqueue({Value(UBits(64, 32))}));

  ASSERT_THAT(
      pi.RunIterationUntilCompleteOrBlocked(),
      IsOkAndHolds(ProcInterpreter::RunResult{.iteration_complete = true,
                                              .progress_made = true,
                                              .blocked_channels = {}}));

  XLS_ASSERT_OK(recv_queue.Enqueue({Value(UBits(128, 32))}));

  ASSERT_THAT(
      pi.RunIterationUntilCompleteOrBlocked(),
      IsOkAndHolds(ProcInterpreter::RunResult{.iteration_complete = true,
                                              .progress_made = true,
                                              .blocked_channels = {}}));

  XLS_ASSERT_OK(recv_queue.Enqueue({Value(UBits(256, 32))}));

  ASSERT_THAT(
      pi.RunIterationUntilCompleteOrBlocked(),
      IsOkAndHolds(ProcInterpreter::RunResult{.iteration_complete = true,
                                              .progress_made = true,
                                              .blocked_channels = {}}));


  EXPECT_EQ(send_queue.size(), 3);

  EXPECT_THAT(send_queue.Dequeue(), IsOkAndHolds(Value(UBits(64, 32))));
  EXPECT_THAT(send_queue.Dequeue(), IsOkAndHolds(Value(UBits(256, 32))));
  EXPECT_THAT(send_queue.Dequeue(), IsOkAndHolds(Value(UBits(512, 32))));
}

// Test FIR filter with accumulator kernel = {1, 10, 100, 1000, 10000, 100000}.
TEST_F(ProcFirFilterTest, FIRAccumulator) {
    auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Value kernel_value,
                           Value::UBitsArray({1, 10, 100, 1000, 10000, 100000},
                                             32));
  // ProcFirFilter pff;
  absl::string_view name = "fir_proc";
  Type* kernel_type = p->GetTypeForValue(kernel_value.element(0));

  XLS_ASSERT_OK_AND_ASSIGN(StreamingChannel* x_in,
                           p->CreateStreamingChannel(
                           absl::StrFormat("%s_x_in", name),
                           ChannelOps::kReceiveOnly, kernel_type));

  XLS_ASSERT_OK_AND_ASSIGN(StreamingChannel* filter_out,
                           p->CreateStreamingChannel(
                           absl::StrFormat("%s_out", name),
                           ChannelOps::kSendOnly, kernel_type));

  XLS_ASSERT_OK_AND_ASSIGN(Proc* fir_proc, CreateFirFilter(name,
                                                               kernel_value,
                                                               x_in,
                                                               filter_out,
                                                               p.get()));

  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<ChannelQueueManager> queue_manager,
      ChannelQueueManager::Create(/*user_defined_queues=*/{}, p.get()));
  ProcInterpreter pi(fir_proc, queue_manager.get());

  XLS_ASSERT_OK_AND_ASSIGN(Channel* send,
                           p.get()->GetChannel(
                               absl::StrFormat("%s_out", name)));
  XLS_ASSERT_OK_AND_ASSIGN(Channel* recv,
                           p.get()->GetChannel(
                               absl::StrFormat("%s_x_in", name)));

  ChannelQueue& send_queue = queue_manager->GetQueue(send);
  ChannelQueue& recv_queue = queue_manager->GetQueue(recv);

  ASSERT_TRUE(send_queue.empty());
  ASSERT_TRUE(recv_queue.empty());

  for (int idx = 1; idx < 8; idx++) {
    XLS_ASSERT_OK(recv_queue.Enqueue({Value(UBits(idx, 32))}));

    ASSERT_THAT(
      pi.RunIterationUntilCompleteOrBlocked(),
      IsOkAndHolds(ProcInterpreter::RunResult{.iteration_complete = true,
                                              .progress_made = true,
                                              .blocked_channels = {}}));
  }

  EXPECT_EQ(send_queue.size(), 7);

  EXPECT_THAT(send_queue.Dequeue(), IsOkAndHolds(Value(UBits(1, 32))));
  EXPECT_THAT(send_queue.Dequeue(), IsOkAndHolds(Value(UBits(12, 32))));
  EXPECT_THAT(send_queue.Dequeue(), IsOkAndHolds(Value(UBits(123, 32))));
  EXPECT_THAT(send_queue.Dequeue(), IsOkAndHolds(Value(UBits(1234, 32))));
  EXPECT_THAT(send_queue.Dequeue(), IsOkAndHolds(Value(UBits(12345, 32))));
  EXPECT_THAT(send_queue.Dequeue(), IsOkAndHolds(Value(UBits(123456, 32))));
  EXPECT_THAT(send_queue.Dequeue(), IsOkAndHolds(Value(UBits(234567, 32))));
}

// Test a FIR filter with single element kernel = {2}
// TODO (kmanav) 2021-10-20: Allow for a single element kernel.

TEST_F(ProcFirFilterTest, DISABLED_FIRScaleFactor) {
    auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Value kernel_value,
                           Value::UBitsArray({2}, 32));
  // ProcFirFilter pff;
  absl::string_view name = "fir_proc";
  Type* kernel_type = p->GetTypeForValue(kernel_value.element(0));

  XLS_ASSERT_OK_AND_ASSIGN(StreamingChannel* x_in,
                           p->CreateStreamingChannel(
                           absl::StrFormat("%s_x_in", name),
                           ChannelOps::kReceiveOnly, kernel_type));

  XLS_ASSERT_OK_AND_ASSIGN(StreamingChannel* filter_out,
                           p->CreateStreamingChannel(
                           absl::StrFormat("%s_out", name),
                           ChannelOps::kSendOnly, kernel_type));

  XLS_ASSERT_OK_AND_ASSIGN(Proc* fir_proc, CreateFirFilter(name,
                                                               kernel_value,
                                                               x_in,
                                                               filter_out,
                                                               p.get()));

  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<ChannelQueueManager> queue_manager,
      ChannelQueueManager::Create(/*user_defined_queues=*/{}, p.get()));
  ProcInterpreter pi(fir_proc, queue_manager.get());

  XLS_ASSERT_OK_AND_ASSIGN(Channel* send,
                           p.get()->GetChannel(
                               absl::StrFormat("%s_out", name)));
  XLS_ASSERT_OK_AND_ASSIGN(Channel* recv,
                           p.get()->GetChannel(
                               absl::StrFormat("%s_x_in", name)));

  ChannelQueue& send_queue = queue_manager->GetQueue(send);
  ChannelQueue& recv_queue = queue_manager->GetQueue(recv);

  ASSERT_TRUE(send_queue.empty());
  ASSERT_TRUE(recv_queue.empty());

  for (int idx = 0; idx < 6; idx++) {
    XLS_ASSERT_OK(recv_queue.Enqueue({Value(UBits(idx, 32))}));

    ASSERT_THAT(
      pi.RunIterationUntilCompleteOrBlocked(),
      IsOkAndHolds(ProcInterpreter::RunResult{.iteration_complete = true,
                                              .progress_made = true,
                                              .blocked_channels = {}}));
  }

  EXPECT_EQ(send_queue.size(), 6);

  EXPECT_THAT(send_queue.Dequeue(), IsOkAndHolds(Value(UBits(0, 32))));
  EXPECT_THAT(send_queue.Dequeue(), IsOkAndHolds(Value(UBits(0, 32))));
  EXPECT_THAT(send_queue.Dequeue(), IsOkAndHolds(Value(UBits(2, 32))));
  EXPECT_THAT(send_queue.Dequeue(), IsOkAndHolds(Value(UBits(4, 32))));
  EXPECT_THAT(send_queue.Dequeue(), IsOkAndHolds(Value(UBits(6, 32))));
  EXPECT_THAT(send_queue.Dequeue(), IsOkAndHolds(Value(UBits(8, 32))));
}

// Compute a triangular blur.
// In reality this has to be normalized, and wouldn't be made of integers.
TEST_F(ProcFirFilterTest, FIRTriangularBlur) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Value kernel_value,
                           Value::UBitsArray({1, 3, 5, 3, 1}, 32));
  // ProcFirFilter pff;
  absl::string_view name = "fir_proc";
  Type* kernel_type = p->GetTypeForValue(kernel_value.element(0));

  XLS_ASSERT_OK_AND_ASSIGN(StreamingChannel* x_in,
                           p->CreateStreamingChannel(
                           absl::StrFormat("%s_x_in", name),
                           ChannelOps::kReceiveOnly, kernel_type));

  XLS_ASSERT_OK_AND_ASSIGN(StreamingChannel* filter_out,
                           p->CreateStreamingChannel(
                           absl::StrFormat("%s_out", name),
                           ChannelOps::kSendOnly, kernel_type));

  XLS_ASSERT_OK_AND_ASSIGN(Proc* fir_proc, CreateFirFilter(name,
                                                               kernel_value,
                                                               x_in,
                                                               filter_out,
                                                               p.get()));

  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<ChannelQueueManager> queue_manager,
      ChannelQueueManager::Create(/*user_defined_queues=*/{}, p.get()));
  ProcInterpreter pi(fir_proc, queue_manager.get());

  XLS_ASSERT_OK_AND_ASSIGN(Channel* send,
                           p.get()->GetChannel(
                               absl::StrFormat("%s_out", name)));
  XLS_ASSERT_OK_AND_ASSIGN(Channel* recv,
                           p.get()->GetChannel(
                               absl::StrFormat("%s_x_in", name)));

  ChannelQueue& send_queue = queue_manager->GetQueue(send);
  ChannelQueue& recv_queue = queue_manager->GetQueue(recv);

  ASSERT_TRUE(send_queue.empty());
  ASSERT_TRUE(recv_queue.empty());

  std::vector<int> values = {2, 10, 4, 3, 8, 20, 9, 3};

  for (int idx = 0; idx < 8; idx++) {
    XLS_ASSERT_OK(recv_queue.Enqueue({Value(UBits(values[idx], 32))}));

    ASSERT_THAT(
      pi.RunIterationUntilCompleteOrBlocked(),
      IsOkAndHolds(ProcInterpreter::RunResult{.iteration_complete = true,
                                              .progress_made = true,
                                              .blocked_channels = {}}));
  }

  EXPECT_EQ(send_queue.size(), 8);

  EXPECT_THAT(send_queue.Dequeue(), IsOkAndHolds(Value(UBits(2, 32))));
  EXPECT_THAT(send_queue.Dequeue(), IsOkAndHolds(Value(UBits(16, 32))));
  EXPECT_THAT(send_queue.Dequeue(), IsOkAndHolds(Value(UBits(44, 32))));
  EXPECT_THAT(send_queue.Dequeue(), IsOkAndHolds(Value(UBits(71, 32))));
  EXPECT_THAT(send_queue.Dequeue(), IsOkAndHolds(Value(UBits(69, 32))));
  EXPECT_THAT(send_queue.Dequeue(), IsOkAndHolds(Value(UBits(81, 32))));
  EXPECT_THAT(send_queue.Dequeue(), IsOkAndHolds(Value(UBits(122, 32))));
  EXPECT_THAT(send_queue.Dequeue(), IsOkAndHolds(Value(UBits(157, 32))));
}

}  // namespace

}  // namespace xls
