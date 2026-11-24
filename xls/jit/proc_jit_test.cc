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

#include "xls/jit/proc_jit.h"

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/log/check.h"
#include "absl/types/span.h"
#include "xls/common/status/matchers.h"
#include "xls/interpreter/channel_queue.h"
#include "xls/interpreter/evaluator_options.h"
#include "xls/interpreter/proc_evaluator.h"
#include "xls/interpreter/proc_evaluator_test_base.h"
#include "xls/ir/bits.h"
#include "xls/ir/channel.h"
#include "xls/ir/channel_ops.h"
#include "xls/ir/events.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/node.h"
#include "xls/ir/package.h"
#include "xls/ir/proc.h"
#include "xls/ir/proc_elaboration.h"
#include "xls/ir/topo_sort.h"
#include "xls/ir/value.h"
#include "xls/jit/function_base_jit.h"
#include "xls/jit/jit_callbacks.h"
#include "xls/jit/jit_channel_queue.h"
#include "xls/jit/jit_evaluator_options.h"
#include "xls/jit/jit_runtime.h"
#include "xls/jit/orc_jit.h"

namespace xls {
namespace {

JitRuntime* GetJitRuntime() {
  static auto orc_jit = OrcJit::Create().value();
  static auto jit_runtime = std::make_unique<JitRuntime>(
      orc_jit->CreateDataLayout().value(), *orc_jit->GetContext());
  return jit_runtime.get();
}

template <bool kWithObserver>
std::unique_ptr<ProcEvaluator> EvaluatorFromProc(
    Proc* proc, ChannelQueueManager* queue_manager) {
  JitChannelQueueManager* jit_queue_manager =
      dynamic_cast<JitChannelQueueManager*>(queue_manager);
  CHECK(jit_queue_manager != nullptr);
  return ProcJit::Create(proc, GetJitRuntime(), jit_queue_manager,
                         EvaluatorOptions(),
                         JitEvaluatorOptions().set_include_observer_callbacks(
                             kWithObserver))
      .value();
}

std::unique_ptr<ChannelQueueManager> QueueManagerForPackage(Package* package) {
  return JitChannelQueueManager::CreateThreadSafe(
             package, std::make_unique<JitRuntime>(
                          GetJitRuntime()->target_layout_info()))
      .value();
}
// Instantiate and run all the tests in proc_evaluator_test_base.cc.
INSTANTIATE_TEST_SUITE_P(
    ProcJitTest, ProcEvaluatorTestBase,
    testing::Values(ProcEvaluatorTestParam(EvaluatorFromProc<false>,
                                           QueueManagerForPackage,
                                           /*supports_observers=*/false),
                    ProcEvaluatorTestParam(EvaluatorFromProc<true>,
                                           QueueManagerForPackage,
                                           /*supports_observers=*/true)));

uint8_t* MisalignedDataPointer(std::vector<uint8_t>& storage) {
  uint8_t* base = storage.data();
  return (reinterpret_cast<uintptr_t>(base) % 2 == 0) ? (base + 1) : base;
}

TEST(ProcJitUnalignedTest, PreservesOutputsAcrossMidTickContinuation) {
  Package package("unaligned_proc_continuation");
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * ch_sync,
      package.CreateStreamingChannel("sync", ChannelOps::kReceiveOnly,
                                     package.GetBitsType(1)));

  ProcBuilder pb("unaligned_continuation_proc", &package);
  BStateElement written = pb.StateElement("written", Value(UBits(10, 32)));
  BStateElement unchanged =
      pb.StateElement("unchanged", Value(UBits(0x123456789abcdef0ULL, 64)));
  pb.StateElement("tok_state", Value::Token());

  // All 0-user sinks appear at the tail of `TopoSort(proc)` in creation order:
  // `next_written` runs before `sync_recv` suspends, and `post_sync_read` runs
  // after `sync_recv` resumes.
  BValue next_written = pb.Next(written, pb.Literal(UBits(99, 32)));
  BValue sync_recv = pb.Receive(ch_sync, pb.Literal(Value::Token()));
  BValue post_sync_read = pb.StateRead(written);
  pb.StateRead(unchanged);

  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, pb.Build());
  XLS_ASSERT_OK(package.SetTop(proc));

  XLS_ASSERT_OK_AND_ASSIGN(std::vector<Node*> topo_order, TopoSort(proc));
  ASSERT_THAT(topo_order,
              ContainsSubsequence({next_written.node(), sync_recv.node(),
                                   post_sync_read.node()}));

  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<OrcJit> orc_jit, OrcJit::Create());
  XLS_ASSERT_OK_AND_ASSIGN(
      JittedFunctionBase jitted_fn,
      JittedFunctionBase::Build(proc, *orc_jit, EvaluatorOptions()));
  ASSERT_EQ(jitted_fn.GetOutputBufferMetadata().size(), 3);
  ASSERT_GT(jitted_fn.GetOutputBufferMetadata()[1].abi_alignment, 1);
  ASSERT_EQ(jitted_fn.GetOutputBufferMetadata()[2].size, 0);

  XLS_ASSERT_OK_AND_ASSIGN(auto data_layout, orc_jit->CreateDataLayout());
  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<JitChannelQueueManager> queue_mgr,
      JitChannelQueueManager::CreateThreadSafe(
          &package, std::make_unique<JitRuntime>(data_layout)));
  XLS_ASSERT_OK_AND_ASSIGN(ProcInstance * proc_instance,
                           queue_mgr->elaboration().GetUniqueInstance(proc));
  std::vector<JitChannelQueue*> queues(jitted_fn.queue_indices().size());
  for (const auto& [channel_name, index] : jitted_fn.queue_indices()) {
    XLS_ASSERT_OK_AND_ASSIGN(Channel * ch, package.GetChannel(channel_name));
    XLS_ASSERT_OK_AND_ASSIGN(ChannelInstance * ch_inst,
                             queue_mgr->elaboration().GetUniqueInstance(ch));
    queues[index] = &queue_mgr->GetJitQueue(ch_inst);
  }
  InstanceContext instance_ctx =
      InstanceContext::CreateForProc(proc_instance, std::move(queues));
  JitRuntime& runtime = queue_mgr->runtime();

  const int64_t size0 = jitted_fn.GetInputBufferMetadata()[0].size;
  const int64_t size1 = jitted_fn.GetInputBufferMetadata()[1].size;
  std::vector<uint8_t> in0_buf(size0 + 8, 0);
  std::vector<uint8_t> in1_buf(size1 + 8, 0);
  std::vector<uint8_t> out0_buf(size0 + 8, 0xaa);
  std::vector<uint8_t> out1_buf(size1 + 8, 0xaa);
  std::vector<uint8_t> temp_buf(jitted_fn.temp_buffer_size() + 8, 0);

  uint8_t* in0_ptr = MisalignedDataPointer(in0_buf);
  uint8_t* in1_ptr = MisalignedDataPointer(in1_buf);
  uint8_t* out0_ptr = MisalignedDataPointer(out0_buf);
  uint8_t* out1_ptr = MisalignedDataPointer(out1_buf);
  uint8_t* temp_ptr = MisalignedDataPointer(temp_buf);

  runtime.BlitValueToBuffer(Value(UBits(10, 32)),
                            proc->GetStateElement(0)->type(),
                            absl::MakeSpan(in0_ptr, size0));
  runtime.BlitValueToBuffer(Value(UBits(0x123456789abcdef0ULL, 64)),
                            proc->GetStateElement(1)->type(),
                            absl::MakeSpan(in1_ptr, size1));

  // Pass `nullptr` for the 0-byte token state element to exercise the zero-size
  // `memcpy` guards alongside misaligned non-empty buffers.
  std::vector<const uint8_t*> input_ptrs = {in0_ptr, in1_ptr, nullptr};
  std::vector<uint8_t*> output_ptrs = {out0_ptr, out1_ptr, nullptr};
  InterpreterEvents events;

  // Partition 0 executes `next_written` (`99`), initializes `unchanged`, and
  // suspends on `sync_recv`.
  int64_t continuation = jitted_fn.RunUnalignedJittedFunction(
      input_ptrs.data(), output_ptrs.data(), temp_ptr, &events, &instance_ctx,
      &runtime, /*continuation=*/0);
  ASSERT_GT(continuation, 0);
  EXPECT_EQ(runtime.UnpackBuffer(out0_ptr, proc->GetStateElement(0)->type()),
            Value(UBits(99, 32)));
  EXPECT_EQ(runtime.UnpackBuffer(out1_ptr, proc->GetStateElement(1)->type()),
            Value(UBits(0x123456789abcdef0ULL, 64)));

  // Unblock `sync_recv` and resume from `continuation`: prior writes and
  // unchanged state elements in `output_ptrs` must survive unaligned bounce
  // buffering and `post_sync_read`.
  XLS_ASSERT_OK_AND_ASSIGN(ChannelQueue * sync_queue,
                           queue_mgr->GetQueueByName("sync", proc->name()));
  XLS_ASSERT_OK(sync_queue->Write(Value(UBits(1, 1))));

  int64_t next_continuation = jitted_fn.RunUnalignedJittedFunction(
      input_ptrs.data(), output_ptrs.data(), temp_ptr, &events, &instance_ctx,
      &runtime, continuation);
  EXPECT_EQ(next_continuation, 0);
  EXPECT_EQ(runtime.UnpackBuffer(out0_ptr, proc->GetStateElement(0)->type()),
            Value(UBits(99, 32)));
  EXPECT_EQ(runtime.UnpackBuffer(out1_ptr, proc->GetStateElement(1)->type()),
            Value(UBits(0x123456789abcdef0ULL, 64)));
}

}  // namespace
}  // namespace xls
