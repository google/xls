// Copyright 2026 The XLS Authors
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

#include "xls/tools/eval_proc.h"

#include <memory>
#include <random>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status_matchers.h"
#include "xls/common/status/matchers.h"
#include "xls/interpreter/channel_queue.h"
#include "xls/interpreter/serial_proc_runtime.h"
#include "xls/ir/bits.h"
#include "xls/ir/channel.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/package.h"
#include "xls/ir/proc.h"
#include "xls/ir/value.h"

namespace xls {
namespace {

using ::absl_testing::IsOk;

TEST(EvalProcTest, AttachRandomInputGeneratorsAndDrainOutputsAcrossRuntimes) {
  Package pkg("proc_scoped_pkg");
  ProcBuilder pb(NewStyleProc(), "accum", &pkg);
  BReceiveChannel in_stream = pb.AddInputChannel(
      "in_stream", pkg.GetBitsType(8), ChannelKind::kStreaming);
  BReceiveChannel in_cfg = pb.AddInputChannel("in_cfg", pkg.GetBitsType(8),
                                              ChannelKind::kSingleValue);
  BSendChannel out_stream = pb.AddOutputChannel(
      "out_stream", pkg.GetBitsType(8), ChannelKind::kStreaming);

  BValue state = pb.ReadStateElement("acc", UBits(0, 8));
  BValue recv_s = pb.Receive(in_stream, pb.Literal(Value::Token()));
  BValue recv_c = pb.Receive(in_cfg, pb.TupleIndex(recv_s, 0));
  BValue sum =
      pb.Add(pb.Add(state, pb.TupleIndex(recv_s, 1)), pb.TupleIndex(recv_c, 1));
  pb.Send(out_stream, pb.TupleIndex(recv_c, 0), sum);
  pb.Next(state, sum);
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, pb.Build());
  ASSERT_THAT(pkg.SetTop(proc), IsOk());

  for (bool use_jit : {false, true}) {
    XLS_ASSERT_OK_AND_ASSIGN(
        std::unique_ptr<SerialProcRuntime> runtime,
        CreateSerialProcRuntime(proc, {.use_jit = use_jit}));
    std::minstd_rand rng(12345);
    ASSERT_THAT(AttachRandomInputGenerators(*runtime, rng, /*pool_size=*/8),
                IsOk());

    for (int i = 0; i < 20; ++i) {
      ASSERT_THAT(runtime->Tick(), IsOk());
    }
    XLS_ASSERT_OK_AND_ASSIGN(
        ChannelQueue * out_queue,
        runtime->queue_manager().GetBoundaryQueueByName("out_stream"));
    EXPECT_EQ(out_queue->GetSize(), 20);

    DrainOutputChannels(*runtime);
    EXPECT_TRUE(out_queue->IsEmpty());
  }
}

}  // namespace
}  // namespace xls
