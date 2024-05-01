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

#include <algorithm>
#include <cstdint>
#include <memory>
#include <type_traits>
#include <vector>

#include "absl/log/check.h"
#include "include/benchmark/benchmark.h"
#include "xls/ir/channel.h"
#include "xls/ir/package.h"
#include "xls/jit/jit_channel_queue.h"
#include "xls/jit/jit_runtime.h"

namespace xls {
namespace {

// Benchmark evaluating writing to the channel then reading from the channel.
// The number of writes are followed by an equal amount of number of reads.
// For a FIFO channel queue, it evaluates the enqueing and dequeing mechanism.
template <typename QueueT,
          typename std::enable_if<std::is_base_of_v<JitChannelQueue, QueueT>,
                                  QueueT>::type* = nullptr>
static void BM_QueueWriteThenRead(benchmark::State& state) {
  int64_t element_size_bytes = state.range(0);

  Package package("benchmark");
  std::unique_ptr<JitRuntime> jit_runtime = JitRuntime::Create().value();
  Channel* channel =
      package
          .CreateStreamingChannel("my_channel", ChannelOps::kSendReceive,
                                  package.GetBitsType(8 * element_size_bytes))
          .value();
  ProcElaboration elaboration =
      ProcElaboration::ElaborateOldStylePackage(&package).value();

  QueueT queue(elaboration.GetUniqueInstance(channel).value(),
               jit_runtime.get());

  int64_t send_count = state.range(1);
  CHECK(queue.IsEmpty());
  std::vector<uint8_t> send_buffer(element_size_bytes);
  std::vector<uint8_t> recv_buffer(element_size_bytes);
  std::fill(send_buffer.begin(), send_buffer.end(), 42);
  for (auto _ : state) {
    for (int64_t i = 0; i < send_count; ++i) {
      queue.WriteRaw(send_buffer.data());
    }
    for (int64_t i = 0; i < send_count; ++i) {
      queue.ReadRaw(recv_buffer.data());
    }
  }
}

// For the following benchmark, the first element in the pair denotes the buffer
// size written/read from the channel queue. The second element in the pair
// denotes the number of writes and/or reads to the channel queue.
BENCHMARK(BM_QueueWriteThenRead<ThreadSafeJitChannelQueue>)
    ->ArgPair(1, 1)
    ->ArgPair(1, 128)
    ->ArgPair(8, 1)
    ->ArgPair(8, 128)
    ->ArgPair(32, 1)
    ->ArgPair(32, 128)
    ->ArgPair(2048, 1)
    ->ArgPair(2048, 128);

BENCHMARK(BM_QueueWriteThenRead<ThreadUnsafeJitChannelQueue>)
    ->ArgPair(1, 1)
    ->ArgPair(1, 128)
    ->ArgPair(8, 1)
    ->ArgPair(8, 128)
    ->ArgPair(32, 1)
    ->ArgPair(32, 128)
    ->ArgPair(2048, 1)
    ->ArgPair(2048, 128);

}  // namespace
}  // namespace xls

BENCHMARK_MAIN();
