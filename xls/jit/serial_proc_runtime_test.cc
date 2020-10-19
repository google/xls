// Copyright 2020 The XLS Authors
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
#include "xls/jit/serial_proc_runtime.h"

#include <thread>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "xls/common/status/matchers.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/ir_parser.h"
#include "xls/ir/package.h"
#include "xls/jit/jit_channel_queue.h"

namespace xls {
namespace {

template <typename T>
void EnqueueData(JitChannelQueue* queue, T data) {
  queue->Send(reinterpret_cast<uint8*>(&data), sizeof(T));
}

template <typename T>
T DequeueData(JitChannelQueue* queue) {
  T data;
  queue->Recv(reinterpret_cast<uint8*>(&data), sizeof(T));
  return data;
}

// This test verifies functionality of a simple X -> A -> B -> Y network without
// internal state. Passes a constant into two procs, with the result that the
// input should be multiplied by 6.
TEST(SerialProcRuntimeTest, SimpleNetwork) {
  constexpr int kNumCycles = 4;
  const std::string kIrText = R"(
package p

chan a_in(data: bits[32], id=0, kind=receive_only, metadata="")
chan a_to_b(data: bits[32], id=1, kind=send_receive, metadata="")
chan b_out(data: bits[32], id=2, kind=send_only, metadata="")

proc a(my_token: token, state: (), init=()) {
  literal.1: bits[32] = literal(value=2)
  receive.2: (token, bits[32]) = receive(my_token, channel_id=0)
  tuple_index.3: token = tuple_index(receive.2, index=0)
  tuple_index.4: bits[32] = tuple_index(receive.2, index=1)
  umul.5: bits[32] = umul(literal.1, tuple_index.4)
  send.6: token = send(tuple_index.3, data=[umul.5], channel_id=1)
  ret tuple.7: (token, ()) = tuple(send.6, state)
}

proc b(my_token: token, state: (), init=()) {
  literal.100: bits[32] = literal(value=3)
  receive.200: (token, bits[32]) = receive(my_token, channel_id=1)
  tuple_index.300: token = tuple_index(receive.200, index=0)
  tuple_index.400: bits[32] = tuple_index(receive.200, index=1)
  umul.500: bits[32] = umul(literal.100, tuple_index.400)
  send.600: token = send(tuple_index.300, data=[umul.500], channel_id=2)
  ret tuple.700: (token, ()) = tuple(send.600, state)
}
)";

  XLS_ASSERT_OK_AND_ASSIGN(auto p, Parser::ParsePackage(kIrText));
  XLS_ASSERT_OK_AND_ASSIGN(auto runtime, SerialProcRuntime::Create(p.get()));
  auto queue_mgr = runtime->queue_mgr();
  XLS_ASSERT_OK_AND_ASSIGN(auto input_queue, queue_mgr->GetQueueById(0));
  XLS_ASSERT_OK_AND_ASSIGN(auto internal_queue, queue_mgr->GetQueueById(1));
  XLS_ASSERT_OK_AND_ASSIGN(auto output_queue, queue_mgr->GetQueueById(2));

  // Prepopulate the non-output queues.
  for (int i = 0; i < kNumCycles; i++) {
    EnqueueData(input_queue, i);
  }

  int dummy = 0;
  EnqueueData(internal_queue, dummy);

  // Run the runtime for those four cycles...
  for (int i = 0; i < kNumCycles; i++) {
    XLS_ASSERT_OK(runtime->Tick());
  }

  // Then verify the output queue contains the right info. We drop one output,
  // since "b" doesn't get the actual input data until cycle 1.
  DequeueData<int>(output_queue);
  for (int i = 0; i < kNumCycles - 1; i++) {
    int result = DequeueData<int>(output_queue);
    EXPECT_EQ(result, i * 6);
  }
}

// Test verifies that an "X"-shaped network can be modeled correctly, i.e.,
// a network that looks like:
//  A   B
//   \ /
//    C
//   / \
//  D   E
//
// Where A and B receive inputs from "outside", and D and E produce outputs.
TEST(SerialProcRuntimeTest, XNetwork) {
  constexpr int kNumCycles = 32;
  const std::string kIrText = R"(
package p

chan i_a(data: bits[32], id=0, kind=receive_only, metadata="")
chan i_b(data: bits[32], id=1, kind=receive_only, metadata="")
chan a_c(data: bits[32], id=2, kind=send_receive, metadata="")
chan b_c(data: bits[32], id=3, kind=send_receive, metadata="")
chan c_d(data: bits[32], id=4, kind=send_receive, metadata="")
chan c_e(data: bits[32], id=5, kind=send_receive, metadata="")
chan d_o(data: bits[32], id=6, kind=send_only, metadata="")
chan e_o(data: bits[32], id=7, kind=send_only, metadata="")

proc a(my_token: token, state: (), init=()) {
  literal.1: bits[32] = literal(value=1)
  receive.2: (token, bits[32]) = receive(my_token, channel_id=0)
  tuple_index.3: token = tuple_index(receive.2, index=0)
  tuple_index.4: bits[32] = tuple_index(receive.2, index=1)
  umul.5: bits[32] = umul(literal.1, tuple_index.4)
  send.6: token = send(tuple_index.3, data=[umul.5], channel_id=2)
  ret tuple.7: (token, ()) = tuple(send.6, state)
}

proc b(my_token: token, state: (), init=()) {
  literal.101: bits[32] = literal(value=2)
  receive.102: (token, bits[32]) = receive(my_token, channel_id=1)
  tuple_index.103: token = tuple_index(receive.102, index=0)
  tuple_index.104: bits[32] = tuple_index(receive.102, index=1)
  umul.105: bits[32] = umul(literal.101, tuple_index.104)
  send.106: token = send(tuple_index.103, data=[umul.105], channel_id=3)
  ret tuple.107: (token, ()) = tuple(send.106, state)
}

proc c(my_token: token, state: (), init=()) {
  literal.201: bits[32] = literal(value=3)
  receive.202: (token, bits[32]) = receive(my_token, channel_id=2)
  tuple_index.203: token = tuple_index(receive.202, index=0)
  tuple_index.204: bits[32] = tuple_index(receive.202, index=1)
  receive.205: (token, bits[32]) = receive(tuple_index.203, channel_id=3)
  tuple_index.206: token = tuple_index(receive.205, index=0)
  tuple_index.207: bits[32] = tuple_index(receive.205, index=1)
  umul.208: bits[32] = umul(literal.201, tuple_index.204)
  umul.209: bits[32] = umul(literal.201, tuple_index.207)
  send.210: token = send(tuple_index.206, data=[umul.208], channel_id=4)
  send.211: token = send(send.210, data=[umul.209], channel_id=5)
  ret tuple.212: (token, ()) = tuple(send.211, state)
}

proc d(my_token: token, state: (), init=()) {
  literal.301: bits[32] = literal(value=4)
  receive.302: (token, bits[32]) = receive(my_token, channel_id=4)
  tuple_index.303: token = tuple_index(receive.302, index=0)
  tuple_index.304: bits[32] = tuple_index(receive.302, index=1)
  umul.305: bits[32] = umul(literal.301, tuple_index.304)
  send.306: token = send(tuple_index.303, data=[umul.305], channel_id=6)
  ret tuple.307: (token, ()) = tuple(send.306, state)
}

proc e(my_token: token, state: (), init=()) {
  literal.401: bits[32] = literal(value=5)
  receive.402: (token, bits[32]) = receive(my_token, channel_id=5)
  tuple_index.403: token = tuple_index(receive.402, index=0)
  tuple_index.404: bits[32] = tuple_index(receive.402, index=1)
  umul.405: bits[32] = umul(literal.401, tuple_index.404)
  send.406: token = send(tuple_index.403, data=[umul.405], channel_id=7)
  ret tuple.407: (token, ()) = tuple(send.406, state)
}
)";
  XLS_ASSERT_OK_AND_ASSIGN(auto p, Parser::ParsePackage(kIrText));
  XLS_ASSERT_OK_AND_ASSIGN(auto runtime, SerialProcRuntime::Create(p.get()));
  auto queue_mgr = runtime->queue_mgr();
  XLS_ASSERT_OK_AND_ASSIGN(auto i_a, queue_mgr->GetQueueById(0));
  XLS_ASSERT_OK_AND_ASSIGN(auto i_b, queue_mgr->GetQueueById(1));
  XLS_ASSERT_OK_AND_ASSIGN(auto d_o, queue_mgr->GetQueueById(6));
  XLS_ASSERT_OK_AND_ASSIGN(auto e_o, queue_mgr->GetQueueById(7));

  // "Prime" the internal queues with garbage data (since it'll be one or two
  // cycles until they get real data).

  for (int i = 0; i < kNumCycles; i++) {
    EnqueueData(i_a, i);
    EnqueueData(i_b, i + 10);
    XLS_ASSERT_OK(runtime->Tick());
  }

  // Now, cut out the garbage data from the output queues, and then verify their
  // contents.
  for (int i = 0; i < kNumCycles - 2; i++) {
    int result = DequeueData<int>(d_o);
    ASSERT_EQ(result, i * 1 * 3 * 4);

    result = DequeueData<int>(e_o);
    ASSERT_EQ(result, (i + 10) * 2 * 3 * 5);
  }
}

// This test verify that state is indeed carried correctly between cycles.
// "a" starts with a "0" state and increments it every time, using it as a
// factor in its umul.
TEST(SerialProcRuntimeTest, CarriesState) {
  constexpr int kNumCycles = 8;
  const std::string kIrText = R"(
package p

chan a_in(data: bits[32], id=0, kind=receive_only, metadata="")
chan a_to_b(data: bits[32], id=1, kind=send_receive, metadata="")
chan b_out(data: bits[32], id=2, kind=send_only, metadata="")

proc a(my_token: token, state: (bits[32]), init=(1)) {
  tuple_index.1: bits[32] = tuple_index(state, index=0)
  receive.2: (token, bits[32]) = receive(my_token, channel_id=0)
  tuple_index.3: token = tuple_index(receive.2, index=0)
  tuple_index.4: bits[32] = tuple_index(receive.2, index=1)
  umul.5: bits[32] = umul(tuple_index.1, tuple_index.4)
  send.6: token = send(tuple_index.3, data=[umul.5], channel_id=1)
  literal.7: bits[32] = literal(value=1)
  add.8: bits[32] = add(tuple_index.1, literal.7)
  tuple.9: (bits[32]) = tuple(add.8)
  ret tuple.10: (token, (bits[32])) = tuple(send.6, tuple.9)
}

proc b(my_token: token, state: (bits[32]), init=()) {
  literal.100: bits[32] = literal(value=3)
  receive.200: (token, bits[32]) = receive(my_token, channel_id=1)
  tuple_index.300: token = tuple_index(receive.200, index=0)
  tuple_index.400: bits[32] = tuple_index(receive.200, index=1)
  umul.500: bits[32] = umul(literal.100, tuple_index.400)
  send.600: token = send(tuple_index.300, data=[umul.500], channel_id=2)
  ret tuple.700: (token, ()) = tuple(send.600, state)
}
)";
  XLS_ASSERT_OK_AND_ASSIGN(auto p, Parser::ParsePackage(kIrText));
  XLS_ASSERT_OK_AND_ASSIGN(auto runtime, SerialProcRuntime::Create(p.get()));
  auto* queue_mgr = runtime->queue_mgr();

  XLS_ASSERT_OK_AND_ASSIGN(auto input_queue, queue_mgr->GetQueueById(0));
  XLS_ASSERT_OK_AND_ASSIGN(auto internal_queue, queue_mgr->GetQueueById(1));
  XLS_ASSERT_OK_AND_ASSIGN(auto output_queue, queue_mgr->GetQueueById(2));

  int dummy = 0;
  EnqueueData(internal_queue, dummy);

  for (int i = 0; i < kNumCycles; i++) {
    EnqueueData(input_queue, i);
    XLS_ASSERT_OK(runtime->Tick());
  }

  // Drop the output from the first cycle; it's not real/valid output.
  DequeueData<int>(output_queue);
  for (int i = 0; i < kNumCycles - 1; i++) {
    int actual = DequeueData<int>(output_queue);
    ASSERT_EQ(actual, i * (i + 1) * 3);
  }
}

// This test verifies that SerialProcRuntime can detect when a network has
// deadlocked (when it's waiting on more data that's not coming).
TEST(SerialProcRuntimeTest, DetectsDeadlock) {
  // Proc A sends one pieces of data to B, but B expects two - the second will
  // never arrive.
  const std::string kIrText = R"(
package p

chan first(data: bits[32], id=1, kind=send_receive, metadata="")
chan second(data: bits[32], id=2, kind=send_receive, metadata="")

proc a(my_token: token, state: bits[1], init=0) {
  literal.1: bits[32] = literal(value=1)
  send.3: token = send(my_token, data=[literal.1], channel_id=1)
  send_if.4: token = send_if(send.3, state, data=[literal.1], channel_id=2)
  ret tuple.5: (token, bits[1]) = tuple(send_if.4, state)
}

proc b(my_token: token, state: (), init=()) {
  receive.101: (token, bits[32]) = receive(my_token, channel_id=1)
  tuple_index.102: token = tuple_index(receive.101, index=0)
  receive.103: (token, bits[32]) = receive(tuple_index.102, channel_id=2)
  tuple_index.104: token = tuple_index(receive.103, index=0)
  ret tuple.107: (token, ()) = tuple(tuple_index.104, state)
}
)";
  XLS_ASSERT_OK_AND_ASSIGN(auto p, Parser::ParsePackage(kIrText));
  XLS_ASSERT_OK_AND_ASSIGN(auto runtime, SerialProcRuntime::Create(p.get()));
  ASSERT_THAT(runtime->Tick(),
              status_testing::StatusIs(absl::StatusCode::kAborted));
}

// Tests that a proc can be blocked (by missing recv data) and then become
// unblocked when data is available.
TEST(SerialProcRuntimeTest, FinishesDelayedCycle) {
  const std::string kIrText = R"(
package p

chan input(data: bits[32], id=0, kind=receive_only, metadata="")
chan a_to_b(data: bits[32], id=1, kind=send_receive, metadata="")
chan output(data: bits[32], id=2, kind=send_only, metadata="")

proc a(my_token: token, state: (), init=()) {
  receive.1: (token, bits[32]) = receive(my_token, channel_id=0)
  tuple_index.2: token = tuple_index(receive.1, index=0)
  tuple_index.3: bits[32] = tuple_index(receive.1, index=1)
  send.4: token = send(tuple_index.2, data=[tuple_index.3], channel_id=1)
  ret tuple.5: (token, ()) = tuple(send.4, state)
}

proc b(my_token: token, state: (), init=()) {
  receive.101: (token, bits[32]) = receive(my_token, channel_id=1)
  tuple_index.102: token = tuple_index(receive.101, index=0)
  tuple_index.103: bits[32] = tuple_index(receive.101, index=1)
  send.104: token = send(tuple_index.102, data=[tuple_index.103], channel_id=2)
  ret tuple.105: (token, ()) = tuple(send.104, state)
}
)";
  XLS_ASSERT_OK_AND_ASSIGN(auto p, Parser::ParsePackage(kIrText));
  XLS_ASSERT_OK_AND_ASSIGN(auto runtime, SerialProcRuntime::Create(p.get()));
  XLS_ASSERT_OK_AND_ASSIGN(auto input_queue,
                           runtime->queue_mgr()->GetQueueById(0));
  std::thread thread([input_queue]() {
    // Give enough time for the network to block, then send in the missing data.
    sleep(1);
    int32 data = 42;
    input_queue->Send(reinterpret_cast<uint8*>(&data), sizeof(data));
  });
  XLS_ASSERT_OK(runtime->Tick());
  XLS_ASSERT_OK_AND_ASSIGN(auto output_queue,
                           runtime->queue_mgr()->GetQueueById(2));

  int32 data;
  output_queue->Recv(reinterpret_cast<uint8*>(&data), sizeof(data));
  EXPECT_EQ(data, 42);
  thread.join();
}

}  // namespace
}  // namespace xls
