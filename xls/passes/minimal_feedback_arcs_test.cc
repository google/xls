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

#include "xls/passes/minimal_feedback_arcs.h"

#include <cstdint>
#include <memory>
#include <string_view>
#include <utility>
#include <vector>

#include "benchmark/benchmark.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "xls/common/status/matchers.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/bits.h"
#include "xls/ir/channel_ops.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/ir_matcher.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/package.h"
#include "xls/ir/value.h"

namespace m = xls::op_matchers;

namespace xls {
namespace {
using ::testing::IsEmpty;
using ::testing::SizeIs;
using ::testing::UnorderedElementsAre;

// Make a proc network with n procs each performing a single receive and send
// in a loop through every proc.
static absl::Status MakeMultiProcLoop(Package* p, int64_t n) {
  std::vector<Channel*> internal_channels;
  for (int64_t i = 0; i < n; ++i) {
    XLS_ASSIGN_OR_RETURN(Channel * ch,
                         p->CreateStreamingChannel(
                             absl::StrFormat("internal%d", i),
                             ChannelOps::kSendReceive, p->GetBitsType(32)));
    internal_channels.push_back(ch);
  }
  for (int64_t i = 0; i < n; ++i) {
    ProcBuilder pb(absl::StrFormat("proc%d", i), p);
    Channel* recv_chan = internal_channels[i];
    Channel* send_chan =
        internal_channels[((i + 1) >= n) ? (i + 1 - n) : (i + 1)];
    BValue recv = pb.Receive(recv_chan, pb.Literal(Value::Token()));
    BValue recv_token = pb.TupleIndex(recv, 0);
    BValue recv_data = pb.TupleIndex(recv, 1);
    pb.Send(send_chan, recv_token, recv_data);
    XLS_RETURN_IF_ERROR(pb.Build().status());
  }

  return absl::OkStatus();
}

// Make a proc network with a single proc performing n loopback send/receive
// pairs in a cycle.
static absl::Status MakeSingleProcLoop(Package* p, int64_t n) {
  std::vector<Channel*> internal_channels;
  for (int64_t i = 0; i < n; ++i) {
    XLS_ASSIGN_OR_RETURN(Channel * ch,
                         p->CreateStreamingChannel(
                             absl::StrFormat("internal%d", i),
                             ChannelOps::kSendReceive, p->GetBitsType(32)));
    internal_channels.push_back(ch);
  }
  ProcBuilder pb("foo", p);
  BValue prev_token = pb.Literal(Value::Token());
  for (int64_t i = 0; i < n; ++i) {
    Channel* recv_chan = internal_channels[i];
    Channel* send_chan =
        internal_channels[((i + 1) >= n) ? (i + 1 - n) : (i + 1)];
    BValue recv = pb.Receive(recv_chan, prev_token);
    BValue recv_token = pb.TupleIndex(recv, 0);
    BValue recv_data = pb.TupleIndex(recv, 1);
    BValue send = pb.Send(send_chan, recv_token, recv_data);
    prev_token = send;
  }
  XLS_RETURN_IF_ERROR(pb.Build().status());
  return absl::OkStatus();
}

// Make n procs that all receive from every other proc, add the result up, and
// then send it to every other proc.
static absl::Status MakeFullyConnectedProcNetwork(Package* p, int64_t n) {
  absl::flat_hash_map<std::pair<int64_t, int64_t>, Channel*> internal_channels;

  for (int64_t i = 0; i < n; ++i) {
    for (int64_t j = 0; j < n; ++j) {
      if (i == j) {
        continue;
      }
      XLS_ASSIGN_OR_RETURN(Channel * ch,
                           p->CreateStreamingChannel(
                               absl::StrFormat("internal%d_%d", i, j),
                               ChannelOps::kSendReceive, p->GetBitsType(32)));
      internal_channels.insert({std::make_pair(i, j), ch});
    }
  }
  for (int64_t i = 0; i < n; ++i) {
    ProcBuilder pb(absl::StrFormat("foo%d", i), p);
    BValue initial_token = pb.Literal(Value::Token());
    BValue sum = pb.Literal(UBits(0, 32));
    std::vector<BValue> recv_tokens;
    for (int64_t j = 0; j < n; ++j) {
      if (i == j) {
        continue;
      }
      Channel* recv_chan = internal_channels[{i, j}];
      BValue recv = pb.Receive(recv_chan, initial_token);
      BValue recv_token = pb.TupleIndex(recv, 0);
      BValue recv_data = pb.TupleIndex(recv, 1);
      sum = pb.Add(sum, recv_data);
      recv_tokens.push_back(recv_token);
    }
    BValue all_recv_token = pb.AfterAll(recv_tokens);
    for (int64_t j = 0; j < n; ++j) {
      if (i == j) {
        continue;
      }
      Channel* send_chan = internal_channels[{j, i}];
      pb.Send(send_chan, all_recv_token, sum);
    }
    XLS_RETURN_IF_ERROR(pb.Build().status());
  }
  return absl::OkStatus();
}

class MinimalFeedbackArcsTest : public IrTestBase {};

TEST_F(MinimalFeedbackArcsTest, SingleProcWithNoFeedback) {
  constexpr std::string_view ir_text = R"(package test
chan in(bits[32], id=0, kind=streaming, ops=receive_only, flow_control=ready_valid)
chan out(bits[32], id=1, kind=streaming, ops=send_only, flow_control=ready_valid)

top proc foo(st: (), init={()}) {
  tkn: token = literal(value=token)
  recv: (token, bits[32]) = receive(tkn, channel=in)
  recv_token: token = tuple_index(recv, index=0)
  recv_data: bits[32] = tuple_index(recv, index=1)
  send_token: token = send(recv_token, recv_data, channel=out)
  next (st)
}
)";
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> p, ParsePackage(ir_text));
  XLS_ASSERT_OK_AND_ASSIGN(auto feedback_channels,
                           MinimalFeedbackArcs(p.get()));
  EXPECT_THAT(feedback_channels, IsEmpty());
}

TEST_F(MinimalFeedbackArcsTest, SingleProcWithLoopbackButNoCycle) {
  constexpr std::string_view ir_text = R"(package test
chan in(bits[32], id=0, kind=streaming, ops=receive_only, flow_control=ready_valid)
chan out(bits[32], id=1, kind=streaming, ops=send_only, flow_control=ready_valid)
chan internal(bits[32], id=2, kind=streaming, ops=send_receive, flow_control=ready_valid)

top proc foo(st: (), init={()}) {
  tkn: token = literal(value=token)
  recv: (token, bits[32]) = receive(tkn, channel=in)
  recv_token: token = tuple_index(recv, index=0)
  recv_data: bits[32] = tuple_index(recv, index=1)
  internal_recv: (token, bits[32]) = receive(tkn, channel=internal)
  internal_recv_token: token = tuple_index(internal_recv, index=0)
  internal_recv_data: bits[32] = tuple_index(internal_recv, index=1)
  sum: bits[32] = add(recv_data, internal_recv_data)
  all_receive_token: token = after_all(recv_token, internal_recv_token)
  send_token: token = send(all_receive_token, sum, channel=out)
  internal_send_token: token = send(recv_token, recv_data, channel=internal)
  next (st)
}
)";
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> p, ParsePackage(ir_text));
  XLS_ASSERT_OK_AND_ASSIGN(auto feedback_channels,
                           MinimalFeedbackArcs(p.get()));
  EXPECT_THAT(feedback_channels, IsEmpty());
}

TEST_F(MinimalFeedbackArcsTest, SingleProcWithLoopbackCycle) {
  constexpr std::string_view ir_text = R"(package test
chan in(bits[32], id=0, kind=streaming, ops=receive_only, flow_control=ready_valid)
chan out(bits[32], id=1, kind=streaming, ops=send_only, flow_control=ready_valid)
chan internal(bits[32], id=2, kind=streaming, ops=send_receive, flow_control=ready_valid)

top proc foo(st: (), init={()}) {
  tkn: token = literal(value=token)
  recv: (token, bits[32]) = receive(tkn, channel=in)
  recv_token: token = tuple_index(recv, index=0)
  recv_data: bits[32] = tuple_index(recv, index=1)
  internal_recv: (token, bits[32]) = receive(tkn, channel=internal)
  internal_recv_token: token = tuple_index(internal_recv, index=0)
  internal_recv_data: bits[32] = tuple_index(internal_recv, index=1)
  sum: bits[32] = add(recv_data, internal_recv_data)
  all_receive_token: token = after_all(recv_token, internal_recv_token)
  send_token: token = send(all_receive_token, sum, channel=out)
  internal_send_token: token = send(all_receive_token, sum, channel=internal)
  next (st)
}
)";
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> p, ParsePackage(ir_text));
  XLS_ASSERT_OK_AND_ASSIGN(auto feedback_channels,
                           MinimalFeedbackArcs(p.get()));
  EXPECT_THAT(feedback_channels, UnorderedElementsAre(m::Channel("internal")));
}

TEST_F(MinimalFeedbackArcsTest, TwoProcsWithSplitAndJoin) {
  // ┌────────────────────┐
  // │ ┌─────┐    ┌─────┐ │
  // │ │ foo │    │ bar │ │
  // └►├─►   │    │ ┌───┼─┘
  //   │  + ─┼───►┼─┤   │
  // ┌►├─►   │    │ └───┼─┐
  // │ └─────┘    └─────┘ │
  // └────────────────────┘
  constexpr std::string_view ir_text = R"(package test
chan in(bits[32], id=0, kind=streaming, ops=receive_only, flow_control=ready_valid)
chan out(bits[32], id=1, kind=streaming, ops=send_only, flow_control=ready_valid)
chan internal0(bits[32], id=2, kind=streaming, ops=send_receive, flow_control=ready_valid)
chan internal1(bits[32], id=3, kind=streaming, ops=send_receive, flow_control=ready_valid)
chan internal2(bits[32], id=4, kind=streaming, ops=send_receive, flow_control=ready_valid)

top proc foo(st: (), init={()}) {
  tkn: token = literal(value=token)
  recv: (token, bits[32]) = receive(tkn, channel=in)
  recv_token: token = tuple_index(recv, index=0)
  recv_data: bits[32] = tuple_index(recv, index=1)
  internal_recv0: (token, bits[32]) = receive(tkn, channel=internal0)
  internal_recv0_token: token = tuple_index(internal_recv0, index=0)
  internal_recv0_data: bits[32] = tuple_index(internal_recv0, index=1)
  internal_recv1: (token, bits[32]) = receive(tkn, channel=internal1)
  internal_recv1_token: token = tuple_index(internal_recv1, index=0)
  internal_recv1_data: bits[32] = tuple_index(internal_recv1, index=1)
  partial_sum: bits[32] = add(internal_recv0_data, internal_recv1_data)
  sum: bits[32] = add(recv_data, partial_sum)
  all_receive_token: token = after_all(recv_token, internal_recv0_token, internal_recv1_token)
  send_token: token = send(all_receive_token, sum, channel=out)
  internal_send_token: token = send(all_receive_token, sum, channel=internal2)
  next (st)
}

proc bar(st: (), init={()}) {
  tkn: token = literal(value=token)
  recv: (token, bits[32]) = receive(tkn, channel=internal2)
  recv_token: token = tuple_index(recv, index=0)
  recv_data: bits[32] = tuple_index(recv, index=1)
  send0_token: token = send(recv_token, recv_data, channel=internal0)
  send1_token: token = send(recv_token, recv_data, channel=internal1)
  next (st)
}
)";
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> p, ParsePackage(ir_text));
  XLS_ASSERT_OK_AND_ASSIGN(auto feedback_channels,
                           MinimalFeedbackArcs(p.get()));
  EXPECT_THAT(feedback_channels, UnorderedElementsAre(m::Channel("internal2")));
}

TEST_F(MinimalFeedbackArcsTest, MultiProcLoop) {
  std::unique_ptr<Package> p = CreatePackage();
  XLS_ASSERT_OK(MakeMultiProcLoop(p.get(), 10));
  XLS_ASSERT_OK_AND_ASSIGN(auto feedback_channels,
                           MinimalFeedbackArcs(p.get()));
  // It only takes one edge to break a single long cycle.
  EXPECT_THAT(feedback_channels, SizeIs(1));
}

TEST_F(MinimalFeedbackArcsTest, SingleProcLoop) {
  std::unique_ptr<Package> p = CreatePackage();
  XLS_ASSERT_OK(MakeSingleProcLoop(p.get(), 10));
  XLS_ASSERT_OK_AND_ASSIGN(auto feedback_channels,
                           MinimalFeedbackArcs(p.get()));
  // It only takes one edge to break a single long cycle.
  EXPECT_THAT(feedback_channels, SizeIs(1));
}

TEST_F(MinimalFeedbackArcsTest, FullyConnectedProcNetwork) {
  std::unique_ptr<Package> p = CreatePackage();
  XLS_ASSERT_OK(MakeFullyConnectedProcNetwork(p.get(), 10));
  XLS_ASSERT_OK_AND_ASSIGN(auto feedback_channels,
                           MinimalFeedbackArcs(p.get()));
  // The number of feedback arcs for a fully connected graph (w/out self-edges)
  // is n(n-1)/2, which is 45 for n=10.
  EXPECT_THAT(feedback_channels, SizeIs(45));
}

void BM_MultiProcLoop(benchmark::State& state) {
  Package p("bm_test");
  XLS_ASSERT_OK(MakeMultiProcLoop(&p, state.range(0)));
  for (auto s : state) {
    XLS_ASSERT_OK_AND_ASSIGN(auto feedback_channels, MinimalFeedbackArcs(&p));
    benchmark::DoNotOptimize(feedback_channels);
  }
}

void BM_SingleProcLoop(benchmark::State& state) {
  Package p("bm_test");
  XLS_ASSERT_OK(MakeSingleProcLoop(&p, state.range(0)));
  for (auto s : state) {
    XLS_ASSERT_OK_AND_ASSIGN(auto feedback_channels, MinimalFeedbackArcs(&p));
    benchmark::DoNotOptimize(feedback_channels);
  }
}

// Fully connected proc networks are expensive inputs for this implementation of
// MinimalFeedackArcs(). This is for a couple reasons:
//  1) GreedyFAS wants to remove sources and sinks greedily, and there are no
//     sources or sinks in a fully connected graph. Every node that gets removed
//     has to be selected as the element with the max degree, which is more
//     expensive to compute than sink or source properties.
//  2) The graph is represented sparsely via head->tail and tail->head maps.
//     Keeping the two maps consistent comes with costs that scale with how the
//     number of edges on each vertex. Fully connected proc networks would be
//     more performant if adjacency were in some dense matrix representation.
//     Note that the fully connected proc network here is implemented with
//     fanned out tokens to the receives which all feed into an after_all, which
//     then fan out to all the sends. It would be much cheaper for the receives
//     and sends to be serially dependent on each other as an implicit
//     intermediate vertex would collapse many of the intermediate edges into
//     one.
void BM_FullyConnectedProcNetwork(benchmark::State& state) {
  Package p("bm_test");
  XLS_ASSERT_OK(MakeFullyConnectedProcNetwork(&p, state.range(0)));
  for (auto s : state) {
    XLS_ASSERT_OK_AND_ASSIGN(auto feedback_channels, MinimalFeedbackArcs(&p));
    benchmark::DoNotOptimize(feedback_channels);
  }
}

BENCHMARK(BM_MultiProcLoop)->Range(10, 5000);
BENCHMARK(BM_SingleProcLoop)->Range(10, 5000);
BENCHMARK(BM_FullyConnectedProcNetwork)->Range(10, 100);

}  // namespace
}  // namespace xls
