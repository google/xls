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

#include "xls/scheduling/proc_clumping_pass.h"

#include <random>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/statusor.h"
#include "xls/common/status/matchers.h"
#include "xls/delay_model/delay_estimators.h"
#include "xls/interpreter/interpreter_proc_runtime.h"
#include "xls/interpreter/proc_runtime.h"
#include "xls/ir/function.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/ir_test_base.h"
#include "xls/scheduling/pipeline_scheduling_pass.h"

namespace xls {
namespace {

using status_testing::IsOkAndHolds;
using status_testing::StatusIs;
using ::testing::HasSubstr;

class ProcClumpingPassTest : public IrTestBase {
 protected:
  ProcClumpingPassTest() = default;

  absl::StatusOr<bool> Run(Package* pkg, const SchedulingPassOptions& options) {
    SchedulingPassResults results;
    SchedulingUnit<> unit{pkg, std::nullopt};
    bool subpass_changed = false;
    bool changed = false;
    XLS_ASSIGN_OR_RETURN(subpass_changed, PipelineSchedulingPass().Run(
                                              &unit, options, &results));
    changed |= subpass_changed;
    XLS_ASSIGN_OR_RETURN(subpass_changed,
                         ProcClumpingPass().Run(&unit, options, &results));
    changed |= subpass_changed;
    return changed;
  }

  void TestSimple(Proc* proc, int64_t pipeline_stages);
};

// Given a proc with one input channel (id = 0, type = bits[1]) and one output
// channel (id = 1, type = bits[1]), send a random string of bits into the input
// channel using the proc interpreter and record the "original" output, then run
// the scheduler and clumping pass, rerun the interpreter, and compare this new
// output against the original output.
void ProcClumpingPassTest::TestSimple(Proc* proc, int64_t pipeline_stages) {
  Package* p = proc->package();

  std::minstd_rand engine;
  std::bernoulli_distribution coin_flip(0.5);

  std::vector<Value> input;
  for (int64_t i = 0; i < 20; ++i) {
    input.push_back(Value(UBits(static_cast<int64_t>(coin_flip(engine)), 1)));
  }

  std::vector<Value> output_expected;
  {
    XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<SerialProcRuntime> spr,
                             CreateInterpreterSerialProcRuntime(p));
    for (const Value& value : input) {
      XLS_ASSERT_OK(spr->queue_manager().queues().at(0)->Write(value));
    }
    XLS_ASSERT_OK(spr->TickUntilBlocked());
    while (std::optional<Value> value_opt =
               spr->queue_manager().queues().at(1)->Read()) {
      output_expected.push_back(value_opt.value());
    }
  }

  SchedulingPassOptions options;
  options.scheduling_options =
      SchedulingOptions().pipeline_stages(pipeline_stages);
  XLS_ASSERT_OK_AND_ASSIGN(options.delay_estimator, GetDelayEstimator("unit"));
  EXPECT_THAT(Run(p, options), IsOkAndHolds(true));

  std::vector<Value> output_actual;
  {
    XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<SerialProcRuntime> spr,
                             CreateInterpreterSerialProcRuntime(p));
    for (const Value& value : input) {
      XLS_ASSERT_OK(spr->queue_manager().queues().at(0)->Write(value));
    }
    XLS_ASSERT_OK(spr->TickUntilBlocked());
    while (std::optional<Value> value_opt =
               spr->queue_manager().queues().at(1)->Read()) {
      output_actual.push_back(value_opt.value());
    }
  }

  EXPECT_EQ(output_expected, output_actual);
}

absl::StatusOr<Channel*> BitRecvChannel(Package* p, std::string_view name) {
  return p->CreateStreamingChannel(name, ChannelOps::kReceiveOnly,
                                   p->GetBitsType(1));
}

absl::StatusOr<Channel*> BitSendChannel(Package* p, std::string_view name) {
  return p->CreateStreamingChannel(name, ChannelOps::kSendOnly,
                                   p->GetBitsType(1));
}

TEST_F(ProcClumpingPassTest, IIGreater1AndMultipleStagesFails) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Channel * in, BitRecvChannel(p.get(), "input"));
  XLS_ASSERT_OK_AND_ASSIGN(Channel * out, BitSendChannel(p.get(), "output"));
  TokenlessProcBuilder pb("p", "tkn", p.get());
  pb.Send(out, pb.Not(pb.Not(pb.Receive(in))));
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, pb.Build({}));
  proc->SetInitiationInterval(5);
  XLS_ASSERT_OK(p->SetTop(proc));

  SchedulingPassOptions options;
  options.scheduling_options = SchedulingOptions().pipeline_stages(10);
  XLS_ASSERT_OK_AND_ASSIGN(options.delay_estimator, GetDelayEstimator("unit"));
  EXPECT_THAT(Run(p.get(), options),
              StatusIs(absl::StatusCode::kInternal,
                       HasSubstr("initiation interval must currently equal the "
                                 "number of pipeline stages")));
}

TEST_F(ProcClumpingPassTest, Simple) {
  // TODO(taktoa): II >= 5 fails because the schedule splits a
  //               receive/tuple_index pair, we should constrain the scheduler
  //               to not do that. Similar issue in the other tests.
  for (int64_t ii = 1; ii < 5; ++ii) {
    auto p = CreatePackage();
    XLS_ASSERT_OK_AND_ASSIGN(Channel * in, BitRecvChannel(p.get(), "input"));
    XLS_ASSERT_OK_AND_ASSIGN(Channel * out, BitSendChannel(p.get(), "output"));
    TokenlessProcBuilder pb("p", "tkn", p.get());
    pb.Send(out, pb.Not(pb.Not(pb.Receive(in))));
    XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, pb.Build({}));
    proc->SetInitiationInterval(ii);
    XLS_ASSERT_OK(p->SetTop(proc));
    TestSimple(proc, ii);
  }
}

TEST_F(ProcClumpingPassTest, SingleBackedge) {
  for (int64_t ii = 1; ii < 6; ++ii) {
    auto p = CreatePackage();
    XLS_ASSERT_OK_AND_ASSIGN(Channel * in, BitRecvChannel(p.get(), "input"));
    XLS_ASSERT_OK_AND_ASSIGN(Channel * out, BitSendChannel(p.get(), "output"));
    TokenlessProcBuilder pb("p", "tkn", p.get());
    BValue state = pb.StateElement("state", Value(UBits(0, 1)));
    BValue next = pb.Not(pb.Not(pb.Xor(pb.Receive(in), state)));
    pb.Send(out, next);
    XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, pb.Build({next}));
    proc->SetInitiationInterval(ii);
    XLS_ASSERT_OK(p->SetTop(proc));
    TestSimple(proc, ii);
  }
}

TEST_F(ProcClumpingPassTest, MultipleBackedge) {
  for (int64_t ii = 1; ii < 6; ++ii) {
    auto p = CreatePackage();
    XLS_ASSERT_OK_AND_ASSIGN(Channel * in, BitRecvChannel(p.get(), "input"));
    XLS_ASSERT_OK_AND_ASSIGN(Channel * out, BitSendChannel(p.get(), "output"));
    TokenlessProcBuilder pb("p", "tkn", p.get());
    BValue state1 = pb.StateElement("state1", Value(UBits(0, 1)));
    BValue state2 = pb.StateElement("state2", Value(UBits(0, 1)));
    BValue next =
        pb.Not(pb.Not(pb.Xor(pb.Receive(in), pb.Xor(state1, state2))));
    pb.Send(out, next);
    XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, pb.Build({next, pb.Not(next)}));
    proc->SetInitiationInterval(ii);
    XLS_ASSERT_OK(p->SetTop(proc));
    TestSimple(proc, ii);
  }
}

TEST_F(ProcClumpingPassTest, MultipleBackedgeInDifferentLogicalCycles) {
  for (int64_t ii = 1; ii < 8; ++ii) {
    auto p = CreatePackage();
    XLS_ASSERT_OK_AND_ASSIGN(Channel * in, BitRecvChannel(p.get(), "input"));
    XLS_ASSERT_OK_AND_ASSIGN(Channel * out, BitSendChannel(p.get(), "output"));
    TokenlessProcBuilder pb("p", "tkn", p.get());
    BValue state1 = pb.StateElement("state1", Value(UBits(0, 1)));
    BValue state2 = pb.StateElement("state2", Value(UBits(0, 1)));
    BValue next1 = pb.Not(pb.Not(pb.Xor(pb.Receive(in), state1)));
    BValue next2 = pb.Not(pb.Not(pb.Xor(next1, state2)));
    pb.Send(out, next2);
    XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, pb.Build({next1, next2}));
    proc->SetInitiationInterval(ii);
    XLS_ASSERT_OK(p->SetTop(proc));
    TestSimple(proc, ii);
  }
}

}  // namespace
}  // namespace xls
