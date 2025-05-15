// Copyright 2024 The XLS Authors
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

#include "xls/scheduling/pipeline_scheduling_pass.h"

#include <filesystem>  // NOLINT
#include <string_view>
#include <utility>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "xls/common/file/get_runfile_path.h"
#include "xls/common/status/matchers.h"
#include "xls/common/status/status_macros.h"
#include "xls/fdo/synthesizer.h"
#include "xls/ir/bits.h"
#include "xls/ir/channel.h"
#include "xls/ir/channel_ops.h"
#include "xls/ir/foreign_function.h"
#include "xls/ir/foreign_function_data.pb.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/package.h"
#include "xls/ir/source_location.h"
#include "xls/ir/value.h"
#include "xls/passes/pass_base.h"
#include "xls/scheduling/scheduling_options.h"
#include "xls/scheduling/scheduling_pass.h"

namespace xls {
namespace {

using ::absl_testing::IsOkAndHolds;
using ::testing::AllOf;
using ::testing::Eq;
using ::testing::HasSubstr;
using ::testing::Pair;
using ::testing::UnorderedElementsAre;

MATCHER_P(SchedulingContextWithElements, matcher, "") {
  return ExplainMatchResult(matcher, arg.schedules(), result_listener);
}

MATCHER(VerifiedPipelineSchedule, "") { return arg.Verify().ok(); }

MATCHER_P3(LatencyIs, node0, node1, latency_matcher, "") {
  if (node0 == nullptr || node1 == nullptr) {
    *result_listener << "A node is nullptr.";
    return false;
  }
  if (node0->function_base() != node1->function_base()) {
    *result_listener << absl::StreamFormat(
        "%v and %v are not the same function bases (%v vs %v).", *node0, *node1,
        *node0->function_base(), *node1->function_base());
    return false;
  }
  if (arg.function_base() != node0->function_base()) {
    *result_listener << "schedule is not the same function base as the nodes.";
    return false;
  }
  if (!(arg.IsScheduled(node0) && arg.IsScheduled(node1))) {
    *result_listener << absl::StreamFormat("%v or %v not scheduled.", *node0,
                                           *node1);
    return false;
  }
  return ExplainMatchResult(
      latency_matcher, arg.cycle(node1) - arg.cycle(node0), result_listener);
}

MATCHER_P(HasDumpIr, matcher, "") {
  return ExplainMatchResult(matcher, arg.DumpIr(), result_listener);
}

using PipelineSchedulingPassTest = IrTestBase;
using RunResultT = std::pair<bool, SchedulingContext>;

absl::StatusOr<RunResultT> RunPipelineSchedulingPass(
    Package* package, const SchedulingOptions& scheduling_options,
    SchedulingContext& context, ::xls::synthesis::Synthesizer* synthesizer) {
  PassResults results;
  TestDelayEstimator delay_estimator;
  SchedulingPassOptions options{.scheduling_options = scheduling_options,
                                .delay_estimator = &delay_estimator,
                                .synthesizer = synthesizer};
  XLS_ASSIGN_OR_RETURN(bool changed, PipelineSchedulingPass().Run(
                                         package, options, &results, context));
  return std::make_pair(changed, context);
}
absl::StatusOr<RunResultT> RunPipelineSchedulingPass(
    Package* p, const SchedulingOptions& scheduling_options,
    ::xls::synthesis::Synthesizer* synthesizer = nullptr) {
  auto context = SchedulingContext::CreateForWholePackage(p);
  return RunPipelineSchedulingPass(p, scheduling_options, context, synthesizer);
}
absl::StatusOr<RunResultT> RunPipelineSchedulingPass(
    FunctionBase* f, const SchedulingOptions& scheduling_options,
    ::xls::synthesis::Synthesizer* synthesizer = nullptr) {
  auto context = SchedulingContext::CreateForSingleFunction(f);
  return RunPipelineSchedulingPass(f->package(), scheduling_options, context,
                                   synthesizer);
}

TEST_F(PipelineSchedulingPassTest, SingleFunction) {
  auto p = CreatePackage();
  FunctionBuilder fb("main", p.get());
  fb.Add(fb.Param("x", p->GetBitsType(32)), fb.Param("y", p->GetBitsType(32)));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());

  EXPECT_THAT(
      RunPipelineSchedulingPass(f, SchedulingOptions().pipeline_stages(2)),
      IsOkAndHolds(
          Pair(true, SchedulingContextWithElements(UnorderedElementsAre(
                         Pair(f, VerifiedPipelineSchedule()))))));
}

TEST_F(PipelineSchedulingPassTest, MultipleProcs) {
  auto p = CreatePackage();
  auto make_proc = [p = p.get()](std::string_view name,
                                 Channel* channel) -> absl::StatusOr<Proc*> {
    ProcBuilder pb(name, p);
    BValue tok = pb.Literal(Value::Token());
    BValue st = pb.StateElement("st", Value(UBits(0, 1)));
    BValue not_st = pb.Not(st);
    BValue lit50 = pb.Literal(UBits(50, 32));
    BValue lit60 = pb.Literal(UBits(60, 32));
    pb.SendIf(channel, tok, st, lit50);
    pb.SendIf(channel, tok, not_st, lit60);
    return pb.Build({not_st});
  };

  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * ch0, p->CreateStreamingChannel("ch0", ChannelOps::kSendOnly,
                                               p->GetBitsType(32)));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * ch1, p->CreateStreamingChannel("ch1", ChannelOps::kSendOnly,
                                               p->GetBitsType(32)));

  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc0, make_proc("proc0", ch0));
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc1, make_proc("proc1", ch1));

  XLS_ASSERT_OK_AND_ASSIGN(
      auto changed_unit, RunPipelineSchedulingPass(
                             p.get(), SchedulingOptions().pipeline_stages(2)));
  EXPECT_THAT(
      changed_unit,
      Pair(true, AllOf(SchedulingContextWithElements(UnorderedElementsAre(
                     Pair(proc0, VerifiedPipelineSchedule()),
                     Pair(proc1, VerifiedPipelineSchedule()))))));
  EXPECT_THAT(changed_unit.second.schedules().at(proc0).ToString(),
              AllOf(HasSubstr("Cycle 0:"),
                    HasSubstr("  st: bits[1] = state_read(state_element=st")));
  EXPECT_THAT(changed_unit.second.schedules().at(proc1).ToString(),
              AllOf(HasSubstr("Cycle 0:"),
                    HasSubstr("  st: bits[1] = state_read(state_element=st")));
}

TEST_F(PipelineSchedulingPassTest, MixedFunctionAndProcScheduling) {
  auto p = CreatePackage();

  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * ch, p->CreateStreamingChannel("ch", ChannelOps::kSendOnly,
                                              p->GetBitsType(1)));
  FunctionBuilder fb("main", p.get());
  fb.Add(fb.Param("x", p->GetBitsType(32)), fb.Param("y", p->GetBitsType(32)));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());

  ProcBuilder pb("pr", p.get());
  BValue tok = pb.Literal(Value::Token());
  BValue st = pb.StateElement("st", Value(UBits(0, 1)));
  BValue not_st = pb.Not(st);
  pb.Send(ch, tok, st);
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, pb.Build({not_st}));

  EXPECT_THAT(RunPipelineSchedulingPass(p.get(),
                                        SchedulingOptions().pipeline_stages(2)),
              IsOkAndHolds(
                  Pair(true, SchedulingContextWithElements(UnorderedElementsAre(
                                 Pair(f, VerifiedPipelineSchedule()),
                                 Pair(proc, VerifiedPipelineSchedule()))))));
}

TEST_F(PipelineSchedulingPassTest, MultipleProcsWithIOConstraint) {
  auto p = CreatePackage();
  auto make_proc = [p = p.get()](
                       std::string_view name, Channel* channel_in,
                       Channel* channel_out) -> absl::StatusOr<Proc*> {
    ProcBuilder pb(name, p);
    BValue tok = pb.Literal(Value::Token());
    BValue st = pb.StateElement("st", Value(UBits(0, 1)));
    BValue not_st = pb.Not(st);
    BValue recv = pb.ReceiveIf(channel_in, tok, st, SourceInfo(), "recv");
    BValue recv_tok = pb.TupleIndex(recv, 0);
    BValue recv_data = pb.TupleIndex(recv, 1);
    pb.SendIf(channel_out, recv_tok, st, recv_data, SourceInfo(), "send");
    return pb.Build({not_st});
  };

  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * ch0, p->CreateStreamingChannel("ch0", ChannelOps::kReceiveOnly,
                                               p->GetBitsType(32)));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * ch1, p->CreateStreamingChannel("ch1", ChannelOps::kSendReceive,
                                               p->GetBitsType(32)));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * ch2, p->CreateStreamingChannel("ch2", ChannelOps::kSendOnly,
                                               p->GetBitsType(32)));

  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc0, make_proc("proc0", ch0, ch1));
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc1, make_proc("proc1", ch1, ch2));

  EXPECT_THAT(
      RunPipelineSchedulingPass(
          p.get(), SchedulingOptions()
                       .pipeline_stages(4)
                       .add_constraint(IOConstraint(
                           /*source_channel=*/"ch0",
                           /*source_direction=*/IODirection::kReceive,
                           /*target_channel=*/"ch1",
                           /*target_direction=*/IODirection::kSend,
                           /*minimum_latency=*/3, /*maximum_latency=*/3))
                       .add_constraint(IOConstraint(
                           /*source_channel=*/"ch1",
                           /*source_direction=*/IODirection::kReceive,
                           /*target_channel=*/"ch2",
                           /*target_direction=*/IODirection::kSend,
                           /*minimum_latency=*/2, /*maximum_latency=*/2))),
      IsOkAndHolds(Pair(
          true,
          SchedulingContextWithElements(UnorderedElementsAre(
              Pair(proc0,
                   AllOf(VerifiedPipelineSchedule(),
                         LatencyIs(proc0->GetNode("recv").value_or(nullptr),
                                   proc0->GetNode("send").value_or(nullptr),
                                   Eq(3)))),
              Pair(proc1,
                   AllOf(VerifiedPipelineSchedule(),
                         LatencyIs(proc1->GetNode("recv").value_or(nullptr),
                                   proc1->GetNode("send").value_or(nullptr),
                                   Eq(2)))))))));
}

TEST_F(PipelineSchedulingPassTest, FdoWithMultipleProcs) {
  auto p = CreatePackage();
  auto make_func =
      [p = p.get()](std::string_view name) -> absl::StatusOr<Function*> {
    FunctionBuilder fb(name, p);
    Type* u64 = p->GetBitsType(64);
    fb.Add(fb.SMul(fb.Param("a", u64), fb.Param("b", u64)), fb.Param("c", u64));
    return fb.Build();
  };

  XLS_ASSERT_OK_AND_ASSIGN(Function * func0, make_func("proc0"));
  XLS_ASSERT_OK_AND_ASSIGN(Function * func1, make_func("proc1"));

  XLS_ASSERT_OK_AND_ASSIGN(std::filesystem::path yosys_path,
                           GetXlsRunfilePath("third_party/yosys/yosys"));
  XLS_ASSERT_OK_AND_ASSIGN(std::filesystem::path sta_path,
                           GetXlsRunfilePath("@org_theopenroadproject/opensta"));
  XLS_ASSERT_OK_AND_ASSIGN(
      std::filesystem::path lib_path,
      GetXlsRunfilePath("@com_google_skywater_pdk_sky130_fd_sc_hd/timing/"
                        "sky130_fd_sc_hd__ff_100C_1v95.lib"));
  auto scheduling_options =
      SchedulingOptions()
          .pipeline_stages(4)
          .clock_period_ps(2000)
          .use_fdo(true)
          .fdo_yosys_path(yosys_path.c_str())
          .fdo_sta_path(sta_path.c_str())
          .fdo_synthesis_libraries(lib_path.c_str())
          .fdo_default_driver_cell("sky130_fd_sc_hd__buf_2")
          .fdo_default_load("sky130_fd_sc_hd__dfxtp_2");
  XLS_ASSERT_OK_AND_ASSIGN(::xls::synthesis::Synthesizer * synthesizer,
                           SetUpSynthesizer(scheduling_options));

  EXPECT_THAT(RunPipelineSchedulingPass(p.get(), scheduling_options,
                                        /*synthesizer=*/synthesizer),
              IsOkAndHolds(
                  Pair(true, SchedulingContextWithElements(UnorderedElementsAre(
                                 Pair(func0, VerifiedPipelineSchedule()),
                                 Pair(func1, VerifiedPipelineSchedule()))))));
  delete synthesizer;
}

TEST_F(PipelineSchedulingPassTest, FunctionWithFFI) {
  auto p = CreatePackage();
  Type* u17 = p->GetBitsType(17);
  Type* u32 = p->GetBitsType(32);

  Function* ffi_fun;
  {
    FunctionBuilder fb("ffi_func", p.get());
    const BValue param_a = fb.Param("a", u32);
    const BValue param_b = fb.Param("b", u17);
    const BValue add = fb.Add(param_a, fb.ZeroExtend(param_b, 32));
    XLS_ASSERT_OK_AND_ASSIGN(ForeignFunctionData ffd,
                             ForeignFunctionDataCreateFromTemplate(
                                 "foo {fn} (.ma({a}), .mb{b}) .out({return})"));
    fb.SetForeignFunctionData(ffd);
    XLS_ASSERT_OK_AND_ASSIGN(ffi_fun, fb.BuildWithReturnValue(add));
  }

  Function* caller;
  {
    FunctionBuilder fb("caller", p.get());
    BValue param_a = fb.Param("a", u32);
    BValue param_b = fb.Param("b", u17);
    fb.Invoke({param_a, param_b}, ffi_fun);
    XLS_ASSERT_OK_AND_ASSIGN(caller, fb.Build());
  }

  EXPECT_THAT(RunPipelineSchedulingPass(p.get(),
                                        SchedulingOptions().pipeline_stages(2)),
              IsOkAndHolds(
                  Pair(true, SchedulingContextWithElements(UnorderedElementsAre(
                                 Pair(caller, VerifiedPipelineSchedule()))))));
}

TEST_F(PipelineSchedulingPassTest, MultiProcWithFFI) {
  auto p = CreatePackage();
  Type* u17 = p->GetBitsType(17);
  Type* u32 = p->GetBitsType(32);

  Function* ffi_fun;
  {
    FunctionBuilder fb("ffi_func", p.get());
    const BValue param_a = fb.Param("a", u32);
    const BValue param_b = fb.Param("b", u17);
    const BValue add = fb.Add(param_a, fb.ZeroExtend(param_b, 32));
    XLS_ASSERT_OK_AND_ASSIGN(ForeignFunctionData ffd,
                             ForeignFunctionDataCreateFromTemplate(
                                 "foo {fn} (.ma({a}), .mb{b}) .out({return})"));
    fb.SetForeignFunctionData(ffd);
    XLS_ASSERT_OK_AND_ASSIGN(ffi_fun, fb.BuildWithReturnValue(add));
  }

  Proc* caller;
  {
    XLS_ASSERT_OK_AND_ASSIGN(
        Channel * ch_a,
        p->CreateStreamingChannel("a", ChannelOps::kReceiveOnly, u32));
    XLS_ASSERT_OK_AND_ASSIGN(
        Channel * ch_b,
        p->CreateStreamingChannel("b", ChannelOps::kReceiveOnly, u17));
    XLS_ASSERT_OK_AND_ASSIGN(
        Channel * ch_c,
        p->CreateStreamingChannel("c", ChannelOps::kSendOnly, u32));

    ProcBuilder pb("caller", p.get());
    BValue recv_a = pb.Receive(ch_a, pb.Literal(Value::Token()));
    BValue recv_b = pb.Receive(ch_b, pb.Literal(Value::Token()));
    BValue result = pb.Invoke(
        {pb.TupleIndex(recv_a, 1), pb.TupleIndex(recv_b, 1)}, ffi_fun);
    pb.Send(ch_c, pb.Literal(Value::Token()), result);
    XLS_ASSERT_OK_AND_ASSIGN(caller, pb.Build({}));
  }

  // Test that multi-proc scheduling doesn't try to schedule the FFI function.
  EXPECT_THAT(
      RunPipelineSchedulingPass(
          p.get(),
          SchedulingOptions().pipeline_stages(2).schedule_all_procs(true)),
      IsOkAndHolds(
          Pair(true, SchedulingContextWithElements(UnorderedElementsAre(
                         Pair(caller, VerifiedPipelineSchedule()))))));
}

TEST_F(PipelineSchedulingPassTest, MultiProcScopedChannels) {
  auto p = CreatePackage();

  // Create leaf proc which adds one to its input.
  Proc* leaf;
  {
    TokenlessProcBuilder pb(NewStyleProc(), "myleaf", "tkn", p.get());
    XLS_ASSERT_OK_AND_ASSIGN(ReceiveChannelInterface * in,
                             pb.AddInputChannel("in", p->GetBitsType(32)));
    XLS_ASSERT_OK_AND_ASSIGN(SendChannelInterface * out,
                             pb.AddOutputChannel("out", p->GetBitsType(32)));

    // Create an optimization opportunity (constant folding).
    BValue one = pb.Add(pb.Literal(UBits(0, 32)), pb.Literal(UBits(1, 32)));

    pb.Send(out, pb.Add(pb.Receive(in), one));
    XLS_ASSERT_OK_AND_ASSIGN(leaf, pb.Build());
  }

  // Create a top proc which instantiates two leaf procs and sends an input
  // value through the chain, accumulates it and then sends to the output.
  Proc* top;
  {
    TokenlessProcBuilder pb(NewStyleProc(), "myproc", "tkn", p.get());
    XLS_ASSERT_OK_AND_ASSIGN(ReceiveChannelInterface * in,
                             pb.AddInputChannel("in", p->GetBitsType(32)));
    XLS_ASSERT_OK_AND_ASSIGN(SendChannelInterface * out,
                             pb.AddOutputChannel("out", p->GetBitsType(32)));

    XLS_ASSERT_OK_AND_ASSIGN(ChannelWithInterfaces tmp0_ch,
                             pb.AddChannel("tmp0", p->GetBitsType(32)));
    XLS_ASSERT_OK_AND_ASSIGN(ChannelWithInterfaces tmp1_ch,
                             pb.AddChannel("tmp1", p->GetBitsType(32)));

    BValue accum = pb.StateElement("accum", Value(UBits(0, 32)));
    XLS_ASSERT_OK(
        pb.InstantiateProc("inst0", leaf, {in, tmp0_ch.send_interface}));
    XLS_ASSERT_OK(pb.InstantiateProc(
        "inst1", leaf, {tmp0_ch.receive_interface, tmp1_ch.send_interface}));
    BValue next_accum = pb.Add(pb.Receive(tmp1_ch.receive_interface), accum);
    pb.Send(out, next_accum);
    XLS_ASSERT_OK(pb.SetAsTop());
    XLS_ASSERT_OK_AND_ASSIGN(top, pb.Build({next_accum}));
  }

  XLS_ASSERT_OK(RunPipelineSchedulingPass(
      p.get(),
      SchedulingOptions().pipeline_stages(2).schedule_all_procs(true)));
}

}  // namespace
}  // namespace xls
