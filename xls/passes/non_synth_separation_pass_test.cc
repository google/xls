// Copyright 2025 The XLS Authors
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

#include "xls/passes/non_synth_separation_pass.h"

#include <optional>
#include <utility>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "xls/common/fuzzing/fuzztest.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "xls/common/status/matchers.h"
#include "xls/common/status/status_macros.h"
#include "xls/fuzzer/ir_fuzzer/ir_fuzz_domain.h"
#include "xls/fuzzer/ir_fuzzer/ir_fuzz_test_library.h"
#include "xls/interpreter/interpreter_proc_runtime.h"
#include "xls/ir/bits.h"
#include "xls/ir/channel_ops.h"
#include "xls/ir/function.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/ir_matcher.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/nodes.h"
#include "xls/ir/package.h"
#include "xls/ir/source_location.h"
#include "xls/ir/value.h"
#include "xls/passes/dce_pass.h"
#include "xls/passes/optimization_pass.h"
#include "xls/passes/pass_base.h"
#include "xls/passes/pass_test_helpers.h"

namespace xls {
namespace {

using ::absl_testing::IsOkAndHolds;
using ::absl_testing::StatusIs;

namespace m = xls::op_matchers;

class NonSynthSeparationPassTest : public IrTestBase {
 protected:
  NonSynthSeparationPassTest() = default;

  absl::StatusOr<bool> Run(Package* p) {
    PassResults results;
    OptimizationContext context;
    return NonSynthSeparationPass().Run(p, OptimizationPassOptions(), &results,
                                        context);
  }
  absl::StatusOr<bool> RunWithDCE(Package* p) {
    PassResults results;
    OptimizationContext context;
    OptimizationCompoundPass pass("TestPass", TestName());
    bool run_result = false;
    pass.Add<RecordIfPassChanged<NonSynthSeparationPass>>(&run_result);
    pass.Add<DeadCodeEliminationPass>();
    XLS_ASSIGN_OR_RETURN(
        bool compound_result,
        pass.Run(p, OptimizationPassOptions(), &results, context));
    return compound_result && run_result;
  }
};

TEST_F(NonSynthSeparationPassTest, FunctionWithNoNonSynthNodesNotCloned) {
  auto p = CreatePackage();
  FunctionBuilder fb("f", p.get());
  fb.Literal(UBits(1, 1));
  XLS_ASSERT_OK(fb.Build());
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(false));
  EXPECT_THAT(p->GetFunctionBases().size(), 1);
}

TEST_F(NonSynthSeparationPassTest, EmptyProcNotCloned) {
  auto p = CreatePackage();
  ProcBuilder pb("proc1", p.get());
  XLS_ASSERT_OK(pb.Build());
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(false));
  EXPECT_THAT(p->GetFunctionBases().size(), 1);
}

TEST_F(NonSynthSeparationPassTest, FunctionIsCloned) {
  auto p = CreatePackage();
  FunctionBuilder fb("f", p.get());
  fb.Assert(fb.Literal(Value::Token()), fb.Literal(UBits(0, 1)), "");
  fb.Literal(UBits(1, 1));
  XLS_ASSERT_OK(fb.Build());
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  XLS_ASSERT_OK(p->GetFunction("non_synth_f"));
}

TEST_F(NonSynthSeparationPassTest, ProcIsCloned) {
  auto p = CreatePackage();
  ProcBuilder pb("proc1", p.get());
  pb.Assert(pb.Literal(Value::Token()), pb.Literal(UBits(0, 1)), "");
  pb.Literal(UBits(1, 1));
  XLS_ASSERT_OK(pb.Build());
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  XLS_ASSERT_OK(p->GetFunction("non_synth_proc1"));
}

TEST_F(NonSynthSeparationPassTest, ProcIsClonedWithSend) {
  auto p = CreatePackage();
  ProcBuilder pb("proc1", p.get());
  pb.Assert(pb.Literal(Value::Token()), pb.Literal(UBits(0, 1)), "");
  XLS_ASSERT_OK_AND_ASSIGN(
      auto channel, p->CreateSingleValueChannel(
                        "channel", ChannelOps::kSendOnly, p->GetBitsType(1)));
  BValue send =
      pb.Send(channel, pb.Literal(Value::Token()), pb.Literal(UBits(0, 1)));
  pb.Identity(send);
  XLS_ASSERT_OK(pb.Build());
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  XLS_ASSERT_OK_AND_ASSIGN(Function * non_synth_proc1,
                           p->GetFunction("non_synth_proc1"));
  // Don't actually give the sends parameters.
  EXPECT_THAT(non_synth_proc1->GetType()->parameters(), testing::IsEmpty());
  EXPECT_THAT(non_synth_proc1->nodes(),
              testing::Contains(m::Identity(m::Literal(Value::Token()))));
}

TEST_F(NonSynthSeparationPassTest, ProcIsClonedWithReceive) {
  auto p = CreatePackage();
  ProcBuilder pb("proc1", p.get());
  pb.Assert(pb.Literal(Value::Token()), pb.Literal(UBits(0, 1)), "");
  XLS_ASSERT_OK_AND_ASSIGN(
      auto channel,
      p->CreateSingleValueChannel("channel", ChannelOps::kReceiveOnly,
                                  p->GetBitsType(12)));
  BValue receive = pb.Receive(channel, pb.Literal(Value::Token()));
  pb.Identity(receive);
  XLS_ASSERT_OK(pb.Build());
  ScopedRecordIr sri(p.get());
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  XLS_ASSERT_OK_AND_ASSIGN(Function * non_synth_proc1,
                           p->GetFunction("non_synth_proc1"));
  EXPECT_THAT(non_synth_proc1->nodes(),
              testing::Contains(m::Identity(m::Type("(token, bits[12])"))));
}

TEST_F(NonSynthSeparationPassTest, ProcIsClonedWithStateRead) {
  auto p = CreatePackage();
  ProcBuilder pb("proc1", p.get());
  pb.Assert(pb.Literal(Value::Token()), pb.Literal(UBits(0, 1)), "");
  BValue state_read = pb.StateElement("state_read", UBits(0, 1));
  pb.Identity(state_read);
  XLS_ASSERT_OK(pb.Build());
  ScopedRecordIr sri(p.get());
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  XLS_ASSERT_OK_AND_ASSIGN(Function * non_synth_proc1,
                           p->GetFunction("non_synth_proc1"));
  XLS_ASSERT_OK_AND_ASSIGN(Node * identity_node,
                           non_synth_proc1->GetNode("identity.11"));
  EXPECT_THAT(identity_node, m::Identity(m::Param()));
}

TEST_F(NonSynthSeparationPassTest, FunctionsAreClonedWithNoDuplicateNames) {
  auto p = CreatePackage();
  FunctionBuilder fb("f", p.get());
  fb.Assert(fb.Literal(Value::Token()), fb.Literal(UBits(0, 1)), "");
  fb.Literal(UBits(1, 1));
  XLS_ASSERT_OK(fb.Build());
  FunctionBuilder non_synth_fb("non_synth_f", p.get());
  non_synth_fb.Assert(non_synth_fb.Literal(Value::Token()),
                      non_synth_fb.Literal(UBits(0, 1)), "");
  non_synth_fb.Literal(UBits(1, 1));
  XLS_ASSERT_OK(non_synth_fb.Build());
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  XLS_ASSERT_OK(p->GetFunction("f"));
  XLS_ASSERT_OK(p->GetFunction("non_synth_f"));
  XLS_ASSERT_OK(p->GetFunction("_non_synth_f"));
  XLS_ASSERT_OK(p->GetFunction("non_synth_non_synth_f"));
}

TEST_F(NonSynthSeparationPassTest, NonSynthFunctionReturnValueIsMadeUseless) {
  auto p = CreatePackage();
  FunctionBuilder fb("f", p.get());
  fb.Assert(fb.Literal(Value::Token()), fb.Literal(UBits(0, 1)), "");
  fb.Literal(UBits(1, 1));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  XLS_ASSERT_OK_AND_ASSIGN(Function * non_synth_f,
                           p->GetFunction("non_synth_f"));
  EXPECT_THAT(f->return_value(), m::Literal(UBits(1, 1)));
  EXPECT_THAT(non_synth_f->return_value(),
              m::Tuple(std::vector<testing::Matcher<const Node*>>()));
}

TEST_F(NonSynthSeparationPassTest, InvokeToNonSynthFunctionIsInserted) {
  auto p = CreatePackage();
  FunctionBuilder fb("f", p.get());
  fb.Assert(fb.Literal(Value::Token()), fb.Literal(UBits(0, 1)), "");
  fb.Literal(UBits(1, 1));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  XLS_ASSERT_OK(p->GetFunction("non_synth_f"));
  XLS_ASSERT_OK_AND_ASSIGN(Node * invoke_node, f->GetNode("invoke.10"));
  EXPECT_THAT(invoke_node,
              m::Invoke(std::vector<testing::Matcher<const Node*>>()));
}

TEST_F(NonSynthSeparationPassTest,
       InvokeToNonSynthFunctionIsInsertedWithParams) {
  auto p = CreatePackage();
  FunctionBuilder fb("f", p.get());
  fb.Assert(fb.Literal(Value::Token()), fb.Literal(UBits(0, 1)), "");
  fb.Param("p0", p->GetBitsType(1));
  fb.Param("p1", p->GetTupleType({}));
  fb.Param("p2", p->GetArrayType(1, p->GetBitsType(1)));
  fb.Literal(UBits(1, 1));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  XLS_ASSERT_OK(p->GetFunction("non_synth_f"));
  XLS_ASSERT_OK_AND_ASSIGN(Node * invoke_node, f->GetNode("invoke.16"));
  EXPECT_THAT(invoke_node,
              m::Invoke(m::Param("p0"), m::Param("p1"), m::Param("p2")));
}

TEST_F(NonSynthSeparationPassTest, InvokeToNonSynthProcIsInsertedWithParams) {
  auto p = CreatePackage();
  ProcBuilder pb("proc1", p.get());
  pb.Assert(pb.Literal(Value::Token()), pb.Literal(UBits(0, 1)), "");
  XLS_ASSERT_OK_AND_ASSIGN(
      auto channel,
      p->CreateSingleValueChannel("channel", ChannelOps::kReceiveOnly,
                                  p->GetBitsType(1)));
  BValue receive = pb.Receive(channel, pb.Literal(Value::Token()));
  pb.Identity(receive);
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc1, pb.Build());
  ScopedRecordIr sri(p.get());
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  XLS_ASSERT_OK(p->GetFunction("non_synth_proc1"));
  XLS_ASSERT_OK_AND_ASSIGN(Node * invoke_node, proc1->GetNode("invoke.18"));
  EXPECT_THAT(invoke_node, m::Invoke(m::Tuple(m::TupleIndex(m::Receive()))));
}

// This test fails if the nodes are not topologically sorted.
TEST_F(NonSynthSeparationPassTest, NonSynthNodesAreRemoved) {
  auto p = CreatePackage();
  FunctionBuilder fb("f", p.get());
  BValue assert_token = fb.Literal(Value::Token());
  BValue assert_condition = fb.Literal(UBits(0, 1));
  BValue assert = fb.Assert(assert_token, assert_condition, "");
  BValue trace_token = fb.Literal(Value::Token());
  BValue trace_condition = fb.Literal(UBits(0, 1));
  BValue trace = fb.Trace(trace_token, trace_condition, {}, "");
  BValue cover_condition = fb.Literal(UBits(0, 1));
  BValue cover = fb.Cover(cover_condition, "cover_label");
  fb.Tuple({assert, trace, cover});
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  XLS_ASSERT_OK_AND_ASSIGN(Function * non_synth_f,
                           p->GetFunction("non_synth_f"));
  EXPECT_THAT(f->return_value(),
              m::Tuple(m::Literal(), m::Literal(),
                       m::Tuple(std::vector<testing::Matcher<const Node*>>())));
  EXPECT_THAT(f->node_count(), 8);
  EXPECT_THAT(non_synth_f->node_count(), 10);
}

TEST_F(NonSynthSeparationPassTest, NonSynthNodesAreRemovedWithDCE) {
  auto p = CreatePackage();
  FunctionBuilder fb("f", p.get());
  BValue assert_token = fb.Literal(Value::Token());
  BValue assert_condition = fb.Literal(UBits(0, 1));
  BValue assert = fb.Assert(assert_token, assert_condition, "");
  BValue trace_token = fb.Literal(Value::Token());
  BValue trace_condition = fb.Literal(UBits(0, 1));
  BValue trace = fb.Trace(trace_token, trace_condition, {}, "");
  BValue cover_condition = fb.Literal(UBits(0, 1));
  BValue cover = fb.Cover(cover_condition, "cover_label");
  fb.Tuple({assert, trace, cover});
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  ASSERT_THAT(RunWithDCE(p.get()), IsOkAndHolds(true));
  XLS_ASSERT_OK_AND_ASSIGN(Function * non_synth_f,
                           p->GetFunction("non_synth_f"));
  EXPECT_THAT(f->return_value(),
              m::Tuple(m::Literal(), m::Literal(),
                       m::Tuple(std::vector<testing::Matcher<const Node*>>())));
  EXPECT_THAT(f->node_count(), 5);
  EXPECT_THAT(non_synth_f->node_count(), 9);
}

// This test fails if the nodes are not topologically sorted.
TEST_F(NonSynthSeparationPassTest, GateNodesAreReplacedWithSelects) {
  auto p = CreatePackage();
  FunctionBuilder fb("f", p.get());
  fb.Assert(fb.Literal(Value::Token()), fb.Literal(UBits(0, 1)), "");
  BValue gate_condition = fb.Literal(UBits(0, 1));
  BValue gate_data = fb.Literal(UBits(0, 2));
  BValue gate = fb.Gate(gate_condition, gate_data);
  fb.Identity(gate);
  fb.Literal(UBits(0, 3));
  XLS_ASSERT_OK(fb.Build());
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  XLS_ASSERT_OK_AND_ASSIGN(Function * non_synth_f,
                           p->GetFunction("non_synth_f"));
  XLS_ASSERT_OK_AND_ASSIGN(Node * select_node, non_synth_f->GetNode("sel.20"));
  EXPECT_THAT(select_node,
              m::Select(m::Literal(UBits(0, 1)), {m::Literal(UBits(0, 2))},
                        m::Literal(UBits(0, 2))));
}

TEST_F(NonSynthSeparationPassTest, ProcHasSameAssertBehavior) {
  auto p = CreatePackage();
  ProcBuilder pb("proc1", p.get());
  BValue read = pb.StateElement("foo", UBits(4, 32));
  pb.Assert(pb.Literal(Value::Token(), SourceInfo(), "tok"),
            pb.Ne(pb.Literal(UBits(7, 32)), read), "foobar");
  pb.Next(read, pb.Add(read, pb.Literal(UBits(1, 32))));
  XLS_ASSERT_OK(pb.Build().status());
  ScopedRecordIr sri(p.get());
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  XLS_ASSERT_OK_AND_ASSIGN(auto eval,
                           CreateInterpreterSerialProcRuntime(p.get()));
  XLS_ASSERT_OK(eval->Tick());
  XLS_ASSERT_OK(eval->Tick());
  XLS_ASSERT_OK(eval->Tick());
  ASSERT_THAT(eval->Tick(), StatusIs(absl::StatusCode::kAborted, "foobar"));
}

TEST_F(NonSynthSeparationPassTest, ProcHasSameTraceBehavior) {
  auto p = CreatePackage();
  ProcBuilder pb("proc1", p.get());
  BValue read = pb.StateElement("foo", UBits(4, 32));
  pb.Trace(pb.Literal(Value::Token(), SourceInfo(), "tok"),
           pb.Literal(UBits(1, 1), SourceInfo(), "true"), {read},
           "value_is_{}");
  pb.Next(read, pb.Add(read, pb.Literal(UBits(1, 32))));
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, pb.Build());
  ScopedRecordIr sri(p.get());
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  XLS_ASSERT_OK_AND_ASSIGN(auto eval,
                           CreateInterpreterSerialProcRuntime(p.get()));
  XLS_ASSERT_OK(eval->Tick());
  XLS_ASSERT_OK(eval->Tick());
  XLS_ASSERT_OK(eval->Tick());
  EXPECT_THAT(eval->GetInterpreterEvents(proc).GetTraceMessageStrings(),
              testing::ElementsAre("value_is_4", "value_is_5", "value_is_6"));
}
TEST_F(NonSynthSeparationPassTest, ProcWithTokenStateElement) {
  auto p = CreatePackage();
  ProcBuilder pb("proc1", p.get());
  BValue read = pb.StateElement("foo", Value::Token());
  BValue v1 = pb.StateElement("bar", UBits(3, 32));
  pb.Next(read, pb.Literal(Value::Token()));
  pb.Next(v1, pb.Literal(UBits(1, 32)));
  pb.Assert(read, pb.Eq(v1, pb.Literal(UBits(1, 32))), "foobar");
  XLS_ASSERT_OK(pb.Build().status());
  ScopedRecordIr sri(p.get());
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  XLS_ASSERT_OK_AND_ASSIGN(Function * ns, p->GetFunction("non_synth_proc1"));
  EXPECT_THAT(ns->params(), testing::ElementsAre(m::Type("bits[32]")));
}

void IrFuzzNonSynthSeparation(FuzzPackageWithArgs fuzz_package_with_args) {
  NonSynthSeparationPass pass;
  OptimizationPassChangesOutputs(std::move(fuzz_package_with_args), pass);
}
FUZZ_TEST(IrFuzzTest, IrFuzzNonSynthSeparation)
    .WithDomains(IrFuzzDomainWithArgs(/*arg_set_count=*/10));

}  // namespace
}  // namespace xls
