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

#include "xls/codegen/finite_state_machine.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/log/log.h"
#include "xls/codegen/vast.h"
#include "xls/common/status/matchers.h"
#include "xls/simulation/verilog_test_base.h"

namespace xls {
namespace verilog {
namespace {

using status_testing::StatusIs;
using ::testing::HasSubstr;

constexpr char kTestName[] = "finite_state_machine_test";
constexpr char kTestdataPath[] = "xls/codegen/testdata";

class FiniteStateMachineTest : public VerilogTestBase {};

TEST_P(FiniteStateMachineTest, TrivialFsm) {
  VerilogFile f = NewVerilogFile();
  Module* module = f.Add(f.Make<Module>(SourceInfo(), TestBaseName()));

  LogicRef* clk =
      module->AddInput("clk", f.ScalarType(SourceInfo()), SourceInfo());
  FsmBuilder fsm("TrivialFsm", module, clk, UseSystemVerilog());
  auto foo = fsm.AddState("Foo");
  auto bar = fsm.AddState("Bar");

  foo->NextState(bar);

  XLS_ASSERT_OK(fsm.Build());
  VLOG(1) << f.Emit();
  ExpectVerilogEqualToGoldenFile(GoldenFilePath(kTestName, kTestdataPath),
                                 f.Emit());
}

TEST_P(FiniteStateMachineTest, TrivialFsmWithOutputs) {
  VerilogFile f = NewVerilogFile();
  Module* module = f.Add(f.Make<Module>(SourceInfo(), TestBaseName()));

  LogicRef* clk =
      module->AddInput("clk", f.ScalarType(SourceInfo()), SourceInfo());
  FsmBuilder fsm("TrivialFsm", module, clk, UseSystemVerilog());
  auto foo = fsm.AddState("Foo");
  auto bar = fsm.AddState("Bar");

  auto baz_out = fsm.AddOutput1("baz", /*default_value=*/0);
  auto qux_out = fsm.AddRegister("qux", 7);

  foo->NextState(bar);
  foo->SetOutput(baz_out, 1);

  bar->NextState(foo);
  // qux counts how many times the state "foo" has been entered.
  bar->SetRegisterNextAsExpression(
      qux_out,
      f.Add(qux_out->logic_ref, f.PlainLiteral(1, SourceInfo()), SourceInfo()));

  XLS_ASSERT_OK(fsm.Build());
  VLOG(1) << f.Emit();
  ExpectVerilogEqualToGoldenFile(GoldenFilePath(kTestName, kTestdataPath),
                                 f.Emit());
}

TEST_P(FiniteStateMachineTest, SimpleFsm) {
  VerilogFile f = NewVerilogFile();
  Module* module = f.Add(f.Make<Module>(SourceInfo(), TestBaseName()));

  LogicRef* clk =
      module->AddInput("clk", f.ScalarType(SourceInfo()), SourceInfo());
  LogicRef* rst_n =
      module->AddInput("rst_n", f.ScalarType(SourceInfo()), SourceInfo());
  LogicRef* ready_in =
      module->AddInput("ready_in", f.ScalarType(SourceInfo()), SourceInfo());
  LogicRef* done_out =
      module->AddOutput("done_out", f.ScalarType(SourceInfo()), SourceInfo());

  // The "done" output is a wire, create a reg copy for assignment in the FSM.
  LogicRef* done =
      module->AddReg("done", f.ScalarType(SourceInfo()), SourceInfo());
  module->Add<ContinuousAssignment>(SourceInfo(), done_out, done);

  FsmBuilder fsm("SimpleFsm", module, clk, UseSystemVerilog(),
                 Reset{rst_n, /*async=*/false, /*active_low=*/true});
  auto idle_state = fsm.AddState("Idle");
  auto busy_state = fsm.AddState("Busy");
  auto done_state = fsm.AddState("Done");

  auto fsm_done_out =
      fsm.AddExistingOutput(done,
                            /*default_value=*/f.PlainLiteral(0, SourceInfo()));

  idle_state->OnCondition(ready_in).NextState(busy_state);
  busy_state->NextState(done_state);
  done_state->SetOutput(fsm_done_out, 1);

  XLS_ASSERT_OK(fsm.Build());
  VLOG(1) << f.Emit();
  ExpectVerilogEqualToGoldenFile(GoldenFilePath(kTestName, kTestdataPath),
                                 f.Emit());
}

TEST_P(FiniteStateMachineTest, FsmWithNestedLogic) {
  VerilogFile f = NewVerilogFile();
  Module* module = f.Add(f.Make<Module>(SourceInfo(), TestBaseName()));

  LogicRef* clk =
      module->AddInput("clk", f.ScalarType(SourceInfo()), SourceInfo());
  LogicRef* rst_n =
      module->AddInput("rst_n", f.ScalarType(SourceInfo()), SourceInfo());
  LogicRef* foo =
      module->AddInput("foo", f.ScalarType(SourceInfo()), SourceInfo());
  LogicRef* bar =
      module->AddInput("bar", f.ScalarType(SourceInfo()), SourceInfo());
  LogicRef* qux =
      module->AddOutput("qux_out", f.ScalarType(SourceInfo()), SourceInfo());

  FsmBuilder fsm("NestLogic", module, clk, UseSystemVerilog(),
                 Reset{rst_n, /*async=*/false, /*active_low=*/true});
  auto a_state = fsm.AddState("A");
  auto b_state = fsm.AddState("B");

  auto fsm_qux_out = fsm.AddOutput("qux", /*width=*/8,
                                   /*default_value=*/0);

  a_state->OnCondition(foo)
      .NextState(b_state)

      // Nested Conditional
      .OnCondition(bar)
      .SetOutput(fsm_qux_out, 42)
      .Else()
      .SetOutput(fsm_qux_out, 123);
  b_state->OnCondition(f.LogicalAnd(foo, bar, SourceInfo())).NextState(a_state);

  XLS_ASSERT_OK(fsm.Build());

  module->Add<ContinuousAssignment>(SourceInfo(), qux, fsm_qux_out->logic_ref);

  VLOG(1) << f.Emit();
  ExpectVerilogEqualToGoldenFile(GoldenFilePath(kTestName, kTestdataPath),
                                 f.Emit());
}

TEST_P(FiniteStateMachineTest, CounterFsm) {
  VerilogFile f = NewVerilogFile();
  Module* module = f.Add(f.Make<Module>(SourceInfo(), TestBaseName()));

  LogicRef* clk =
      module->AddInput("clk", f.ScalarType(SourceInfo()), SourceInfo());
  LogicRef* rst =
      module->AddInput("rst", f.ScalarType(SourceInfo()), SourceInfo());
  FsmBuilder fsm("CounterFsm", module, clk, UseSystemVerilog(),
                 Reset{rst, /*async=*/true, /*active_low=*/false});
  auto foo = fsm.AddState("Foo");
  auto bar = fsm.AddState("Bar");
  auto qux = fsm.AddState("Qux");

  auto counter = fsm.AddDownCounter("counter", 6);
  foo->SetCounter(counter, 42).NextState(bar);
  bar->OnCounterIsZero(counter).NextState(qux);
  qux->NextState(foo);

  XLS_ASSERT_OK(fsm.Build());
  VLOG(1) << f.Emit();
  ExpectVerilogEqualToGoldenFile(GoldenFilePath(kTestName, kTestdataPath),
                                 f.Emit());
}

TEST_P(FiniteStateMachineTest, ComplexFsm) {
  VerilogFile f = NewVerilogFile();
  Module* module = f.Add(f.Make<Module>(SourceInfo(), TestBaseName()));

  LogicRef* clk =
      module->AddInput("clk", f.ScalarType(SourceInfo()), SourceInfo());
  LogicRef* foo_in =
      module->AddInput("foo_in", f.ScalarType(SourceInfo()), SourceInfo());
  LogicRef* bar_in =
      module->AddOutput("bar_in", f.ScalarType(SourceInfo()), SourceInfo());
  LogicRef* qux_in =
      module->AddOutput("qux_in", f.ScalarType(SourceInfo()), SourceInfo());

  FsmBuilder fsm("ComplexFsm", module, clk, UseSystemVerilog());
  auto hungry = fsm.AddState("Hungry");
  auto sad = fsm.AddState("Sad");
  auto happy = fsm.AddState("Happy");
  auto awake = fsm.AddState("Awake");
  auto sleepy = fsm.AddState("Sleepy");

  auto sleep = fsm.AddOutput1("sleep", 0);
  auto walk = fsm.AddOutput1("walk", 0);
  auto run = fsm.AddOutput1("run", 1);
  auto die = fsm.AddOutput1("die", 1);

  hungry->OnCondition(foo_in).NextState(happy).Else().NextState(sad);
  hungry->OnCondition(qux_in).SetOutput(walk, 0).SetOutput(die, 1);

  sad->NextState(awake);
  sad->SetOutput(walk, 0);
  sad->SetOutput(run, 1);

  awake->NextState(sleepy);

  sleepy->OnCondition(bar_in)
      .NextState(hungry)
      .ElseOnCondition(qux_in)
      .NextState(sad);

  happy->OnCondition(bar_in).SetOutput(die, 0);
  happy->OnCondition(foo_in).NextState(hungry).SetOutput(sleep, 1);

  XLS_ASSERT_OK(fsm.Build());
  VLOG(1) << f.Emit();
  ExpectVerilogEqualToGoldenFile(GoldenFilePath(kTestName, kTestdataPath),
                                 f.Emit());
}

TEST_P(FiniteStateMachineTest, OutputAssignments) {
  // Test various conditional and unconditional assignments of output regs in
  // different states. Verify the proper insertion of assignment of default
  // values to the outputs such that each code path has exactly one assignment
  // per output.
  VerilogFile f = NewVerilogFile();
  Module* module = f.Add(f.Make<Module>(SourceInfo(), TestBaseName()));

  LogicRef* clk =
      module->AddInput("clk", f.ScalarType(SourceInfo()), SourceInfo());
  LogicRef* rst_n =
      module->AddInput("rst_n", f.ScalarType(SourceInfo()), SourceInfo());

  LogicRef* a = module->AddInput("a", f.ScalarType(SourceInfo()), SourceInfo());
  LogicRef* b = module->AddInput("b", f.ScalarType(SourceInfo()), SourceInfo());

  FsmBuilder fsm("SimpleFsm", module, clk, UseSystemVerilog(),
                 Reset{rst_n, /*async=*/false, /*active_low=*/true});
  auto out_42 = fsm.AddOutput("out_42", /*width=*/8, /*default_value=*/42);
  auto out_123 = fsm.AddOutput("out_123", /*width=*/8, /*default_value=*/123);

  auto idle_state = fsm.AddState("Idle");
  idle_state->NextState(idle_state);

  {
    auto state = fsm.AddState("AssignmentToDefaultValue");
    state->SetOutput(out_42, 42);
    state->SetOutput(out_123, 123);
    state->NextState(idle_state);
  }

  {
    auto state = fsm.AddState("AssignmentToNondefaultValue");
    state->SetOutput(out_42, 33);
    state->SetOutput(out_123, 22);
    state->NextState(idle_state);
  }

  {
    auto state = fsm.AddState("ConditionalAssignToDefaultValue");
    state->OnCondition(a).SetOutput(out_42, 42);
    state->OnCondition(b).SetOutput(out_123, 123);
    state->NextState(idle_state);
  }

  {
    auto state = fsm.AddState("ConditionalAssignToNondefaultValue");
    state->OnCondition(a).SetOutput(out_42, 1);
    state->OnCondition(b).SetOutput(out_123, 2).Else().SetOutput(out_123, 4);
    state->NextState(idle_state);
  }

  {
    auto state = fsm.AddState("NestedConditionalAssignToNondefaultValue");
    state->OnCondition(a).OnCondition(b).SetOutput(out_42, 1).Else().SetOutput(
        out_123, 7);
    state->NextState(idle_state);
  }

  {
    auto state = fsm.AddState("AssignToNondefaultValueAtDifferentDepths");
    ConditionalFsmBlock& if_a = state->OnCondition(a);
    if_a.SetOutput(out_42, 1);
    if_a.Else().OnCondition(b).SetOutput(out_42, 77);
    state->NextState(idle_state);
  }

  XLS_ASSERT_OK(fsm.Build());
  VLOG(1) << f.Emit();

  ExpectVerilogEqualToGoldenFile(GoldenFilePath(kTestName, kTestdataPath),
                                 f.Emit());
}

TEST_P(FiniteStateMachineTest, MultipleAssignments) {
  VerilogFile f = NewVerilogFile();
  Module* module = f.Add(f.Make<Module>(SourceInfo(), TestBaseName()));

  LogicRef* clk =
      module->AddInput("clk", f.ScalarType(SourceInfo()), SourceInfo());
  LogicRef* rst_n =
      module->AddInput("rst_n", f.ScalarType(SourceInfo()), SourceInfo());

  LogicRef* a = module->AddInput("a", f.ScalarType(SourceInfo()), SourceInfo());

  FsmBuilder fsm("SimpleFsm", module, clk, UseSystemVerilog(),
                 Reset{rst_n, /*async=*/false, /*active_low=*/true});
  auto out = fsm.AddOutput("out", /*width=*/8, /*default_value=*/42);

  auto state = fsm.AddState("State");
  state->SetOutput(out, 123);
  state->OnCondition(a).SetOutput(out, 44);

  VLOG(1) << f.Emit();
  EXPECT_THAT(
      fsm.Build(),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("Output \"out\" may be assigned more than once")));
}

TEST_P(FiniteStateMachineTest, MultipleConditionalAssignments) {
  VerilogFile f = NewVerilogFile();
  Module* module = f.Add(f.Make<Module>(SourceInfo(), TestBaseName()));

  LogicRef* clk =
      module->AddInput("clk", f.ScalarType(SourceInfo()), SourceInfo());
  LogicRef* rst_n =
      module->AddInput("rst_n", f.ScalarType(SourceInfo()), SourceInfo());

  LogicRef* a = module->AddInput("a", f.ScalarType(SourceInfo()), SourceInfo());
  LogicRef* b = module->AddInput("b", f.ScalarType(SourceInfo()), SourceInfo());

  FsmBuilder fsm("SimpleFsm", module, clk, UseSystemVerilog(),
                 Reset{rst_n, /*async=*/false, /*active_low=*/true});
  auto out = fsm.AddOutput("out", /*width=*/8, /*default_value=*/42);

  auto state = fsm.AddState("State");
  state->OnCondition(a).SetOutput(out, 44);
  // Even setting output to same value is an error.
  state->OnCondition(b).SetOutput(out, 44);

  VLOG(1) << f.Emit();
  EXPECT_THAT(
      fsm.Build(),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("Output \"out\" may be assigned more than once")));
}

INSTANTIATE_TEST_SUITE_P(FiniteStateMachineTestInstantiation,
                         FiniteStateMachineTest,
                         testing::ValuesIn(kDefaultSimulationTargets),
                         ParameterizedTestName<FiniteStateMachineTest>);

}  // namespace
}  // namespace verilog
}  // namespace xls
