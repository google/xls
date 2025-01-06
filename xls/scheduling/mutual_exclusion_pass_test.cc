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

#include "xls/scheduling/mutual_exclusion_pass.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string_view>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xls/common/status/matchers.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/bits.h"
#include "xls/ir/channel.h"
#include "xls/ir/channel_ops.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/ir_matcher.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/op.h"
#include "xls/ir/package.h"
#include "xls/ir/value.h"
#include "xls/ir/verifier.h"
#include "xls/passes/cse_pass.h"
#include "xls/passes/dce_pass.h"
#include "xls/passes/optimization_pass.h"
#include "xls/passes/pass_base.h"
#include "xls/scheduling/scheduling_options.h"
#include "xls/scheduling/scheduling_pass.h"

namespace xls {
namespace {

using ::absl_testing::IsOkAndHolds;
using ::absl_testing::StatusIs;
using ::testing::HasSubstr;
using ::testing::Optional;

namespace m = ::xls::op_matchers;

class SimplificationPass : public OptimizationCompoundPass {
 public:
  explicit SimplificationPass()
      : OptimizationCompoundPass("simp", "Simplification") {
    Add<DeadCodeEliminationPass>();
    Add<CsePass>();
  }
};

using MutualExclusionPassTest = IrTestBase;

absl::StatusOr<bool> RunMutualExclusionPass(
    SchedulingUnit&& unit,
    const SchedulingPassOptions& options = SchedulingPassOptions()) {
  PassResults results;
  bool changed = false;
  bool subpass_changed;
  {
    SchedulingPassResults scheduling_results;
    XLS_ASSIGN_OR_RETURN(
        subpass_changed,
        MutualExclusionPass().Run(&unit, options, &scheduling_results));
    changed = changed || subpass_changed;
  }
  XLS_ASSIGN_OR_RETURN(
      subpass_changed,
      SimplificationPass().Run(unit.GetPackage(), OptimizationPassOptions(),
                               &results));
  changed = changed || subpass_changed;
  return changed;
}

absl::StatusOr<bool> RunMutualExclusionPass(
    Package* p,
    const SchedulingPassOptions& options = SchedulingPassOptions()) {
  return RunMutualExclusionPass(SchedulingUnit::CreateForWholePackage(p),
                                options);
}
absl::StatusOr<bool> RunMutualExclusionPass(
    FunctionBase* f,
    const SchedulingPassOptions& options = SchedulingPassOptions()) {
  return RunMutualExclusionPass(SchedulingUnit::CreateForSingleFunction(f),
                                options);
}

absl::StatusOr<bool> RunMutualExclusionPass(FunctionBase* f,
                                            const SchedulingOptions& options) {
  SchedulingPassOptions pass_options;
  pass_options.scheduling_options = options;
  return RunMutualExclusionPass(SchedulingUnit::CreateForSingleFunction(f),
                                pass_options);
}

absl::StatusOr<Proc*> CreateTwoParallelSendsProc(Package* p,
                                                 std::string_view name,
                                                 Channel* channel) {
  ProcBuilder pb(name, p);
  BValue tok = pb.StateElement("__token", Value::Token());
  BValue st = pb.StateElement("__state", Value(UBits(0, 1)));
  BValue not_st = pb.Not(st);
  BValue lit50 = pb.Literal(UBits(50, 32));
  BValue lit60 = pb.Literal(UBits(60, 32));
  BValue send0 = pb.SendIf(channel, tok, st, lit50);
  BValue send1 = pb.SendIf(channel, tok, not_st, lit60);
  return pb.Build({pb.AfterAll({send0, send1}), not_st});
}

absl::StatusOr<Node*> FindOp(FunctionBase* f, Op op) {
  Node* result = nullptr;
  for (Node* node : f->nodes()) {
    if (node->op() == op) {
      result = node;
      break;
    }
  }
  if (result == nullptr) {
    return absl::InternalError("Could not find op");
  }
  return result;
}

int64_t NumberOfOp(FunctionBase* f, Op op) {
  int64_t result = 0;
  for (Node* node : f->nodes()) {
    if (node->op() == op) {
      ++result;
    }
  }
  return result;
}

TEST_F(MutualExclusionPassTest, TwoParallelSends) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * test_channel,
      p->CreateStreamingChannel("test_channel", ChannelOps::kSendOnly,
                                p->GetBitsType(32)));
  XLS_ASSERT_OK_AND_ASSIGN(
      Proc * proc, CreateTwoParallelSendsProc(p.get(), "main", test_channel));
  EXPECT_THAT(RunMutualExclusionPass(proc), IsOkAndHolds(true));
  EXPECT_EQ(NumberOfOp(proc, Op::kSend), 1);
  XLS_EXPECT_OK(VerifyProc(proc, true));
}

TEST_F(MutualExclusionPassTest,
       TwoParallelSendsWithSmallRlimitAndRequiredMerging) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * test_channel,
      p->CreateStreamingChannel("test_channel", ChannelOps::kSendOnly,
                                p->GetBitsType(32)));
  XLS_ASSERT_OK_AND_ASSIGN(
      Proc * proc, CreateTwoParallelSendsProc(p.get(), "main", test_channel));
  EXPECT_THAT(RunMutualExclusionPass(
                  proc, SchedulingOptions().mutual_exclusion_z3_rlimit(1)),
              IsOkAndHolds(true));
  EXPECT_EQ(NumberOfOp(proc, Op::kSend), 1);
  XLS_EXPECT_OK(VerifyProc(proc, true));
}

TEST_F(MutualExclusionPassTest,
       TwoParallelSendsWithSmallRlimitAndOptionalMerging) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * test_channel,
      p->CreateStreamingChannel(
          "test_channel", ChannelOps::kSendOnly, p->GetBitsType(32),
          /*initial_values=*/{}, /*fifo_config=*/std::nullopt,
          /*flow_control=*/FlowControl::kReadyValid,
          /*strictness=*/ChannelStrictness::kArbitraryStaticOrder));
  XLS_ASSERT_OK_AND_ASSIGN(
      Proc * proc, CreateTwoParallelSendsProc(p.get(), "main", test_channel));
  EXPECT_THAT(RunMutualExclusionPass(
                  proc, SchedulingOptions().mutual_exclusion_z3_rlimit(1)),
              IsOkAndHolds(false));
  EXPECT_EQ(NumberOfOp(proc, Op::kSend), 2);
}

TEST_F(MutualExclusionPassTest, ThreeParallelSends) {
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> p, ParsePackage(R"(
     package test_module

     chan test_channel(
       bits[32], id=0, kind=streaming, ops=send_only,
       flow_control=ready_valid, metadata="""""")

     top proc main(__state: bits[2], init={0}) {
       literal.1: bits[2] = literal(value=1)
       add.2: bits[2] = add(literal.1, __state)
       zero_ext.3: bits[32] = zero_ext(add.2, new_bit_count=32)
       literal.4: bits[32] = literal(value=50)
       literal.5: bits[32] = literal(value=60)
       literal.6: bits[32] = literal(value=70)
       eq.7: bits[1] = eq(zero_ext.3, literal.4)
       eq.8: bits[1] = eq(zero_ext.3, literal.5)
       eq.9: bits[1] = eq(zero_ext.3, literal.6)
       __token: token = literal(value=token, id=1000)
       send.10: token = send(__token, literal.4, predicate=eq.7, channel=test_channel)
       send.11: token = send(__token, literal.5, predicate=eq.8, channel=test_channel)
       send.12: token = send(__token, literal.6, predicate=eq.9, channel=test_channel)
       next (add.2)
     }
  )"));
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, p->GetTopAsProc());
  EXPECT_THAT(RunMutualExclusionPass(proc), IsOkAndHolds(true));
  EXPECT_EQ(NumberOfOp(proc, Op::kSend), 1);
  XLS_EXPECT_OK(VerifyProc(proc, true));
  XLS_ASSERT_OK_AND_ASSIGN(Node * one_hot_select, FindOp(proc, Op::kOneHotSel));
  EXPECT_THAT(
      one_hot_select,
      m::OneHotSelect(m::Concat(*proc->GetNode("eq.7"), *proc->GetNode("eq.8"),
                                *proc->GetNode("eq.9")),
                      {// Note the reversed order, because concat is
                       // in diagrammatic order but one_hot_sel uses
                       // little endian.
                       *proc->GetNode("literal.6"), *proc->GetNode("literal.5"),
                       *proc->GetNode("literal.4")}));
}

TEST_F(MutualExclusionPassTest, TwoSequentialSends) {
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> p, ParsePackage(R"(
     package test_module

     chan test_channel(
       bits[32], id=0, kind=streaming, ops=send_only,
       flow_control=ready_valid, metadata="""""")

     top proc main(__state: bits[1], init={0}) {
       not.1: bits[1] = not(__state)
       literal.2: bits[32] = literal(value=50)
       literal.3: bits[32] = literal(value=60)
       __token: token = literal(value=token, id=1000)
       send.4: token = send(__token, literal.2, predicate=__state, channel=test_channel)
       send.5: token = send(send.4, literal.3, predicate=not.1, channel=test_channel)
       next (not.1)
     }
  )"));
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, p->GetTopAsProc());
  EXPECT_THAT(RunMutualExclusionPass(proc), IsOkAndHolds(true));
  EXPECT_EQ(NumberOfOp(proc, Op::kSend), 1);
}

TEST_F(MutualExclusionPassTest, TwoSequentialSendsWithInterveningIO) {
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> p, ParsePackage(R"(
     package test_module

     chan test_channel(
       bits[32], id=0, kind=streaming, ops=send_only,
       flow_control=ready_valid, metadata="""""")

     chan other_channel(
       bits[32], id=1, kind=streaming, ops=send_only,
       flow_control=ready_valid, metadata="""""")

     top proc main(__state: bits[1], init={0}) {
       not.1: bits[1] = not(__state)
       literal.2: bits[32] = literal(value=50)
       literal.3: bits[32] = literal(value=60)
       __token: token = literal(value=token, id=1000)
       send.4: token = send(__token, literal.2, predicate=__state, channel=test_channel)
       send.5: token = send(send.4, literal.2, channel=other_channel)
       send.6: token = send(send.5, literal.3, predicate=not.1, channel=test_channel)
       next (not.1)
     }
  )"));
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, p->GetTopAsProc());
  EXPECT_THAT(RunMutualExclusionPass(proc), IsOkAndHolds(false));
  EXPECT_EQ(NumberOfOp(proc, Op::kSend), 3);
}

TEST_F(MutualExclusionPassTest, Complex) {
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> p, ParsePackage(R"(
     package test_module

     chan test_channel(
       bits[2], id=0, kind=streaming, ops=send_only,
       flow_control=ready_valid, metadata="""""")

     chan other_channel(
       bits[2], id=1, kind=streaming, ops=send_only,
       flow_control=ready_valid, metadata="""""")

     top proc main(__state: bits[2], init={0}) {
       literal.1: bits[2] = literal(value=1)
       add.2: bits[2] = add(__state, literal.1)
       literal.3: bits[2] = literal(value=0)
       literal.4: bits[2] = literal(value=1)
       literal.5: bits[2] = literal(value=2)
       eq.6: bits[1] = eq(__state, literal.3)
       eq.7: bits[1] = eq(__state, literal.4)
       eq.8: bits[1] = eq(__state, literal.5)
       __token: token = literal(value=token, id=1000)
       send.9: token = send(__token, literal.3, predicate=eq.6, channel=test_channel)
       send.10: token = send(send.9, literal.3, channel=other_channel)
       send.11: token = send(send.10, literal.4, predicate=eq.7, channel=test_channel)
       send.12: token = send(__token, literal.5, predicate=eq.8, channel=test_channel)
       next (add.2)
     }
  )"));
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, p->GetTopAsProc());
  EXPECT_THAT(RunMutualExclusionPass(proc), IsOkAndHolds(true));
  EXPECT_EQ(NumberOfOp(proc, Op::kSend), 3);
}

TEST_F(MutualExclusionPassTest, TwoParallelReceives) {
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> p, ParsePackage(R"(
     package test_module

     chan test_channel(
       bits[32], id=0, kind=streaming, ops=send_receive,
       flow_control=ready_valid, metadata="""""")

     top proc main(__state: bits[1], init={0}) {
       __token: token = literal(value=token, id=1000)
       not.1: bits[1] = not(__state)
       receive.2: (token, bits[32]) = receive(__token, predicate=__state, channel=test_channel)
       tuple_index.3: token = tuple_index(receive.2, index=0)
       tuple_index.4: bits[32] = tuple_index(receive.2, index=1)
       receive.5: (token, bits[32]) = receive(__token, predicate=not.1, channel=test_channel)
       tuple_index.6: token = tuple_index(receive.5, index=0)
       tuple_index.7: bits[32] = tuple_index(receive.5, index=1)
       add.8: bits[32] = add(tuple_index.4, tuple_index.7)
       after_all.9: token = after_all(tuple_index.3, tuple_index.6)
       send.10: token = send(after_all.9, add.8, channel=test_channel)
       next (not.1)
     }
  )"));
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, p->GetTopAsProc());
  EXPECT_THAT(RunMutualExclusionPass(proc), IsOkAndHolds(true));
  EXPECT_EQ(NumberOfOp(proc, Op::kReceive), 1);
  XLS_EXPECT_OK(VerifyProc(proc, true));
}

TEST_F(MutualExclusionPassTest, TwoSequentialReceives) {
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> p, ParsePackage(R"(
     package test_module

     chan test_channel(
       bits[32], id=0, kind=streaming, ops=send_receive,
       flow_control=ready_valid, metadata="""""")

     top proc main(__state: bits[1], init={0}) {
       __token: token = literal(value=token, id=1000)
       not.1: bits[1] = not(__state)
       receive.2: (token, bits[32]) = receive(__token, predicate=__state, channel=test_channel)
       tuple_index.3: token = tuple_index(receive.2, index=0)
       tuple_index.4: bits[32] = tuple_index(receive.2, index=1)
       receive.5: (token, bits[32]) = receive(tuple_index.3, predicate=not.1, channel=test_channel)
       tuple_index.6: token = tuple_index(receive.5, index=0)
       tuple_index.7: bits[32] = tuple_index(receive.5, index=1)
       add.8: bits[32] = add(tuple_index.4, tuple_index.7)
       after_all.9: token = after_all(tuple_index.3, tuple_index.6)
       send.10: token = send(after_all.9, add.8, channel=test_channel)
       next (not.1)
     }
  )"));
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, p->GetTopAsProc());
  EXPECT_THAT(RunMutualExclusionPass(proc), IsOkAndHolds(true));
  EXPECT_EQ(NumberOfOp(proc, Op::kReceive), 1);
}

TEST_F(MutualExclusionPassTest, TwoSequentialReceivesWithInterveningIO) {
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> p, ParsePackage(R"(
     package test_module

     chan test_channel(
       bits[32], id=0, kind=streaming, ops=send_receive,
       flow_control=ready_valid, metadata="""""")

     chan other_channel(
       bits[32], id=1, kind=streaming, ops=send_only,
       flow_control=ready_valid, metadata="""""")

     top proc main(__state: bits[1], init={0}) {
       __token: token = literal(value=token, id=1000)
       not.1: bits[1] = not(__state)
       receive.2: (token, bits[32]) = receive(__token, predicate=__state, channel=test_channel)
       tuple_index.3: token = tuple_index(receive.2, index=0)
       tuple_index.4: bits[32] = tuple_index(receive.2, index=1)
       send.5: token = send(tuple_index.3, tuple_index.4, channel=other_channel)
       receive.6: (token, bits[32]) = receive(send.5, predicate=not.1, channel=test_channel)
       tuple_index.7: token = tuple_index(receive.6, index=0)
       tuple_index.8: bits[32] = tuple_index(receive.6, index=1)
       add.9: bits[32] = add(tuple_index.4, tuple_index.8)
       after_all.10: token = after_all(tuple_index.3, tuple_index.7)
       send.11: token = send(after_all.10, add.9, channel=test_channel)
       next (not.1)
     }
  )"));
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, p->GetTopAsProc());
  EXPECT_THAT(RunMutualExclusionPass(proc), IsOkAndHolds(false));
  EXPECT_EQ(NumberOfOp(proc, Op::kReceive), 2);
}

TEST_F(MutualExclusionPassTest, TwoSequentialReceivesWithDataDep) {
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> p, ParsePackage(R"(
     package test_module

     chan test_channel(
       bits[1], id=0, kind=streaming, ops=send_receive,
       flow_control=ready_valid, metadata="""""")

     top proc main(__state: bits[1], init={0}) {
       __token: token = literal(value=token, id=1000)
       not.1: bits[1] = not(__state)
       receive.2: (token, bits[1]) = receive(__token, predicate=__state, channel=test_channel)
       tuple_index.3: token = tuple_index(receive.2, index=0)
       tuple_index.4: bits[1] = tuple_index(receive.2, index=1)
       and.11: bits[1] = and(not.1, tuple_index.4)
       receive.5: (token, bits[1]) = receive(tuple_index.3, predicate=and.11, channel=test_channel)
       tuple_index.6: token = tuple_index(receive.5, index=0)
       tuple_index.7: bits[1] = tuple_index(receive.5, index=1)
       add.8: bits[1] = add(tuple_index.4, tuple_index.7)
       after_all.9: token = after_all(tuple_index.3, tuple_index.6)
       send.10: token = send(after_all.9, add.8, channel=test_channel)
       next (not.1)
     }
  )"));
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, p->GetTopAsProc());
  EXPECT_THAT(RunMutualExclusionPass(proc), IsOkAndHolds(false));
  EXPECT_EQ(NumberOfOp(proc, Op::kReceive), 2);
}

TEST_F(MutualExclusionPassTest, TwoReceivesDependingOnReceive) {
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> p, ParsePackage(R"(
     package test_module

     chan test_channel(
       bits[32], id=0, kind=streaming, ops=send_receive,
       flow_control=ready_valid, metadata="""""")

     chan other_channel(
       bits[32], id=1, kind=streaming, ops=receive_only,
       flow_control=ready_valid, metadata="""""")

     top proc main(__state: bits[1], init={0}) {
       __token: token = literal(value=token, id=1000)
       not.1: bits[1] = not(__state)
       receive.2: (token, bits[32]) = receive(__token, channel=other_channel)
       tuple_index.3: token = tuple_index(receive.2, index=0)
       tuple_index.4: bits[32] = tuple_index(receive.2, index=1)
       receive.5: (token, bits[32]) = receive(tuple_index.3, predicate=__state, channel=test_channel)
       tuple_index.7: token = tuple_index(receive.5, index=0)
       tuple_index.8: bits[32] = tuple_index(receive.5, index=1)
       receive.9: (token, bits[32]) = receive(tuple_index.3, predicate=not.1, channel=test_channel)
       tuple_index.10: token = tuple_index(receive.9, index=0)
       tuple_index.11: bits[32] = tuple_index(receive.9, index=1)
       add.12: bits[32] = add(tuple_index.4, tuple_index.8)
       add.13: bits[32] = add(add.12, tuple_index.11)
       after_all.14: token = after_all(tuple_index.7, tuple_index.10)
       send.15: token = send(after_all.14, add.13, channel=test_channel)
       next (not.1)
     }
  )"));
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, p->GetTopAsProc());
  EXPECT_THAT(RunMutualExclusionPass(proc), IsOkAndHolds(true));
  EXPECT_EQ(NumberOfOp(proc, Op::kReceive), 2);
}

TEST_F(MutualExclusionPassTest, SelectPredicates) {
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> p, ParsePackage(R"(
     package test_module

     top proc main(
       selector1: bits[1],
       selector2: bits[1],
       input1: bits[1],
       input2: bits[1],
       input3: bits[1],
       init={0, 0, 0, 0, 0}
     ) {
       not_input1: bits[1] = not(input1)
       not_input2: bits[1] = not(input2)
       not_input3: bits[1] = not(input3)
       select1: bits[1] = sel(selector1, cases=[not_input1, not_input2])
       select2: bits[1] = sel(selector2, cases=[select1, not_input3])
       zero: bits[1] = literal(value=0)
       next (zero, zero, zero, zero, select2)
     }
  )"));
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, p->GetTopAsProc());
  Predicates preds;
  XLS_ASSERT_OK(AddSelectPredicates(&preds, proc));
  EXPECT_THAT(
      preds.GetPredicate(*proc->GetNode("selector1")),
      Optional(m::And(m::Eq(*proc->GetNode("selector2"), m::Literal(0)))));
  EXPECT_FALSE(preds.GetPredicate(*proc->GetNode("selector2")).has_value());
  EXPECT_THAT(
      preds.GetPredicate(*proc->GetNode("input1")),
      Optional(m::And(m::Eq(*proc->GetNode("selector1"), m::Literal(0)),
                      m::Eq(*proc->GetNode("selector2"), m::Literal(0)))));
  EXPECT_THAT(
      preds.GetPredicate(*proc->GetNode("input2")),
      Optional(m::And(m::Eq(*proc->GetNode("selector1"), m::Literal(1)),
                      m::Eq(*proc->GetNode("selector2"), m::Literal(0)))));
  EXPECT_THAT(
      preds.GetPredicate(*proc->GetNode("input3")),
      Optional(m::And(m::Eq(*proc->GetNode("selector2"), m::Literal(1)))));
  EXPECT_THAT(
      preds.GetPredicate(*proc->GetNode("not_input1")),
      Optional(m::And(m::Eq(*proc->GetNode("selector1"), m::Literal(0)),
                      m::Eq(*proc->GetNode("selector2"), m::Literal(0)))));
  EXPECT_THAT(
      preds.GetPredicate(*proc->GetNode("not_input2")),
      Optional(m::And(m::Eq(*proc->GetNode("selector1"), m::Literal(1)),
                      m::Eq(*proc->GetNode("selector2"), m::Literal(0)))));
  EXPECT_THAT(
      preds.GetPredicate(*proc->GetNode("not_input3")),
      Optional(m::And(m::Eq(*proc->GetNode("selector2"), m::Literal(1)))));
  EXPECT_THAT(
      preds.GetPredicate(*proc->GetNode("select1")),
      Optional(m::And(m::Eq(*proc->GetNode("selector2"), m::Literal(0)))));
  EXPECT_FALSE(preds.GetPredicate(*proc->GetNode("select2")).has_value());
  EXPECT_FALSE(preds.GetPredicate(*proc->GetNode("zero")).has_value());
}

TEST_F(MutualExclusionPassTest, SelectPredicatesCaseFanout) {
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> p, ParsePackage(R"(
     package test_module

     top proc main(
       selector1: bits[1],
       selector2: bits[1],
       input1: bits[1],
       input2: bits[1],
       input3: bits[1],
       init={0, 0, 0, 0, 0}
     ) {
       not_input1: bits[1] = not(input1)
       not_input2: bits[1] = not(input2)
       not_input3: bits[1] = not(input3)
       select1: bits[1] = sel(selector1, cases=[not_input1, not_input2])
       select2: bits[1] = sel(selector2, cases=[select1, not_input3])
       anded: bits[1] = and(select2, not_input3)
       zero: bits[1] = literal(value=0)
       next (zero, zero, zero, zero, anded)
     }
  )"));
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, p->GetTopAsProc());
  Predicates preds;
  XLS_ASSERT_OK(AddSelectPredicates(&preds, proc));
  EXPECT_THAT(
      preds.GetPredicate(*proc->GetNode("selector1")),
      Optional(m::And(m::Eq(*proc->GetNode("selector2"), m::Literal(0)))));
  EXPECT_FALSE(preds.GetPredicate(*proc->GetNode("selector2")).has_value());
  EXPECT_THAT(
      preds.GetPredicate(*proc->GetNode("input1")),
      Optional(m::And(m::Eq(*proc->GetNode("selector1"), m::Literal(0)),
                      m::Eq(*proc->GetNode("selector2"), m::Literal(0)))));
  EXPECT_THAT(
      preds.GetPredicate(*proc->GetNode("input2")),
      Optional(m::And(m::Eq(*proc->GetNode("selector1"), m::Literal(1)),
                      m::Eq(*proc->GetNode("selector2"), m::Literal(0)))));
  EXPECT_FALSE(preds.GetPredicate(*proc->GetNode("input3")).has_value());
  EXPECT_THAT(
      preds.GetPredicate(*proc->GetNode("not_input1")),
      Optional(m::And(m::Eq(*proc->GetNode("selector1"), m::Literal(0)),
                      m::Eq(*proc->GetNode("selector2"), m::Literal(0)))));
  EXPECT_THAT(
      preds.GetPredicate(*proc->GetNode("not_input2")),
      Optional(m::And(m::Eq(*proc->GetNode("selector1"), m::Literal(1)),
                      m::Eq(*proc->GetNode("selector2"), m::Literal(0)))));
  EXPECT_FALSE(preds.GetPredicate(*proc->GetNode("not_input3")).has_value());
  EXPECT_THAT(
      preds.GetPredicate(*proc->GetNode("select1")),
      Optional(m::And(m::Eq(*proc->GetNode("selector2"), m::Literal(0)))));
  EXPECT_FALSE(preds.GetPredicate(*proc->GetNode("select2")).has_value());
  EXPECT_FALSE(preds.GetPredicate(*proc->GetNode("anded")).has_value());
  EXPECT_FALSE(preds.GetPredicate(*proc->GetNode("zero")).has_value());
}

TEST_F(MutualExclusionPassTest, SelectPredicatesImplicitUses) {
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> p, ParsePackage(R"(
     package test_module

     top proc main(
       selector1: bits[1],
       selector2: bits[1],
       input1: bits[1],
       input2: bits[1],
       input3: bits[1],
       init={0, 0, 0, 0, 0}
     ) {
       not_input1: bits[1] = not(input1)
       not_input2: bits[1] = not(input2)
       not_input3: bits[1] = not(input3)
       select1: bits[1] = sel(selector1, cases=[not_input1, not_input2])
       select2: bits[1] = sel(selector2, cases=[select1, not_input3])
       zero: bits[1] = literal(value=0)
       next (zero, zero, zero, input2, select2)
     }
  )"));
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, p->GetTopAsProc());
  Predicates preds;
  XLS_ASSERT_OK(AddSelectPredicates(&preds, proc));
  EXPECT_THAT(
      preds.GetPredicate(*proc->GetNode("selector1")),
      Optional(m::And(m::Eq(*proc->GetNode("selector2"), m::Literal(0)))));
  EXPECT_FALSE(preds.GetPredicate(*proc->GetNode("selector2")).has_value());
  EXPECT_THAT(
      preds.GetPredicate(*proc->GetNode("input1")),
      Optional(m::And(m::Eq(*proc->GetNode("selector1"), m::Literal(0)),
                      m::Eq(*proc->GetNode("selector2"), m::Literal(0)))));
  EXPECT_FALSE(preds.GetPredicate(*proc->GetNode("input2")).has_value());
  EXPECT_THAT(
      preds.GetPredicate(*proc->GetNode("input3")),
      Optional(m::And(m::Eq(*proc->GetNode("selector2"), m::Literal(1)))));
  EXPECT_THAT(
      preds.GetPredicate(*proc->GetNode("not_input1")),
      Optional(m::And(m::Eq(*proc->GetNode("selector1"), m::Literal(0)),
                      m::Eq(*proc->GetNode("selector2"), m::Literal(0)))));
  EXPECT_THAT(
      preds.GetPredicate(*proc->GetNode("not_input2")),
      Optional(m::And(m::Eq(*proc->GetNode("selector1"), m::Literal(1)),
                      m::Eq(*proc->GetNode("selector2"), m::Literal(0)))));
  EXPECT_THAT(
      preds.GetPredicate(*proc->GetNode("not_input3")),
      Optional(m::And(m::Eq(*proc->GetNode("selector2"), m::Literal(1)))));
  EXPECT_THAT(
      preds.GetPredicate(*proc->GetNode("select1")),
      Optional(m::And(m::Eq(*proc->GetNode("selector2"), m::Literal(0)))));
  EXPECT_FALSE(preds.GetPredicate(*proc->GetNode("select2")).has_value());
  EXPECT_FALSE(preds.GetPredicate(*proc->GetNode("zero")).has_value());
}

TEST_F(MutualExclusionPassTest, TwoProcsBothHavingTwoParallelSends) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * test_channel0,
      p->CreateStreamingChannel("test_channel0", ChannelOps::kSendOnly,
                                p->GetBitsType(32)));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * test_channel1,
      p->CreateStreamingChannel("test_channel1", ChannelOps::kSendOnly,
                                p->GetBitsType(32)));
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc0, CreateTwoParallelSendsProc(
                                             p.get(), "proc0", test_channel0));
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc1, CreateTwoParallelSendsProc(
                                             p.get(), "proc1", test_channel1));
  EXPECT_THAT(RunMutualExclusionPass(p.get()), IsOkAndHolds(true));
  EXPECT_EQ(NumberOfOp(proc0, Op::kSend), 1);
  EXPECT_EQ(NumberOfOp(proc1, Op::kSend), 1);
  XLS_EXPECT_OK(VerifyProc(proc0, true));
  XLS_EXPECT_OK(VerifyProc(proc1, true));
}

TEST_F(MutualExclusionPassTest, RequiredMergeFails) {
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> p, ParsePackage(R"(
     package test_module

     chan cin0(
       bits[32], id=0, kind=streaming, ops=receive_only,
       flow_control=ready_valid, metadata="""""")

     chan cin1(
       bits[32], id=1, kind=streaming, ops=receive_only,
       flow_control=ready_valid, metadata="""""")

     top proc main(s: bits[1], init={0}) {
       not_s: bits[1] = not(s)
       tok: token = literal(value=token)
       rcv0_not: (token, bits[32]) = receive(tok, predicate=not_s, channel=cin0)
       rcv1_not: (token, bits[32]) = receive(tok, predicate=not_s, channel=cin1)
       tok0: token = tuple_index(rcv0_not, index=0)
       tok1: token = tuple_index(rcv1_not, index=0)
       rcv0: (token, bits[32]) = receive(tok1, predicate=s, channel=cin0)
       rcv1: (token, bits[32]) = receive(tok0, predicate=s, channel=cin1)
       next (not_s)
     }
  )"));
  EXPECT_THAT(RunMutualExclusionPass(p.get()),
              StatusIs(absl::StatusCode::kFailedPrecondition,
                       HasSubstr("without creating a cycle")));
}

TEST_F(MutualExclusionPassTest, AvoidsCycles) {
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> p, ParsePackage(R"(
     package test_module

     chan cin0(
       bits[32], id=0, kind=streaming, ops=receive_only,
       flow_control=ready_valid, strictness=arbitrary_static_order,
       metadata="""""")

     chan cin1(
       bits[32], id=1, kind=streaming, ops=receive_only,
       flow_control=ready_valid, strictness=arbitrary_static_order,
       metadata="""""")

     top proc main(s: bits[1], init={0}) {
       not_s: bits[1] = not(s)
       tok: token = literal(value=token)
       rcv0_not: (token, bits[32]) = receive(tok, predicate=not_s, channel=cin0)
       rcv1_not: (token, bits[32]) = receive(tok, predicate=not_s, channel=cin1)
       tok0: token = tuple_index(rcv0_not, index=0)
       tok1: token = tuple_index(rcv1_not, index=0)
       rcv0: (token, bits[32]) = receive(tok1, predicate=s, channel=cin0)
       rcv1: (token, bits[32]) = receive(tok0, predicate=s, channel=cin1)
       next (not_s)
     }
  )"));
  EXPECT_THAT(RunMutualExclusionPass(p.get()), IsOkAndHolds(true));
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, p->GetTopAsProc());
  EXPECT_EQ(NumberOfOp(proc, Op::kReceive), 3);
}

}  // namespace
}  // namespace xls
