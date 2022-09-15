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

#include "xls/passes/mutual_exclusion_pass.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/statusor.h"
#include "xls/common/status/matchers.h"
#include "xls/ir/function.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/ir_matcher.h"
#include "xls/ir/ir_test_base.h"
#include "xls/passes/cse_pass.h"
#include "xls/passes/dce_pass.h"
#include "xls/passes/passes.h"

namespace xls {
namespace {

using status_testing::IsOkAndHolds;
namespace m = ::xls::op_matchers;

class SimplificationPass : public CompoundPass {
 public:
  explicit SimplificationPass() : CompoundPass("simp", "Simplification") {
    Add<DeadCodeEliminationPass>();
    Add<CsePass>();
  }
};

class MutualExclusionPassTest : public IrTestBase {
 protected:
  MutualExclusionPassTest() = default;

  absl::StatusOr<bool> Run(FunctionBase* f) {
    PassResults results;
    bool changed = false;
    bool subpass_changed;
    XLS_ASSIGN_OR_RETURN(
        subpass_changed,
        MutualExclusionPass().RunOnFunctionBase(f, PassOptions(), &results));
    changed |= subpass_changed;
    XLS_ASSIGN_OR_RETURN(
        subpass_changed,
        SimplificationPass().Run(f->package(), PassOptions(), &results));
    changed |= subpass_changed;
    return changed;
  }
};

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
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> p, ParsePackage(R"(
     package test_module

     chan test_channel(
       bits[32], id=0, kind=streaming, ops=send_only,
       flow_control=ready_valid, metadata="""""")

     top proc main(__token: token, __state: bits[1], init={0}) {
       not.1: bits[1] = not(__state)
       literal.2: bits[32] = literal(value=50)
       literal.3: bits[32] = literal(value=60)
       send.4: token = send(__token, literal.2, predicate=__state, channel_id=0)
       send.5: token = send(__token, literal.3, predicate=not.1, channel_id=0)
       after_all.6: token = after_all(send.4, send.5)
       next (after_all.6, not.1)
     }
  )"));
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, p->GetTopAsProc());
  EXPECT_THAT(Run(proc), IsOkAndHolds(true));
  EXPECT_EQ(NumberOfOp(proc, Op::kSend), 1);
  XLS_EXPECT_OK(VerifyProc(proc, true));
}

TEST_F(MutualExclusionPassTest, ThreeParallelSends) {
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> p, ParsePackage(R"(
     package test_module

     chan test_channel(
       bits[32], id=0, kind=streaming, ops=send_only,
       flow_control=ready_valid, metadata="""""")

     top proc main(__token: token, __state: bits[2], init={0}) {
       literal.1: bits[2] = literal(value=1)
       add.2: bits[2] = add(literal.1, __state)
       zero_ext.3: bits[32] = zero_ext(add.2, new_bit_count=32)
       literal.4: bits[32] = literal(value=50)
       literal.5: bits[32] = literal(value=60)
       literal.6: bits[32] = literal(value=70)
       eq.7: bits[1] = eq(zero_ext.3, literal.4)
       eq.8: bits[1] = eq(zero_ext.3, literal.5)
       eq.9: bits[1] = eq(zero_ext.3, literal.6)
       send.10: token = send(__token, literal.4, predicate=eq.7, channel_id=0)
       send.11: token = send(__token, literal.5, predicate=eq.8, channel_id=0)
       send.12: token = send(__token, literal.6, predicate=eq.9, channel_id=0)
       after_all.13: token = after_all(send.10, send.11, send.12)
       next (after_all.13, add.2)
     }
  )"));
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, p->GetTopAsProc());
  EXPECT_THAT(Run(proc), IsOkAndHolds(true));
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

     top proc main(__token: token, __state: bits[1], init={0}) {
       not.1: bits[1] = not(__state)
       literal.2: bits[32] = literal(value=50)
       literal.3: bits[32] = literal(value=60)
       send.4: token = send(__token, literal.2, predicate=__state, channel_id=0)
       send.5: token = send(send.4, literal.3, predicate=not.1, channel_id=0)
       next (send.5, not.1)
     }
  )"));
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, p->GetTopAsProc());
  EXPECT_THAT(Run(proc), IsOkAndHolds(false));
  EXPECT_EQ(NumberOfOp(proc, Op::kSend), 2);
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

     top proc main(__token: token, __state: bits[1], init={0}) {
       not.1: bits[1] = not(__state)
       literal.2: bits[32] = literal(value=50)
       literal.3: bits[32] = literal(value=60)
       send.4: token = send(__token, literal.2, predicate=__state, channel_id=0)
       send.5: token = send(send.4, literal.2, channel_id=1)
       send.6: token = send(send.5, literal.3, predicate=not.1, channel_id=0)
       next (send.6, not.1)
     }
  )"));
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, p->GetTopAsProc());
  EXPECT_THAT(Run(proc), IsOkAndHolds(false));
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

     top proc main(__token: token, __state: bits[2], init={0}) {
       literal.1: bits[2] = literal(value=1)
       add.2: bits[2] = add(__state, literal.1)
       literal.3: bits[2] = literal(value=0)
       literal.4: bits[2] = literal(value=1)
       literal.5: bits[2] = literal(value=2)
       eq.6: bits[1] = eq(__state, literal.3)
       eq.7: bits[1] = eq(__state, literal.4)
       eq.8: bits[1] = eq(__state, literal.5)
       send.9: token = send(__token, literal.3, predicate=eq.6, channel_id=0)
       send.10: token = send(send.9, literal.3, channel_id=1)
       send.11: token = send(send.10, literal.4, predicate=eq.7, channel_id=0)
       send.12: token = send(__token, literal.5, predicate=eq.8, channel_id=0)
       after_all.13: token = after_all(send.11, send.12)
       next (after_all.13, add.2)
     }
  )"));
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, p->GetTopAsProc());
  EXPECT_THAT(Run(proc), IsOkAndHolds(true));
  EXPECT_EQ(NumberOfOp(proc, Op::kSend), 3);
}

TEST_F(MutualExclusionPassTest, TwoParallelReceives) {
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> p, ParsePackage(R"(
     package test_module

     chan test_channel(
       bits[32], id=0, kind=streaming, ops=send_receive,
       flow_control=ready_valid, metadata="""""")

     top proc main(__token: token, __state: bits[1], init={0}) {
       not.1: bits[1] = not(__state)
       receive.2: (token, bits[32]) = receive(__token, predicate=__state, channel_id=0)
       tuple_index.3: token = tuple_index(receive.2, index=0)
       tuple_index.4: bits[32] = tuple_index(receive.2, index=1)
       receive.5: (token, bits[32]) = receive(__token, predicate=not.1, channel_id=0)
       tuple_index.6: token = tuple_index(receive.5, index=0)
       tuple_index.7: bits[32] = tuple_index(receive.5, index=1)
       add.8: bits[32] = add(tuple_index.4, tuple_index.7)
       after_all.9: token = after_all(tuple_index.3, tuple_index.6)
       send.10: token = send(after_all.9, add.8, channel_id=0)
       next (send.10, not.1)
     }
  )"));
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, p->GetTopAsProc());
  EXPECT_THAT(Run(proc), IsOkAndHolds(true));
  EXPECT_EQ(NumberOfOp(proc, Op::kReceive), 1);
  XLS_EXPECT_OK(VerifyProc(proc, true));
}

TEST_F(MutualExclusionPassTest, TwoSequentialReceives) {
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> p, ParsePackage(R"(
     package test_module

     chan test_channel(
       bits[32], id=0, kind=streaming, ops=send_receive,
       flow_control=ready_valid, metadata="""""")

     top proc main(__token: token, __state: bits[1], init={0}) {
       not.1: bits[1] = not(__state)
       receive.2: (token, bits[32]) = receive(__token, predicate=__state, channel_id=0)
       tuple_index.3: token = tuple_index(receive.2, index=0)
       tuple_index.4: bits[32] = tuple_index(receive.2, index=1)
       receive.5: (token, bits[32]) = receive(tuple_index.3, predicate=not.1, channel_id=0)
       tuple_index.6: token = tuple_index(receive.5, index=0)
       tuple_index.7: bits[32] = tuple_index(receive.5, index=1)
       add.8: bits[32] = add(tuple_index.4, tuple_index.7)
       after_all.9: token = after_all(tuple_index.3, tuple_index.6)
       send.10: token = send(after_all.9, add.8, channel_id=0)
       next (send.10, not.1)
     }
  )"));
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, p->GetTopAsProc());
  EXPECT_THAT(Run(proc), IsOkAndHolds(false));
  EXPECT_EQ(NumberOfOp(proc, Op::kReceive), 2);
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

     top proc main(__token: token, __state: bits[1], init={0}) {
       not.1: bits[1] = not(__state)
       receive.2: (token, bits[32]) = receive(__token, predicate=__state, channel_id=0)
       tuple_index.3: token = tuple_index(receive.2, index=0)
       tuple_index.4: bits[32] = tuple_index(receive.2, index=1)
       send.5: token = send(tuple_index.3, tuple_index.4, channel_id=1)
       receive.6: (token, bits[32]) = receive(send.5, predicate=not.1, channel_id=0)
       tuple_index.7: token = tuple_index(receive.6, index=0)
       tuple_index.8: bits[32] = tuple_index(receive.6, index=1)
       add.9: bits[32] = add(tuple_index.4, tuple_index.8)
       after_all.10: token = after_all(tuple_index.3, tuple_index.7)
       send.11: token = send(after_all.10, add.9, channel_id=0)
       next (send.11, not.1)
     }
  )"));
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, p->GetTopAsProc());
  EXPECT_THAT(Run(proc), IsOkAndHolds(false));
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

     top proc main(__token: token, __state: bits[1], init={0}) {
       not.1: bits[1] = not(__state)
       receive.2: (token, bits[32]) = receive(__token, channel_id=1)
       tuple_index.3: token = tuple_index(receive.2, index=0)
       tuple_index.4: bits[32] = tuple_index(receive.2, index=1)
       receive.5: (token, bits[32]) = receive(tuple_index.3, predicate=__state, channel_id=0)
       tuple_index.7: token = tuple_index(receive.5, index=0)
       tuple_index.8: bits[32] = tuple_index(receive.5, index=1)
       receive.9: (token, bits[32]) = receive(tuple_index.3, predicate=not.1, channel_id=0)
       tuple_index.10: token = tuple_index(receive.9, index=0)
       tuple_index.11: bits[32] = tuple_index(receive.9, index=1)
       add.12: bits[32] = add(tuple_index.4, tuple_index.8)
       add.13: bits[32] = add(add.12, tuple_index.11)
       after_all.14: token = after_all(tuple_index.7, tuple_index.10)
       send.15: token = send(after_all.14, add.13, channel_id=0)
       next (send.15, not.1)
     }
  )"));
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, p->GetTopAsProc());
  EXPECT_THAT(Run(proc), IsOkAndHolds(true));
  EXPECT_EQ(NumberOfOp(proc, Op::kReceive), 2);
}

}  // namespace
}  // namespace xls
