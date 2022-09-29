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

#include "xls/passes/dfe_pass.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xls/common/status/matchers.h"
#include "xls/ir/function.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/ir_test_base.h"
#include "xls/passes/pass_base.h"

namespace xls {
namespace {

using status_testing::IsOkAndHolds;
using status_testing::StatusIs;

class DeadFunctionEliminationPassTest : public IrTestBase {
 protected:
  DeadFunctionEliminationPassTest() = default;

  absl::StatusOr<bool> Run(Package* p) {
    PassResults results;
    return DeadFunctionEliminationPass().Run(p, PassOptions(), &results);
  }

  absl::StatusOr<Function*> MakeFunction(std::string_view name, Package* p) {
    FunctionBuilder fb(name, p);
    fb.Param("arg", p->GetBitsType(32));
    return fb.Build();
  }
};

TEST_F(DeadFunctionEliminationPassTest, NoDeadFunctions) {
  auto p = std::make_unique<Package>(TestName());
  XLS_ASSERT_OK_AND_ASSIGN(Function * a, MakeFunction("a", p.get()));
  XLS_ASSERT_OK_AND_ASSIGN(Function * b, MakeFunction("b", p.get()));
  FunctionBuilder fb("the_entry", p.get());
  BValue x = fb.Param("x", p->GetBitsType(32));
  fb.Add(fb.Invoke({x}, a), fb.Invoke({x}, b));
  XLS_ASSERT_OK(fb.Build().status());
  XLS_ASSERT_OK(p->SetTopByName("the_entry"));

  EXPECT_EQ(p->functions().size(), 3);
  EXPECT_THAT(Run(p.get()), IsOkAndHolds(false));
  EXPECT_EQ(p->functions().size(), 3);
}

TEST_F(DeadFunctionEliminationPassTest, OneDeadFunction) {
  auto p = std::make_unique<Package>(TestName());
  XLS_ASSERT_OK_AND_ASSIGN(Function * a, MakeFunction("a", p.get()));
  XLS_ASSERT_OK(MakeFunction("dead", p.get()).status());
  FunctionBuilder fb("the_entry", p.get());
  BValue x = fb.Param("x", p->GetBitsType(32));
  fb.Add(fb.Invoke({x}, a), fb.Invoke({x}, a));
  XLS_ASSERT_OK(fb.Build().status());
  XLS_ASSERT_OK(p->SetTopByName("the_entry"));

  EXPECT_EQ(p->functions().size(), 3);
  EXPECT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_EQ(p->functions().size(), 2);
}

TEST_F(DeadFunctionEliminationPassTest, OneDeadFunctionButNoEntry) {
  // If no entry function is specified, then DFS cannot happen as all functions
  // are live.
  auto p = std::make_unique<Package>(TestName());
  XLS_ASSERT_OK_AND_ASSIGN(Function * a, MakeFunction("a", p.get()));
  XLS_ASSERT_OK(MakeFunction("dead", p.get()).status());
  FunctionBuilder fb("blah", p.get());
  BValue x = fb.Param("x", p->GetBitsType(32));
  fb.Add(fb.Invoke({x}, a), fb.Invoke({x}, a));
  XLS_ASSERT_OK(fb.Build().status());

  EXPECT_EQ(p->functions().size(), 3);
  EXPECT_THAT(Run(p.get()), IsOkAndHolds(false));
  EXPECT_EQ(p->functions().size(), 3);
}

TEST_F(DeadFunctionEliminationPassTest, ProcCallingFunction) {
  auto p = std::make_unique<Package>(TestName());
  XLS_ASSERT_OK_AND_ASSIGN(Function * f,
                           MakeFunction("called_by_proc", p.get()));
  XLS_ASSERT_OK(MakeFunction("not_called_by_proc", p.get()).status());

  TokenlessProcBuilder b(TestName(), "tkn", p.get());
  b.StateElement("st", Value(UBits(0, 32)));
  BValue invoke = b.Invoke({b.GetStateParam(0)}, f);
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, b.Build({invoke}));
  XLS_ASSERT_OK(p->SetTop(proc));

  EXPECT_EQ(p->functions().size(), 2);
  XLS_EXPECT_OK(p->GetFunction("not_called_by_proc").status());

  EXPECT_THAT(Run(p.get()), IsOkAndHolds(true));

  EXPECT_EQ(p->functions().size(), 1);
  EXPECT_THAT(p->GetFunction("not_called_by_proc"),
              StatusIs(absl::StatusCode::kNotFound));
}

TEST_F(DeadFunctionEliminationPassTest, MultipleProcs) {
  auto p = std::make_unique<Package>(TestName());
  Type* u32 = p->GetBitsType(32);

  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * ch_in_a,
      p->CreateStreamingChannel("in_a", ChannelOps::kReceiveOnly, u32));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * ch_out_a,
      p->CreateStreamingChannel("out_a", ChannelOps::kSendOnly, u32));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * ch_a_to_b,
      p->CreateStreamingChannel("a_to_b", ChannelOps::kSendReceive, u32));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * ch_b_to_a,
      p->CreateStreamingChannel("b_to_a", ChannelOps::kSendReceive, u32));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * ch_in_c,
      p->CreateStreamingChannel("in_c", ChannelOps::kReceiveOnly, u32));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * ch_out_c,
      p->CreateStreamingChannel("out_c", ChannelOps::kSendOnly, u32));

  {
    TokenlessProcBuilder b("A", "tkn", p.get());
    b.Send(ch_a_to_b, b.Receive(ch_in_a));
    b.Send(ch_out_a, b.Receive(ch_b_to_a));
    XLS_ASSERT_OK_AND_ASSIGN(Proc * a, b.Build({}));
    XLS_ASSERT_OK(p->SetTop(a));
  }
  {
    TokenlessProcBuilder b("B", "tkn", p.get());
    b.Send(ch_b_to_a, b.Receive(ch_a_to_b));
    XLS_ASSERT_OK(b.Build({}).status());
  }
  {
    TokenlessProcBuilder b("C", "tkn", p.get());
    b.Send(ch_out_c, b.Receive(ch_in_c));
    XLS_ASSERT_OK(b.Build({}).status());
  }

  // Proc "C" should be removed as well as the its channels.
  EXPECT_EQ(p->procs().size(), 3);
  XLS_EXPECT_OK(p->GetProc("C").status());
  EXPECT_EQ(p->channels().size(), 6);
  XLS_EXPECT_OK(p->GetChannel("in_c").status());
  XLS_EXPECT_OK(p->GetChannel("out_c").status());

  EXPECT_THAT(Run(p.get()), IsOkAndHolds(true));

  EXPECT_EQ(p->procs().size(), 2);
  EXPECT_THAT(p->GetProc("C"), StatusIs(absl::StatusCode::kNotFound));
  EXPECT_EQ(p->channels().size(), 4);
  EXPECT_THAT(p->GetChannel("in_c").status(),
              StatusIs(absl::StatusCode::kNotFound));
  EXPECT_THAT(p->GetChannel("out_c"), StatusIs(absl::StatusCode::kNotFound));
}

TEST_F(DeadFunctionEliminationPassTest, MapAndCountedFor) {
  // If no entry function is specified, then DFS cannot happen as all functions
  // are live.
  auto p = std::make_unique<Package>(TestName());
  XLS_ASSERT_OK_AND_ASSIGN(Function * a, MakeFunction("a", p.get()));
  Function* body;
  {
    FunctionBuilder fb("jesse_the_loop_body", p.get());
    fb.Param("i", p->GetBitsType(32));
    fb.Param("arg", p->GetBitsType(32));
    fb.Literal(UBits(123, 32));
    XLS_ASSERT_OK_AND_ASSIGN(body, fb.Build());
  }
  FunctionBuilder fb("the_entry", p.get());
  BValue x = fb.Param("x", p->GetBitsType(32));
  BValue ar = fb.Param("ar", p->GetArrayType(42, p->GetBitsType(32)));
  BValue mapped_ar = fb.Map(ar, a);
  BValue for_loop = fb.CountedFor(x, /*trip_count=*/42, /*stride=*/1, body);
  fb.Tuple({mapped_ar, for_loop});

  XLS_ASSERT_OK(fb.Build().status());
  XLS_ASSERT_OK(p->SetTopByName("the_entry"));

  EXPECT_EQ(p->functions().size(), 3);
  EXPECT_THAT(Run(p.get()), IsOkAndHolds(false));
  EXPECT_EQ(p->functions().size(), 3);
}

TEST_F(DeadFunctionEliminationPassTest, BlockWithInstantiation) {
  auto p = CreatePackage();
  Type* u32 = p->GetBitsType(32);

  auto build_subblock = [&](std::string_view name) -> absl::StatusOr<Block*> {
    BlockBuilder bb(name, p.get());
    bb.OutputPort("out", bb.InputPort("in", u32));
    return bb.Build();
  };

  XLS_ASSERT_OK_AND_ASSIGN(Block * used_subblock,
                           build_subblock("used_subblock"));
  XLS_ASSERT_OK(build_subblock("unused_subblock").status());

  BlockBuilder bb("my_block", p.get());
  XLS_ASSERT_OK_AND_ASSIGN(
      Instantiation * instantiation,
      bb.block()->AddBlockInstantiation("inst", used_subblock));
  BValue in = bb.InputPort("in0", u32);
  bb.InstantiationInput(instantiation, "in", in);
  BValue inst_out = bb.InstantiationOutput(instantiation, "out");
  bb.OutputPort("out", inst_out);
  XLS_ASSERT_OK_AND_ASSIGN(Block * top, bb.Build());
  XLS_ASSERT_OK(p->SetTop(top));

  EXPECT_EQ(p->blocks().size(), 3);
  XLS_EXPECT_OK(p->GetBlock("unused_subblock").status());

  EXPECT_THAT(Run(p.get()), IsOkAndHolds(true));

  EXPECT_EQ(p->blocks().size(), 2);
  EXPECT_THAT(p->GetBlock("unused_subblock"),
              StatusIs(absl::StatusCode::kNotFound));
}

}  // namespace
}  // namespace xls
