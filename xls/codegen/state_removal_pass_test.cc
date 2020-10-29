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

#include "xls/codegen/state_removal_pass.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "xls/common/status/matchers.h"
#include "xls/ir/ir_matcher.h"
#include "xls/ir/ir_test_base.h"

namespace m = ::xls::op_matchers;

namespace xls {
namespace {

using status_testing::IsOkAndHolds;

class StateRemovalPassTest : public IrTestBase {
 protected:
  absl::StatusOr<bool> Run(Proc* proc) {
    PassResults results;
    XLS_ASSIGN_OR_RETURN(bool changed, StateRemovalPass().RunOnProc(
                                           proc, PassOptions(), &results));
    return changed;
  }
};

TEST_F(StateRemovalPassTest, SimpleProc) {
  auto p = CreatePackage();
  ProcBuilder pb(TestName(), /*init_value=*/Value(UBits(42, 32)), "tkn", "st",
                 p.get());
  BValue add = pb.Add(pb.Literal(UBits(1, 32)), pb.GetStateParam());
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, pb.Build(pb.GetTokenParam(), add));

  EXPECT_EQ(proc->StateType(), p->GetBitsType(32));

  EXPECT_THAT(Run(proc), IsOkAndHolds(true));

  EXPECT_EQ(proc->StateType(), p->GetTupleType({}));

  XLS_ASSERT_OK_AND_ASSIGN(Channel * state_channel,
                           p->GetChannel(StateRemovalPass::kStateChannelName));
  EXPECT_EQ(state_channel->data_elements().size(), 1);
  EXPECT_EQ(state_channel->data_element(0).name, "st");
  EXPECT_EQ(state_channel->data_element(0).type, p->GetBitsType(32));

  EXPECT_THAT(
      proc->NextToken(),
      m::AfterAll(m::Param(),
                  m::Send(m::TupleIndex(m::Receive(), /*index=*/0),
                          /*data=*/{m::Add(
                              m::Literal(),
                              m::TupleIndex(m::Receive(), /*index=*/1))})));
}

TEST_F(StateRemovalPassTest, ProcWithNilState) {
  auto p = CreatePackage();
  ProcBuilder pb(TestName(), /*init_value=*/Value::Tuple({}), "tkn", "st",
                 p.get());
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc,
                           pb.Build(pb.GetTokenParam(), pb.Tuple({})));

  EXPECT_THAT(Run(proc), IsOkAndHolds(false));
}

}  // namespace
}  // namespace xls
