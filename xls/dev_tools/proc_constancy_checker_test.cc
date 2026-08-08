// Copyright 2026 The XLS Authors
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

#include "xls/dev_tools/proc_constancy_checker.h"

#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/statusor.h"
#include "xls/common/status/matchers.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/bits.h"
#include "xls/ir/channel.h"
#include "xls/ir/channel_ops.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/node.h"
#include "xls/ir/package.h"
#include "xls/ir/proc.h"
#include "xls/ir/source_location.h"
#include "xls/ir/value.h"

namespace xls {
namespace {

using ::testing::UnorderedElementsAre;

class ProcConstancyCheckerTest : public IrTestBase {
 protected:
  absl::StatusOr<Proc*> BuildTestProc(Package* p) {
    ProcBuilder pb(TestName(), p);
    XLS_ASSIGN_OR_RETURN(
        auto out_ch, p->CreateStreamingChannel("out_ch", ChannelOps::kSendOnly,
                                               p->GetBitsType(32)));
    auto tok = pb.ReadStateElement("tok", Value::Token());
    auto state = pb.ReadStateElement("st", Value(UBits(0, 32)));
    auto lit1 = pb.Literal(UBits(1, 32));
    auto add1 = pb.Add(state, lit1, SourceInfo(), "add1");
    auto cond = pb.Literal(UBits(1, 1));
    pb.Assert(tok, cond, "test assert");
    auto snd_tok = pb.Send(out_ch, tok, add1);
    pb.Next(state, add1);
    pb.Next(tok, snd_tok);
    XLS_ASSIGN_OR_RETURN(Proc * proc, pb.Build());
    XLS_RETURN_IF_ERROR(p->SetTop(proc));
    return proc;
  }
};

TEST_F(ProcConstancyCheckerTest, GetNonConstantNodes) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, BuildTestProc(p.get()));
  XLS_ASSERT_OK(StripNonSynthNodes(p.get(), proc));
  XLS_ASSERT_OK_AND_ASSIGN(std::vector<Node*> targets,
                           GetNodesFilteringNonSynthAndTrivialConstants(proc));
  std::vector<std::string> names;
  for (Node* n : targets) {
    names.push_back(n->GetName());
  }
  EXPECT_THAT(names, UnorderedElementsAre("add1"));
}

TEST_F(ProcConstancyCheckerTest, UnrollProcForConstancyTest) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, BuildTestProc(p.get()));
  XLS_ASSERT_OK_AND_ASSIGN((auto [func, map]), UnrollProcForConstancy(proc, 3));
  EXPECT_NE(func, nullptr);
  EXPECT_FALSE(map.empty());
}

}  // namespace
}  // namespace xls
