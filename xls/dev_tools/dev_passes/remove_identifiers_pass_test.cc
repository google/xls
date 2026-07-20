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

#include "xls/dev_tools/dev_passes/remove_identifiers_pass.h"

#include <array>
#include <string_view>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "xls/common/status/matchers.h"
#include "xls/ir/bits.h"
#include "xls/ir/function.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/package.h"
#include "xls/ir/value.h"
#include "xls/passes/optimization_pass.h"
#include "xls/passes/pass_base.h"

namespace xls {
namespace {

using absl_testing::IsOkAndHolds;

class RemoveIdentifiersPassTest : public IrTestBase {
 protected:
  absl::StatusOr<bool> Run(Package* p) {
    RemoveIdentifiersPass pass;
    PassResults results;
    OptimizationContext context;
    return pass.Run(p, {}, &results, context);
  }
};

TEST_F(RemoveIdentifiersPassTest, BasicFunction) {
  static constexpr std::array<std::string_view, 4> kSecrets{
      "the_answer_func", "foo", "bar", "ultimate_question"};
  auto p = CreatePackage();
  FunctionBuilder fb("the_answer_func", p.get());
  fb.Add(fb.Param("foo", p->GetBitsType(32)),
         fb.Param("bar", p->GetBitsType(32)), SourceInfo(),
         "ultimate_question");
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  XLS_ASSERT_OK(p->SetTop(f));

  EXPECT_THAT(Run(p.get()), IsOkAndHolds(true));

  EXPECT_THAT(p->functions(), testing::SizeIs(1));

  for (auto secret : kSecrets) {
    EXPECT_THAT(p->DumpIr(), testing::Not(testing::ContainsRegex(secret)));
  }
}

TEST_F(RemoveIdentifiersPassTest, BasicProc) {
  static constexpr std::array<std::string_view, 3> kSecrets{
      "the_answer_proc", "secret_tunnel", "astounding"};
  auto p = CreatePackage();
  ProcBuilder pb(NewStyleProc{}, "the_answer_proc", p.get());
  pb.AddChannel("secret_tunnel", p->GetBitsType(32));
  auto start_param = pb.StateElement("astounding", Value(UBits(42, 32)));
  pb.Next(start_param, start_param);
  XLS_ASSERT_OK_AND_ASSIGN(auto* orig_proc, pb.Build());
  XLS_ASSERT_OK(p->SetTop(orig_proc));

  EXPECT_THAT(Run(p.get()), IsOkAndHolds(true));

  EXPECT_THAT(p->procs(), testing::SizeIs(1));

  for (auto secret : kSecrets) {
    EXPECT_THAT(p->DumpIr(), testing::Not(testing::ContainsRegex(secret)));
  }
}

}  // namespace
}  // namespace xls
