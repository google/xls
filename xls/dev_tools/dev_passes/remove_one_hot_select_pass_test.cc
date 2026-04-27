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

#include "xls/dev_tools/dev_passes/remove_one_hot_select_pass.h"

#include <memory>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "xls/common/status/matchers.h"
#include "xls/ir/function.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/ir_matcher.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/package.h"
#include "xls/passes/optimization_pass.h"
#include "xls/passes/pass_base.h"
#include "xls/solvers/z3_ir_equivalence_testutils.h"

namespace m = xls::op_matchers;

namespace xls {
namespace {

using absl_testing::IsOkAndHolds;
using solvers::z3::ScopedVerifyEquivalence;

class RemoveOneHotSelectPassTest : public IrTestBase {
 protected:
  absl::StatusOr<bool> Run(Package* p) {
    OptimizationCompoundPass pass("test_pass", "test_pass");
    pass.Add<RemoveOneHotSelectPass>();
    PassResults results;
    OptimizationContext context;
    return pass.Run(p, {}, &results, context);
  }
};

TEST_F(RemoveOneHotSelectPassTest, TwoBitSelector) {
  auto p = CreatePackage();
  FunctionBuilder fb("f", p.get());
  BValue sel = fb.Param("sel", p->GetBitsType(2));
  BValue x = fb.Param("x", p->GetBitsType(32));
  BValue y = fb.Param("y", p->GetBitsType(32));
  fb.OneHotSelect(sel, {x, y});
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());

  ScopedVerifyEquivalence sve(f);
  EXPECT_THAT(Run(p.get()), IsOkAndHolds(true));

  EXPECT_THAT(
      f->return_value(),
      m::Select(m::Param("sel"), {m::Literal(0), m::Param("x"), m::Param("y"),
                                  m::Or(m::Param("x"), m::Param("y"))}));
}

TEST_F(RemoveOneHotSelectPassTest, OneBitSelector) {
  auto p = CreatePackage();
  FunctionBuilder fb("f", p.get());
  BValue sel = fb.Param("sel", p->GetBitsType(1));
  BValue x = fb.Param("x", p->GetBitsType(32));
  fb.OneHotSelect(sel, {x});
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());

  ScopedVerifyEquivalence sve(f);
  EXPECT_THAT(Run(p.get()), IsOkAndHolds(true));

  EXPECT_THAT(f->return_value(),
              m::Select(m::Param("sel"), {m::Literal(0), m::Param("x")}));
}

TEST_F(RemoveOneHotSelectPassTest, ExceedsMaxBitsUnchanged) {
  auto p = CreatePackage();
  FunctionBuilder fb("f", p.get());
  BValue sel =
      fb.Param("sel", p->GetBitsType(RemoveOneHotSelectPass::kMaxBits + 1));
  std::vector<BValue> cases;
  for (int i = 0; i < RemoveOneHotSelectPass::kMaxBits + 1; ++i) {
    cases.push_back(fb.Param(absl::StrCat("c_", i), p->GetBitsType(32)));
  }
  fb.OneHotSelect(sel, cases);
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());

  ScopedVerifyEquivalence sve(f);
  EXPECT_THAT(Run(p.get()), IsOkAndHolds(false));

  EXPECT_THAT(f->return_value(), m::OneHotSelect());
}

}  // namespace
}  // namespace xls
