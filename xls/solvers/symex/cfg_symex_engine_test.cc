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

#include "xls/solvers/symex/cfg_symex_engine.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status.h"
#include "xls/common/status/matchers.h"
#include "xls/ir/bits.h"
#include "xls/ir/function.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/package.h"
#include "z3/src/api/z3.h"  // IWYU pragma: keep
#include "z3/src/api/z3_api.h"

namespace xls::solvers::symex {
namespace {

using ::absl_testing::StatusIs;

class CfgSymExEngineTest : public IrTestBase {
 protected:
  void SetUp() override {
    IrTestBase::SetUp();
    config_ = Z3_mk_config();
    ctx_ = Z3_mk_context(config_);
  }

  void TearDown() override {
    if (ctx_ != nullptr) {
      Z3_del_context(ctx_);
    }
    if (config_ != nullptr) {
      Z3_del_config(config_);
    }
    IrTestBase::TearDown();
  }

  Z3_config config_ = nullptr;
  Z3_context ctx_ = nullptr;
};

TEST_F(CfgSymExEngineTest, InitializesWithDefaultOptions) {
  CfgSymExEngine engine(ctx_);
  EXPECT_EQ(engine.total_explored_paths(), 0);
  EXPECT_EQ(engine.feasible_paths(), 0);
}

TEST_F(CfgSymExEngineTest, RespectsCustomOptions) {
  SymExOptions options;
  options.max_paths = 42;
  options.max_depth = 128;

  CfgSymExEngine engine(ctx_, options);
  EXPECT_EQ(engine.total_explored_paths(), 0);
  EXPECT_EQ(engine.feasible_paths(), 0);
}

TEST_F(CfgSymExEngineTest, ExploresDirectIrSelectMux) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  auto cond = fb.Param("cond", p->GetBitsType(1));
  auto a = fb.Param("a", p->GetBitsType(8));
  auto b = fb.Param("b", p->GetBitsType(8));
  fb.Select(cond, {a, b});
  XLS_ASSERT_OK_AND_ASSIGN(Function * func, fb.Build());

  CfgSymExEngine engine(ctx_);
  EXPECT_THAT(engine.ExplorePaths(func).status(),
              StatusIs(absl::StatusCode::kUnimplemented));
}

TEST_F(CfgSymExEngineTest, ExploresDirectIrThreeWayCompareMuxTree) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  Type* u32 = p->GetBitsType(32);
  BValue a = fb.Param("a", u32);
  BValue b = fb.Param("b", u32);
  BValue ugt = fb.UGt(a, b);
  BValue eq = fb.Eq(a, b);
  BValue lt_res = fb.Literal(UBits(10, 32));
  BValue eq_res = fb.Literal(UBits(20, 32));
  BValue ugt_res = fb.Literal(UBits(30, 32));
  BValue sub_select = fb.Select(eq, {lt_res, eq_res});
  fb.Select(ugt, {sub_select, ugt_res});
  XLS_ASSERT_OK_AND_ASSIGN(Function * func, fb.Build());

  CfgSymExEngine engine(ctx_);
  EXPECT_THAT(engine.ExplorePaths(func).status(),
              StatusIs(absl::StatusCode::kUnimplemented));
}

}  // namespace
}  // namespace xls::solvers::symex
