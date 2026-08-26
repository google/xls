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

#include "xls/solvers/symex/symbolic_path.h"

#include <memory>
#include <utility>
#include <vector>

#include "gtest/gtest.h"
#include "absl/status/status.h"
#include "xls/common/status/matchers.h"
#include "xls/ir/bits.h"
#include "xls/ir/function.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/package.h"
#include "xls/ir/value.h"
#include "xls/solvers/z3_utils.h"
#include "z3/src/api/z3.h"

namespace xls::solvers::symex {
namespace {

using ::absl_testing::StatusIs;

class SymbolicPathTest : public ::testing::Test {
 protected:
  void SetUp() override {
    config_ = Z3_mk_config();
    ctx_ = Z3_mk_context(config_);
    package_ = std::make_unique<xls::Package>("test_pkg");
  }

  void TearDown() override {
    package_.reset();
    if (ctx_ != nullptr) {
      Z3_del_context(ctx_);
    }
    if (config_ != nullptr) {
      Z3_del_config(config_);
    }
  }

  Z3_config config_ = nullptr;
  Z3_context ctx_ = nullptr;
  std::unique_ptr<xls::Package> package_;
};

TEST_F(SymbolicPathTest, InitializesDefaultFields) {
  SymbolicPath path;
  EXPECT_EQ(path.path_condition, nullptr);
  EXPECT_EQ(path.return_value, nullptr);
  EXPECT_FALSE(path.is_feasible);
  EXPECT_TRUE(path.branch_decisions.empty());
}

TEST_F(SymbolicPathTest, SolveInfeasiblePathFailsPrecondition) {
  SymbolicPath path;
  path.is_feasible = false;
  EXPECT_THAT(path.Solve(ctx_, {}).status(),
              StatusIs(absl::StatusCode::kFailedPrecondition));
}

TEST_F(SymbolicPathTest, SolvesSatisfiablePathConditionAndExtractsValues) {
  xls::FunctionBuilder fb("solve_test_fn", package_.get());
  fb.Param("x", package_->GetBitsType(32));
  fb.Param("y", package_->GetBitsType(32));
  XLS_ASSERT_OK_AND_ASSIGN(xls::Function * fn, fb.Build());

  Z3_symbol x_sym = Z3_mk_string_symbol(ctx_, "x");
  Z3_symbol y_sym = Z3_mk_string_symbol(ctx_, "y");
  Z3_sort bv32 = Z3_mk_bv_sort(ctx_, 32);
  Z3_ast x_ast = Z3_mk_const(ctx_, x_sym, bv32);
  Z3_ast y_ast = Z3_mk_const(ctx_, y_sym, bv32);

  // Path condition: x == 10 AND y == 32
  Z3_ast val10 = xls::solvers::z3::BitsToZ3(ctx_, xls::UBits(10, 32));
  Z3_ast val32 = xls::solvers::z3::BitsToZ3(ctx_, xls::UBits(32, 32));
  Z3_ast eq_x = Z3_mk_eq(ctx_, x_ast, val10);
  Z3_ast eq_y = Z3_mk_eq(ctx_, y_ast, val32);
  Z3_ast conds[2] = {eq_x, eq_y};
  Z3_ast path_cond = Z3_mk_and(ctx_, 2, conds);

  SymbolicPath path;
  path.is_feasible = true;
  path.path_condition = path_cond;
  path.node_translations[fn->param(0)] = x_ast;
  path.node_translations[fn->param(1)] = y_ast;

  std::vector<const xls::Param*> params = {fn->param(0), fn->param(1)};
  XLS_ASSERT_OK_AND_ASSIGN(auto solution, path.Solve(ctx_, params));

  ASSERT_EQ(solution.size(), 2);
  EXPECT_EQ(solution[0].first, fn->param(0));
  EXPECT_EQ(solution[0].second, xls::Value(xls::UBits(10, 32)));
  EXPECT_EQ(solution[1].first, fn->param(1));
  EXPECT_EQ(solution[1].second, xls::Value(xls::UBits(32, 32)));
}

TEST_F(SymbolicPathTest, SolvesUnsatisfiablePathReturnsNotFound) {
  xls::FunctionBuilder fb("unsat_test_fn", package_.get());
  fb.Param("x", package_->GetBitsType(32));
  XLS_ASSERT_OK_AND_ASSIGN(xls::Function * fn, fb.Build());

  Z3_symbol x_sym = Z3_mk_string_symbol(ctx_, "x");
  Z3_sort bv32 = Z3_mk_bv_sort(ctx_, 32);
  Z3_ast x_ast = Z3_mk_const(ctx_, x_sym, bv32);

  // Path condition: x == 10 AND x == 20 (UNSAT)
  Z3_ast val10 = xls::solvers::z3::BitsToZ3(ctx_, xls::UBits(10, 32));
  Z3_ast val20 = xls::solvers::z3::BitsToZ3(ctx_, xls::UBits(20, 32));
  Z3_ast eq1 = Z3_mk_eq(ctx_, x_ast, val10);
  Z3_ast eq2 = Z3_mk_eq(ctx_, x_ast, val20);
  Z3_ast conds[2] = {eq1, eq2};
  Z3_ast path_cond = Z3_mk_and(ctx_, 2, conds);

  SymbolicPath path;
  path.is_feasible = true;
  path.path_condition = path_cond;
  path.node_translations[fn->param(0)] = x_ast;

  EXPECT_THAT(path.Solve(ctx_, {fn->param(0)}).status(),
              StatusIs(absl::StatusCode::kNotFound));
}

}  // namespace
}  // namespace xls::solvers::symex
