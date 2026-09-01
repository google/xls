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

#include <cstdint>
#include <optional>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/types/span.h"
#include "xls/common/status/matchers.h"
#include "xls/interpreter/function_interpreter.h"
#include "xls/ir/bits.h"
#include "xls/ir/bits_ops.h"
#include "xls/ir/events.h"
#include "xls/ir/function.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/package.h"
#include "xls/ir/value.h"
#include "xls/solvers/symex/concolic_input_spec.h"
#include "xls/solvers/symex/symbolic_path.h"
#include "z3/src/api/z3.h"  // IWYU pragma: keep
#include "z3/src/api/z3_api.h"

namespace xls::solvers::symex {
namespace {

using ::testing::Optional;

// Returns true if cond1 and cond2 cannot be simultaneously satisfied.
bool AreMutuallyExclusive(Z3_context ctx, Z3_ast cond1, Z3_ast cond2) {
  Z3_solver solver = Z3_mk_solver(ctx);
  Z3_solver_inc_ref(ctx, solver);
  Z3_solver_assert(ctx, solver, cond1);
  Z3_solver_assert(ctx, solver, cond2);
  Z3_lbool result = Z3_solver_check(ctx, solver);
  Z3_solver_dec_ref(ctx, solver);
  return result == Z3_L_FALSE;
}

// Returns true if the disjunction of all path conditions covers 100% of the
// input domain (i.e. check(!combined_path_condition) == UNSAT).
bool IsExhaustiveCoverage(Z3_context ctx,
                          absl::Span<const SymbolicPath> paths) {
  if (paths.empty()) {
    return false;
  }
  std::vector<Z3_ast> conds;
  conds.reserve(paths.size());
  for (const SymbolicPath& path : paths) {
    conds.push_back(path.path_condition);
  }
  Z3_ast combined = Z3_mk_or(ctx, conds.size(), conds.data());
  Z3_ast not_combined = Z3_mk_not(ctx, combined);

  Z3_solver solver = Z3_mk_solver(ctx);
  Z3_solver_inc_ref(ctx, solver);
  Z3_solver_assert(ctx, solver, not_combined);
  Z3_lbool result = Z3_solver_check(ctx, solver);
  Z3_solver_dec_ref(ctx, solver);
  return result == Z3_L_FALSE;
}

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

TEST_F(CfgSymExEngineTest, ExploresDirectIrSelectMux) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  auto cond = fb.Param("cond", p->GetBitsType(1));
  auto a = fb.Param("a", p->GetBitsType(8));
  auto b = fb.Param("b", p->GetBitsType(8));
  fb.Select(cond, {a, b});
  XLS_ASSERT_OK_AND_ASSIGN(Function * func, fb.Build());

  CfgSymExEngine engine(ctx_);
  XLS_ASSERT_OK_AND_ASSIGN(std::vector<SymbolicPath> paths,
                           engine.ExplorePaths(func));
  ASSERT_EQ(paths.size(), 2);

  // Path 0: case 0 (a).
  ASSERT_EQ(paths[0].branch_decisions.size(), 1);
  EXPECT_EQ(paths[0].branch_decisions[0].arm_index, 0);
  EXPECT_FALSE(paths[0].branch_decisions[0].is_default());
  EXPECT_THAT(paths[0].GetParamValue("cond"), Optional(Value(UBits(0, 1))));
  XLS_ASSERT_OK_AND_ASSIGN(InterpreterResult<Value> interp0,
                           InterpretFunction(func, paths[0].input_values()));
  EXPECT_THAT(paths[0].GetParamValue("a"), Optional(interp0.value));

  // Path 1: case 1 (b).
  ASSERT_EQ(paths[1].branch_decisions.size(), 1);
  EXPECT_EQ(paths[1].branch_decisions[0].arm_index, 1);
  EXPECT_FALSE(paths[1].branch_decisions[0].is_default());
  EXPECT_THAT(paths[1].GetParamValue("cond"), Optional(Value(UBits(1, 1))));
  XLS_ASSERT_OK_AND_ASSIGN(InterpreterResult<Value> interp1,
                           InterpretFunction(func, paths[1].input_values()));
  EXPECT_THAT(paths[1].GetParamValue("b"), Optional(interp1.value));

  // Verify paths are mutually exclusive and collectively exhaustive.
  EXPECT_TRUE(AreMutuallyExclusive(ctx_, paths[0].path_condition,
                                   paths[1].path_condition));
  EXPECT_TRUE(IsExhaustiveCoverage(ctx_, paths));
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
  XLS_ASSERT_OK_AND_ASSIGN(std::vector<SymbolicPath> paths,
                           engine.ExplorePaths(func));
  ASSERT_EQ(paths.size(), 3);

  std::vector<uint64_t> results;
  for (const SymbolicPath& path : paths) {
    ASSERT_EQ(path.generated_test.size(), 2);
    std::optional<Value> a_val = path.GetParamValue("a");
    std::optional<Value> b_val = path.GetParamValue("b");
    ASSERT_TRUE(a_val.has_value());
    ASSERT_TRUE(b_val.has_value());
    XLS_ASSERT_OK_AND_ASSIGN(InterpreterResult<Value> interp_res,
                             InterpretFunction(func, path.input_values()));
    XLS_ASSERT_OK_AND_ASSIGN(uint64_t result_int,
                             interp_res.value.bits().ToUint64());
    results.push_back(result_int);

    // Check witness input values match the branch taken.
    if (result_int == 10) {
      EXPECT_TRUE(bits_ops::ULessThan(a_val->bits(), b_val->bits()));
    } else if (result_int == 20) {
      EXPECT_EQ(a_val->bits(), b_val->bits());
    } else if (result_int == 30) {
      EXPECT_TRUE(bits_ops::UGreaterThan(a_val->bits(), b_val->bits()));
    } else {
      FAIL() << "Unexpected interpreter result: " << result_int;
    }
  }

  // Verify all 3 outcomes (10: lt, 20: eq, 30: ugt) were explored.
  EXPECT_THAT(results, testing::UnorderedElementsAre(10, 20, 30));

  // Verify pairwise mutual exclusivity and collective exhaustiveness.
  for (int i = 0; i < paths.size(); ++i) {
    for (int j = i + 1; j < paths.size(); ++j) {
      EXPECT_TRUE(AreMutuallyExclusive(ctx_, paths[i].path_condition,
                                       paths[j].path_condition));
    }
  }
  EXPECT_TRUE(IsExhaustiveCoverage(ctx_, paths));
}

TEST_F(CfgSymExEngineTest, PrunesPathsWithConcreteParam) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  auto cond = fb.Param("cond", p->GetBitsType(1));
  auto a = fb.Param("a", p->GetBitsType(8));
  auto b = fb.Param("b", p->GetBitsType(8));
  fb.Select(cond, {a, b});
  XLS_ASSERT_OK_AND_ASSIGN(Function * func, fb.Build());

  CfgSymExEngine engine(ctx_);

  // Fix cond = false (arm 0, returning 'a').
  ConcolicInputSpec spec0;
  spec0.BindParam("cond", Value::Bool(false));
  XLS_ASSERT_OK_AND_ASSIGN(
      std::vector<SymbolicPath> paths0,
      engine.ExplorePaths(func, {.concrete_inputs = spec0}));
  ASSERT_EQ(paths0.size(), 1);
  EXPECT_EQ(paths0[0].branch_decisions[0].arm_index, 0);
  EXPECT_THAT(paths0[0].GetParamValue("cond"), Optional(Value::Bool(false)));
  XLS_ASSERT_OK_AND_ASSIGN(InterpreterResult<Value> interp0,
                           InterpretFunction(func, paths0[0].input_values()));
  EXPECT_THAT(paths0[0].GetParamValue("a"), Optional(interp0.value));

  // Fix cond = true (arm 1, returning 'b').
  ConcolicInputSpec spec1;
  spec1.BindParam("cond", Value::Bool(true));
  XLS_ASSERT_OK_AND_ASSIGN(
      std::vector<SymbolicPath> paths1,
      engine.ExplorePaths(func, {.concrete_inputs = spec1}));
  ASSERT_EQ(paths1.size(), 1);
  EXPECT_EQ(paths1[0].branch_decisions[0].arm_index, 1);
  EXPECT_THAT(paths1[0].GetParamValue("cond"), Optional(Value::Bool(true)));
  XLS_ASSERT_OK_AND_ASSIGN(InterpreterResult<Value> interp1,
                           InterpretFunction(func, paths1[0].input_values()));
  EXPECT_THAT(paths1[0].GetParamValue("b"), Optional(interp1.value));
}

TEST_F(CfgSymExEngineTest, StopsAtMaxPathsLimit) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  auto s = fb.Param("s", p->GetBitsType(2));
  auto c0 = fb.Literal(UBits(10, 32));
  auto c1 = fb.Literal(UBits(20, 32));
  auto c2 = fb.Literal(UBits(30, 32));
  auto c3 = fb.Literal(UBits(40, 32));
  fb.Select(s, {c0, c1, c2, c3});
  XLS_ASSERT_OK_AND_ASSIGN(Function * func, fb.Build());

  CfgSymExEngine engine(ctx_);

  // Exploring with max_paths = 2 on a 4-path function returns exactly 2 paths.
  XLS_ASSERT_OK_AND_ASSIGN(std::vector<SymbolicPath> paths,
                           engine.ExplorePaths(func, {.max_paths = 2}));
  EXPECT_EQ(paths.size(), 2);

  // Exploring with max_paths = 0 returns 0 paths.
  XLS_ASSERT_OK_AND_ASSIGN(std::vector<SymbolicPath> paths_zero,
                           engine.ExplorePaths(func, {.max_paths = 0}));
  EXPECT_EQ(paths_zero.size(), 0);
}
}  // namespace
}  // namespace xls::solvers::symex
