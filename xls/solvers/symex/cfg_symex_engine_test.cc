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
#include "xls/ir/package.h"
#include "xls/ir/value.h"
#include "xls/solvers/symex/symbolic_path.h"
#include "xls/solvers/symex/test_util.h"
#include "z3/src/api/z3.h"  // IWYU pragma: keep
#include "z3/src/api/z3_api.h"

namespace xls::solvers::symex {
namespace {

using ::testing::ElementsAre;
using ::testing::Optional;
using ::testing::UnorderedElementsAre;

using CfgSymExEngineTest = SymExTestBase;

TEST_F(CfgSymExEngineTest, ExploresNonBranchingFunction) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  auto a = fb.Param("a", p->GetBitsType(8));
  auto b = fb.Param("b", p->GetBitsType(8));
  fb.Add(a, b);
  XLS_ASSERT_OK_AND_ASSIGN(Function * func, fb.Build());

  CfgSymExEngine engine(ctx_);
  XLS_ASSERT_OK_AND_ASSIGN(std::vector<SymbolicPath> paths,
                           engine.ExplorePaths(func));
  ASSERT_EQ(paths.size(), 1);
  EXPECT_TRUE(paths[0].branch_decisions.empty());
  EXPECT_TRUE(IsExhaustiveCoverage(ctx_, paths));
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
  XLS_ASSERT_OK_AND_ASSIGN(std::vector<SymbolicPath> paths,
                           engine.ExplorePaths(func));
  ASSERT_EQ(paths.size(), 2);

  // Path 0: case 0 (a).
  EXPECT_THAT(
      paths[0].branch_decisions,
      ElementsAre(BranchDecisionIs(/*arm_index=*/0, /*is_default=*/false)));
  EXPECT_THAT(paths[0].GetParamValue("cond"), Optional(Value(UBits(0, 1))));
  XLS_ASSERT_OK_AND_ASSIGN(InterpreterResult<Value> interp0,
                           InterpretFunction(func, paths[0].input_values()));
  EXPECT_THAT(paths[0].GetParamValue("a"), Optional(interp0.value));

  // Path 1: case 1 (b).
  EXPECT_THAT(
      paths[1].branch_decisions,
      ElementsAre(BranchDecisionIs(/*arm_index=*/1, /*is_default=*/false)));
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
  EXPECT_THAT(results, UnorderedElementsAre(10, 20, 30));

  // Verify pairwise mutual exclusivity and collective exhaustiveness.
  for (int i = 0; i < paths.size(); ++i) {
    for (int j = i + 1; j < paths.size(); ++j) {
      EXPECT_TRUE(AreMutuallyExclusive(ctx_, paths[i].path_condition,
                                       paths[j].path_condition));
    }
  }
  EXPECT_TRUE(IsExhaustiveCoverage(ctx_, paths));
}

}  // namespace
}  // namespace xls::solvers::symex
