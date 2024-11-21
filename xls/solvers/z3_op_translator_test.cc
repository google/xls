// Copyright 2023 The XLS Authors
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

#include "xls/solvers/z3_op_translator.h"

#include "gtest/gtest.h"
#include "xls/common/logging/log_lines.h"
#include "xls/solvers/z3_utils.h"
#include "z3/src/api/z3.h"  // IWYU pragma: keep
#include "z3/src/api/z3_api.h"

namespace xls::solvers::z3 {
namespace {

class Z3OpTranslatorTest : public testing::Test {
 public:
  Z3OpTranslatorTest() {
    config_ = Z3_mk_config();
    Z3_set_param_value(config_, "proof", "true");
    ctx_ = Z3_mk_context(config_);
    solver_ = CreateSolver(ctx_, /*num_threads=*/1);
  }

  ~Z3OpTranslatorTest() override {
    Z3_solver_dec_ref(ctx_, solver_);
    Z3_del_context(ctx_);
    Z3_del_config(config_);
  }

 protected:
  Z3_config config_;
  Z3_context ctx_;
  Z3_solver solver_;
};

// Simple scenario where we create one-bit parameters and check there is a
// scenario in which they cannot be proven equal (should be trivial, but
// exercises all the basic machinery).
TEST_F(Z3OpTranslatorTest, OneBitParamsNe) {
  Z3OpTranslator t(ctx_);
  Z3_ast p = t.MakeBvParam(1, "p");
  Z3_ast q = t.MakeBvParam(1, "q");
  Z3_ast params_equal = t.EqBool(p, q);
  EXPECT_EQ(t.GetSortName(params_equal), "Bool");

  // Assert a constraint into the solver.
  Z3_solver_assert(ctx_, solver_, params_equal);

  Z3_lbool satisfiable = Z3_solver_check(ctx_, solver_);
  EXPECT_EQ(satisfiable, Z3_L_TRUE);

  // Note: we shouldn't assert on the model results as they may not be uniquely
  // satisfying solution.
  XLS_LOG_LINES(
      INFO, SolverResultToString(ctx_, solver_, satisfiable, /*hexify=*/false));

  // It is satisfiable -- i.e. we found a set of values for which the assertion
  // holds.
  EXPECT_EQ(Z3_solver_check(ctx_, solver_), Z3_L_TRUE);
}

// Demonstrates what the bit ordering is for the "extract" operation.
TEST_F(Z3OpTranslatorTest, TwoBitParamsExtract) {
  Z3OpTranslator t(ctx_);
  Z3_ast p = t.MakeBvParam(2, "p");
  Z3_ast q = t.MakeBvParam(2, "q");
  Z3_ast p0 = t.Extract(p, 0);
  Z3_ast q0 = t.Extract(q, 0);
  Z3_ast p1 = t.Extract(p, 1);
  Z3_ast q1 = t.Extract(q, 1);

  // We assert that the 0 bits are not equal and the 1 bits are equal.
  Z3_ast bit0_ne = t.NeBool(p0, q0);
  EXPECT_EQ(t.GetSortName(bit0_ne), "Bool");
  Z3_solver_assert(ctx_, solver_, bit0_ne);

  Z3_ast bit1_eq = t.EqBool(p1, q1);
  EXPECT_EQ(t.GetSortName(bit1_eq), "Bool");
  Z3_solver_assert(ctx_, solver_, bit1_eq);

  Z3_lbool satisfiable = Z3_solver_check(ctx_, solver_);
  EXPECT_EQ(satisfiable, Z3_L_TRUE);

  // Note: we shouldn't assert on the model results as they may not be uniquely
  // satisfying solution.
  XLS_LOG_LINES(
      INFO, SolverResultToString(ctx_, solver_, satisfiable, /*hexify=*/false));

  // It is satisfiable -- i.e. we found a set of values for which the assertion
  // holds.
  EXPECT_EQ(Z3_solver_check(ctx_, solver_), Z3_L_TRUE);
}

// Demonstrates sign extension from one bit to two bits.
TEST_F(Z3OpTranslatorTest, SignExtOneBitAllOnesOrAllZeros) {
  Z3OpTranslator t(ctx_);
  Z3_ast p = t.MakeBvParam(1, "p");
  Z3_ast s = t.SignExt(p, /*new_bit_count=*/2);
  Z3_ast is_all_z = t.EqZero(s);
  Z3_ast all_one = t.Fill(true, /*bit_count=*/2);
  Z3_ast is_all_o = t.Eq(s, all_one);
  Z3_ast all_z_or_o = t.Or(is_all_o, is_all_z);
  Z3_ast all_z_or_o_bool = t.EqZeroBool(all_z_or_o);

  // We assert that all_z_or_o is false. When we get unsat, we have proof by
  // contradiction.
  EXPECT_EQ(t.GetSortName(all_z_or_o_bool), "Bool");
  Z3_solver_assert(ctx_, solver_, all_z_or_o_bool);

  // We cannot find an example where the signext of a single bit value is /not/
  // all-zeros or all-ones.
  Z3_lbool satisfiable = Z3_solver_check(ctx_, solver_);
  EXPECT_EQ(satisfiable, Z3_L_FALSE);
}

}  // namespace
}  // namespace xls::solvers::z3
