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

#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "gtest/gtest.h"
#include "absl/types/span.h"
#include "xls/common/file/filesystem.h"
#include "xls/common/file/get_runfile_path.h"
#include "xls/common/status/matchers.h"
#include "xls/interpreter/function_interpreter.h"
#include "xls/ir/bits.h"
#include "xls/ir/events.h"
#include "xls/ir/function.h"
#include "xls/ir/ir_parser.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/package.h"
#include "xls/ir/value.h"
#include "xls/solvers/symex/cfg_symex_engine.h"
#include "xls/solvers/symex/concolic_input_spec.h"
#include "xls/solvers/symex/symbolic_path.h"
#include "z3/src/api/z3.h"  // IWYU pragma: keep
#include "z3/src/api/z3_api.h"

namespace xls::solvers::symex {
namespace {

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

class SymExE2eTest : public IrTestBase {
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

TEST_F(SymExE2eTest, ExploresAndSolvesLinearizedAluExecutionPaths) {
  XLS_ASSERT_OK_AND_ASSIGN(
      std::filesystem::path ir_path,
      GetXlsRunfilePath("xls/solvers/symex/testdata/execute_alu.ir"));
  XLS_ASSERT_OK_AND_ASSIGN(std::string ir_text, GetFileContents(ir_path));
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> package,
                           Parser::ParsePackage(ir_text));
  XLS_ASSERT_OK_AND_ASSIGN(Function * func,
                           package->GetFunction("__execute_alu__execute_alu"));

  ASSERT_NE(func, nullptr);
  EXPECT_EQ(func->params().size(), 3);

  CfgSymExEngine engine(ctx_);
  XLS_ASSERT_OK_AND_ASSIGN(std::vector<SymbolicPath> paths,
                           engine.ExplorePaths(func));

  // 6 feasible paths explored due to eager evaluation of nested sub-muxes in
  // hardware IR.
  EXPECT_EQ(paths.size(), 6);

  for (const SymbolicPath& path : paths) {
    EXPECT_FALSE(path.branch_decisions.empty());
    ASSERT_EQ(path.generated_test.size(), 3);

    std::optional<Value> op_val = path.GetParamValue("op");
    std::optional<Value> a_val = path.GetParamValue("a");
    std::optional<Value> b_val = path.GetParamValue("b");
    ASSERT_TRUE(op_val.has_value());
    ASSERT_TRUE(a_val.has_value());
    ASSERT_TRUE(b_val.has_value());

    XLS_ASSERT_OK_AND_ASSIGN(uint64_t op, op_val->bits().ToUint64());
    XLS_ASSERT_OK_AND_ASSIGN(uint64_t a, a_val->bits().ToUint64());
    XLS_ASSERT_OK_AND_ASSIGN(uint64_t b, b_val->bits().ToUint64());

    XLS_ASSERT_OK_AND_ASSIGN(InterpreterResult<Value> interp_res,
                             InterpretFunction(func, path.input_values()));
    ASSERT_TRUE(interp_res.value.IsTuple());
    ASSERT_EQ(interp_res.value.elements().size(), 2);

    XLS_ASSERT_OK_AND_ASSIGN(uint64_t status,
                             interp_res.value.elements()[0].bits().ToUint64());
    XLS_ASSERT_OK_AND_ASSIGN(uint64_t result_val,
                             interp_res.value.elements()[1].bits().ToUint64());

    // Verify interpreter output matches DSLX ALU specification.
    if (op == 0) {  // ADD
      if (a + b > 255) {
        EXPECT_EQ(status, 1);  // OVERFLOW
        EXPECT_EQ(result_val, 0);
      } else {
        EXPECT_EQ(status, 0);  // OK
        EXPECT_EQ(result_val, (a + b) & 0xFF);
      }
    } else if (op == 1) {    // AND
      EXPECT_EQ(status, 0);  // OK
      EXPECT_EQ(result_val, a & b);
    } else {                 // INVALID_OP
      EXPECT_EQ(status, 2);  // INVALID_OP
      EXPECT_EQ(result_val, 0);
    }
  }

  // Verify pairwise mutual exclusivity and collective exhaustiveness across
  // all paths.
  for (int i = 0; i < paths.size(); ++i) {
    for (int j = i + 1; j < paths.size(); ++j) {
      EXPECT_TRUE(AreMutuallyExclusive(ctx_, paths[i].path_condition,
                                       paths[j].path_condition));
    }
  }
  EXPECT_TRUE(IsExhaustiveCoverage(ctx_, paths));
}

TEST_F(SymExE2eTest, ExploresConcolicAluWithConcreteOp) {
  XLS_ASSERT_OK_AND_ASSIGN(
      std::filesystem::path ir_path,
      GetXlsRunfilePath("xls/solvers/symex/testdata/execute_alu.ir"));
  XLS_ASSERT_OK_AND_ASSIGN(std::string ir_text, GetFileContents(ir_path));
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> package,
                           Parser::ParsePackage(ir_text));
  XLS_ASSERT_OK_AND_ASSIGN(Function * func,
                           package->GetFunction("__execute_alu__execute_alu"));

  CfgSymExEngine engine(ctx_);
  ConcolicInputSpec inputs;
  inputs.BindParam("op", Value(UBits(0, 2)));
  XLS_ASSERT_OK_AND_ASSIGN(
      std::vector<SymbolicPath> paths,
      engine.ExplorePaths(func, {.concrete_inputs = inputs}));

  // With concrete op=0 (ADD), all AND and INVALID_OP paths are pruned.
  // Exactly 2 feasible paths remain: ADD OK and ADD OVERFLOW.
  EXPECT_EQ(paths.size(), 2);

  for (const SymbolicPath& path : paths) {
    std::optional<Value> op_val = path.GetParamValue("op");
    std::optional<Value> a_val = path.GetParamValue("a");
    std::optional<Value> b_val = path.GetParamValue("b");
    ASSERT_TRUE(op_val.has_value());
    ASSERT_TRUE(a_val.has_value());
    ASSERT_TRUE(b_val.has_value());

    EXPECT_EQ(*op_val, Value(UBits(0, 2)));
    XLS_ASSERT_OK_AND_ASSIGN(uint64_t a, a_val->bits().ToUint64());
    XLS_ASSERT_OK_AND_ASSIGN(uint64_t b, b_val->bits().ToUint64());

    XLS_ASSERT_OK_AND_ASSIGN(InterpreterResult<Value> interp_res,
                             InterpretFunction(func, path.input_values()));
    ASSERT_TRUE(interp_res.value.IsTuple());
    ASSERT_EQ(interp_res.value.elements().size(), 2);

    XLS_ASSERT_OK_AND_ASSIGN(uint64_t status,
                             interp_res.value.elements()[0].bits().ToUint64());
    XLS_ASSERT_OK_AND_ASSIGN(uint64_t result_val,
                             interp_res.value.elements()[1].bits().ToUint64());

    if (a + b > 255) {
      EXPECT_EQ(status, 1);  // OVERFLOW
      EXPECT_EQ(result_val, 0);
    } else {
      EXPECT_EQ(status, 0);  // OK
      EXPECT_EQ(result_val, (a + b) & 0xFF);
    }
  }
}

}  // namespace
}  // namespace xls::solvers::symex
