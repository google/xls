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

#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/statusor.h"
#include "xls/common/file/filesystem.h"
#include "xls/common/file/get_runfile_path.h"
#include "xls/common/status/matchers.h"
#include "xls/common/status/status_macros.h"
#include "xls/interpreter/function_interpreter.h"
#include "xls/ir/events.h"
#include "xls/ir/function.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/nodes.h"
#include "xls/ir/package.h"
#include "xls/ir/value.h"
#include "xls/solvers/symex/symbolic_path.h"
#include "xls/solvers/symex/symex_engine.h"
#include "z3/src/api/z3_api.h"

namespace xls::solvers::symex {
namespace {

using ::testing::SizeIs;

class SymexE2eTest : public IrTestBase {
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

  bool AreMutuallyExclusive(Z3_ast cond1, Z3_ast cond2) {
    Z3_solver solver = Z3_mk_solver(ctx_);
    Z3_solver_inc_ref(ctx_, solver);
    Z3_solver_assert(ctx_, solver, cond1);
    Z3_solver_assert(ctx_, solver, cond2);
    Z3_lbool result = Z3_solver_check(ctx_, solver);
    Z3_solver_dec_ref(ctx_, solver);
    return result == Z3_L_FALSE;
  }

  absl::StatusOr<std::unique_ptr<Package>> LoadAluPackage() {
    XLS_ASSIGN_OR_RETURN(
        std::filesystem::path ir_path,
        GetXlsRunfilePath("xls/solvers/symex/testdata/execute_alu.ir"));
    XLS_ASSIGN_OR_RETURN(std::string ir_text, GetFileContents(ir_path));
    return ParsePackage(ir_text);
  }

  Z3_config config_ = nullptr;
  Z3_context ctx_ = nullptr;
};

TEST_F(SymexE2eTest, ExecuteAluExploresPaths) {
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> p, LoadAluPackage());
  XLS_ASSERT_OK_AND_ASSIGN(Function * fn,
                           p->GetFunction("__execute_alu__execute_alu"));

  XLS_ASSERT_OK_AND_ASSIGN(SymExEngine engine, SymExEngine::Create(ctx_));
  XLS_ASSERT_OK_AND_ASSIGN(std::vector<SymbolicPath> paths,
                           engine.ExplorePaths(fn));

  // Without branch visibility optimization, exhaustive exploration across all
  // topological selects produces 3 (op) x 2 (overflow) = 6 paths.
  // (Follow-up CL will prune inactive branches to 4 paths via DAG guard
  // propagation).
  ASSERT_THAT(paths, SizeIs(6));

  // Verify pairwise mutual exclusivity.
  for (size_t i = 0; i < paths.size(); ++i) {
    for (size_t j = i + 1; j < paths.size(); ++j) {
      EXPECT_TRUE(AreMutuallyExclusive(paths[i].path_condition,
                                       paths[j].path_condition));
    }
  }

  // Verify interpreter execution of generated test inputs.
  for (const SymbolicPath& path : paths) {
    std::vector<Value> args;
    for (Param* param : fn->params()) {
      std::optional<Value> v = path.GetParamValue(param->name());
      ASSERT_TRUE(v.has_value());
      args.push_back(*v);
    }
    XLS_ASSERT_OK_AND_ASSIGN(
        Value result, DropInterpreterEvents(InterpretFunction(fn, args)));
    EXPECT_TRUE(result.IsTuple());
  }
}

}  // namespace
}  // namespace xls::solvers::symex
