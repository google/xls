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

#include <filesystem>
#include <memory>
#include <string>

#include "gtest/gtest.h"
#include "absl/status/status.h"
#include "xls/common/file/filesystem.h"
#include "xls/common/file/get_runfile_path.h"
#include "xls/common/status/matchers.h"
#include "xls/ir/function.h"
#include "xls/ir/ir_parser.h"
#include "xls/ir/package.h"
#include "xls/solvers/symex/cfg_symex_engine.h"
#include "xls/solvers/symex/symbolic_path.h"
#include "xls/solvers/symex/symex_engine.h"
#include "z3/src/api/z3.h"

namespace xls::solvers::symex {
namespace {

using ::absl_testing::StatusIs;

class SymExE2eTest : public ::testing::Test {
 protected:
  void SetUp() override {
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
  }

  Z3_config config_ = nullptr;
  Z3_context ctx_ = nullptr;
};

TEST_F(SymExE2eTest, ExploresAndSolvesAllFourAluExecutionPaths) {
  XLS_ASSERT_OK_AND_ASSIGN(
      std::filesystem::path ir_path,
      GetXlsRunfilePath("xls/solvers/symex/testdata/execute_alu.ir"));
  XLS_ASSERT_OK_AND_ASSIGN(std::string ir_text, GetFileContents(ir_path));
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<xls::Package> package,
                           xls::Parser::ParsePackage(ir_text));
  XLS_ASSERT_OK_AND_ASSIGN(xls::Function * func,
                           package->GetFunction("__execute_alu__execute_alu"));

  ASSERT_NE(func, nullptr);
  EXPECT_EQ(func->params().size(), 3);

  CfgSymExEngine engine(ctx_);
  XLS_ASSERT_OK_AND_ASSIGN(std::vector<SymbolicPath> paths,
                           engine.ExplorePaths(func));

  EXPECT_EQ(paths.size(), 4);
  EXPECT_EQ(engine.feasible_paths(), 4);

  for (const SymbolicPath& path : paths) {
    EXPECT_TRUE(path.is_feasible);
    EXPECT_NE(path.path_condition, nullptr);
    EXPECT_NE(path.return_value, nullptr);

    XLS_ASSERT_OK_AND_ASSIGN(auto solution, path.Solve(ctx_, func->params()));
    ASSERT_EQ(solution.size(), 3);
  }
}

}  // namespace
}  // namespace xls::solvers::symex
