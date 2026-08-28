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

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/types/span.h"
#include "xls/common/file/filesystem.h"
#include "xls/common/file/get_runfile_path.h"
#include "xls/common/status/matchers.h"
#include "xls/interpreter/function_interpreter.h"
#include "xls/ir/events.h"
#include "xls/ir/function.h"
#include "xls/ir/ir_parser.h"
#include "xls/ir/package.h"
#include "xls/ir/value.h"
#include "xls/solvers/symex/cfg_symex_engine.h"
#include "xls/solvers/symex/symbolic_path.h"
#include "xls/solvers/symex/test_util.h"
#include "z3/src/api/z3.h"  // IWYU pragma: keep
#include "z3/src/api/z3_api.h"

namespace xls::solvers::symex {
namespace {

using SymExE2eTest = SymExTestBase;

struct AluExpected {
  uint64_t status;
  uint64_t result;
};

// Reference specification oracle for execute_alu.
AluExpected SymexAluOracle(uint64_t op, uint64_t a, uint64_t b) {
  if (op == 0) {  // ADD
    if (a + b > 255) {
      return {.status = 1, .result = 0};  // OVERFLOW
    }
    return {.status = 0, .result = (a + b) & 0xFF};  // OK
  }
  if (op == 1) {                            // AND
    return {.status = 0, .result = a & b};  // OK
  }
  return {.status = 2, .result = 0};  // INVALID_OP
}

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

    // Verify interpreter output matches DSLX ALU oracle specification.
    AluExpected expected = SymexAluOracle(op, a, b);
    EXPECT_EQ(status, expected.status);
    EXPECT_EQ(result_val, expected.result);
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

}  // namespace
}  // namespace xls::solvers::symex
