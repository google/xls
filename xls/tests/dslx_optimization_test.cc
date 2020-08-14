// Copyright 2020 Google LLC
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

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "xls/common/file/get_runfile_path.h"
#include "xls/common/file/temp_file.h"
#include "xls/common/logging/logging.h"
#include "xls/common/status/matchers.h"
#include "xls/common/subprocess.h"
#include "xls/ir/function.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/op.h"
#include "xls/ir/package.h"
#include "xls/passes/standard_pipeline.h"

namespace xls {
namespace {

// Tests that DSLX constructs are properly optimized. This test is in C++
// because the complicated bit (matching of the IR) use a C++ API.
class DslxOptimizationTest : public IrTestBase {
 protected:
  xabsl::StatusOr<std::unique_ptr<VerifiedPackage>> DslxToIr(
      absl::string_view dslx) {
    XLS_ASSIGN_OR_RETURN(TempFile dslx_temp, TempFile::CreateWithContent(dslx));
    std::string ir_converter_main_path =
        GetXlsRunfilePath("xls/dslx/ir_converter_main").string();
    std::pair<std::string, std::string> stdout_stderr;
    XLS_ASSIGN_OR_RETURN(
        stdout_stderr,
        InvokeSubprocess({ir_converter_main_path, dslx_temp.path().string()}));
    return ParsePackage(stdout_stderr.first);
  }

  // Returns true if the given IR function has a node with the given op.
  bool FunctionHasOp(Function* function, Op op) {
    for (Node* node : function->nodes()) {
      if (node->op() == op) {
        return true;
      }
    }
    return false;
  }
};

TEST_F(DslxOptimizationTest, StdFindIndexOfLiteralArray) {
  std::string input = R"(import std

const A = u32[3]:[10, 20, 30];

fn main(i: u32) -> (bool, u32) {
  std::find_index(A, i)
})";
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> package, DslxToIr(input));
  XLS_ASSERT_OK(RunStandardPassPipeline(package.get()).status());
  XLS_ASSERT_OK_AND_ASSIGN(Function * entry, package->EntryFunction());
  XLS_VLOG(1) << package->DumpIr();
  // Verify that no ArrayIndex or ArrayUpdate operations exist in the IR.
  // TODO(b/159035667): The optimized IR is much more complicated than it should
  // be. When optimizations have been added which simplify the IR, add stricter
  // tests here.
  EXPECT_FALSE(FunctionHasOp(entry, Op::kArrayIndex));
  EXPECT_FALSE(FunctionHasOp(entry, Op::kArrayUpdate));
  EXPECT_FALSE(FunctionHasOp(entry, Op::kReverse));
  EXPECT_FALSE(FunctionHasOp(entry, Op::kOr));
}

}  // namespace
}  // namespace xls
