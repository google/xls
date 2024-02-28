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

#include <cstdlib>
#include <filesystem>  // NOLINT
#include <optional>
#include <string_view>
#include <utility>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xls/common/file/filesystem.h"
#include "xls/common/file/get_runfile_path.h"
#include "xls/common/file/temp_directory.h"
#include "xls/common/logging/logging.h"
#include "xls/common/status/matchers.h"
#include "xls/common/subprocess.h"
#include "xls/dslx/interp_value_utils.h"
#include "xls/fuzzer/cpp_run_fuzz.h"
#include "xls/fuzzer/cpp_sample_runner.h"
#include "xls/fuzzer/run_fuzz.h"
#include "xls/fuzzer/sample.h"

namespace xls {
namespace {

absl::StatusOr<std::filesystem::path> GetParseIrPath() {
  return GetXlsRunfilePath("xls/tools/parse_ir");
}

using status_testing::IsOkAndHolds;
using status_testing::StatusIs;
using ::testing::AllOf;
using ::testing::Contains;
using ::testing::HasSubstr;

class RunFuzzMinimizerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    XLS_ASSERT_OK_AND_ASSIGN(temp_dir_, TempDirectory::Create());
  }

  void TearDown() override {
    // Take ownership of the `temp_dir_` so it will be destroyed on return; this
    // lets us use early-exit control flow.
    TempDirectory temp_dir = *std::move(temp_dir_);
    temp_dir_.reset();

    // If the test failed, preserve the outputs in the undeclared outputs
    // directory (assuming one exists).
    if (!HasFailure()) {
      return;
    }

    const char* test_undeclared_outputs_dir =
        getenv("TEST_UNDECLARED_OUTPUTS_DIR");
    if (test_undeclared_outputs_dir == nullptr) {
      return;
    }
    std::filesystem::path undeclared_outputs_dir(test_undeclared_outputs_dir);

    const testing::TestInfo* test_info =
        testing::UnitTest::GetInstance()->current_test_info();
    if (test_info == nullptr) {
      return;
    }
    CHECK(test_info->name() != nullptr);

    std::filesystem::path test_outputs_path =
        undeclared_outputs_dir / test_info->name();
    if (test_info->type_param() != nullptr) {
      test_outputs_path /= test_info->type_param();
    }
    if (test_info->value_param() != nullptr) {
      test_outputs_path /= test_info->value_param();
    }
    CHECK(std::filesystem::create_directories(test_outputs_path));
    std::filesystem::copy(temp_dir.path(), test_outputs_path,
                          std::filesystem::copy_options::recursive);
  }

  std::filesystem::path GetTempPath() { return temp_dir_->path(); }

 private:
  std::optional<TempDirectory> temp_dir_;
};

TEST_F(RunFuzzMinimizerTest, MinimizeIRMinimizationPossible) {
  // Add an invalid codegen flag to inject an error into the running of the
  // sample. The error is unconditional so IR minimization should be able to
  // reduce the sample to a minimal function (just returns a literal).
  SampleOptions sample_options;
  sample_options.set_input_is_dslx(true);
  sample_options.set_ir_converter_args({"--top=main"});
  sample_options.set_codegen(true);
  sample_options.set_codegen_args({"--invalid_flag!!!"});

  XLS_ASSERT_OK_AND_ASSIGN(ArgsBatch args_batch,
                           dslx::ParseArgsBatch("bits[8]:7\nbits[8]:100"));
  Sample sample("fn main(x: u8) -> u8 { -x }", sample_options, args_batch);
  ASSERT_THAT(
      RunSample(sample, GetTempPath()),
      StatusIs(absl::StatusCode::kInternal,
               HasSubstr("codegen_main returned non-zero exit status")));
  XLS_ASSERT_OK_AND_ASSIGN(
      std::optional<std::filesystem::path> minimized_ir_path,
      MinimizeIr(sample, GetTempPath()));
  ASSERT_THAT(GetDirectoryEntries(GetTempPath()),
              IsOkAndHolds(Contains(GetTempPath() / "ir_minimizer_test.sh")));
  ASSERT_NE(minimized_ir_path, std::nullopt);

  // Validate the minimized IR was reduced to return a literal...
  ASSERT_THAT(GetFileContents(*minimized_ir_path),
              IsOkAndHolds(AllOf(HasSubstr("package "), HasSubstr("fn "),
                                 HasSubstr("ret literal"))));
  // ... and verify the minimized IR parses.
  XLS_ASSERT_OK_AND_ASSIGN(std::filesystem::path parse_ir_path,
                           GetParseIrPath());
  XLS_ASSERT_OK(SubprocessErrorAsStatus(
      InvokeSubprocess({parse_ir_path.string(), minimized_ir_path->string()})));
}

TEST_F(RunFuzzMinimizerTest, MinimizeIRNoMinimizationPossible) {
  // Verify that IR minimization at least generates a minimization test script
  // and doesn't blow up if the IR is not minimizable. In this case, "not
  // minimizable" means that no error is ever generated when running the sample.
  SampleOptions sample_options;
  sample_options.set_input_is_dslx(true);
  sample_options.set_ir_converter_args({"--top=main"});

  XLS_ASSERT_OK_AND_ASSIGN(ArgsBatch args_batch,
                           dslx::ParseArgsBatch("bits[8]:7\nbits[8]:100"));
  Sample sample("fn main(x: u8) -> u8 { -x }", sample_options, args_batch);
  XLS_ASSERT_OK(RunSample(sample, GetTempPath()));
  XLS_ASSERT_OK_AND_ASSIGN(
      std::optional<std::filesystem::path> minimized_ir_path,
      MinimizeIr(sample, GetTempPath()));
  EXPECT_THAT(GetDirectoryEntries(GetTempPath()),
              IsOkAndHolds(Contains(GetTempPath() / "ir_minimizer_test.sh")));
  EXPECT_EQ(minimized_ir_path, std::nullopt);
}

TEST_F(RunFuzzMinimizerTest, MinimizeJITInterpreterMismatch) {
  SampleOptions sample_options;
  sample_options.set_input_is_dslx(true);
  sample_options.set_ir_converter_args({"--top=main"});

  XLS_ASSERT_OK_AND_ASSIGN(ArgsBatch args_batch,
                           dslx::ParseArgsBatch("bits[8]:0xff\nbits[8]:0x42"));
  Sample sample("fn main(x: u8) -> u8 { -x }", sample_options, args_batch);
  XLS_ASSERT_OK(RunSample(sample, GetTempPath()));
  XLS_ASSERT_OK_AND_ASSIGN(
      std::optional<std::filesystem::path> minimized_ir_path,
      MinimizeIr(sample, GetTempPath(), /*inject_jit_result=*/"bits[32]:0x0"));
  ASSERT_THAT(GetDirectoryEntries(GetTempPath()),
              IsOkAndHolds(Contains(GetTempPath() / "ir_minimizer_test.sh")));
  ASSERT_NE(minimized_ir_path, std::nullopt);

  // Validate the minimized IR was reduced to return a literal...
  ASSERT_THAT(GetFileContents(*minimized_ir_path),
              IsOkAndHolds(AllOf(HasSubstr("package "), HasSubstr("fn "),
                                 HasSubstr("ret literal"))));
  // ... and verify the minimized IR parses.
  XLS_ASSERT_OK_AND_ASSIGN(std::filesystem::path parse_ir_path,
                           GetParseIrPath());
  XLS_ASSERT_OK(SubprocessErrorAsStatus(
      InvokeSubprocess({parse_ir_path.string(), minimized_ir_path->string()})));
}

}  // namespace
}  // namespace xls
