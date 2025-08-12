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

#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <optional>
#include <random>
#include <utility>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/flags/flag.h"
#include "absl/log/check.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "xls/common/file/filesystem.h"
#include "xls/common/file/temp_directory.h"
#include "xls/common/status/matchers.h"
#include "xls/common/status/status_macros.h"
#include "xls/dslx/frontend/pos.h"
#include "xls/fuzzer/ast_generator.h"
#include "xls/fuzzer/run_fuzz.h"
#include "xls/fuzzer/sample.h"

ABSL_FLAG(bool, wide, false, "Run with wide bits types.");
ABSL_FLAG(bool, generate_proc, false, "Generate a proc sample.");

namespace xls {
namespace {

constexpr int64_t kCallsPerSample = 8;
constexpr int64_t kSampleCount = 20;
constexpr int64_t kProcTicks = 100;

using ::absl_testing::IsOkAndHolds;
using ::testing::IsSupersetOf;

// Returns the directory in which to write crashers.
//
// Crashers are written to the undeclared outputs directory, if it is available.
// Otherwise a temporary directory is created.
absl::StatusOr<std::optional<std::filesystem::path>> GetCrasherDir() {
  char* undeclared_outputs_dir = std::getenv("TEST_UNDECLARED_OUTPUTS_DIR");
  if (undeclared_outputs_dir == nullptr) {
    return std::nullopt;
  }
  std::filesystem::path crasher_dir =
      std::filesystem::path(undeclared_outputs_dir) / "crashers";
  XLS_RETURN_IF_ERROR(RecursivelyCreateDir(crasher_dir));
  return std::move(crasher_dir);
}

class RunFuzzCodegenTest : public ::testing::Test {
 protected:
  void SetUp() final {
    XLS_ASSERT_OK_AND_ASSIGN(crasher_dir_, GetCrasherDir());
    XLS_ASSERT_OK_AND_ASSIGN(temp_dir_, TempDirectory::Create());
  }

  void TearDown() final {
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

  dslx::AstGeneratorOptions GetAstGeneratorOptions() {
    return dslx::AstGeneratorOptions{
        .max_width_bits_types = absl::GetFlag(FLAGS_wide) ? 128 : 64,
        .emit_gate = false,
        .generate_proc = absl::GetFlag(FLAGS_generate_proc),
    };
  }

  SampleOptions GetSampleOptions() {
    SampleOptions options;
    options.set_input_is_dslx(true);
    options.set_ir_converter_args({"--top=main"});
    options.set_calls_per_sample(
        absl::GetFlag(FLAGS_generate_proc) ? 0 : kCallsPerSample);
    options.set_proc_ticks(absl::GetFlag(FLAGS_generate_proc) ? kProcTicks : 0);
    options.set_convert_to_ir(true);
    options.set_optimize_ir(true);
    options.set_use_system_verilog(false);
    options.set_codegen(true);
    options.set_simulate(true);
    return options;
  }

  absl::Status RunFuzz(int64_t seed) {
    std::mt19937_64 rng(seed);
    return GenerateSampleAndRun(file_table_, rng, GetAstGeneratorOptions(),
                                GetSampleOptions(), /*run_dir=*/GetTempPath(),
                                crasher_dir_)
        .status();
  }

 private:
  dslx::FileTable file_table_;
  std::optional<std::filesystem::path> crasher_dir_;
  std::optional<TempDirectory> temp_dir_;
};

TEST_F(RunFuzzCodegenTest, FilesGenerated) {
  XLS_EXPECT_OK(RunFuzz(42));
  EXPECT_THAT(
      GetDirectoryEntries(GetTempPath()),
      IsOkAndHolds(IsSupersetOf(
          {GetTempPath() / "sample.ir", GetTempPath() / "sample.opt.ir",
           GetTempPath() / "sample.block.ir", GetTempPath() / "sample.v"})));
}

class RunFuzzCodegenSeededTest : public RunFuzzCodegenTest,
                                 public ::testing::WithParamInterface<int64_t> {
};

TEST_P(RunFuzzCodegenSeededTest, TestSeed) {
  const int64_t base_seed = GetParam();
  for (int i = 0; i < kSampleCount; ++i) {
    const int64_t sample_seed = kSampleCount * base_seed + i;
    XLS_EXPECT_OK(RunFuzz(sample_seed)) << absl::StreamFormat(
        "For seed %d, sample #%d of %d failed (sample seed = %d).", base_seed,
        i, kSampleCount, sample_seed);
  }
}

INSTANTIATE_TEST_SUITE_P(RunFuzzCodegenSeededTest, RunFuzzCodegenSeededTest,
                         ::testing::Range<int64_t>(0, 50));

}  // namespace
}  // namespace xls
