// Copyright 2023 The XLS Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <algorithm>
#include <cstdlib>
#include <filesystem>  // NOLINT
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "gtest/gtest.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/inlined_vector.h"
#include "absl/flags/flag.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_replace.h"
#include "xls/codegen/codegen_options.h"
#include "xls/codegen/codegen_result.h"
#include "xls/codegen/combinational_generator.h"
#include "xls/codegen/vast/vast.h"
#include "xls/common/file/filesystem.h"
#include "xls/common/file/get_runfile_path.h"
#include "xls/common/file/temp_directory.h"
#include "xls/common/source_location.h"
#include "xls/common/status/matchers.h"
#include "xls/common/status/status_macros.h"
#include "xls/common/subprocess.h"
#include "xls/contrib/xlscc/unit_tests/unit_test.h"
#include "xls/interpreter/function_interpreter.h"
#include "xls/ir/bits.h"
#include "xls/ir/events.h"
#include "xls/ir/function.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/nodes.h"
#include "xls/ir/value.h"
#include "xls/passes/optimization_pass_pipeline.h"
#include "xls/simulation/default_verilog_simulator.h"
#include "xls/simulation/module_simulator.h"
#include "xls/simulation/verilog_simulator.h"

ABSL_FLAG(int, sample_count, 1, "number of samples to generate.");
ABSL_FLAG(int, seed, 1, "seed for pseudo-randomizer");
ABSL_FLAG(std::string, crash_path, "", "path at which to place crash data.");
ABSL_FLAG(std::string, input_path, "", "path at which to find test files.");
ABSL_FLAG(bool, run_failed, false, "run failed tests");

namespace xlscc {

class GeneratedTester : public XlsccTestBase {
 public:
  GeneratedTester()
      : verilog_simulator_(xls::verilog::GetDefaultVerilogSimulator()) {}
  void TestBody() override {}

  absl::Status RunExisting(const std::filesystem::path& cc_filename,
                           const std::filesystem::path& exec_filename) {
    XLS_ASSIGN_OR_RETURN(std::string test_content,
                         xls::GetFileContents(cc_filename));

    XLS_ASSIGN_OR_RETURN(xls::SubprocessResult result_value,
                         xls::InvokeSubprocess({std::string(exec_filename)}));

    std::string stripped_result = absl::StrReplaceAll(
        result_value.stdout_content, {{"0b", ""}, {".", ""}});
    std::reverse(stripped_result.begin(), stripped_result.end());
    absl::InlinedVector<bool, 1> expected_in;
    expected_in.reserve(stripped_result.size());
    for (char& ch : stripped_result) {
      expected_in.push_back(ch == '1');
    }
    auto expected = xls::Value(xls::Bits(expected_in));
    XLS_ASSIGN_OR_RETURN(auto calc_result, RunIntTest(expected, test_content));
    if (calc_result != expected) {
      std::cout << "ac: " << expected << " xls: " << calc_result << '\n';
      return absl::InternalError("test failed");
    }
    return absl::OkStatus();
  }

 private:
  absl::StatusOr<xls::Value> RunOptimized(
      const absl::flat_hash_map<std::string, xls::Value>& args,
      const xls::InterpreterEvents& unopt_events) {
    // Run main pipeline.
    (void)xls::RunOptimizationPassPipeline(package_.get());

    // Run interpreter on optimized IR.
    {
      XLS_ASSIGN_OR_RETURN(xls::Function * main, package_->GetTopAsFunction());
      XLS_ASSIGN_OR_RETURN(xls::InterpreterResult<xls::Value> result,
                           xls::InterpretFunctionKwargs(main, args));
      XLS_RETURN_IF_ERROR(xls::InterpreterEventsToStatus(result.events));
      return result.value;
    }
  }

  absl::StatusOr<xls::Value> RunSimulated(
      const absl::flat_hash_map<std::string, xls::Value>& args) {
    CHECK_EQ(package_->functions().size(), 1);
    std::optional<xls::FunctionBase*> top = package_->GetTop();
    CHECK(top.has_value());
    CHECK(top.value()->IsFunction());

    XLS_ASSIGN_OR_RETURN(
        xls::verilog::CodegenResult result,
        xls::verilog::GenerateCombinationalModule(
            top.value(),
            xls::verilog::CodegenOptions().use_system_verilog(false)));

    VLOG(3) << "Verilog text:\n" << result.verilog_text;
    xls::verilog::ModuleSimulator simulator(
        result.signature, result.verilog_text, xls::verilog::FileType::kVerilog,
        verilog_simulator_.get());
    XLS_ASSIGN_OR_RETURN(xls::Value actual, simulator.RunFunction(args));
    return actual;
  }

  absl::StatusOr<xls::Value> RunAndExpectEqGenerated(
      const xls::Value& expected, std::string_view cpp_source,
      xabsl::SourceLocation loc, std::vector<std::string_view>& clang_argv) {
    XLS_ASSIGN_OR_RETURN(std::string ir,
                         SourceToIr(cpp_source, nullptr, clang_argv));

    XLS_ASSIGN_OR_RETURN(package_, ParsePackage(ir));

    absl::flat_hash_map<std::string, xls::Value> args;
    xls::InterpreterEvents unopt_events;

    // Run interpreter on unoptimized IR.
    {
      XLS_ASSIGN_OR_RETURN(xls::Function * entry, package_->GetTopAsFunction());
      XLS_ASSIGN_OR_RETURN(xls::InterpreterResult<xls::Value> result,
                           xls::InterpretFunctionKwargs(entry, args));
      XLS_RETURN_IF_ERROR(xls::InterpreterEventsToStatus(result.events));
      unopt_events = result.events;
      if (expected != result.value) {
        return result.value;
      }
    }

    xls::Value result;
    XLS_ASSIGN_OR_RETURN(result, RunOptimized(args, unopt_events));
    if (expected != result) {
      return result;
    }
    XLS_ASSIGN_OR_RETURN(result, RunSimulated(args));
    return result;
  }

  absl::StatusOr<xls::Value> RunIntTest(
      const xls::Value& expected, std::string_view cpp_source,
      xabsl::SourceLocation loc = xabsl::SourceLocation::current()) {
    XLS_ASSIGN_OR_RETURN(
        std::string ac_int_path,
        xls::GetXlsRunfilePath("external/com_github_hlslibs_ac_types/include/ac_int.h"));
    XLS_ASSIGN_OR_RETURN(
        std::string xls_int_path,
        xls::GetXlsRunfilePath("xls/contrib/xlscc/synth_only/xls_int.h"));
    XLS_ASSIGN_OR_RETURN(
        std::string xls_compat_path,
        xls::GetXlsRunfilePath(
            "xls/contrib/xlscc/synth_only/ac_compat/ac_int.h"));

    // Get the path that includes the ac_datatypes folder, so that the
    //  ac_datatypes headers can be included with the form:
    // #include "ac_datatypes/include/include/foo.h"
    auto ac_int_dir = std::filesystem::path(ac_int_path);
    ac_int_dir = ac_int_dir.parent_path().parent_path();
    std::string ac_include = std::string("-I") + ac_int_dir.string();

    std::string xls_int_dir = std::filesystem::path(xls_int_path).parent_path();
    std::string xls_include = std::string("-I") + xls_int_dir;

    std::string xls_compat_dir =
        std::filesystem::path(xls_compat_path).parent_path();
    std::string xls_compat_include = std::string("-I") + xls_compat_dir;

    std::vector<std::string_view> argv;
    argv.push_back(xls_include);
    argv.push_back(ac_include);
    argv.push_back(xls_compat_include);
    argv.push_back("-D__SYNTHESIS__");
    XLS_ASSIGN_OR_RETURN(
        xls::Value result,
        RunAndExpectEqGenerated(expected, cpp_source, loc, argv));
    return result;
  }

  std::unique_ptr<xls::verilog::VerilogSimulator> verilog_simulator_;
};

TEST_F(GeneratedTester, Simple) {
  xlscc::GeneratedTester tester;
  int seed = absl::GetFlag(FLAGS_seed);
  std::string input_path = absl::GetFlag(FLAGS_input_path);
  std::string crash_path = absl::GetFlag(FLAGS_crash_path);
  bool run_failed = absl::GetFlag(FLAGS_run_failed);

  absl::StatusOr<xls::TempDirectory> temp_dir_class =
      xls::TempDirectory::Create();
  std::string temp_dir;
  if (temp_dir_class.ok()) {
    temp_dir = temp_dir_class.value().path();
    LOG(INFO) << "using temp directory: " << temp_dir;
  } else {
    LOG(QFATAL) << "Failed to create temp directory: ";
  }

  std::srand(seed);

  if (!run_failed) {
    if (!crash_path.empty()) {
      if (!std::filesystem::create_directory(crash_path)) {
        LOG(WARNING) << "failed to create crash path: " << crash_path;
      }
    } else {
      crash_path = temp_dir;
    }
    XLS_ASSERT_OK_AND_ASSIGN(
        std::filesystem::path exec_filename,
        xls::GetXlsRunfilePath(input_path + std::to_string(seed)));

    std::filesystem::path cc_filepath = exec_filename.string() + ".cc";
    absl::Status status = RunExisting(cc_filepath, exec_filename);
    XLS_EXPECT_OK(status) << "failed test case: " << std::to_string(seed) << " "
                          << cc_filepath;
  } else {
    absl::StatusOr<std::vector<std::filesystem::path>> files =
        xls::GetDirectoryEntries(crash_path);
    if (files.ok()) {
      for (const std::filesystem::path& file : files.value()) {
        if (file.extension() != ".cc") {
          continue;
        }
        std::string exe_path = file.string();
        exe_path = exe_path.substr(0, exe_path.size() - 3);
        absl::Status test_result = tester.RunExisting(file, exe_path);
        if (!test_result.ok()) {
          std::cout << "test failed: " << file << '\n';
        } else {
          std::cout << "test succeeded: " << file << '\n';
        }
      }
    }
  }
}

}  // namespace xlscc

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
