// Copyright 2020 The XLS Authors
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

#ifndef XLS_SIMULATION_VERILOG_TEST_BASE_H_
#define XLS_SIMULATION_VERILOG_TEST_BASE_H_

#include <cctype>
#include <filesystem>  // NOLINT
#include <memory>
#include <ostream>
#include <string>
#include <string_view>
#include <vector>

#include "gtest/gtest.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_replace.h"
#include "absl/strings/str_split.h"
#include "absl/types/span.h"
#include "xls/codegen/codegen_options.h"
#include "xls/codegen/module_signature.h"
#include "xls/codegen/vast/vast.h"
#include "xls/common/golden_files.h"
#include "xls/common/logging/log_lines.h"
#include "xls/common/source_location.h"
#include "xls/common/status/matchers.h"
#include "xls/simulation/module_simulator.h"
#include "xls/simulation/module_testbench.h"
#include "xls/simulation/verilog_include.h"
#include "xls/simulation/verilog_simulator.h"
#include "xls/simulation/verilog_simulators.h"

namespace xls {
namespace verilog {

// VerilogTestBase is a test base class parameterized in two dimensions: (1)
// Verilog simulator (e.g., "iverilog"), and (2) whether or not to use
// SystemVerilog or Verilog as the codegen target language. Tests which generate
// or simulate Verilog can use this as a base class to get coverage across
// multiple simulators and SystemVerilog/Verilog code generation.
//
// To use, derive from the test class, define each test using TEST_P, and
// include an INSTANTIATE_TEST_SUITE_P call.  For example:
//
//   class FooTest : public VerilogTestBase {
//     ...
//   };
//
//   TEST_P(....) { }
//
//   INSTANTIATE_TEST_SUITE_P(FooTestInstantiation, FooTest,
//                            testing::ValuesIn(kDefaultSimulationTargets),
//                            ParameterizedTestName<FooTest>());
//
// VerilogTestBase::GetSimulator() and VerilogTestBase::UseSystemVerilog()
// methods return the parameterized verilog simulator and use SystemVerilog
// options.

// Struct which the test is parameterized on.
struct SimulationTarget {
  // Name of the simulator (e.g., "iverilog"). This string is passed to
  // xls::verilog::GetVerilogSimulator.
  std::string_view simulator;
  // Whether or not to use SystemVerilog.
  bool use_system_verilog;
};

// Improve test error messages by adding a stream overload for SimulationTarget.
inline std::ostream& operator<<(std::ostream& os, const SimulationTarget& t) {
  os << absl::StreamFormat("%s/%s", t.simulator,
                           t.use_system_verilog ? "SystemVerilog" : "Verilog");
  return os;
}

#include "xls/simulation/simulation_targets.inc"  // IWYU pragma: keep

// Returns the name of the parameterized test from the Paramtype info. Use in
// INSTANTIATE_TEST_SUITE_P invocation so tests have meaningful names (e.g.,
// TestTheThing/FancySimulatorSystemVerilog) rather than numeric enumerations
// (TestTheThing/1). See above for example usage.
template <typename TestT>
std::string ParameterizedTestName(
    const testing::TestParamInfo<typename TestT::ParamType>& info) {
  // Underscores and dashes not allowed in test names. Strip them out and
  // replace string with camel case. For example, "fancy-sim" becomes
  // "FancySim".
  std::vector<std::string> parts =
      absl::StrSplit(info.param.simulator, absl::ByAnyChar("-_"));
  for (std::string& part : parts) {
    part = std::string(1, toupper(part[0])) + part.substr(1);
  }
  return absl::StrCat(absl::StrJoin(parts, ""), info.param.use_system_verilog
                                                    ? "SystemVerilog"
                                                    : "Verilog");
}

// Base class for parameterized tests that produce Verilog files.
//
// See VerilogTestBase for the common case where ParamType is SimulationTarget.
template <typename ParamType>
class VerilogTestBaseWithParam : public testing::TestWithParam<ParamType> {
 protected:
  VerilogTestBaseWithParam() = default;

  // Returns the name of the test. Include the parameterization. Example:
  // TestTheThing/FancySimulatorVerilog.
  static std::string TestName() {
    return ::testing::UnitTest::GetInstance()->current_test_info()->name();
  }

  // Returns the name of the test with slashes converted to underscores.
  static std::string SanitizedTestName() {
    std::string name =
        ::testing::UnitTest::GetInstance()->current_test_info()->name();
    return absl::StrReplaceAll(name, {{"/", "_"}});
  }

  // Returns the base name of the test (the string which appears in the TEST_P
  // macro). For example, for the test defined as:
  //
  //   TEST_P(FooTest, TestTheThing) { ... }
  //
  // TestBaseName returns "TestTheThing".
  std::string TestBaseName() {
    std::vector<std::string> parts = absl::StrSplit(TestName(), '/');
    CHECK_EQ(parts.size(), 2);
    return parts.front();
  }

  // Returns a newly created VerilogFile of the type (SystemVerilog or Verilog)
  // indicated by the TestParam.
  VerilogFile NewVerilogFile() { return VerilogFile(GetFileType()); }

  // Returns a module testbench for testing the given verilog and signature. The
  // underlying simulator and file type is determined by the test parameters.
  absl::StatusOr<std::unique_ptr<ModuleTestbench>> NewModuleTestbench(
      std::string_view verilog_text, const ModuleSignature& signature,
      absl::Span<const VerilogInclude> includes = {}) {
    return ModuleTestbench::CreateFromVerilogText(verilog_text, GetFileType(),
                                                  signature, GetSimulator(),
                                                  /*reset_dut=*/true, includes);
  }

  ModuleSimulator NewModuleSimulator(
      std::string_view verilog_text, const ModuleSignature& signature,
      absl::Span<const VerilogInclude> includes = {}) {
    return ModuleSimulator(signature, verilog_text, GetFileType(),
                           GetSimulator(), includes);
  }

  // Returns a CodegenOptions data structure with the system verilog option
  // set to the test param value.
  CodegenOptions codegen_options() {
    return CodegenOptions().use_system_verilog(UseSystemVerilog());
  }

  virtual SimulationTarget GetSimulationTarget() const = 0;

  // Returns the Verilog simulator as determined by the test parameter.
  VerilogSimulator* GetSimulator() {
    return GetVerilogSimulator(GetSimulationTarget().simulator).value();
  }

  // Returns whether or not the SystemVerilog or Verilog should be used as the
  // codegen target language as determined by the test parameters.
  bool UseSystemVerilog() const {
    return GetSimulationTarget().use_system_verilog;
  }

  // Returns the type of file (SystemVerilog or Verilog) being tested by this
  // test instance.
  FileType GetFileType() const {
    return UseSystemVerilog() ? FileType::kSystemVerilog : FileType::kVerilog;
  }

  // Validates the given verilog text by running it through the verilog
  // simulator. No functional testing is performed.
  absl::Status ValidateVerilog(
      std::string_view text,
      absl::Span<const VerilogSimulator::MacroDefinition> macro_definitions =
          {},
      absl::Span<const VerilogInclude> includes = {}) {
    return GetSimulator()->RunSyntaxChecking(text, GetFileType(),
                                             macro_definitions, includes);
  }

  // EXPECTs that the given strings are equal and are valid Verilog. The
  // includes and macro definitions are used for validating the verilog.
  void ExpectVerilogEqual(std::string_view expected, std::string_view actual,
                          absl::Span<const VerilogSimulator::MacroDefinition>
                              macro_definitions = {},
                          absl::Span<const VerilogInclude> includes = {}) {
    if (VLOG_IS_ON(1)) {
      XLS_LOG_LINES(INFO, absl::StrCat("Actual Verilog:\n", actual));
      XLS_LOG_LINES(INFO, absl::StrCat("Expected Verilog:\n", expected));
    }
    EXPECT_EQ(expected, actual);
    XLS_EXPECT_OK(ValidateVerilog(actual, macro_definitions, includes));
  }

  // EXPECTs that the given text is equal to the golden reference file
  // specified by golden_file_path. Also EXPECTs that the text is valid
  // Verilog.
  //
  // `golden_file_path` should be relative to the main XLS source directory.
  //
  // To update golden file run test binary directly (not via bazel run/test)
  // with --test_update_golden_files:
  //
  //   ./path/to/foo_test --test_update_golden_files
  //
  void ExpectVerilogEqualToGoldenFile(
      const std::filesystem::path& golden_file_path, std::string_view text,
      absl::Span<const VerilogSimulator::MacroDefinition> macro_definitions =
          {},
      absl::Span<const VerilogInclude> includes = {},
      xabsl::SourceLocation loc = xabsl::SourceLocation::current()) {
    ExpectEqualToGoldenFile(golden_file_path, text, loc);
    XLS_EXPECT_OK(ValidateVerilog(text, macro_definitions, includes));
  }

  // Returns the path to the testdata file associated with a unit test. The
  // return path has the form:
  //
  //  ${XLS top}/${testdata_dir}/${test_file_name}_${TestBaseName}.v
  //
  // If UseSystemVerilog() is true the file extension will be ".sv" or ".v"
  // otherwise. test_file_name should be the name of the test file without the
  // .cc extension. testdata_dir should the path the testdata directory
  // relative to the XLS source top.
  virtual std::filesystem::path GoldenFilePath(
      std::string_view test_file_name,
      const std::filesystem::path& testdata_dir) {
    // We suffix the golden reference files with "txt" on top of the extension
    // just to indicate they're compiler byproduct comparison points and not
    // Verilog files that have been written by hand.
    std::string filename =
        absl::StrFormat("%s_%s.%s", test_file_name, TestBaseName(),
                        UseSystemVerilog() ? "svtxt" : "vtxt");
    return testdata_dir / filename;
  }
};

class VerilogTestBase : public VerilogTestBaseWithParam<SimulationTarget> {
  SimulationTarget GetSimulationTarget() const final { return GetParam(); }
};

}  // namespace verilog
}  // namespace xls

#endif  // XLS_SIMULATION_VERILOG_TEST_BASE_H_
