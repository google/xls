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

#ifndef XLS_CODEGEN_VERILOG_TEST_BASE_H_
#define XLS_CODEGEN_VERILOG_TEST_BASE_H_

#include <filesystem>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xls/codegen/vast.h"
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
  std::string simulator;
  // Whether or not to use SystemVerilog.
  bool use_system_verilog;
};

// Improve test error messages by adding a stream overload for SimulationTarget.
inline std::ostream& operator<<(std::ostream& os, SimulationTarget t) {
  os << absl::StreamFormat("%s/%s", t.simulator,
                           t.use_system_verilog ? "SystemVerilog" : "Verilog");
  return os;
}

#include "xls/simulation/simulation_targets.inc"

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

class VerilogTestBase : public testing::TestWithParam<SimulationTarget> {
 protected:
  VerilogTestBase() = default;

  // Returns the name of the test. Include the parameterization. Example:
  // TestTheThing/FancySimulatorVerilog.
  std::string TestName() {
    return ::testing::UnitTest::GetInstance()->current_test_info()->name();
  }

  // Returns the base name of the test (the string which appears in the TEST_P
  // macro). For example, for the test defined as:
  //
  //   TEST_P(FooTest, TestTheThing) { ... }
  //
  // TestBaseName returns "TestTheThing".
  std::string TestBaseName() {
    std::vector<std::string> parts = absl::StrSplit(TestName(), '/');
    XLS_CHECK_EQ(parts.size(), 2);
    return parts.front();
  }

  // Returns the Verilog simulator as determined by the test parameter.
  VerilogSimulator* GetSimulator() {
    return GetVerilogSimulator(GetParam().simulator).value();
  }

  // Returns whether or not the SystemVerilog or Verilog should be used as the
  // codegen target language as determined by the test parameters.
  bool UseSystemVerilog() { return GetParam().use_system_verilog; }

  // Validates the given verilog text by running it through the verilog
  // simulator. No functional testing is performed.
  absl::Status ValidateVerilog(absl::string_view text,
                               absl::Span<const VerilogInclude> includes = {});

  // EXPECTs that the given strings are equal and are valid Verilog. The
  // includes are used for validating the verilog.
  void ExpectVerilogEqual(absl::string_view expected, absl::string_view actual,
                          absl::Span<const VerilogInclude> includes = {});

  // EXPECTs that the given text is equal to the golden reference file specified
  // by golden_file_path. Also EXPECTs that the text is valid Verilog.
  //
  // `golden_file_path` should be relative to the main XLS source directory.
  //
  // To update golden file run test binary directly (not via bazel run/test)
  // with --test_update_golden_files:
  //
  //   ./path/to/foo_test --test_update_golden_files
  //
  void ExpectVerilogEqualToGoldenFile(
      const std::filesystem::path& golden_file_path, absl::string_view text,
      absl::Span<const VerilogInclude> includes = {});

  // Returns the path to the testdata file associated with a unit test. The
  // return path has the form:
  //
  //  ${XLS top}/${testdata_dir}/${test_file_name}_${TestBaseName}.v
  //
  // If UseSystemVerilog() is true the file extension will be ".sv" or ".v"
  // otherwise. test_file_name should be the name of the test file without the
  // .cc extension. testdata_dir should the path the the testdata directory
  // relative to the XLS source top.
  std::filesystem::path GoldenFilePath(
      absl::string_view test_file_name,
      const std::filesystem::path& testdata_dir);
};

}  // namespace verilog
}  // namespace xls

#endif  // XLS_CODEGEN_VERILOG_TEST_BASE_H_
