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

#include "xls/spin/spin_runner.h"

#include <cstdint>
#include <filesystem>
#include <string>
#include <string_view>
#include <tuple>

#include "absl/log/log.h"
#include "absl/log/scoped_mock_log.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "xls/common/file/filesystem.h"
#include "xls/common/file/get_runfile_path.h"
#include "xls/common/file/temp_directory.h"
#include "xls/common/status/matchers.h"
#include "xls/dslx/create_import_data.h"
#include "xls/dslx/parse_and_typecheck.h"

namespace xls::spin {
namespace {

using ::absl::ScopedMockLog;
using ::absl_testing::IsOk;
using ::testing::_;
using ::testing::HasSubstr;

class PromelaSpinRunnerTest : public testing::Test {};

/* RunSpinCheck */

TEST_F(PromelaSpinRunnerTest, RunSpinCheck_NoTestProc) {
  constexpr std::string_view kDslx = R"(
proc SimpleProc {
    config() {}
    init { () }
    next(state: ()) {}
}
)";
  ScopedMockLog log;
  EXPECT_CALL(log, Log(absl::LogSeverity::kWarning, _,
                       HasSubstr("no #[test_proc] found")));
  log.StartCapturingLogs();
  EXPECT_THAT(RunSpinCheck(kDslx, "simple.x", "simple"), IsOk());
  log.StopCapturingLogs();
}

TEST_F(PromelaSpinRunnerTest, RunSpinCheck_Passthrough) {
  XLS_ASSERT_OK_AND_ASSIGN(
      auto src, GetXlsRunfilePath("xls/spin/testdata/passthrough.x"));
  XLS_ASSERT_OK_AND_ASSIGN(std::string dslx_source, GetFileContents(src));
  SpinRunOptions opts;
  opts.exec_type = SpinExecutionType::kGuided;
  opts.type_inference_v2 = true;
  ScopedMockLog log;
  EXPECT_CALL(log, Log(absl::LogSeverity::kInfo, _,
                       HasSubstr("Starting SPIN guided check")));
  EXPECT_CALL(log, Log(absl::LogSeverity::kInfo, _,
                       HasSubstr("Running SPIN simulation on")));
  log.StartCapturingLogs();
  EXPECT_THAT(RunSpinCheck(dslx_source, src.string(), "passthrough", opts),
              IsOk());
  log.StopCapturingLogs();
}

TEST_F(PromelaSpinRunnerTest, RunSpinCheck_Exhaustive) {
  XLS_ASSERT_OK_AND_ASSIGN(
      auto src, GetXlsRunfilePath("xls/spin/testdata/passthrough.x"));
  XLS_ASSERT_OK_AND_ASSIGN(std::string dslx_source, GetFileContents(src));
  SpinRunOptions opts;
  opts.exec_type = SpinExecutionType::kExhaustive;
  opts.type_inference_v2 = true;
  ScopedMockLog log;
  EXPECT_CALL(log, Log(absl::LogSeverity::kInfo, _,
                       HasSubstr("Starting SPIN exhaustive check")));
  EXPECT_CALL(log, Log(absl::LogSeverity::kInfo, _,
                       HasSubstr("Running SPIN exhaustive search on")));
  log.StartCapturingLogs();
  EXPECT_THAT(RunSpinCheck(dslx_source, src.string(), "passthrough", opts),
              IsOk());
  log.StopCapturingLogs();
}

constexpr std::string_view kTwoTestProcsDslx = R"(
#![feature(type_inference_v2)]

#[test_proc]
proc FirstTest {
    terminator: chan<bool> out;
    config(terminator: chan<bool> out) { (terminator,) }
    init { () }
    next(state: ()) {
        let tok = send(join(), terminator, true);
    }
}

#[test_proc]
proc SecondTest {
    terminator: chan<bool> out;
    config(terminator: chan<bool> out) { (terminator,) }
    init { () }
    next(state: ()) {
        let tok = send(join(), terminator, true);
    }
}
)";

TEST_F(PromelaSpinRunnerTest, RunSpinCheck_MultipleProcs_WithFilter) {
  XLS_ASSERT_OK_AND_ASSIGN(auto tmp, TempDirectory::Create("two_procs"));
  const std::filesystem::path src = tmp.path() / "test.x";
  XLS_ASSERT_OK(SetFileContents(src, kTwoTestProcsDslx));
  SpinRunOptions opts;
  opts.exec_type = SpinExecutionType::kGuided;
  opts.type_inference_v2 = true;
  opts.test_filter = "SecondTest";
  ScopedMockLog log;
  EXPECT_CALL(log, Log(absl::LogSeverity::kInfo, _,
                       HasSubstr("Converting test proc 'SecondTest'")));
  log.StartCapturingLogs();
  EXPECT_THAT(RunSpinCheck(kTwoTestProcsDslx, src.string(), "test", opts),
              IsOk());
  log.StopCapturingLogs();
}

TEST_F(PromelaSpinRunnerTest, RunSpinCheck_MultipleProcs_WithoutFilter) {
  XLS_ASSERT_OK_AND_ASSIGN(auto tmp, TempDirectory::Create("two_procs"));
  const std::filesystem::path src = tmp.path() / "test.x";
  XLS_ASSERT_OK(SetFileContents(src, kTwoTestProcsDslx));
  SpinRunOptions opts;
  opts.exec_type = SpinExecutionType::kGuided;
  opts.type_inference_v2 = true;
  ScopedMockLog log;
  EXPECT_CALL(log, Log(absl::LogSeverity::kWarning, _,
                       HasSubstr("multiple #[test_proc]")));
  EXPECT_CALL(log, Log(absl::LogSeverity::kInfo, _,
                       HasSubstr("Converting test proc 'FirstTest'")));
  log.StartCapturingLogs();
  EXPECT_THAT(RunSpinCheck(kTwoTestProcsDslx, src.string(), "test", opts),
              IsOk());
  log.StopCapturingLogs();
}

/* BuildDslxChannelNameMap */

// Returns the DslxChannelNameMap entry for (proc, instance, var), or "".
static std::string MapLookup(const DslxChannelNameMap& m, std::string_view proc,
                             int64_t instance, std::string_view var) {
  auto it = m.find(
      std::make_tuple(std::string(proc), instance, std::string(var)));
  return (it != m.end()) ? it->second : "";
}

TEST_F(PromelaSpinRunnerTest, BuildDslxChannelNameMap_ChannelDecl) {
  constexpr std::string_view kDslx = R"(
#![feature(type_inference_v2)]
#[test_proc]
proc SimpleTest {
    terminator: chan<bool> out;
    config(terminator: chan<bool> out) {
        let (req_s, req_r) = chan<u32>("req");
        (terminator,)
    }
    init { () }
    next(state: ()) { let tok = send(join(), terminator, true); }
}
)";
  dslx::ImportData import_data = dslx::CreateImportDataForTest();
  XLS_ASSERT_OK_AND_ASSIGN(
      dslx::TypecheckedModule tm,
      dslx::ParseAndTypecheck(kDslx, "test.x", "test", &import_data));
  DslxChannelNameMap map = BuildDslxChannelNameMap(*tm.module);
  EXPECT_EQ(MapLookup(map, "SimpleTest", 0, "req_s"), "req");
  EXPECT_EQ(MapLookup(map, "SimpleTest", 0, "req_r"), "req");
}

TEST_F(PromelaSpinRunnerTest, BuildDslxChannelNameMap_MultipleInstances) {
  // Same proc type spawned twice with different channel args: each instance
  // must map its parameter to its own ChannelDecl string.
  constexpr std::string_view kDslx = R"(
#![feature(type_inference_v2)]
proc Sub {
    req_r: chan<u32> in;
    config(req_r: chan<u32> in) { (req_r,) }
    init { () }
    next(state: ()) { let (tok, _) = recv(join(), req_r); }
}
#[test_proc]
proc ParentTest {
    terminator: chan<bool> out;
    config(terminator: chan<bool> out) {
        let (ch_a_s, ch_a_r) = chan<u32>("ch_a");
        let (ch_b_s, ch_b_r) = chan<u32>("ch_b");
        spawn Sub(ch_a_r);
        spawn Sub(ch_b_r);
        (terminator,)
    }
    init { () }
    next(state: ()) { let tok = send(join(), terminator, true); }
}
)";
  dslx::ImportData import_data = dslx::CreateImportDataForTest();
  XLS_ASSERT_OK_AND_ASSIGN(
      dslx::TypecheckedModule tm,
      dslx::ParseAndTypecheck(kDslx, "test.x", "test", &import_data));
  DslxChannelNameMap map = BuildDslxChannelNameMap(*tm.module);
  EXPECT_EQ(MapLookup(map, "Sub", 0, "req_r"), "ch_a");
  EXPECT_EQ(MapLookup(map, "Sub", 1, "req_r"), "ch_b");
}

}  // namespace
}  // namespace xls::spin
