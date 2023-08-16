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

#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/strings/str_replace.h"
#include "xls/common/file/named_pipe.h"
#include "xls/common/file/temp_directory.h"
#include "xls/common/logging/logging.h"
#include "xls/common/status/matchers.h"
#include "xls/common/thread.h"
#include "xls/ir/bits_ops.h"
#include "xls/ir/number_parser.h"
#include "xls/simulation/verilog_test_base.h"

namespace xls {
namespace verilog {
namespace {

using ::testing::HasSubstr;
using ::testing::Not;

// Simple testbench which read 32-bit hex values from the file `{{input}}` and
// writes the value + 42 to the file `{{output}}``.
constexpr std::string_view kInputOutputTestbench = R"(
module testbench;
  reg [31:0] input_value;
  integer cnt, in_fd, out_fd;

  initial begin
    in_fd = $fopen("{{input}}","r");
    if (in_fd == 0) begin
      $display("FAILED: Cannot open input file `{{input}}`.");
      $finish;
    end
    out_fd = $fopen("{{output}}","w");
    if (out_fd == 0) begin
      $display("FAILED: Cannot open output file `{{output}}`.");
      $finish;
    end
    while ($feof(in_fd) == 0) begin
      cnt = $fscanf(in_fd, "%x\n", input_value);
      if (cnt == 0) begin
        $display("FAILED: $fscanf failed.");
        $finish;
      end
      $fwriteh(out_fd, input_value + 32'd42);
      $fwrite(out_fd, "\n");
      #1;
    end
    $display("SUCCESS");
    $finish;
  end
endmodule
)";

class TestbenchIoTest : public VerilogTestBase {};

TEST_P(TestbenchIoTest, SimpleInputOutput) {
  XLS_ASSERT_OK_AND_ASSIGN(TempDirectory temp_dir, TempDirectory::Create());
  XLS_ASSERT_OK_AND_ASSIGN(NamedPipe input_pipe,
                           NamedPipe::Create(temp_dir.path() / "input_pipe"));
  XLS_ASSERT_OK_AND_ASSIGN(NamedPipe output_pipe,
                           NamedPipe::Create(temp_dir.path() / "output_pipe"));
  std::string verilog = absl::StrReplaceAll(
      kInputOutputTestbench, {{"{{input}}", input_pipe.path().string()},
                              {"{{output}}", output_pipe.path().string()}});

  const int64_t kSampleCount = 10;
  Thread write_thread([&]() {
    FileLineWriter fw = input_pipe.OpenForWriting().value();
    XLS_VLOG(1) << "Feeding input:";
    for (int32_t i = 0; i < kSampleCount; ++i) {
      XLS_VLOG(1) << absl::StreamFormat("  input: %x", i);
      XLS_CHECK_OK(fw.WriteLine(absl::StrFormat("%x\n", i)));
    }
  });

  // Read all lines from the output named pipe and return as int32_t's.
  auto read_lines_and_convert = [&]() -> absl::StatusOr<std::vector<int32_t>> {
    XLS_ASSIGN_OR_RETURN(FileLineReader fr, output_pipe.OpenForReading());
    XLS_VLOG(1) << "Reading output:";
    std::vector<int32_t> values;
    while (true) {
      XLS_ASSIGN_OR_RETURN(std::optional<std::string> line, fr.ReadLine());
      if (!line.has_value()) {
        // EOF reached which means the writer has closed the pipe.
        break;
      }
      XLS_VLOG(1) << absl::StrFormat("Read line `%s`", *line);
      XLS_ASSIGN_OR_RETURN(
          Bits value, ParseUnsignedNumberWithoutPrefix(
                          *line, FormatPreference::kHex, /*bit_count=*/32));
      XLS_VLOG(1) << absl::StreamFormat("Value: %v", value);
      values.push_back(static_cast<int32_t>(value.ToUint64().value()));
    }
    return values;
  };

  std::vector<int32_t> results;
  Thread read_thread([&]() {
    absl::StatusOr<std::vector<int32_t>> values_or = read_lines_and_convert();
    XLS_CHECK_OK(values_or.status());
    results = std::move(values_or.value());
  });

  std::pair<std::string, std::string> stdout_stderr;
  XLS_ASSERT_OK_AND_ASSIGN(stdout_stderr,
                           GetSimulator()->Run(verilog, GetFileType()));
  read_thread.Join();
  write_thread.Join();

  EXPECT_THAT(stdout_stderr.first, HasSubstr("SUCCESS"));
  EXPECT_THAT(stdout_stderr.first, Not(HasSubstr("FAILED")));

  ASSERT_EQ(results.size(), kSampleCount);
  for (int32_t i = 0; i < kSampleCount; ++i) {
    EXPECT_EQ(results[i], i + 42);
  }
}

TEST_P(TestbenchIoTest, BadInput) {
  XLS_ASSERT_OK_AND_ASSIGN(TempDirectory temp_dir, TempDirectory::Create());
  XLS_ASSERT_OK_AND_ASSIGN(NamedPipe input_pipe,
                           NamedPipe::Create(temp_dir.path() / "input_pipe"));
  XLS_ASSERT_OK_AND_ASSIGN(NamedPipe output_pipe,
                           NamedPipe::Create(temp_dir.path() / "output_pipe"));
  std::string verilog = absl::StrReplaceAll(
      kInputOutputTestbench, {{"{{input}}", input_pipe.path().string()},
                              {"{{output}}", output_pipe.path().string()}});

  Thread write_thread([&]() {
    FileLineWriter fw = input_pipe.OpenForWriting().value();
    XLS_CHECK_OK(fw.WriteLine("thisisnotahexnumber"));
  });

  Thread read_thread([&]() {
    // Open the pipe to unblock simulation.
    output_pipe.OpenForReading().value();
  });

  std::pair<std::string, std::string> stdout_stderr;
  XLS_ASSERT_OK_AND_ASSIGN(stdout_stderr,
                           GetSimulator()->Run(verilog, GetFileType()));
  read_thread.Join();
  write_thread.Join();

  EXPECT_THAT(stdout_stderr.first, Not(HasSubstr("SUCCESS")));
  EXPECT_THAT(stdout_stderr.first, HasSubstr("FAILED: $fscanf failed"));
}

TEST_P(TestbenchIoTest, InvalidInputFile) {
  XLS_ASSERT_OK_AND_ASSIGN(TempDirectory temp_dir, TempDirectory::Create());
  XLS_ASSERT_OK_AND_ASSIGN(NamedPipe output_pipe,
                           NamedPipe::Create(temp_dir.path() / "output_pipe"));
  std::string verilog = absl::StrReplaceAll(
      kInputOutputTestbench, {{"{{input}}", "/not/a/filename"},
                              {"{{output}}", output_pipe.path().string()}});

  std::pair<std::string, std::string> stdout_stderr;
  XLS_ASSERT_OK_AND_ASSIGN(stdout_stderr,
                           GetSimulator()->Run(verilog, GetFileType()));

  EXPECT_THAT(stdout_stderr.first, Not(HasSubstr("SUCCESS")));
  EXPECT_THAT(stdout_stderr.first, HasSubstr("FAILED: Cannot open input file"));
}

TEST_P(TestbenchIoTest, InvalidOutputFile) {
  XLS_ASSERT_OK_AND_ASSIGN(TempDirectory temp_dir, TempDirectory::Create());
  XLS_ASSERT_OK_AND_ASSIGN(NamedPipe input_pipe,
                           NamedPipe::Create(temp_dir.path() / "input_pipe"));
  std::string verilog = absl::StrReplaceAll(
      kInputOutputTestbench, {{"{{input}}", input_pipe.path().string()},
                              {"{{output}}", "/not/a/filename"}});

  // We need to open the input pipe to unblock the $fopen in the testbench.
  Thread write_thread(
      [&]() { XLS_CHECK_OK(input_pipe.OpenForWriting().status()); });

  std::pair<std::string, std::string> stdout_stderr;
  XLS_ASSERT_OK_AND_ASSIGN(stdout_stderr,
                           GetSimulator()->Run(verilog, GetFileType()));
  write_thread.Join();

  EXPECT_THAT(stdout_stderr.first, Not(HasSubstr("SUCCESS")));
  EXPECT_THAT(stdout_stderr.first,
              HasSubstr("FAILED: Cannot open output file"));
}

INSTANTIATE_TEST_SUITE_P(TestbenchIoTestInstantiation, TestbenchIoTest,
                         testing::ValuesIn(kDefaultSimulationTargets),
                         ParameterizedTestName<TestbenchIoTest>);

}  // namespace
}  // namespace verilog
}  // namespace xls
