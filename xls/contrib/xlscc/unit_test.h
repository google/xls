// Copyright 2021 The XLS Authors
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

#ifndef XLS_CONTRIB_XLSCC_UNIT_TEST_H_
#define XLS_CONTRIB_XLSCC_UNIT_TEST_H_

#include <cstdint>
#include <cstdio>
#include <memory>
#include <vector>

#include "gtest/gtest.h"
#include "absl/strings/str_format.h"
#include "xls/common/file/temp_file.h"
#include "xls/common/status/status_macros.h"
#include "xls/contrib/xlscc/cc_parser.h"
#include "xls/contrib/xlscc/hls_block.pb.h"
#include "xls/contrib/xlscc/translator.h"
#include "xls/ir/bits.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/value.h"

// Support for XLS[cc] related tests, such as invoking XLS[cc]
//  with the appropriate parameters for the test environment
class XlsccTestBase : public xls::IrTestBase {
 public:
  void Run(const absl::flat_hash_map<std::string, uint64_t>& args,
           uint64_t expected, std::string_view cpp_source,
           xabsl::SourceLocation loc = xabsl::SourceLocation::current(),
           std::vector<std::string_view> clang_argv = {});

  void Run(const absl::flat_hash_map<std::string, xls::Value>& args,
           xls::Value expected, std::string_view cpp_source,
           xabsl::SourceLocation loc = xabsl::SourceLocation::current(),
           std::vector<std::string_view> clang_argv = {});

  void RunWithStatics(
      const absl::flat_hash_map<std::string, xls::Value>& args,
      const absl::Span<xls::Value>& expected_outputs,
      std::string_view cpp_source,
      xabsl::SourceLocation loc = xabsl::SourceLocation::current());

  absl::Status ScanFile(xls::TempFile& temp,
                        std::vector<std::string_view> clang_argv = {},
                        bool io_test_mode = false,
                        bool error_on_init_interval = false);

  absl::Status ScanFile(std::string_view cpp_src,
                        std::vector<std::string_view> clang_argv = {},
                        bool io_test_mode = false,
                        bool error_on_init_interval = false);

  // Overload which takes a translator as a parameter rather than constructing
  // and using the translator_ data member.
  static absl::Status ScanTempFileWithContent(
      xls::TempFile& temp, std::vector<std::string_view> argv,
      xlscc::CCParser* translator, const char* top_name = "my_package");

  static absl::Status ScanTempFileWithContent(
      std::string_view cpp_src, std::vector<std::string_view> argv,
      xlscc::CCParser* translator, const char* top_name = "my_package");

  absl::StatusOr<std::string> SourceToIr(
      xls::TempFile& temp, xlscc::GeneratedFunction** pfunc = nullptr,
      std::vector<std::string_view> clang_argv = {}, bool io_test_mode = false);

  absl::StatusOr<std::string> SourceToIr(
      std::string_view cpp_src, xlscc::GeneratedFunction** pfunc = nullptr,
      std::vector<std::string_view> clang_argv = {}, bool io_test_mode = false);

  struct IOOpTest {
    IOOpTest(std::string name, int value, bool condition)
        : name(name),
          value(xls::Value(xls::SBits(value, 32))),
          condition(condition) {}
    IOOpTest(std::string name, xls::Value value, bool condition)
        : name(name), value(value), condition(condition) {}

    std::string name;
    xls::Value value;
    bool condition;
  };

  void ProcTest(std::string content, std::optional<xlscc::HLSBlock> block_spec,
                const absl::flat_hash_map<std::string, std::list<xls::Value>>&
                    inputs_by_channel,
                const absl::flat_hash_map<std::string, std::list<xls::Value>>&
                    outputs_by_channel,
                const int min_ticks = 1, const int max_ticks = 100,
                int top_level_init_interval = 0);

  absl::StatusOr<uint64_t> GetStateBitsForProcNameContains(
      std::string_view name_cont);

  absl::StatusOr<xlscc_metadata::MetadataOutput> GenerateMetadata();

  absl::StatusOr<xlscc::HLSBlock> GetBlockSpec();

  std::unique_ptr<xls::Package> package_;
  std::unique_ptr<xlscc::Translator> translator_;
  xlscc::HLSBlock block_spec_;
};

#endif  // XLS_CONTRIB_XLSCC_UNIT_TEST_H_
