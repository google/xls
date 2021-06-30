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

#include <cstdio>
#include <memory>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/strings/str_format.h"
#include "xls/codegen/combinational_generator.h"
#include "xls/common/file/temp_file.h"
#include "xls/common/status/matchers.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/contrib/xlscc/translator.h"
#include "xls/contrib/xlscc/unit_test.h"
#include "xls/interpreter/channel_queue.h"
#include "xls/interpreter/function_interpreter.h"
#include "xls/interpreter/proc_interpreter.h"
#include "xls/ir/bits.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/nodes.h"
#include "xls/ir/value.h"

// Support for XLS[cc] related tests, such as invoking XLS[cc]
//  with the appropriate parameters for the test environment
class XlsccTestBase : public xls::IrTestBase {
 public:
  void Run(const absl::flat_hash_map<std::string, uint64_t>& args,
           uint64_t expected, absl::string_view cpp_source,
           xabsl::SourceLocation loc = xabsl::SourceLocation::current(),
           std::vector<absl::string_view> clang_argv = {});

  void Run(const absl::flat_hash_map<std::string, xls::Value>& args,
           xls::Value expected, absl::string_view cpp_source,
           xabsl::SourceLocation loc = xabsl::SourceLocation::current(),
           std::vector<absl::string_view> clang_argv = {});

  absl::Status ScanFile(absl::string_view cpp_src,
                        std::vector<absl::string_view> argv = {});

  // Overload which takes a translator as a parameter rather than constructing
  // and using the translator_ data member.
  static absl::Status ScanFile(absl::string_view cpp_src,
                               std::vector<absl::string_view> argv,
                               xlscc::Translator* translator);

  absl::StatusOr<std::string> SourceToIr(
      absl::string_view cpp_src, xlscc::GeneratedFunction** pfunc = nullptr,
      std::vector<absl::string_view> clang_argv = {});

  struct IOOpTest {
    IOOpTest(std::string name, int value, bool condition)
        : name(name), value(value), condition(condition) {}

    std::string name;
    int value;
    bool condition;
  };

  void IOTest(std::string content, std::list<IOOpTest> inputs,
              std::list<IOOpTest> outputs,
              absl::flat_hash_map<std::string, xls::Value> args = {});

  void ProcTest(std::string content, const xlscc::HLSBlock& block_spec,
                const absl::flat_hash_map<std::string, std::vector<xls::Value>>&
                    inputs_by_channel,
                const absl::flat_hash_map<std::string, std::vector<xls::Value>>&
                    outputs_by_channel);

  std::unique_ptr<xlscc::Translator> translator_;
};

#endif  // XLS_CONTRIB_XLSCC_UNIT_TEST_H_
