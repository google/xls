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
#include <list>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "gtest/gtest.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/log_sink.h"
#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "xls/common/file/temp_file.h"
#include "xls/common/status/status_macros.h"
#include "xls/contrib/xlscc/cc_parser.h"
#include "xls/contrib/xlscc/hls_block.pb.h"
#include "xls/contrib/xlscc/translator.h"
#include "xls/ir/bits.h"
#include "xls/ir/events.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/node.h"
#include "xls/ir/value.h"

struct CapturedLogEntry {
  CapturedLogEntry();
  explicit CapturedLogEntry(const ::absl::LogEntry& entry);

  std::string text_message;
  absl::LogSeverity log_severity;
  int verbosity = 0;
  std::string source_filename;
  std::string source_basename;
  int source_line = 0;
  bool prefix = false;
};

// Support for XLS[cc] related tests, such as invoking XLS[cc]
//  with the appropriate parameters for the test environment
class XlsccTestBase : public xls::IrTestBase, public ::absl::LogSink {
 public:
  XlsccTestBase();

  ~XlsccTestBase() override;

  void Send(const ::absl::LogEntry& entry) override;

  void Run(const absl::flat_hash_map<std::string, uint64_t>& args,
           uint64_t expected, std::string_view cpp_source,
           xabsl::SourceLocation loc = xabsl::SourceLocation::current(),
           std::vector<std::string_view> clang_argv = {},
           int64_t max_unroll_iters = 0);

  void Run(const absl::flat_hash_map<std::string, xls::Value>& args,
           xls::Value expected, std::string_view cpp_source,
           xabsl::SourceLocation loc = xabsl::SourceLocation::current(),
           std::vector<std::string_view> clang_argv = {},
           int64_t max_unroll_iters = 0);

  absl::StatusOr<std::vector<std::string>> GetClangArgForIntTest() const;

  void RunAcDatatypeTest(
      const absl::flat_hash_map<std::string, uint64_t>& args, uint64_t expected,
      std::string_view cpp_source,
      xabsl::SourceLocation loc = xabsl::SourceLocation::current());

  void RunWithStatics(
      const absl::flat_hash_map<std::string, xls::Value>& args,
      const absl::Span<xls::Value>& expected_outputs,
      std::string_view cpp_source,
      xabsl::SourceLocation loc = xabsl::SourceLocation::current());

  absl::Status ScanFile(xls::TempFile& temp,
                        std::vector<std::string_view> clang_argv = {},
                        bool io_test_mode = false,
                        bool error_on_init_interval = false,
                        bool error_on_uninitialized = false,
                        xls::SourceLocation loc = xls::SourceLocation(),
                        bool fail_xlscc_check = false,
                        int64_t max_unroll_iters = 0,
                        const char* top_class_name = "");

  absl::Status ScanFile(std::string_view cpp_src,
                        std::vector<std::string_view> clang_argv = {},
                        bool io_test_mode = false,
                        bool error_on_init_interval = false,
                        bool error_on_uninitialized = false,
                        xls::SourceLocation loc = xls::SourceLocation(),
                        bool fail_xlscc_check = false,
                        int64_t max_unroll_iters = 0,
                        const char* top_class_name = "");

  // Overload which takes a translator as a parameter rather than constructing
  // and using the translator_ data member.
  static absl::Status ScanTempFileWithContent(
      xls::TempFile& temp, std::vector<std::string_view> argv,
      xlscc::CCParser* translator, const char* top_name = "my_package",
      const char* top_class_name = "");

  static absl::Status ScanTempFileWithContent(
      std::string_view cpp_src, std::vector<std::string_view> argv,
      xlscc::CCParser* translator, const char* top_name = "my_package",
      const char* top_class_name = "");

  absl::StatusOr<std::string> SourceToIr(
      xls::TempFile& temp, xlscc::GeneratedFunction** pfunc = nullptr,
      const std::vector<std::string_view>& clang_argv = {},
      bool io_test_mode = false, int64_t max_unroll_iters = 0);

  absl::StatusOr<std::string> SourceToIr(
      std::string_view cpp_src, xlscc::GeneratedFunction** pfunc = nullptr,
      const std::vector<std::string_view>& clang_argv = {},
      bool io_test_mode = false, int64_t max_unroll_iters = 0);

  struct IOOpTest {
    IOOpTest(std::string name, int value, bool condition)
        : name(name),
          value(xls::Value(xls::SBits(value, 32))),
          condition(condition) {}
    IOOpTest(std::string name, xls::Value value, bool condition)
        : name(name), value(value), condition(condition) {}
    IOOpTest(std::string name, xls::Value value, const char* message,
             xlscc::TraceType trace_type, std::string label = "")
        : name(name),
          value(value),
          message(message),
          trace_type(trace_type),
          label(label) {}

    std::string name;
    xls::Value value;
    bool condition = true;
    std::string message;
    xlscc::TraceType trace_type = xlscc::TraceType::kNull;
    std::string label;
  };

  void ProcTest(std::string_view content,
                std::optional<xlscc::HLSBlock> block_spec,
                const absl::flat_hash_map<std::string, std::list<xls::Value>>&
                    inputs_by_channel,
                const absl::flat_hash_map<std::string, std::list<xls::Value>>&
                    outputs_by_channel,
                int min_ticks = 1, int max_ticks = 100,
                int top_level_init_interval = 0,
                const char* top_class_name = "",
                absl::Status expected_tick_status = absl::OkStatus(),
                const absl::flat_hash_map<std::string, xls::InterpreterEvents>&
                    expected_events_by_proc_name = {});

  void IOTest(std::string_view content, std::list<IOOpTest> inputs,
              std::list<IOOpTest> outputs,
              absl::flat_hash_map<std::string, xls::Value> args = {});

  absl::StatusOr<uint64_t> GetStateBitsForProcNameContains(
      std::string_view name_cont);
  absl::StatusOr<uint64_t> GetBitsForChannelNameContains(
      std::string_view name_cont);

  absl::StatusOr<xlscc_metadata::MetadataOutput> GenerateMetadata();

  absl::StatusOr<xlscc::HLSBlock> GetBlockSpec();

  absl::StatusOr<std::vector<xls::Node*>> GetIOOpsForChannel(
      xls::FunctionBase* proc, std::string_view channel);
  static absl::Status TokensForNode(
      xls::Node* node, absl::flat_hash_set<xls::Node*>& predecessors);
  absl::StatusOr<bool> NodeIsAfterTokenWise(xls::Proc* proc, xls::Node* before,
                                            xls::Node* after);

  absl::StatusOr<std::vector<xls::Node*>> GetOpsForChannelNameContains(
      std::string_view channel);

  static void GetTokenOperandsDeeply(xls::Node* node,
                                     absl::flat_hash_set<xls::Node*>& operands);

  absl::StatusOr<absl::flat_hash_map<xls::Node*, int64_t>>
  GetStatesByIONodeForFSMProc(std::string_view func_name);

  std::unique_ptr<xls::Package> package_;
  std::unique_ptr<xlscc::Translator> translator_;
  xlscc::HLSBlock block_spec_;
  bool generate_fsms_for_pipelined_loops_ = false;
  bool merge_states_ = false;
  bool split_states_on_channel_ops_ = false;

 protected:
  std::vector<CapturedLogEntry> log_entries_;
};

#endif  // XLS_CONTRIB_XLSCC_UNIT_TEST_H_
