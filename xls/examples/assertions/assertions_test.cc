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

#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "xls/codegen/module_signature.h"
#include "xls/codegen/module_signature.pb.h"
#include "xls/codegen/vast/vast.h"
#include "xls/common/file/filesystem.h"
#include "xls/common/file/get_runfile_path.h"
#include "xls/common/logging/log_lines.h"
#include "xls/common/status/matchers.h"
#include "xls/ir/bits.h"
#include "xls/ir/value.h"
#include "xls/simulation/default_verilog_simulator.h"
#include "xls/simulation/module_simulator.h"
#include "xls/simulation/verilog_simulator.h"

namespace xls {
namespace {

using ::absl_testing::StatusIs;
using ::testing::ContainerEq;
using ::testing::HasSubstr;

TEST(AssertionsMainTest, CombinationalTest) {
  XLS_ASSERT_OK_AND_ASSIGN(
      std::filesystem::path verilog_path,
      GetXlsRunfilePath("xls/examples/assertions/assertions_comb.sv"));
  XLS_ASSERT_OK_AND_ASSIGN(
      std::filesystem::path signature_path,
      GetXlsRunfilePath(
          "xls/examples/assertions/assertions_comb.sig.textproto"));

  std::unique_ptr<verilog::VerilogSimulator> verilog_simulator =
      verilog::GetDefaultVerilogSimulator();
  if (!verilog_simulator->DoesSupportSystemVerilog() ||
      !verilog_simulator->DoesSupportAssertions()) {
    return;
  }

  XLS_ASSERT_OK_AND_ASSIGN(std::string verilog_text,
                           GetFileContents(verilog_path));
  XLS_VLOG_LINES(3, verilog_text);

  verilog::ModuleSignatureProto signature_proto;
  XLS_ASSERT_OK(ParseTextProtoFile(signature_path, &signature_proto));
  XLS_ASSERT_OK_AND_ASSIGN(
      verilog::ModuleSignature signature,
      verilog::ModuleSignature::FromProto(signature_proto));

  verilog::ModuleSimulator simulator(signature, verilog_text,
                                     verilog::FileType::kSystemVerilog,
                                     verilog_simulator.get());

  // Test normal operation
  {
    std::vector<absl::flat_hash_map<std::string, Value>> args_sets;
    args_sets.push_back({{"y", Value(UBits(4, 32))}});
    args_sets.push_back({{"y", Value(UBits(10, 32))}});
    args_sets.push_back({{"y", Value(UBits(14, 32))}});

    std::vector<Value> expected_outputs;
    expected_outputs.push_back(Value(UBits(34, 32)));
    expected_outputs.push_back(Value(UBits(10, 32)));
    expected_outputs.push_back(Value(UBits(14, 32)));

    XLS_ASSERT_OK_AND_ASSIGN(std::vector<Value> outputs,
                             simulator.RunBatched(args_sets));

    EXPECT_THAT(outputs, ContainerEq(expected_outputs));
  }

  // Test assertion failure.
  {
    std::vector<absl::flat_hash_map<std::string, Value>> args_sets;
    args_sets.push_back({{"y", Value(UBits(9, 32))}});

    std::vector<Value> expected_outputs;
    expected_outputs.push_back(Value(UBits(0, 32)));

    EXPECT_THAT(simulator.RunBatched(args_sets),
                StatusIs(absl::StatusCode::kAborted,
                         HasSubstr("Assertion failure via assert!")));
  }
}

TEST(AssertionsMainTest, PipelinedTest) {
  XLS_ASSERT_OK_AND_ASSIGN(
      std::filesystem::path verilog_path,
      GetXlsRunfilePath("xls/examples/assertions/assertions_4_stages.sv"));
  XLS_ASSERT_OK_AND_ASSIGN(
      std::filesystem::path signature_path,
      GetXlsRunfilePath(
          "xls/examples/assertions/assertions_4_stages.sig.textproto"));

  std::unique_ptr<verilog::VerilogSimulator> verilog_simulator =
      verilog::GetDefaultVerilogSimulator();

  if (!verilog_simulator->DoesSupportSystemVerilog() ||
      !verilog_simulator->DoesSupportAssertions()) {
    return;
  }

  XLS_ASSERT_OK_AND_ASSIGN(std::string verilog_text,
                           GetFileContents(verilog_path));
  XLS_VLOG_LINES(3, verilog_text);

  verilog::ModuleSignatureProto signature_proto;
  XLS_ASSERT_OK(ParseTextProtoFile(signature_path, &signature_proto));
  XLS_ASSERT_OK_AND_ASSIGN(
      verilog::ModuleSignature signature,
      verilog::ModuleSignature::FromProto(signature_proto));

  verilog::ModuleSimulator simulator(signature, verilog_text,
                                     verilog::FileType::kSystemVerilog,
                                     verilog_simulator.get());

  // Test normal operation
  {
    std::vector<absl::flat_hash_map<std::string, Value>> args_sets;
    args_sets.push_back({{"y", Value(UBits(4, 32))}});
    args_sets.push_back({{"y", Value(UBits(10, 32))}});
    args_sets.push_back({{"y", Value(UBits(14, 32))}});

    std::vector<Value> expected_outputs;
    expected_outputs.push_back(Value(UBits(34, 32)));
    expected_outputs.push_back(Value(UBits(10, 32)));
    expected_outputs.push_back(Value(UBits(14, 32)));

    XLS_ASSERT_OK_AND_ASSIGN(std::vector<Value> outputs,
                             simulator.RunBatched(args_sets));

    EXPECT_THAT(outputs, ContainerEq(expected_outputs));
  }

  // Test assertion failure.
  {
    std::vector<absl::flat_hash_map<std::string, Value>> args_sets;
    args_sets.push_back({{"y", Value(UBits(15, 32))}});

    std::vector<Value> expected_outputs;
    expected_outputs.push_back(Value(UBits(0, 32)));

    EXPECT_THAT(simulator.RunBatched(args_sets),
                StatusIs(absl::StatusCode::kAborted,
                         HasSubstr("Assertion failure via assert!")));
  }
}

}  // namespace
}  // namespace xls
