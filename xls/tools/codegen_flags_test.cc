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

#include "xls/tools/codegen_flags.h"

#include <string>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/flags/parse.h"
#include "absl/flags/reflection.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "xls/common/file/temp_file.h"
#include "xls/common/status/matchers.h"
#include "xls/tools/codegen_flags.pb.h"

namespace xls {
namespace {

using ::absl_testing::StatusIs;
using ::testing::HasSubstr;

void ParseFlagsForTest(std::vector<std::string> args) {
  std::vector<char*> argv;
  argv.reserve(args.size() + 1);
  std::string prog = "codegen_flags_test";
  argv.push_back(prog.data());
  for (std::string& arg : args) {
    argv.push_back(arg.data());
  }
  absl::ParseCommandLine(static_cast<int>(argv.size()), argv.data());
}

TEST(CodegenFlagsTest, PreservePortsFlagPropagated) {
  absl::FlagSaver flag_saver;
  ParseFlagsForTest({"--nopreserve_ports"});

  XLS_ASSERT_OK_AND_ASSIGN(CodegenFlagsProto proto, GetCodegenFlags());
  EXPECT_TRUE(proto.has_preserve_ports());
  EXPECT_FALSE(proto.preserve_ports());
}

TEST(CodegenFlagsTest, MaxTraceVerbosityFlagPropagated) {
  absl::FlagSaver flag_saver;
  ParseFlagsForTest({"--max_trace_verbosity=3"});

  XLS_ASSERT_OK_AND_ASSIGN(CodegenFlagsProto proto, GetCodegenFlags());
  EXPECT_TRUE(proto.has_max_trace_verbosity());
  EXPECT_EQ(proto.max_trace_verbosity(), 3);
}

class CodegenFlagsProtoConflictTest
    : public ::testing::TestWithParam<std::string> {};

TEST_P(CodegenFlagsProtoConflictTest, RejectsCombiningProtoAndIndividualFlag) {
  absl::FlagSaver flag_saver;
  XLS_ASSERT_OK_AND_ASSIGN(
      TempFile proto_file,
      TempFile::CreateWithContent("generator: GENERATOR_KIND_PIPELINE\n"));
  ParseFlagsForTest(
      {"--codegen_options_proto=" + proto_file.path().string(), GetParam()});

  EXPECT_THAT(GetCodegenFlags(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Cannot combine 'codegen_options_proto'")));
}

INSTANTIATE_TEST_SUITE_P(IndividualFlags, CodegenFlagsProtoConflictTest,
                         ::testing::Values("--generator=combinational",
                                           "--flop_inputs_kind=skid",
                                           "--flop_outputs_kind=skid",
                                           "--nopreserve_ports",
                                           "--max_trace_verbosity=1",
                                           "--nomaterialize_internal_fifos"));

}  // namespace
}  // namespace xls
