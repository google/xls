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
using ::testing::ElementsAre;
using ::testing::HasSubstr;
using ::testing::IsEmpty;

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

TEST(CodegenFlagsTest, ProtoDefaultInstanceMatchesFlagDefaults) {
  const CodegenFlagsProto& defaults = CodegenFlagsProto::default_instance();

  EXPECT_EQ(defaults.generator(), GENERATOR_KIND_PIPELINE);
  EXPECT_TRUE(defaults.flop_inputs());
  EXPECT_TRUE(defaults.flop_outputs());
  EXPECT_EQ(defaults.flop_inputs_kind(), IO_KIND_FLOP);
  EXPECT_EQ(defaults.flop_outputs_kind(), IO_KIND_FLOP);
  EXPECT_TRUE(defaults.flop_single_value_channels());
  EXPECT_EQ(defaults.output_port_name(), "out");
  EXPECT_TRUE(defaults.reset_data_path());
  EXPECT_TRUE(defaults.use_system_verilog());
  EXPECT_EQ(defaults.max_inline_depth(), 5);
  EXPECT_EQ(defaults.streaming_channel_valid_suffix(), "_vld");
  EXPECT_EQ(defaults.streaming_channel_ready_suffix(), "_rdy");
  EXPECT_TRUE(defaults.gate_recvs());
  EXPECT_TRUE(defaults.array_index_bounds_checking());
  EXPECT_EQ(defaults.register_merge_strategy(), STRATEGY_IDENTITY_ONLY);
  EXPECT_TRUE(defaults.emit_sv_types());
  EXPECT_EQ(defaults.simulation_macro_name(), "SIMULATION");
  EXPECT_EQ(defaults.fifo_module(), "xls_fifo_wrapper");
  EXPECT_TRUE(defaults.add_invariant_assertions());
  EXPECT_EQ(defaults.source_annotation_strategy(), ANNOTATION_STRATEGY_NONE);
  EXPECT_TRUE(defaults.preserve_ports());

  XLS_ASSERT_OK_AND_ASSIGN(CodegenFlagsProto from_flags, GetCodegenFlags());
  EXPECT_EQ(from_flags.generator(), defaults.generator());
  EXPECT_EQ(from_flags.flop_inputs(), defaults.flop_inputs());
  EXPECT_EQ(from_flags.flop_outputs(), defaults.flop_outputs());
  EXPECT_EQ(from_flags.flop_inputs_kind(), defaults.flop_inputs_kind());
  EXPECT_EQ(from_flags.flop_outputs_kind(), defaults.flop_outputs_kind());
  EXPECT_EQ(from_flags.flop_single_value_channels(),
            defaults.flop_single_value_channels());
  EXPECT_EQ(from_flags.output_port_name(), defaults.output_port_name());
  EXPECT_EQ(from_flags.reset_data_path(), defaults.reset_data_path());
  EXPECT_EQ(from_flags.use_system_verilog(), defaults.use_system_verilog());
  EXPECT_EQ(from_flags.max_inline_depth(), defaults.max_inline_depth());
  EXPECT_EQ(from_flags.streaming_channel_valid_suffix(),
            defaults.streaming_channel_valid_suffix());
  EXPECT_EQ(from_flags.streaming_channel_ready_suffix(),
            defaults.streaming_channel_ready_suffix());
  EXPECT_EQ(from_flags.gate_recvs(), defaults.gate_recvs());
  EXPECT_EQ(from_flags.array_index_bounds_checking(),
            defaults.array_index_bounds_checking());
  EXPECT_EQ(from_flags.register_merge_strategy(),
            defaults.register_merge_strategy());
  EXPECT_EQ(from_flags.emit_sv_types(), defaults.emit_sv_types());
  EXPECT_EQ(from_flags.simulation_macro_name(),
            defaults.simulation_macro_name());
  EXPECT_EQ(from_flags.fifo_module(), defaults.fifo_module());
  EXPECT_EQ(from_flags.add_invariant_assertions(),
            defaults.add_invariant_assertions());
  EXPECT_EQ(from_flags.source_annotation_strategy(),
            defaults.source_annotation_strategy());
  EXPECT_EQ(from_flags.preserve_ports(), defaults.preserve_ports());
  EXPECT_TRUE(from_flags.has_assertion_macros());
  EXPECT_THAT(from_flags.assertion_macros().macro_names(),
              ElementsAre("ASSERT_ON"));
}

TEST(CodegenFlagsTest, MaterializeInternalFifosSetsEmptyFifoModule) {
  absl::FlagSaver flag_saver;
  ParseFlagsForTest({"--materialize_internal_fifos"});

  XLS_ASSERT_OK_AND_ASSIGN(CodegenFlagsProto proto, GetCodegenFlags());
  EXPECT_TRUE(proto.has_fifo_module());
  EXPECT_THAT(proto.fifo_module(), IsEmpty());
}

TEST(CodegenFlagsTest, AssertionMacroNamesEmptyFlagSetsEmptyConfig) {
  absl::FlagSaver flag_saver;
  ParseFlagsForTest({"--assertion_macro_names="});

  XLS_ASSERT_OK_AND_ASSIGN(CodegenFlagsProto proto, GetCodegenFlags());
  EXPECT_TRUE(proto.has_assertion_macros());
  EXPECT_THAT(proto.assertion_macros().macro_names(), IsEmpty());
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
