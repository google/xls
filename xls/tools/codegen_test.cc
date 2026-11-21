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

#include "xls/tools/codegen.h"

#include <memory>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "xls/codegen/codegen_options.h"
#include "xls/common/status/matchers.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/function.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/package.h"
#include "xls/tools/codegen_flags.pb.h"
#include "xls/tools/scheduling_options_flags.pb.h"

namespace xls {
namespace {

using ::absl_testing::StatusIs;
using ::testing::AllOf;
using ::testing::ElementsAre;
using ::testing::HasSubstr;
using ::testing::IsEmpty;

class CodegenTest : public IrTestBase {
 protected:
  absl::StatusOr<std::unique_ptr<VerifiedPackage>> CreateSimplePackage() {
    std::unique_ptr<VerifiedPackage> p = CreatePackage();
    FunctionBuilder fb("simple", p.get());
    fb.Add(fb.Param("x", p->GetBitsType(8)), fb.Param("y", p->GetBitsType(8)));
    XLS_ASSIGN_OR_RETURN(Function * f, fb.Build());
    XLS_RETURN_IF_ERROR(p->SetTop(f));
    return p;
  }
};

TEST_F(CodegenTest, PipelineFailsEarlyWithoutPipelineStagesOrClockPeriod) {
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedPackage> p,
                           CreateSimplePackage());
  CodegenFlagsProto codegen_flags;
  codegen_flags.set_generator(GENERATOR_KIND_PIPELINE);
  // Leave delay_model unset too so this verifies timing options are validated
  // before delay model setup.
  SchedulingOptionsFlagsProto scheduling_flags;

  EXPECT_THAT(ScheduleAndCodegen(p.get(), scheduling_flags, codegen_flags,
                                 /*with_delay_model=*/true),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       AllOf(HasSubstr("--pipeline_stages"),
                             HasSubstr("--clock_period_ps"))));
}

TEST_F(CodegenTest, PipelineRejectsNegativeTimingOptions) {
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedPackage> p,
                           CreateSimplePackage());
  CodegenFlagsProto codegen_flags;
  codegen_flags.set_generator(GENERATOR_KIND_PIPELINE);

  SchedulingOptionsFlagsProto bad_stages;
  bad_stages.set_delay_model("unit");
  bad_stages.set_pipeline_stages(-1);
  EXPECT_THAT(ScheduleAndCodegen(p.get(), bad_stages, codegen_flags,
                                 /*with_delay_model=*/true),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("--pipeline_stages must be positive")));

  SchedulingOptionsFlagsProto bad_clock;
  bad_clock.set_delay_model("unit");
  bad_clock.set_clock_period_ps(-100);
  EXPECT_THAT(ScheduleAndCodegen(p.get(), bad_clock, codegen_flags,
                                 /*with_delay_model=*/true),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("--clock_period_ps must be positive")));
}

TEST_F(CodegenTest, CombinationalDoesNotRequireTimingOptions) {
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedPackage> p,
                           CreateSimplePackage());
  CodegenFlagsProto codegen_flags;
  codegen_flags.set_generator(GENERATOR_KIND_COMBINATIONAL);
  SchedulingOptionsFlagsProto scheduling_flags;

  XLS_EXPECT_OK(ScheduleAndCodegen(p.get(), scheduling_flags, codegen_flags,
                                   /*with_delay_model=*/false));
}

TEST_F(CodegenTest, CodegenOptionsFromProtoPropagatesMaxTraceVerbosity) {
  CodegenFlagsProto proto;
  proto.set_max_trace_verbosity(3);

  XLS_ASSERT_OK_AND_ASSIGN(verilog::CodegenOptions options,
                           CodegenOptionsFromProto(proto));
  EXPECT_EQ(options.max_trace_verbosity(), 3);
}

TEST_F(CodegenTest, CodegenOptionsFromProtoPropagatesPreservePorts) {
  CodegenFlagsProto default_proto;
  XLS_ASSERT_OK_AND_ASSIGN(verilog::CodegenOptions default_options,
                           CodegenOptionsFromProto(default_proto));
  EXPECT_TRUE(default_options.preserve_ports());

  CodegenFlagsProto disabled_proto;
  disabled_proto.set_preserve_ports(false);
  XLS_ASSERT_OK_AND_ASSIGN(verilog::CodegenOptions disabled_options,
                           CodegenOptionsFromProto(disabled_proto));
  EXPECT_FALSE(disabled_options.preserve_ports());
}

TEST_F(CodegenTest, CodegenOptionsFromProtoAddInvariantAssertions) {
  for (GeneratorKind kind :
       {GENERATOR_KIND_COMBINATIONAL, GENERATOR_KIND_PIPELINE}) {
    CodegenFlagsProto default_proto;
    default_proto.set_generator(kind);
    XLS_ASSERT_OK_AND_ASSIGN(verilog::CodegenOptions default_options,
                             CodegenOptionsFromProto(default_proto));
    EXPECT_TRUE(default_options.add_invariant_assertions());

    CodegenFlagsProto disabled_proto;
    disabled_proto.set_generator(kind);
    disabled_proto.set_add_invariant_assertions(false);
    XLS_ASSERT_OK_AND_ASSIGN(verilog::CodegenOptions disabled_options,
                             CodegenOptionsFromProto(disabled_proto));
    EXPECT_FALSE(disabled_options.add_invariant_assertions());
  }
}

TEST_F(CodegenTest, EmptyProtoMatchesDefaultCodegenOptions) {
  CodegenFlagsProto empty_proto;
  XLS_ASSERT_OK_AND_ASSIGN(verilog::CodegenOptions from_empty,
                           CodegenOptionsFromProto(empty_proto));
  verilog::CodegenOptions cpp_defaults;

  EXPECT_TRUE(from_empty.emit_as_pipeline());
  EXPECT_TRUE(from_empty.flop_inputs());
  EXPECT_TRUE(from_empty.flop_outputs());
  EXPECT_EQ(from_empty.flop_inputs_kind(), cpp_defaults.flop_inputs_kind());
  EXPECT_EQ(from_empty.flop_outputs_kind(), cpp_defaults.flop_outputs_kind());
  EXPECT_TRUE(from_empty.flop_single_value_channels());
  EXPECT_EQ(from_empty.output_port_name(), cpp_defaults.output_port_name());
  EXPECT_EQ(from_empty.use_system_verilog(), cpp_defaults.use_system_verilog());
  EXPECT_TRUE(from_empty.use_system_verilog());
  EXPECT_EQ(from_empty.separate_lines(), cpp_defaults.separate_lines());
  EXPECT_EQ(from_empty.max_inline_depth(), cpp_defaults.max_inline_depth());
  EXPECT_EQ(from_empty.max_inline_depth(), 5);
  EXPECT_EQ(from_empty.streaming_channel_valid_suffix(),
            cpp_defaults.streaming_channel_valid_suffix());
  EXPECT_EQ(from_empty.streaming_channel_ready_suffix(),
            cpp_defaults.streaming_channel_ready_suffix());
  EXPECT_EQ(from_empty.gate_recvs(), cpp_defaults.gate_recvs());
  EXPECT_TRUE(from_empty.gate_recvs());
  EXPECT_EQ(from_empty.array_index_bounds_checking(),
            cpp_defaults.array_index_bounds_checking());
  EXPECT_TRUE(from_empty.array_index_bounds_checking());
  EXPECT_EQ(from_empty.register_merge_strategy(),
            verilog::CodegenOptions::RegisterMergeStrategy::kIdentityOnly);
  EXPECT_EQ(from_empty.emit_sv_types(), cpp_defaults.emit_sv_types());
  EXPECT_EQ(from_empty.simulation_macro_name(),
            cpp_defaults.simulation_macro_name());
  EXPECT_THAT(from_empty.assertion_macro_names(), ElementsAre("ASSERT_ON"));
  EXPECT_EQ(from_empty.fifo_module(), cpp_defaults.fifo_module());
  EXPECT_EQ(from_empty.fifo_module(), "xls_fifo_wrapper");
  EXPECT_EQ(from_empty.add_invariant_assertions(),
            cpp_defaults.add_invariant_assertions());
  EXPECT_EQ(from_empty.source_annotation_strategy(),
            cpp_defaults.source_annotation_strategy());
  EXPECT_EQ(from_empty.preserve_ports(), cpp_defaults.preserve_ports());
  EXPECT_EQ(from_empty.max_trace_verbosity(),
            cpp_defaults.max_trace_verbosity());
}

TEST_F(CodegenTest, ExplicitEmptyAssertionMacrosDisablesGuards) {
  CodegenFlagsProto proto;
  proto.mutable_assertion_macros();

  XLS_ASSERT_OK_AND_ASSIGN(verilog::CodegenOptions options,
                           CodegenOptionsFromProto(proto));
  EXPECT_THAT(options.assertion_macro_names(), IsEmpty());
}

TEST_F(CodegenTest, LegacyAssertionMacroNamesStillSupported) {
  CodegenFlagsProto proto;
  proto.add_assertion_macro_names("!SYNTHESIS");
  proto.add_assertion_macro_names("ASSERT_ON");

  XLS_ASSERT_OK_AND_ASSIGN(verilog::CodegenOptions options,
                           CodegenOptionsFromProto(proto));
  EXPECT_THAT(options.assertion_macro_names(),
              ElementsAre("!SYNTHESIS", "ASSERT_ON"));
}

}  // namespace
}  // namespace xls
