// Copyright 2022 The XLS Authors
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

#include "xls/codegen/ram_rewrite_pass.h"

#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <ostream>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/base/no_destructor.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_replace.h"
#include "absl/types/span.h"
#include "absl/types/variant.h"
#include "xls/codegen/block_conversion.h"
#include "xls/codegen/codegen_options.h"
#include "xls/codegen/codegen_pass.h"
#include "xls/codegen/codegen_pass_pipeline.h"
#include "xls/codegen/module_signature.pb.h"
#include "xls/codegen/ram_configuration.h"
#include "xls/common/casts.h"
#include "xls/common/proto_test_utils.h"
#include "xls/common/status/matchers.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/common/visitor.h"
#include "xls/estimators/delay_model/delay_estimators.h"
#include "xls/ir/block.h"
#include "xls/ir/channel.h"
#include "xls/ir/ir_matcher.h"
#include "xls/ir/ir_parser.h"
#include "xls/ir/nodes.h"
#include "xls/ir/package.h"
#include "xls/ir/proc.h"
#include "xls/passes/pass_base.h"
#include "xls/scheduling/pipeline_schedule.h"
#include "xls/scheduling/run_pipeline_schedule.h"
#include "xls/scheduling/scheduling_options.h"

namespace xls {
namespace verilog {
namespace {

namespace m = xls::op_matchers;

using proto_testing::EqualsProto;
using status_testing::StatusIs;
using testing::AllOf;
using testing::AnyOf;
using testing::Contains;
using testing::Eq;
using testing::HasSubstr;
using ::testing::IsSupersetOf;
using testing::Not;

class PortByNameMatcher : public ::testing::MatcherInterface<Block::Port> {
 public:
  explicit PortByNameMatcher(std::string_view port_name)
      : port_name_(port_name) {}
  PortByNameMatcher(const PortByNameMatcher&) = default;

  bool MatchAndExplain(
      const Block::Port port,
      ::testing::MatchResultListener* listener) const override {
    return absl::visit(
        Visitor{
            [=, this](Block::ClockPort* p) {
              *listener << "ClockPort(" << p->name << ")";
              if (p->name != port_name_) {
                *listener << " does not match expected " << port_name_ << ".";
                return false;
              }
              return true;
            },
            [=, this](InputPort* p) {
              *listener << "InputPort(" << p->name() << ")";
              if (p->name() != port_name_) {
                *listener << " does not match expected " << port_name_ << ".";
                return false;
              }
              return true;
            },
            [=, this](OutputPort* p) {
              *listener << "OutputPort(" << p->name() << ")";
              if (p->name() != port_name_) {
                *listener << " does not match expected " << port_name_ << ".";
                return false;
              }
              return true;
            },
        },
        port);
  }
  void DescribeTo(::std::ostream* os) const override {
    *os << "Port(" << port_name_ << ")";
  }

 protected:
  std::string_view port_name_;
};

inline ::testing::Matcher<::xls::Block::Port> PortByName(
    std::string_view port_name) {
  return ::testing::MakeMatcher(new PortByNameMatcher(port_name));
}

class PortProtoByNameMatcher : public ::testing::MatcherInterface<PortProto> {
 public:
  explicit PortProtoByNameMatcher(std::string_view port_name)
      : port_name_(port_name) {}
  PortProtoByNameMatcher(const PortProtoByNameMatcher&) = default;

  bool MatchAndExplain(
      const PortProto port_proto,
      ::testing::MatchResultListener* listener) const override {
    *listener << "PortProto(" << port_proto.name() << ")";
    if (port_proto.name() != port_name_) {
      *listener << " does not match expected " << port_name_ << ".";
      return false;
    }
    return true;
  }

  void DescribeTo(::std::ostream* os) const override {
    *os << "PortProto(" << port_name_ << ")";
  }

 protected:
  std::string_view port_name_;
};

inline ::testing::Matcher<::xls::verilog::PortProto> PortProtoByName(
    std::string_view port_name) {
  return ::testing::MakeMatcher(new PortProtoByNameMatcher(port_name));
}

CodegenPass* DefaultCodegenPassPipeline() {
  static absl::NoDestructor<std::unique_ptr<CodegenCompoundPass>> singleton(
      CreateCodegenPassPipeline());
  return singleton->get();
}

CodegenPass* RamRewritePassOnly() {
  static absl::NoDestructor<RamRewritePass> singleton;
  return singleton.get();
}

std::string_view CodegenPassName(CodegenPass const* pass) {
  if (pass == DefaultCodegenPassPipeline()) {
    return "DefaultCodegenPassPipeline";
  }
  if (pass == RamRewritePassOnly()) {
    return "RamRewritePassOnly";
  }
  // We're seeing an unknown codegen pass, so error
  LOG(FATAL) << "Unknown codegen pass!";
  return "";
}

struct RamChannelRewriteTestParam {
  std::string_view test_name;
  // IR must contain a proc named "my_proc"
  std::string_view ir_text;
  int64_t pipeline_stages;
  absl::Span<const std::string_view> ram_config_strings;
  bool expect_read_mask;
  bool expect_write_mask;
  // Type of data held by each ram in the same order as ram_config_strings
  absl::Span<const std::string_view> ram_contents;
};

class RamRewritePassTest
    : public testing::TestWithParam<
          std::tuple<RamChannelRewriteTestParam, CodegenPass*>> {
 protected:
  CodegenOptions GetCodegenOptions() const {
    auto& param = std::get<0>(GetParam());
    CodegenOptions codegen_options;
    codegen_options.flop_inputs(false)
        .flop_outputs(false)
        .clock_name("clk")
        .reset("rst", false, false, false)
        .streaming_channel_data_suffix("_data")
        .streaming_channel_valid_suffix("_valid")
        .streaming_channel_ready_suffix("_ready")
        .module_name("pipelined_proc");
    std::vector<std::unique_ptr<RamConfiguration>> ram_configurations;
    ram_configurations.reserve(param.ram_config_strings.size());
    for (std::string_view config_string : param.ram_config_strings) {
      std::unique_ptr<RamConfiguration> config =
          RamConfiguration::ParseString(config_string).value();
      ram_configurations.push_back(std::move(config));
    }
    codegen_options.ram_configurations(ram_configurations);
    return codegen_options;
  }

  bool ExpectReadMask() const {
    auto& param = std::get<0>(GetParam());
    auto codegen_pass = std::get<1>(GetParam());
    // If we're only running the ram rewrite pass (which is not compound), even
    // masks with type () will not be removed. They should always exist.
    return param.expect_read_mask || !codegen_pass->IsCompound();
  }

  bool ExpectWriteMask() const {
    auto& param = std::get<0>(GetParam());
    auto codegen_pass = std::get<1>(GetParam());
    // If we're only running the ram rewrite pass (which is not compound), even
    // masks with type () will not be removed. They should always exist.
    return param.expect_write_mask || !codegen_pass->IsCompound();
  }

  absl::StatusOr<CodegenPassUnit> ScheduleAndBlockConvert(
      Package const* package, const CodegenOptions& codegen_options) {
    auto& param = std::get<0>(GetParam());

    XLS_ASSIGN_OR_RETURN(Proc * proc, package->GetProc("my_proc"));

    auto scheduling_options =
        SchedulingOptions().pipeline_stages(param.pipeline_stages);
    // Add constraints for each ram config to be scheduled according to the
    // config's latency
    for (const auto& ram_config : codegen_options.ram_configurations()) {
      if (ram_config->ram_kind() == "1RW") {
        auto* ram1rw_config = down_cast<Ram1RWConfiguration*>(ram_config.get());
        scheduling_options.add_constraint(IOConstraint(
            ram1rw_config->rw_port_configuration().request_channel_name,
            IODirection::kSend,
            ram1rw_config->rw_port_configuration().response_channel_name,
            IODirection::kReceive,
            /*minimum_latency=*/ram1rw_config->latency(),
            /*maximum_latency=*/ram1rw_config->latency()));
        scheduling_options.add_constraint(IOConstraint(
            ram1rw_config->rw_port_configuration().request_channel_name,
            IODirection::kSend,
            ram1rw_config->rw_port_configuration()
                .write_completion_channel_name,
            IODirection::kReceive,
            /*minimum_latency=*/ram1rw_config->latency(),
            /*maximum_latency=*/ram1rw_config->latency()));
        continue;
      }
      if (ram_config->ram_kind() == "1R1W") {
        auto* ram1r1w_config =
            down_cast<Ram1R1WConfiguration*>(ram_config.get());
        scheduling_options.add_constraint(IOConstraint(
            ram1r1w_config->r_port_configuration().request_channel_name,
            IODirection::kSend,
            ram1r1w_config->r_port_configuration().response_channel_name,
            IODirection::kReceive,
            /*minimum_latency=*/ram1r1w_config->latency(),
            /*maximum_latency=*/ram1r1w_config->latency()));
        scheduling_options.add_constraint(IOConstraint(
            ram1r1w_config->w_port_configuration().request_channel_name,
            IODirection::kSend,
            ram1r1w_config->w_port_configuration()
                .write_completion_channel_name,
            IODirection::kReceive,
            /*minimum_latency=*/ram1r1w_config->latency(),
            /*maximum_latency=*/ram1r1w_config->latency()));
        continue;
      }
    }

    XLS_ASSIGN_OR_RETURN(auto delay_estimator, GetDelayEstimator("unit"));
    XLS_ASSIGN_OR_RETURN(
        PipelineSchedule schedule,
        RunPipelineSchedule(proc, *delay_estimator, scheduling_options));

    return FunctionBaseToPipelinedBlock(schedule, codegen_options, proc);
  }

  std::string_view TypeOfRam(std::string_view ram_name) const {
    auto& param = std::get<0>(GetParam());
    for (int64_t i = 0; i < param.ram_config_strings.size(); ++i) {
      if (param.ram_config_strings[i].starts_with(ram_name)) {
        return param.ram_contents[i];
      }
    }
    ADD_FAILURE() << "Unable to find configuration for " << ram_name;
    return "";
  }
};

TEST_P(RamRewritePassTest, PortsUpdated) {
  auto& param = std::get<0>(GetParam());
  CodegenOptions codegen_options = GetCodegenOptions();
  CodegenPassOptions pass_options{
      .codegen_options = codegen_options,
  };

  XLS_ASSERT_OK_AND_ASSIGN(auto package, Parser::ParsePackage(param.ir_text));
  XLS_ASSERT_OK_AND_ASSIGN(
      CodegenPassUnit unit,
      ScheduleAndBlockConvert(package.get(), codegen_options));
  auto pipeline = std::get<1>(GetParam());
  CodegenPassResults results;
  XLS_ASSERT_OK_AND_ASSIGN(bool changed,
                           pipeline->Run(&unit, pass_options, &results));

  EXPECT_TRUE(changed);

  for (const auto& config : codegen_options.ram_configurations()) {
    std::vector<std::string_view> old_channel_names;
    EXPECT_THAT(config->ram_kind(), AnyOf(Eq("1RW"), Eq("1R1W")));
    if (config->ram_kind() == "1RW") {
      auto* ram1rw_config = down_cast<Ram1RWConfiguration*>(config.get());
      EXPECT_THAT(
          unit.top_block->GetPorts(),
          AllOf(Contains(
                    PortByName(absl::StrFormat("%s_addr", config->ram_name()))),
                Contains(PortByName(
                    absl::StrFormat("%s_rd_data", config->ram_name()))),
                Contains(
                    PortByName(absl::StrFormat("%s_re", config->ram_name()))),
                Contains(PortByName(
                    absl::StrFormat("%s_wr_data", config->ram_name()))),
                Contains(
                    PortByName(absl::StrFormat("%s_we", config->ram_name())))));
      XLS_ASSERT_OK_AND_ASSIGN(InputPort * rd_input,
                               unit.top_block->GetInputPort(absl::StrFormat(
                                   "%s_rd_data", config->ram_name())));
      EXPECT_THAT(rd_input->GetType(), m::Type(TypeOfRam(config->ram_name())));
      XLS_ASSERT_OK_AND_ASSIGN(OutputPort * wr_output,
                               unit.top_block->GetOutputPort(absl::StrFormat(
                                   "%s_wr_data", config->ram_name())));
      EXPECT_THAT(wr_output->operand(0)->GetType(),
                  m::Type(TypeOfRam(config->ram_name())));
      if (ExpectReadMask()) {
        EXPECT_THAT(unit.top_block->GetPorts(),
                    Contains(PortByName(
                        absl::StrFormat("%s_rd_mask", config->ram_name()))));
      } else {
        EXPECT_THAT(unit.top_block->GetPorts(),
                    Not(Contains(PortByName(
                        absl::StrFormat("%s_rd_mask", config->ram_name())))));
      }
      if (ExpectWriteMask()) {
        EXPECT_THAT(unit.top_block->GetPorts(),
                    Contains(PortByName(
                        absl::StrFormat("%s_wr_mask", config->ram_name()))));
      } else {
        EXPECT_THAT(unit.top_block->GetPorts(),
                    Not(Contains(PortByName(
                        absl::StrFormat("%s_wr_mask", config->ram_name())))));
      }

      old_channel_names.push_back(
          ram1rw_config->rw_port_configuration().request_channel_name);
      old_channel_names.push_back(
          ram1rw_config->rw_port_configuration().response_channel_name);
    } else if (config->ram_kind() == "1R1W") {
      auto* ram1r1w_config = down_cast<Ram1R1WConfiguration*>(config.get());
      EXPECT_THAT(unit.top_block->GetPorts(),
                  AllOf(Contains(PortByName(
                            absl::StrFormat("%s_rd_en", config->ram_name()))),
                        Contains(PortByName(
                            absl::StrFormat("%s_rd_addr", config->ram_name()))),
                        Contains(PortByName(
                            absl::StrFormat("%s_rd_data", config->ram_name()))),
                        Contains(PortByName(
                            absl::StrFormat("%s_wr_en", config->ram_name()))),
                        Contains(PortByName(
                            absl::StrFormat("%s_wr_addr", config->ram_name()))),
                        Contains(PortByName(absl::StrFormat(
                            "%s_wr_data", config->ram_name())))));
      if (ExpectReadMask()) {
        EXPECT_THAT(unit.top_block->GetPorts(),
                    Contains(PortByName(
                        absl::StrFormat("%s_rd_mask", config->ram_name()))));
      } else {
        EXPECT_THAT(unit.top_block->GetPorts(),
                    Not(Contains(PortByName(
                        absl::StrFormat("%s_rd_mask", config->ram_name())))));
      }
      if (ExpectWriteMask()) {
        EXPECT_THAT(unit.top_block->GetPorts(),
                    Contains(PortByName(
                        absl::StrFormat("%s_wr_mask", config->ram_name()))));
      } else {
        EXPECT_THAT(unit.top_block->GetPorts(),
                    Not(Contains(PortByName(
                        absl::StrFormat("%s_wr_mask", config->ram_name())))));
      }
      old_channel_names.push_back(
          ram1r1w_config->r_port_configuration().request_channel_name);
      old_channel_names.push_back(
          ram1r1w_config->r_port_configuration().response_channel_name);
      old_channel_names.push_back(
          ram1r1w_config->w_port_configuration().request_channel_name);
    }
    for (auto old_channel_name : old_channel_names) {
      EXPECT_THAT(
          unit.top_block->GetPorts(),
          Not(AnyOf(
              Contains(PortByName(absl::StrCat(old_channel_name, "_valid"))),
              Contains(PortByName(absl::StrCat(old_channel_name, "_data"))),
              Contains(PortByName(absl::StrCat(old_channel_name, "_ready"))))));
    }
    EXPECT_THAT(unit.top_block->GetPorts(),
                Not(AnyOf(Contains(PortByName(
                              absl::StrFormat("%s_valid", config->ram_name()))),
                          Contains(PortByName(
                              absl::StrFormat("%s_data", config->ram_name()))),
                          Contains(PortByName(absl::StrFormat(
                              "%s_ready", config->ram_name()))))));
  }
}

TEST_P(RamRewritePassTest, ModuleSignatureUpdated) {
  // Module signature is generated by other codegen passes, only run this test
  // if we're running the full pass pipeline.
  if (std::get<1>(GetParam()) != DefaultCodegenPassPipeline()) {
    GTEST_SKIP();
  }

  auto& param = std::get<0>(GetParam());
  CodegenOptions codegen_options = GetCodegenOptions();
  CodegenPassOptions pass_options{
      .codegen_options = codegen_options,
  };

  XLS_ASSERT_OK_AND_ASSIGN(auto package, Parser::ParsePackage(param.ir_text));
  XLS_ASSERT_OK_AND_ASSIGN(
      CodegenPassUnit unit,
      ScheduleAndBlockConvert(package.get(), codegen_options));
  auto pipeline = std::get<1>(GetParam());
  CodegenPassResults results;
  XLS_ASSERT_OK_AND_ASSIGN(bool changed,
                           pipeline->Run(&unit, pass_options, &results));

  EXPECT_TRUE(changed);

  ASSERT_TRUE(unit.metadata.contains(unit.top_block) &&
              unit.metadata.at(unit.top_block).signature.has_value());
  for (const auto& config : codegen_options.ram_configurations()) {
    absl::flat_hash_set<std::string_view> channel_names;
    EXPECT_THAT(config->ram_kind(), AnyOf(Eq("1RW"), Eq("1R1W")));
    if (config->ram_kind() == "1RW") {
      auto* ram1rw_config = down_cast<Ram1RWConfiguration*>(config.get());
      bool found = false;
      for (auto& ram : unit.metadata.at(unit.top_block).signature->rams()) {
        if (ram.ram_oneof_case() != RamProto::RamOneofCase::kRam1Rw) {
          continue;
        }
        if (ram.ram_1rw().rw_port().request().name() ==
                ram1rw_config->rw_port_configuration().request_channel_name &&
            ram.ram_1rw().rw_port().response().name() ==
                ram1rw_config->rw_port_configuration().response_channel_name) {
          found = true;
        }
        // Check the port information matches.
        EXPECT_THAT(
            unit.metadata.at(unit.top_block).signature->proto().data_ports(),
            IsSupersetOf({
                EqualsProto(ram.ram_1rw().rw_port().response().read_data()),
                EqualsProto(ram.ram_1rw().rw_port().request().address()),
                EqualsProto(ram.ram_1rw().rw_port().request().read_enable()),
                EqualsProto(ram.ram_1rw().rw_port().request().write_data()),
                EqualsProto(ram.ram_1rw().rw_port().request().write_enable()),
            }))
            << "missing 1rw ports";
      }
      EXPECT_TRUE(found);
      channel_names.insert(
          ram1rw_config->rw_port_configuration().request_channel_name);
      channel_names.insert(
          ram1rw_config->rw_port_configuration().response_channel_name);
      channel_names.insert(
          ram1rw_config->rw_port_configuration().write_completion_channel_name);
      XLS_ASSERT_OK_AND_ASSIGN(
          Channel * req_channel,
          package->GetChannel(
              ram1rw_config->rw_port_configuration().request_channel_name));
      // req is (addr, wr_data, wr_mask, rd_mask, we, re), so wr_mask_idx=2 and
      // rd_mask_idx=3.
      // If a mask is an empty tuple, expect to see it 0 times (i.e. be absent)
      // in the signature, otherwise expect it present once.
      int write_mask_times =
          req_channel->type()->AsTupleOrDie()->element_type(2)->IsEqualTo(
              package->GetTupleType({}))
              ? 0
              : 1;
      int read_mask_times =
          req_channel->type()->AsTupleOrDie()->element_type(3)->IsEqualTo(
              package->GetTupleType({}))
              ? 0
              : 1;
      EXPECT_THAT(unit.metadata.at(unit.top_block).signature->data_outputs(),
                  AllOf(Contains(PortProtoByName(
                            absl::StrCat(ram1rw_config->ram_name(), "_addr"))),
                        Contains(PortProtoByName(absl::StrCat(
                            ram1rw_config->ram_name(), "_wr_data"))),
                        Contains(PortProtoByName(
                            absl::StrCat(ram1rw_config->ram_name(), "_we"))),
                        Contains(PortProtoByName(
                            absl::StrCat(ram1rw_config->ram_name(), "_re"))),
                        Contains(PortProtoByName(absl::StrCat(
                                     ram1rw_config->ram_name(), "_wr_mask")))
                            .Times(write_mask_times),
                        Contains(PortProtoByName(absl::StrCat(
                                     ram1rw_config->ram_name(), "_rd_mask")))
                            .Times(read_mask_times)));
      EXPECT_THAT(unit.metadata.at(unit.top_block).signature->data_inputs(),
                  Contains(PortProtoByName(
                      absl::StrCat(ram1rw_config->ram_name(), "_rd_data"))));
    } else if (config->ram_kind() == "1R1W") {
      auto* ram1r1w_config = down_cast<Ram1R1WConfiguration*>(config.get());
      bool found = false;
      for (auto& ram : unit.metadata.at(unit.top_block).signature->rams()) {
        if (ram.ram_oneof_case() != RamProto::RamOneofCase::kRam1R1W) {
          continue;
        }
        if (ram.ram_1r1w().r_port().request().name() ==
                ram1r1w_config->r_port_configuration().request_channel_name &&
            ram.ram_1r1w().r_port().response().name() ==
                ram1r1w_config->r_port_configuration().response_channel_name) {
          found = true;
        }
        // Check the port information matches.
        EXPECT_THAT(
            unit.metadata.at(unit.top_block).signature->proto().data_ports(),
            IsSupersetOf({
                EqualsProto(ram.ram_1r1w().r_port().response().data()),
                EqualsProto(ram.ram_1r1w().r_port().request().address()),
                EqualsProto(ram.ram_1r1w().r_port().request().enable()),
                EqualsProto(ram.ram_1r1w().w_port().request().data()),
                EqualsProto(ram.ram_1r1w().w_port().request().address()),
                EqualsProto(ram.ram_1r1w().w_port().request().enable()),
            }))
            << "missing 1r1w ports";
      }
      EXPECT_TRUE(found);
      channel_names.insert(
          ram1r1w_config->r_port_configuration().request_channel_name);
      channel_names.insert(
          ram1r1w_config->r_port_configuration().response_channel_name);
      channel_names.insert(
          ram1r1w_config->w_port_configuration().request_channel_name);
      channel_names.insert(
          ram1r1w_config->w_port_configuration().write_completion_channel_name);
      // If a mask is an empty tuple, expect to see it 0 times (i.e. be absent)
      // in the signature, otherwise expect it present once.
      XLS_ASSERT_OK_AND_ASSIGN(
          Channel * r_channel,
          package->GetChannel(
              ram1r1w_config->r_port_configuration().request_channel_name));
      // (rd_addr, rd_mask) -> mask_idx = 1
      // Empty tuple means no read mask.
      int read_mask_times =
          r_channel->type()->AsTupleOrDie()->element_type(1)->IsEqualTo(
              package->GetTupleType({}))
              ? 0
              : 1;
      XLS_ASSERT_OK_AND_ASSIGN(
          Channel * w_channel,
          package->GetChannel(
              ram1r1w_config->w_port_configuration().request_channel_name));
      // (wr_addr, wr_data, wr_mask) -> mask_idx = 2
      // Empty tuple means no write mask.
      int write_mask_times =
          w_channel->type()->AsTupleOrDie()->element_type(2)->IsEqualTo(
              package->GetTupleType({}))
              ? 0
              : 1;
      EXPECT_THAT(unit.metadata.at(unit.top_block).signature->data_outputs(),
                  AllOf(Contains(PortProtoByName(absl::StrCat(
                            ram1r1w_config->ram_name(), "_rd_addr"))),
                        Contains(PortProtoByName(absl::StrCat(
                            ram1r1w_config->ram_name(), "_wr_addr"))),
                        Contains(PortProtoByName(absl::StrCat(
                            ram1r1w_config->ram_name(), "_wr_data"))),
                        Contains(PortProtoByName(absl::StrCat(
                            ram1r1w_config->ram_name(), "_wr_en"))),
                        Contains(PortProtoByName(absl::StrCat(
                            ram1r1w_config->ram_name(), "_rd_en"))),
                        Contains(PortProtoByName(absl::StrCat(
                                     ram1r1w_config->ram_name(), "_rd_mask")))
                            .Times(read_mask_times),
                        Contains(PortProtoByName(absl::StrCat(
                                     ram1r1w_config->ram_name(), "_wr_mask")))
                            .Times(write_mask_times)));
      EXPECT_THAT(unit.metadata.at(unit.top_block).signature->data_inputs(),
                  Contains(PortProtoByName(
                      absl::StrCat(ram1r1w_config->ram_name(), "_rd_data"))));
    }
    for (auto& channel :
         unit.metadata.at(unit.top_block).signature->streaming_channels()) {
      EXPECT_THAT(channel_names, Not(Contains(Eq(channel.name()))));
    }
    for (auto& channel_name : channel_names) {
      EXPECT_THAT(unit.metadata.at(unit.top_block).signature->data_inputs(),
                  Not(Contains(AnyOf(
                      PortProtoByName(channel_name),
                      PortProtoByName(absl::StrCat(channel_name, "_valid")),
                      PortProtoByName(absl::StrCat(channel_name, "_ready"))))));
      EXPECT_THAT(unit.metadata.at(unit.top_block).signature->data_outputs(),
                  Not(Contains(AnyOf(
                      PortProtoByName(channel_name),
                      PortProtoByName(absl::StrCat(channel_name, "_valid")),
                      PortProtoByName(absl::StrCat(channel_name, "_ready"))))));
    }
  }
}

TEST_P(RamRewritePassTest, WriteCompletionRemoved) {
  auto& param = std::get<0>(GetParam());
  CodegenOptions codegen_options = GetCodegenOptions();
  CodegenPassOptions pass_options{
      .codegen_options = codegen_options,
  };

  XLS_ASSERT_OK_AND_ASSIGN(auto package, Parser::ParsePackage(param.ir_text));
  XLS_ASSERT_OK_AND_ASSIGN(
      CodegenPassUnit unit,
      ScheduleAndBlockConvert(package.get(), codegen_options));
  auto pipeline = std::get<1>(GetParam());
  CodegenPassResults results;
  XLS_ASSERT_OK_AND_ASSIGN(bool changed,
                           pipeline->Run(&unit, pass_options, &results));

  EXPECT_TRUE(changed);

  for (const auto& config : codegen_options.ram_configurations()) {
    if (config->ram_kind() == "1RW") {
      auto* ram1rw_config = down_cast<Ram1RWConfiguration*>(config.get());
      std::string_view wr_comp_name =
          ram1rw_config->rw_port_configuration().write_completion_channel_name;
      EXPECT_THAT(unit.top_block->GetPorts(),
                  Not(Contains(AnyOf(
                      PortByName(absl::StrCat(wr_comp_name, "_ready")),
                      PortByName(wr_comp_name),
                      PortByName(absl::StrCat(wr_comp_name, "_valid"))))));
    } else if (config->ram_kind() == "1R1W") {
      auto* ram1r1w_config = down_cast<Ram1R1WConfiguration*>(config.get());
      std::string_view wr_comp_name =
          ram1r1w_config->w_port_configuration().write_completion_channel_name;
      EXPECT_THAT(unit.top_block->GetPorts(),
                  Not(Contains(AnyOf(
                      PortByName(absl::StrCat(wr_comp_name, "_ready")),
                      PortByName(wr_comp_name),
                      PortByName(absl::StrCat(wr_comp_name, "_valid"))))));
    }
  }
}

// Tests implicitly rely on rams being named ram0, ram1, and so on.
constexpr std::string_view kSingle1RW[] = {"ram0:1RW:req:resp:wr_comp"};

constexpr std::string_view kThree1RW[] = {
    std::string_view("ram0:1RW:req0:resp0:wr_comp0"),
    std::string_view("ram1:1RW:req1:resp1:wr_comp1"),
    std::string_view("ram2:1RW:req2:resp2:wr_comp2"),
};

constexpr std::string_view kSingle1R1W[] = {
    "ram0:1R1W:rd_req:rd_resp:wr_req:wr_comp"};

constexpr std::string_view kThree1R1W[] = {
    std::string_view("ram0:1R1W:rd_req0:rd_resp0:wr_req0:wr_comp0"),
    std::string_view("ram1:1R1W:rd_req1:rd_resp1:wr_req1:wr_comp1"),
    std::string_view("ram2:1R1W:rd_req2:rd_resp2:wr_req2:wr_comp2"),
};

constexpr std::string_view k1RWAnd1R1W[] = {
    std::string_view("ram0:1RW:req0:resp0:wr_comp0"),
    std::string_view("ram1:1R1W:rd_req1:rd_resp1:wr_req1:wr_comp1"),
};

constexpr std::array<std::string_view, 1> k1Ram32Bit{"bits[32]"};
constexpr std::array<std::string_view, 2> k2Ram32Bit{"bits[32]", "bits[32]"};
constexpr std::array<std::string_view, 3> k3Ram32Bit{"bits[32]", "bits[32]",
                                                     "bits[32]"};

constexpr RamChannelRewriteTestParam kTestParameters[] = {
    RamChannelRewriteTestParam{
        .test_name = "Simple32Bit1RW",
        .ir_text = R"(package  test
chan req((bits[32], bits[32], (), (), bits[1], bits[1]), id=0, kind=streaming, ops=send_only, flow_control=ready_valid, metadata="""""")
chan resp((bits[32]), id=1, kind=streaming, ops=receive_only, flow_control=ready_valid, metadata="""""")
chan wr_comp((), id=2, kind=streaming, ops=receive_only, flow_control=ready_valid, metadata="""""")

proc my_proc(__state: bits[32], init={0}) {
  __token: token = literal(value=token)
  empty_tuple: () = literal(value=())
  true_lit: bits[1] = literal(value=1)
  false_lit: bits[1] = literal(value=0)
  to_send: (bits[32], bits[32], (), (), bits[1], bits[1]) = tuple(__state, __state, empty_tuple, empty_tuple, true_lit, false_lit)
  send_token: token = send(__token, to_send, channel=req)
  rcv: (token, (bits[32])) = receive(send_token, channel=resp)
  rcv_token: token = tuple_index(rcv, index=0)
  wr_comp_rcv: (token, ()) = receive(rcv_token, channel=wr_comp)
  one_lit: bits[32] = literal(value=1)
  next_state: bits[32] = add(__state, one_lit)
  next_state_value: () = next_value(param=__state, value=next_state)
}
  )",
        .pipeline_stages = 2,
        .ram_config_strings = kSingle1RW,
        .ram_contents = k1Ram32Bit,
    },
    RamChannelRewriteTestParam{
        .test_name = "Simple32Bit1RWWithMask",
        .ir_text = R"(package  test
chan req((bits[32], bits[32], bits[4], bits[4], bits[1], bits[1]), id=0, kind=streaming, ops=send_only, flow_control=ready_valid, metadata="""""")
chan resp((bits[32]), id=1, kind=streaming, ops=receive_only, flow_control=ready_valid, metadata="""""")
chan wr_comp((), id=2, kind=streaming, ops=receive_only, flow_control=ready_valid, metadata="""""")

proc my_proc(__state: bits[32], init={0}) {
  __token: token = literal(value=token)
  empty_tuple: () = literal(value=())
  true_lit: bits[1] = literal(value=1)
  false_lit: bits[1] = literal(value=0)
  all_mask: bits[4] = literal(value=0xf)
  to_send: (bits[32], bits[32], bits[4], bits[4], bits[1], bits[1]) = tuple(__state, __state, all_mask, all_mask, true_lit, false_lit)
  send_token: token = send(__token, to_send, channel=req)
  rcv: (token, (bits[32])) = receive(send_token, channel=resp)
  rcv_token: token = tuple_index(rcv, index=0)
  wr_comp_rcv: (token, ()) = receive(rcv_token, channel=wr_comp)
  one_lit: bits[32] = literal(value=1)
  next_state: bits[32] = add(__state, one_lit)
  next_state_value: () = next_value(param=__state, value=next_state)
}
  )",
        .pipeline_stages = 2,
        .ram_config_strings = kSingle1RW,
        .expect_read_mask = true,
        .expect_write_mask = true,
        .ram_contents = k1Ram32Bit,
    },
    RamChannelRewriteTestParam{
        .test_name = "Simple32Bit1RWWithExtraneousChannels",
        .ir_text = R"(package  test
chan req((bits[32], bits[32], (), (), bits[1], bits[1]), id=3, kind=streaming, ops=send_only, flow_control=ready_valid, metadata="""""")
chan resp((bits[32]), id=1, kind=streaming, ops=receive_only, flow_control=ready_valid, metadata="""""")
chan wr_comp((), id=4, kind=streaming, ops=receive_only, flow_control=ready_valid, metadata="""""")
chan extra0(bits[1], id=0, kind=streaming, ops=receive_only, flow_control=ready_valid, metadata="""""")
chan extra1(bits[1], id=2, kind=streaming, ops=receive_only, flow_control=ready_valid, metadata="""""")

proc my_proc(__state: bits[32], init={0}) {
  __token: token = literal(value=token)
  true_lit: bits[1] = literal(value=1)
  false_lit: bits[1] = literal(value=0)
  empty_tuple: () = literal(value=())
  to_send: (bits[32], bits[32], (), (), bits[1], bits[1]) = tuple(__state, __state, empty_tuple, empty_tuple, true_lit, false_lit)
  send_token: token = send(__token, to_send, channel=req)
  rcv: (token, (bits[32])) = receive(send_token, channel=resp)
  rcv_token: token = tuple_index(rcv, index=0)
  one_lit: bits[32] = literal(value=1)
  extra0_rcv: (token, bits[1]) = receive(rcv_token, channel=extra0)
  extra0_token: token = tuple_index(extra0_rcv, index=0)
  extra1_rcv: (token, bits[1]) = receive(extra0_token, channel=extra1)
  extra1_token: token = tuple_index(extra1_rcv, index=0)
  wr_comp_rcv: (token, ()) = receive(extra1_token, channel=wr_comp)
  next_state: bits[32] = add(__state, one_lit)
  next_state_value: () = next_value(param=__state, value=next_state)
}
  )",
        .pipeline_stages = 4,
        .ram_config_strings = kSingle1RW,
        .ram_contents = k1Ram32Bit,
    },
    RamChannelRewriteTestParam{
        .test_name = "32BitWithThree1RWRams",
        .ir_text = R"(package  test
chan req0((bits[32], bits[32], (), (), bits[1], bits[1]), id=0, kind=streaming, ops=send_only, flow_control=ready_valid, metadata="""""")
chan resp0((bits[32]), id=1, kind=streaming, ops=receive_only, flow_control=ready_valid, metadata="""""")
chan wr_comp0((), id=6, kind=streaming, ops=receive_only, flow_control=ready_valid, metadata="""""")
chan req1((bits[32], bits[32], (), (), bits[1], bits[1]), id=2, kind=streaming, ops=send_only, flow_control=ready_valid, metadata="""""")
chan resp1((bits[32]), id=3, kind=streaming, ops=receive_only, flow_control=ready_valid, metadata="""""")
chan wr_comp1((), id=7, kind=streaming, ops=receive_only, flow_control=ready_valid, metadata="""""")
chan req2((bits[32], bits[32], (), (), bits[1], bits[1]), id=4, kind=streaming, ops=send_only, flow_control=ready_valid, metadata="""""")
chan resp2((bits[32]), id=5, kind=streaming, ops=receive_only, flow_control=ready_valid, metadata="""""")
chan wr_comp2((), id=8, kind=streaming, ops=receive_only, flow_control=ready_valid, metadata="""""")

proc my_proc(__state: (), init={()}) {
  __token: token = literal(value=token)
  true_lit: bits[1] = literal(value=1)
  false_lit: bits[1] = literal(value=0)
  empty_lit: bits[32] = literal(value=0)
  empty_tuple: () = literal(value=())
  to_send0: (bits[32], bits[32], (), (), bits[1], bits[1]) = tuple(empty_lit, empty_lit, empty_tuple, empty_tuple, false_lit, true_lit)
  send0_token: token = send(__token, to_send0, channel=req0)
  rcv0: (token, (bits[32])) = receive(send0_token, channel=resp0)
  rcv0_tuple: (bits[32]) = tuple_index(rcv0, index=1)
  rcv0_data: bits[32] = tuple_index(rcv0_tuple, index=0)
  to_send1: (bits[32], bits[32], (), (), bits[1], bits[1]) = tuple(rcv0_data, empty_lit, empty_tuple, empty_tuple, false_lit, true_lit)
  send1_token: token = send(__token, to_send1, channel=req1)
  rcv1: (token, (bits[32])) = receive(send1_token, channel=resp1)
  rcv1_tuple: (bits[32]) = tuple_index(rcv1, index=1)
  rcv1_data: bits[32] = tuple_index(rcv1_tuple, index=0)
  to_send2: (bits[32], bits[32], (), (), bits[1], bits[1]) = tuple(rcv1_data, empty_lit, empty_tuple, empty_tuple, false_lit, true_lit)
  send2_token: token = send(__token, to_send2, channel=req2)
  rcv2: (token, (bits[32])) = receive(send2_token, channel=resp2)
  rcv2_tuple: (bits[32]) = tuple_index(rcv2, index=1)
  rcv2_data: bits[32] = tuple_index(rcv2_tuple, index=0)
  wr_comp0_rcv: (token, ()) = receive(send0_token, channel=wr_comp0)
  wr_comp1_rcv: (token, ()) = receive(send1_token, channel=wr_comp1)
  wr_comp2_rcv: (token, ()) = receive(send2_token, channel=wr_comp2)
  next_state: () = tuple()
  next_state_value: () = next_value(param=__state, value=next_state)
}
  )",
        .pipeline_stages = 20,
        .ram_config_strings = kThree1RW,
        .ram_contents = k3Ram32Bit,
    },
    RamChannelRewriteTestParam{
        .test_name = "Simple32Bit1R1W",
        .ir_text = R"(package  test
chan rd_req((bits[32], ()), id=0, kind=streaming, ops=send_only, flow_control=ready_valid, metadata="""""")
chan rd_resp((bits[32]), id=1, kind=streaming, ops=receive_only, flow_control=ready_valid, metadata="""""")
chan wr_req((bits[32], bits[32], ()), id=2, kind=streaming, ops=send_only, flow_control=ready_valid, metadata="""""")
chan wr_comp((), id=3, kind=streaming, ops=receive_only, flow_control=ready_valid, metadata="""""")

proc my_proc(__state: bits[32], init={0}) {
  __token: token = literal(value=token)
  empty_tuple: () = literal(value=())
  to_send0: (bits[32], ()) = tuple(__state, empty_tuple)
  send_token0: token = send(__token, to_send0, channel=rd_req)
  rcv: (token, (bits[32])) = receive(send_token0, channel=rd_resp)
  rcv_token: token = tuple_index(rcv, index=0)
  to_send1: (bits[32], bits[32], ()) = tuple(__state, __state, empty_tuple)
  send_token1: token = send(rcv_token, to_send1, channel=wr_req)
  wr_comp_rcv: (token, ()) = receive(send_token1, channel=wr_comp)
  one_lit: bits[32] = literal(value=1)
  next_state: bits[32] = add(__state, one_lit)
  next_state_value: () = next_value(param=__state, value=next_state)
}
  )",
        .pipeline_stages = 3,
        .ram_config_strings = kSingle1R1W,
        .ram_contents = k1Ram32Bit,
    },
    RamChannelRewriteTestParam{
        .test_name = "Simple32Bit1R1WWithMask",
        .ir_text = R"(package  test
chan rd_req((bits[32], bits[4]), id=0, kind=streaming, ops=send_only, flow_control=ready_valid, metadata="""""")
chan rd_resp((bits[32]), id=1, kind=streaming, ops=receive_only, flow_control=ready_valid, metadata="""""")
chan wr_req((bits[32], bits[32], bits[4]), id=2, kind=streaming, ops=send_only, flow_control=ready_valid, metadata="""""")
chan wr_comp((), id=3, kind=streaming, ops=receive_only, flow_control=ready_valid, metadata="""""")

proc my_proc(__state: bits[32], init={0}) {
  __token: token = literal(value=token)
  all_mask: bits[4] = literal(value=0xf)
  empty_tuple: () = literal(value=())
  to_send0: (bits[32], bits[4]) = tuple(__state, all_mask)
  send_token0: token = send(__token, to_send0, channel=rd_req)
  rcv: (token, (bits[32])) = receive(send_token0, channel=rd_resp)
  rcv_token: token = tuple_index(rcv, index=0)
  to_send1: (bits[32], bits[32], bits[4]) = tuple(__state, __state, all_mask)
  send_token1: token = send(rcv_token, to_send1, channel=wr_req)
  wr_comp_rcv: (token, ()) = receive(send_token1, channel=wr_comp)
  one_lit: bits[32] = literal(value=1)
  next_state: bits[32] = add(__state, one_lit)
  next_state_value: () = next_value(param=__state, value=next_state)
}
  )",
        .pipeline_stages = 3,
        .ram_config_strings = kSingle1R1W,
        .expect_read_mask = true,
        .expect_write_mask = true,
        .ram_contents = k1Ram32Bit,
    },
    RamChannelRewriteTestParam{
        .test_name = "Simple32Bit1R1WWithExtraneousChannels",
        .ir_text = R"(package  test
chan rd_req((bits[32], ()), id=0, kind=streaming, ops=send_only, flow_control=ready_valid, metadata="""""")
chan rd_resp((bits[32]), id=1, kind=streaming, ops=receive_only, flow_control=ready_valid, metadata="""""")
chan wr_req((bits[32], bits[32], ()), id=2, kind=streaming, ops=send_only, flow_control=ready_valid, metadata="""""")
chan wr_comp((), id=5, kind=streaming, ops=receive_only, flow_control=ready_valid, metadata="""""")
chan extra0(bits[1], id=3, kind=streaming, ops=receive_only, flow_control=ready_valid, metadata="""""")
chan extra1(bits[1], id=4, kind=streaming, ops=receive_only, flow_control=ready_valid, metadata="""""")

proc my_proc(__state: bits[32], init={0}) {
  __token: token = literal(value=token)
  empty_tuple: () = literal(value=())
  to_send0: (bits[32], ()) = tuple(__state, empty_tuple)
  send_token0: token = send(__token, to_send0, channel=rd_req)
  rcv: (token, (bits[32])) = receive(send_token0, channel=rd_resp)
  rcv_token: token = tuple_index(rcv, index=0)
  to_send1: (bits[32], bits[32], ()) = tuple(__state, __state, empty_tuple)
  send_token1: token = send(rcv_token, to_send1, channel=wr_req)
  extra0_rcv: (token, bits[1]) = receive(send_token1, channel=extra0)
  extra0_token: token = tuple_index(extra0_rcv, index=0)
  extra1_rcv: (token, bits[1]) = receive(extra0_token, channel=extra1)
  extra1_token: token = tuple_index(extra1_rcv, index=0)
  wr_comp_rcv: (token, ()) = receive(extra1_token, channel=wr_comp)
  one_lit: bits[32] = literal(value=1)
  next_state: bits[32] = add(__state, one_lit)
  next_state_value: () = next_value(param=__state, value=next_state)
}
  )",
        .pipeline_stages = 3,
        .ram_config_strings = kSingle1R1W,
        .ram_contents = k1Ram32Bit,
    },
    RamChannelRewriteTestParam{
        .test_name = "32BitWithThree1R1WRams",
        .ir_text = R"(package  test
chan rd_req0((bits[32], ()), id=0, kind=streaming, ops=send_only, flow_control=ready_valid, metadata="""""")
chan rd_resp0((bits[32]), id=1, kind=streaming, ops=receive_only, flow_control=ready_valid, metadata="""""")
chan wr_req0((bits[32], bits[32], ()), id=2, kind=streaming, ops=send_only, flow_control=ready_valid, metadata="""""")
chan wr_comp0((), id=9, kind=streaming, ops=receive_only, flow_control=ready_valid, metadata="""""")
chan rd_req1((bits[32], ()), id=3, kind=streaming, ops=send_only, flow_control=ready_valid, metadata="""""")
chan rd_resp1((bits[32]), id=4, kind=streaming, ops=receive_only, flow_control=ready_valid, metadata="""""")
chan wr_req1((bits[32], bits[32], ()), id=5, kind=streaming, ops=send_only, flow_control=ready_valid, metadata="""""")
chan wr_comp1((), id=10, kind=streaming, ops=receive_only, flow_control=ready_valid, metadata="""""")
chan rd_req2((bits[32], ()), id=6, kind=streaming, ops=send_only, flow_control=ready_valid, metadata="""""")
chan rd_resp2((bits[32]), id=7, kind=streaming, ops=receive_only, flow_control=ready_valid, metadata="""""")
chan wr_req2((bits[32], bits[32], ()), id=8, kind=streaming, ops=send_only, flow_control=ready_valid, metadata="""""")
chan wr_comp2((), id=11, kind=streaming, ops=receive_only, flow_control=ready_valid, metadata="""""")

proc my_proc(__state: bits[32], init={0}) {
  __token: token = literal(value=token)
  empty_tuple: () = literal(value=())
  to_send0: (bits[32], ()) = tuple(__state, empty_tuple)
  send_token0: token = send(__token, to_send0, channel=rd_req0)
  rcv: (token, (bits[32])) = receive(send_token0, channel=rd_resp0)
  rcv_token: token = tuple_index(rcv, index=0)
  to_send1: (bits[32], bits[32], ()) = tuple(__state, __state, empty_tuple)
  send_token1: token = send(rcv_token, to_send1, channel=wr_req0)
  send_token2: token = send(send_token1, to_send0, channel=rd_req1)
  rcv1: (token, (bits[32])) = receive(send_token2, channel=rd_resp1)
  rcv_token1: token = tuple_index(rcv1, index=0)
  to_send2: (bits[32], bits[32], ()) = tuple(__state, __state, empty_tuple)
  send_token3: token = send(rcv_token1, to_send2, channel=wr_req1)
  send_token4: token = send(send_token3, to_send0, channel=rd_req2)
  rcv2: (token, (bits[32])) = receive(send_token4, channel=rd_resp2)
  rcv_token2: token = tuple_index(rcv2, index=0)
  to_send3: (bits[32], bits[32], ()) = tuple(__state, __state, empty_tuple)
  send_token5: token = send(rcv_token2, to_send2, channel=wr_req2)
  wr_comp0_rcv: (token, ()) = receive(send_token1, channel=wr_comp0)
  wr_comp1_rcv: (token, ()) = receive(send_token3, channel=wr_comp1)
  wr_comp2_rcv: (token, ()) = receive(send_token5, channel=wr_comp2)
  one_lit: bits[32] = literal(value=1)
  next_state: bits[32] = add(__state, one_lit)
  next_state_value: () = next_value(param=__state, value=next_state)
}
  )",
        .pipeline_stages = 20,
        .ram_config_strings = kThree1R1W,
        .ram_contents = k3Ram32Bit,
    },
    RamChannelRewriteTestParam{
        .test_name = "Simple32Bit1RWAnd1R1W",
        .ir_text = R"(package  test
chan req0((bits[32], bits[32], (), (), bits[1], bits[1]), id=0, kind=streaming, ops=send_only, flow_control=ready_valid, metadata="""""")
chan resp0((bits[32]), id=1, kind=streaming, ops=receive_only, flow_control=ready_valid, metadata="""""")
chan wr_comp0((), id=5, kind=streaming, ops=receive_only, flow_control=ready_valid, metadata="""""")
chan rd_req1((bits[32], ()), id=2, kind=streaming, ops=send_only, flow_control=ready_valid, metadata="""""")
chan rd_resp1((bits[32]), id=3, kind=streaming, ops=receive_only, flow_control=ready_valid, metadata="""""")
chan wr_req1((bits[32], bits[32], ()), id=4, kind=streaming, ops=send_only, flow_control=ready_valid, metadata="""""")
chan wr_comp1((), id=6, kind=streaming, ops=receive_only, flow_control=ready_valid, metadata="""""")


proc my_proc(__state: bits[32], init={0}) {
  __token: token = literal(value=token)
  empty_tuple: () = literal(value=())
  true_lit: bits[1] = literal(value=1)
  false_lit: bits[1] = literal(value=0)
  to_send0: (bits[32], bits[32], (), (), bits[1], bits[1]) = tuple(__state, __state, empty_tuple, empty_tuple, true_lit, false_lit)
  send_token0: token = send(__token, to_send0, channel=req0)
  rcv0: (token, (bits[32])) = receive(send_token0, channel=resp0)
  rcv_token0: token = tuple_index(rcv0, index=0)
  one_lit: bits[32] = literal(value=1)
  to_send1: (bits[32], ()) = tuple(__state, empty_tuple)
  send_token1: token = send(rcv_token0, to_send1, channel=rd_req1)
  rcv1: (token, (bits[32])) = receive(send_token1, channel=rd_resp1)
  rcv_token1: token = tuple_index(rcv1, index=0)
  to_send2: (bits[32], bits[32], ()) = tuple(__state, __state, empty_tuple)
  send_token2: token = send(rcv_token1, to_send2, channel=wr_req1)
  wr_comp0_rcv: (token, ()) = receive(send_token0, channel=wr_comp0)
  wr_comp1_rcv: (token, ()) = receive(send_token2, channel=wr_comp1)
  next_state: bits[32] = add(__state, one_lit)
  next_state_value: () = next_value(param=__state, value=next_state)
}
  )",
        .pipeline_stages = 4,
        .ram_config_strings = k1RWAnd1R1W,
        .ram_contents = k2Ram32Bit,
    },
};

INSTANTIATE_TEST_SUITE_P(
    RamRewritePassTestInstantiation, RamRewritePassTest,
    ::testing::Combine(testing::ValuesIn(kTestParameters),
                       testing::Values(DefaultCodegenPassPipeline(),
                                       RamRewritePassOnly())),
    [](const testing::TestParamInfo<
        std::tuple<RamChannelRewriteTestParam, CodegenPass*>>& info) {
      return absl::StrCat(std::get<0>(info.param).test_name, "_",
                          CodegenPassName(std::get<1>(info.param)));
    });

// Expects channels named "req" and "resp" and a top level proc named "my_proc".
absl::StatusOr<Block*> MakeBlockAndRunPasses(Package* package,
                                             std::string_view ram_kind) {
  std::vector<std::unique_ptr<RamConfiguration>> ram_configurations;
  if (ram_kind == "1RW") {
    ram_configurations.push_back(std::make_unique<Ram1RWConfiguration>(
        "ram", 1, "req", "resp", "wr_comp"));
  } else if (ram_kind == "1R1W") {
    ram_configurations.push_back(std::make_unique<Ram1R1WConfiguration>(
        "ram", 1, "rd_req", "rd_resp", "wr_req", "wr_comp"));
  } else {
    return absl::InvalidArgumentError(
        absl::StrFormat("Unrecognized ram_kind %s.", ram_kind));
  }

  CodegenOptions codegen_options;
  codegen_options.flop_inputs(false)
      .flop_outputs(false)
      .clock_name("clk")
      .reset("rst", false, false, false)
      .streaming_channel_data_suffix("_data")
      .streaming_channel_valid_suffix("_valid")
      .streaming_channel_ready_suffix("_ready")
      .module_name("pipelined_proc")
      .ram_configurations(ram_configurations);
  CodegenPassOptions pass_options{
      .codegen_options = codegen_options,
  };

  XLS_ASSIGN_OR_RETURN(Proc * proc, package->GetProc("my_proc"));

  auto scheduling_options = SchedulingOptions();
  // Add constraints for each req/resp pair to be scheduled one cycle apart, and
  // put enough stages in the pipeline for all of our scenarios.
  if (ram_kind == "1RW") {
    scheduling_options.pipeline_stages(2)
        .add_constraint(IOConstraint(
            "req", IODirection::kSend, "resp", IODirection::kReceive,
            /*minimum_latency=*/1, /*maximum_latency=*/1))
        .add_constraint(IOConstraint(
            "req", IODirection::kSend, "wr_comp", IODirection::kReceive,
            /*minimum_latency=*/1, /*maximum_latency=*/1));
  } else if (ram_kind == "1R1W") {
    scheduling_options.pipeline_stages(3)
        .add_constraint(IOConstraint(
            "rd_req", IODirection::kSend, "rd_resp", IODirection::kReceive,
            /*minimum_latency=*/1, /*maximum_latency=*/1))
        .add_constraint(IOConstraint(
            "wr_req", IODirection::kSend, "wr_comp", IODirection::kReceive,
            /*minimum_latency=*/1, /*maximum_latency=*/1));
  }
  XLS_ASSIGN_OR_RETURN(auto delay_estimator, GetDelayEstimator("unit"));
  XLS_ASSIGN_OR_RETURN(
      PipelineSchedule schedule,
      RunPipelineSchedule(proc, *delay_estimator, scheduling_options));
  XLS_ASSIGN_OR_RETURN(
      CodegenPassUnit unit,
      FunctionBaseToPipelinedBlock(schedule, codegen_options, proc));
  XLS_RET_CHECK_OK(RunCodegenPassPipeline(pass_options, unit.top_block));
  return unit.top_block;
}

struct TestProc1RWVars {
  std::optional<std::string_view> req_type = std::nullopt;
  std::optional<std::string_view> resp_type = std::nullopt;
  std::optional<std::string_view> wr_comp_type = std::nullopt;
  std::optional<std::string_view> req_chan_params = std::nullopt;
  std::optional<std::string_view> resp_chan_params = std::nullopt;
  std::optional<std::string_view> wr_comp_chan_params = std::nullopt;
  std::optional<std::string_view> send_value = std::nullopt;
};

std::string MakeTestProc1RW(TestProc1RWVars vars) {
  return absl::StrReplaceAll(
      R"(
  package test
chan req($req_type, id=0, $req_chan_params, metadata="""""")
chan resp($resp_type, id=1, $resp_chan_params, metadata="""""")
chan wr_comp($wr_comp_type, id=2, $wr_comp_chan_params, metadata="""""")

proc my_proc(__state: bits[32], init={0}) {
  __token: token = literal(value=token)
  all_mask: bits[4] = literal(value=0xf)
  true_lit: bits[1] = literal(value=1)
  false_lit: bits[1] = literal(value=0)
  empty_tuple: () = literal(value=())
  to_send: $req_type = $send_value
  send_token: token = send(__token, to_send, channel=req)
  rcv: (token, $resp_type) = receive(send_token, channel=resp)
  rcv_token: token = tuple_index(rcv, index=0)
  wr_comp_rcv: (token, $wr_comp_type) = receive(rcv_token, channel=wr_comp)
  one_lit: bits[32] = literal(value=1)
  next_state: bits[32] = add(__state, one_lit)
  next_state_value: () = next_value(param=__state, value=next_state)
}
  )",
      {
          {"$req_type", vars.req_type.value_or(
                            "(bits[32], bits[32], (), (), bits[1], bits[1])")},
          {"$resp_type", vars.resp_type.value_or("(bits[32])")},
          {"$wr_comp_type", vars.wr_comp_type.value_or("()")},
          {"$req_chan_params",
           vars.req_chan_params.value_or(
               "kind=streaming, flow_control=ready_valid, ops=send_only")},
          {"$resp_chan_params",
           vars.resp_chan_params.value_or(
               "kind=streaming, flow_control=ready_valid, ops=receive_only")},
          {"$wr_comp_chan_params",
           vars.wr_comp_chan_params.value_or(
               "kind=streaming, flow_control=ready_valid, ops=receive_only")},
          {"$send_value",
           vars.send_value.value_or("tuple(__state, __state, empty_tuple, "
                                    "empty_tuple, true_lit, false_lit)")},
      });
}

struct TestProc1R1WVars {
  std::optional<std::string_view> rd_req_type = std::nullopt;
  std::optional<std::string_view> rd_resp_type = std::nullopt;
  std::optional<std::string_view> wr_req_type = std::nullopt;
  std::optional<std::string_view> wr_comp_type = std::nullopt;
  std::optional<std::string_view> rd_req_chan_params = std::nullopt;
  std::optional<std::string_view> rd_resp_chan_params = std::nullopt;
  std::optional<std::string_view> wr_req_chan_params = std::nullopt;
  std::optional<std::string_view> wr_comp_chan_params = std::nullopt;
  std::optional<std::string_view> rd_send_value = std::nullopt;
  std::optional<std::string_view> wr_send_value = std::nullopt;
};

std::string MakeTestProc1R1W(TestProc1R1WVars vars) {
  return absl::StrReplaceAll(
      R"(
package test

chan rd_req($rd_req_type, id=0, $rd_req_chan_params, metadata="""""")
chan rd_resp($rd_resp_type, id=1, $rd_resp_chan_params, metadata="""""")
chan wr_req($wr_req_type, id=2, $wr_req_chan_params, metadata="""""")
chan wr_comp($wr_comp_type, id=3, $wr_comp_chan_params, metadata="""""")

proc my_proc(__state: bits[32], init={0}) {
  __token: token = literal(value=token)
  true_lit: bits[1] = literal(value=1)
  false_lit: bits[1] = literal(value=0)
  all_mask: bits[4] = literal(value=0xf)
  empty_tuple: () = literal(value=())
  to_send0: $rd_req_type = $rd_send_value
  send_token0: token = send(__token, to_send0, channel=rd_req)
  rcv: (token, $rd_resp_type) = receive(send_token0, channel=rd_resp)
  rcv_token: token = tuple_index(rcv, index=0)
  to_send1: $wr_req_type = $wr_send_value
  send_token1: token = send(rcv_token, to_send1, channel=wr_req)
  wr_comp_recv: (token, $wr_comp_type) = receive(send_token1, channel=wr_comp)
  one_lit: bits[32] = literal(value=1)
  next_state: bits[32] = add(__state, one_lit)
  next_state_value: () = next_value(param=__state, value=next_state)
}
  )",
      {
          {"$rd_req_type", vars.rd_req_type.value_or("(bits[32], ())")},
          {"$rd_resp_type", vars.rd_resp_type.value_or("(bits[32])")},
          {"$wr_req_type",
           vars.wr_req_type.value_or("(bits[32], bits[32], ())")},
          {"$wr_comp_type", vars.wr_comp_type.value_or("()")},
          {"$rd_req_chan_params",
           vars.rd_req_chan_params.value_or(
               "kind=streaming, flow_control=ready_valid, ops=send_only")},
          {"$rd_resp_chan_params",
           vars.rd_resp_chan_params.value_or(
               "kind=streaming, flow_control=ready_valid, ops=receive_only")},
          {"$wr_req_chan_params",
           vars.wr_req_chan_params.value_or(
               "kind=streaming, flow_control=ready_valid, ops=send_only")},
          {"$wr_comp_chan_params",
           vars.wr_comp_chan_params.value_or(
               "kind=streaming, flow_control=ready_valid, ops=receive_only")},
          {"$rd_send_value",
           vars.rd_send_value.value_or("tuple(__state, empty_tuple)")},
          {"$wr_send_value",
           vars.wr_send_value.value_or("tuple(__state, __state, empty_tuple)")},
      });
}

// Tests for checking invalid inputs on 1rw RAMs.
TEST(RamRewritePassInvalidInputsTest, TestDefaultsWork1RW) {
  std::string ir_text = MakeTestProc1RW({});
  XLS_ASSERT_OK_AND_ASSIGN(auto package, Parser::ParsePackage(ir_text));
  XLS_EXPECT_OK(MakeBlockAndRunPasses(package.get(), /*ram_kind=*/"1RW"));
}

TEST(RamRewritePassInvalidInputsTest, TestWriteMaskWorks1RW) {
  std::string ir_text = MakeTestProc1RW(
      {.req_type = "(bits[32], bits[32], bits[4], (), bits[1], bits[1])",
       .send_value = "tuple(__state, __state, all_mask, empty_tuple, true_lit, "
                     "false_lit)"});
  XLS_ASSERT_OK_AND_ASSIGN(auto package, Parser::ParsePackage(ir_text));
  XLS_EXPECT_OK(MakeBlockAndRunPasses(package.get(), /*ram_kind=*/"1RW"));
}

TEST(RamRewritePassInvalidInputsTest, TestReadMaskWorks1RW) {
  std::string ir_text = MakeTestProc1RW(
      {.req_type = "(bits[32], bits[32], (), bits[4], bits[1], bits[1])",
       .send_value = "tuple(__state, __state, empty_tuple, all_mask, true_lit, "
                     "false_lit)"});
  XLS_ASSERT_OK_AND_ASSIGN(auto package, Parser::ParsePackage(ir_text));
  XLS_EXPECT_OK(MakeBlockAndRunPasses(package.get(), /*ram_kind=*/"1RW"));
}

TEST(RamRewritePassInvalidInputsTest, InvalidChannelFlowControl1RW) {
  // Try single_value channels instead of streaming
  std::string ir_text = MakeTestProc1RW(
      TestProc1RWVars{.req_chan_params = "kind=single_value, ops=send_only"});
  // The channels are single_value, so ready/valid ports are missing and the
  // pass will error.
  XLS_ASSERT_OK_AND_ASSIGN(auto package, Parser::ParsePackage(ir_text));
  EXPECT_THAT(MakeBlockAndRunPasses(package.get(), /*ram_kind=*/"1RW"),
              StatusIs(absl::StatusCode::kInternal));
}

TEST(RamRewritePassInvalidInputsTest, InvalidReqChannelTypeNotTuple1RW) {
  // Try bits type instead of tuple for req channel
  std::string ir_text = MakeTestProc1RW(TestProc1RWVars{
      .req_type = "bits[32]", .send_value = "add(__state, __state)"});
  XLS_ASSERT_OK_AND_ASSIGN(auto package, Parser::ParsePackage(ir_text));
  EXPECT_THAT(MakeBlockAndRunPasses(package.get(), /*ram_kind=*/"1RW"),
              StatusIs(absl::StatusCode::kInternal,
                       HasSubstr("Request must be a tuple type")));
}

TEST(RamRewritePassInvalidInputsTest, InvalidRespChannelTypeNotTuple1RW) {
  // Try bits type instead of tuple for resp channel
  std::string ir_text =
      MakeTestProc1RW(TestProc1RWVars{.resp_type = "bits[32]"});
  XLS_ASSERT_OK_AND_ASSIGN(auto package, Parser::ParsePackage(ir_text));
  EXPECT_THAT(MakeBlockAndRunPasses(package.get(), /*ram_kind=*/"1RW"),
              StatusIs(absl::StatusCode::kInternal,
                       HasSubstr("Response must be a tuple type")));
}

TEST(RamRewritePassInvalidInputsTest, WrongNumberRequestChannelEntries1RW) {
  // Add an extra field to the req channel
  std::string ir_text = MakeTestProc1RW(TestProc1RWVars{
      .req_type = "(bits[32], bits[32], (), (), bits[1], bits[1], bits[1])",
      .send_value =
          "tuple(__state, __state, empty_tuple, empty_tuple, true_lit, "
          "false_lit, false_lit)"});
  XLS_ASSERT_OK_AND_ASSIGN(auto package, Parser::ParsePackage(ir_text));
  EXPECT_THAT(
      MakeBlockAndRunPasses(package.get(), /*ram_kind=*/"1RW"),
      StatusIs(absl::StatusCode::kInternal,
               HasSubstr("Request must be a tuple type with 6 elements")));
}

TEST(RamRewritePassInvalidInputsTest, WrongNumberResponseChannelEntries1RW) {
  // Add an extra field to the response channel
  std::string ir_text =
      MakeTestProc1RW(TestProc1RWVars{.resp_type = "(bits[32], bits[1])"});
  XLS_ASSERT_OK_AND_ASSIGN(auto package, Parser::ParsePackage(ir_text));
  EXPECT_THAT(
      MakeBlockAndRunPasses(package.get(), /*ram_kind=*/"1RW"),
      StatusIs(absl::StatusCode::kInternal,
               HasSubstr("Response must be a tuple type with 1 element")));
}

TEST(RamRewritePassInvalidInputsTest, RequestElementNotBits1RW) {
  // Replace re with a token (re must be bits[1])
  std::string ir_text = MakeTestProc1RW(TestProc1RWVars{
      .req_type = "(bits[32], bits[32], (), (), bits[1], token)",
      .send_value = "tuple(__state, __state, empty_tuple, empty_tuple, "
                    "true_lit, __token)",
  });
  XLS_ASSERT_OK_AND_ASSIGN(auto package, Parser::ParsePackage(ir_text));
  EXPECT_THAT(
      MakeBlockAndRunPasses(package.get(), /*ram_kind=*/"1RW"),
      StatusIs(absl::StatusCode::kInternal,
               HasSubstr(
                   "Request element re (idx=5) must be type bits, got token")));
}

TEST(RamRewritePassInvalidInputsTest, ResponseElementNotBits1RW) {
  // Replace re with a token (re must be bits[1])
  std::string ir_text =
      MakeTestProc1RW(TestProc1RWVars{.resp_type = "(token)"});
  XLS_ASSERT_OK_AND_ASSIGN(auto package, Parser::ParsePackage(ir_text));
  EXPECT_THAT(MakeBlockAndRunPasses(package.get(), /*ram_kind=*/"1RW"),
              StatusIs(absl::StatusCode::kInternal,
                       HasSubstr("Response element rd_data (idx=0) must not "
                                 "contain token, got token.")));
}

TEST(RamRewritePassInvalidInputsTest, WeMustBeWidth1For1RW) {
  // Replace we with bits[32] (must be bits[1])
  std::string ir_text = MakeTestProc1RW(TestProc1RWVars{
      .req_type = "(bits[32], bits[32], (), (), bits[32], bits[1])",
      .send_value = "tuple(__state, __state, empty_tuple, empty_tuple, "
                    "__state, true_lit)",
  });
  XLS_ASSERT_OK_AND_ASSIGN(auto package, Parser::ParsePackage(ir_text));
  EXPECT_THAT(
      MakeBlockAndRunPasses(package.get(), /*ram_kind=*/"1RW"),
      StatusIs(absl::StatusCode::kInternal,
               HasSubstr("Request element we (idx=2) must be type bits[1]")));
}

TEST(RamRewritePassInvalidInputsTest, ReMustBeWidth1For1RW) {
  // Replace re with bits[32] (must be bits[1])
  std::string ir_text = MakeTestProc1RW(TestProc1RWVars{
      .req_type = "(bits[32], bits[32], (), (), bits[1], bits[32])",
      .send_value = "tuple(__state, __state, empty_tuple, empty_tuple, "
                    "true_lit, __state)",
  });
  XLS_ASSERT_OK_AND_ASSIGN(auto package, Parser::ParsePackage(ir_text));
  EXPECT_THAT(
      MakeBlockAndRunPasses(package.get(), /*ram_kind=*/"1RW"),
      StatusIs(absl::StatusCode::kInternal,
               HasSubstr("Request element re (idx=3) must be type bits[1]")));
}

// Tests for checking invalid inputs on 1r1w RAMs.
TEST(RamRewritePassInvalidInputsTest, TestDefaultsWork1R1W) {
  std::string ir_text = MakeTestProc1R1W({});
  XLS_ASSERT_OK_AND_ASSIGN(auto package, Parser::ParsePackage(ir_text));
  XLS_EXPECT_OK(MakeBlockAndRunPasses(package.get(), /*ram_kind=*/"1R1W"));
}

TEST(RamRewritePassInvalidInputsTest, TestWriteMaskWorks1R1W) {
  std::string ir_text =
      MakeTestProc1R1W({.wr_req_type = "(bits[32], bits[32], bits[4])",
                        .wr_send_value = "tuple(__state, __state, all_mask)"});
  XLS_ASSERT_OK_AND_ASSIGN(auto package, Parser::ParsePackage(ir_text));
  XLS_EXPECT_OK(MakeBlockAndRunPasses(package.get(), /*ram_kind=*/"1R1W"));
}

TEST(RamRewritePassInvalidInputsTest, TestReadMaskWorks1R1W) {
  std::string ir_text =
      MakeTestProc1R1W({.rd_req_type = "(bits[32], bits[4])",
                        .rd_send_value = "tuple(__state, all_mask)"});
  XLS_ASSERT_OK_AND_ASSIGN(auto package, Parser::ParsePackage(ir_text));
  XLS_EXPECT_OK(MakeBlockAndRunPasses(package.get(), /*ram_kind=*/"1R1W"));
}

TEST(RamRewritePassInvalidInputsTest,
     InvalidReadRequestChannelFlowControl1R1W) {
  // Try single_value channels instead of streaming
  std::string ir_text = MakeTestProc1R1W(TestProc1R1WVars{
      .rd_req_chan_params = "kind=single_value, ops=send_only"});
  // The channels are single_value, so ready/valid ports are missing and the
  // pass will error.
  XLS_ASSERT_OK_AND_ASSIGN(auto package, Parser::ParsePackage(ir_text));
  EXPECT_THAT(MakeBlockAndRunPasses(package.get(), /*ram_kind=*/"1R1W"),
              StatusIs(absl::StatusCode::kInternal));
}

TEST(RamRewritePassInvalidInputsTest,
     InvalidReadResponseChannelFlowControl1R1W) {
  // Try single_value channels instead of streaming
  std::string ir_text = MakeTestProc1R1W(TestProc1R1WVars{
      .rd_resp_chan_params = "kind=single_value, ops=receive_only"});
  // The channels are single_value, so ready/valid ports are missing and the
  // pass will error.
  XLS_ASSERT_OK_AND_ASSIGN(auto package, Parser::ParsePackage(ir_text));
  EXPECT_THAT(MakeBlockAndRunPasses(package.get(), /*ram_kind=*/"1R1W"),
              StatusIs(absl::StatusCode::kInternal));
}

TEST(RamRewritePassInvalidInputsTest, InvalidWriteChannelFlowControl1R1W) {
  // Try single_value channels instead of streaming
  std::string ir_text = MakeTestProc1R1W(TestProc1R1WVars{
      .wr_req_chan_params = "kind=single_value, ops=send_only"});
  // The channels are single_value, so ready/valid ports are missing and the
  // pass will error.
  XLS_ASSERT_OK_AND_ASSIGN(auto package, Parser::ParsePackage(ir_text));
  EXPECT_THAT(MakeBlockAndRunPasses(package.get(), /*ram_kind=*/"1R1W"),
              StatusIs(absl::StatusCode::kInternal));
}

TEST(RamRewritePassInvalidInputsTest, InvalidReadReqChannelTypeNotTuple1R1W) {
  // Try bits type instead of tuple for req channel
  std::string ir_text = MakeTestProc1R1W(TestProc1R1WVars{
      .rd_req_type = "bits[32]", .rd_send_value = "add(__state, __state)"});
  XLS_ASSERT_OK_AND_ASSIGN(auto package, Parser::ParsePackage(ir_text));
  EXPECT_THAT(MakeBlockAndRunPasses(package.get(), /*ram_kind=*/"1R1W"),
              StatusIs(absl::StatusCode::kInternal,
                       HasSubstr("rd_req must be a tuple type")));
}

TEST(RamRewritePassInvalidInputsTest, InvalidWriteReqChannelTypeNotTuple1R1W) {
  // Try bits type instead of tuple for req channel
  std::string ir_text = MakeTestProc1R1W(TestProc1R1WVars{
      .wr_req_type = "bits[32]", .wr_send_value = "add(__state, __state)"});
  XLS_ASSERT_OK_AND_ASSIGN(auto package, Parser::ParsePackage(ir_text));
  EXPECT_THAT(MakeBlockAndRunPasses(package.get(), /*ram_kind=*/"1R1W"),
              StatusIs(absl::StatusCode::kInternal,
                       HasSubstr("wr_req must be a tuple type")));
}

TEST(RamRewritePassInvalidInputsTest, InvalidReadRespChannelTypeNotTuple1R1W) {
  // Try bits type instead of tuple for resp channel
  std::string ir_text =
      MakeTestProc1R1W(TestProc1R1WVars{.rd_resp_type = "bits[32]"});
  XLS_ASSERT_OK_AND_ASSIGN(auto package, Parser::ParsePackage(ir_text));
  EXPECT_THAT(MakeBlockAndRunPasses(package.get(), /*ram_kind=*/"1R1W"),
              StatusIs(absl::StatusCode::kInternal,
                       HasSubstr("rd_resp must be a tuple type")));
}

TEST(RamRewritePassInvalidInputsTest,
     WrongNumberReadRequestChannelEntries1R1W) {
  // Add an extra field to the req channel
  std::string ir_text = MakeTestProc1R1W(TestProc1R1WVars{
      .rd_req_type = "(bits[32], (), bits[1])",
      .rd_send_value = "tuple(__state, empty_tuple, true_lit)"});
  XLS_ASSERT_OK_AND_ASSIGN(auto package, Parser::ParsePackage(ir_text));
  EXPECT_THAT(
      MakeBlockAndRunPasses(package.get(), /*ram_kind=*/"1R1W"),
      StatusIs(absl::StatusCode::kInternal,
               HasSubstr("rd_req must be a tuple type with 2 elements")));
}

TEST(RamRewritePassInvalidInputsTest,
     WrongNumberWriteRequestChannelEntries1R1W) {
  // Add an extra field to the req channel
  std::string ir_text = MakeTestProc1R1W(TestProc1R1WVars{
      .wr_req_type = "(bits[32], bits[32], (), bits[1])",
      .wr_send_value = "tuple(__state, __state, empty_tuple, true_lit)"});
  XLS_ASSERT_OK_AND_ASSIGN(auto package, Parser::ParsePackage(ir_text));
  EXPECT_THAT(
      MakeBlockAndRunPasses(package.get(), /*ram_kind=*/"1R1W"),
      StatusIs(absl::StatusCode::kInternal,
               HasSubstr("wr_req must be a tuple type with 3 elements")));
}

TEST(RamRewritePassInvalidInputsTest,
     WrongNumberReadResponseChannelEntries1R1W) {
  // Add an extra field to the response channel
  std::string ir_text =
      MakeTestProc1R1W(TestProc1R1WVars{.rd_resp_type = "(bits[32], bits[1])"});
  XLS_ASSERT_OK_AND_ASSIGN(auto package, Parser::ParsePackage(ir_text));
  EXPECT_THAT(
      MakeBlockAndRunPasses(package.get(), /*ram_kind=*/"1R1W"),
      StatusIs(absl::StatusCode::kInternal,
               HasSubstr("rd_resp must be a tuple type with 1 element")));
}

TEST(RamRewritePassInvalidInputsTest, ReadRequestElementNotBits1R1W) {
  // Replace re with a token (addr must be bits)
  std::string ir_text = MakeTestProc1R1W(TestProc1R1WVars{
      .rd_req_type = "(token, ())",
      .rd_send_value = "tuple(__token, empty_tuple)",
  });
  XLS_ASSERT_OK_AND_ASSIGN(auto package, Parser::ParsePackage(ir_text));
  EXPECT_THAT(
      MakeBlockAndRunPasses(package.get(), /*ram_kind=*/"1R1W"),
      StatusIs(
          absl::StatusCode::kInternal,
          HasSubstr(
              "rd_req element rd_addr (idx=0) must be type bits, got token")));
}

TEST(RamRewritePassInvalidInputsTest, WriteRequestElementNotBits1R1W) {
  // Replace re with a token (data must be bits[1])
  std::string ir_text = MakeTestProc1R1W(TestProc1R1WVars{
      .wr_req_type = "(bits[32], token, ())",
      .wr_send_value = "tuple(__state, __token, empty_tuple)",
  });
  XLS_ASSERT_OK_AND_ASSIGN(auto package, Parser::ParsePackage(ir_text));
  EXPECT_THAT(MakeBlockAndRunPasses(package.get(), /*ram_kind=*/"1R1W"),
              StatusIs(absl::StatusCode::kInternal,
                       HasSubstr("wr_req element wr_data (idx=1) must not "
                                 "contain token, got token")));
}

TEST(RamRewritePassInvalidInputsTest, ReadResponseElementNotBits1R1W) {
  // Replace rd_data with a token (data must be bits[1])
  std::string ir_text =
      MakeTestProc1R1W(TestProc1R1WVars{.rd_resp_type = "(token)"});
  XLS_ASSERT_OK_AND_ASSIGN(auto package, Parser::ParsePackage(ir_text));
  EXPECT_THAT(MakeBlockAndRunPasses(package.get(), /*ram_kind=*/"1R1W"),
              StatusIs(absl::StatusCode::kInternal,
                       HasSubstr("rd_resp element rd_data (idx=0) must not "
                                 "contain token, got token.")));
}

}  // namespace
}  // namespace verilog
}  // namespace xls
