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

#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_replace.h"
#include "xls/codegen/block_conversion.h"
#include "xls/codegen/codegen_options.h"
#include "xls/codegen/codegen_pass_pipeline.h"
#include "xls/codegen/ram_configuration.h"
#include "xls/common/casts.h"
#include "xls/common/status/matchers.h"
#include "xls/common/status/status_macros.h"
#include "xls/common/visitor.h"
#include "xls/delay_model/delay_estimators.h"
#include "xls/ir/ir_parser.h"
#include "xls/ir/proc.h"

namespace xls {
namespace verilog {
namespace {

using status_testing::StatusIs;
using testing::AllOf;
using testing::AnyOf;
using testing::Contains;
using testing::Eq;
using testing::HasSubstr;
using testing::Not;

class PortByNameMatcher : public ::testing::MatcherInterface<Block::Port> {
 public:
  explicit PortByNameMatcher(std::string_view port_name)
      : port_name_(port_name) {}
  PortByNameMatcher(const PortByNameMatcher&) = default;

  bool MatchAndExplain(
      const Block::Port port,
      ::testing::MatchResultListener* listener) const override {
    return std::visit(
        Visitor{
            [=](Block::ClockPort* p) {
              *listener << "ClockPort(" << p->name << ")";
              if (p->name != port_name_) {
                *listener << " does not match expected " << port_name_ << ".";
                return false;
              }
              return true;
            },
            [=](InputPort* p) {
              *listener << "InputPort(" << p->name() << ")";
              if (p->name() != port_name_) {
                *listener << " does not match expected " << port_name_ << ".";
                return false;
              }
              return true;
            },
            [=](OutputPort* p) {
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

CodegenPass* DefaultCodegenPassPipeline() {
  static CodegenCompoundPass* singleton = CreateCodegenPassPipeline().release();
  return singleton;
}

CodegenPass* RamRewritePassOnly() {
  static RamRewritePass* singleton = new RamRewritePass();
  return singleton;
}

std::string_view CodegenPassName(CodegenPass const* pass) {
  if (pass == DefaultCodegenPassPipeline()) {
    return "DefaultCodegenPassPipeline";
  }
  if (pass == RamRewritePassOnly()) {
    return "RamRewritePassOnly";
  }
  // We're seeing an unknown codegen pass, so error
  XLS_LOG(FATAL) << "Unknown codegen pass!";
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

  absl::StatusOr<Block*> MakeBlock(Package const* package,
                                   const CodegenOptions& codegen_options) {
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
        continue;
      }
      if (ram_config->ram_kind() == "1R1W") {
        auto* ram1rw_config =
            down_cast<Ram1R1WConfiguration*>(ram_config.get());
        scheduling_options.add_constraint(IOConstraint(
            ram1rw_config->r_port_configuration().request_channel_name,
            IODirection::kSend,
            ram1rw_config->r_port_configuration().response_channel_name,
            IODirection::kReceive,
            /*minimum_latency=*/ram1rw_config->latency(),
            /*maximum_latency=*/ram1rw_config->latency()));
        // No write response port, no constraint to add.
        // TODO(rigge): add write completion scheduling constraint.
        continue;
      }
    }

    XLS_ASSIGN_OR_RETURN(auto delay_estimator, GetDelayEstimator("unit"));
    XLS_ASSIGN_OR_RETURN(
        PipelineSchedule schedule,
        PipelineSchedule::Run(proc, *delay_estimator, scheduling_options));

    return ProcToPipelinedBlock(schedule, codegen_options, proc);
  }
};

TEST_P(RamRewritePassTest, PortsUpdated) {
  auto& param = std::get<0>(GetParam());
  CodegenOptions codegen_options = GetCodegenOptions();
  CodegenPassOptions pass_options{
      .codegen_options = codegen_options,
  };

  XLS_ASSERT_OK_AND_ASSIGN(auto package, Parser::ParsePackage(param.ir_text));
  XLS_ASSERT_OK_AND_ASSIGN(auto block,
                           MakeBlock(package.get(), codegen_options));
  auto pipeline = std::get<1>(GetParam());
  CodegenPassUnit unit(block->package(), block);
  PassResults results;
  XLS_ASSERT_OK_AND_ASSIGN(bool changed,
                           pipeline->Run(&unit, pass_options, &results));

  EXPECT_TRUE(changed);

  for (const auto& config : codegen_options.ram_configurations()) {
    std::vector<std::string_view> old_channel_names;
    EXPECT_THAT(config->ram_kind(), AnyOf(Eq("1RW"), Eq("1R1W")));
    if (config->ram_kind() == "1RW") {
      auto* ram1rw_config = down_cast<Ram1RWConfiguration*>(config.get());
      EXPECT_THAT(
          block->GetPorts(),
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
      if (ExpectReadMask()) {
        EXPECT_THAT(block->GetPorts(), Contains(PortByName(absl::StrFormat(
                                           "%s_rd_mask", config->ram_name()))));
      } else {
        EXPECT_THAT(block->GetPorts(),
                    Not(Contains(PortByName(
                        absl::StrFormat("%s_rd_mask", config->ram_name())))));
      }
      if (ExpectWriteMask()) {
        EXPECT_THAT(block->GetPorts(), Contains(PortByName(absl::StrFormat(
                                           "%s_wr_mask", config->ram_name()))));
      } else {
        EXPECT_THAT(block->GetPorts(),
                    Not(Contains(PortByName(
                        absl::StrFormat("%s_wr_mask", config->ram_name())))));
      }

      old_channel_names.push_back(
          ram1rw_config->rw_port_configuration().request_channel_name);
      old_channel_names.push_back(
          ram1rw_config->rw_port_configuration().response_channel_name);
    } else if (config->ram_kind() == "1R1W") {
      auto* ram1r1w_config = down_cast<Ram1R1WConfiguration*>(config.get());
      EXPECT_THAT(block->GetPorts(),
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
        EXPECT_THAT(block->GetPorts(), Contains(PortByName(absl::StrFormat(
                                           "%s_rd_mask", config->ram_name()))));
      } else {
        EXPECT_THAT(block->GetPorts(),
                    Not(Contains(PortByName(
                        absl::StrFormat("%s_rd_mask", config->ram_name())))));
      }
      if (ExpectWriteMask()) {
        EXPECT_THAT(block->GetPorts(), Contains(PortByName(absl::StrFormat(
                                           "%s_wr_mask", config->ram_name()))));
      } else {
        EXPECT_THAT(block->GetPorts(),
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
          block->GetPorts(),
          Not(AnyOf(
              Contains(PortByName(absl::StrCat(old_channel_name, "_valid"))),
              Contains(PortByName(absl::StrCat(old_channel_name, "_data"))),
              Contains(PortByName(absl::StrCat(old_channel_name, "_ready"))))));
    }
    EXPECT_THAT(block->GetPorts(),
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
  XLS_ASSERT_OK_AND_ASSIGN(auto block,
                           MakeBlock(package.get(), codegen_options));
  auto pipeline = std::get<1>(GetParam());
  CodegenPassUnit unit(block->package(), block);
  PassResults results;
  XLS_ASSERT_OK_AND_ASSIGN(bool changed,
                           pipeline->Run(&unit, pass_options, &results));

  EXPECT_TRUE(changed);

  EXPECT_TRUE(unit.signature.has_value());
  for (const auto& config : codegen_options.ram_configurations()) {
    absl::flat_hash_set<std::string_view> channel_names;
    EXPECT_THAT(config->ram_kind(), AnyOf(Eq("1RW"), Eq("1R1W")));
    if (config->ram_kind() == "1RW") {
      auto* ram1rw_config = down_cast<Ram1RWConfiguration*>(config.get());
      bool found = false;
      for (auto& ram : unit.signature->rams()) {
        if (ram.ram_oneof_case() != RamProto::RamOneofCase::kRam1Rw) {
          continue;
        }
        if (ram.ram_1rw().rw_port().request().name() ==
                ram1rw_config->rw_port_configuration().request_channel_name &&
            ram.ram_1rw().rw_port().response().name() ==
                ram1rw_config->rw_port_configuration().response_channel_name) {
          found = true;
        }
      }
      EXPECT_TRUE(found);
      channel_names.insert(
          ram1rw_config->rw_port_configuration().request_channel_name);
      channel_names.insert(
          ram1rw_config->rw_port_configuration().response_channel_name);
    } else if (config->ram_kind() == "1R1W") {
      auto* ram1r1w_config = down_cast<Ram1R1WConfiguration*>(config.get());
      bool found = false;
      for (auto& ram : unit.signature->rams()) {
        if (ram.ram_oneof_case() != RamProto::RamOneofCase::kRam1R1W) {
          continue;
        }
        if (ram.ram_1r1w().r_port().request().name() ==
                ram1r1w_config->r_port_configuration().request_channel_name &&
            ram.ram_1r1w().r_port().response().name() ==
                ram1r1w_config->r_port_configuration().response_channel_name) {
          found = true;
        }
      }
      EXPECT_TRUE(found);
      channel_names.insert(
          ram1r1w_config->r_port_configuration().request_channel_name);
      channel_names.insert(
          ram1r1w_config->r_port_configuration().response_channel_name);
      channel_names.insert(
          ram1r1w_config->w_port_configuration().request_channel_name);
    }
    for (auto& channel : unit.signature->streaming_channels()) {
      EXPECT_THAT(channel_names, Not(Contains(Eq(channel.name()))));
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
  XLS_ASSERT_OK_AND_ASSIGN(auto block,
                           MakeBlock(package.get(), codegen_options));
  auto pipeline = std::get<1>(GetParam());
  CodegenPassUnit unit(block->package(), block);
  PassResults results;
  XLS_ASSERT_OK_AND_ASSIGN(bool changed,
                           pipeline->Run(&unit, pass_options, &results));

  EXPECT_TRUE(changed);

  for (const auto& config : codegen_options.ram_configurations()) {
    if (config->ram_kind() == "1RW") {
      auto* ram1rw_config = down_cast<Ram1RWConfiguration*>(config.get());
      std::string_view wr_comp_name =
          ram1rw_config->rw_port_configuration().write_completion_channel_name;
      EXPECT_THAT(block->GetPorts(),
                  Not(Contains(AnyOf(
                      PortByName(absl::StrCat(wr_comp_name, "_ready")),
                      PortByName(wr_comp_name),
                      PortByName(absl::StrCat(wr_comp_name, "_valid"))))));
    } else if (config->ram_kind() == "1R1W") {
      auto* ram1r1w_config = down_cast<Ram1R1WConfiguration*>(config.get());
      std::string_view wr_comp_name =
          ram1r1w_config->w_port_configuration().write_completion_channel_name;
      EXPECT_THAT(block->GetPorts(),
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

constexpr RamChannelRewriteTestParam kTestParameters[] = {
    RamChannelRewriteTestParam{
        .test_name = "Simple32Bit1RW",
        .ir_text = R"(package  test
chan req((bits[32], bits[32], (), (), bits[1], bits[1]), id=0, kind=streaming, ops=send_only, flow_control=ready_valid, metadata="""""")
chan resp((bits[32]), id=1, kind=streaming, ops=receive_only, flow_control=ready_valid, metadata="""""")
chan wr_comp((), id=2, kind=streaming, ops=receive_only, flow_control=ready_valid, metadata="""""")

proc my_proc(__token: token, __state: bits[32], init={0}) {
  empty_tuple: () = literal(value=())
  true_lit: bits[1] = literal(value=1)
  false_lit: bits[1] = literal(value=0)
  to_send: (bits[32], bits[32], (), (), bits[1], bits[1]) = tuple(__state, __state, empty_tuple, empty_tuple, true_lit, false_lit)
  send_token: token = send(__token, to_send, channel_id=0)
  rcv: (token, (bits[32])) = receive(send_token, channel_id=1)
  rcv_token: token = tuple_index(rcv, index=0)
  wr_comp_rcv: (token, ()) = receive(rcv_token, channel_id=2)
  wr_comp_token: token = tuple_index(wr_comp_rcv, index=0)
  one_lit: bits[32] = literal(value=1)
  next_state: bits[32] = add(__state, one_lit)
  next (wr_comp_token, next_state)
}
  )",
        .pipeline_stages = 2,
        .ram_config_strings = kSingle1RW,
    },
    RamChannelRewriteTestParam{
        .test_name = "Simple32Bit1RWWithMask",
        .ir_text = R"(package  test
chan req((bits[32], bits[32], bits[4], bits[4], bits[1], bits[1]), id=0, kind=streaming, ops=send_only, flow_control=ready_valid, metadata="""""")
chan resp((bits[32]), id=1, kind=streaming, ops=receive_only, flow_control=ready_valid, metadata="""""")
chan wr_comp((), id=2, kind=streaming, ops=receive_only, flow_control=ready_valid, metadata="""""")

proc my_proc(__token: token, __state: bits[32], init={0}) {
  empty_tuple: () = literal(value=())
  true_lit: bits[1] = literal(value=1)
  false_lit: bits[1] = literal(value=0)
  all_mask: bits[4] = literal(value=0xf)
  to_send: (bits[32], bits[32], bits[4], bits[4], bits[1], bits[1]) = tuple(__state, __state, all_mask, all_mask, true_lit, false_lit)
  send_token: token = send(__token, to_send, channel_id=0)
  rcv: (token, (bits[32])) = receive(send_token, channel_id=1)
  rcv_token: token = tuple_index(rcv, index=0)
  wr_comp_rcv: (token, ()) = receive(rcv_token, channel_id=2)
  wr_comp_token: token = tuple_index(wr_comp_rcv, index=0)
  one_lit: bits[32] = literal(value=1)
  next_state: bits[32] = add(__state, one_lit)
  next (wr_comp_token, next_state)
}
  )",
        .pipeline_stages = 2,
        .ram_config_strings = kSingle1RW,
        .expect_read_mask = true,
        .expect_write_mask = true,
    },
    RamChannelRewriteTestParam{
        .test_name = "Simple32Bit1RWWithExtraneousChannels",
        .ir_text = R"(package  test
chan req((bits[32], bits[32], (), (), bits[1], bits[1]), id=3, kind=streaming, ops=send_only, flow_control=ready_valid, metadata="""""")
chan resp((bits[32]), id=1, kind=streaming, ops=receive_only, flow_control=ready_valid, metadata="""""")
chan wr_comp((), id=4, kind=streaming, ops=receive_only, flow_control=ready_valid, metadata="""""")
chan extra0(bits[1], id=0, kind=streaming, ops=receive_only, flow_control=ready_valid, metadata="""""")
chan extra1(bits[1], id=2, kind=streaming, ops=receive_only, flow_control=ready_valid, metadata="""""")

proc my_proc(__token: token, __state: bits[32], init={0}) {
  true_lit: bits[1] = literal(value=1)
  false_lit: bits[1] = literal(value=0)
  empty_tuple: () = literal(value=())
  to_send: (bits[32], bits[32], (), (), bits[1], bits[1]) = tuple(__state, __state, empty_tuple, empty_tuple, true_lit, false_lit)
  send_token: token = send(__token, to_send, channel_id=3)
  rcv: (token, (bits[32])) = receive(send_token, channel_id=1)
  rcv_token: token = tuple_index(rcv, index=0)
  one_lit: bits[32] = literal(value=1)
  extra0_rcv: (token, bits[1]) = receive(rcv_token, channel_id=0)
  extra0_token: token = tuple_index(extra0_rcv, index=0)
  extra1_rcv: (token, bits[1]) = receive(extra0_token, channel_id=2)
  extra1_token: token = tuple_index(extra1_rcv, index=0)
  wr_comp_rcv: (token, ()) = receive(extra1_token, channel_id=4)
  wr_comp_token: token = tuple_index(wr_comp_rcv, index=0)
  next_state: bits[32] = add(__state, one_lit)
  next (wr_comp_token, next_state)
}
  )",
        .pipeline_stages = 4,
        .ram_config_strings = kSingle1RW,
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

proc my_proc(__token: token, __state: (), init={()}) {
  true_lit: bits[1] = literal(value=1)
  false_lit: bits[1] = literal(value=0)
  empty_lit: bits[32] = literal(value=0)
  empty_tuple: () = literal(value=())
  to_send0: (bits[32], bits[32], (), (), bits[1], bits[1]) = tuple(empty_lit, empty_lit, empty_tuple, empty_tuple, false_lit, true_lit)
  send0_token: token = send(__token, to_send0, channel_id=0)
  rcv0: (token, (bits[32])) = receive(send0_token, channel_id=1)
  rcv0_token: token = tuple_index(rcv0, index=0)
  rcv0_tuple: (bits[32]) = tuple_index(rcv0, index=1)
  rcv0_data: bits[32] = tuple_index(rcv0_tuple, index=0)
  to_send1: (bits[32], bits[32], (), (), bits[1], bits[1]) = tuple(rcv0_data, empty_lit, empty_tuple, empty_tuple, false_lit, true_lit)
  send1_token: token = send(__token, to_send1, channel_id=2)
  rcv1: (token, (bits[32])) = receive(send1_token, channel_id=3)
  rcv1_token: token = tuple_index(rcv1, index=0)
  rcv1_tuple: (bits[32]) = tuple_index(rcv1, index=1)
  rcv1_data: bits[32] = tuple_index(rcv1_tuple, index=0)
  to_send2: (bits[32], bits[32], (), (), bits[1], bits[1]) = tuple(rcv1_data, empty_lit, empty_tuple, empty_tuple, false_lit, true_lit)
  send2_token: token = send(__token, to_send2, channel_id=4)
  rcv2: (token, (bits[32])) = receive(send2_token, channel_id=5)
  rcv2_token: token = tuple_index(rcv2, index=0)
  rcv2_tuple: (bits[32]) = tuple_index(rcv2, index=1)
  rcv2_data: bits[32] = tuple_index(rcv2_tuple, index=0)
  wr_comp0_rcv: (token, ()) = receive(rcv2_token, channel_id=6)
  wr_comp0_token: token = tuple_index(wr_comp0_rcv, index=0)
  wr_comp1_rcv: (token, ()) = receive(rcv2_token, channel_id=7)
  wr_comp1_token: token = tuple_index(wr_comp1_rcv, index=0)
  wr_comp2_rcv: (token, ()) = receive(rcv2_token, channel_id=8)
  wr_comp2_token: token = tuple_index(wr_comp2_rcv, index=0)
  after_all_token: token = after_all(rcv0_token, rcv1_token, rcv2_token, wr_comp0_token, wr_comp1_token, wr_comp2_token)
  next_state: () = tuple()
  next (after_all_token, next_state)
}
  )",
        .pipeline_stages = 20,
        .ram_config_strings = kThree1RW,
    },
    RamChannelRewriteTestParam{
        .test_name = "Simple32Bit1R1W",
        .ir_text = R"(package  test
chan rd_req((bits[32], ()), id=0, kind=streaming, ops=send_only, flow_control=ready_valid, metadata="""""")
chan rd_resp((bits[32]), id=1, kind=streaming, ops=receive_only, flow_control=ready_valid, metadata="""""")
chan wr_req((bits[32], bits[32], ()), id=2, kind=streaming, ops=send_only, flow_control=ready_valid, metadata="""""")
chan wr_comp((), id=3, kind=streaming, ops=receive_only, flow_control=ready_valid, metadata="""""")

proc my_proc(__token: token, __state: bits[32], init={0}) {
  empty_tuple: () = literal(value=())
  to_send0: (bits[32], ()) = tuple(__state, empty_tuple)
  send_token0: token = send(__token, to_send0, channel_id=0)
  rcv: (token, (bits[32])) = receive(send_token0, channel_id=1)
  rcv_token: token = tuple_index(rcv, index=0)
  to_send1: (bits[32], bits[32], ()) = tuple(__state, __state, empty_tuple)
  send_token1: token = send(rcv_token, to_send1, channel_id=2)
  wr_comp_rcv: (token, ()) = receive(send_token1, channel_id=3)
  wr_comp_token: token = tuple_index(wr_comp_rcv, index=0)
  one_lit: bits[32] = literal(value=1)
  next_state: bits[32] = add(__state, one_lit)
  next (wr_comp_token, next_state)
}
  )",
        .pipeline_stages = 2,
        .ram_config_strings = kSingle1R1W,
    },
    RamChannelRewriteTestParam{
        .test_name = "Simple32Bit1R1WWithMask",
        .ir_text = R"(package  test
chan rd_req((bits[32], bits[4]), id=0, kind=streaming, ops=send_only, flow_control=ready_valid, metadata="""""")
chan rd_resp((bits[32]), id=1, kind=streaming, ops=receive_only, flow_control=ready_valid, metadata="""""")
chan wr_req((bits[32], bits[32], bits[4]), id=2, kind=streaming, ops=send_only, flow_control=ready_valid, metadata="""""")
chan wr_comp((), id=3, kind=streaming, ops=receive_only, flow_control=ready_valid, metadata="""""")

proc my_proc(__token: token, __state: bits[32], init={0}) {
  all_mask: bits[4] = literal(value=0xf)
  empty_tuple: () = literal(value=())
  to_send0: (bits[32], bits[4]) = tuple(__state, all_mask)
  send_token0: token = send(__token, to_send0, channel_id=0)
  rcv: (token, (bits[32])) = receive(send_token0, channel_id=1)
  rcv_token: token = tuple_index(rcv, index=0)
  to_send1: (bits[32], bits[32], bits[4]) = tuple(__state, __state, all_mask)
  send_token1: token = send(rcv_token, to_send1, channel_id=2)
  wr_comp_rcv: (token, ()) = receive(send_token1, channel_id=3)
  wr_comp_token: token = tuple_index(wr_comp_rcv, index=0)
  one_lit: bits[32] = literal(value=1)
  next_state: bits[32] = add(__state, one_lit)
  next (wr_comp_token, next_state)
}
  )",
        .pipeline_stages = 2,
        .ram_config_strings = kSingle1R1W,
        .expect_read_mask = true,
        .expect_write_mask = true,
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

proc my_proc(__token: token, __state: bits[32], init={0}) {
  empty_tuple: () = literal(value=())
  to_send0: (bits[32], ()) = tuple(__state, empty_tuple)
  send_token0: token = send(__token, to_send0, channel_id=0)
  rcv: (token, (bits[32])) = receive(send_token0, channel_id=1)
  rcv_token: token = tuple_index(rcv, index=0)
  to_send1: (bits[32], bits[32], ()) = tuple(__state, __state, empty_tuple)
  send_token1: token = send(rcv_token, to_send1, channel_id=2)
  extra0_rcv: (token, bits[1]) = receive(send_token1, channel_id=3)
  extra0_token: token = tuple_index(extra0_rcv, index=0)
  extra1_rcv: (token, bits[1]) = receive(extra0_token, channel_id=4)
  extra1_token: token = tuple_index(extra1_rcv, index=0)
  wr_comp_rcv: (token, ()) = receive(extra1_token, channel_id=5)
  wr_comp_token: token = tuple_index(wr_comp_rcv, index=0)
  one_lit: bits[32] = literal(value=1)
  next_state: bits[32] = add(__state, one_lit)
  next (wr_comp_token, next_state)
}
  )",
        .pipeline_stages = 2,
        .ram_config_strings = kSingle1R1W,
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

proc my_proc(__token: token, __state: bits[32], init={0}) {
  empty_tuple: () = literal(value=())
  to_send0: (bits[32], ()) = tuple(__state, empty_tuple)
  send_token0: token = send(__token, to_send0, channel_id=0)
  rcv: (token, (bits[32])) = receive(send_token0, channel_id=1)
  rcv_token: token = tuple_index(rcv, index=0)
  to_send1: (bits[32], bits[32], ()) = tuple(__state, __state, empty_tuple)
  send_token1: token = send(rcv_token, to_send1, channel_id=2)
  send_token2: token = send(send_token1, to_send0, channel_id=3)
  rcv1: (token, (bits[32])) = receive(send_token2, channel_id=4)
  rcv_token1: token = tuple_index(rcv1, index=0)
  to_send2: (bits[32], bits[32], ()) = tuple(__state, __state, empty_tuple)
  send_token3: token = send(rcv_token1, to_send2, channel_id=5)
  send_token4: token = send(send_token3, to_send0, channel_id=6)
  rcv2: (token, (bits[32])) = receive(send_token4, channel_id=7)
  rcv_token2: token = tuple_index(rcv2, index=0)
  to_send3: (bits[32], bits[32], ()) = tuple(__state, __state, empty_tuple)
  send_token5: token = send(rcv_token2, to_send2, channel_id=8)
  wr_comp0_rcv: (token, ()) = receive(send_token5, channel_id=9)
  wr_comp0_token: token = tuple_index(wr_comp0_rcv, index=0)
  wr_comp1_rcv: (token, ()) = receive(wr_comp0_token, channel_id=10)
  wr_comp1_token: token = tuple_index(wr_comp1_rcv, index=0)
  wr_comp2_rcv: (token, ()) = receive(wr_comp1_token, channel_id=11)
  wr_comp2_token: token = tuple_index(wr_comp2_rcv, index=0)
  one_lit: bits[32] = literal(value=1)
  next_state: bits[32] = add(__state, one_lit)
  next (wr_comp2_token, next_state)
}
  )",
        .pipeline_stages = 20,
        .ram_config_strings = kThree1R1W,
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


proc my_proc(__token: token, __state: bits[32], init={0}) {
  empty_tuple: () = literal(value=())
  true_lit: bits[1] = literal(value=1)
  false_lit: bits[1] = literal(value=0)
  to_send0: (bits[32], bits[32], (), (), bits[1], bits[1]) = tuple(__state, __state, empty_tuple, empty_tuple, true_lit, false_lit)
  send_token0: token = send(__token, to_send0, channel_id=0)
  rcv0: (token, (bits[32])) = receive(send_token0, channel_id=1)
  rcv_token0: token = tuple_index(rcv0, index=0)
  one_lit: bits[32] = literal(value=1)
  to_send1: (bits[32], ()) = tuple(__state, empty_tuple)
  send_token1: token = send(rcv_token0, to_send1, channel_id=2)
  rcv1: (token, (bits[32])) = receive(send_token1, channel_id=3)
  rcv_token1: token = tuple_index(rcv1, index=0)
  to_send2: (bits[32], bits[32], ()) = tuple(__state, __state, empty_tuple)
  send_token2: token = send(rcv_token1, to_send2, channel_id=4)
  wr_comp0_rcv: (token, ()) = receive(send_token2, channel_id=5)
  wr_comp0_token: token = tuple_index(wr_comp0_rcv, index=0)
  wr_comp1_rcv: (token, ()) = receive(wr_comp0_token, channel_id=6)
  wr_comp1_token: token = tuple_index(wr_comp1_rcv, index=0)
  next_state: bits[32] = add(__state, one_lit)
  next (wr_comp1_token, next_state)
}
  )",
        .pipeline_stages = 4,
        .ram_config_strings = k1RWAnd1R1W,
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

  auto scheduling_options = SchedulingOptions().pipeline_stages(2);
  // Add constraints for each req/resp pair to be scheduled one cycle apart
  scheduling_options.add_constraint(
      IOConstraint("req", IODirection::kSend, "resp", IODirection::kReceive,
                   /*minimum_latency=*/1, /*maximum_latency=*/1));
  scheduling_options.add_constraint(
      IOConstraint("req", IODirection::kSend, "wr_comp", IODirection::kReceive,
                   /*minimum_latency=*/1, /*maximum_latency=*/1));
  XLS_ASSIGN_OR_RETURN(auto delay_estimator, GetDelayEstimator("unit"));
  XLS_ASSIGN_OR_RETURN(
      PipelineSchedule schedule,
      PipelineSchedule::Run(proc, *delay_estimator, scheduling_options));
  XLS_ASSIGN_OR_RETURN(Block * block,
                       ProcToPipelinedBlock(schedule, codegen_options, proc));
  XLS_RET_CHECK_OK(RunCodegenPassPipeline(pass_options, block));
  return block;
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

proc my_proc(__token: token, __state: bits[32], init={0}) {
  all_mask: bits[4] = literal(value=0xf)
  true_lit: bits[1] = literal(value=1)
  false_lit: bits[1] = literal(value=0)
  empty_tuple: () = literal(value=())
  to_send: $req_type = $send_value
  send_token: token = send(__token, to_send, channel_id=0)
  rcv: (token, $resp_type) = receive(send_token, channel_id=1)
  rcv_token: token = tuple_index(rcv, index=0)
  wr_comp_rcv: (token, $wr_comp_type) = receive(rcv_token, channel_id=2)
  wr_comp_token: token = tuple_index(wr_comp_rcv, index=0)
  one_lit: bits[32] = literal(value=1)
  next_state: bits[32] = add(__state, one_lit)
  next (wr_comp_token, next_state)
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

proc my_proc(__token: token, __state: bits[32], init={0}) {
  true_lit: bits[1] = literal(value=1)
  false_lit: bits[1] = literal(value=0)
  all_mask: bits[4] = literal(value=0xf)
  empty_tuple: () = literal(value=())
  to_send0: $rd_req_type = $rd_send_value
  send_token0: token = send(__token, to_send0, channel_id=0)
  rcv: (token, $rd_resp_type) = receive(send_token0, channel_id=1)
  rcv_token: token = tuple_index(rcv, index=0)
  to_send1: $wr_req_type = $wr_send_value
  send_token1: token = send(rcv_token, to_send1, channel_id=2)
  wr_comp_recv: (token, $wr_comp_type) = receive(send_token1, channel_id=3)
  wr_comp_token: token = tuple_index(wr_comp_recv, index=0)
  one_lit: bits[32] = literal(value=1)
  next_state: bits[32] = add(__state, one_lit)
  next (wr_comp_token, next_state)
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
