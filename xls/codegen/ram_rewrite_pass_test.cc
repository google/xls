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
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_replace.h"
#include "xls/codegen/block_conversion.h"
#include "xls/codegen/codegen_options.h"
#include "xls/codegen/codegen_pass_pipeline.h"
#include "xls/common/status/matchers.h"
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
  absl::Span<const std::pair<std::string_view, std::string_view>>
      req_resp_channel_pairs;
};

class RamRewritePassTest
    : public testing::TestWithParam<
          std::tuple<RamChannelRewriteTestParam, CodegenPass*>> {
 protected:
  CodegenOptions GetCodegenOptions() {
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
    ram_configurations.reserve(param.req_resp_channel_pairs.size());
    int ram_id = 0;
    for (auto const& [req_name, resp_name] : param.req_resp_channel_pairs) {
      ram_configurations.push_back(std::make_unique<Ram1RWConfiguration>(
          /*ram_name=*/absl::StrFormat("ram%d", ram_id),
          /*latency=*/1,
          /*request_name=*/req_name,
          /*response_name=*/resp_name));
      ram_id++;
    }
    codegen_options.ram_configurations(ram_configurations);
    return codegen_options;
  }

  absl::StatusOr<Block*> MakeBlock(Package const* package,
                                   const CodegenOptions& codegen_options) {
    auto& param = std::get<0>(GetParam());

    XLS_ASSIGN_OR_RETURN(Proc * proc, package->GetProc("my_proc"));

    auto scheduling_options =
        SchedulingOptions().pipeline_stages(param.pipeline_stages);
    // Add constraints for each req/resp pair to be scheduled one cycle apart
    for (auto& [req_channel, resp_channel] : param.req_resp_channel_pairs) {
      scheduling_options.add_constraint(IOConstraint(
          req_channel, IODirection::kSend, resp_channel, IODirection::kReceive,
          /*minimum_latency=*/1, /*maximum_latency=*/1));
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

  int ram_id = 0;
  for (auto& [req_channel, resp_channel] : param.req_resp_channel_pairs) {
    EXPECT_THAT(
        block->GetPorts(),
        Not(AnyOf(Contains(PortByName(absl::StrCat(req_channel, "_valid"))),
                  Contains(PortByName(absl::StrCat(req_channel, "_data"))),
                  Contains(PortByName(absl::StrCat(req_channel, "_ready"))))));
    EXPECT_THAT(
        block->GetPorts(),
        AllOf(Contains(PortByName(absl::StrFormat("ram%d_addr", ram_id))),
              Contains(PortByName(absl::StrFormat("ram%d_wr_data", ram_id))),
              Contains(PortByName(absl::StrFormat("ram%d_we", ram_id))),
              Contains(PortByName(absl::StrFormat("ram%d_re", ram_id)))));
    EXPECT_THAT(
        block->GetPorts(),
        Not(AnyOf(
            Contains(PortByName(absl::StrFormat("ram%d_valid", ram_id))),
            Contains(PortByName(absl::StrFormat("ram%d_data", ram_id))),
            Contains(PortByName(absl::StrFormat("ram%d_ready", ram_id))))));
    EXPECT_THAT(block->GetPorts(),
                Contains(PortByName(absl::StrFormat("ram%d_rd_data", ram_id))));
    ram_id++;
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
  for (auto& [req_channel, resp_channel] : param.req_resp_channel_pairs) {
    for (auto& channel : unit.signature->streaming_channels()) {
      EXPECT_NE(req_channel, channel.name());
      EXPECT_NE(resp_channel, channel.name());
    }
    bool found = false;
    for (auto& ram : unit.signature->rams()) {
      EXPECT_EQ(ram.ram_oneof_case(), RamProto::RamOneofCase::kRam1Rw);
      if (ram.ram_1rw().rw_port().request().name() == req_channel &&
          ram.ram_1rw().rw_port().response().name() == resp_channel) {
        found = true;
      }
    }
    EXPECT_TRUE(found);
  }
}

constexpr std::pair<std::string_view, std::string_view> kSingleReqResp[] = {
    {"req", "resp"}};

constexpr std::pair<std::string_view, std::string_view> kThreeReqResp[] = {
    {"req0", "resp0"},
    {"req1", "resp1"},
    {"req2", "resp2"},
};

constexpr RamChannelRewriteTestParam kTestParameters[] = {
    RamChannelRewriteTestParam{
        .test_name = "Simple32Bit",
        .ir_text = R"(package  test
chan req((bits[32], bits[32], bits[1], bits[1]), id=0, kind=streaming, ops=send_only, flow_control=ready_valid, metadata="""""")
chan resp((bits[32]), id=1, kind=streaming, ops=receive_only, flow_control=ready_valid, metadata="""""")

proc my_proc(__token: token, __state: bits[32], init={0}) {
  true_lit: bits[1] = literal(value=1)
  false_lit: bits[1] = literal(value=0)
  to_send: (bits[32], bits[32], bits[1], bits[1]) = tuple(__state, __state, true_lit, false_lit)
  send_token: token = send(__token, to_send, channel_id=0)
  rcv: (token, (bits[32])) = receive(send_token, channel_id=1)
  rcv_token: token = tuple_index(rcv, index=0)
  one_lit: bits[32] = literal(value=1)
  next_state: bits[32] = add(__state, one_lit)
  next (rcv_token, next_state)
}
  )",
        .pipeline_stages = 2,
        .req_resp_channel_pairs = kSingleReqResp,
    },
    RamChannelRewriteTestParam{
        .test_name = "Simple32BitWithExtraneousChannels",
        .ir_text = R"(package  test
chan req((bits[32], bits[32], bits[1], bits[1]), id=3, kind=streaming, ops=send_only, flow_control=ready_valid, metadata="""""")
chan resp((bits[32]), id=1, kind=streaming, ops=receive_only, flow_control=ready_valid, metadata="""""")
chan extra0(bits[1], id=0, kind=streaming, ops=receive_only, flow_control=ready_valid, metadata="""""")
chan extra1(bits[1], id=2, kind=streaming, ops=receive_only, flow_control=ready_valid, metadata="""""")

proc my_proc(__token: token, __state: bits[32], init={0}) {
  true_lit: bits[1] = literal(value=1)
  false_lit: bits[1] = literal(value=0)
  to_send: (bits[32], bits[32], bits[1], bits[1]) = tuple(__state, __state, true_lit, false_lit)
  send_token: token = send(__token, to_send, channel_id=3)
  rcv: (token, (bits[32])) = receive(send_token, channel_id=1)
  rcv_token: token = tuple_index(rcv, index=0)
  one_lit: bits[32] = literal(value=1)
  extra0_rcv: (token, bits[1]) = receive(rcv_token, channel_id=0)
  extra0_token: token = tuple_index(extra0_rcv, index=0)
  extra1_rcv: (token, bits[1]) = receive(extra0_token, channel_id=2)
  extra1_token: token = tuple_index(extra1_rcv, index=0)
  next_state: bits[32] = add(__state, one_lit)
  next (extra1_token, next_state)
}
  )",
        .pipeline_stages = 4,
        .req_resp_channel_pairs = kSingleReqResp,
    },
    RamChannelRewriteTestParam{
        .test_name = "32BitWithThreeRamChannels",
        .ir_text = R"(package  test
chan req0((bits[32], bits[32], bits[1], bits[1]), id=0, kind=streaming, ops=send_only, flow_control=ready_valid, metadata="""""")
chan resp0((bits[32]), id=1, kind=streaming, ops=receive_only, flow_control=ready_valid, metadata="""""")
chan req1((bits[32], bits[32], bits[1], bits[1]), id=2, kind=streaming, ops=send_only, flow_control=ready_valid, metadata="""""")
chan resp1((bits[32]), id=3, kind=streaming, ops=receive_only, flow_control=ready_valid, metadata="""""")
chan req2((bits[32], bits[32], bits[1], bits[1]), id=4, kind=streaming, ops=send_only, flow_control=ready_valid, metadata="""""")
chan resp2((bits[32]), id=5, kind=streaming, ops=receive_only, flow_control=ready_valid, metadata="""""")

proc my_proc(__token: token, __state: (), init={()}) {
  true_lit: bits[1] = literal(value=1)
  false_lit: bits[1] = literal(value=0)
  empty_lit: bits[32] = literal(value=0)
  to_send0: (bits[32], bits[32], bits[1], bits[1]) = tuple(empty_lit, empty_lit, false_lit, true_lit)
  send0_token: token = send(__token, to_send0, channel_id=0)
  rcv0: (token, (bits[32])) = receive(send0_token, channel_id=1)
  rcv0_token: token = tuple_index(rcv0, index=0)
  rcv0_tuple: (bits[32]) = tuple_index(rcv0, index=1)
  rcv0_data: bits[32] = tuple_index(rcv0_tuple, index=0)
  to_send1: (bits[32], bits[32], bits[1], bits[1]) = tuple(rcv0_data, empty_lit, false_lit, true_lit)
  send1_token: token = send(__token, to_send1, channel_id=2)
  rcv1: (token, (bits[32])) = receive(send1_token, channel_id=3)
  rcv1_token: token = tuple_index(rcv1, index=0)
  rcv1_tuple: (bits[32]) = tuple_index(rcv1, index=1)
  rcv1_data: bits[32] = tuple_index(rcv1_tuple, index=0)
  to_send2: (bits[32], bits[32], bits[1], bits[1]) = tuple(rcv1_data, empty_lit, false_lit, true_lit)
  send2_token: token = send(__token, to_send2, channel_id=4)
  rcv2: (token, (bits[32])) = receive(send2_token, channel_id=5)
  rcv2_token: token = tuple_index(rcv2, index=0)
  rcv2_tuple: (bits[32]) = tuple_index(rcv2, index=1)
  rcv2_data: bits[32] = tuple_index(rcv2_tuple, index=0)
  after_all_token: token = after_all(rcv0_token, rcv1_token, rcv2_token)
  next_state: () = tuple()
  next (after_all_token, next_state)
}
  )",
        .pipeline_stages = 20,
        .req_resp_channel_pairs = kThreeReqResp,
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
absl::StatusOr<Block*> MakeBlockAndRunPasses(Package* package) {
  std::vector<std::unique_ptr<RamConfiguration>> ram_configurations;
  ram_configurations.push_back(
      std::make_unique<Ram1RWConfiguration>("ram", 1, "req", "resp"));
  CodegenOptions codegen_options;
  codegen_options.flop_inputs(false)
      .flop_outputs(false)
      .clock_name("clk")
      .reset("rst", false, false, false)
      .streaming_channel_data_suffix("_data")
      .streaming_channel_valid_suffix("_valid")
      .streaming_channel_ready_suffix("_ready")
      .module_name("pipelined_proc")
      .ram_configurations(std::move(ram_configurations));
  CodegenPassOptions pass_options{
      .codegen_options = codegen_options,
  };

  XLS_ASSIGN_OR_RETURN(Proc * proc, package->GetProc("my_proc"));

  auto scheduling_options = SchedulingOptions().pipeline_stages(2);
  // Add constraints for each req/resp pair to be scheduled one cycle apart
  scheduling_options.add_constraint(
      IOConstraint("req", IODirection::kSend, "resp", IODirection::kReceive,
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

struct TestProcVars {
  std::optional<std::string_view> req_type = std::nullopt;
  std::optional<std::string_view> resp_type = std::nullopt;
  std::optional<std::string_view> req_chan_params = std::nullopt;
  std::optional<std::string_view> resp_chan_params = std::nullopt;
  std::optional<std::string_view> send_value = std::nullopt;
};

std::string MakeTestProc(TestProcVars vars) {
  return absl::StrReplaceAll(
      R"(
  package test
chan req($req_type, id=0, $req_chan_params, metadata="""""")
chan resp($resp_type, id=1, $resp_chan_params, metadata="""""")

proc my_proc(__token: token, __state: bits[32], init={0}) {
  true_lit: bits[1] = literal(value=1)
  false_lit: bits[1] = literal(value=0)
  to_send: $req_type = $send_value
  send_token: token = send(__token, to_send, channel_id=0)
  rcv: (token, $resp_type) = receive(send_token, channel_id=1)
  rcv_token: token = tuple_index(rcv, index=0)
  one_lit: bits[32] = literal(value=1)
  next_state: bits[32] = add(__state, one_lit)
  next (rcv_token, next_state)
}
  )",
      {
          {"$req_type",
           vars.req_type.value_or("(bits[32], bits[32], bits[1], bits[1])")},
          {"$resp_type", vars.resp_type.value_or("(bits[32])")},
          {"$req_chan_params",
           vars.req_chan_params.value_or(
               "kind=streaming, flow_control=ready_valid, ops=send_only")},
          {"$resp_chan_params",
           vars.resp_chan_params.value_or(
               "kind=streaming, flow_control=ready_valid, ops=receive_only")},
          {"$send_value", vars.send_value.value_or(
                              "tuple(__state, __state, true_lit, false_lit)")},
      });
}

// Tests for checking invalid inputs
TEST(RamRewritePassInvalidInputsTest, InvalidChannelFlowControl) {
  // Try single_value channels instead of streaming
  std::string ir_text = MakeTestProc(
      TestProcVars{.req_chan_params = "kind=single_value, ops=send_only"});
  // The channels are single_value, so ready/valid ports are missing and the
  // pass will error.
  XLS_ASSERT_OK_AND_ASSIGN(auto package, Parser::ParsePackage(ir_text));
  EXPECT_THAT(MakeBlockAndRunPasses(package.get()),
              StatusIs(absl::StatusCode::kInternal));
}

TEST(RamRewritePassInvalidInputsTest, InvalidReqChannelTypeNotTuple) {
  // Try bits type instead of tuple for req channel
  std::string ir_text = MakeTestProc(TestProcVars{
      .req_type = "bits[32]", .send_value = "add(__state, __state)"});
  XLS_ASSERT_OK_AND_ASSIGN(auto package, Parser::ParsePackage(ir_text));
  EXPECT_THAT(MakeBlockAndRunPasses(package.get()),
              StatusIs(absl::StatusCode::kInternal,
                       HasSubstr("Request must be a tuple type")));
}

TEST(RamRewritePassInvalidInputsTest, InvalidRespChannelTypeNotTuple) {
  // Try bits type instead of tuple for resp channel
  std::string ir_text = MakeTestProc(TestProcVars{.resp_type = "bits[32]"});
  XLS_ASSERT_OK_AND_ASSIGN(auto package, Parser::ParsePackage(ir_text));
  EXPECT_THAT(MakeBlockAndRunPasses(package.get()),
              StatusIs(absl::StatusCode::kInternal,
                       HasSubstr("Response must be a tuple type")));
}

TEST(RamRewritePassInvalidInputsTest, WrongNumberRequestChannelEntries) {
  // Add an extra field to the req channel
  std::string ir_text = MakeTestProc(TestProcVars{
      .req_type = "(bits[32], bits[32], bits[1], bits[1], bits[1])",
      .send_value = "tuple(__state, __state, true_lit, false_lit, false_lit)"});
  XLS_ASSERT_OK_AND_ASSIGN(auto package, Parser::ParsePackage(ir_text));
  EXPECT_THAT(
      MakeBlockAndRunPasses(package.get()),
      StatusIs(absl::StatusCode::kInternal,
               HasSubstr("Request must be a tuple type with 4 elements")));
}

TEST(RamRewritePassInvalidInputsTest, WrongNumberResponseChannelEntries) {
  // Add an extra field to the response channel
  std::string ir_text =
      MakeTestProc(TestProcVars{.resp_type = "(bits[32], bits[1])"});
  XLS_ASSERT_OK_AND_ASSIGN(auto package, Parser::ParsePackage(ir_text));
  EXPECT_THAT(
      MakeBlockAndRunPasses(package.get()),
      StatusIs(absl::StatusCode::kInternal,
               HasSubstr("Response must be a tuple type with 1 element")));
}

TEST(RamRewritePassInvalidInputsTest, RequestElementNotBits) {
  // Replace re with a token (re must be bits[1])
  std::string ir_text = MakeTestProc(TestProcVars{
      .req_type = "(bits[32], bits[32], bits[1], token)",
      .send_value = "tuple(__state, __state, true_lit, __token)",
  });
  XLS_ASSERT_OK_AND_ASSIGN(auto package, Parser::ParsePackage(ir_text));
  EXPECT_THAT(MakeBlockAndRunPasses(package.get()),
              StatusIs(absl::StatusCode::kInternal,
                       HasSubstr("element must be type bits, got token")));
}

TEST(RamRewritePassInvalidInputsTest, ResponseElementNotBits) {
  // Replace re with a token (re must be bits[1])
  std::string ir_text = MakeTestProc(TestProcVars{.resp_type = "(token)"});
  XLS_ASSERT_OK_AND_ASSIGN(auto package, Parser::ParsePackage(ir_text));
  EXPECT_THAT(MakeBlockAndRunPasses(package.get()),
              StatusIs(absl::StatusCode::kInternal,
                       HasSubstr("element must be type bits, got token")));
}

TEST(RamRewritePassInvalidInputsTest, WeMustBeWidth1) {
  // Replace we with bits[32] (must be bits[1])
  std::string ir_text = MakeTestProc(TestProcVars{
      .req_type = "(bits[32], bits[32], bits[32], bits[1])",
      .send_value = "tuple(__state, __state, __state, true_lit)",
  });
  XLS_ASSERT_OK_AND_ASSIGN(auto package, Parser::ParsePackage(ir_text));
  EXPECT_THAT(MakeBlockAndRunPasses(package.get()),
              StatusIs(absl::StatusCode::kInternal,
                       HasSubstr("Request we element must have width 1")));
}

TEST(RamRewritePassInvalidInputsTest, ReMustBeWidth1) {
  // Replace re with bits[32] (must be bits[1])
  std::string ir_text = MakeTestProc(TestProcVars{
      .req_type = "(bits[32], bits[32], bits[1], bits[32])",
      .send_value = "tuple(__state, __state, true_lit, __state)",
  });
  XLS_ASSERT_OK_AND_ASSIGN(auto package, Parser::ParsePackage(ir_text));
  EXPECT_THAT(MakeBlockAndRunPasses(package.get()),
              StatusIs(absl::StatusCode::kInternal,
                       HasSubstr("Request re element must have width 1")));
}

TEST(RamRewritePassInvalidInputsTest, RespDataWidthMustMatchReqDataWidth) {
  // Try a 64-bit response (must match 32-bit request data width)
  std::string ir_text = MakeTestProc(TestProcVars{
      .resp_type = "(bits[64])",
  });
  XLS_ASSERT_OK_AND_ASSIGN(auto package, Parser::ParsePackage(ir_text));
  EXPECT_THAT(
      MakeBlockAndRunPasses(package.get()),
      StatusIs(absl::StatusCode::kInternal,
               HasSubstr("Response rd_data element (width=64) must have the "
                         "same width as request wr_data element (width=32)")));
}

}  // namespace
}  // namespace verilog
}  // namespace xls
