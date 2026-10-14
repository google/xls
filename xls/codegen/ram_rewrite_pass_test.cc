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
#include <variant>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_replace.h"
#include "absl/types/span.h"
#include "absl/types/variant.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "xls/codegen/block_conversion.h"
#include "xls/codegen/codegen_options.h"
#include "xls/codegen/codegen_pass.h"
#include "xls/codegen/codegen_pass_pipeline.h"
#include "xls/codegen/module_signature.h"
#include "xls/codegen/module_signature.pb.h"
#include "xls/codegen/port_legalization_pass.h"
#include "xls/codegen/ram_configuration.h"
#include "xls/codegen_v_1_5/codegen.h"
#include "xls/common/file/filesystem.h"
#include "xls/common/proto_test_utils.h"
#include "xls/common/status/matchers.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/common/undeclared_outputs.h"
#include "xls/common/visitor.h"
#include "xls/estimators/delay_model/delay_estimators.h"
#include "xls/interpreter/block_interpreter.h"
#include "xls/ir/bits.h"
#include "xls/ir/block.h"
#include "xls/ir/channel.h"
#include "xls/ir/ir_matcher.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/nodes.h"
#include "xls/ir/package.h"
#include "xls/ir/proc.h"
#include "xls/ir/value.h"
#include "xls/ir/verifier.h"
#include "xls/passes/optimization_pass.h"
#include "xls/passes/pass_base.h"
#include "xls/scheduling/pipeline_schedule.h"
#include "xls/scheduling/run_pipeline_schedule.h"
#include "xls/scheduling/scheduling_options.h"

namespace xls {
namespace verilog {
namespace {

namespace m = xls::op_matchers;

using ::absl_testing::StatusIs;
using ::testing::AllOf;
using ::testing::AnyOf;
using ::testing::Contains;
using ::testing::Eq;
using ::testing::HasSubstr;
using ::testing::IsSupersetOf;
using ::testing::Not;
using ::xls::proto_testing::EqualsProto;

class PortByNameMatcher : public ::testing::MatcherInterface<Block::Port> {
 public:
  explicit PortByNameMatcher(std::string_view port_name)
      : port_name_(port_name) {}

  bool MatchAndExplain(
      const Block::Port port,
      ::testing::MatchResultListener* listener) const override {
    return absl::visit(
        Visitor{
            [=, this](ClockPort* p) {
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
  std::string port_name_;
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
  std::string port_name_;
};

inline ::testing::Matcher<::xls::verilog::PortProto> PortProtoByName(
    std::string_view port_name) {
  return ::testing::MakeMatcher(new PortProtoByNameMatcher(port_name));
}

enum class CodegenPassType {
  kDefault,
  kRamRewritePassOnly,
};

std::unique_ptr<CodegenPass> GetCodegenPass(CodegenPassType type,
                                            OptimizationContext& context) {
  switch (type) {
    case CodegenPassType::kDefault:
      return CreateCodegenPassPipeline(context);
    case CodegenPassType::kRamRewritePassOnly:
      return std::make_unique<RamRewritePass>();
  }
}

std::string_view CodegenPassName(CodegenPassType type) {
  switch (type) {
    case CodegenPassType::kDefault:
      return "DefaultCodegenPassPipeline";
    case CodegenPassType::kRamRewritePassOnly:
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

template <typename Sink>
void AbslStringify(Sink& sink, RamChannelRewriteTestParam param) {
  absl::Format(&sink, "RamChannelRewriteTestParam {\n");
  absl::Format(&sink, "  test_name = %s\n", param.test_name);
  absl::Format(&sink, "  pipeline_stages = %d\n", param.pipeline_stages);
  absl::Format(&sink, "  ram_config_strings = {%s}\n",
               absl::StrJoin(param.ram_config_strings, ", "));
  absl::Format(&sink, "  expect_read_mask = %d\n", param.expect_read_mask);
  absl::Format(&sink, "  expect_write_mask = %d\n", param.expect_write_mask);
  absl::Format(&sink, "  ram_contents = {%s}\n",
               absl::StrJoin(param.ram_contents, ", "));
  absl::Format(&sink, "}");
}

class RamRewritePassTest
    : public testing::TestWithParam<
          std::tuple<RamChannelRewriteTestParam, CodegenPassType>> {
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
    std::vector<RamConfiguration> ram_configurations;
    ram_configurations.reserve(param.ram_config_strings.size());
    for (std::string_view config_string : param.ram_config_strings) {
      RamConfiguration config = ParseRamConfiguration(config_string).value();
      ram_configurations.push_back(std::move(config));
    }
    codegen_options.ram_configurations(ram_configurations);
    return codegen_options;
  }

  bool ExpectReadMask() const {
    auto& param = std::get<0>(GetParam());
    auto codegen_pass_type = std::get<1>(GetParam());
    // If we're only running the ram rewrite pass, even masks with type () will
    // not be removed. They should always exist.
    return param.expect_read_mask ||
           codegen_pass_type == CodegenPassType::kRamRewritePassOnly;
  }

  bool ExpectWriteMask() const {
    auto& param = std::get<0>(GetParam());
    auto codegen_pass_type = std::get<1>(GetParam());
    // If we're only running the ram rewrite pass, even masks with type () will
    // not be removed. They should always exist.
    return param.expect_write_mask ||
           codegen_pass_type == CodegenPassType::kRamRewritePassOnly;
  }

  absl::StatusOr<CodegenContext> ScheduleAndBlockConvert(
      Package const* package, const CodegenOptions& codegen_options) {
    auto& param = std::get<0>(GetParam());

    XLS_ASSIGN_OR_RETURN(Proc * proc, package->GetProc("my_proc"));

    auto scheduling_options =
        SchedulingOptions().pipeline_stages(param.pipeline_stages);
    // Add constraints for each ram config to be scheduled according to the
    // config's latency
    for (const auto& ram_config : codegen_options.ram_configurations()) {
      if (std::holds_alternative<Ram1RWConfiguration>(ram_config)) {
        auto ram1rw_config = std::get<Ram1RWConfiguration>(ram_config);
        scheduling_options.add_constraint(IOConstraint(
            ram1rw_config.rw_port_configuration().request_channel_name,
            IODirection::kSend,
            ram1rw_config.rw_port_configuration().response_channel_name,
            IODirection::kReceive,
            /*minimum_latency=*/ram1rw_config.latency(),
            /*maximum_latency=*/ram1rw_config.latency()));
        scheduling_options.add_constraint(IOConstraint(
            ram1rw_config.rw_port_configuration().request_channel_name,
            IODirection::kSend,
            ram1rw_config.rw_port_configuration().write_completion_channel_name,
            IODirection::kReceive,
            /*minimum_latency=*/ram1rw_config.latency(),
            /*maximum_latency=*/ram1rw_config.latency()));
        continue;
      }
      if (std::holds_alternative<Ram1R1WConfiguration>(ram_config)) {
        auto ram1r1w_config = std::get<Ram1R1WConfiguration>(ram_config);
        scheduling_options.add_constraint(IOConstraint(
            ram1r1w_config.r_port_configuration().request_channel_name,
            IODirection::kSend,
            ram1r1w_config.r_port_configuration().response_channel_name,
            IODirection::kReceive,
            /*minimum_latency=*/ram1r1w_config.latency(),
            /*maximum_latency=*/ram1r1w_config.latency()));
        scheduling_options.add_constraint(IOConstraint(
            ram1r1w_config.w_port_configuration().request_channel_name,
            IODirection::kSend,
            ram1r1w_config.w_port_configuration().write_completion_channel_name,
            IODirection::kReceive,
            /*minimum_latency=*/ram1r1w_config.latency(),
            /*maximum_latency=*/ram1r1w_config.latency()));
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

TEST(RamRewriteHandshakeTest, Held1RWRequestIssuesOnceAfterResponseStall) {
  constexpr std::string_view kIr = R"(package test

top block held_valid_ram_request(
    clk: clock, rst: bits[1], source_addr: bits[4],
    source_valid: bits[1], source_write: bits[1], source_ready: bits[1],
    consumer_ready: bits[1], consumer_data: bits[32], consumer_valid: bits[1],
    req_data: (bits[4], bits[32], (), (), bits[1], bits[1]),
    req_valid: bits[1], req_ready: bits[1],
    resp_data: (bits[32]), resp_valid: bits[1], resp_ready: bits[1],
    wr_comp_data: (), wr_comp_valid: bits[1], wr_comp_ready: bits[1]) {
  #![reset(port="rst", asynchronous=false, active_low=false)]
  #![channel_ports(name=req, type=(bits[4], bits[32], (), (), bits[1], bits[1]), direction=send, kind=streaming, data_port=req_data, ready_port=req_ready, valid_port=req_valid)]
  #![channel_ports(name=resp, type=(bits[32]), direction=receive, kind=streaming, data_port=resp_data, ready_port=resp_ready, valid_port=resp_valid)]
  #![channel_ports(name=wr_comp, type=(), direction=receive, kind=streaming, data_port=wr_comp_data, ready_port=wr_comp_ready, valid_port=wr_comp_valid)]
  rst: bits[1] = input_port(name=rst)
  source_addr: bits[4] = input_port(name=source_addr)
  source_valid: bits[1] = input_port(name=source_valid)
  source_write: bits[1] = input_port(name=source_write)
  consumer_ready: bits[1] = input_port(name=consumer_ready)
  req_ready: bits[1] = input_port(name=req_ready)
  resp_data: (bits[32]) = input_port(name=resp_data)
  resp_valid: bits[1] = input_port(name=resp_valid)
  wr_comp_data: () = input_port(name=wr_comp_data)
  wr_comp_valid: bits[1] = input_port(name=wr_comp_valid)
  one1: bits[1] = literal(value=1)
  write_data: bits[32] = literal(value=0x55)
  empty: () = tuple()
  source_read: bits[1] = not(source_write)
  held_valid: bits[1] = identity(source_valid)
  req_tuple: (bits[4], bits[32], (), (), bits[1], bits[1]) = tuple(source_addr, write_data, empty, empty, source_write, source_read)
  rd_value: bits[32] = tuple_index(resp_data, index=0)
  req_data: () = output_port(req_tuple, name=req_data)
  req_valid: () = output_port(held_valid, name=req_valid)
  source_ready: () = output_port(req_ready, name=source_ready)
  consumer_data: () = output_port(rd_value, name=consumer_data)
  consumer_valid: () = output_port(resp_valid, name=consumer_valid)
  resp_ready: () = output_port(consumer_ready, name=resp_ready)
  wr_comp_ready: () = output_port(one1, name=wr_comp_ready)
}
)";

  XLS_ASSERT_OK_AND_ASSIGN(auto package, IrTestBase::ParsePackage(kIr));
  XLS_ASSERT_OK_AND_ASSIGN(Block * block,
                           package->GetBlock("held_valid_ram_request"));
  const CodegenOptions codegen_options =
      CodegenOptions()
          .reset("rst", false, false, false)
          .ram_configurations(
              {Ram1RWConfiguration("ram", 1, "req", "resp", "wr_comp")});
  const CodegenPassOptions pass_options{.codegen_options = codegen_options};
  CodegenContext context(block);
  PassResults results;
  XLS_ASSERT_OK_AND_ASSIGN(
      bool changed,
      RamRewritePass().Run(package.get(), pass_options, &results, context));
  EXPECT_TRUE(changed);
  // Legalize empty mask ports before using the integer-valued interpreter.
  XLS_ASSERT_OK(PortLegalizationPass()
                    .Run(package.get(), pass_options, &results, context)
                    .status());

  for (uint64_t second_is_write : {0, 1}) {
    SCOPED_TRACE(second_is_write);
    std::vector<absl::flat_hash_map<std::string, uint64_t>> inputs = {
        {{"rst", 1},
         {"source_addr", 0},
         {"source_valid", 0},
         {"source_write", 0},
         {"consumer_ready", 1},
         {"ram_rd_data", 0}},
        {{"rst", 0},
         {"source_addr", 1},
         {"source_valid", 1},
         {"source_write", 0},
         {"consumer_ready", 1},
         {"ram_rd_data", 0}},
        {{"rst", 0},
         {"source_addr", 2},
         {"source_valid", 1},
         {"source_write", second_is_write},
         {"consumer_ready", 0},
         {"ram_rd_data", 0xa1}},
        {{"rst", 0},
         {"source_addr", 2},
         {"source_valid", 1},
         {"source_write", second_is_write},
         {"consumer_ready", 0},
         {"ram_rd_data", 0}},
        {{"rst", 0},
         {"source_addr", 2},
         {"source_valid", 1},
         {"source_write", second_is_write},
         {"consumer_ready", 1},
         {"ram_rd_data", 0}},
        {{"rst", 0},
         {"source_addr", 0},
         {"source_valid", 0},
         {"source_write", 0},
         {"consumer_ready", 1},
         {"ram_rd_data", 0xb2}},
        {{"rst", 0},
         {"source_addr", 0},
         {"source_valid", 0},
         {"source_write", 0},
         {"consumer_ready", 1},
         {"ram_rd_data", 0}},
    };
    XLS_ASSERT_OK_AND_ASSIGN(auto outputs,
                             InterpretSequentialBlock(block, inputs));
    EXPECT_EQ(outputs[1].at("ram_re"), 1);
    for (int64_t cycle : {2, 3}) {
      EXPECT_EQ(outputs[cycle].at("source_ready"), 0);
      EXPECT_EQ(outputs[cycle].at("ram_re"), 0);
      EXPECT_EQ(outputs[cycle].at("ram_we"), 0);
      EXPECT_EQ(outputs[cycle].at("consumer_valid"), 1);
      EXPECT_EQ(outputs[cycle].at("consumer_data"), 0xa1);
    }
    EXPECT_EQ(outputs[4].at("source_ready"), 1);
    EXPECT_EQ(outputs[4].at("ram_re"), 1 - second_is_write);
    EXPECT_EQ(outputs[4].at("ram_we"), second_is_write);
    EXPECT_EQ(outputs[4].at("consumer_data"), 0xa1);
    EXPECT_EQ(outputs[5].at("consumer_valid"), 1 - second_is_write);
    if (second_is_write == 0) {
      EXPECT_EQ(outputs[5].at("consumer_data"), 0xb2);
    }
    EXPECT_EQ(outputs[6].at("consumer_valid"), 0);
    uint64_t read_count = 0;
    uint64_t write_count = 0;
    for (const auto& cycle : outputs) {
      read_count += cycle.at("ram_re");
      write_count += cycle.at("ram_we");
    }
    EXPECT_EQ(read_count, 2 - second_is_write);
    EXPECT_EQ(write_count, second_is_write);
  }
}

TEST(RamRewriteHandshakeTest, HeldReadIssuesOnceAfterResponseStall) {
  constexpr std::string_view kIr = R"(package test

top block held_valid_ram_request(
    clk: clock, rst: bits[1], source_addr: bits[4],
    source_valid: bits[1], source_ready: bits[1],
    consumer_ready: bits[1], consumer_data: bits[32],
    consumer_valid: bits[1], rd_req_data: (bits[4], ()),
    rd_req_valid: bits[1], rd_req_ready: bits[1],
    rd_resp_data: (bits[32]), rd_resp_valid: bits[1],
    rd_resp_ready: bits[1],
    wr_req_data: (bits[4], bits[32], ()), wr_req_valid: bits[1],
    wr_req_ready: bits[1], wr_comp_data: (),
    wr_comp_valid: bits[1], wr_comp_ready: bits[1]) {
  #![reset(port="rst", asynchronous=false, active_low=false)]
  #![channel_ports(name=rd_req, type=(bits[4], ()), direction=send, kind=streaming, data_port=rd_req_data, ready_port=rd_req_ready, valid_port=rd_req_valid)]
  #![channel_ports(name=rd_resp, type=(bits[32]), direction=receive, kind=streaming, data_port=rd_resp_data, ready_port=rd_resp_ready, valid_port=rd_resp_valid)]
  #![channel_ports(name=wr_req, type=(bits[4], bits[32], ()), direction=send, kind=streaming, data_port=wr_req_data, ready_port=wr_req_ready, valid_port=wr_req_valid)]
  #![channel_ports(name=wr_comp, type=(), direction=receive, kind=streaming, data_port=wr_comp_data, ready_port=wr_comp_ready, valid_port=wr_comp_valid)]
  rst: bits[1] = input_port(name=rst)
  source_addr: bits[4] = input_port(name=source_addr)
  source_valid: bits[1] = input_port(name=source_valid)
  consumer_ready: bits[1] = input_port(name=consumer_ready)
  rd_req_ready: bits[1] = input_port(name=rd_req_ready)
  rd_resp_data: (bits[32]) = input_port(name=rd_resp_data)
  rd_resp_valid: bits[1] = input_port(name=rd_resp_valid)
  wr_req_ready: bits[1] = input_port(name=wr_req_ready)
  wr_comp_data: () = input_port(name=wr_comp_data)
  wr_comp_valid: bits[1] = input_port(name=wr_comp_valid)
  zero1: bits[1] = literal(value=0)
  one1: bits[1] = literal(value=1)
  zero4: bits[4] = literal(value=0)
  zero32: bits[32] = literal(value=0)
  empty: () = tuple()
  held_valid: bits[1] = identity(source_valid)
  write_valid: bits[1] = identity(zero1)
  rd_tuple: (bits[4], ()) = tuple(source_addr, empty)
  wr_tuple: (bits[4], bits[32], ()) = tuple(zero4, zero32, empty)
  rd_value: bits[32] = tuple_index(rd_resp_data, index=0)
  rd_req_data: () = output_port(rd_tuple, name=rd_req_data)
  rd_req_valid: () = output_port(held_valid, name=rd_req_valid)
  source_ready: () = output_port(rd_req_ready, name=source_ready)
  consumer_data: () = output_port(rd_value, name=consumer_data)
  consumer_valid: () = output_port(rd_resp_valid, name=consumer_valid)
  rd_resp_ready: () = output_port(consumer_ready, name=rd_resp_ready)
  wr_req_data: () = output_port(wr_tuple, name=wr_req_data)
  wr_req_valid: () = output_port(write_valid, name=wr_req_valid)
  wr_comp_ready: () = output_port(one1, name=wr_comp_ready)
}
)";

  XLS_ASSERT_OK_AND_ASSIGN(auto package, IrTestBase::ParsePackage(kIr));
  XLS_ASSERT_OK_AND_ASSIGN(Block * block,
                           package->GetBlock("held_valid_ram_request"));

  std::vector<RamConfiguration> ram_configurations;
  ram_configurations.push_back(
      Ram1R1WConfiguration("ram", 1, "rd_req", "rd_resp", "wr_req", "wr_comp"));
  const CodegenOptions codegen_options =
      CodegenOptions()
          .reset("rst", false, false, false)
          .ram_configurations(ram_configurations);
  const CodegenPassOptions pass_options{.codegen_options = codegen_options};
  CodegenContext context(block);
  PassResults results;
  XLS_ASSERT_OK_AND_ASSIGN(
      bool changed,
      RamRewritePass().Run(package.get(), pass_options, &results, context));
  EXPECT_TRUE(changed);
  // Legalize empty mask ports before using the integer-valued interpreter.
  XLS_ASSERT_OK(PortLegalizationPass()
                    .Run(package.get(), pass_options, &results, context)
                    .status());

  std::vector<absl::flat_hash_map<std::string, uint64_t>> inputs = {
      {{"rst", 1},
       {"source_addr", 0},
       {"source_valid", 0},
       {"consumer_ready", 1},
       {"ram_rd_data", 0}},
      {{"rst", 0},
       {"source_addr", 1},
       {"source_valid", 1},
       {"consumer_ready", 1},
       {"ram_rd_data", 0}},
      {{"rst", 0},
       {"source_addr", 2},
       {"source_valid", 1},
       {"consumer_ready", 0},
       {"ram_rd_data", 0xa1}},
      {{"rst", 0},
       {"source_addr", 2},
       {"source_valid", 1},
       {"consumer_ready", 0},
       {"ram_rd_data", 0}},
      {{"rst", 0},
       {"source_addr", 2},
       {"source_valid", 1},
       {"consumer_ready", 1},
       {"ram_rd_data", 0}},
      {{"rst", 0},
       {"source_addr", 0},
       {"source_valid", 0},
       {"consumer_ready", 1},
       {"ram_rd_data", 0xb2}},
      {{"rst", 0},
       {"source_addr", 0},
       {"source_valid", 0},
       {"consumer_ready", 1},
       {"ram_rd_data", 0}},
  };
  XLS_ASSERT_OK_AND_ASSIGN(auto outputs,
                           InterpretSequentialBlock(block, inputs));

  // The second request remains valid throughout the response stall, but it
  // must issue to the physical RAM exactly once, when request ready returns.
  EXPECT_EQ(outputs[1].at("ram_rd_en"), 1);
  EXPECT_EQ(outputs[2].at("source_ready"), 0);
  EXPECT_EQ(outputs[2].at("ram_rd_en"), 0);
  EXPECT_EQ(outputs[3].at("source_ready"), 0);
  EXPECT_EQ(outputs[3].at("ram_rd_en"), 0);
  EXPECT_EQ(outputs[4].at("source_ready"), 1);
  EXPECT_EQ(outputs[4].at("ram_rd_en"), 1);
  int64_t read_count = 0;
  for (const auto& cycle_outputs : outputs) {
    read_count += cycle_outputs.at("ram_rd_en");
  }
  EXPECT_EQ(read_count, 2);

  EXPECT_EQ(outputs[2].at("consumer_valid"), 1);
  EXPECT_EQ(outputs[2].at("consumer_data"), 0xa1);
  EXPECT_EQ(outputs[3].at("consumer_valid"), 1);
  EXPECT_EQ(outputs[3].at("consumer_data"), 0xa1);
  EXPECT_EQ(outputs[4].at("consumer_valid"), 1);
  EXPECT_EQ(outputs[4].at("consumer_data"), 0xa1);
  EXPECT_EQ(outputs[5].at("consumer_valid"), 1);
  EXPECT_EQ(outputs[5].at("consumer_data"), 0xb2);
}

class GeneratedRamStallTest : public testing::TestWithParam<bool> {};

TEST_P(GeneratedRamStallTest, PreservesTransactionsDuringOutputStalls) {
  const bool single_port = GetParam();
  const std::string_view channels = single_port ? R"(
chan req((bits[4], bits[32], (), (), bits[1], bits[1]), id=4, kind=streaming, ops=send_only, flow_control=ready_valid)
)"
                                                : R"(
chan rd_req((bits[4], ()), id=4, kind=streaming, ops=send_only, flow_control=ready_valid)
chan wr_req((bits[4], bits[32], ()), id=5, kind=streaming, ops=send_only, flow_control=ready_valid)
)";
  const std::string_view requests = single_port ? R"(
  request: (bits[4], bits[32], (), (), bits[1], bits[1]) = tuple(addr, write_data, empty, empty, write, read)
  read_token: token = send(command_token, request, channel=req)
  write_token: token = identity(read_token)
)"
                                                : R"(
  read_request: (bits[4], ()) = tuple(addr, empty)
  write_request: (bits[4], bits[32], ()) = tuple(addr, write_data, empty)
  read_token: token = send(command_token, read_request, predicate=read, channel=rd_req)
  write_token: token = send(command_token, write_request, predicate=write, channel=wr_req)
)";
  constexpr std::string_view kProcTemplate = R"(package test
chan cmd(bits[5], id=0, kind=streaming, ops=receive_only, flow_control=ready_valid)
chan out(bits[32], id=1, kind=streaming, ops=send_only, flow_control=ready_valid)
chan resp((bits[32]), id=2, kind=streaming, ops=receive_only, flow_control=ready_valid)
chan wr_comp((), id=3, kind=streaming, ops=receive_only, flow_control=ready_valid)
$CHANNELS

top proc reader(state: (), init={()}) {
  t: token = literal(value=token)
  command: (token, bits[5]) = receive(t, channel=cmd)
  command_token: token = tuple_index(command, index=0)
  command_data: bits[5] = tuple_index(command, index=1)
  addr: bits[4] = bit_slice(command_data, start=0, width=4)
  write: bits[1] = bit_slice(command_data, start=4, width=1)
  read: bits[1] = not(write)
  write_data: bits[32] = literal(value=85)
  empty: () = tuple()
$REQUESTS
  response: (token, (bits[32])) = receive(read_token, predicate=read, channel=resp)
  response_token: token = tuple_index(response, index=0)
  response_tuple: (bits[32]) = tuple_index(response, index=1)
  response_data: bits[32] = tuple_index(response_tuple, index=0)
  completion: (token, ()) = receive(write_token, predicate=write, channel=wr_comp)
  completion_token: token = tuple_index(completion, index=0)
  done: token = after_all(response_token, completion_token)
  output_token: token = send(done, response_data, predicate=read, channel=out)
  next_state: () = next_value(state_element=state, value=state)
}
)";
  const std::string ir = absl::StrReplaceAll(
      kProcTemplate, {{"$CHANNELS", channels}, {"$REQUESTS", requests}});
  XLS_ASSERT_OK_AND_ASSIGN(auto package, IrTestBase::ParsePackage(ir));
  XLS_ASSERT_OK_AND_ASSIGN(
      auto ram_config,
      ParseRamConfiguration(single_port
                                ? "ram:1RW:req:resp:wr_comp"
                                : "ram:1R1W:rd_req:resp:wr_req:wr_comp"));
  CodegenOptions options;
  options.module_name("ram_stall")
      .clock_name("clk")
      .reset("rst", false, false, false)
      .flop_inputs(false)
      .flop_outputs(false)
      .streaming_channel_data_suffix("_data")
      .streaming_channel_valid_suffix("_valid")
      .streaming_channel_ready_suffix("_ready")
      .ram_configurations({ram_config});
  SchedulingOptions scheduling;
  scheduling.pipeline_stages(2);
  for (const IOConstraint& constraint :
       GetRamConfigurationIOConstraints(ram_config)) {
    scheduling.add_constraint(constraint);
  }
  XLS_ASSERT_OK_AND_ASSIGN(auto delay_estimator, GetDelayEstimator("unit"));
  XLS_ASSERT_OK_AND_ASSIGN(
      auto result, xls::codegen::Codegen(package.get(), options, scheduling,
                                         delay_estimator));
  if (auto output_dir = GetUndeclaredOutputDirectory();
      output_dir.has_value()) {
    XLS_ASSERT_OK(SetFileContents(
        *output_dir / (single_port ? "ram_1rw.v" : "ram_1r1w.v"),
        result.verilog_text));
  }
  XLS_ASSERT_OK_AND_ASSIGN(Block * block, package->GetTopAsBlock());

  // Commands are held until accepted. A write to an otherwise unread address
  // avoids depending on the RAM's simultaneous read/write collision policy.
  const std::vector<uint64_t> commands = {1, 2, 0x17, 3, 4, 5, 6};
  const std::vector<uint64_t> expected_reads = {1, 2, 3, 4, 5, 6};
  const std::vector<uint64_t> expected_responses = {0xa1, 0xa2, 0xa3,
                                                    0xa4, 0xa5, 0xa6};
  // The final case is an always-ready control: its stall is beyond the run.
  for (int64_t stall_start : {2, 3, 4, 5, 6, 7, 128}) {
    SCOPED_TRACE(stall_start);
    XLS_ASSERT_OK_AND_ASSIGN(auto continuation,
                             kInterpreterBlockEvaluator.NewContinuation(block));
    std::vector<uint64_t> memory(16);
    for (int64_t address = 0; address < memory.size(); ++address) {
      memory[address] = 0xa0 + address;
    }
    int64_t sent = 0;
    uint64_t response = 0;
    std::vector<uint64_t> physical_reads;
    std::vector<uint64_t> physical_writes;
    std::vector<uint64_t> received;
    // Run well past the final expected transfer to detect extra responses too.
    for (int64_t cycle = 0; cycle < 96; ++cycle) {
      const bool reset = cycle == 0;
      const bool valid = !reset && sent < commands.size();
      const bool ready = cycle < stall_start || cycle >= stall_start + 16;
      XLS_ASSERT_OK(continuation->RunOneCycle({
          {"rst", Value(UBits(reset, 1))},
          {"cmd_data", Value(UBits(valid ? commands[sent] : 0, 5))},
          {"cmd_valid", Value(UBits(valid, 1))},
          {"out_ready", Value(UBits(ready, 1))},
          {"ram_rd_data", Value(UBits(response, 32))},
      }));
      auto output = [&](std::string_view name) {
        return continuation->output_ports().at(name).bits().ToUint64().value();
      };
      if (reset) {
        continue;
      }
      if (valid && output("cmd_ready")) {
        ++sent;
      }
      if (ready && output("out_valid")) {
        received.push_back(output("out_data"));
      }
      // The external RAM returns data exactly one cycle after a physical read.
      // Poison idle cycles so a spurious response-valid cannot pass unnoticed.
      response = 0xdeadbeef;
      if (output(single_port ? "ram_re" : "ram_rd_en")) {
        const uint64_t address =
            output(single_port ? "ram_addr" : "ram_rd_addr");
        physical_reads.push_back(address);
        response = memory.at(address);
      }
      if (output(single_port ? "ram_we" : "ram_wr_en")) {
        const uint64_t address =
            output(single_port ? "ram_addr" : "ram_wr_addr");
        physical_writes.push_back(address);
        memory.at(address) = output("ram_wr_data");
      }
    }
    EXPECT_EQ(sent, commands.size());
    EXPECT_EQ(physical_reads, expected_reads);
    EXPECT_THAT(physical_writes, testing::ElementsAre(7));
    EXPECT_EQ(memory[7], 0x55);
    EXPECT_EQ(received, expected_responses);
  }
}

INSTANTIATE_TEST_SUITE_P(RamKinds, GeneratedRamStallTest,
                         testing::Values(true, false),
                         [](const testing::TestParamInfo<bool>& info) {
                           return info.param ? "SinglePort" : "SeparatePorts";
                         });

TEST_P(RamRewritePassTest, PortsUpdated) {
  auto& param = std::get<0>(GetParam());
  const CodegenOptions codegen_options = GetCodegenOptions();
  const CodegenPassOptions pass_options{
      .codegen_options = codegen_options,
  };

  XLS_ASSERT_OK_AND_ASSIGN(auto package,
                           IrTestBase::ParsePackage(param.ir_text));
  XLS_ASSERT_OK_AND_ASSIGN(
      CodegenContext context,
      ScheduleAndBlockConvert(package.get(), codegen_options));
  PassResults results;
  OptimizationContext opt_context;
  auto pipeline = GetCodegenPass(std::get<1>(GetParam()), opt_context);
  XLS_ASSERT_OK_AND_ASSIGN(
      bool changed,
      pipeline->Run(package.get(), pass_options, &results, context));

  EXPECT_TRUE(changed);

  for (const auto& config : codegen_options.ram_configurations()) {
    std::vector<std::string> old_channel_names;
    if (std::holds_alternative<Ram1RWConfiguration>(config)) {
      auto ram1rw_config = std::get<Ram1RWConfiguration>(config);
      EXPECT_THAT(context.top_block()->GetPorts(),
                  AllOf(Contains(PortByName(absl::StrFormat(
                            "%s_addr", ram1rw_config.ram_name()))),
                        Contains(PortByName(absl::StrFormat(
                            "%s_rd_data", ram1rw_config.ram_name()))),
                        Contains(PortByName(absl::StrFormat(
                            "%s_re", ram1rw_config.ram_name()))),
                        Contains(PortByName(absl::StrFormat(
                            "%s_wr_data", ram1rw_config.ram_name()))),
                        Contains(PortByName(absl::StrFormat(
                            "%s_we", ram1rw_config.ram_name())))));
      XLS_ASSERT_OK_AND_ASSIGN(
          InputPort * rd_input,
          context.top_block()->GetInputPort(
              absl::StrFormat("%s_rd_data", ram1rw_config.ram_name())));
      EXPECT_THAT(rd_input->GetType(),
                  m::Type(TypeOfRam(ram1rw_config.ram_name())));
      XLS_ASSERT_OK_AND_ASSIGN(
          OutputPort * wr_output,
          context.top_block()->GetOutputPort(
              absl::StrFormat("%s_wr_data", ram1rw_config.ram_name())));
      EXPECT_THAT(wr_output->operand(0)->GetType(),
                  m::Type(TypeOfRam(ram1rw_config.ram_name())));
      if (ExpectReadMask()) {
        EXPECT_THAT(context.top_block()->GetPorts(),
                    Contains(PortByName(absl::StrFormat(
                        "%s_rd_mask", ram1rw_config.ram_name()))));
      } else {
        EXPECT_THAT(context.top_block()->GetPorts(),
                    Not(Contains(PortByName(absl::StrFormat(
                        "%s_rd_mask", ram1rw_config.ram_name())))));
      }
      if (ExpectWriteMask()) {
        EXPECT_THAT(context.top_block()->GetPorts(),
                    Contains(PortByName(absl::StrFormat(
                        "%s_wr_mask", ram1rw_config.ram_name()))));
      } else {
        EXPECT_THAT(context.top_block()->GetPorts(),
                    Not(Contains(PortByName(absl::StrFormat(
                        "%s_wr_mask", ram1rw_config.ram_name())))));
      }

      old_channel_names.push_back(
          ram1rw_config.rw_port_configuration().request_channel_name);
      old_channel_names.push_back(
          ram1rw_config.rw_port_configuration().response_channel_name);
    } else if (std::holds_alternative<Ram1R1WConfiguration>(config)) {
      auto ram1r1w_config = std::get<Ram1R1WConfiguration>(config);
      EXPECT_THAT(context.top_block()->GetPorts(),
                  AllOf(Contains(PortByName(absl::StrFormat(
                            "%s_rd_en", ram1r1w_config.ram_name()))),
                        Contains(PortByName(absl::StrFormat(
                            "%s_rd_addr", ram1r1w_config.ram_name()))),
                        Contains(PortByName(absl::StrFormat(
                            "%s_rd_data", ram1r1w_config.ram_name()))),
                        Contains(PortByName(absl::StrFormat(
                            "%s_wr_en", ram1r1w_config.ram_name()))),
                        Contains(PortByName(absl::StrFormat(
                            "%s_wr_addr", ram1r1w_config.ram_name()))),
                        Contains(PortByName(absl::StrFormat(
                            "%s_wr_data", ram1r1w_config.ram_name())))));
      if (ExpectReadMask()) {
        EXPECT_THAT(context.top_block()->GetPorts(),
                    Contains(PortByName(absl::StrFormat(
                        "%s_rd_mask", ram1r1w_config.ram_name()))));
      } else {
        EXPECT_THAT(context.top_block()->GetPorts(),
                    Not(Contains(PortByName(absl::StrFormat(
                        "%s_rd_mask", ram1r1w_config.ram_name())))));
      }
      if (ExpectWriteMask()) {
        EXPECT_THAT(context.top_block()->GetPorts(),
                    Contains(PortByName(absl::StrFormat(
                        "%s_wr_mask", ram1r1w_config.ram_name()))));
      } else {
        EXPECT_THAT(context.top_block()->GetPorts(),
                    Not(Contains(PortByName(absl::StrFormat(
                        "%s_wr_mask", ram1r1w_config.ram_name())))));
      }
      old_channel_names.push_back(
          ram1r1w_config.r_port_configuration().request_channel_name);
      old_channel_names.push_back(
          ram1r1w_config.r_port_configuration().response_channel_name);
      old_channel_names.push_back(
          ram1r1w_config.w_port_configuration().request_channel_name);
    }
    for (const auto& old_channel_name : old_channel_names) {
      EXPECT_THAT(
          context.top_block()->GetPorts(),
          Not(AnyOf(
              Contains(PortByName(absl::StrCat(old_channel_name, "_valid"))),
              Contains(PortByName(absl::StrCat(old_channel_name, "_data"))),
              Contains(PortByName(absl::StrCat(old_channel_name, "_ready"))))));
    }
    EXPECT_THAT(context.top_block()->GetPorts(),
                Not(AnyOf(Contains(PortByName(absl::StrFormat(
                              "%s_valid", RamConfigurationRamName(config)))),
                          Contains(PortByName(absl::StrFormat(
                              "%s_data", RamConfigurationRamName(config)))),
                          Contains(PortByName(absl::StrFormat(
                              "%s_ready", RamConfigurationRamName(config)))))));
  }
}

TEST_P(RamRewritePassTest, ModuleSignatureUpdated) {
  // Module signature is generated by other codegen passes, only run this test
  // if we're running the full pass pipeline.
  if (std::get<1>(GetParam()) != CodegenPassType::kDefault) {
    GTEST_SKIP();
  }

  auto& param = std::get<0>(GetParam());
  const CodegenOptions codegen_options = GetCodegenOptions();
  const CodegenPassOptions pass_options{
      .codegen_options = codegen_options,
  };

  XLS_ASSERT_OK_AND_ASSIGN(auto package,
                           IrTestBase::ParsePackage(param.ir_text));
  XLS_ASSERT_OK_AND_ASSIGN(
      CodegenContext context,
      ScheduleAndBlockConvert(package.get(), codegen_options));
  PassResults results;
  OptimizationContext opt_context;
  auto pipeline = GetCodegenPass(std::get<1>(GetParam()), opt_context);
  XLS_ASSERT_OK_AND_ASSIGN(
      bool changed,
      pipeline->Run(package.get(), pass_options, &results, context));

  EXPECT_TRUE(changed);

  ASSERT_TRUE(context.HasMetadataForBlock(context.top_block()) &&
              context.top_block()->GetSignature().has_value());
  XLS_ASSERT_OK_AND_ASSIGN(
      ModuleSignature signature,
      ModuleSignature::FromProto(*context.top_block()->GetSignature()));
  for (const auto& config : codegen_options.ram_configurations()) {
    absl::flat_hash_set<std::string> channel_names;
    if (std::holds_alternative<Ram1RWConfiguration>(config)) {
      auto ram1rw_config = std::get<Ram1RWConfiguration>(config);
      bool found = false;
      for (auto& ram : signature.rams()) {
        if (ram.ram_oneof_case() != RamProto::RamOneofCase::kRam1Rw) {
          continue;
        }
        if (ram.ram_1rw().rw_port().request().name() ==
                ram1rw_config.rw_port_configuration().request_channel_name &&
            ram.ram_1rw().rw_port().response().name() ==
                ram1rw_config.rw_port_configuration().response_channel_name) {
          found = true;
        }
        // Check the port information matches.
        EXPECT_THAT(
            signature.proto().data_ports(),
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
          ram1rw_config.rw_port_configuration().request_channel_name);
      channel_names.insert(
          ram1rw_config.rw_port_configuration().response_channel_name);
      channel_names.insert(
          ram1rw_config.rw_port_configuration().write_completion_channel_name);
      XLS_ASSERT_OK_AND_ASSIGN(
          Channel * req_channel,
          package->GetChannel(
              ram1rw_config.rw_port_configuration().request_channel_name));
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
      EXPECT_THAT(signature.data_outputs(),
                  AllOf(Contains(PortProtoByName(
                            absl::StrCat(ram1rw_config.ram_name(), "_addr"))),
                        Contains(PortProtoByName(absl::StrCat(
                            ram1rw_config.ram_name(), "_wr_data"))),
                        Contains(PortProtoByName(
                            absl::StrCat(ram1rw_config.ram_name(), "_we"))),
                        Contains(PortProtoByName(
                            absl::StrCat(ram1rw_config.ram_name(), "_re"))),
                        Contains(PortProtoByName(absl::StrCat(
                                     ram1rw_config.ram_name(), "_wr_mask")))
                            .Times(write_mask_times),
                        Contains(PortProtoByName(absl::StrCat(
                                     ram1rw_config.ram_name(), "_rd_mask")))
                            .Times(read_mask_times)));
      EXPECT_THAT(signature.data_inputs(),
                  Contains(PortProtoByName(
                      absl::StrCat(ram1rw_config.ram_name(), "_rd_data"))));
    } else if (std::holds_alternative<Ram1R1WConfiguration>(config)) {
      auto ram1r1w_config = std::get<Ram1R1WConfiguration>(config);
      bool found = false;
      for (auto& ram : signature.rams()) {
        if (ram.ram_oneof_case() != RamProto::RamOneofCase::kRam1R1W) {
          continue;
        }
        if (ram.ram_1r1w().r_port().request().name() ==
                ram1r1w_config.r_port_configuration().request_channel_name &&
            ram.ram_1r1w().r_port().response().name() ==
                ram1r1w_config.r_port_configuration().response_channel_name) {
          found = true;
        }
        // Check the port information matches.
        EXPECT_THAT(
            signature.proto().data_ports(),
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
          ram1r1w_config.r_port_configuration().request_channel_name);
      channel_names.insert(
          ram1r1w_config.r_port_configuration().response_channel_name);
      channel_names.insert(
          ram1r1w_config.w_port_configuration().request_channel_name);
      channel_names.insert(
          ram1r1w_config.w_port_configuration().write_completion_channel_name);
      // If a mask is an empty tuple, expect to see it 0 times (i.e. be absent)
      // in the signature, otherwise expect it present once.
      XLS_ASSERT_OK_AND_ASSIGN(
          Channel * r_channel,
          package->GetChannel(
              ram1r1w_config.r_port_configuration().request_channel_name));
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
              ram1r1w_config.w_port_configuration().request_channel_name));
      // (wr_addr, wr_data, wr_mask) -> mask_idx = 2
      // Empty tuple means no write mask.
      int write_mask_times =
          w_channel->type()->AsTupleOrDie()->element_type(2)->IsEqualTo(
              package->GetTupleType({}))
              ? 0
              : 1;
      EXPECT_THAT(signature.data_outputs(),
                  AllOf(Contains(PortProtoByName(absl::StrCat(
                            ram1r1w_config.ram_name(), "_rd_addr"))),
                        Contains(PortProtoByName(absl::StrCat(
                            ram1r1w_config.ram_name(), "_wr_addr"))),
                        Contains(PortProtoByName(absl::StrCat(
                            ram1r1w_config.ram_name(), "_wr_data"))),
                        Contains(PortProtoByName(
                            absl::StrCat(ram1r1w_config.ram_name(), "_wr_en"))),
                        Contains(PortProtoByName(
                            absl::StrCat(ram1r1w_config.ram_name(), "_rd_en"))),
                        Contains(PortProtoByName(absl::StrCat(
                                     ram1r1w_config.ram_name(), "_rd_mask")))
                            .Times(read_mask_times),
                        Contains(PortProtoByName(absl::StrCat(
                                     ram1r1w_config.ram_name(), "_wr_mask")))
                            .Times(write_mask_times)));
      EXPECT_THAT(signature.data_inputs(),
                  Contains(PortProtoByName(
                      absl::StrCat(ram1r1w_config.ram_name(), "_rd_data"))));
    }
    for (auto& channel : signature.GetChannelInterfaces()) {
      EXPECT_THAT(channel_names, Not(Contains(Eq(channel.channel_name()))));
    }
    for (auto& channel_name : channel_names) {
      EXPECT_THAT(signature.data_inputs(),
                  Not(Contains(AnyOf(
                      PortProtoByName(channel_name),
                      PortProtoByName(absl::StrCat(channel_name, "_valid")),
                      PortProtoByName(absl::StrCat(channel_name, "_ready"))))));
      EXPECT_THAT(signature.data_outputs(),
                  Not(Contains(AnyOf(
                      PortProtoByName(channel_name),
                      PortProtoByName(absl::StrCat(channel_name, "_valid")),
                      PortProtoByName(absl::StrCat(channel_name, "_ready"))))));
    }
  }
}

TEST_P(RamRewritePassTest, WriteCompletionRemoved) {
  auto& param = std::get<0>(GetParam());
  const CodegenOptions codegen_options = GetCodegenOptions();
  const CodegenPassOptions pass_options{
      .codegen_options = codegen_options,
  };

  XLS_ASSERT_OK_AND_ASSIGN(auto package,
                           IrTestBase::ParsePackage(param.ir_text));
  XLS_ASSERT_OK_AND_ASSIGN(
      CodegenContext context,
      ScheduleAndBlockConvert(package.get(), codegen_options));
  PassResults results;
  OptimizationContext opt_context;
  auto pipeline = GetCodegenPass(std::get<1>(GetParam()), opt_context);
  XLS_ASSERT_OK_AND_ASSIGN(
      bool changed,
      pipeline->Run(package.get(), pass_options, &results, context));

  EXPECT_TRUE(changed);

  for (const auto& config : codegen_options.ram_configurations()) {
    if (std::holds_alternative<Ram1RWConfiguration>(config)) {
      auto ram1rw_config = std::get<Ram1RWConfiguration>(config);
      std::string_view wr_comp_name =
          ram1rw_config.rw_port_configuration().write_completion_channel_name;
      EXPECT_THAT(context.top_block()->GetPorts(),
                  Not(Contains(AnyOf(
                      PortByName(absl::StrCat(wr_comp_name, "_ready")),
                      PortByName(wr_comp_name),
                      PortByName(absl::StrCat(wr_comp_name, "_valid"))))));
    } else if (std::holds_alternative<Ram1R1WConfiguration>(config)) {
      auto ram1r1w_config = std::get<Ram1R1WConfiguration>(config);
      std::string_view wr_comp_name =
          ram1r1w_config.w_port_configuration().write_completion_channel_name;
      EXPECT_THAT(context.top_block()->GetPorts(),
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
chan req((bits[32], bits[32], (), (), bits[1], bits[1]), id=0, kind=streaming, ops=send_only, flow_control=ready_valid)
chan resp((bits[32]), id=1, kind=streaming, ops=receive_only, flow_control=ready_valid)
chan wr_comp((), id=2, kind=streaming, ops=receive_only, flow_control=ready_valid)

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
  next_state_value: () = next_value(state_element=__state, value=next_state)
}
  )",
        .pipeline_stages = 2,
        .ram_config_strings = kSingle1RW,
        .ram_contents = k1Ram32Bit,
    },
    RamChannelRewriteTestParam{
        .test_name = "Simple32Bit1RWWithMask",
        .ir_text = R"(package  test
chan req((bits[32], bits[32], bits[4], bits[4], bits[1], bits[1]), id=0, kind=streaming, ops=send_only, flow_control=ready_valid)
chan resp((bits[32]), id=1, kind=streaming, ops=receive_only, flow_control=ready_valid)
chan wr_comp((), id=2, kind=streaming, ops=receive_only, flow_control=ready_valid)

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
  next_state_value: () = next_value(state_element=__state, value=next_state)
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
chan req((bits[32], bits[32], (), (), bits[1], bits[1]), id=3, kind=streaming, ops=send_only, flow_control=ready_valid)
chan resp((bits[32]), id=1, kind=streaming, ops=receive_only, flow_control=ready_valid)
chan wr_comp((), id=4, kind=streaming, ops=receive_only, flow_control=ready_valid)
chan extra0(bits[1], id=0, kind=streaming, ops=receive_only, flow_control=ready_valid)
chan extra1(bits[1], id=2, kind=streaming, ops=receive_only, flow_control=ready_valid)

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
  next_state_value: () = next_value(state_element=__state, value=next_state)
}
  )",
        .pipeline_stages = 4,
        .ram_config_strings = kSingle1RW,
        .ram_contents = k1Ram32Bit,
    },
    RamChannelRewriteTestParam{
        .test_name = "32BitWithThree1RWRams",
        .ir_text = R"(package  test
chan req0((bits[32], bits[32], (), (), bits[1], bits[1]), id=0, kind=streaming, ops=send_only, flow_control=ready_valid)
chan resp0((bits[32]), id=1, kind=streaming, ops=receive_only, flow_control=ready_valid)
chan wr_comp0((), id=6, kind=streaming, ops=receive_only, flow_control=ready_valid)
chan req1((bits[32], bits[32], (), (), bits[1], bits[1]), id=2, kind=streaming, ops=send_only, flow_control=ready_valid)
chan resp1((bits[32]), id=3, kind=streaming, ops=receive_only, flow_control=ready_valid)
chan wr_comp1((), id=7, kind=streaming, ops=receive_only, flow_control=ready_valid)
chan req2((bits[32], bits[32], (), (), bits[1], bits[1]), id=4, kind=streaming, ops=send_only, flow_control=ready_valid)
chan resp2((bits[32]), id=5, kind=streaming, ops=receive_only, flow_control=ready_valid)
chan wr_comp2((), id=8, kind=streaming, ops=receive_only, flow_control=ready_valid)

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
  next_state_value: () = next_value(state_element=__state, value=next_state)
}
  )",
        .pipeline_stages = 20,
        .ram_config_strings = kThree1RW,
        .ram_contents = k3Ram32Bit,
    },
    RamChannelRewriteTestParam{
        .test_name = "Simple32Bit1R1W",
        .ir_text = R"(package  test
chan rd_req((bits[32], ()), id=0, kind=streaming, ops=send_only, flow_control=ready_valid)
chan rd_resp((bits[32]), id=1, kind=streaming, ops=receive_only, flow_control=ready_valid)
chan wr_req((bits[32], bits[32], ()), id=2, kind=streaming, ops=send_only, flow_control=ready_valid)
chan wr_comp((), id=3, kind=streaming, ops=receive_only, flow_control=ready_valid)

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
  next_state_value: () = next_value(state_element=__state, value=next_state)
}
  )",
        .pipeline_stages = 3,
        .ram_config_strings = kSingle1R1W,
        .ram_contents = k1Ram32Bit,
    },
    RamChannelRewriteTestParam{
        .test_name = "Simple32Bit1R1WWithMask",
        .ir_text = R"(package  test
chan rd_req((bits[32], bits[4]), id=0, kind=streaming, ops=send_only, flow_control=ready_valid)
chan rd_resp((bits[32]), id=1, kind=streaming, ops=receive_only, flow_control=ready_valid)
chan wr_req((bits[32], bits[32], bits[4]), id=2, kind=streaming, ops=send_only, flow_control=ready_valid)
chan wr_comp((), id=3, kind=streaming, ops=receive_only, flow_control=ready_valid)

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
  next_state_value: () = next_value(state_element=__state, value=next_state)
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
chan rd_req((bits[32], ()), id=0, kind=streaming, ops=send_only, flow_control=ready_valid)
chan rd_resp((bits[32]), id=1, kind=streaming, ops=receive_only, flow_control=ready_valid)
chan wr_req((bits[32], bits[32], ()), id=2, kind=streaming, ops=send_only, flow_control=ready_valid)
chan wr_comp((), id=5, kind=streaming, ops=receive_only, flow_control=ready_valid)
chan extra0(bits[1], id=3, kind=streaming, ops=receive_only, flow_control=ready_valid)
chan extra1(bits[1], id=4, kind=streaming, ops=receive_only, flow_control=ready_valid)

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
  next_state_value: () = next_value(state_element=__state, value=next_state)
}
  )",
        .pipeline_stages = 3,
        .ram_config_strings = kSingle1R1W,
        .ram_contents = k1Ram32Bit,
    },
    RamChannelRewriteTestParam{
        .test_name = "32BitWithThree1R1WRams",
        .ir_text = R"(package  test
chan rd_req0((bits[32], ()), id=0, kind=streaming, ops=send_only, flow_control=ready_valid)
chan rd_resp0((bits[32]), id=1, kind=streaming, ops=receive_only, flow_control=ready_valid)
chan wr_req0((bits[32], bits[32], ()), id=2, kind=streaming, ops=send_only, flow_control=ready_valid)
chan wr_comp0((), id=9, kind=streaming, ops=receive_only, flow_control=ready_valid)
chan rd_req1((bits[32], ()), id=3, kind=streaming, ops=send_only, flow_control=ready_valid)
chan rd_resp1((bits[32]), id=4, kind=streaming, ops=receive_only, flow_control=ready_valid)
chan wr_req1((bits[32], bits[32], ()), id=5, kind=streaming, ops=send_only, flow_control=ready_valid)
chan wr_comp1((), id=10, kind=streaming, ops=receive_only, flow_control=ready_valid)
chan rd_req2((bits[32], ()), id=6, kind=streaming, ops=send_only, flow_control=ready_valid)
chan rd_resp2((bits[32]), id=7, kind=streaming, ops=receive_only, flow_control=ready_valid)
chan wr_req2((bits[32], bits[32], ()), id=8, kind=streaming, ops=send_only, flow_control=ready_valid)
chan wr_comp2((), id=11, kind=streaming, ops=receive_only, flow_control=ready_valid)

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
  next_state_value: () = next_value(state_element=__state, value=next_state)
}
  )",
        .pipeline_stages = 20,
        .ram_config_strings = kThree1R1W,
        .ram_contents = k3Ram32Bit,
    },
    RamChannelRewriteTestParam{
        .test_name = "Simple32Bit1RWAnd1R1W",
        .ir_text = R"(package  test
chan req0((bits[32], bits[32], (), (), bits[1], bits[1]), id=0, kind=streaming, ops=send_only, flow_control=ready_valid)
chan resp0((bits[32]), id=1, kind=streaming, ops=receive_only, flow_control=ready_valid)
chan wr_comp0((), id=5, kind=streaming, ops=receive_only, flow_control=ready_valid)
chan rd_req1((bits[32], ()), id=2, kind=streaming, ops=send_only, flow_control=ready_valid)
chan rd_resp1((bits[32]), id=3, kind=streaming, ops=receive_only, flow_control=ready_valid)
chan wr_req1((bits[32], bits[32], ()), id=4, kind=streaming, ops=send_only, flow_control=ready_valid)
chan wr_comp1((), id=6, kind=streaming, ops=receive_only, flow_control=ready_valid)


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
  next_state_value: () = next_value(state_element=__state, value=next_state)
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
                       testing::Values(CodegenPassType::kDefault,
                                       CodegenPassType::kRamRewritePassOnly)),
    [](const testing::TestParamInfo<
        std::tuple<RamChannelRewriteTestParam, CodegenPassType>>& info) {
      return absl::StrCat(std::get<0>(info.param).test_name, "_",
                          CodegenPassName(std::get<1>(info.param)));
    });

// Expects channels named "req" and "resp" and a top level proc named "my_proc".
absl::StatusOr<Block*> MakeBlockAndRunPasses(Package* package,
                                             std::string_view ram_kind) {
  std::vector<RamConfiguration> ram_configurations;
  if (ram_kind == "1RW") {
    ram_configurations.push_back(
        Ram1RWConfiguration("ram", 1, "req", "resp", "wr_comp"));
  } else if (ram_kind == "1R1W") {
    ram_configurations.push_back(Ram1R1WConfiguration(
        "ram", 1, "rd_req", "rd_resp", "wr_req", "wr_comp"));
  } else {
    return absl::InvalidArgumentError(
        absl::StrFormat("Unrecognized ram_kind %s.", ram_kind));
  }

  const CodegenOptions codegen_options =
      CodegenOptions()
          .flop_inputs(false)
          .flop_outputs(false)
          .clock_name("clk")
          .reset("rst", false, false, false)
          .streaming_channel_data_suffix("_data")
          .streaming_channel_valid_suffix("_valid")
          .streaming_channel_ready_suffix("_ready")
          .module_name("pipelined_proc")
          .ram_configurations(ram_configurations);
  const CodegenPassOptions pass_options{
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
      CodegenContext context,
      FunctionBaseToPipelinedBlock(schedule, codegen_options, proc));
  OptimizationContext opt_context;
  XLS_RET_CHECK_OK(
      RunCodegenPassPipeline(pass_options, context.top_block(), opt_context));
  return context.top_block();
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
chan req($req_type, id=0, $req_chan_params)
chan resp($resp_type, id=1, $resp_chan_params)
chan wr_comp($wr_comp_type, id=2, $wr_comp_chan_params)

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
  next_state_value: () = next_value(state_element=__state, value=next_state)
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

chan rd_req($rd_req_type, id=0, $rd_req_chan_params)
chan rd_resp($rd_resp_type, id=1, $rd_resp_chan_params)
chan wr_req($wr_req_type, id=2, $wr_req_chan_params)
chan wr_comp($wr_comp_type, id=3, $wr_comp_chan_params)

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
  next_state_value: () = next_value(state_element=__state, value=next_state)
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
  XLS_ASSERT_OK_AND_ASSIGN(auto package, IrTestBase::ParsePackage(ir_text));
  XLS_EXPECT_OK(MakeBlockAndRunPasses(package.get(), /*ram_kind=*/"1RW"));
}

TEST(RamRewritePassInvalidInputsTest, TestWriteMaskWorks1RW) {
  std::string ir_text = MakeTestProc1RW(
      {.req_type = "(bits[32], bits[32], bits[4], (), bits[1], bits[1])",
       .send_value = "tuple(__state, __state, all_mask, empty_tuple, true_lit, "
                     "false_lit)"});
  XLS_ASSERT_OK_AND_ASSIGN(auto package, IrTestBase::ParsePackage(ir_text));
  XLS_EXPECT_OK(MakeBlockAndRunPasses(package.get(), /*ram_kind=*/"1RW"));
}

TEST(RamRewritePassInvalidInputsTest, TestReadMaskWorks1RW) {
  std::string ir_text = MakeTestProc1RW(
      {.req_type = "(bits[32], bits[32], (), bits[4], bits[1], bits[1])",
       .send_value = "tuple(__state, __state, empty_tuple, all_mask, true_lit, "
                     "false_lit)"});
  XLS_ASSERT_OK_AND_ASSIGN(auto package, IrTestBase::ParsePackage(ir_text));
  XLS_EXPECT_OK(MakeBlockAndRunPasses(package.get(), /*ram_kind=*/"1RW"));
}

TEST(RamRewritePassInvalidInputsTest, InvalidChannelFlowControl1RW) {
  // Try single_value channels instead of streaming
  std::string ir_text = MakeTestProc1RW(
      TestProc1RWVars{.req_chan_params = "kind=single_value, ops=send_only"});
  // The channels are single_value, so ready/valid ports are missing and the
  // pass will error.
  XLS_ASSERT_OK_AND_ASSIGN(auto package, IrTestBase::ParsePackage(ir_text));
  EXPECT_THAT(MakeBlockAndRunPasses(package.get(), /*ram_kind=*/"1RW"),
              StatusIs(absl::StatusCode::kInternal));
}

TEST(RamRewritePassInvalidInputsTest, InvalidReqChannelTypeNotTuple1RW) {
  // Try bits type instead of tuple for req channel
  std::string ir_text = MakeTestProc1RW(TestProc1RWVars{
      .req_type = "bits[32]", .send_value = "add(__state, __state)"});
  XLS_ASSERT_OK_AND_ASSIGN(auto package, IrTestBase::ParsePackage(ir_text));
  EXPECT_THAT(MakeBlockAndRunPasses(package.get(), /*ram_kind=*/"1RW"),
              StatusIs(absl::StatusCode::kInternal,
                       HasSubstr("Request must be a tuple type")));
}

TEST(RamRewritePassInvalidInputsTest, InvalidRespChannelTypeNotTuple1RW) {
  // Try bits type instead of tuple for resp channel
  std::string ir_text =
      MakeTestProc1RW(TestProc1RWVars{.resp_type = "bits[32]"});
  XLS_ASSERT_OK_AND_ASSIGN(auto package, IrTestBase::ParsePackage(ir_text));
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
  XLS_ASSERT_OK_AND_ASSIGN(auto package, IrTestBase::ParsePackage(ir_text));
  EXPECT_THAT(
      MakeBlockAndRunPasses(package.get(), /*ram_kind=*/"1RW"),
      StatusIs(absl::StatusCode::kInternal,
               HasSubstr("Request must be a tuple type with 6 elements")));
}

TEST(RamRewritePassInvalidInputsTest, WrongNumberResponseChannelEntries1RW) {
  // Add an extra field to the response channel
  std::string ir_text =
      MakeTestProc1RW(TestProc1RWVars{.resp_type = "(bits[32], bits[1])"});
  XLS_ASSERT_OK_AND_ASSIGN(auto package, IrTestBase::ParsePackage(ir_text));
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
  XLS_ASSERT_OK_AND_ASSIGN(auto package, IrTestBase::ParsePackage(ir_text));
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
  XLS_ASSERT_OK_AND_ASSIGN(auto package, IrTestBase::ParsePackage(ir_text));
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
  XLS_ASSERT_OK_AND_ASSIGN(auto package, IrTestBase::ParsePackage(ir_text));
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
  XLS_ASSERT_OK_AND_ASSIGN(auto package, IrTestBase::ParsePackage(ir_text));
  EXPECT_THAT(
      MakeBlockAndRunPasses(package.get(), /*ram_kind=*/"1RW"),
      StatusIs(absl::StatusCode::kInternal,
               HasSubstr("Request element re (idx=3) must be type bits[1]")));
}

// Tests for checking invalid inputs on 1r1w RAMs.
TEST(RamRewritePassInvalidInputsTest, TestDefaultsWork1R1W) {
  std::string ir_text = MakeTestProc1R1W({});
  XLS_ASSERT_OK_AND_ASSIGN(auto package, IrTestBase::ParsePackage(ir_text));
  XLS_EXPECT_OK(MakeBlockAndRunPasses(package.get(), /*ram_kind=*/"1R1W"));
}

TEST(RamRewritePassInvalidInputsTest, TestWriteMaskWorks1R1W) {
  std::string ir_text =
      MakeTestProc1R1W({.wr_req_type = "(bits[32], bits[32], bits[4])",
                        .wr_send_value = "tuple(__state, __state, all_mask)"});
  XLS_ASSERT_OK_AND_ASSIGN(auto package, IrTestBase::ParsePackage(ir_text));
  XLS_EXPECT_OK(MakeBlockAndRunPasses(package.get(), /*ram_kind=*/"1R1W"));
}

TEST(RamRewritePassInvalidInputsTest, TestReadMaskWorks1R1W) {
  std::string ir_text =
      MakeTestProc1R1W({.rd_req_type = "(bits[32], bits[4])",
                        .rd_send_value = "tuple(__state, all_mask)"});
  XLS_ASSERT_OK_AND_ASSIGN(auto package, IrTestBase::ParsePackage(ir_text));
  XLS_EXPECT_OK(MakeBlockAndRunPasses(package.get(), /*ram_kind=*/"1R1W"));
}

TEST(RamRewritePassInvalidInputsTest,
     InvalidReadRequestChannelFlowControl1R1W) {
  // Try single_value channels instead of streaming
  std::string ir_text = MakeTestProc1R1W(TestProc1R1WVars{
      .rd_req_chan_params = "kind=single_value, ops=send_only"});
  // The channels are single_value, so ready/valid ports are missing and the
  // pass will error.
  XLS_ASSERT_OK_AND_ASSIGN(auto package, IrTestBase::ParsePackage(ir_text));
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
  XLS_ASSERT_OK_AND_ASSIGN(auto package, IrTestBase::ParsePackage(ir_text));
  EXPECT_THAT(MakeBlockAndRunPasses(package.get(), /*ram_kind=*/"1R1W"),
              StatusIs(absl::StatusCode::kInternal));
}

TEST(RamRewritePassInvalidInputsTest, InvalidWriteChannelFlowControl1R1W) {
  // Try single_value channels instead of streaming
  std::string ir_text = MakeTestProc1R1W(TestProc1R1WVars{
      .wr_req_chan_params = "kind=single_value, ops=send_only"});
  // The channels are single_value, so ready/valid ports are missing and the
  // pass will error.
  XLS_ASSERT_OK_AND_ASSIGN(auto package, IrTestBase::ParsePackage(ir_text));
  EXPECT_THAT(MakeBlockAndRunPasses(package.get(), /*ram_kind=*/"1R1W"),
              StatusIs(absl::StatusCode::kInternal));
}

TEST(RamRewritePassInvalidInputsTest, InvalidReadReqChannelTypeNotTuple1R1W) {
  // Try bits type instead of tuple for req channel
  std::string ir_text = MakeTestProc1R1W(TestProc1R1WVars{
      .rd_req_type = "bits[32]", .rd_send_value = "add(__state, __state)"});
  XLS_ASSERT_OK_AND_ASSIGN(auto package, IrTestBase::ParsePackage(ir_text));
  EXPECT_THAT(MakeBlockAndRunPasses(package.get(), /*ram_kind=*/"1R1W"),
              StatusIs(absl::StatusCode::kInternal,
                       HasSubstr("rd_req must be a tuple type")));
}

TEST(RamRewritePassInvalidInputsTest, InvalidWriteReqChannelTypeNotTuple1R1W) {
  // Try bits type instead of tuple for req channel
  std::string ir_text = MakeTestProc1R1W(TestProc1R1WVars{
      .wr_req_type = "bits[32]", .wr_send_value = "add(__state, __state)"});
  XLS_ASSERT_OK_AND_ASSIGN(auto package, IrTestBase::ParsePackage(ir_text));
  EXPECT_THAT(MakeBlockAndRunPasses(package.get(), /*ram_kind=*/"1R1W"),
              StatusIs(absl::StatusCode::kInternal,
                       HasSubstr("wr_req must be a tuple type")));
}

TEST(RamRewritePassInvalidInputsTest, InvalidReadRespChannelTypeNotTuple1R1W) {
  // Try bits type instead of tuple for resp channel
  std::string ir_text =
      MakeTestProc1R1W(TestProc1R1WVars{.rd_resp_type = "bits[32]"});
  XLS_ASSERT_OK_AND_ASSIGN(auto package, IrTestBase::ParsePackage(ir_text));
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
  XLS_ASSERT_OK_AND_ASSIGN(auto package, IrTestBase::ParsePackage(ir_text));
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
  XLS_ASSERT_OK_AND_ASSIGN(auto package, IrTestBase::ParsePackage(ir_text));
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
  XLS_ASSERT_OK_AND_ASSIGN(auto package, IrTestBase::ParsePackage(ir_text));
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
  XLS_ASSERT_OK_AND_ASSIGN(auto package, IrTestBase::ParsePackage(ir_text));
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
  XLS_ASSERT_OK_AND_ASSIGN(auto package, IrTestBase::ParsePackage(ir_text));
  EXPECT_THAT(MakeBlockAndRunPasses(package.get(), /*ram_kind=*/"1R1W"),
              StatusIs(absl::StatusCode::kInternal,
                       HasSubstr("wr_req element wr_data (idx=1) must not "
                                 "contain token, got token")));
}

TEST(RamRewritePassInvalidInputsTest, ReadResponseElementNotBits1R1W) {
  // Replace rd_data with a token (data must be bits[1])
  std::string ir_text =
      MakeTestProc1R1W(TestProc1R1WVars{.rd_resp_type = "(token)"});
  XLS_ASSERT_OK_AND_ASSIGN(auto package, IrTestBase::ParsePackage(ir_text));
  EXPECT_THAT(MakeBlockAndRunPasses(package.get(), /*ram_kind=*/"1R1W"),
              StatusIs(absl::StatusCode::kInternal,
                       HasSubstr("rd_resp element rd_data (idx=0) must not "
                                 "contain token, got token.")));
}

}  // namespace
}  // namespace verilog
}  // namespace xls
