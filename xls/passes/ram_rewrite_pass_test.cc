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

#include "xls/passes/ram_rewrite_pass.h"

#include <array>
#include <optional>
#include <string>

#include "gtest/gtest.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "xls/common/status/matchers.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/channel.h"
#include "xls/ir/channel_ops.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/ir_matcher.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/value.h"
#include "xls/passes/pass_base.h"
#include "xls/passes/standard_pipeline.h"

namespace xls {
namespace {

namespace m = xls::op_matchers;
using status_testing::IsOkAndHolds;
using status_testing::StatusIs;
using testing::HasSubstr;

class RamRewritePassTest : public IrTestBase {
 protected:
  RamRewritePassTest() = default;

  absl::StatusOr<bool> Run(Package* p,
                           absl::Span<RamRewrite const> ram_rewrites) {
    PassResults results;
    return CreateStandardPassPipeline()->Run(
        p,
        PassOptions{.ram_rewrites = std::vector<RamRewrite>(
                        ram_rewrites.begin(), ram_rewrites.end())},
        &results);
  }

  std::unique_ptr<TokenlessProcBuilder> MakeProcBuilder(Package* p,
                                                        std::string_view name) {
    return std::make_unique<TokenlessProcBuilder>(
        /*name=*/name, /*token_name=*/"tok",
        /*package=*/p, /*should_verify=*/true);
  }

  absl::StatusOr<std::array<Channel*, 4>> MakeAbstractRam(
      Package* p, const RamConfig& config, std::string_view name_prefix,
      Type* data_type) {
    std::array<Channel*, 4> channels;
    int64_t data_width = data_type->GetFlatBitCount();
    if (config.kind != RamKind::kAbstract) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "RAM must be abstract kind, got %s.", RamKindToString(config.kind)));
    }
    Type* addr_type = p->GetBitsType(config.addr_width());
    Type* mask_type = GetMaskType(p, config.mask_width(data_width));

    Type* read_req_type =
        p->GetTupleType(std::vector<Type*>{addr_type, mask_type});
    Type* read_resp_type = p->GetTupleType(std::vector<Type*>{data_type});
    Type* write_req_type =
        p->GetTupleType(std::vector<Type*>{addr_type, data_type, mask_type});
    Type* write_resp_type = p->GetTupleType(std::vector<Type*>{});

    XLS_ASSIGN_OR_RETURN(
        Channel * read_req,
        p->CreateStreamingChannel(absl::StrCat(name_prefix, "_read_req"),
                                  ChannelOps::kSendOnly, read_req_type));
    channels[0] = read_req;

    XLS_ASSIGN_OR_RETURN(
        Channel * read_resp,
        p->CreateStreamingChannel(absl::StrCat(name_prefix, "_read_resp"),
                                  ChannelOps::kReceiveOnly, read_resp_type));
    channels[1] = read_resp;

    XLS_ASSIGN_OR_RETURN(
        Channel * write_req,
        p->CreateStreamingChannel(absl::StrCat(name_prefix, "_write_req"),
                                  ChannelOps::kSendOnly, write_req_type));
    channels[2] = write_req;

    XLS_ASSIGN_OR_RETURN(
        Channel * write_resp,
        p->CreateStreamingChannel(absl::StrCat(name_prefix, "_write_resp"),
                                  ChannelOps::kReceiveOnly, write_resp_type));
    channels[3] = write_resp;
    return channels;
  }
};

TEST_F(RamRewritePassTest, NoRamRewrites) {
  auto p = std::make_unique<Package>(TestName());
  auto pb = MakeProcBuilder(p.get(), "p");
  RamConfig config{.kind = RamKind::kAbstract,
                   .depth = 1024,
                   .word_partition_size = std::nullopt,
                   .initial_value = std::nullopt};
  XLS_ASSERT_OK_AND_ASSIGN(auto abstract_ram_channels,
                           MakeAbstractRam(p.get(), config, "ram",
                                           /*data_type=*/p->GetBitsType(32)));
  auto& [ram_read_req, ram_read_resp, ram_write_req, ram_write_resp] =
      abstract_ram_channels;

  pb->Send(ram_read_req,
           pb->Literal(Value::Tuple({Value(UBits(0, 10)), Value::Tuple({})})));
  pb->Receive(ram_read_resp);
  pb->Send(ram_write_req,
           pb->Literal(Value::Tuple(
               {Value(UBits(0, 10)), Value(UBits(0, 32)), Value::Tuple({})})));
  pb->Receive(ram_write_resp);
  XLS_ASSERT_OK(pb->Build({}).status());
  XLS_ASSERT_OK(p->SetTopByName("p"));

  XLS_EXPECT_OK(Run(p.get(), {}));
  EXPECT_EQ(p->procs().size(), 1);
  EXPECT_EQ(p->channels().size(), 4);
  EXPECT_THAT(p->GetChannel("ram_read_req").value(),
              m::ChannelWithType("(bits[10], ())"));
  EXPECT_THAT(p->GetChannel("ram_read_resp").value(),
              m::ChannelWithType("(bits[32])"));
  EXPECT_THAT(p->GetChannel("ram_write_req").value(),
              m::ChannelWithType("(bits[10], bits[32], ())"));
  EXPECT_THAT(p->GetChannel("ram_write_resp").value(),
              m::ChannelWithType("()"));
}

TEST_F(RamRewritePassTest, SingleAbstractTo1RWRewrite) {
  auto p = std::make_unique<Package>(TestName());
  auto pb = MakeProcBuilder(p.get(), "p");
  RamConfig config_abstract{.kind = RamKind::kAbstract,
                            .depth = 1024,
                            .word_partition_size = std::nullopt,
                            .initial_value = std::nullopt};
  RamConfig config_1rw = config_abstract;
  config_1rw.kind = RamKind::k1RW;
  XLS_ASSERT_OK_AND_ASSIGN(
      auto abstract_ram_channels,
      MakeAbstractRam(p.get(), config_abstract, "ram_abstract",
                      /*data_type=*/p->GetBitsType(32)));
  auto& [ram_abstract_read_req, ram_abstract_read_resp, ram_abstract_write_req,
         ram_abstract_write_resp] = abstract_ram_channels;

  pb->Send(ram_abstract_read_req,
           pb->Literal(Value::Tuple({Value(UBits(0, 10)), Value::Tuple({})})));
  pb->Receive(ram_abstract_read_resp);
  pb->Send(ram_abstract_write_req,
           pb->Literal(Value::Tuple(
               {Value(UBits(0, 10)), Value(UBits(0, 32)), Value::Tuple({})})));
  pb->Receive(ram_abstract_write_resp);

  XLS_ASSERT_OK(pb->Build({}).status());
  XLS_ASSERT_OK(p->SetTopByName("p"));

  std::vector<RamRewrite> ram_rewrites{RamRewrite{
      .from_config = config_abstract,
      .from_channels_logical_to_physical =
          absl::flat_hash_map<std::string, std::string>{
              {"abstract_read_req", "ram_abstract_read_req"},
              {"abstract_read_resp", "ram_abstract_read_resp"},
              {"abstract_write_req", "ram_abstract_write_req"},
              {"write_completion", "ram_abstract_write_resp"},
          },
      .to_config = config_1rw,
      .to_name_prefix = "ram_1rw",
      .model_builder = std::nullopt,
  }};
  EXPECT_THAT(Run(p.get(), ram_rewrites), IsOkAndHolds(true));
  EXPECT_EQ(p->procs().size(), 1);
  EXPECT_EQ(p->channels().size(), 3);
  EXPECT_THAT(
      p->GetChannel("ram_1rw_req").value(),
      m::ChannelWithType("(bits[10], bits[32], (), (), bits[1], bits[1])"));
  EXPECT_THAT(p->GetChannel("ram_1rw_resp").value(),
              m::ChannelWithType("(bits[32])"));
  EXPECT_THAT(p->GetChannel("ram_1rw_write_completion").value(),
              m::ChannelWithType("()"));
}

TEST_F(RamRewritePassTest, SingleAbstractTo1RWRewriteDataIsTuple) {
  auto p = std::make_unique<Package>(TestName());
  auto pb = MakeProcBuilder(p.get(), "p");
  RamConfig config_abstract{.kind = RamKind::kAbstract,
                            .depth = 1024,
                            .word_partition_size = std::nullopt,
                            .initial_value = std::nullopt};
  RamConfig config_1rw = config_abstract;
  config_1rw.kind = RamKind::k1RW;
  XLS_ASSERT_OK_AND_ASSIGN(
      auto abstract_ram_channels,
      MakeAbstractRam(
          p.get(), config_abstract, "ram_abstract",
          /*data_type=*/
          p->GetTupleType({p->GetBitsType(32), p->GetBitsType(1)})));
  auto& [ram_abstract_read_req, ram_abstract_read_resp, ram_abstract_write_req,
         ram_abstract_write_resp] = abstract_ram_channels;

  pb->Send(ram_abstract_read_req, pb->Literal(Value::Tuple({
                                      Value(UBits(0, 10)),  // addr
                                      Value::Tuple({})      // mask
                                  })));
  pb->Receive(ram_abstract_read_resp);
  pb->Send(ram_abstract_write_req,
           pb->Literal(Value::Tuple({
               Value(UBits(0, 10)),                                      // addr
               Value::Tuple({Value(UBits(0, 32)), Value(UBits(0, 1))}),  // data
               Value::Tuple({}),                                         // mask
           })));
  pb->Receive(ram_abstract_write_resp);

  XLS_ASSERT_OK(pb->Build({}).status());
  XLS_ASSERT_OK(p->SetTopByName("p"));

  std::vector<RamRewrite> ram_rewrites{RamRewrite{
      .from_config = config_abstract,
      .from_channels_logical_to_physical =
          absl::flat_hash_map<std::string, std::string>{
              {"abstract_read_req", "ram_abstract_read_req"},
              {"abstract_read_resp", "ram_abstract_read_resp"},
              {"abstract_write_req", "ram_abstract_write_req"},
              {"write_completion", "ram_abstract_write_resp"},
          },
      .to_config = config_1rw,
      .to_name_prefix = "ram_1rw",
      .model_builder = std::nullopt,
  }};
  EXPECT_THAT(Run(p.get(), ram_rewrites), IsOkAndHolds(true));
  EXPECT_EQ(p->procs().size(), 1);
  EXPECT_EQ(p->channels().size(), 3);
  EXPECT_THAT(p->GetChannel("ram_1rw_req").value(),
              m::ChannelWithType(
                  "(bits[10], (bits[32], bits[1]), (), (), bits[1], bits[1])"));
  EXPECT_THAT(p->GetChannel("ram_1rw_resp").value(),
              m::ChannelWithType("((bits[32], bits[1]))"));
  EXPECT_THAT(p->GetChannel("ram_1rw_write_completion").value(),
              m::ChannelWithType("()"));
}

TEST_F(RamRewritePassTest, SingleAbstractTo1RWWithMaskRewrite) {
  auto p = std::make_unique<Package>(TestName());
  auto pb = MakeProcBuilder(p.get(), "p");
  RamConfig config_abstract{.kind = RamKind::kAbstract,
                            .depth = 1024,
                            .word_partition_size = 8,
                            .initial_value = std::nullopt};
  RamConfig config_1rw = config_abstract;
  config_1rw.kind = RamKind::k1RW;
  XLS_ASSERT_OK_AND_ASSIGN(
      auto abstract_ram_channels,
      MakeAbstractRam(p.get(), config_abstract, "ram_abstract",
                      /*data_type=*/p->GetBitsType(32)));
  auto& [ram_abstract_read_req, ram_abstract_read_resp, ram_abstract_write_req,
         ram_abstract_write_resp] = abstract_ram_channels;

  pb->Send(
      ram_abstract_read_req,
      pb->Literal(Value::Tuple({Value(UBits(0, 10)), Value(UBits(0xf, 4))})));
  pb->Receive(ram_abstract_read_resp);
  pb->Send(ram_abstract_write_req,
           pb->Literal(Value::Tuple({Value(UBits(0, 10)), Value(UBits(0, 32)),
                                     Value(UBits(0xf, 4))})));
  pb->Receive(ram_abstract_write_resp);

  XLS_ASSERT_OK(pb->Build({}).status());
  XLS_ASSERT_OK(p->SetTopByName("p"));

  std::vector<RamRewrite> ram_rewrites{RamRewrite{
      .from_config = config_abstract,
      .from_channels_logical_to_physical =
          absl::flat_hash_map<std::string, std::string>{
              {"abstract_read_req", "ram_abstract_read_req"},
              {"abstract_read_resp", "ram_abstract_read_resp"},
              {"abstract_write_req", "ram_abstract_write_req"},
              {"write_completion", "ram_abstract_write_resp"},
          },
      .to_config = config_1rw,
      .to_name_prefix = "ram_1rw",
      .model_builder = std::nullopt,
  }};
  EXPECT_THAT(Run(p.get(), ram_rewrites), IsOkAndHolds(true));
  EXPECT_EQ(p->procs().size(), 1);
  EXPECT_EQ(p->channels().size(), 3);
  EXPECT_THAT(p->GetChannel("ram_1rw_req").value(),
              m::ChannelWithType(
                  "(bits[10], bits[32], bits[4], bits[4], bits[1], bits[1])"));
  EXPECT_THAT(p->GetChannel("ram_1rw_resp").value(),
              m::ChannelWithType("(bits[32])"));
  EXPECT_THAT(p->GetChannel("ram_1rw_write_completion").value(),
              m::ChannelWithType("()"));
}

TEST_F(RamRewritePassTest, MultipleAbstractTo1RWRewrite) {
  auto p = std::make_unique<Package>(TestName());
  auto pb = MakeProcBuilder(p.get(), "p");
  RamConfig config_abstract{.kind = RamKind::kAbstract,
                            .depth = 1024,
                            .word_partition_size = std::nullopt,
                            .initial_value = std::nullopt};
  RamConfig config_1rw = config_abstract;
  config_1rw.kind = RamKind::k1RW;
  XLS_ASSERT_OK_AND_ASSIGN(
      auto abstract0_ram_channels,
      MakeAbstractRam(p.get(), config_abstract, "ram_abstract0",
                      /*data_type=*/p->GetBitsType(32)));
  auto& [ram_abstract0_read_req, ram_abstract0_read_resp,
         ram_abstract0_write_req, ram_abstract0_write_resp] =
      abstract0_ram_channels;
  XLS_ASSERT_OK_AND_ASSIGN(
      auto abstract1_ram_channels,
      MakeAbstractRam(p.get(), config_abstract, "ram_abstract1",
                      /*data_type=*/p->GetBitsType(32)));
  auto& [ram_abstract1_read_req, ram_abstract1_read_resp,
         ram_abstract1_write_req, ram_abstract1_write_resp] =
      abstract1_ram_channels;

  pb->Send(ram_abstract0_read_req,
           pb->Literal(Value::Tuple({Value(UBits(0, 10)), Value::Tuple({})})));
  pb->Receive(ram_abstract0_read_resp);
  pb->Send(ram_abstract0_write_req,
           pb->Literal(Value::Tuple(
               {Value(UBits(0, 10)), Value(UBits(0, 32)), Value::Tuple({})})));
  pb->Receive(ram_abstract0_write_resp);

  pb->Send(ram_abstract1_read_req,
           pb->Literal(Value::Tuple({Value(UBits(0, 10)), Value::Tuple({})})));
  pb->Receive(ram_abstract1_read_resp);
  pb->Send(ram_abstract1_write_req,
           pb->Literal(Value::Tuple(
               {Value(UBits(0, 10)), Value(UBits(0, 32)), Value::Tuple({})})));
  pb->Receive(ram_abstract1_write_resp);

  XLS_ASSERT_OK(pb->Build({}).status());
  XLS_ASSERT_OK(p->SetTopByName("p"));

  std::vector<RamRewrite> ram_rewrites{
      RamRewrite{
          .from_config = config_abstract,
          .from_channels_logical_to_physical =
              absl::flat_hash_map<std::string, std::string>{
                  {"abstract_read_req", "ram_abstract0_read_req"},
                  {"abstract_read_resp", "ram_abstract0_read_resp"},
                  {"abstract_write_req", "ram_abstract0_write_req"},
                  {"write_completion", "ram_abstract0_write_resp"},
              },
          .to_config = config_1rw,
          .to_name_prefix = "ram_1rw_0",
          .model_builder = std::nullopt,
      },
      RamRewrite{
          .from_config = config_abstract,
          .from_channels_logical_to_physical =
              absl::flat_hash_map<std::string, std::string>{
                  {"abstract_read_req", "ram_abstract1_read_req"},
                  {"abstract_read_resp", "ram_abstract1_read_resp"},
                  {"abstract_write_req", "ram_abstract1_write_req"},
                  {"write_completion", "ram_abstract1_write_resp"},
              },
          .to_config = config_1rw,
          .to_name_prefix = "ram_1rw_1",
          .model_builder = std::nullopt,
      }};
  EXPECT_THAT(Run(p.get(), ram_rewrites), IsOkAndHolds(true));
  EXPECT_EQ(p->procs().size(), 1);
  EXPECT_EQ(p->channels().size(), 6);
  EXPECT_THAT(
      p->GetChannel("ram_1rw_0_req").value(),
      m::ChannelWithType("(bits[10], bits[32], (), (), bits[1], bits[1])"));
  EXPECT_THAT(p->GetChannel("ram_1rw_0_resp").value(),
              m::ChannelWithType("(bits[32])"));
  EXPECT_THAT(p->GetChannel("ram_1rw_0_write_completion").value(),
              m::ChannelWithType("()"));
  EXPECT_THAT(
      p->GetChannel("ram_1rw_1_req").value(),
      m::ChannelWithType("(bits[10], bits[32], (), (), bits[1], bits[1])"));
  EXPECT_THAT(p->GetChannel("ram_1rw_1_resp").value(),
              m::ChannelWithType("(bits[32])"));
  EXPECT_THAT(p->GetChannel("ram_1rw_1_write_completion").value(),
              m::ChannelWithType("()"));
}

TEST_F(RamRewritePassTest, SingleAbstractTo1RWRewriteWithWidthMismatch) {
  auto p = std::make_unique<Package>(TestName());
  auto pb = MakeProcBuilder(p.get(), "p");
  RamConfig config_abstract{.kind = RamKind::kAbstract,
                            .depth = 1024,
                            .word_partition_size = std::nullopt,
                            .initial_value = std::nullopt};
  RamConfig config_1rw = config_abstract;
  config_1rw.kind = RamKind::k1RW;
  XLS_ASSERT_OK_AND_ASSIGN(
      auto abstract_ram_channels,
      MakeAbstractRam(p.get(), config_abstract, "ram_abstract",
                      /*data_type=*/p->GetBitsType(32)));
  auto& [ram_abstract_read_req, ram_abstract_read_resp, ram_abstract_write_req,
         ram_abstract_write_resp] = abstract_ram_channels;

  pb->Send(ram_abstract_read_req,
           pb->Literal(Value::Tuple({Value(UBits(0, 10)), Value::Tuple({})})));
  pb->Receive(ram_abstract_read_resp);
  pb->Send(ram_abstract_write_req,
           pb->Literal(Value::Tuple(
               {Value(UBits(0, 10)), Value(UBits(0, 32)), Value::Tuple({})})));
  pb->Receive(ram_abstract_write_resp);

  XLS_ASSERT_OK(pb->Build({}).status());
  XLS_ASSERT_OK(p->SetTopByName("p"));

  // Change depth, pass should fail because address field doesn't match
  // expectation.
  config_abstract.depth = 4096;
  std::vector<RamRewrite> ram_rewrites{RamRewrite{
      .from_config = config_abstract,
      .from_channels_logical_to_physical =
          absl::flat_hash_map<std::string, std::string>{
              {"abstract_read_req", "ram_abstract_read_req"},
              {"abstract_read_resp", "ram_abstract_read_resp"},
              {"abstract_write_req", "ram_abstract_write_req"},
              {"write_completion", "ram_abstract_write_resp"},
          },
      .to_config = config_1rw,
      .to_name_prefix = "ram_1rw",
      .model_builder = std::nullopt,
  }};
  EXPECT_THAT(
      Run(p.get(), ram_rewrites),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr(
                   "Expected addr (tuple read_req element 0 of 2) to have type "
                   "bits[12], got bits[10]")));
}

TEST_F(RamRewritePassTest, SingleAbstractTo1R1WRewrite) {
  auto p = std::make_unique<Package>(TestName());
  auto pb = MakeProcBuilder(p.get(), "p");
  RamConfig config_abstract{.kind = RamKind::kAbstract,
                            .depth = 1024,
                            .word_partition_size = std::nullopt,
                            .initial_value = std::nullopt};
  RamConfig config_1r1w = config_abstract;
  config_1r1w.kind = RamKind::k1R1W;
  XLS_ASSERT_OK_AND_ASSIGN(
      auto abstract_ram_channels,
      MakeAbstractRam(p.get(), config_abstract, "ram_abstract",
                      /*data_type=*/p->GetBitsType(32)));
  auto& [ram_abstract_read_req, ram_abstract_read_resp, ram_abstract_write_req,
         ram_abstract_write_resp] = abstract_ram_channels;

  pb->Send(ram_abstract_read_req,
           pb->Literal(Value::Tuple({Value(UBits(0, 10)), Value::Tuple({})})));
  pb->Receive(ram_abstract_read_resp);
  pb->Send(ram_abstract_write_req,
           pb->Literal(Value::Tuple(
               {Value(UBits(0, 10)), Value(UBits(0, 32)), Value::Tuple({})})));
  pb->Receive(ram_abstract_write_resp);

  XLS_ASSERT_OK(pb->Build({}).status());
  XLS_ASSERT_OK(p->SetTopByName("p"));

  std::vector<RamRewrite> ram_rewrites{RamRewrite{
      .from_config = config_abstract,
      .from_channels_logical_to_physical =
          absl::flat_hash_map<std::string, std::string>{
              {"abstract_read_req", "ram_abstract_read_req"},
              {"abstract_read_resp", "ram_abstract_read_resp"},
              {"abstract_write_req", "ram_abstract_write_req"},
              {"write_completion", "ram_abstract_write_resp"},
          },
      .to_config = config_1r1w,
      .to_name_prefix = "ram_1r1w",
      .model_builder = std::nullopt,
  }};
  EXPECT_THAT(Run(p.get(), ram_rewrites), IsOkAndHolds(true));
  EXPECT_EQ(p->procs().size(), 1);
  EXPECT_EQ(p->channels().size(), 4);
  EXPECT_THAT(p->GetChannel("ram_1r1w_read_req").value(),
              m::ChannelWithType("(bits[10], ())"));
  EXPECT_THAT(p->GetChannel("ram_1r1w_read_resp").value(),
              m::ChannelWithType("(bits[32])"));
  EXPECT_THAT(p->GetChannel("ram_1r1w_write_req").value(),
              m::ChannelWithType("(bits[10], bits[32], ())"));
  EXPECT_THAT(p->GetChannel("ram_1r1w_write_completion").value(),
              m::ChannelWithType("()"));
}

TEST_F(RamRewritePassTest, SingleAbstractTo1R1WRewriteDataIsTuple) {
  auto p = std::make_unique<Package>(TestName());
  auto pb = MakeProcBuilder(p.get(), "p");
  RamConfig config_abstract{.kind = RamKind::kAbstract,
                            .depth = 1024,
                            .word_partition_size = std::nullopt,
                            .initial_value = std::nullopt};
  RamConfig config_1r1w = config_abstract;
  config_1r1w.kind = RamKind::k1R1W;
  XLS_ASSERT_OK_AND_ASSIGN(
      auto abstract_ram_channels,
      MakeAbstractRam(
          p.get(), config_abstract, "ram_abstract",
          /*data_type=*/
          p->GetTupleType({p->GetBitsType(32), p->GetBitsType(32)})));
  auto& [ram_abstract_read_req, ram_abstract_read_resp, ram_abstract_write_req,
         ram_abstract_write_resp] = abstract_ram_channels;

  pb->Send(ram_abstract_read_req,
           pb->Literal(Value::Tuple({Value(UBits(0, 10)), Value::Tuple({})})));
  pb->Receive(ram_abstract_read_resp);
  pb->Send(ram_abstract_write_req,
           pb->Literal(Value::Tuple(
               {Value(UBits(0, 10)),
                Value::Tuple({Value(UBits(0, 32)), Value(UBits(0, 32))}),
                Value::Tuple({})})));
  pb->Receive(ram_abstract_write_resp);

  XLS_ASSERT_OK(pb->Build({}).status());
  XLS_ASSERT_OK(p->SetTopByName("p"));

  std::vector<RamRewrite> ram_rewrites{RamRewrite{
      .from_config = config_abstract,
      .from_channels_logical_to_physical =
          absl::flat_hash_map<std::string, std::string>{
              {"abstract_read_req", "ram_abstract_read_req"},
              {"abstract_read_resp", "ram_abstract_read_resp"},
              {"abstract_write_req", "ram_abstract_write_req"},
              {"write_completion", "ram_abstract_write_resp"},
          },
      .to_config = config_1r1w,
      .to_name_prefix = "ram_1r1w",
      .model_builder = std::nullopt,
  }};
  EXPECT_THAT(Run(p.get(), ram_rewrites), IsOkAndHolds(true));
  EXPECT_EQ(p->procs().size(), 1);
  EXPECT_EQ(p->channels().size(), 4);
  EXPECT_THAT(p->GetChannel("ram_1r1w_read_req").value(),
              m::ChannelWithType("(bits[10], ())"));
  EXPECT_THAT(p->GetChannel("ram_1r1w_read_resp").value(),
              m::ChannelWithType("((bits[32], bits[32]))"));
  EXPECT_THAT(p->GetChannel("ram_1r1w_write_req").value(),
              m::ChannelWithType("(bits[10], (bits[32], bits[32]), ())"));
  EXPECT_THAT(p->GetChannel("ram_1r1w_write_completion").value(),
              m::ChannelWithType("()"));
}

TEST_F(RamRewritePassTest, SingleAbstractTo1R1WWithMaskRewrite) {
  auto p = std::make_unique<Package>(TestName());
  auto pb = MakeProcBuilder(p.get(), "p");
  RamConfig config_abstract{.kind = RamKind::kAbstract,
                            .depth = 1024,
                            .word_partition_size = 8,
                            .initial_value = std::nullopt};
  RamConfig config_1r1w = config_abstract;
  config_1r1w.kind = RamKind::k1R1W;
  XLS_ASSERT_OK_AND_ASSIGN(
      auto abstract_ram_channels,
      MakeAbstractRam(p.get(), config_abstract, "ram_abstract",
                      /*data_type=*/p->GetBitsType(32)));
  auto& [ram_abstract_read_req, ram_abstract_read_resp, ram_abstract_write_req,
         ram_abstract_write_resp] = abstract_ram_channels;

  pb->Send(
      ram_abstract_read_req,
      pb->Literal(Value::Tuple({Value(UBits(0, 10)), Value(UBits(0xf, 4))})));
  pb->Receive(ram_abstract_read_resp);
  pb->Send(ram_abstract_write_req,
           pb->Literal(Value::Tuple({Value(UBits(0, 10)), Value(UBits(0, 32)),
                                     Value(UBits(0xf, 4))})));
  pb->Receive(ram_abstract_write_resp);

  XLS_ASSERT_OK(pb->Build({}).status());
  XLS_ASSERT_OK(p->SetTopByName("p"));

  std::vector<RamRewrite> ram_rewrites{RamRewrite{
      .from_config = config_abstract,
      .from_channels_logical_to_physical =
          absl::flat_hash_map<std::string, std::string>{
              {"abstract_read_req", "ram_abstract_read_req"},
              {"abstract_read_resp", "ram_abstract_read_resp"},
              {"abstract_write_req", "ram_abstract_write_req"},
              {"write_completion", "ram_abstract_write_resp"},
          },
      .to_config = config_1r1w,
      .to_name_prefix = "ram_1r1w",
      .model_builder = std::nullopt,
  }};
  EXPECT_THAT(Run(p.get(), ram_rewrites), IsOkAndHolds(true));
  EXPECT_EQ(p->procs().size(), 1);
  EXPECT_EQ(p->channels().size(), 4);
  EXPECT_THAT(p->GetChannel("ram_1r1w_read_req").value(),
              m::ChannelWithType("(bits[10], bits[4])"));
  EXPECT_THAT(p->GetChannel("ram_1r1w_read_resp").value(),
              m::ChannelWithType("(bits[32])"));
  EXPECT_THAT(p->GetChannel("ram_1r1w_write_req").value(),
              m::ChannelWithType("(bits[10], bits[32], bits[4])"));
  EXPECT_THAT(p->GetChannel("ram_1r1w_write_completion").value(),
              m::ChannelWithType("()"));
}

TEST_F(RamRewritePassTest, MultipleAbstractTo1R1WRewrite) {
  auto p = std::make_unique<Package>(TestName());
  auto pb = MakeProcBuilder(p.get(), "p");
  RamConfig config_abstract{.kind = RamKind::kAbstract,
                            .depth = 1024,
                            .word_partition_size = std::nullopt,
                            .initial_value = std::nullopt};
  RamConfig config_1r1w = config_abstract;
  config_1r1w.kind = RamKind::k1R1W;
  XLS_ASSERT_OK_AND_ASSIGN(
      auto abstract0_ram_channels,
      MakeAbstractRam(p.get(), config_abstract, "ram_abstract0",
                      /*data_type=*/p->GetBitsType(32)));
  auto& [ram_abstract0_read_req, ram_abstract0_read_resp,
         ram_abstract0_write_req, ram_abstract0_write_resp] =
      abstract0_ram_channels;
  XLS_ASSERT_OK_AND_ASSIGN(
      auto abstract1_ram_channels,
      MakeAbstractRam(p.get(), config_abstract, "ram_abstract1",
                      /*data_type=*/p->GetBitsType(32)));
  auto& [ram_abstract1_read_req, ram_abstract1_read_resp,
         ram_abstract1_write_req, ram_abstract1_write_resp] =
      abstract1_ram_channels;

  pb->Send(ram_abstract0_read_req,
           pb->Literal(Value::Tuple({Value(UBits(0, 10)), Value::Tuple({})})));
  pb->Receive(ram_abstract0_read_resp);
  pb->Send(ram_abstract0_write_req,
           pb->Literal(Value::Tuple(
               {Value(UBits(0, 10)), Value(UBits(0, 32)), Value::Tuple({})})));
  pb->Receive(ram_abstract0_write_resp);

  pb->Send(ram_abstract1_read_req,
           pb->Literal(Value::Tuple({Value(UBits(0, 10)), Value::Tuple({})})));
  pb->Receive(ram_abstract1_read_resp);
  pb->Send(ram_abstract1_write_req,
           pb->Literal(Value::Tuple(
               {Value(UBits(0, 10)), Value(UBits(0, 32)), Value::Tuple({})})));
  pb->Receive(ram_abstract1_write_resp);

  XLS_ASSERT_OK(pb->Build({}).status());
  XLS_ASSERT_OK(p->SetTopByName("p"));

  std::vector<RamRewrite> ram_rewrites{
      RamRewrite{
          .from_config = config_abstract,
          .from_channels_logical_to_physical =
              absl::flat_hash_map<std::string, std::string>{
                  {"abstract_read_req", "ram_abstract0_read_req"},
                  {"abstract_read_resp", "ram_abstract0_read_resp"},
                  {"abstract_write_req", "ram_abstract0_write_req"},
                  {"write_completion", "ram_abstract0_write_resp"},
              },
          .to_config = config_1r1w,
          .to_name_prefix = "ram_1r1w_0",
          .model_builder = std::nullopt,
      },
      RamRewrite{
          .from_config = config_abstract,
          .from_channels_logical_to_physical =
              absl::flat_hash_map<std::string, std::string>{
                  {"abstract_read_req", "ram_abstract1_read_req"},
                  {"abstract_read_resp", "ram_abstract1_read_resp"},
                  {"abstract_write_req", "ram_abstract1_write_req"},
                  {"write_completion", "ram_abstract1_write_resp"},
              },
          .to_config = config_1r1w,
          .to_name_prefix = "ram_1r1w_1",
          .model_builder = std::nullopt,
      }};
  EXPECT_THAT(Run(p.get(), ram_rewrites), IsOkAndHolds(true));
  EXPECT_EQ(p->procs().size(), 1);
  EXPECT_EQ(p->channels().size(), 8);
  EXPECT_THAT(p->GetChannel("ram_1r1w_0_read_req").value(),
              m::ChannelWithType("(bits[10], ())"));
  EXPECT_THAT(p->GetChannel("ram_1r1w_0_read_resp").value(),
              m::ChannelWithType("(bits[32])"));
  EXPECT_THAT(p->GetChannel("ram_1r1w_0_write_req").value(),
              m::ChannelWithType("(bits[10], bits[32], ())"));
  EXPECT_THAT(p->GetChannel("ram_1r1w_0_write_completion").value(),
              m::ChannelWithType("()"));
  EXPECT_THAT(p->GetChannel("ram_1r1w_1_read_req").value(),
              m::ChannelWithType("(bits[10], ())"));
  EXPECT_THAT(p->GetChannel("ram_1r1w_1_read_resp").value(),
              m::ChannelWithType("(bits[32])"));
  EXPECT_THAT(p->GetChannel("ram_1r1w_1_write_req").value(),
              m::ChannelWithType("(bits[10], bits[32], ())"));
  EXPECT_THAT(p->GetChannel("ram_1r1w_1_write_completion").value(),
              m::ChannelWithType("()"));
}

TEST_F(RamRewritePassTest, MultipleAbstractTo1RWAnd1R1WRewrite) {
  auto p = std::make_unique<Package>(TestName());
  auto pb = MakeProcBuilder(p.get(), "p");
  RamConfig config_abstract{.kind = RamKind::kAbstract,
                            .depth = 1024,
                            .word_partition_size = std::nullopt,
                            .initial_value = std::nullopt};
  RamConfig config_1rw = config_abstract;
  config_1rw.kind = RamKind::k1RW;
  RamConfig config_1r1w = config_abstract;
  config_1r1w.kind = RamKind::k1R1W;
  XLS_ASSERT_OK_AND_ASSIGN(
      auto abstract0_ram_channels,
      MakeAbstractRam(p.get(), config_abstract, "ram_abstract0",
                      /*data_type=*/p->GetBitsType(32)));
  auto& [ram_abstract0_read_req, ram_abstract0_read_resp,
         ram_abstract0_write_req, ram_abstract0_write_resp] =
      abstract0_ram_channels;
  XLS_ASSERT_OK_AND_ASSIGN(
      auto abstract1_ram_channels,
      MakeAbstractRam(p.get(), config_abstract, "ram_abstract1",
                      /*data_type=*/p->GetBitsType(32)));
  auto& [ram_abstract1_read_req, ram_abstract1_read_resp,
         ram_abstract1_write_req, ram_abstract1_write_resp] =
      abstract1_ram_channels;

  pb->Send(ram_abstract0_read_req,
           pb->Literal(Value::Tuple({Value(UBits(0, 10)), Value::Tuple({})})));
  pb->Receive(ram_abstract0_read_resp);
  pb->Send(ram_abstract0_write_req,
           pb->Literal(Value::Tuple(
               {Value(UBits(0, 10)), Value(UBits(0, 32)), Value::Tuple({})})));
  pb->Receive(ram_abstract0_write_resp);

  pb->Send(ram_abstract1_read_req,
           pb->Literal(Value::Tuple({Value(UBits(0, 10)), Value::Tuple({})})));
  pb->Receive(ram_abstract1_read_resp);
  pb->Send(ram_abstract1_write_req,
           pb->Literal(Value::Tuple(
               {Value(UBits(0, 10)), Value(UBits(0, 32)), Value::Tuple({})})));
  pb->Receive(ram_abstract1_write_resp);

  XLS_ASSERT_OK(pb->Build({}).status());
  XLS_ASSERT_OK(p->SetTopByName("p"));

  std::vector<RamRewrite> ram_rewrites{
      RamRewrite{
          .from_config = config_abstract,
          .from_channels_logical_to_physical =
              absl::flat_hash_map<std::string, std::string>{
                  {"abstract_read_req", "ram_abstract0_read_req"},
                  {"abstract_read_resp", "ram_abstract0_read_resp"},
                  {"abstract_write_req", "ram_abstract0_write_req"},
                  {"write_completion", "ram_abstract0_write_resp"},
              },
          .to_config = config_1rw,
          .to_name_prefix = "ram_1rw_0",
          .model_builder = std::nullopt,
      },
      RamRewrite{
          .from_config = config_abstract,
          .from_channels_logical_to_physical =
              absl::flat_hash_map<std::string, std::string>{
                  {"abstract_read_req", "ram_abstract1_read_req"},
                  {"abstract_read_resp", "ram_abstract1_read_resp"},
                  {"abstract_write_req", "ram_abstract1_write_req"},
                  {"write_completion", "ram_abstract1_write_resp"},
              },
          .to_config = config_1r1w,
          .to_name_prefix = "ram_1r1w_1",
          .model_builder = std::nullopt,
      }};
  EXPECT_THAT(Run(p.get(), ram_rewrites), IsOkAndHolds(true));
  EXPECT_EQ(p->procs().size(), 1);
  EXPECT_EQ(p->channels().size(), 7);
  EXPECT_THAT(
      p->GetChannel("ram_1rw_0_req").value(),
      m::ChannelWithType("(bits[10], bits[32], (), (), bits[1], bits[1])"));
  EXPECT_THAT(p->GetChannel("ram_1rw_0_resp").value(),
              m::ChannelWithType("(bits[32])"));
  EXPECT_THAT(p->GetChannel("ram_1rw_0_write_completion").value(),
              m::ChannelWithType("()"));
  EXPECT_THAT(p->GetChannel("ram_1r1w_1_read_req").value(),
              m::ChannelWithType("(bits[10], ())"));
  EXPECT_THAT(p->GetChannel("ram_1r1w_1_read_resp").value(),
              m::ChannelWithType("(bits[32])"));
  EXPECT_THAT(p->GetChannel("ram_1r1w_1_write_req").value(),
              m::ChannelWithType("(bits[10], bits[32], ())"));
  EXPECT_THAT(p->GetChannel("ram_1r1w_1_write_completion").value(),
              m::ChannelWithType("()"));
}
}  // namespace
}  // namespace xls
