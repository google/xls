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

#include "xls/simulation/generic/config.h"

#include <fcntl.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>

#include <cstdint>
#include <fstream>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "gmock/gmock.h"
#include "google/protobuf/util/message_differencer.h"
#include "gtest/gtest.h"
#include "xls/common/logging/log_flags.h"
#include "xls/common/logging/logging.h"
#include "xls/common/status/matchers.h"

namespace xls::simulation::generic {
namespace {

class ConfigChannelTest : public ::testing::Test {
 protected:
  ConfigChannelTest() : name("name"), offset(0xDEADBEEF), dma_id(0) {}
  static void SetUpTestSuite() { absl::SetFlag(&FLAGS_logtostderr, true); }

  absl::StatusOr<ConfigChannel> cfg;
  std::string name;
  uint64_t offset;
  uint64_t dma_id;
};

TEST_F(ConfigChannelTest, MakeConfigChannel) {
  ::testing::internal::CaptureStderr();
  XLS_ASSERT_OK_AND_ASSIGN(
      cfg, ConfigChannel::MakeConfigChannel(name, offset, dma_id));
  // Expect empty stderr
  EXPECT_THAT(::testing::internal::GetCapturedStderr(), ::testing::IsEmpty());
  EXPECT_EQ(cfg.value().GetName(), name);
  XLS_EXPECT_OK_AND_EQ(cfg.value().GetOffset(), offset);
  XLS_EXPECT_OK_AND_EQ(cfg.value().GetDMAID(), dma_id);
}

TEST_F(ConfigChannelTest, MakeConfigChannelOffsetNullopt) {
  ::testing::internal::CaptureStderr();
  XLS_ASSERT_OK_AND_ASSIGN(
      cfg, ConfigChannel::MakeConfigChannel(name, std::nullopt, dma_id));
  // Expect empty stderr
  EXPECT_THAT(::testing::internal::GetCapturedStderr(), ::testing::IsEmpty());
  EXPECT_EQ(cfg.value().GetName(), name);
  EXPECT_TRUE(absl::IsNotFound(cfg.value().GetOffset().status()));
  XLS_EXPECT_OK_AND_EQ(cfg.value().GetDMAID(), dma_id);
}

TEST_F(ConfigChannelTest, MakeConfigChannelDmaIdNullopt) {
  ::testing::internal::CaptureStderr();
  XLS_ASSERT_OK_AND_ASSIGN(
      cfg, ConfigChannel::MakeConfigChannel(name, offset, std::nullopt));
  // Expect empty stderr
  EXPECT_THAT(::testing::internal::GetCapturedStderr(), ::testing::IsEmpty());
  EXPECT_EQ(cfg.value().GetName(), name);
  XLS_EXPECT_OK_AND_EQ(cfg.value().GetOffset(), offset);
  EXPECT_TRUE(absl::IsNotFound(cfg.value().GetDMAID().status()));
}

TEST_F(ConfigChannelTest, MakeConfigChannelDmaIdAndOffsetNullopt) {
  ::testing::internal::CaptureStderr();
  EXPECT_TRUE(absl::IsInvalidArgument(
      ConfigChannel::MakeConfigChannel(name, std::nullopt, std::nullopt)
          .status()));
  EXPECT_THAT(::testing::internal::GetCapturedStderr(),
              ::testing::HasSubstr("does not define offset or DMA ID"));
}

class ConfigAxiStreamLikeChannelTest : public ::testing::Test {
 protected:
  ConfigAxiStreamLikeChannelTest()
      : name("name"),
        data_idxs({0, 1, 2, 3, 4, 5, 6, 7}),
        offset(0xDEADBEEF),
        dma_id(0),
        keep_idx(0),
        last_idx(0) {}
  static void SetUpTestSuite() { absl::SetFlag(&FLAGS_logtostderr, true); }

  absl::StatusOr<ConfigAxiStreamLikeChannel> cfg;
  std::string name;
  std::vector<uint64_t> data_idxs;
  uint64_t offset;
  uint64_t dma_id;
  uint64_t keep_idx;
  uint64_t last_idx;
};

TEST_F(ConfigAxiStreamLikeChannelTest, MakeConfigAxiStreamLikeChannel) {
  ::testing::internal::CaptureStderr();
  XLS_ASSERT_OK_AND_ASSIGN(
      cfg, ConfigAxiStreamLikeChannel::MakeConfigAxiStreamLikeChannel(
               name, data_idxs, offset, dma_id, keep_idx, last_idx));
  // Expect empty stderr
  EXPECT_THAT(::testing::internal::GetCapturedStderr(), ::testing::IsEmpty());
  EXPECT_EQ(cfg.value().GetDataIdxs(), data_idxs);
  XLS_EXPECT_OK_AND_EQ(cfg.value().GetKeepIdx(), keep_idx);
  XLS_EXPECT_OK_AND_EQ(cfg.value().GetLastIdx(), last_idx);
}

TEST_F(ConfigAxiStreamLikeChannelTest, GetDataIdxsEmpty) {
  data_idxs = {};
  ::testing::internal::CaptureStderr();
  EXPECT_TRUE(absl::IsInvalidArgument(
      ConfigAxiStreamLikeChannel::MakeConfigAxiStreamLikeChannel(
          name, data_idxs, offset, dma_id, keep_idx, last_idx)
          .status()));
  EXPECT_THAT(::testing::internal::GetCapturedStderr(),
              ::testing::HasSubstr("does not define any data indices"));
}

TEST_F(ConfigAxiStreamLikeChannelTest, GetKeepIdxNullopt) {
  ::testing::internal::CaptureStderr();
  XLS_ASSERT_OK_AND_ASSIGN(
      cfg, ConfigAxiStreamLikeChannel::MakeConfigAxiStreamLikeChannel(
               name, data_idxs, offset, dma_id, std::nullopt, last_idx));
  // Expect empty stderr
  EXPECT_THAT(::testing::internal::GetCapturedStderr(), ::testing::IsEmpty());
  EXPECT_EQ(cfg.value().GetDataIdxs(), data_idxs);
  EXPECT_TRUE(absl::IsNotFound(cfg.value().GetKeepIdx().status()));
  XLS_EXPECT_OK_AND_EQ(cfg.value().GetLastIdx(), last_idx);
}

TEST_F(ConfigAxiStreamLikeChannelTest, GetLastIdxNullopt) {
  ::testing::internal::CaptureStderr();
  XLS_ASSERT_OK_AND_ASSIGN(
      cfg, ConfigAxiStreamLikeChannel::MakeConfigAxiStreamLikeChannel(
               name, data_idxs, offset, dma_id, keep_idx, std::nullopt));
  // Expect empty stderr
  EXPECT_THAT(::testing::internal::GetCapturedStderr(), ::testing::IsEmpty());
  EXPECT_EQ(cfg.value().GetDataIdxs(), data_idxs);
  XLS_EXPECT_OK_AND_EQ(cfg.value().GetKeepIdx(), keep_idx);
  EXPECT_TRUE(absl::IsNotFound(cfg.value().GetLastIdx().status()));
}

class ChannelManagerTest : public ::testing::Test {
 protected:
  ChannelManagerTest()
      : base_address(0xDEADBEEF), channels_num(8), channel_width(64) {}
  static void SetUpTestSuite() { absl::SetFlag(&FLAGS_logtostderr, true); }
  void SetUp() override { SetupChannels(); }
  virtual void SetupChannels() = 0;

  uint64_t base_address;
  uint8_t channels_num;
  uint8_t channel_width;
};

class RegularChannelManagerTest : public ChannelManagerTest {
 protected:
  void SetupChannels() override {
    channels.clear();
    for (uint8_t i = 0; i < channels_num; i++) {
      auto cfg = ConfigChannel::MakeConfigChannel(
          "RegularChannel" + std::to_string(i), (channel_width * i), i);
      channels.push_back(cfg.value());
    }
  }

  void CompareChannels(std::vector<ConfigChannel> tested_channels) {
    uint64_t i = 0;
    for (auto channel : tested_channels) {
      EXPECT_EQ(channel.GetName(), channels[i].GetName());
      EXPECT_EQ(channel.GetOffset(), channels[i].GetOffset());
      EXPECT_EQ(channel.GetDMAID(), channels[i].GetDMAID());
      i++;
    }
  }

  std::vector<ConfigChannel> channels;
};

class ConfigSingleValueManagerTest : public RegularChannelManagerTest {
 protected:
  ConfigSingleValueManagerTest() : runtime_status_offset(0xFEEBDAED) {}

  uint64_t runtime_status_offset;
};

TEST_F(ConfigSingleValueManagerTest, ConfigSingleValueManager) {
  ::testing::internal::CaptureStderr();
  ConfigSingleValueManager cfg =
      ConfigSingleValueManager(base_address, runtime_status_offset, channels);
  // Expect empty stderr
  EXPECT_THAT(::testing::internal::GetCapturedStderr(), ::testing::IsEmpty());
  EXPECT_EQ(cfg.GetBaseAddress(), base_address);
  EXPECT_EQ(cfg.GetRuntimeStatusOffset(), runtime_status_offset);
  EXPECT_EQ(cfg.GetChannels().size(), channels.size());
  CompareChannels(cfg.GetChannels());
}

using ConfigStreamManagerTest = RegularChannelManagerTest;

TEST_F(ConfigStreamManagerTest, ConfigStreamManager) {
  ::testing::internal::CaptureStderr();
  ConfigStreamManager cfg = ConfigStreamManager(base_address, channels);
  // Expect empty stderr
  EXPECT_THAT(::testing::internal::GetCapturedStderr(), ::testing::IsEmpty());
  EXPECT_EQ(cfg.GetBaseAddress(), base_address);
  EXPECT_EQ(cfg.GetChannels().size(), channels.size());
  CompareChannels(cfg.GetChannels());
}

class ConfigDmaAxiStreamLikeManagerTest : public ChannelManagerTest {
 protected:
  void SetupChannels() override {
    std::vector<uint64_t> data_idxs = {0, 1, 2, 3, 4, 5, 6, 7};

    channels.clear();
    for (uint8_t i = 0; i < channels_num; i++) {
      absl::StatusOr<ConfigAxiStreamLikeChannel> cfg =
          ConfigAxiStreamLikeChannel::MakeConfigAxiStreamLikeChannel(
              "AxiStreamLikeChannel" + std::to_string(i), data_idxs,
              (channel_width * i), i, i, i);
      channels.push_back(cfg.value());
    }
  }

  void CompareChannels(
      std::vector<ConfigAxiStreamLikeChannel> tested_channels) {
    uint64_t i = 0;
    for (auto channel : tested_channels) {
      EXPECT_EQ(channel.GetName(), channels[i].GetName());
      EXPECT_EQ(channel.GetOffset(), channels[i].GetOffset());
      EXPECT_EQ(channel.GetDMAID(), channels[i].GetDMAID());
      EXPECT_EQ(channel.GetKeepIdx(), channels[i].GetKeepIdx());
      EXPECT_EQ(channel.GetLastIdx(), channels[i].GetLastIdx());
      EXPECT_EQ(channel.GetDataIdxs(), channels[i].GetDataIdxs());
      i++;
    }
  }

  std::vector<ConfigAxiStreamLikeChannel> channels;
};

TEST_F(ConfigDmaAxiStreamLikeManagerTest, ConfigDmaAxiStramLikeManager) {
  ::testing::internal::CaptureStderr();
  ConfigDmaAxiStreamLikeManager cfg =
      ConfigDmaAxiStreamLikeManager(base_address, channels);
  // Expect empty stderr
  EXPECT_THAT(::testing::internal::GetCapturedStderr(), ::testing::IsEmpty());
  EXPECT_EQ(cfg.GetBaseAddress(), base_address);
  EXPECT_EQ(cfg.GetChannels().size(), channels.size());
  CompareChannels(cfg.GetChannels());
}

class ConfigWriter {
 public:
  inline static const std::vector<std::string> import_path = {
      "arbitrary/import/path/a", "arbitrary/import/path/b",
      "arbitrary/import/path/c", "arbitrary/import/path/d"};

  inline static const std::string path_to_dslx_design =
      "arbitrary/path/to/dslx/design.x";
  inline static const std::string dslx_top_level_process =
      "dslx_top_level_process_name";
  inline static const std::string path_to_ir_design =
      "arbitrary/path/to/ir/design.ir";
  inline static const std::string ir_top_level_process =
      "ir_top_level_process_name";

  static const uint64_t address_mask = 0xFFFFFFFFFFFFFFFF;

  static const uint8_t max_single_value_channels = 64;
  static const uint64_t example_single_value_channel_width = 64;
  static const uint64_t svm_base_address = 0;
  // runtime status is the first channel in single value manager address space
  static const uint64_t svm_runtime_status_offset = svm_base_address;

  static const uint8_t max_stream_channels = 32;
  static const uint64_t example_stream_channel_width = 256;
  static const uint64_t sm_base_address =
      svm_base_address +
      (max_single_value_channels * example_single_value_channel_width);

  static const uint8_t max_dma_stream_channels = 16;
  static const uint64_t example_dma_stream_channel_width = 512;
  static const uint64_t dma_sm_base_address =
      sm_base_address + (max_stream_channels * example_stream_channel_width);

  static const uint8_t max_dma_axi_stream_channels = 8;
  static const uint64_t example_dma_axi_stream_channel_width = 2048;
  static const uint64_t dma_axi_stream_base_address =
      dma_sm_base_address +
      (max_dma_stream_channels * example_dma_stream_channel_width);

  void SetupChannelConfigProto(ConfigProto_Channel* cfg, std::string dslx_name,
                               std::string ir_name, uint64_t offset,
                               uint64_t dma_id) {
    cfg->set_dslx_name(dslx_name);
    cfg->set_ir_name(ir_name);
    cfg->set_in_manager_offset(offset);
    cfg->set_dma_id(dma_id);
  }

  void SetupAxiStreamLikeChannelConfigProto(
      ConfigProto_AXIStreamLikeChannel* cfg, std::string dslx_name,
      std::string ir_name, uint64_t offset, uint64_t dma_id,
      std::vector<uint64_t> data_idxs, uint64_t keep_idx, uint64_t last_idx) {
    ConfigProto_Channel* channel = cfg->mutable_base_config();
    SetupChannelConfigProto(channel, dslx_name, ir_name, offset, dma_id);
    for (auto data_idx : data_idxs) {
      cfg->add_dataidxs(data_idx);
    }
    cfg->set_keepidx(keep_idx);
    cfg->set_lastidx(last_idx);
  }

  void SetupSvmConfigProto(ConfigProto_SingleValueManager* cfg, uint64_t ba,
                           uint64_t runtime_status_offset, uint8_t max_channels,
                           uint64_t channel_width) {
    cfg->set_base_address(ba);
    cfg->set_runtime_status_offset(runtime_status_offset);

    for (uint8_t i = 0; i < max_channels; i++) {
      std::string dslx_name = "DslxSingleValueChannel" + std::to_string(i);
      std::string ir_name = "IrSingleValueChannel" + std::to_string(i);
      uint64_t offset = i * channel_width;
      uint64_t dma_id = i;
      ConfigProto_Channel* channel = cfg->add_single_value_channels();
      SetupChannelConfigProto(channel, dslx_name, ir_name, offset, dma_id);
    }
  }

  void SetupSmConfigProto(ConfigProto_StreamManager* cfg, uint64_t ba,
                          uint8_t max_channels, uint64_t channel_width) {
    cfg->set_base_address(ba);

    for (uint8_t i = 0; i < max_channels; i++) {
      std::string dslx_name = "DslxStreamChannel" + std::to_string(i);
      std::string ir_name = "IrStreamChannel" + std::to_string(i);
      uint64_t offset = i * channel_width;
      uint64_t dma_id = i;
      ConfigProto_Channel* channel = cfg->add_stream_channels();
      SetupChannelConfigProto(channel, dslx_name, ir_name, offset, dma_id);
    }
  }

  void SetupDmaAxiSmConfigProto(ConfigProto_DMAAXIStreamLikeManager* cfg,
                                uint64_t ba, uint8_t max_channels,
                                uint64_t channel_width) {
    cfg->set_base_address(ba);
    for (uint8_t i = 0; i < max_channels; i++) {
      std::string dslx_name = "DslxAxiStreamLikeChannel" + std::to_string(i);
      std::string ir_name = "IrAxiStreamLikeChannel" + std::to_string(i);
      uint64_t offset = i * channel_width;
      uint64_t dma_id = i;
      std::vector<uint64_t> data_idxs = {0, 1, 2, 3, 4, 5, 6, 7};
      uint64_t keep_idx = i;
      uint64_t last_idx = i;
      ConfigProto_AXIStreamLikeChannel* channel = cfg->add_axi_channels();
      SetupAxiStreamLikeChannelConfigProto(channel, dslx_name, ir_name, offset,
                                           dma_id, data_idxs, keep_idx,
                                           last_idx);
    }
  }

  ConfigProto& SetupConfigProto() {
    for (auto path : import_path) {
      cfg_proto.add_import_path(path);
    }

    cfg_proto.set_path_to_dslx_design(path_to_dslx_design);
    cfg_proto.set_dslx_top_level_process(dslx_top_level_process);
    cfg_proto.set_path_to_ir_design(path_to_ir_design);
    cfg_proto.set_ir_top_level_process(ir_top_level_process);
    cfg_proto.set_address_mask(address_mask);

    svm_config = cfg_proto.mutable_svm_config();
    SetupSvmConfigProto(svm_config, svm_base_address, svm_runtime_status_offset,
                        max_single_value_channels,
                        example_single_value_channel_width);

    sm_config = cfg_proto.mutable_sm_config();
    SetupSmConfigProto(sm_config, sm_base_address, max_stream_channels,
                       example_stream_channel_width);

    dma_sm_config = cfg_proto.mutable_dma_sm_config();
    SetupSmConfigProto(dma_sm_config, dma_sm_base_address,
                       max_dma_stream_channels,
                       example_dma_stream_channel_width);

    dma_axi_sm_config = cfg_proto.mutable_dma_axi_sm_config();
    SetupDmaAxiSmConfigProto(dma_axi_sm_config, dma_axi_stream_base_address,
                             max_dma_axi_stream_channels,
                             example_dma_axi_stream_channel_width);

    return cfg_proto;
  }

  void WriteBinaryConfigProto(std::string filename) {
    std::fstream output(filename,
                        std::ios::out | std::ios::trunc | std::ios::binary);
    if (!cfg_proto.SerializeToOstream(&output)) {
      XLS_LOG(ERROR) << "Failed to write binary config file";
    }
    output.close();
  }

  void WriteTextConfigProto(std::string filename) {
    int fd = open(filename.c_str(), O_CREAT | O_WRONLY | O_TRUNC,
                  S_IRUSR | S_IWUSR | S_IRGRP | S_IWGRP | S_IROTH | S_IWOTH);
    google::protobuf::io::FileOutputStream output =
        google::protobuf::io::FileOutputStream(fd);
    if (!google::protobuf::TextFormat::Print(cfg_proto, &output)) {
      XLS_LOG(ERROR) << "Failed to write text config file";
    }
    output.Close();
    close(fd);
  }

  ConfigProto cfg_proto;
  ConfigProto_SingleValueManager* svm_config;
  ConfigProto_StreamManager* sm_config;
  ConfigProto_StreamManager* dma_sm_config;
  ConfigProto_DMAAXIStreamLikeManager* dma_axi_sm_config;
};

class MakeProtoForConfigFileTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() { absl::SetFlag(&FLAGS_logtostderr, true); }
  explicit MakeProtoForConfigFileTest(std::string cfg_proto_file_path)
      : cfg_proto_file_path(cfg_proto_file_path) {
    cfg_writer.SetupConfigProto();
  }

  absl::StatusOr<std::unique_ptr<ConfigProto>> status_or_cfg_proto;
  ConfigWriter cfg_writer;
  ConfigType cfg_type;
  std::string cfg_proto_file_path;
};

class MakeProtoForConfigFileBinaryTest
    : public MakeProtoForConfigFileTest,
      public testing::WithParamInterface<ConfigType> {
 protected:
  MakeProtoForConfigFileBinaryTest()
      : MakeProtoForConfigFileTest("/tmp/cfg_proto.bin") {
    cfg_writer.WriteBinaryConfigProto(cfg_proto_file_path);
  }
};

TEST_P(MakeProtoForConfigFileBinaryTest, RoundTripCheck) {
  cfg_type = GetParam();

  ::testing::internal::CaptureStderr();
  status_or_cfg_proto = MakeProtoForConfigFile(cfg_proto_file_path, cfg_type);
  // Expect empty stderr
  EXPECT_THAT(::testing::internal::GetCapturedStderr(), ::testing::IsEmpty());

  if (cfg_type == ConfigType::kNone) {
    XLS_EXPECT_OK(status_or_cfg_proto);
    EXPECT_TRUE(google::protobuf::util::MessageDifferencer::Equivalent(
        *(status_or_cfg_proto.value()), cfg_writer.cfg_proto));
  } else if (cfg_type == ConfigType::kBinproto) {
    XLS_EXPECT_OK(status_or_cfg_proto);
    EXPECT_TRUE(google::protobuf::util::MessageDifferencer::Equivalent(
        *(status_or_cfg_proto.value()), cfg_writer.cfg_proto));
  } else if (cfg_type == ConfigType::kTextproto) {
    EXPECT_TRUE(absl::IsFailedPrecondition(status_or_cfg_proto.status()));
  } else {
    EXPECT_TRUE(absl::IsFailedPrecondition(status_or_cfg_proto.status()));
  }
}

INSTANTIATE_TEST_SUITE_P(MakeProtoForConfigFileBinaryTestParametrized,
                         MakeProtoForConfigFileBinaryTest,
                         testing::Values(ConfigType::kNone,
                                         ConfigType::kBinproto,
                                         ConfigType::kTextproto));

class MakeProtoForConfigFileTextTest
    : public MakeProtoForConfigFileTest,
      public testing::WithParamInterface<ConfigType> {
 protected:
  MakeProtoForConfigFileTextTest()
      : MakeProtoForConfigFileTest("/tmp/cfg_proto.txtpb") {
    cfg_writer.WriteTextConfigProto(cfg_proto_file_path);
  }
};

TEST_P(MakeProtoForConfigFileTextTest, RoundTripCheck) {
  cfg_type = GetParam();

  ::testing::internal::CaptureStderr();
  status_or_cfg_proto = MakeProtoForConfigFile(cfg_proto_file_path, cfg_type);
  // Expect empty stderr
  EXPECT_THAT(::testing::internal::GetCapturedStderr(), ::testing::IsEmpty());

  // Expected to fail because MakeProtoForConfigFile() with ConfigType::kNone
  // calls ResolveConfigType() which resolves the config type based on
  // absl flag 'FLAGS_config_type' which has default value set to 'binproto'.
  // This causes Text Proto file to be incorrectly parsed as Binary Proto file
  if (cfg_type == ConfigType::kNone) {
    EXPECT_TRUE(absl::IsFailedPrecondition(status_or_cfg_proto.status()));
  } else if (cfg_type == ConfigType::kBinproto) {
    EXPECT_TRUE(absl::IsFailedPrecondition(status_or_cfg_proto.status()));
  } else if (cfg_type == ConfigType::kTextproto) {
    XLS_EXPECT_OK(status_or_cfg_proto);
    EXPECT_TRUE(google::protobuf::util::MessageDifferencer::Equivalent(
        *(status_or_cfg_proto.value()), cfg_writer.cfg_proto));
  } else {
    EXPECT_TRUE(absl::IsFailedPrecondition(status_or_cfg_proto.status()));
  }
}

INSTANTIATE_TEST_SUITE_P(MakeProtoForConfigFileTextTestParametrized,
                         MakeProtoForConfigFileTextTest,
                         testing::Values(ConfigType::kNone,
                                         ConfigType::kBinproto,
                                         ConfigType::kTextproto));

class ResolveConfigTypeTest
    : public ::testing::Test,
      public ::testing::WithParamInterface<std::string> {
 protected:
  static void SetUpTestSuite() { absl::SetFlag(&FLAGS_logtostderr, true); }
};

TEST_P(ResolveConfigTypeTest, ResolveConfigType) {
  std::string cfg_type_str = GetParam();
  if (cfg_type_str != "default_cfg_type")
    absl::SetFlag(&FLAGS_config_type, cfg_type_str);
  ::testing::internal::CaptureStderr();
  absl::StatusOr<ConfigType> cfg_type = ResolveConfigType();
  // Expect empty stderr
  EXPECT_THAT(::testing::internal::GetCapturedStderr(), ::testing::IsEmpty());

  if (cfg_type_str == "textproto") {
    XLS_EXPECT_OK(cfg_type);
  } else if (cfg_type_str == "binproto") {
    XLS_EXPECT_OK(cfg_type);
  } else if (cfg_type_str == "default_cfg_type") {
    XLS_EXPECT_OK(cfg_type);  // cfg_type flag defaults to 'binproto'
  } else if (cfg_type_str == "") {
    EXPECT_TRUE(absl::IsInvalidArgument(cfg_type.status()));
  } else {
    EXPECT_TRUE(absl::IsInvalidArgument(cfg_type.status()));
  }
}

INSTANTIATE_TEST_SUITE_P(ResolveConfigTypeTestParametrized,
                         ResolveConfigTypeTest,
                         testing::Values("default_cfg_type", "",
                                         "not_recognized", "textproto",
                                         "binproto"));

class ConfigTest : public MakeProtoForConfigFileTest,
                   public testing::WithParamInterface<SimulationType> {
 protected:
  ConfigTest() : MakeProtoForConfigFileTest("/tmp/cfg_proto.bin") {
    cfg_writer.WriteBinaryConfigProto(cfg_proto_file_path);
    cfg_type = ConfigType::kNone;
  }

  void compare_channels(ConfigChannel* channel,
                        const ConfigProto_Channel& ref_channel,
                        SimulationType sim_type) {
    if (sim_type == SimulationType::kDSLX)
      EXPECT_EQ(channel->GetName(), ref_channel.dslx_name());
    else if (sim_type == SimulationType::kIR)
      EXPECT_EQ(channel->GetName(), ref_channel.ir_name());
    else
      FAIL() << "Unexpected SimulationType";

    XLS_EXPECT_OK_AND_EQ(channel->GetOffset(), ref_channel.in_manager_offset());
    XLS_EXPECT_OK_AND_EQ(channel->GetDMAID(), ref_channel.dma_id());
  }

  void compare_axi_stream_like_channels(
      ConfigAxiStreamLikeChannel* channel,
      const ConfigProto_AXIStreamLikeChannel& ref_channel,
      SimulationType sim_type) {
    compare_channels(channel, ref_channel.base_config(), sim_type);

    XLS_EXPECT_OK_AND_EQ(channel->GetKeepIdx(), ref_channel.keepidx());
    XLS_EXPECT_OK_AND_EQ(channel->GetLastIdx(), ref_channel.lastidx());

    auto data_idxs = channel->GetDataIdxs();
    EXPECT_GT(data_idxs.size(), 0);
    EXPECT_EQ(data_idxs.size(), ref_channel.dataidxs_size());
    for (int i = 0; i < data_idxs.size(); i++) {
      EXPECT_EQ(data_idxs[i], ref_channel.dataidxs(i));
    }
  }

  SimulationType sim_type;
};

TEST_P(ConfigTest, GetImportPaths) {
  XLS_ASSERT_OK_AND_ASSIGN(
      auto cfg_proto, MakeProtoForConfigFile(cfg_proto_file_path, cfg_type));
  ConfigProto reference = *cfg_proto;
  auto cfg_wrapper = Config(std::move(cfg_proto), GetParam());

  std::vector<std::filesystem::path> import_paths =
      cfg_wrapper.GetImportPaths();
  if (GetParam() == SimulationType::kDSLX) {
    EXPECT_GT(import_paths.size(), 0);
    EXPECT_EQ(import_paths.size(), reference.import_path_size());
  } else if (GetParam() == SimulationType::kIR) {
    EXPECT_EQ(import_paths.size(), 0);
  } else {
    FAIL() << "Unexpected SimulationType";
  }
  for (int i = 0; i < import_paths.size(); i++) {
    EXPECT_EQ(import_paths[i].string(), reference.import_path(i));
  }
}

TEST_P(ConfigTest, GetDesignPath) {
  XLS_ASSERT_OK_AND_ASSIGN(
      auto cfg_proto, MakeProtoForConfigFile(cfg_proto_file_path, cfg_type));
  ConfigProto reference = *cfg_proto;
  auto cfg_wrapper = Config(std::move(cfg_proto), GetParam());

  XLS_ASSERT_OK_AND_ASSIGN(std::filesystem::path design_path,
                           cfg_wrapper.GetDesignPath());
  if (GetParam() == SimulationType::kDSLX) {
    EXPECT_EQ(design_path.string(), reference.path_to_dslx_design());
  } else if (GetParam() == SimulationType::kIR) {
    EXPECT_EQ(design_path.string(), reference.path_to_ir_design());
  } else {
    FAIL() << "Unexpected SimulationType";
  }
}

TEST_P(ConfigTest, GetDesignName) {
  XLS_ASSERT_OK_AND_ASSIGN(
      auto cfg_proto, MakeProtoForConfigFile(cfg_proto_file_path, cfg_type));
  ConfigProto reference = *cfg_proto;
  auto cfg_wrapper = Config(std::move(cfg_proto), GetParam());

  XLS_ASSERT_OK_AND_ASSIGN(std::string design_name,
                           cfg_wrapper.GetDesignName());
  if (GetParam() == SimulationType::kDSLX) {
    EXPECT_EQ(design_name, reference.dslx_top_level_process());
  } else if (GetParam() == SimulationType::kIR) {
    EXPECT_EQ(design_name, reference.ir_top_level_process());
  } else {
    FAIL() << "Unexpected SimulationType";
  }
}

TEST_P(ConfigTest, GetAddressMask) {
  XLS_ASSERT_OK_AND_ASSIGN(
      auto cfg_proto, MakeProtoForConfigFile(cfg_proto_file_path, cfg_type));
  ConfigProto reference = *cfg_proto;
  auto cfg_wrapper = Config(std::move(cfg_proto), GetParam());

  XLS_EXPECT_OK_AND_EQ(cfg_wrapper.GetAddressMask(), reference.address_mask());
}

TEST_P(ConfigTest, GetSingleValueManagerConfig) {
  XLS_ASSERT_OK_AND_ASSIGN(
      auto cfg_proto, MakeProtoForConfigFile(cfg_proto_file_path, cfg_type));
  ConfigProto reference = *cfg_proto;
  auto ref_svm_cfg = reference.svm_config();
  sim_type = GetParam();
  auto cfg_wrapper = Config(std::move(cfg_proto), sim_type);

  XLS_ASSERT_OK_AND_ASSIGN(auto svm_cfg,
                           cfg_wrapper.GetSingleValueManagerConfig());
  EXPECT_EQ(svm_cfg->GetBaseAddress(), ref_svm_cfg.base_address());
  EXPECT_EQ(svm_cfg->GetRuntimeStatusOffset(),
            ref_svm_cfg.runtime_status_offset());
  auto channels = svm_cfg->GetChannels();
  EXPECT_GT(channels.size(), 0);
  EXPECT_EQ(channels.size(), ref_svm_cfg.single_value_channels_size());
  for (int i = 0; i < channels.size(); i++) {
    compare_channels(&channels[i], ref_svm_cfg.single_value_channels(i),
                     sim_type);
  }
}

TEST_P(ConfigTest, GetStreamManagerConfig) {
  XLS_ASSERT_OK_AND_ASSIGN(
      auto cfg_proto, MakeProtoForConfigFile(cfg_proto_file_path, cfg_type));
  ConfigProto reference = *cfg_proto;
  auto ref_sm_cfg = reference.sm_config();
  sim_type = GetParam();
  auto cfg_wrapper = Config(std::move(cfg_proto), sim_type);

  XLS_ASSERT_OK_AND_ASSIGN(auto sm_cfg, cfg_wrapper.GetStreamManagerConfig());
  EXPECT_EQ(sm_cfg->GetBaseAddress(), ref_sm_cfg.base_address());
  auto channels = sm_cfg->GetChannels();
  EXPECT_GT(channels.size(), 0);
  EXPECT_EQ(channels.size(), ref_sm_cfg.stream_channels_size());
  for (int i = 0; i < channels.size(); i++) {
    compare_channels(&channels[i], ref_sm_cfg.stream_channels(i), sim_type);
  }
}

TEST_P(ConfigTest, GetDmaStreamManagerConfig) {
  XLS_ASSERT_OK_AND_ASSIGN(
      auto cfg_proto, MakeProtoForConfigFile(cfg_proto_file_path, cfg_type));
  ConfigProto reference = *cfg_proto;
  auto ref_dma_sm_cfg = reference.dma_sm_config();
  sim_type = GetParam();
  auto cfg_wrapper = Config(std::move(cfg_proto), sim_type);

  XLS_ASSERT_OK_AND_ASSIGN(auto dma_sm_cfg,
                           cfg_wrapper.GetDmaStreamManagerConfig());
  EXPECT_EQ(dma_sm_cfg->GetBaseAddress(), ref_dma_sm_cfg.base_address());
  auto channels = dma_sm_cfg->GetChannels();
  EXPECT_GT(channels.size(), 0);
  EXPECT_EQ(channels.size(), ref_dma_sm_cfg.stream_channels_size());
  for (int i = 0; i < channels.size(); i++) {
    compare_channels(&channels[i], ref_dma_sm_cfg.stream_channels(i), sim_type);
  }
}

TEST_P(ConfigTest, GetDmaAxiStreamLikeManagerConfig) {
  XLS_ASSERT_OK_AND_ASSIGN(
      auto cfg_proto, MakeProtoForConfigFile(cfg_proto_file_path, cfg_type));
  ConfigProto reference = *cfg_proto;
  auto ref_dma_axi_sm_cfg = reference.dma_axi_sm_config();
  sim_type = GetParam();
  auto cfg_wrapper = Config(std::move(cfg_proto), sim_type);

  XLS_ASSERT_OK_AND_ASSIGN(auto dma_axi_sm_cfg,
                           cfg_wrapper.GetDmaAxiStreamLikeManagerConfig());
  EXPECT_EQ(dma_axi_sm_cfg->GetBaseAddress(),
            ref_dma_axi_sm_cfg.base_address());
  auto channels = dma_axi_sm_cfg->GetChannels();
  EXPECT_GT(channels.size(), 0);
  EXPECT_EQ(channels.size(), ref_dma_axi_sm_cfg.axi_channels_size());
  for (int i = 0; i < channels.size(); i++) {
    compare_axi_stream_like_channels(
        &channels[i], ref_dma_axi_sm_cfg.axi_channels(i), sim_type);
  }
}

INSTANTIATE_TEST_SUITE_P(ConfigTestParametrized, ConfigTest,
                         testing::Values(SimulationType::kDSLX,
                                         SimulationType::kIR));

}  // namespace
}  // namespace xls::simulation::generic
