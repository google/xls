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

#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xls/common/file/filesystem.h"
#include "xls/common/logging/logging.h"
#include "xls/common/status/status_macros.h"

ABSL_FLAG(
    std::string, config_type, "binproto",
    "Configuration file format, can be either: 'textproto', or 'binproto'");

namespace xls::simulation::generic {

absl::StatusOr<ConfigType> ResolveConfigType() {
  std::string config_type_str = absl::GetFlag(FLAGS_config_type);
  absl::StatusOr<ConfigType> config_type;
  if (config_type_str == "textproto") {
    config_type = ConfigType::kTextproto;
  } else if (config_type_str == "binproto") {
    config_type = ConfigType::kBinproto;
  } else {
    return absl::InvalidArgumentError(
        "Invalid -config_type flag: " + config_type_str +
        "; must be one of textproto|binproto");
  }

  return config_type;
}

absl::StatusOr<std::unique_ptr<ConfigProto>> MakeProtoForConfigFile(
    std::string_view config_file_path, ConfigType config_type) {
  absl::Status success;
  absl::StatusOr<std::unique_ptr<ConfigProto>> design_config =
      std::make_unique<ConfigProto>();

  if (config_type == ConfigType::kNone) {
    ConfigType& temp(config_type);
    XLS_ASSIGN_OR_RETURN(temp, ResolveConfigType());
  }
  if (config_type == ConfigType::kTextproto) {
    success = ParseTextProtoFile(config_file_path, design_config.value().get());
  } else {
    success = ParseProtobinFile(config_file_path, design_config.value().get());
  }
  if (!success.ok()) {
    design_config = success;
  }
  return design_config;
}

Config::Config(std::unique_ptr<ConfigProto> proto, SimulationType sim_type)
    : proto_(std::move(proto)), sim_type_(sim_type) {}

std::vector<std::filesystem::path> Config::GetImportPaths() const {
  std::vector<std::filesystem::path> import_paths;
  if (this->sim_type_ == SimulationType::kDSLX) {
    import_paths.reserve(this->proto_->import_path_size());
    for (int i = 0; i < this->proto_->import_path_size(); ++i) {
      import_paths.push_back(
          std::filesystem::path(this->proto_->import_path(i)));
    }
  }
  return import_paths;
}

absl::StatusOr<std::filesystem::path> Config::GetDesignPath() const {
  std::string type("DSLX");
  if (this->sim_type_ == SimulationType::kIR) type = std::string("IR");
  absl::StatusOr<std::filesystem::path> result =
      absl::NotFoundError("Config has no value for " + type + " file path");
  if (this->sim_type_ == SimulationType::kDSLX) {
    if (this->proto_->has_path_to_dslx_design())
      result = std::filesystem::path(this->proto_->path_to_dslx_design());
  } else if (this->sim_type_ == SimulationType::kIR) {
    if (this->proto_->has_path_to_ir_design())
      result = std::filesystem::path(this->proto_->path_to_ir_design());
  } else {
    return absl::InvalidArgumentError("Invalid sim type");
  }
  return result;
}

absl::StatusOr<std::string> Config::GetDesignName() const {
  std::string type("DSLX");
  if (this->sim_type_ == SimulationType::kIR) type = std::string("IR");
  absl::StatusOr<std::string> result =
      absl::NotFoundError("Config has no value for " + type + " file path");
  if (this->sim_type_ == SimulationType::kDSLX) {
    if (this->proto_->has_dslx_top_level_process())
      result = this->proto_->dslx_top_level_process();
  } else if (this->sim_type_ == SimulationType::kIR) {
    if (this->proto_->has_ir_top_level_process())
      result = this->proto_->ir_top_level_process();
  }
  return result;
}

absl::StatusOr<uint64_t> Config::GetAddressMask() const {
  uint64_t value = this->proto_->address_mask();
  absl::StatusOr<uint64_t> result = value;
  char hex_result_[100];
  snprintf(hex_result_, sizeof(hex_result_) - 1, "%" PRIx64, value);
  std::string hex_result(hex_result_);
  value += 1;
  if ((value & (value >> 1)) != 0)
    result = absl::InvalidArgumentError("Address Mask " + hex_result +
                                        " is not continous");
  return result;
}

absl::StatusOr<std::optional<ConfigSingleValueManager>>
Config::GetSingleValueManagerConfig() const {
  if (!this->proto_->has_svm_config()) {
    return std::nullopt;
  }
  auto svm_config_ = this->proto_->svm_config();
  std::vector<ConfigChannel> channels_;
  channels_.reserve(svm_config_.single_value_channels_size());
  for (int i = 0; i < svm_config_.single_value_channels_size(); ++i) {
    auto channel_config = svm_config_.single_value_channels(i);
    std::string name;
    if (this->sim_type_ == SimulationType::kDSLX) {
      if (!channel_config.has_dslx_name()) continue;
      name = channel_config.dslx_name();
    } else if (this->sim_type_ == SimulationType::kIR) {
      if (!channel_config.has_ir_name()) continue;
      name = channel_config.ir_name();
    }
    std::optional<uint64_t> ba_ = std::nullopt;
    if (channel_config.has_in_manager_offset())
      ba_ = channel_config.in_manager_offset();
    std::optional<uint64_t> id_ = std::nullopt;
    if (channel_config.has_dma_id()) id_ = channel_config.dma_id();
    XLS_ASSIGN_OR_RETURN(ConfigChannel channel_,
                         ConfigChannel::MakeConfigChannel(name, ba_, id_));
    channels_.push_back(channel_);
  }
  return ConfigSingleValueManager(svm_config_.base_address(),
                                  svm_config_.runtime_status_offset(),
                                  channels_);
}

absl::StatusOr<std::optional<ConfigStreamManager>>
Config::GetStreamManagerConfig() const {
  if (!this->proto_->has_sm_config()) {
    return std::nullopt;
  }
  auto sm_config_ = this->proto_->sm_config();
  std::vector<ConfigChannel> channels_;
  channels_.reserve(sm_config_.stream_channels_size());
  for (int i = 0; i < sm_config_.stream_channels_size(); ++i) {
    auto channel_config = sm_config_.stream_channels(i);
    std::string name;
    if (this->sim_type_ == SimulationType::kDSLX) {
      if (!channel_config.has_dslx_name()) continue;
      name = channel_config.dslx_name();
    } else if (this->sim_type_ == SimulationType::kIR) {
      if (!channel_config.has_ir_name()) continue;
      name = channel_config.ir_name();
    }
    std::optional<uint64_t> ba_ = std::nullopt;
    if (channel_config.has_in_manager_offset())
      ba_ = channel_config.in_manager_offset();
    std::optional<uint64_t> id_ = std::nullopt;
    if (channel_config.has_dma_id()) id_ = channel_config.dma_id();
    XLS_ASSIGN_OR_RETURN(ConfigChannel channel_,
                         ConfigChannel::MakeConfigChannel(name, ba_, id_));
    channels_.push_back(channel_);
  }
  return ConfigStreamManager(sm_config_.base_address(), channels_);
}

absl::StatusOr<std::optional<ConfigStreamManager>>
Config::GetDmaStreamManagerConfig() const {
  if (!this->proto_->has_dma_sm_config()) {
    return std::nullopt;
  }
  auto dma_sm_config_ = this->proto_->dma_sm_config();
  std::vector<ConfigChannel> channels_;
  channels_.reserve(dma_sm_config_.stream_channels_size());
  for (int i = 0; i < dma_sm_config_.stream_channels_size(); ++i) {
    auto channel_config = dma_sm_config_.stream_channels(i);
    std::string name;
    if (this->sim_type_ == SimulationType::kDSLX) {
      if (!channel_config.has_dslx_name()) continue;
      name = channel_config.dslx_name();
    } else if (this->sim_type_ == SimulationType::kIR) {
      if (!channel_config.has_ir_name()) continue;
      name = channel_config.ir_name();
    }
    std::optional<uint64_t> ba_ = std::nullopt;
    if (channel_config.has_in_manager_offset())
      ba_ = channel_config.in_manager_offset();
    std::optional<uint64_t> id_ = std::nullopt;
    if (channel_config.has_dma_id()) id_ = channel_config.dma_id();
    XLS_ASSIGN_OR_RETURN(ConfigChannel channel_,
                         ConfigChannel::MakeConfigChannel(name, ba_, id_));
    channels_.push_back(channel_);
  }
  return ConfigStreamManager(dma_sm_config_.base_address(), channels_);
}

ConfigDmaAxiStreamLikeManager::ConfigDmaAxiStreamLikeManager(
    uint64_t base_address, std::vector<ConfigAxiStreamLikeChannel> channels)
    : base_address_(base_address), channels_(channels) {}

uint64_t ConfigDmaAxiStreamLikeManager::GetBaseAddress() const {
  return this->base_address_;
}

const std::vector<ConfigAxiStreamLikeChannel>&
ConfigDmaAxiStreamLikeManager::GetChannels() const {
  return this->channels_;
}

absl::StatusOr<std::optional<ConfigDmaAxiStreamLikeManager>>
Config::GetDmaAxiStreamLikeManagerConfig() const {
  if (!this->proto_->has_dma_axi_sm_config()) {
    return std::nullopt;
  }
  auto dma_axi_sm_config_ = this->proto_->dma_axi_sm_config();
  std::vector<ConfigAxiStreamLikeChannel> channels_;
  channels_.reserve(dma_axi_sm_config_.axi_channels_size());
  for (int i = 0; i < dma_axi_sm_config_.axi_channels_size(); ++i) {
    auto axi_config = dma_axi_sm_config_.axi_channels(i);
    auto base_config = axi_config.base_config();
    std::string name;
    if (this->sim_type_ == SimulationType::kDSLX) {
      if (!base_config.has_dslx_name()) continue;
      name = base_config.dslx_name();
    } else if (this->sim_type_ == SimulationType::kIR) {
      if (!base_config.has_ir_name()) continue;
      name = base_config.ir_name();
    }
    std::optional<uint64_t> ba_ = std::nullopt;
    if (base_config.has_in_manager_offset())
      ba_ = base_config.in_manager_offset();
    std::optional<uint64_t> id_ = std::nullopt;
    if (base_config.has_dma_id()) id_ = base_config.dma_id();

    std::vector<uint64_t> data_idxs_;
    data_idxs_.reserve(axi_config.dataidxs_size());
    for (int j = 0; j < axi_config.dataidxs_size(); ++j) {
      data_idxs_.push_back(axi_config.dataidxs(j));
    }

    std::optional<uint64_t> keep_ = std::nullopt;
    if (axi_config.has_keepidx()) keep_ = axi_config.keepidx();
    std::optional<uint64_t> last_ = std::nullopt;
    if (axi_config.has_lastidx()) last_ = axi_config.lastidx();

    XLS_ASSIGN_OR_RETURN(
        ConfigAxiStreamLikeChannel channel_,
        ConfigAxiStreamLikeChannel::MakeConfigAxiStreamLikeChannel(
            name, data_idxs_, ba_, id_, keep_, last_));
    channels_.push_back(channel_);
  }
  return ConfigDmaAxiStreamLikeManager(dma_axi_sm_config_.base_address(),
                                       channels_);
}

/* static */ absl::StatusOr<ConfigChannel> ConfigChannel::MakeConfigChannel(
    std::string name, std::optional<uint64_t> offset,
    std::optional<uint64_t> dma_id) {
  absl::StatusOr<ConfigChannel> result = absl::InvalidArgumentError(
      "Channel \"" + name + "\" has invalid config!");
  if (offset.has_value() || dma_id.has_value())
    result = ConfigChannel(name, offset, dma_id);
  XLS_LOG_IF(ERROR, !result.ok())
      << "Channel \"" << name << "\" does not define offset or DMA ID";
  return result;
}

ConfigChannel::ConfigChannel(std::string name, std::optional<uint64_t> offset,
                             std::optional<uint64_t> dma_id)
    : name_(name), offset_(offset), dma_id_(dma_id) {}

std::string_view ConfigChannel::GetName() const { return this->name_; }

absl::StatusOr<uint64_t> ConfigChannel::GetOffset() const {
  absl::StatusOr<uint64_t> result =
      absl::NotFoundError(this->name_ + " doesn't define offset.");
  if (this->offset_.has_value()) result = this->offset_.value();
  return result;
}

absl::StatusOr<uint64_t> ConfigChannel::GetDMAID() const {
  absl::StatusOr<uint64_t> result =
      absl::NotFoundError(this->name_ + " doesn't define DMA ID.");
  if (this->dma_id_.has_value()) result = this->dma_id_.value();
  return result;
}

ConfigSingleValueManager::ConfigSingleValueManager(
    uint64_t base_address, uint64_t runtime_status_offset,
    std::vector<ConfigChannel> channels)
    : base_address_(base_address),
      runtime_status_offset_(runtime_status_offset),
      channels_(channels) {}

uint64_t ConfigSingleValueManager::GetBaseAddress() const {
  return this->base_address_;
}

uint64_t ConfigSingleValueManager::GetRuntimeStatusOffset() const {
  return this->runtime_status_offset_;
}

const std::vector<ConfigChannel>& ConfigSingleValueManager::GetChannels()
    const {
  return this->channels_;
}

ConfigStreamManager::ConfigStreamManager(uint64_t base_address,
                                         std::vector<ConfigChannel> channels)
    : base_address_(base_address), channels_(channels) {}

uint64_t ConfigStreamManager::GetBaseAddress() const {
  return this->base_address_;
}

const std::vector<ConfigChannel>& ConfigStreamManager::GetChannels() const {
  return this->channels_;
}

/* static */ absl::StatusOr<ConfigAxiStreamLikeChannel>
ConfigAxiStreamLikeChannel::MakeConfigAxiStreamLikeChannel(
    std::string name, std::vector<uint64_t> data_idxs,
    std::optional<uint64_t> base_address, std::optional<uint64_t> dma_id,
    std::optional<uint64_t> keep_idx, std::optional<uint64_t> last_idx) {
  absl::StatusOr<ConfigAxiStreamLikeChannel> result =
      absl::InvalidArgumentError(name + " has invalid config!");
  bool has_address = base_address.has_value() || dma_id.has_value();
  bool has_data = data_idxs.size() > 0;
  if (has_address && has_data)
    result = ConfigAxiStreamLikeChannel(name, data_idxs, base_address, dma_id,
                                        keep_idx, last_idx);

  XLS_LOG_IF(ERROR, !has_address)
      << name << " doesn't define base address nor DMA ID";

  XLS_LOG_IF(ERROR, !has_data)
      << "Channel \"" << name << "\" does not define any data indices";
  return result;
}

absl::StatusOr<uint64_t> ConfigAxiStreamLikeChannel::GetLastIdx() const {
  absl::StatusOr<uint64_t> result =
      absl::NotFoundError(this->name_ + " doesn't define last index.");
  if (this->last_idx_.has_value()) result = this->last_idx_.value();
  return result;
}

absl::StatusOr<uint64_t> ConfigAxiStreamLikeChannel::GetKeepIdx() const {
  absl::StatusOr<uint64_t> result =
      absl::NotFoundError(this->name_ + " doesn't define keep index.");
  if (this->keep_idx_.has_value()) result = this->keep_idx_.value();
  return result;
}

const std::vector<uint64_t>& ConfigAxiStreamLikeChannel::GetDataIdxs() const {
  return this->data_idxs_;
}

ConfigAxiStreamLikeChannel::ConfigAxiStreamLikeChannel(
    std::string name, std::vector<uint64_t> data_idxs,
    std::optional<uint64_t> offset, std::optional<uint64_t> dma_id,
    std::optional<uint64_t> keep_idx, std::optional<uint64_t> last_idx)
    : ConfigChannel(name, offset, dma_id),
      keep_idx_(keep_idx),
      last_idx_(last_idx),
      data_idxs_(data_idxs) {}

}  // namespace xls::simulation::generic
