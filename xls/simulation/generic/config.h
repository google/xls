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

#ifndef XLS_SIMULATION_GENERIC_CONFIG_H_
#define XLS_SIMULATION_GENERIC_CONFIG_H_

#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/flags/declare.h"
#include "absl/flags/flag.h"
#include "absl/status/statusor.h"
#include "xls/simulation/generic/config.pb.h"

ABSL_DECLARE_FLAG(std::string, config_type);

namespace xls::simulation::generic {

enum class ConfigType {
  kTextproto,
  kBinproto,
  kNone,
};

enum class SimulationType {
  kDSLX,
  kIR,
};

absl::StatusOr<std::unique_ptr<ConfigProto>> MakeProtoForConfigFile(
    std::string_view, ConfigType = ConfigType::kNone);

absl::StatusOr<ConfigType> ResolveConfigType();

class ConfigSingleValueManager;
class ConfigStreamManager;
class ConfigDmaAxiStreamLikeManager;

class Config {
 public:
  Config(std::unique_ptr<ConfigProto>, SimulationType);

  std::vector<std::filesystem::path> GetImportPaths() const;
  absl::StatusOr<std::filesystem::path> GetDesignPath() const;
  absl::StatusOr<std::string> GetDesignName() const;
  absl::StatusOr<uint64_t> GetAddressMask() const;

  absl::StatusOr<std::optional<ConfigSingleValueManager>>
  GetSingleValueManagerConfig() const;
  absl::StatusOr<std::optional<ConfigStreamManager>> GetStreamManagerConfig()
      const;
  absl::StatusOr<std::optional<ConfigStreamManager>> GetDmaStreamManagerConfig()
      const;
  absl::StatusOr<std::optional<ConfigDmaAxiStreamLikeManager>>
  GetDmaAxiStreamLikeManagerConfig() const;

 private:
  Config();
  std::unique_ptr<ConfigProto> proto_;
  SimulationType sim_type_;
};

class ConfigChannel {
 public:
  static absl::StatusOr<ConfigChannel> MakeConfigChannel(
      std::string, std::optional<uint64_t> = std::nullopt,
      std::optional<uint64_t> = std::nullopt);

  std::string_view GetName() const;
  absl::StatusOr<uint64_t> GetOffset() const;
  absl::StatusOr<uint64_t> GetDMAID() const;

 protected:
  ConfigChannel();
  ConfigChannel(std::string, std::optional<uint64_t> = std::nullopt,
                std::optional<uint64_t> = std::nullopt);

  std::string name_;
  std::optional<uint64_t> offset_;
  std::optional<uint64_t> dma_id_;
};

class ConfigSingleValueManager {
 public:
  ConfigSingleValueManager(uint64_t, uint64_t, std::vector<ConfigChannel>);

  uint64_t GetBaseAddress() const;
  uint64_t GetRuntimeStatusOffset() const;
  const std::vector<ConfigChannel>& GetChannels() const;

 private:
  ConfigSingleValueManager();
  uint64_t base_address_;
  uint64_t runtime_status_offset_;
  std::vector<ConfigChannel> channels_;
};

class ConfigStreamManager {
 public:
  ConfigStreamManager(uint64_t, std::vector<ConfigChannel>);

  uint64_t GetBaseAddress() const;
  const std::vector<ConfigChannel>& GetChannels() const;

 private:
  ConfigStreamManager();
  uint64_t base_address_;
  std::vector<ConfigChannel> channels_;
};

class ConfigAxiStreamLikeChannel : public ConfigChannel {
 public:
  static absl::StatusOr<ConfigAxiStreamLikeChannel>
      MakeConfigAxiStreamLikeChannel(std::string, std::vector<uint64_t>,
                                     std::optional<uint64_t> = std::nullopt,
                                     std::optional<uint64_t> = std::nullopt,
                                     std::optional<uint64_t> = std::nullopt,
                                     std::optional<uint64_t> = std::nullopt);

  absl::StatusOr<uint64_t> GetKeepIdx() const;
  absl::StatusOr<uint64_t> GetLastIdx() const;
  const std::vector<uint64_t>& GetDataIdxs() const;

 private:
  ConfigAxiStreamLikeChannel();
  ConfigAxiStreamLikeChannel(std::string, std::vector<uint64_t>,
                             std::optional<uint64_t> = std::nullopt,
                             std::optional<uint64_t> = std::nullopt,
                             std::optional<uint64_t> = std::nullopt,
                             std::optional<uint64_t> = std::nullopt);

  std::optional<uint64_t> keep_idx_;
  std::optional<uint64_t> last_idx_;
  std::vector<uint64_t> data_idxs_;
};

class ConfigDmaAxiStreamLikeManager {
 public:
  ConfigDmaAxiStreamLikeManager(uint64_t,
                                std::vector<ConfigAxiStreamLikeChannel>);

  uint64_t GetBaseAddress() const;
  const std::vector<ConfigAxiStreamLikeChannel>& GetChannels() const;

 private:
  ConfigDmaAxiStreamLikeManager();
  uint64_t base_address_;
  std::vector<ConfigAxiStreamLikeChannel> channels_;
};

}  // namespace xls::simulation::generic
#endif  // XLS_SIMULATION_GENERIC_CONFIG_H_
