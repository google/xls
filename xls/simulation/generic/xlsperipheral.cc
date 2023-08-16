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

#include "xls/simulation/generic/xlsperipheral.h"

#include <cstdint>
#include <initializer_list>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>

#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "xls/common/file/filesystem.h"
#include "xls/common/logging/logging.h"
#include "xls/common/status/ret_check.h"
#include "xls/ir/ir_parser.h"
#include "xls/ir/package.h"
#include "xls/simulation/generic/axi_stream_like_dma_endpoint.h"
#include "xls/simulation/generic/config.h"
#include "xls/simulation/generic/dmastreammanager.h"
#include "xls/simulation/generic/ichannelmanager.h"
#include "xls/simulation/generic/iconnection.h"
#include "xls/simulation/generic/ir_axistreamlike.h"
#include "xls/simulation/generic/ir_single_value.h"
#include "xls/simulation/generic/ir_stream.h"
#include "xls/simulation/generic/runtime_manager.h"
#include "xls/simulation/generic/singlevaluemanager.h"
#include "xls/simulation/generic/stream_dma_endpoint.h"
#include "xls/simulation/generic/streammanager.h"

namespace xls::simulation::generic {

// Load config file
// Assumptions: text proto, IR simulation
// TODO(rdobrodii): 2023-09-05 extract this from the context string?
absl::StatusOr<Config> LoadConfig(std::string_view context) {
  ConfigType config_type = ConfigType::kTextproto;
  SimulationType sim_type = SimulationType::kIR;

  auto proto = MakeProtoForConfigFile(context, config_type);
  if (!proto.ok()) {
    return absl::InternalError(
        absl::StrFormat("Error loading config file '%s': %s", context,
                        proto.status().ToString()));
  }
  return Config(std::move(proto.value()), sim_type);
}

absl::StatusOr<std::optional<StreamManager>> SetUpStreamManager(
    const Config& config, RuntimeManager* runtime) {
  absl::StatusOr<std::optional<ConfigStreamManager>> sm_config =
      config.GetStreamManagerConfig();
  if (!sm_config.ok()) {
    XLS_LOG(WARNING) << "Error loading Stream configuration: "
                     << sm_config.status().ToString();
    return std::nullopt;
  }
  if (!sm_config->has_value()) {
    return std::nullopt;
  }

  absl::StatusOr<StreamManager> sm = StreamManager::Build(
      sm_config->value().GetBaseAddress(),
      [&](const auto& RegisterStream) -> absl::Status {
        for (auto& ch_config : sm_config->value().GetChannels()) {
          std::string_view ch_name = ch_config.GetName();

          XLS_ASSIGN_OR_RETURN(
              auto* queue,
              runtime->runtime().queue_manager().GetQueueByName(ch_name));

          if (queue->channel()->kind() != ChannelKind::kStreaming) {
            return absl::InternalError(
                absl::StrFormat("Channel \"%s\" is not a stream.", ch_name));
          }

          XLS_ASSIGN_OR_RETURN(IRStream stream, IRStream::MakeIRStream(queue));

          XLS_ASSIGN_OR_RETURN(IChannelManager::channel_addr_t offset,
                               ch_config.GetOffset());

          XLS_LOG(INFO) << "Registering channel \"" << ch_name << "\"";
          RegisterStream(offset, new IRStream(std::move(stream)));
        }
        return absl::OkStatus();
      });
  if (!sm.ok()) {
    return sm.status();
  }
  return std::make_optional(std::move(sm.value()));
}

absl::StatusOr<std::unique_ptr<Package>> LoadPackage(const Config& config) {
  // Describe the config
  XLS_ASSIGN_OR_RETURN(std::filesystem::path design_path,
                       config.GetDesignPath());
  XLS_LOG(INFO) << "Setting up simulation of the following design: "
                << design_path;

  XLS_ASSIGN_OR_RETURN(std::string design_contents,
                       xls::GetFileContents(design_path));

  auto package = xls::Parser::ParsePackage(design_contents);
  if (!package.ok()) {
    XLS_LOG(FATAL) << "Failed to parse package " << design_path
                   << "\n Reason: " << package.status();
  }
  return std::move(package.value());
}

static absl::StatusOr<std::optional<SingleValueManager>>
SetUpSingleValueManager(const Config& config, RuntimeManager* runtime) {
  absl::StatusOr<std::optional<ConfigSingleValueManager>> svm_config =
      config.GetSingleValueManagerConfig();
  if (!svm_config.ok()) {
    XLS_LOG(WARNING) << "Error loading SingleValue configuration: "
                     << svm_config.status().ToString();
    return std::nullopt;
  }
  if (!svm_config->has_value()) {
    return std::nullopt;
  }
  if (runtime == nullptr) {
    return absl::InvalidArgumentError("Undefined runtime!");
  }
  SingleValueManager svm(svm_config->value().GetBaseAddress());
  XLS_RETURN_IF_ERROR(
      svm.RegisterIRegister(std::move(runtime->GetRuntimeStatusOwnership()),
                            svm_config->value().GetRuntimeStatusOffset()));

  for (auto& ch_config : svm_config->value().GetChannels()) {
    std::string_view ch_name = ch_config.GetName();

    XLS_ASSIGN_OR_RETURN(
        auto* queue,
        runtime->runtime().queue_manager().GetQueueByName(ch_name));

    if (queue->channel()->kind() != ChannelKind::kSingleValue) {
      return absl::InternalError(
          absl::StrFormat("Channel \"%s\" is not a register.", ch_name));
    }

    XLS_ASSIGN_OR_RETURN(IRSingleValue singlevalue,
                         IRSingleValue::MakeIRSingleValue(queue));

    XLS_ASSIGN_OR_RETURN(uint64_t offset, ch_config.GetOffset());

    XLS_LOG(INFO) << "Registering single value channel \"" << ch_name << "\"";
    XLS_RETURN_IF_ERROR(svm.RegisterIRegister(
        std::make_unique<IRSingleValue>(std::move(singlevalue)), offset));
  }

  return svm;
}

static absl::StatusOr<IRStream> CreateStreamFromConfig(
    RuntimeManager* runtime, const ConfigChannel& config) {
  std::string_view ch_name = config.GetName();
  XLS_ASSIGN_OR_RETURN(
      auto* queue, runtime->runtime().queue_manager().GetQueueByName(ch_name));

  if (queue->channel()->kind() != ChannelKind::kStreaming) {
    return absl::InternalError(
        absl::StrFormat("Channel \"%s\" is not a stream.", ch_name));
  }

  XLS_ASSIGN_OR_RETURN(IRStream stream, IRStream::MakeIRStream(queue));
  XLS_LOG(INFO) << "Created channel \"" << ch_name << "\"";

  return std::move(stream);
}

static absl::StatusOr<std::optional<DmaStreamManager>> SetUpDmaStreamManager(
    IMasterPort* bus_master_port, const Config& config,
    RuntimeManager* runtime) {
  // DMA Stream Manager not declared in config, skip it
  absl::StatusOr<std::optional<ConfigStreamManager>> dma_config =
      config.GetDmaStreamManagerConfig();
  if (!dma_config.ok()) {
    XLS_LOG(WARNING) << "Error loading DMA configuration: "
                     << dma_config.status().ToString();
    return std::nullopt;
  }
  if (!dma_config->has_value()) {
    return std::nullopt;
  }

  DmaStreamManager dma = DmaStreamManager(dma_config->value().GetBaseAddress());
  for (auto& ch_config : dma_config->value().GetChannels()) {
    XLS_ASSIGN_OR_RETURN(auto stream,
                         CreateStreamFromConfig(runtime, ch_config));
    XLS_ASSIGN_OR_RETURN(uint64_t dma_id, ch_config.GetDMAID());
    XLS_RETURN_IF_ERROR(dma.RegisterEndpoint(
        std::unique_ptr<StreamDmaEndpoint>(new StreamDmaEndpoint(
            std::unique_ptr<IStream>(new IRStream(std::move(stream))))),
        dma_id, bus_master_port));
  }

  return dma;
}

static absl::StatusOr<IrAxiStreamLike> CreateAXIStreamFromConfig(
    RuntimeManager* runtime, const ConfigAxiStreamLikeChannel& config) {
  std::string_view ch_name = config.GetName();
  XLS_ASSIGN_OR_RETURN(
      auto* queue, runtime->runtime().queue_manager().GetQueueByName(ch_name));

  if (queue->channel()->kind() != ChannelKind::kStreaming) {
    return absl::InternalError(
        absl::StrFormat("Channel \"%s\" is not a stream.", ch_name));
  }

  bool multisymbol;
  std::optional<uint64_t> tkeep_index;
  if (!config.GetKeepIdx().ok()) {
    multisymbol = false;
  } else {
    multisymbol = true;
    tkeep_index = config.GetKeepIdx().value();
  }
  std::optional<uint64_t> tlast_index;
  if (config.GetLastIdx().ok()) {
    tlast_index = config.GetLastIdx().value();
  }

  uint64_t data_index = config.GetDataIdxs()[0];
  XLS_ASSIGN_OR_RETURN(IrAxiStreamLike stream,
                       IrAxiStreamLike::Make(queue, multisymbol, data_index,
                                             tlast_index, tkeep_index));
  XLS_LOG(INFO) << "Created channel \"" << ch_name << "\"";

  return std::move(stream);
}

static absl::StatusOr<std::optional<DmaStreamManager>> SetUpDmaAXIStreamManager(
    IMasterPort* bus_master_port, const Config& config,
    RuntimeManager* runtime) {
  // AXI DMA Stream Manager not declared in config, skip it
  absl::StatusOr<std::optional<ConfigDmaAxiStreamLikeManager>> dma_config =
      config.GetDmaAxiStreamLikeManagerConfig();
  if (!dma_config.ok()) {
    XLS_LOG(WARNING) << "Error loading AXI DMA configuration: "
                     << dma_config.status().ToString();
    return std::nullopt;
  }
  if (!dma_config->has_value()) {
    return std::nullopt;
  }

  DmaStreamManager dma = DmaStreamManager(dma_config->value().GetBaseAddress());
  for (auto& ch_config : dma_config->value().GetChannels()) {
    XLS_ASSIGN_OR_RETURN(auto stream,
                         CreateAXIStreamFromConfig(runtime, ch_config));
    XLS_ASSIGN_OR_RETURN(uint64_t dma_id, ch_config.GetDMAID());
    XLS_RETURN_IF_ERROR(dma.RegisterEndpoint(
        std::unique_ptr<AxiStreamLikeDmaEndpoint>(
            new AxiStreamLikeDmaEndpoint(std::unique_ptr<IrAxiStreamLike>(
                new IrAxiStreamLike(std::move(stream))))),
        dma_id, bus_master_port));
  }

  return dma;
}

absl::StatusOr<XlsPeripheral> XlsPeripheral::Make(IConnection& connection,
                                                  std::string_view context) {
  XLS_LOG(INFO) << "Creating generic::XlsPeripheral. Config: " << context;

  XLS_ASSIGN_OR_RETURN(Config config, LoadConfig(context));
  XLS_ASSIGN_OR_RETURN(std::unique_ptr<Package> package, LoadPackage(config));

  // Real init happens on `Reset` call.
  return XlsPeripheral(std::move(config), connection, std::move(package),
                       nullptr);
}

XlsPeripheral::XlsPeripheral(Config&& config, IConnection& connection,
                             std::unique_ptr<Package> package,
                             std::unique_ptr<RuntimeManager> runtime)
    : config_(std::move(config)),
      connection_(connection),
      package_(std::move(package)),
      runtime_(std::move(runtime)) {}

uint64_t GetByteWidth(AccessWidth access) {
  switch (access) {
    case AccessWidth::BYTE:
      return 1;
    case AccessWidth::WORD:
      return 2;
    case AccessWidth::DWORD:
      return 4;
    case AccessWidth::QWORD:
      return 8;
    default:
      XLS_LOG(ERROR) << "Unhandled AccessWidth!";
      XLS_DCHECK(false);
  }
  return 0;
}

absl::Status XlsPeripheral::CheckRequest(uint64_t addr, AccessWidth width) {
  if (runtime_ == nullptr) {
    XLS_LOG(FATAL) << "Device has not been initialized";
    return absl::FailedPreconditionError("Device has not been initialized");
  }

  uint64_t access_width = GetByteWidth(width);
  bool in_range = false;
  bool possible_unaligned = false;

  for (auto& manager : managers_) {
    if (manager->InRange(addr) && manager->InRange(addr + access_width - 1)) {
      in_range = true;
      break;
    }
    if (manager->InRange(addr)) {
      in_range = true;
    }
    if (manager->InRange(addr + access_width - 1)) {
      possible_unaligned = true;
    }
    if (in_range && possible_unaligned) {
      return absl::InvalidArgumentError(
          "Access to multiple managers at once is not supported");
    }
  }

  if (!in_range) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Access at %016x - has no mapping", addr));
  }
  return absl::OkStatus();
}

absl::StatusOr<uint64_t> XlsPeripheral::HandleRead(uint64_t addr,
                                                   AccessWidth width) {
  absl::StatusOr<uint64_t> value;
  for (auto& manager : managers_) {
    if (manager->InRange(addr)) {
      switch (width) {
        case AccessWidth::BYTE:
          value = manager->ReadU8AtAddress(addr);
          break;
        case AccessWidth::WORD:
          value = manager->ReadU16AtAddress(addr);
          break;
        case AccessWidth::DWORD:
          value = manager->ReadU32AtAddress(addr);
          break;
        case AccessWidth::QWORD:
          value = manager->ReadU64AtAddress(addr);
          break;
      }
    }
  }
  return value;
}

absl::Status XlsPeripheral::HandleWrite(uint64_t addr, AccessWidth width,
                                        uint64_t payload) {
  absl::Status status;
  for (auto& manager : managers_) {
    if (manager->InRange(addr)) {
      switch (width) {
        case AccessWidth::BYTE:
          status = manager->WriteU8AtAddress(addr, payload);
          break;
        case AccessWidth::WORD:
          status = manager->WriteU16AtAddress(addr, payload);
          break;
        case AccessWidth::DWORD:
          status = manager->WriteU32AtAddress(addr, payload);
          break;
        case AccessWidth::QWORD:
          status = manager->WriteU64AtAddress(addr, payload);
          break;
      }
    }
  }
  return status;
}

absl::StatusOr<IRQEnum> XlsPeripheral::HandleIRQ() {
  if (runtime_ == nullptr) {
    XLS_LOG(FATAL) << "Device has not been initialized";
    return absl::FailedPreconditionError("Device has not been initialized");
  }
  // Simulation only uses IRQ 0 if ever used
  bool irq = false;
  for (auto& manager : managers_) {
    if (auto interrupt_manager = dynamic_cast<IIRQ*>(manager.get());
        interrupt_manager != nullptr) {
      if (auto s = interrupt_manager->UpdateIRQ(); !s.ok()) {
        XLS_LOG(FATAL) << s.message();
        return s;
      }
      irq |= interrupt_manager->GetIRQ();
    }
  }
  if (last_irq_ != irq) {
    last_irq_ = irq;
    if (irq > 0)
      return IRQEnum::SetIRQ;
    else
      return IRQEnum::UnsetIRQ;
  }

  return IRQEnum::NoChange;
}

absl::Status XlsPeripheral::HandleTick() {
  if (runtime_ == nullptr) {
    XLS_LOG(FATAL) << "Device has not been initialized";
    return absl::FailedPreconditionError("Device has not been initialized");
  }

  absl::Status tick_status;
  // Tick is special: response must be sent over a request channel
  tick_status = runtime_->Update();
  // This is horrible, but in order to avoid it we would have to change the
  // way ProcRuntime::Tick() reports deadlocks. Those are almost never
  // errors in case of co-simulation, instead they usually mean that the
  // device is idle (not in use).
  if (!tick_status.ok()) {
    if (tick_status.message().find_first_of("Proc network is deadlocked.") !=
        0) {
      XLS_LOG(FATAL) << tick_status.message();
      return tick_status;
    }
  }

  for (auto& manager : managers_) {
    if (auto active_manager = dynamic_cast<IActive*>(manager.get());
        active_manager != nullptr) {
      if (auto s = active_manager->Update(); !s.ok()) {
        XLS_LOG(FATAL) << s.message();
        return s;
      }
    }
  }

  return absl::OkStatus();
}

absl::Status XlsPeripheral::Reset() {
  this->managers_.clear();

  auto runtime_man = RuntimeManager::Create(package_.get(), false);
  if (!runtime_man.ok()) {
    return absl::InternalError("Failed to initialize runtime: " +
                               runtime_man.status().ToString());
  }

  XLS_ASSIGN_OR_RETURN(auto svm_option,
                       SetUpSingleValueManager(config_, runtime_man->get()));
  if (svm_option.has_value()) {
    this->managers_.emplace_back(std::unique_ptr<IChannelManager>(
        new SingleValueManager(std::move(svm_option.value()))));
  }

  XLS_ASSIGN_OR_RETURN(std::optional<StreamManager> sm_option,
                       SetUpStreamManager(config_, runtime_man->get()));
  if (sm_option.has_value()) {
    this->managers_.emplace_back(std::unique_ptr<IChannelManager>(
        new StreamManager(std::move(sm_option.value()))));
  }

  XLS_ASSIGN_OR_RETURN(std::optional<DmaStreamManager> dsm_option,
                       SetUpDmaStreamManager(connection_.GetMasterPort(),
                                             config_, runtime_man->get()));
  if (dsm_option.has_value()) {
    this->managers_.emplace_back(std::unique_ptr<IChannelManager>(
        new DmaStreamManager(std::move(dsm_option.value()))));
  }

  XLS_ASSIGN_OR_RETURN(std::optional<DmaStreamManager> dasm_option,
                       SetUpDmaAXIStreamManager(connection_.GetMasterPort(),
                                                config_, runtime_man->get()));
  if (dasm_option.has_value()) {
    this->managers_.emplace_back(std::unique_ptr<IChannelManager>(
        new DmaStreamManager(std::move(dasm_option.value()))));
  }

  this->runtime_ = std::move(runtime_man.value());

  XLS_LOG(INFO) << "Peripheral has been reset";
  return absl::OkStatus();
}

}  // namespace xls::simulation::generic
