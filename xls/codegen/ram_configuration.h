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

#ifndef XLS_CODEGEN_RAM_CONFIGURATION_H_
#define XLS_CODEGEN_RAM_CONFIGURATION_H_

#include <memory>
#include <string>

#include "absl/status/statusor.h"

namespace xls::verilog {

struct RamRWPortConfiguration {
  std::string request_channel_name;
  std::string response_channel_name;
};

// Abstract base class for a RAM configuration.
class RamConfiguration {
 public:
  virtual ~RamConfiguration() = default;

  static absl::StatusOr<std::unique_ptr<RamConfiguration>> ParseString(
      std::string_view text);
  virtual std::unique_ptr<RamConfiguration> Clone() const = 0;

  std::string_view ram_name() const { return ram_name_; }
  int64_t latency() const { return latency_; }
  virtual std::string_view ram_kind() const = 0;

 protected:
  RamConfiguration(std::string_view ram_name, int64_t latency)
      : ram_name_(ram_name), latency_(latency) {}
  std::string ram_name_;
  int64_t latency_;
};

// Configuration for a single-port RAM.
class Ram1RWConfiguration : public RamConfiguration {
 public:
  Ram1RWConfiguration(std::string_view ram_name, int64_t latency,
                      std::string_view request_name,
                      std::string_view response_name)
      : RamConfiguration(ram_name, latency),
        rw_port_configuration_(RamRWPortConfiguration{
            .request_channel_name = std::string{request_name},
            .response_channel_name = std::string{response_name}}) {}

  static absl::StatusOr<std::unique_ptr<Ram1RWConfiguration>> ParseSplitString(
      absl::Span<const std::string_view> fields);

  std::unique_ptr<RamConfiguration> Clone() const override;

  std::string_view ram_kind() const override { return "1RW"; }

  const RamRWPortConfiguration& rw_port_configuration() const {
    return rw_port_configuration_;
  }

 private:
  RamRWPortConfiguration rw_port_configuration_;
};

}  // namespace xls::verilog

#endif  // XLS_CODEGEN_RAM_CONFIGURATION_H_
