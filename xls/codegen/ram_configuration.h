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

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/status/statusor.h"
#include "xls/scheduling/scheduling_options.h"

namespace xls::verilog {

struct RamRWPortConfiguration {
  std::string request_channel_name;
  std::string response_channel_name;
  std::string write_completion_channel_name;
};

struct RamRPortConfiguration {
  std::string request_channel_name;
  std::string response_channel_name;
};

struct RamWPortConfiguration {
  std::string request_channel_name;
  std::string write_completion_channel_name;
};

// Abstract base class for a RAM configuration.
class RamConfiguration {
 public:
  // Parses a split configuration string of the form
  // "(ram_name, ram_kind[, <kind-specific configuration>])".
  static absl::StatusOr<std::unique_ptr<RamConfiguration>> ParseString(
      std::string_view text);

  virtual ~RamConfiguration() = default;

  virtual std::unique_ptr<RamConfiguration> Clone() const = 0;
  virtual std::vector<IOConstraint> GetIOConstraints() const = 0;

  std::string_view ram_name() const { return ram_name_; }
  int64_t latency() const { return latency_; }
  std::string_view ram_kind() const { return ram_kind_; }

 protected:
  RamConfiguration(std::string_view ram_name, int64_t latency,
                   std::string ram_kind)
      : ram_name_(ram_name),
        ram_kind_(std::move(ram_kind)),
        latency_(latency) {}

  std::string ram_name_;
  std::string ram_kind_;
  int64_t latency_;
};

// Configuration for a single-port RAM.
class Ram1RWConfiguration : public RamConfiguration {
 public:
  Ram1RWConfiguration(std::string_view ram_name, int64_t latency,
                      std::string_view request_name,
                      std::string_view response_name,
                      std::string_view write_completion_name)
      : RamConfiguration(ram_name, latency, /*ram_kind=*/"1RW"),
        rw_port_configuration_(RamRWPortConfiguration{
            .request_channel_name = std::string{request_name},
            .response_channel_name = std::string{response_name},
            .write_completion_channel_name = std::string{write_completion_name},
        }) {}

  std::unique_ptr<RamConfiguration> Clone() const override;
  std::vector<IOConstraint> GetIOConstraints() const override;

  const RamRWPortConfiguration& rw_port_configuration() const {
    return rw_port_configuration_;
  }

 private:
  RamRWPortConfiguration rw_port_configuration_;
};

// Configuration for a pseudo dual-port RAM.
class Ram1R1WConfiguration : public RamConfiguration {
 public:
  Ram1R1WConfiguration(std::string_view ram_name, int64_t latency,
                       std::string_view read_request_name,
                       std::string_view read_response_name,
                       std::string_view write_request_name,
                       std::string_view write_completion_name)
      : RamConfiguration(ram_name, latency, /*ram_kind=*/"1R1W"),
        r_port_configuration_(RamRPortConfiguration{
            .request_channel_name = std::string{read_request_name},
            .response_channel_name = std::string{read_response_name},
        }),
        w_port_configuration_(RamWPortConfiguration{
            .request_channel_name = std::string{write_request_name},
            .write_completion_channel_name =
                std::string{write_completion_name}}) {}

  std::unique_ptr<RamConfiguration> Clone() const override;
  std::vector<IOConstraint> GetIOConstraints() const override;

  const RamRPortConfiguration& r_port_configuration() const {
    return r_port_configuration_;
  }
  const RamWPortConfiguration& w_port_configuration() const {
    return w_port_configuration_;
  }

 private:
  RamRPortConfiguration r_port_configuration_;
  RamWPortConfiguration w_port_configuration_;
};

}  // namespace xls::verilog

#endif  // XLS_CODEGEN_RAM_CONFIGURATION_H_
