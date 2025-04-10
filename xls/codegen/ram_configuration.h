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

#include <array>
#include <cstdint>
#include <string>
#include <string_view>
#include <variant>

#include "absl/status/statusor.h"
#include "absl/types/span.h"
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

// Configuration for a single-port RAM.
class Ram1RWConfiguration {
 public:
  Ram1RWConfiguration(std::string_view ram_name, int64_t latency,
                      std::string_view request_name,
                      std::string_view response_name,
                      std::string_view write_completion_name)
      : ram_name_(ram_name),
        latency_(latency),
        rw_port_configuration_(RamRWPortConfiguration{
            .request_channel_name = std::string{request_name},
            .response_channel_name = std::string{response_name},
            .write_completion_channel_name = std::string{write_completion_name},
        }),
        io_constraints_(MakeIOConstraints(rw_port_configuration_, latency_)) {}

  std::string_view ram_name() const { return ram_name_; }
  int64_t latency() const { return latency_; }
  absl::Span<IOConstraint const> io_constraints() const {
    return io_constraints_;
  }

  const RamRWPortConfiguration& rw_port_configuration() const {
    return rw_port_configuration_;
  }

 private:
  // Used by constructor to populate `io_constraints_`.
  static std::array<IOConstraint, 2> MakeIOConstraints(
      const RamRWPortConfiguration& rw_port_configuration, int64_t latency);

  std::string ram_name_;
  int64_t latency_;

  RamRWPortConfiguration rw_port_configuration_;

  std::array<IOConstraint, 2> io_constraints_;
};

// Configuration for a pseudo dual-port RAM.
class Ram1R1WConfiguration {
 public:
  Ram1R1WConfiguration(std::string_view ram_name, int64_t latency,
                       std::string_view read_request_name,
                       std::string_view read_response_name,
                       std::string_view write_request_name,
                       std::string_view write_completion_name)
      : ram_name_(ram_name),
        latency_(latency),
        r_port_configuration_(RamRPortConfiguration{
            .request_channel_name = std::string{read_request_name},
            .response_channel_name = std::string{read_response_name},
        }),
        w_port_configuration_(RamWPortConfiguration{
            .request_channel_name = std::string{write_request_name},
            .write_completion_channel_name =
                std::string{write_completion_name}}),
        io_constraints_(MakeIOConstraints(r_port_configuration_,
                                          w_port_configuration_, latency)) {}

  std::string_view ram_name() const { return ram_name_; }
  int64_t latency() const { return latency_; }
  absl::Span<IOConstraint const> io_constraints() const {
    return io_constraints_;
  }
  const RamRPortConfiguration& r_port_configuration() const {
    return r_port_configuration_;
  }
  const RamWPortConfiguration& w_port_configuration() const {
    return w_port_configuration_;
  }

 private:
  // Used by constructor to populate `io_constraints_`.
  static std::array<IOConstraint, 2> MakeIOConstraints(
      const RamRPortConfiguration& r_port_configuration,
      const RamWPortConfiguration& w_port_configuration, int64_t latency);

  std::string ram_name_;
  int64_t latency_;

  RamRPortConfiguration r_port_configuration_;
  RamWPortConfiguration w_port_configuration_;

  std::array<IOConstraint, 2> io_constraints_;
};

using RamConfiguration =
    std::variant<Ram1RWConfiguration, Ram1R1WConfiguration>;

// Parses a split configuration string of the form
// "(ram_name, ram_kind[, <kind-specific configuration>])".
absl::StatusOr<RamConfiguration> ParseRamConfiguration(std::string_view text);

absl::Span<IOConstraint const> GetRamConfigurationIOConstraints(
    const RamConfiguration& ram_configuration);

std::string_view RamConfigurationRamName(
    const RamConfiguration& ram_configuration);

}  // namespace xls::verilog

#endif  // XLS_CODEGEN_RAM_CONFIGURATION_H_
