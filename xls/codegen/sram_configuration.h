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

#ifndef XLS_CODEGEN_SRAM_CONFIGURATION_H_
#define XLS_CODEGEN_SRAM_CONFIGURATION_H_

#include <memory>
#include <string>

#include "absl/status/statusor.h"

namespace xls::verilog {

struct SramRWPortConfiguration {
  std::string request_channel_name;
  std::string response_channel_name;
};

// Abstract base class for an SRAM configuration.
class SramConfiguration {
 public:
  virtual ~SramConfiguration() = default;

  static absl::StatusOr<std::unique_ptr<SramConfiguration>> ParseString(
      std::string_view text);
  virtual std::unique_ptr<SramConfiguration> Clone() const = 0;

  std::string_view sram_name() const { return sram_name_; }
  int64_t latency() const { return latency_; }
  virtual std::string_view sram_kind() const = 0;

 protected:
  SramConfiguration(std::string_view sram_name, int64_t latency)
      : sram_name_(sram_name), latency_(latency) {}
  std::string sram_name_;
  int64_t latency_;
};

// Sram configuration for a single-port SRAM.
class Sram1RWConfiguration : public SramConfiguration {
 public:
  Sram1RWConfiguration(std::string_view sram_name, int64_t latency,
                       std::string_view request_name,
                       std::string_view response_name)
      : SramConfiguration(sram_name, latency),
        rw_port_configuration_(SramRWPortConfiguration{
            .request_channel_name = std::string{request_name},
            .response_channel_name = std::string{response_name}}) {}

  static absl::StatusOr<std::unique_ptr<Sram1RWConfiguration>> ParseSplitString(
      absl::Span<const std::string_view> fields);

  std::unique_ptr<SramConfiguration> Clone() const override;

  std::string_view sram_kind() const override { return "1RW"; }

  const SramRWPortConfiguration& rw_port_configuration() const {
    return rw_port_configuration_;
  }

 private:
  SramRWPortConfiguration rw_port_configuration_;
};

}  // namespace xls::verilog

#endif  // XLS_CODEGEN_SRAM_CONFIGURATION_H_
