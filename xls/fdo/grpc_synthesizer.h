// Copyright 2024 The XLS Authors
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

#ifndef XLS_FDO_GRPC_SYNTHESIZER_H_
#define XLS_FDO_GRPC_SYNTHESIZER_H_

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>

#include "absl/status/statusor.h"
#include "xls/fdo/synthesizer.h"

namespace xls {
namespace synthesis {

inline constexpr int64_t kDefaultFrequencyHz = 1e9;

// Factory parameters for the gRPC-client Synthesizer. The parameters are:
// `server_and_port`: The gRPC endpoint that the `Synthesizer` object should
//      send requests to. e.g.: "ipv4:///0.0.0.0:10000"
// `frequency_hz`: The target frequency any designs that will be synthesized.
class GrpcSynthesizerParameters : public SynthesizerParameters {
 public:
  explicit GrpcSynthesizerParameters(std::string_view server_and_port,
                                     int64_t frequency_hz = kDefaultFrequencyHz)
      : SynthesizerParameters("grpc"),
        server_and_port_(server_and_port),
        frequency_hz_(frequency_hz) {}

  const std::string& server_and_port() const { return server_and_port_; }

  int64_t frequency_hz() const { return frequency_hz_; }

 private:
  const std::string server_and_port_;
  const int64_t frequency_hz_;
};

// A factory that deals out `Synthesizer` objects that use a gRPC client to talk
// to a `SynthesizerService`.
class GrpcSynthesizerFactory : public SynthesizerFactory {
 public:
  explicit GrpcSynthesizerFactory() : SynthesizerFactory("grpc") {}

  absl::StatusOr<std::unique_ptr<Synthesizer>> CreateSynthesizer(
      const SynthesizerParameters& parameters) override;
};

}  // namespace synthesis
}  // namespace xls

#endif  // XLS_FDO_GRPC_SYNTHESIZER_H_
