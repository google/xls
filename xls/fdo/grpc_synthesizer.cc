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

#include "xls/fdo/grpc_synthesizer.h"

#include <cstdint>
#include <memory>
#include <string_view>

#include "absl/log/check.h"
#include "absl/status/statusor.h"
#include "xls/common/casts.h"
#include "xls/common/module_initializer.h"
#include "xls/common/status/status_macros.h"
#include "xls/fdo/synthesizer.h"
#include "xls/synthesis/synthesis.pb.h"
#include "xls/synthesis/synthesis_client.h"

namespace xls {
namespace synthesis {
namespace {

// A Synthesizer implementation that invokes a `SynthesizerService` via a gRPC
// client.
class GrpcSynthesizer : public Synthesizer {
 public:
  explicit GrpcSynthesizer(const GrpcSynthesizerParameters& params)
      : Synthesizer("grpc"), params_(params) {}

  absl::StatusOr<int64_t> SynthesizeVerilogAndGetDelay(
      std::string_view verilog_text,
      std::string_view top_module_name) const override {
    xls::synthesis::CompileRequest request;
    request.set_top_module_name(top_module_name);
    request.set_target_frequency_hz(params_.frequency_hz());
    request.set_module_text(verilog_text);

    XLS_ASSIGN_OR_RETURN(CompileResponse response,
                         xls::synthesis::SynthesizeViaClient(
                             params_.server_and_port(), request));
    const int64_t clock_period_ps =
        static_cast<int64_t>(1e12) / params_.frequency_hz();
    return response.slack_ps() == 0 ? 0 : clock_period_ps - response.slack_ps();
  }

 private:
  const GrpcSynthesizerParameters params_;
};

}  // namespace

absl::StatusOr<std::unique_ptr<Synthesizer>>
GrpcSynthesizerFactory::CreateSynthesizer(
    const SynthesizerParameters& parameters) {
  return std::make_unique<GrpcSynthesizer>(
      down_cast<const GrpcSynthesizerParameters&>(parameters));
}

XLS_REGISTER_MODULE_INITIALIZER(grpc_synthesizer_factory, {
  CHECK_OK(GetSynthesizerManagerSingleton().RegisterSynthesizer(
      std::make_unique<GrpcSynthesizerFactory>()));
});

}  // namespace synthesis
}  // namespace xls
