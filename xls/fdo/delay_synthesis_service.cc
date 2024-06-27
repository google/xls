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

#include "xls/fdo/delay_synthesis_service.h"

#include <cstdint>
#include <string>

#include "absl/status/statusor.h"
#include "grpcpp/server_context.h"
#include "grpcpp/support/status.h"

namespace xls {
namespace synthesis {

::grpc::Status DelaySynthesisService::Compile(
    ::grpc::ServerContext* server_context, const CompileRequest* request,
    CompileResponse* result) {
  absl::StatusOr<int64_t> delay = synthesizer_->SynthesizeVerilogAndGetDelay(
      request->module_text(), request->top_module_name());
  if (!delay.ok()) {
    return ::grpc::Status(grpc::StatusCode::INTERNAL,
                          std::string(delay.status().message()));
  }
  if (*delay != 0) {
    result->set_max_frequency_hz(static_cast<int64_t>(1e12) / *delay);
  }
  // Currently there is no way to convey a target frequency to a Synthesizer
  // object.
  result->set_insensitive_to_target_freq(true);
  return ::grpc::Status::OK;
}

}  // namespace synthesis
}  // namespace xls
