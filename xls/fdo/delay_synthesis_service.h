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

#ifndef XLS_FDO_DELAY_SYNTHESIS_SERVICE_H_
#define XLS_FDO_DELAY_SYNTHESIS_SERVICE_H_

#include <memory>
#include <utility>

#include "grpcpp/server_context.h"
#include "grpcpp/support/status.h"
#include "xls/fdo/synthesizer.h"
#include "xls/synthesis/synthesis.pb.h"
#include "xls/synthesis/synthesis.pb.h"
#include "xls/synthesis/synthesis_service.grpc.pb.h"

namespace xls {
namespace synthesis {

// A `SynthesisService` backed by a `Synthesizer`, which should be created by
// the caller via `SynthesizerFactory`. Currently, the service only produces
// delay values and not a netlist, and is intended for enabling the use of any
// synthesizer with tools like `delay_info_main` that target a gRPC server.
class DelaySynthesisService : public SynthesisService::Service {
 public:
  explicit DelaySynthesisService(std::unique_ptr<Synthesizer> synthesizer)
      : synthesizer_(std::move(synthesizer)) {}

  grpc::Status Compile(grpc::ServerContext* server_context,
                       const CompileRequest* request,
                       CompileResponse* result) override;

 private:
  std::unique_ptr<Synthesizer> synthesizer_;
};

}  // namespace synthesis
}  // namespace xls

#endif  // XLS_FDO_DELAY_SYNTHESIS_SERVICE_H_
