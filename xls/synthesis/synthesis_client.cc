// Copyright 2023 Google LLC
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

#include "xls/synthesis/synthesis_client.h"

#include <memory>
#include <string>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "grpcpp/channel.h"
#include "grpcpp/client_context.h"
#include "grpcpp/create_channel.h"
#include "grpcpp/security/credentials.h"
#include "grpcpp/support/status.h"
#include "xls/common/status/status_macros.h"
#include "xls/synthesis/credentials.h"
#include "xls/synthesis/synthesis.pb.h"
#include "xls/synthesis/synthesis_service.grpc.pb.h"

namespace xls {
namespace synthesis {

static absl::Status GrpcToAbslStatus(const grpc::Status& grpc_status) {
  return absl::Status(
      // this assumes that the status code enums match up
      static_cast<absl::StatusCode>(static_cast<int>(grpc_status.error_code())),
      grpc_status.error_message());
}

// This creates a new channel and stub *each* invocation
absl::StatusOr<CompileResponse> SynthesizeViaClient(
    const std::string& server, const CompileRequest& request) {
  // Create a channel, a logical connection an endpoint.
  std::shared_ptr<grpc::ChannelCredentials> creds = GetChannelCredentials();
  std::shared_ptr<grpc::Channel> channel = grpc::CreateChannel(server, creds);
  // Creation of a RPC stub for the channel.
  std::unique_ptr<SynthesisService::Stub> stub(
      SynthesisService::NewStub(channel));

  // Context for the client. It could be used to convey extra information to
  // the server and/or tweak certain RPC behaviors.
  grpc::ClientContext context;

  // The actual RPC.
  CompileResponse response;
  XLS_RETURN_IF_ERROR(
      GrpcToAbslStatus(stub->Compile(&context, request, &response)));
  return response;
}

}  // namespace synthesis
}  // namespace xls
