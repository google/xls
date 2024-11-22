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

#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "google/protobuf/text_format.h"
#include "xls/common/exit_status.h"
#include "xls/common/file/filesystem.h"
#include "xls/common/init_xls.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/passes/optimization_pass.h"
#include "xls/passes/optimization_pass_pipeline.h"
#include "xls/passes/pass_pipeline.pb.h"

const char kUsage[] = R"(
  Dump the default optimization pipeline as a PassPipelineProto.
)";

ABSL_FLAG(std::optional<std::string>, textproto_out, std::nullopt,
          "Location to write the pipeline in textproto format.");
ABSL_FLAG(std::optional<std::string>, proto_out, std::nullopt,
          "Location to write the pipeline in binary proto format.");

namespace xls {
namespace {

absl::Status RealMain() {
  std::unique_ptr<OptimizationPass> passes = CreateOptimizationPassPipeline();
  if (!absl::GetFlag(FLAGS_textproto_out) && !absl::GetFlag(FLAGS_proto_out)) {
    return absl::InvalidArgumentError(
        "One of --proto_out or --textproto_out is required");
  }
  PassPipelineProto res;
  XLS_ASSIGN_OR_RETURN(*res.mutable_top(), passes->ToProto());
  if (absl::GetFlag(FLAGS_textproto_out)) {
    std::string out;
    XLS_RET_CHECK(google::protobuf::TextFormat::PrintToString(res, &out))
        << "Unable to serialize proto.";
    XLS_RETURN_IF_ERROR(
        SetFileContents(*absl::GetFlag(FLAGS_textproto_out), out));
  }
  if (absl::GetFlag(FLAGS_proto_out)) {
    XLS_RETURN_IF_ERROR(SetFileContents(*absl::GetFlag(FLAGS_proto_out),
                                        res.SerializeAsString()));
  }
  return absl::OkStatus();
}

}  // namespace
}  // namespace xls

int main(int argc, char** argv) {
  std::vector<std::string_view> positional_arguments =
      xls::InitXls(kUsage, argc, argv);

  if (!positional_arguments.empty()) {
    LOG(QFATAL) << "Expected invocation: " << argv[0];
  }

  return xls::ExitStatus(xls::RealMain());
}
