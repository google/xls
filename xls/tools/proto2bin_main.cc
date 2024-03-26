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

#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "xls/common/exit_status.h"
#include "xls/common/file/filesystem.h"
#include "xls/common/init_xls.h"
#include "xls/common/logging/log_lines.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/contrib/xlscc/hls_block.pb.h"
#include "xls/ir/ram_rewrite.pb.h"

const char kUsage[] = R"(
Simplified utility to convert a textproto to a binary proto.  Used
to to provide binary protos to xls rules and command line
interfaces that utilize binary protos for configuration (ex. xlscc).
)";

ABSL_FLAG(std::string, message, "",
          "Message to read textproto as.  Supported: [xlscc.HLSBlock, "
          "xls.RamRewritesProto]");
ABSL_FLAG(std::string, output, "", "Output file to write binary proto to.");

namespace xls {
namespace {

absl::StatusOr<std::unique_ptr<google::protobuf::Message>> MakeProtoForMessageType(
    std::string_view message_type) {
  if (message_type == "xlscc.HLSBlock") {
    return std::make_unique<xlscc::HLSBlock>();
  }
  if (message_type == "xls.RamRewritesProto") {
    return std::make_unique<xls::RamRewritesProto>();
  }

  return absl::UnimplementedError(
      absl::StrFormat("Unsupported proto message type: %s", message_type));
}

absl::Status RealMain(std::string_view textproto_path,
                      std::string_view message_type,
                      std::string_view output_path) {
  LOG(INFO) << absl::StrFormat("Converting type %s textproto %s to binproto %s",
                               message_type, textproto_path, output_path);

  XLS_ASSIGN_OR_RETURN(std::unique_ptr<google::protobuf::Message> proto,
                       MakeProtoForMessageType(message_type));

  CHECK_OK(ParseTextProtoFile(textproto_path, proto.get()));

  VLOG(1) << "Proto contents:";
  XLS_VLOG_LINES(1, proto->DebugString());

  CHECK_OK(SetProtobinFile(output_path, *proto));

  return absl::OkStatus();
}

}  // namespace
}  // namespace xls

int main(int argc, char** argv) {
  std::vector<std::string_view> positional_arguments =
      xls::InitXls(kUsage, argc, argv);

  if (positional_arguments.size() != 1) {
    LOG(QFATAL) << absl::StreamFormat("Expected invocation: %s TEXT_PROTO_FILE",
                                      argv[0]);
  }

  std::string_view text_proto_path = positional_arguments[0];

  if (absl::GetFlag(FLAGS_message).empty()) {
    LOG(QFATAL) << "--message (proto message type) required.";
  }

  if (absl::GetFlag(FLAGS_output).empty()) {
    LOG(QFATAL) << "--output (binary proto output file path) required.";
  }

  return xls::ExitStatus(xls::RealMain(text_proto_path,
                                       absl::GetFlag(FLAGS_message),
                                       absl::GetFlag(FLAGS_output)));
}
