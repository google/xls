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

#include <filesystem>  // NOLINT
#include <iostream>
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
#include "xls/dev_tools/extract_interface.h"
#include "xls/ir/xls_ir_interface.pb.h"
#include "xls/public/ir_parser.h"

const char kUsage[] = R"(
  Parse an IR file and write a PackageInterfaceProto to stdout.
)";

ABSL_FLAG(bool, binary_proto, false, "Print as a binary proto");

namespace xls {
namespace {

absl::Status RealMain(bool binary_proto, const std::filesystem::path& path) {
  XLS_ASSIGN_OR_RETURN(auto ir_text, GetFileContents(path));
  XLS_ASSIGN_OR_RETURN(auto package, ParsePackage(ir_text, path.string()));
  PackageInterfaceProto proto = ExtractPackageInterface(package.get());

  std::string output;
  if (binary_proto) {
    output = proto.SerializeAsString();
  } else {
    XLS_RET_CHECK(google::protobuf::TextFormat::PrintToString(proto, &output));
  }
  std::cout << output;
  return absl::OkStatus();
}

}  // namespace
}  // namespace xls

int main(int argc, char** argv) {
  std::vector<std::string_view> positional_arguments =
      xls::InitXls(kUsage, argc, argv);

  if (positional_arguments.size() != 1) {
    LOG(QFATAL) << "Expected invocation: " << argv[0] << " <ir_file>";
  }

  return xls::ExitStatus(xls::RealMain(absl::GetFlag(FLAGS_binary_proto),
                                       positional_arguments[0]));
}
