// Copyright 2025 The XLS Authors
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

#include <string>
#include <string_view>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_replace.h"
#include "xls/common/exit_status.h"
#include "xls/common/file/filesystem.h"
#include "xls/common/init_xls.h"
#include "xls/common/status/status_macros.h"
#include "xls/passes/optimization_pass_registry.h"
#include "xls/passes/tools/generate_documentation_proto.h"
#include "xls/passes/tools/pass_documentation.pb.h"

const char kUsage[] = R"(
Extract file comments from registered passes.
)";

ABSL_FLAG(std::string, header, "", "header to extract data from");
ABSL_FLAG(std::string, strip_prefix, "", "prefix to strip from file names.");
ABSL_FLAG(std::string, output, "/dev/stdout", "output file");
ABSL_FLAG(std::vector<std::string>, copts, {},
          "Comma-separated copts needed to parse the pass declarations. The "
          "text __ESCAPED_COMMA__ will be replaced with a ',' before use.");

namespace xls {
namespace {

std::vector<std::string> Unescape(std::vector<std::string> inp) {
  for (auto& v : inp) {
    std::string tmp = std::move(v);
    v = absl::StrReplaceAll(tmp, {{"__ESCAPED_COMMA__", ","}});
  }
  return inp;
}

absl::Status RealMain() {
  std::vector<std::string> unescaped_copts =
      Unescape(absl::GetFlag(FLAGS_copts));

  XLS_ASSIGN_OR_RETURN(
      PassDocumentationProto res,
      GenerateDocumentationProto(GetOptimizationRegistry(),
                                 absl::GetFlag(FLAGS_header), unescaped_copts));
  for (int i = 0; i < res.passes_size(); ++i) {
    PassDocumentationProto::OnePass* pass = res.mutable_passes(i);
    if (pass->has_file() &&
        pass->file().starts_with(absl::GetFlag(FLAGS_strip_prefix))) {
      *pass->mutable_file() =
          pass->file().substr(absl::GetFlag(FLAGS_strip_prefix).size());
    }
  }
  XLS_RETURN_IF_ERROR(
      SetFileContents(absl::GetFlag(FLAGS_output), res.SerializeAsString()));
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
