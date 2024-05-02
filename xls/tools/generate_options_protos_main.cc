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

#include <string_view>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "xls/common/exit_status.h"
#include "xls/common/init_xls.h"
#include "xls/common/status/status_macros.h"
#include "xls/tools/codegen_flags.h"
#include "xls/tools/scheduling_options_flags.h"

static constexpr std::string_view kUsage = R"(
Create a scheduling options and codegen options proto with values populated
from flags or default values if a flag is not specified.
)";

namespace xls {
namespace {

absl::Status RealMain() {
  XLS_RETURN_IF_ERROR(GetSchedulingOptionsFlagsProto().status());
  XLS_RETURN_IF_ERROR(GetCodegenFlags().status());
  return absl::OkStatus();
}

}  // namespace
}  // namespace xls

int main(int argc, char** argv) {
  std::vector<std::string_view> positional_arguments =
      xls::InitXls(kUsage, argc, argv);

  if (!absl::GetFlag(FLAGS_scheduling_options_used_textproto_file) &&
      !absl::GetFlag(FLAGS_codegen_options_used_textproto_file)) {
    LOG(QFATAL)
        << "Requires at least one of --scheduling_options_used_textproto_file "
           "or --codegen_options_used_textproto_file";
  }

  if (!positional_arguments.empty()) {
    LOG(QFATAL) << "Expected invocation: " << argv[0];
  }

  return xls::ExitStatus(xls::RealMain());
}
