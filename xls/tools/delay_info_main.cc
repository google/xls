// Copyright 2021 The XLS Authors
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

// Takes in an IR file and produces an IR file that has been run through the
// standard optimization pipeline.

#include <memory>
#include <string_view>
#include <vector>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "xls/common/exit_status.h"
#include "xls/common/init_xls.h"
#include "xls/common/status/status_macros.h"
#include "xls/estimators/delay_model/delay_info.pb.h"
#include "xls/tools/delay_info_flags.h"
#include "xls/tools/delay_info_flags.pb.h"
#include "xls/tools/delay_info_printer.h"

static constexpr std::string_view kUsage = R"(

Dumps delay information about an XLS function including per-node delay
information and critical-path. Example invocations:

Emit delay information about a function:
   delay_info_main --delay_model=unit --top=ENTRY IR_FILE

Emit delay information about a function including per-stage critical path
information:
   delay_info_main --delay_model=unit \
     --schedule_path=SCHEDULE_FILE \
     --top=ENTRY \
     IR_FILE
)";

namespace xls::tools {
namespace {

absl::Status RealMain(std::string_view input_path) {
  std::unique_ptr<DelayInfoPrinter> printer = CreateDelayInfoPrinter();
  XLS_RETURN_IF_ERROR(printer->Init(GetDelayInfoFlagsProto(input_path)));
  return printer->GenerateApplicableInfo();
}

}  // namespace
}  // namespace xls::tools

int main(int argc, char** argv) {
  std::vector<std::string_view> positional_arguments =
      xls::InitXls(kUsage, argc, argv);

  if (positional_arguments.empty()) {
    LOG(QFATAL) << absl::StreamFormat("Expected invocation: %s <path>",
                                      argv[0]);
  }

  return xls::ExitStatus(xls::tools::RealMain(positional_arguments[0]));
}
