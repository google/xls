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

#include <optional>
#include <string_view>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "xls/common/exit_status.h"
#include "xls/common/init_xls.h"
#include "xls/tools/tool_timeout.h"

const char kUsage[] = R"(
Dummy tool that just waits a particular amount of time to test timeout functionality.
)";

ABSL_FLAG(std::optional<absl::Duration>, wait_for, std::nullopt,
          "pause execution time.");

namespace xls {
namespace {

absl::Status RealMain() {
  auto timeout = StartTimeoutTimer();
  absl::SleepFor(
      absl::GetFlag(FLAGS_wait_for).value_or(absl::InfiniteDuration()));
  std::cout << "Waited for "
            << absl::GetFlag(FLAGS_wait_for).value_or(absl::InfiniteDuration())
            << "\n";
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
