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

#include <cstdlib>
#include <string>
#include <string_view>

#include "absl/flags/flag.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "libs/json11/json11.hpp"
#include "xls/common/exit_status.h"
#include "xls/common/file/filesystem.h"
#include "xls/common/init_xls.h"
#include "xls/common/status/status_macros.h"

const char kUsage[] = R"(
A dummy JSON metrics script meant to mimic the behavior of an OpenROAD metrics
script wrapped using bazel_rules_hdl modular hardware flows.

Invocation:

  env CONSTANT_CLOCK_PERIOD_PS=1000 OUTPUT_METRICS=metrics.json \
  dummy_metrics_main --critical_path_ps=1723.41

  will result in the JSON for slack_ps=-723.41 written to metrics.json
)";

ABSL_FLAG(double, critical_path_ps, 1000,
          "The critical path length used to compute the reported slack value.");

enum class ErrorMode { kNone, kNotJson, kNoSlackPs, kNotDouble };

// Parse a string to an ErrorMode flag
static bool AbslParseFlag(std::string_view text, ErrorMode* mode,
                          std::string* error) {
  if (text == "none") {
    *mode = ErrorMode::kNone;
    return true;
  }
  if (text == "not_json") {
    *mode = ErrorMode::kNotJson;
    return true;
  }
  if (text == "no_slack_ps") {
    *mode = ErrorMode::kNoSlackPs;
    return true;
  }
  if (text == "not_double") {
    *mode = ErrorMode::kNotDouble;
    return true;
  }
  *error = "unknown value for enumeration";
  return false;
}

// AbslUnparseFlag converts from an ErrorMode to a string
static std::string AbslUnparseFlag(ErrorMode mode) {
  switch (mode) {
    case ErrorMode::kNone:
      return "none";
    case ErrorMode::kNotJson:
      return "not_json";
    case ErrorMode::kNoSlackPs:
      return "no_slack_ps";
    case ErrorMode::kNotDouble:
      return "not_double";
    default:
      return absl::StrCat(mode);
  }
}

ABSL_FLAG(ErrorMode, error_mode, ErrorMode::kNone,
          "When not none, kind of error response to return (not_json, "
          "no_slack_ps, not_double).");

namespace xls {
namespace synthesis {
namespace {

absl::Status RealMain() {
  double critical_path_ps = absl::GetFlag(FLAGS_critical_path_ps);

  const char* clock_period_ps_str = std::getenv("CONSTANT_CLOCK_PERIOD_PS");
  if (clock_period_ps_str == nullptr) {
    return absl::InternalError(
        "CONSTANT_CLOCK_PERIOD_PS environment variable not set.");
  }

  double clock_period_ps = 0;
  if (!absl::SimpleAtod(clock_period_ps_str, &clock_period_ps)) {
    return absl::InternalError(absl::StrCat(
        "CONSTANT_CLOCK_PERIOD_PS does not contain a valid double, is",
        clock_period_ps_str, " instead."));
  }

  double slack_ps = clock_period_ps - critical_path_ps;

  const char* metrics_file_path = ::getenv("OUTPUT_METRICS");
  if (metrics_file_path == nullptr) {
    return absl::InternalError("OUTPUT_METRICS environment variable not set.");
  }

  std::string contents;
  switch (absl::GetFlag(FLAGS_error_mode)) {
    case ErrorMode::kNotJson: {
      contents = "not json";
      break;
    }
    case ErrorMode::kNoSlackPs: {
      json11::Json bad_json(json11::Json::object{
          {"not_slack_ps", json11::Json(absl::StrCat(slack_ps))}});
      contents = bad_json.dump();
      break;
    }
    case ErrorMode::kNotDouble: {
      json11::Json bad_json(
          json11::Json::object{{"slack_ps", json11::Json("no slack")}});
      contents = bad_json.dump();
      break;
    }
    default: {
      // OpenROAD metrics JSON returns numbers as strings.
      json11::Json metrics_json(json11::Json::object{
          {"slack_ps", json11::Json(absl::StrCat(slack_ps))}});
      contents = metrics_json.dump();
      break;
    }
  }
  return xls::SetFileContents(metrics_file_path, contents);
}

}  // namespace
}  // namespace synthesis
}  // namespace xls

int main(int argc, char** argv) {
  xls::InitXls(kUsage, argc, argv);

  return xls::ExitStatus(xls::synthesis::RealMain());
}
