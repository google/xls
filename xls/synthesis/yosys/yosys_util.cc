//
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

#include "xls/synthesis/yosys/yosys_util.h"

#include <cstdint>
#include <string>
#include <string_view>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/ascii.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_split.h"
#include "libs/json11/json11.hpp"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "re2/re2.h"

namespace xls {
namespace synthesis {

absl::StatusOr<int64_t> ParseNextpnrOutput(std::string_view nextpnr_output) {
  bool found = false;
  double max_mhz = 0.0;

  // We're looking for lines of the form:
  //
  //   Info: Max frequency for clock 'foo': 125.28 MHz (PASS at 100.00 MHz)
  //
  // And we want to extract 125.28.

  // Finds the last line following the above pattern.
  static constexpr LazyRE2 max_frequency_regex = {
      .pattern_ = R"(Info: Max frequency for clock '.*': ([\d\.]+) MHz)"};
  double parsed_max_mhz = 0.0;
  while (RE2::FindAndConsume(&nextpnr_output, *max_frequency_regex,
                             &parsed_max_mhz)) {
    found = true;
    max_mhz = parsed_max_mhz;
  }

  if (!found) {
    return absl::NotFoundError(
        "Could not find maximum frequency in nextpnr output.");
  }

  return static_cast<int64_t>(max_mhz * 1e6);
}

absl::StatusOr<YosysSynthesisStatistics> ParseYosysJsonOutput(
    std::string_view json_content) {
  YosysSynthesisStatistics stats;
  stats.area = -1.0f;
  stats.sequential_area = -1.0f;

  std::string err;
  const auto json = json11::Json::parse(std::string(json_content), err);
  if (!err.empty()) {
    return absl::InternalError(absl::StrFormat(
        "ParseYosysJsonOutput JSON parse error: %s \n content: %s", err,
        json_content));
  }

  if (!json["design"].is_object()) {
    return absl::InternalError(
        "ParseYosysJsonOutput could not find root 'design' object in JSON");
  }

  const auto& design = json["design"];

  if (design["num_cells_by_type"].is_object()) {
    for (auto const& [cell_name, count] :
         design["num_cells_by_type"].object_items()) {
      stats.cell_histogram[cell_name] = count.int_value();
    }
  }

  if (design["area"].is_number()) {
    stats.area = design["area"].number_value();
  }

  if (design["sequential_area"].is_number()) {
    stats.sequential_area = design["sequential_area"].number_value();
  }

  return stats;
}

absl::StatusOr<STAStatistics> ParseOpenSTAOutput(std::string_view sta_output) {
  STAStatistics stats;

  std::string clk_period_ps;
  std::string freq_mhz;
  float tmp_float = 0.0;
  std::string slack_ps;
  bool period_ok = false, slack_ok = false;

  static constexpr LazyRE2 clk_period_regex = {
      .pattern_ = R"(op_clk period_min = (\d+\.\d+) fmax = (\d+\.\d+))"};
  static constexpr LazyRE2 slack_regex = {
      .pattern_ = R"(^(?:worst slack(?: max)?) (-?\d+.\d+))"};

  for (std::string_view line : absl::StrSplit(sta_output, '\n')) {
    line = absl::StripAsciiWhitespace(line);
    // We're looking for lines with this outline, all in ps, freq in mhz
    //
    //   clk period_min = 116.71 fmax = 8568.43
    //   worst slack -96.67
    // And we want to extract 116.71 and 8568.43 and -96.67

    if (RE2::PartialMatch(line, *clk_period_regex, &clk_period_ps, &freq_mhz)) {
      XLS_RET_CHECK(absl::SimpleAtof(clk_period_ps, &stats.period_ps));
      stats.max_frequency_hz = static_cast<int64_t>(
          1e12 / stats.period_ps);  // use clk in ps for accuracy
      period_ok = true;
    }

    if (RE2::PartialMatch(line, *slack_regex, &slack_ps)) {
      XLS_RET_CHECK(absl::SimpleAtof(slack_ps, &tmp_float));
      stats.slack_ps = static_cast<int64_t>(tmp_float);
      // ensure that negative slack (even small) is reported as negative!
      if (tmp_float < 0.0 && stats.slack_ps == 0) {
        stats.slack_ps = -1;
      }
      slack_ok = true;
    }
  }
  bool opensta_ok = period_ok && slack_ok;
  XLS_RET_CHECK(opensta_ok) << "\"Problem parsing results from OpenSTA. "
                               "OpenSTA may have exited unexpectedly.\"";

  return stats;
}  // ParseOpenSTAOutput
}  // namespace synthesis
}  // namespace xls
