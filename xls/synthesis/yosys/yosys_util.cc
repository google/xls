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
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/ascii.h"
#include "absl/strings/match.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_split.h"
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
  // TODO(meheff): Use regular expressions for this. Unfortunately using RE2
  // causes multiple definition link errors when building yosys_server_test.
  for (auto line : absl::StrSplit(nextpnr_output, '\n')) {
    if (absl::StartsWith(line, "Info: Max frequency for clock") &&
        absl::StrContains(line, " MHz ")) {
      std::vector<std::string_view> tokens = absl::StrSplit(line, ' ');
      for (int64_t i = 1; i < tokens.size(); ++i) {
        if (tokens[i] == "MHz") {
          if (absl::SimpleAtod(tokens[i - 1], &max_mhz)) {
            found = true;
            break;
          }
        }
      }
    }
  }
  if (!found) {
    return absl::NotFoundError(
        "Could not find maximum frequency in nextpnr output.");
  }

  return static_cast<int64_t>(max_mhz * 1e6);
}

absl::StatusOr<YosysSynthesisStatistics> ParseYosysOutput(
    std::string_view yosys_output) {
  YosysSynthesisStatistics stats;
  stats.area = -1.0f;
  stats.sequential_area = -1.0f;
  std::vector<std::string> lines = absl::StrSplit(yosys_output, '\n');
  std::vector<std::string>::iterator parse_line_itr = lines.begin();

  // Advance parse_line_index until a line starting with 'key' is found.
  // Return false if 'key' is not found, otherwise true.
  auto parse_until_found_string_starts_with = [&](std::string_view key) {
    for (; parse_line_itr != lines.end(); ++parse_line_itr) {
      if (absl::StartsWith(*parse_line_itr, key)) {
        return true;
      }
    }
    return false;
  };

  // Advance parse_line_index until a line containing 'key' is found.
  // Return false if 'key' is not found, otherwise true.
  auto parse_until_found_string_contains = [&](std::string_view key) {
    for (; parse_line_itr != lines.end(); ++parse_line_itr) {
      if (absl::StrContains(*parse_line_itr, key)) {
        return true;
      }
    }
    return false;
  };

  // Find the XLS marker for the statistics section - the set of statistics
  // we are interested in is printed after this marker.
  if (!parse_until_found_string_starts_with(
          "XLS marker: statistics section starts here")) {
    return absl::InternalError(
        "ParseYosysOutput could not find the term \"XLS marker: statistics "
        "section starts here\" in the yosys output");
  }
  // Find the "Number of cells:" line. The cell histogram is printed immediately
  // after this line.
  if (!parse_until_found_string_contains("Number of cells:")) {
    return absl::InternalError(
        "ParseYosysOutput could not find the term \"Number of cells:\" in the "
        "yosys output");
  }
  parse_line_itr++;

  // Process cell histogram.
  for (; parse_line_itr != lines.end(); ++parse_line_itr) {
    int64_t cell_count;
    std::string cell_name;
    if (RE2::FullMatch(*parse_line_itr, "\\s+(\\w+)\\s+(\\d+)\\s*", &cell_name,
                       &cell_count)) {
      XLS_RET_CHECK(!stats.cell_histogram.contains(cell_name));
      stats.cell_histogram[cell_name] = cell_count;
    } else {
      break;
    }
  }

  constexpr std::string_view floating_point_regex = "([0-9\\+\\-e\\.]+)";
  // The stats related to area, if exists, should come immediately after the
  // cell histogram. We need to be careful of not jumping to another set of
  // statistics. Ideally, we should have an end marker as the area part might or
  // might not exist. However, we find that printing another marker after
  // calling `yosys stats` might result in the marker getting mixed in the
  // statistics output. Therefore, we use the following heuristic: if we see the
  // "Number of cells:" line again, we are probably in another set of
  // statistics, and we should stop parsing.
  while (parse_line_itr != lines.end() &&
         !absl::StrContains(*parse_line_itr, "Number of cells:")) {
    double parsed_area = -1.0f;
    double parsed_sequential_area = -1.0f;
    std::string parsed_cell_type_area_unknown;
    // TODO(hnpl): We should use LazyRE2 for this and the rest of the file.
    if (RE2::FullMatch(*parse_line_itr,
                       "\\s+Area for cell type (\\w+) is unknown!",
                       &parsed_cell_type_area_unknown)) {
      stats.cell_type_with_unknown_area.push_back(
          parsed_cell_type_area_unknown);
    }
    if (RE2::FullMatch(
            *parse_line_itr,
            absl::StrCat("\\s+Chip area for .+: ", floating_point_regex),
            &parsed_area)) {
      stats.area = parsed_area;
    }
    if (RE2::PartialMatch(
            *parse_line_itr,
            absl::StrCat("\\s+of which used for sequential elements: ",
                         floating_point_regex, " "),
            &parsed_sequential_area)) {
      stats.sequential_area = parsed_sequential_area;
      break;
    }
    ++parse_line_itr;
  }

  return stats;
}  //  ParseYosysOutput

absl::StatusOr<STAStatistics> ParseOpenSTAOutput(std::string_view sta_output) {
  STAStatistics stats;

  std::string clk_period_ps;
  std::string freq_mhz;
  float tmp_float = 0.0;
  std::string slack_ps;
  bool period_ok = false, slack_ok = false;

  for (std::string_view line : absl::StrSplit(sta_output, '\n')) {
    line = absl::StripAsciiWhitespace(line);
    // We're looking for lines with this outline, all in ps, freq in mhz
    //
    //   clk period_min = 116.71 fmax = 8568.43
    //   worst slack -96.67
    // And we want to extract 116.71 and 8568.43 and -96.67

    if (RE2::PartialMatch(line,
                          R"(op_clk period_min = (\d+\.\d+) fmax = (\d+\.\d+))",
                          &clk_period_ps, &freq_mhz)) {
      XLS_RET_CHECK(absl::SimpleAtof(clk_period_ps, &stats.period_ps));
      stats.max_frequency_hz = static_cast<int64_t>(
          1e12 / stats.period_ps);  // use clk in ps for accuracy
      period_ok = true;
    }

    if (RE2::PartialMatch(line, R"(^worst slack (-?\d+.\d+))", &slack_ps)) {
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
