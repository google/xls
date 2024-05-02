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

#include "xls/ir/format_strings.h"

#include <algorithm>
#include <cstdint>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_replace.h"
#include "absl/types/span.h"

namespace xls {

absl::StatusOr<std::vector<FormatStep>> ParseFormatString(
    std::string_view format_string) {
  std::vector<FormatStep> steps;

  int64_t i = 0;
  auto consume_substr = [&i, format_string](std::string_view m) -> bool {
    if (format_string.substr(i, m.length()) == m) {
      i = i + m.length();
      return true;
    }
    return false;
  };

  std::string fragment;
  fragment.reserve(format_string.length());

  auto push_fragment = [&fragment, &steps]() {
    if (!fragment.empty()) {
      steps.push_back(fragment);
      fragment.clear();
    }
  };

  while (i < format_string.length()) {
    if (consume_substr("{{")) {
      absl::StrAppend(&fragment, "{{");
      continue;
    }
    if (consume_substr("}}")) {
      absl::StrAppend(&fragment, "}}");
      continue;
    }
    if (consume_substr("{}")) {
      push_fragment();
      steps.push_back(FormatPreference::kDefault);
      continue;
    }
    if (consume_substr("{:u}")) {
      push_fragment();
      steps.push_back(FormatPreference::kUnsignedDecimal);
      continue;
    }
    if (consume_substr("{:d}")) {
      push_fragment();
      steps.push_back(FormatPreference::kSignedDecimal);
      continue;
    }
    if (consume_substr("{:x}")) {
      push_fragment();
      steps.push_back(FormatPreference::kPlainHex);
      continue;
    }
    if (consume_substr("{:#x}")) {
      push_fragment();
      steps.push_back(FormatPreference::kHex);
      continue;
    }
    if (consume_substr("{:b}")) {
      push_fragment();
      steps.push_back(FormatPreference::kPlainBinary);
      continue;
    }
    if (consume_substr("{:#b}")) {
      push_fragment();
      steps.push_back(FormatPreference::kBinary);
      continue;
    }
    if (format_string[i] == '{') {
      size_t close_pos = format_string.find('}', i);
      if (close_pos != std::string_view::npos) {
        return absl::InvalidArgumentError(absl::StrFormat(
            "Invalid or unsupported format specifier \"%s\" in format string "
            "\"%s\"",
            format_string.substr(i, close_pos - i + 1), format_string));
      }
      return absl::InvalidArgumentError(absl::StrFormat(
          "{ without matching } at position %d in format string \"%s\"", i,
          format_string));
    }
    if (format_string[i] == '}') {
      return absl::InvalidArgumentError(absl::StrFormat(
          "} with no preceding { at position %d in format string \"%s\"", i,
          format_string));
    }

    fragment += format_string[i];
    i = i + 1;
  }

  push_fragment();
  return steps;
}

std::vector<FormatPreference> OperandPreferencesFromFormat(
    absl::Span<const FormatStep> format) {
  std::vector<FormatPreference> preferences;
  for (const FormatStep& step : format) {
    if (std::holds_alternative<FormatPreference>(step)) {
      preferences.push_back(std::get<FormatPreference>(step));
    }
  }
  return preferences;
}

int64_t OperandsExpectedByFormat(absl::Span<const FormatStep> format) {
  return std::count_if(format.begin(), format.end(),
                       [](const FormatStep& step) {
                         return std::holds_alternative<FormatPreference>(step);
                       });
}

std::string StepsToXlsFormatString(absl::Span<const FormatStep> format) {
  return absl::StrJoin(
      format, "", [](std::string* out, const FormatStep& step) {
        if (std::holds_alternative<FormatPreference>(step)) {
          absl::StrAppend(out, FormatPreferenceToXlsSpecifier(
                                   std::get<FormatPreference>(step)));
        } else {
          absl::StrAppend(out, std::get<std::string>(step));
        }
      });
}

std::string StepsToVerilogFormatString(absl::Span<const FormatStep> format) {
  return absl::StrJoin(
      format, "", [](std::string* out, const FormatStep& step) {
        if (std::holds_alternative<FormatPreference>(step)) {
          absl::StrAppend(out, FormatPreferenceToVerilogSpecifier(
                                   std::get<FormatPreference>(step)));
        } else {
          // Convert {{ and }} to { and }.
          std::string step_str = absl::StrReplaceAll(
              std::get<std::string>(step), {{"{{", "{"}, {"}}", "}"}});

          absl::StrAppend(out, step_str);
        }
      });
}

}  // namespace xls
