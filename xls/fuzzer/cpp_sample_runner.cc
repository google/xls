// Copyright 2023 The XLS Authors
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

#include "xls/fuzzer/cpp_sample_runner.h"

#include <algorithm>
#include <cstddef>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "xls/common/status/status_macros.h"
#include "xls/dslx/interp_value.h"
#include "xls/ir/format_preference.h"
#include "xls/ir/value.h"

namespace xls {
namespace {

absl::StatusOr<std::string> ToIrString(const dslx::InterpValue& v) {
  XLS_ASSIGN_OR_RETURN(xls::Value value, v.ConvertToIr());
  return value.ToString(FormatPreference::kHex);
}

}  // namespace

absl::Status CompareResultsFunction(
    const absl::flat_hash_map<std::string, absl::Span<const dslx::InterpValue>>&
        results,
    const ArgsBatch* maybe_args_batch) {
  if (results.empty()) {
    return absl::OkStatus();
  }

  // We sort the results so we iterate in stable / repeatable ways.
  std::vector<std::pair<std::string, absl::Span<const dslx::InterpValue>>>
      sorted_results_items(results.begin(), results.end());
  std::sort(
      sorted_results_items.begin(), sorted_results_items.end(),
      [](const auto& lhs, const auto& rhs) { return lhs.first < rhs.first; });

  if (maybe_args_batch != nullptr) {
    // Check length is the same as results.
    for (const auto& [k, v] : sorted_results_items) {
      if (maybe_args_batch->size() != v.size()) {
        return absl::InvalidArgumentError(
            absl::StrFormat("SampleError: Results for %s has %d values, "
                            "argument batch has %d values",
                            k, v.size(), maybe_args_batch->size()));
      }
    }
  }

  std::optional<std::string> reference;
  for (const auto& [name, result_values] : sorted_results_items) {
    if (!reference.has_value()) {
      reference = name;
      continue;
    }

    if (results.at(reference.value()).size() != result_values.size()) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "SampleError: Results for %s has %d value(s), %s has %d value(s)",
          reference.value(), results.at(reference.value()).size(), name,
          result_values.size()));
    }

    for (size_t i = 0; i < result_values.size(); ++i) {
      const dslx::InterpValue& ref_result = results.at(reference.value()).at(i);
      const dslx::InterpValue& value = result_values.at(i);
      if (ref_result.Eq(value)) {
        continue;
      }
      // Bin all of the sources by whether they match the reference or 'values'.
      // This helps identify which of the two is likely correct.
      std::string args_str = "(args unknown)";
      if (maybe_args_batch != nullptr) {
        const std::vector<dslx::InterpValue>& batch_item =
            maybe_args_batch->at(i);
        args_str = absl::StrJoin(
            batch_item, "; ", [](std::string* out, const dslx::InterpValue& v) {
              absl::StrAppend(out, ToIrString(v).value());
            });
      }
      std::vector<std::string> reference_matches;
      std::vector<std::string> values_matches;
      for (const auto& [k, vs] : results) {
        if (vs[i].Eq(ref_result)) {
          reference_matches.push_back(k);
        }
        if (vs[i].Eq(value)) {
          values_matches.push_back(k);
        }
      }
      std::sort(reference_matches.begin(), reference_matches.end());
      std::sort(values_matches.begin(), values_matches.end());

      XLS_ASSIGN_OR_RETURN(std::string ref_result_str, ToIrString(ref_result));
      XLS_ASSIGN_OR_RETURN(std::string value_str, ToIrString(value));
      return absl::InvalidArgumentError(absl::StrFormat(
          "SampleError: Result miscompare for sample %d:\nargs: %s\n%s =\n   "
          "%s\n%s =\n   %s",
          i, args_str, absl::StrJoin(reference_matches, ", "), ref_result_str,
          absl::StrJoin(values_matches, ", "), value_str));
    }
  }

  return absl::OkStatus();
}

}  // namespace xls
