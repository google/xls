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

#include "xls/dev_tools/pipeline_metrics.h"

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <optional>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

#include "google/protobuf/duration.pb.h"
#include "absl/container/flat_hash_map.h"
#include "absl/functional/function_ref.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/time/time.h"
#include "xls/ir/package.h"
#include "xls/passes/pass_metrics.pb.h"

namespace xls {
namespace {

int64_t DurationToMs(absl::Duration duration) {
  return duration / absl::Milliseconds(1);
}

absl::Duration ProtoDurationToAbslDuration(
    const google::protobuf::Duration& proto) {
  return absl::Seconds(proto.seconds()) + absl::Nanoseconds(proto.nanos());
}

// Struct holding the aggregation of multiple PassResultProtos.
struct AggregateMetrics {
  std::string pass_name;
  int64_t run_count = 0;
  int64_t changed_count = 0;
  absl::Duration run_duration;
  TransformMetrics metrics;

  AggregateMetrics operator+(const AggregateMetrics& other) const {
    AggregateMetrics result;
    result.pass_name = (pass_name.empty() || pass_name == other.pass_name)
                           ? other.pass_name
                           : "";
    result.run_count = run_count + other.run_count;
    result.changed_count = changed_count + other.changed_count;
    result.run_duration = run_duration + other.run_duration;
    result.metrics = metrics + other.metrics;
    return result;
  }

  static AggregateMetrics FromProto(const PassResultProto& proto) {
    AggregateMetrics metrics;
    metrics.pass_name = proto.pass_name();
    metrics.run_count = 1;
    metrics.changed_count = proto.changed() ? 1 : 0;
    metrics.run_duration = ProtoDurationToAbslDuration(proto.pass_duration());
    metrics.metrics = TransformMetrics::FromProto(proto.metrics());
    return metrics;
  }
};

void AggregatePassResultsInternal(
    const PassResultProto& proto,
    absl::flat_hash_map<std::string, AggregateMetrics>& metrics_map) {
  if (proto.nested_results().empty()) {
    // Non-compound pass.
    AggregateMetrics& metrics = metrics_map[proto.pass_name()];
    metrics = metrics + AggregateMetrics::FromProto(proto);
  } else {
    for (const PassResultProto& nested_proto : proto.nested_results()) {
      AggregatePassResultsInternal(nested_proto, metrics_map);
    }
  }
}

// Recursively walk the pass results within `proto` and aggregate the metrics by
// pass name. Returns a vector sorted (decreasing) by run time.
std::vector<AggregateMetrics> AggregatePassResults(
    const PassResultProto& proto) {
  absl::flat_hash_map<std::string, AggregateMetrics> metrics_map;
  AggregatePassResultsInternal(proto, metrics_map);
  std::vector<AggregateMetrics> metrics;
  for (auto& [_, m] : metrics_map) {
    metrics.push_back(m);
  }
  std::sort(metrics.begin(), metrics.end(),
            [&](const AggregateMetrics& a, const AggregateMetrics& b) {
              // Sort by time (at the same resolution we show), breaking ties by
              // # of times run, # of times changed, and finally pass name.
              auto key = [](const AggregateMetrics& x) {
                return std::tuple(DurationToMs(x.run_duration), x.run_count,
                                  x.changed_count, x.pass_name);
              };
              // Sort high to low.
              return key(a) > key(b);
            });
  return metrics;
}

void BuildHierarchicalTableInternal(
    const PassResultProto& proto, int64_t indent,
    std::vector<std::string>& lines,
    std::optional<AggregateMetrics>& collapsed_summary_metrics) {
  std::string indent_str(indent * 2, ' ');
  if (proto.nested_results().empty()) {
    // Collapse sequences of non-compound passes into a single line.
    if (!collapsed_summary_metrics.has_value()) {
      collapsed_summary_metrics = AggregateMetrics::FromProto(proto);
    } else {
      *collapsed_summary_metrics =
          *collapsed_summary_metrics + AggregateMetrics::FromProto(proto);
    }
    return;
  }

  auto add_line = [&](std::string_view pass_name,
                      const AggregateMetrics& metrics) {
    lines.push_back(absl::StrFormat(
        "%-55s %6dms         %4d/%4d          %8d(+)/%8d(-)/%8d(R)        "
        " %8d(-)/%8d(R)",
        indent_str + std::string{pass_name}, DurationToMs(metrics.run_duration),
        metrics.changed_count, metrics.run_count, metrics.metrics.nodes_added,
        metrics.metrics.nodes_removed, metrics.metrics.nodes_replaced,
        metrics.metrics.operands_removed, metrics.metrics.operands_replaced));
  };

  auto maybe_add_summary_line = [&](bool extra_indent) {
    if (collapsed_summary_metrics.has_value()) {
      add_line(absl::StrFormat("%s[%d passes run]", extra_indent ? "  " : "",
                               collapsed_summary_metrics->run_count),
               *collapsed_summary_metrics);
      collapsed_summary_metrics.reset();
    }
  };

  maybe_add_summary_line(false);

  std::vector<std::pair<int64_t, int64_t>> intervals;
  if (proto.fixed_point_iterations() > 0) {
    // Fixed-point pass. Break the nested results into iterations.
    int64_t end = 0;
    int64_t pass_count =
        proto.nested_results().size() / proto.fixed_point_iterations();
    while (end < proto.nested_results().size()) {
      int64_t next_end =
          std::min(int64_t{proto.nested_results().size()}, end + pass_count);
      intervals.push_back({end, next_end});
      end = next_end;
    }
  } else {
    // Non-fixed point pass. Aggregate all nested results together.
    intervals.push_back({0, proto.nested_results().size()});
  }

  int64_t iteration = 0;
  for (auto [start, end] : intervals) {
    AggregateMetrics interval_metrics;
    for (int64_t i = start; i < end; ++i) {
      interval_metrics = interval_metrics +
                         AggregateMetrics::FromProto(proto.nested_results()[i]);
    }
    std::string pass_name =
        proto.fixed_point_iterations() > 0
            ? absl::StrFormat("%s [iter #%d]", proto.pass_name(), iteration)
            : proto.pass_name();
    add_line(pass_name, interval_metrics);
    for (int64_t i = start; i < end; ++i) {
      const PassResultProto& nested_proto = proto.nested_results()[i];
      BuildHierarchicalTableInternal(nested_proto, indent + 1, lines,
                                     collapsed_summary_metrics);
    }

    maybe_add_summary_line(true);

    ++iteration;
  }
}

// Returns the lines of a table which mirrors the hierarchical structure of the
// (compound) passes which generated the metrics in `proto`.
std::string BuildHierarchicalTable(const PassResultProto& proto) {
  std::vector<std::string> lines;
  std::optional<AggregateMetrics> collapsed_summary_metrics;
  BuildHierarchicalTableInternal(proto, 0, lines, collapsed_summary_metrics);
  return absl::StrCat(absl::StrJoin(lines, "\n"), "\n");
}

// Returns the sum of leaf pass duration for those pass results which for which
// filter function is true.
absl::Duration TotalTime(
    const PassResultProto& result,
    absl::FunctionRef<bool(const PassResultProto& result)> filter) {
  if (!filter(result)) {
    return absl::Duration();
  }
  if (result.nested_results().empty()) {
    return ProtoDurationToAbslDuration(result.pass_duration());
  }
  absl::Duration total;
  for (const PassResultProto& nested_result : result.nested_results()) {
    total += TotalTime(nested_result, filter);
  }
  return total;
}

}  // namespace

std::string SummarizePipelineMetrics(const PipelineMetricsProto& metrics) {
  // The metrics object is recursive. Aggregate the results by pass name.
  std::vector<AggregateMetrics> aggregate_metrics =
      AggregatePassResults(metrics.pass_results());
  std::string str = "Aggregate pass statistics:\n\n";
  absl::StrAppendFormat(
      &str,
      "%-30s   Duration  Runs: changed/total  Nodes "
      "added(+)/removed(-)/replaced(R)   Operands removed(-)/replaced(R)\n",
      "Pass name");
  std::string divider = std::string(135, '-') + "\n";
  absl::StrAppend(&str, divider);
  auto make_line = [](const AggregateMetrics& metric) {
    return absl::StrFormat(
        "  %-30s %6dms       %4d/%4d         %8d(+)/%8d(-)/%8d(R)        "
        " %8d(-)/%8d(R)\n",
        metric.pass_name, DurationToMs(metric.run_duration),
        metric.changed_count, metric.run_count, metric.metrics.nodes_added,
        metric.metrics.nodes_removed, metric.metrics.nodes_replaced,
        metric.metrics.operands_removed, metric.metrics.operands_replaced);
  };
  for (const AggregateMetrics& metric : aggregate_metrics) {
    absl::StrAppend(&str, make_line(metric));
  }
  absl::StrAppend(&str, divider);
  AggregateMetrics total = std::accumulate(
      aggregate_metrics.begin(), aggregate_metrics.end(), AggregateMetrics());
  total.pass_name = "Total";
  absl::StrAppend(&str, make_line(total));

  // Add line showing the amount of time spent running passes which did not
  // change the IR.
  int64_t changed_ms = DurationToMs(TotalTime(
      metrics.pass_results(),
      [](const PassResultProto& result) { return result.changed(); }));
  int64_t total_ms = DurationToMs(
      TotalTime(metrics.pass_results(),
                [](const PassResultProto& result) { return true; }));
  absl::StrAppendFormat(
      &str,
      "\nTotal time on passes which changed IR (changed/total): %6dms/%6dms "
      "(%s)\n",
      changed_ms, total_ms,
      total_ms == 0 ? "0.0%"
                    : absl::StrFormat("%0.1f%%", 100.0 * (float)changed_ms /
                                                     (float)total_ms));

  absl::StrAppend(&str, "\nHierarchical pass statistics:\n\n");
  absl::StrAppendFormat(
      &str,
      "%-55s Duration     Runs: changed/total   Nodes "
      "added(+)/removed(-)/replaced(R)   Operands removed(-)/replaced(R)\n",
      "Pass name");
  absl::StrAppend(&str, std::string(161, '-') + "\n");
  absl::StrAppend(&str, BuildHierarchicalTable(metrics.pass_results()));
  return str;
}

}  // namespace xls
