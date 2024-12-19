// Copyright 2020 The XLS Authors
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

#include "xls/passes/pass_base.h"

#include <algorithm>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include "google/protobuf/duration.pb.h"
#include "absl/strings/str_format.h"
#include "absl/time/time.h"
#include "xls/ir/package.h"
#include "xls/passes/pass_metrics.pb.h"

namespace xls {

void CompoundPassResult::AddSinglePassResult(std::string_view pass_name,
                                             bool changed,
                                             absl::Duration duration,
                                             const TransformMetrics& metrics) {
  SinglePassResult& result = pass_results_[pass_name];
  ++result.run_count;
  result.changed_count += changed ? 1 : 0;
  result.duration = result.duration + duration;
  result.metrics = result.metrics + metrics;
}

void CompoundPassResult::AccumulateCompoundPassResult(
    const CompoundPassResult& other) {
  changed_ = changed_ || other.changed_;
  for (const auto& [pass_name, other_pass_result] : other.pass_results_) {
    SinglePassResult& pass_result = pass_results_[pass_name];
    pass_result.changed_count += other_pass_result.changed_count;
    pass_result.run_count += other_pass_result.run_count;
    pass_result.duration = pass_result.duration + other_pass_result.duration;
    pass_result.metrics = pass_result.metrics + other_pass_result.metrics;
  }
}

std::string CompoundPassResult::ToString() const {
  std::string s;
  std::vector<std::string> pass_names;
  for (const auto& [name, _] : pass_results_) {
    pass_names.push_back(name);
  }
  std::sort(pass_names.begin(), pass_names.end(),
            [&](const std::string& a, const std::string& b) {
              return pass_results_.at(a).duration >
                     pass_results_.at(b).duration;
            });

  s = "Compound pass statistics:\n";
  for (const std::string& name : pass_names) {
    SinglePassResult result = pass_results_.at(name);
    absl::StrAppendFormat(
        &s, "  %15s: changed %d/%d, total time %s, metrics %s\n", name,
        result.changed_count, result.run_count, FormatDuration(result.duration),
        result.metrics.ToString());
  }
  return s;
}

PassResultProto SinglePassResult::ToProto() const {
  PassResultProto res;
  res.set_run_count(run_count);
  res.set_changed_count(changed_count);
  *res.mutable_metrics() = metrics.ToProto();

  absl::Duration rem;
  int64_t s = absl::IDivDuration(duration, absl::Seconds(1), &rem);
  int64_t n = absl::IDivDuration(duration, absl::Nanoseconds(1), &rem);
  res.mutable_pass_duration()->set_seconds(s);
  res.mutable_pass_duration()->set_nanos(n);
  return res;
}

PipelineMetricsProto CompoundPassResult::ToProto() const {
  PipelineMetricsProto res;
  for (const auto& [name, result] : pass_results_) {
    res.mutable_pass_results()->insert({name, result.ToProto()});
  }
  return res;
}

}  // namespace xls
