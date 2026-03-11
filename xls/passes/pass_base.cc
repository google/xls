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

#include <cstdint>

#include "google/protobuf/duration.pb.h"
#include "absl/time/time.h"
#include "xls/ir/package.h"
#include "xls/passes/pass_metrics.pb.h"

namespace xls {
namespace {

void InvocationToProto(const PassInvocation& invocation,
                       PassMetricsProto* proto) {
  proto->set_pass_name(invocation.pass_name());
  proto->set_changed(invocation.ir_changed());
  absl::Duration rem;
  int64_t s =
      absl::IDivDuration(invocation.run_duration(), absl::Seconds(1), &rem);
  int64_t n =
      absl::IDivDuration(invocation.run_duration(), absl::Nanoseconds(1), &rem);
  proto->mutable_duration()->set_seconds(s);
  proto->mutable_duration()->set_nanos(n);
  *proto->mutable_transformation_metrics() = invocation.metrics().ToProto();
  if (invocation.fixed_point_iterations() > 0) {
    proto->set_fixed_point_iterations(invocation.fixed_point_iterations());
  }
  for (const auto& nested_invocation : invocation.nested_invocations()) {
    InvocationToProto(*nested_invocation, proto->add_nested_results());
  }
  for (int64_t pass_number : invocation.all_pass_numbers()) {
    proto->add_pass_numbers(pass_number);
  }
}

}  // namespace

PassPipelineMetricsProto PassResults::ToProto() const {
  PassPipelineMetricsProto proto;
  proto.set_total_passes(total_invocations_);
  // Generally the results should have only been used for a single root compound
  // pass so skip emitting the 'root' invocation if that seems to be the case.
  if (root_invocation_->nested_invocations().size() == 1) {
    InvocationToProto(*root_invocation_->nested_invocations().front(),
                      proto.mutable_pass_metrics());
  } else {
    InvocationToProto(*root_invocation_, proto.mutable_pass_metrics());
    // Clear most of the data from the root invocation since it doesn't really
    // exist.
    proto.mutable_pass_metrics()->clear_duration();
    proto.mutable_pass_metrics()->clear_transformation_metrics();
    proto.mutable_pass_metrics()->clear_pass_numbers();
    proto.mutable_pass_metrics()->clear_changed();
  }
  return proto;
}

}  // namespace xls
