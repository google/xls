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
  proto->set_pass_name(invocation.pass_name);
  proto->set_changed(invocation.ir_changed);
  absl::Duration rem;
  int64_t s =
      absl::IDivDuration(invocation.run_duration, absl::Seconds(1), &rem);
  int64_t n =
      absl::IDivDuration(invocation.run_duration, absl::Nanoseconds(1), &rem);
  proto->mutable_duration()->set_seconds(s);
  proto->mutable_duration()->set_nanos(n);
  *proto->mutable_transformation_metrics() = invocation.metrics.ToProto();
  proto->set_fixed_point_iterations(invocation.fixed_point_iterations);
  for (const PassInvocation& nested_invocation :
       invocation.nested_invocations) {
    InvocationToProto(nested_invocation, proto->add_nested_results());
  }
}

}  // namespace

PassPipelineMetricsProto PassResults::ToProto() const {
  PassPipelineMetricsProto proto;
  proto.set_total_passes(total_invocations);
  InvocationToProto(invocation, proto.mutable_pass_metrics());
  return proto;
}

}  // namespace xls
