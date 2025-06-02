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

#include "xls/tools/typecheck_flags.h"

#include <iostream>
#include <string>

#include "absl/flags/flag.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xls/tools/typecheck_flags.pb.h"

ABSL_FLAG(std::string, typecheck_trace_out_dir, "",
          "The directory in which to output type inference trace artifacts.");

ABSL_FLAG(
    bool, typecheck_dump_inference_table, false,
    "Whether to dump the inference table to a file in `trace_output_dir`.");

ABSL_FLAG(bool, typecheck_dump_traces, false,
          "Whether to dump per-module traces of the details of what was done "
          "during type inference in `trace_ouput_dir`.");

namespace xls {

absl::StatusOr<TypecheckFlagsProto> GetTypecheckFlagsProto() {
  TypecheckFlagsProto proto;
  std::string trace_out_dir = absl::GetFlag(FLAGS_typecheck_trace_out_dir);
  bool dump_inference_table =
      absl::GetFlag(FLAGS_typecheck_dump_inference_table);
  bool dump_traces = absl::GetFlag(FLAGS_typecheck_dump_traces);

  if (trace_out_dir.empty()) {
    if (dump_inference_table || dump_traces) {
      return absl::InvalidArgumentError(
          "In order to output inference table or traces, "
          "--typecheck_trace_out_dir must be specified.");
    }
  } else {
    proto.set_trace_out_dir(trace_out_dir);
    if (!dump_inference_table && !dump_traces) {
      std::cerr << "Warning: --typecheck_trace_out_dir used without any flags "
                   "for generating trace artifacts.\n";
    }
  }

  proto.set_dump_inference_table(dump_inference_table);
  proto.set_dump_traces(dump_traces);
  return proto;
}

}  // namespace xls
