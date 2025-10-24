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

#include <optional>
#include <string>
#include <string_view>

#include "absl/flags/flag.h"
#include "xls/tools/delay_info_flags.pb.h"

ABSL_FLAG(
    std::string, top, "",
    "The name of the top entity. Currently, only functions are supported. "
    "Function to emit delay information about.");
ABSL_FLAG(std::string, schedule_path, "",
          "Optional path to a pipeline schedule to use for emitting per-stage "
          "critical paths.");
ABSL_FLAG(bool, schedule, false,
          "Run scheduling to generate a schedule for delay analysis, rather "
          "than reading a schedule via --schedule_path.");
ABSL_FLAG(bool, compare_to_synthesis, false,
          "Whether to compare the delay info from the XLS delay model to "
          "synthesizer output.");
ABSL_FLAG(std::string, synthesis_server, "ipv4:///0.0.0.0:10000",
          "The address, including port, of the gRPC server to use with "
          "--compare_to_synthesis.");
ABSL_FLAG(int, abs_delay_diff_min_ps, 0,
          "Return an error exit code if the absolute value of `synthesized "
          "delay - delay model prediction` is below this threshold. This "
          "enables use of delay_info_main as a helper for ir_minimizer_main, "
          "to find the minimal IR exhibiting a minimum difference. "
          "`compare_to_synthesis` must also be true.");
ABSL_FLAG(std::optional<int>, stage, std::nullopt,
          "Only analyze the specified, zero-based stage of the pipeline.");
ABSL_FLAG(std::optional<std::string>, proto_out, std::nullopt,
          "File to write a binary xls.DelayInfoProto to containing delay info "
          "of the input.");

namespace xls {

DelayInfoFlagsProto GetDelayInfoFlagsProto(std::string_view input_path) {
#define POPULATE_FLAG(__x)                         \
  {                                                \
    if (FLAGS_##__x.IsSpecifiedOnCommandLine()) {  \
      proto.set_##__x(absl::GetFlag(FLAGS_##__x)); \
    }                                              \
  }
#define POPULATE_OPTIONAL_FLAG(__x)                       \
  {                                                       \
    if (auto optional_value = absl::GetFlag(FLAGS_##__x); \
        optional_value.has_value()) {                     \
      proto.set_##__x(*optional_value);                   \
    }                                                     \
  }

  DelayInfoFlagsProto proto;
  proto.set_input_path(input_path);
  POPULATE_FLAG(top)
  POPULATE_FLAG(schedule_path)
  POPULATE_FLAG(schedule)
  POPULATE_FLAG(compare_to_synthesis)
  POPULATE_FLAG(abs_delay_diff_min_ps)
  POPULATE_OPTIONAL_FLAG(stage)
  POPULATE_OPTIONAL_FLAG(proto_out)

  // We want this even when defaulted (which is most common).
  proto.set_synthesis_server(absl::GetFlag(FLAGS_synthesis_server));
  return proto;

#undef POPULATE_OPTIONAL_FLAG
#undef POPULATE_FLAG
}

}  // namespace xls
