// Copyright 2026 The XLS Authors
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

// XLS IR -> Promela code generator for the SPIN model checker.
// Usage: promela_converter_main [--output=PATH] IR_FILE  (stdout if --output omitted)

#include <iostream>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_split.h"
#include "xls/common/exit_status.h"
#include "xls/common/file/filesystem.h"
#include "xls/common/init_xls.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/ir_parser.h"
#include "xls/spin/promela_generator.h"

ABSL_FLAG(std::string, output, "",
          "Path for the generated Promela file. Writes to stdout if empty.");

ABSL_FLAG(std::string, top, "",
          "Name of the top function or proc. Required when the package "
          "contains multiple entry points and no top is already set.");

ABSL_FLAG(bool, emit_source_locations, false,
          "Annotate each generated Promela statement with a comment of the "
          "form /* filename:line:col */ derived from the XLS IR source "
          "location. Useful for tracing generated Promela back to the "
          "original DSLX source.");

ABSL_FLAG(bool, emit_source_hints, false,
          "Annotate each generated Promela statement with a comment "
          "containing the original IR node expression, e.g.: "
          "/* ir: add.1: bits[32] = add(a, b, id=1) */");

ABSL_FLAG(int64_t, channel_depth, 8,
          "Buffer depth for every declared Promela channel "
          "(chan x = [N] of {T}). Should match the FIFO depth of the "
          "corresponding XLS channel.");

ABSL_FLAG(bool, emit_termination_hook, false,
          "Append a blocking receive on the terminator channel to the init "
          "block. init blocks until the test proc signals completion, making "
          "the test end point visible in the simulation trace.");

ABSL_FLAG(bool, assert_send_on_full_channel, false,
          "Prefix every send operation with assert(len(ch) < DEPTH). "
          "Turns a blocked-on-full-channel situation into an explicit SPIN "
          "assertion violation during exhaustive verification. Useful for "
          "proving that no execution path sends to a saturated channel.");

ABSL_FLAG(std::string, worst_case_throughput, "",
          "Comma-separated list of ProcName:N pairs specifying the worst-case "
          "throughput for individual procs (e.g. 'SlowProc:4,OtherProc:2'). "
          "ProcName must match the sanitised Promela proctype name as emitted "
          "in the generated 'proctype <name>() {' header. N is the number of "
          "loop iterations between productive steps; procs not listed run "
          "every iteration (N=1).");

ABSL_FLAG(bool, emit_progress_labels, false,
          "Prefix each channel send and receive with a SPIN progress label "
          "(progress_recv_<chan>: or progress_send_<chan>:). "
          "Progress labels are used with spin -search -DNP for non-progress "
          "cycle (livelock) detection: a cycle that visits no progress-labeled "
          "state is flagged as a livelock.");

namespace {
constexpr std::string_view kUsage = R"(
Generates Promela source code from an XLS IR file for model checking with
the SPIN verifier (https://spinroot.com).

XLS functions are emitted as Promela inline procedures.
XLS procs are emitted as Promela proctypes.
XLS channels are emitted as Promela chan declarations.

Example:
  promela_converter_main my_design.ir
  promela_converter_main --output=model.pml my_design.ir
)";
}  // namespace

namespace xls {
namespace {

absl::Status RealMain(std::string_view ir_path) {
  if (ir_path == "-") {
    ir_path = "/dev/stdin";
  }

  LOG(INFO) << "Parsing IR from '" << ir_path << "'";
  XLS_ASSIGN_OR_RETURN(std::string ir_contents, GetFileContents(ir_path));
  XLS_ASSIGN_OR_RETURN(std::unique_ptr<Package> package,
                       Parser::ParsePackage(ir_contents, ir_path));

  const std::string top_name = absl::GetFlag(FLAGS_top);
  if (!top_name.empty() && !package->GetTop().has_value()) {
    XLS_RETURN_IF_ERROR(package->SetTopByName(top_name));
  }

  spin::PromelaGeneratorOptions options;
  options.emit_source_locations = absl::GetFlag(FLAGS_emit_source_locations);
  options.emit_source_hints = absl::GetFlag(FLAGS_emit_source_hints);
  options.channel_depth = absl::GetFlag(FLAGS_channel_depth);
  options.emit_termination_hook = absl::GetFlag(FLAGS_emit_termination_hook);
  options.assert_send_on_full_channel =
      absl::GetFlag(FLAGS_assert_send_on_full_channel);
  options.emit_progress_labels = absl::GetFlag(FLAGS_emit_progress_labels);

  const std::string throughput_flag =
      absl::GetFlag(FLAGS_worst_case_throughput);
  if (!throughput_flag.empty()) {
    for (std::string_view entry : absl::StrSplit(throughput_flag, ',')) {
      std::vector<std::string_view> parts = absl::StrSplit(entry, ':');
      if (parts.size() != 2) {
        return absl::InvalidArgumentError(absl::StrFormat(
            "--worst_case_throughput: expected 'Name:N', got '%s'", entry));
      }
      int64_t throughput_value;
      if (!absl::SimpleAtoi(parts[1], &throughput_value) ||
          throughput_value < 1) {
        return absl::InvalidArgumentError(absl::StrFormat(
            "--worst_case_throughput: N must be a positive integer, got '%s'",
            parts[1]));
      }
      options.worst_case_throughput[std::string(parts[0])] = throughput_value;
    }
  }

  LOG(INFO) << "Generating Promela for package '" << package->name() << "'";
  XLS_ASSIGN_OR_RETURN(
      std::string promela_text,
      spin::PromelaGenerator::Generate(package.get(), options));

  const std::string output_path = absl::GetFlag(FLAGS_output);
  if (output_path.empty()) {
    LOG(INFO) << "Writing Promela to stdout";
    std::cout << promela_text;
  } else {
    LOG(INFO) << "Writing Promela to '" << output_path << "'";
    XLS_RETURN_IF_ERROR(SetFileContents(output_path, promela_text));
  }

  return absl::OkStatus();
}

}  // namespace
}  // namespace xls

int main(int argc, char** argv) {
  std::vector<std::string_view> positional_arguments =
      xls::InitXls(kUsage, argc, argv);

  if (positional_arguments.size() != 1) {
    LOG(QFATAL) << absl::StreamFormat("Expected invocation: %s IR_FILE",
                                      argv[0]);
  }
  std::string_view ir_path = positional_arguments[0];
  return xls::ExitStatus(xls::RealMain(ir_path));
}
