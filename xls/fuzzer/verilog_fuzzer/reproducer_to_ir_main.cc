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

#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "xls/common/fuzzing/fuzztest.h"
#include "absl/flags/flag.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "xls/common/exit_status.h"
#include "xls/common/file/filesystem.h"
#include "xls/common/init_xls.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/fuzzer/ir_fuzzer/ir_fuzz_domain.h"
#include "xls/fuzzer/ir_fuzzer/reproducer_repo_files.h"
#include "xls/fuzzer/verilog_fuzzer/verilog_fuzz_domain.h"

constexpr std::string_view kUsage = R"(
Parse a codegen_fuzz_test reproducer and print the generated IR and config.
)";

ABSL_FLAG(std::string, ir_out, "/dev/stdout", "File to dump the IR to.");
ABSL_FLAG(std::string, codegen_options_textproto_out, "/dev/null",
          "File to dump the codegen options to.");
ABSL_FLAG(std::string, codegen_options_proto_out, "/dev/null",
          "File to dump the codegen options to.");
ABSL_FLAG(std::string, scheduling_options_textproto_out, "/dev/null",
          "File to dump the scheduling options to.");
ABSL_FLAG(std::string, scheduling_options_proto_out, "/dev/null",
          "File to dump the scheduling options to.");
ABSL_FLAG(std::string, test_case, "",
          "What test this is from. Must be either "
          "'CodegenSucceedsForEveryFunctionWithNoPipelineLimit' or "
          "'CodegenSucceedsOrThrowsReasonableError'.");

namespace xls {
namespace {

absl::Status RealMain(std::string_view file_v) {
  std::string file = std::string(file_v);
  if (IsFuzztestReproPath(file)) {
    XLS_ASSIGN_OR_RETURN(
        file, FuzztestRepoToFilePath(file),
        _ << "Unable to find file for fuzztest repo target: " << file);
  }
  XLS_ASSIGN_OR_RETURN(std::string repro, GetFileContents(file), _ << file);
  VerilogGenerator gen;
  if (absl::GetFlag(FLAGS_test_case) ==
      "CodegenSucceedsForEveryFunctionWithNoPipelineLimit") {
    auto domain = VerilogGeneratorDomain(
        IrFuzzDomain(), fuzztest::Just(DefaultSchedulingOptions()),
        fuzztest::Just(DefaultCodegenOptions()));
    XLS_ASSIGN_OR_RETURN(
        auto [v], fuzztest::unstable::ParseReproducerValue(repro, domain));
    gen = std::move(v);
  } else {
    XLS_RET_CHECK_EQ(absl::GetFlag(FLAGS_test_case),
                     "CodegenSucceedsOrThrowsReasonableError");
    auto domain = VerilogGeneratorDomain(IrFuzzDomain(),
                                         NoFdoSchedulingOptionsFlagsDomain(),
                                         CodegenFlagsDomain());
    XLS_ASSIGN_OR_RETURN(
        auto [v], fuzztest::unstable::ParseReproducerValue(repro, domain));
    gen = std::move(v);
  }
  XLS_RETURN_IF_ERROR(
      SetFileContents(absl::GetFlag(FLAGS_ir_out), gen.package->DumpIr()));
  XLS_RETURN_IF_ERROR(SetTextProtoFile(
      absl::GetFlag(FLAGS_codegen_options_textproto_out), gen.codegen_options));
  XLS_RETURN_IF_ERROR(
      SetFileContents(absl::GetFlag(FLAGS_codegen_options_proto_out),
                      gen.codegen_options.SerializeAsString()));
  XLS_RETURN_IF_ERROR(
      SetTextProtoFile(absl::GetFlag(FLAGS_scheduling_options_textproto_out),
                       gen.scheduling_options));
  XLS_RETURN_IF_ERROR(
      SetFileContents(absl::GetFlag(FLAGS_scheduling_options_proto_out),
                      gen.scheduling_options.SerializeAsString()));
  return absl::OkStatus();
}

}  // namespace
}  // namespace xls

int main(int argc, char** argv) {
  std::vector<std::string_view> positional_arguments =
      xls::InitXls(kUsage, argc, argv);

  if (positional_arguments.size() != 1) {
    LOG(QFATAL) << "Expected invocation: " << argv[0] << " <repro_file>";
  }

  return xls::ExitStatus(xls::RealMain(positional_arguments[0]));
}
