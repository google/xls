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

// Compares a SPIN trace against a DSLX trace per channel (order must match).
// Usage: promela_trace_compare --spin_trace=<spin.json>
//          --dslx_trace=<dslx.textproto> [--dslx_file=<src.x>]

#include <array>
#include <filesystem>
#include <iostream>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "xls/common/file/filesystem.h"
#include "xls/common/init_xls.h"
#include "xls/common/status/status_macros.h"
#include "xls/dslx/create_import_data.h"
#include "xls/dslx/default_dslx_stdlib_path.h"
#include "xls/dslx/frontend/proc.h"
#include "xls/dslx/ir_convert/ir_converter.h"
#include "xls/dslx/parse_and_typecheck.h"
#include "xls/dslx/virtualizable_file_system.h"
#include "xls/dslx/warning_kind.h"
#include "xls/spin/spin_runner.h"
#include "xls/spin/trace_compare.h"

ABSL_FLAG(std::string, spin_trace, "", "Path to SPIN simulation trace JSON.");
ABSL_FLAG(std::string, dslx_trace, "",
          "Path to DSLX interpreter trace textproto.");
ABSL_FLAG(std::string, terminator_channel, "",
          "Channel name used as a termination signal. When non-empty, "
          "both traces are truncated after the first SEND on this channel.");
ABSL_FLAG(std::string, dslx_file, "",
          "Path to the DSLX source file. Derives channel-name rewriting and "
          "proc-hierarchy paths from the source.");
ABSL_FLAG(std::string, dslx_stdlib_path, "",
          "Path to the DSLX standard library. Defaults to the built-in path.");

namespace xls::spin {
namespace {

absl::Status Run() {
  std::string spin_path = absl::GetFlag(FLAGS_spin_trace);
  std::string dslx_path = absl::GetFlag(FLAGS_dslx_trace);
  if (spin_path.empty() || dslx_path.empty()) {
    return absl::InvalidArgumentError(
        "--spin_trace and --dslx_trace are required.");
  }

  LOG(INFO) << "Comparing SPIN trace '" << spin_path << "' against DSLX trace '"
            << dslx_path << "'";
  std::string spin_json, dslx_text;
  XLS_ASSIGN_OR_RETURN(spin_json, GetFileContents(spin_path));
  XLS_ASSIGN_OR_RETURN(dslx_text, GetFileContents(dslx_path));

  ProcInstPaths proc_paths;
  DslxChannelNameMap channel_name_map;

  std::string dslx_file_path = absl::GetFlag(FLAGS_dslx_file);
  if (dslx_file_path.empty()) {
    LOG(WARNING) << "No --dslx_file provided; channel keys will use bare "
                    "names without proc-hierarchy prefixes.";
  }
  if (!dslx_file_path.empty()) {
    std::string dslx_source;
    XLS_ASSIGN_OR_RETURN(dslx_source, GetFileContents(dslx_file_path));
    std::string stdlib_path = absl::GetFlag(FLAGS_dslx_stdlib_path);
    if (stdlib_path.empty()) {
      stdlib_path = std::string(xls::kDefaultDslxStdlibPath);
    }
    std::string module_name =
        std::filesystem::path(dslx_file_path).stem().string();
    dslx::ImportData import_data(dslx::CreateImportData(
        stdlib_path, {}, dslx::kAllWarningsSet,
        std::make_unique<dslx::RealFilesystem>()));
    XLS_ASSIGN_OR_RETURN(
        dslx::TypecheckedModule tm,
        dslx::ParseAndTypecheck(dslx_source, dslx_file_path, module_name,
                                &import_data));
    channel_name_map = BuildDslxChannelNameMap(*tm.module);

    std::vector<dslx::TestProc*> test_procs = tm.module->GetTestProcs();
    if (!test_procs.empty()) {
      std::string top = test_procs[0]->proc()->identifier();
      if (test_procs.size() > 1) {
        LOG(WARNING) << "Multiple test procs found; using '" << top
                     << "' for proc-hierarchy resolution";
      }
      bool printed_error = false;
      dslx::ConvertOptions convert_options = {
          .convert_tests = true,
          .lower_to_proc_scoped_channels = true,
      };
      std::array<std::string_view, 1> module_path{dslx_file_path};
      XLS_ASSIGN_OR_RETURN(
          dslx::PackageConversionData conv,
          dslx::ConvertFilesToPackage(module_path, stdlib_path, {},
                                      convert_options, top, module_name,
                                      &printed_error));
      XLS_ASSIGN_OR_RETURN(proc_paths, BuildProcInstPathsForSpin(conv.package.get()));
    }
  }

  const std::string terminator_channel =
      absl::GetFlag(FLAGS_terminator_channel);
  XLS_ASSIGN_OR_RETURN(TraceMap spin_events,
                       ParseSpinTrace(spin_json, proc_paths,
                                      terminator_channel));
  XLS_ASSIGN_OR_RETURN(TraceMap dslx_events,
                       ParseDslxTrace(dslx_text, terminator_channel,
                                      channel_name_map));
  absl::Status result = CompareTraces(spin_events, dslx_events);
  if (result.ok()) {
    LOG(INFO) << "Trace comparison passed";
  }
  return result;
}

}  // namespace
}  // namespace xls::spin

int main(int argc, char** argv) {
  xls::InitXls(argv[0], argc, argv);
  absl::Status status = xls::spin::Run();
  if (!status.ok()) {
    std::cerr << status.message() << "\n";
    return 1;
  }
  return 0;
}
