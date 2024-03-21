// Copyright 2020 Google LLC
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

#include "xls/synthesis/yosys/yosys_synthesis_service.h"

#include <cstdint>
#include <filesystem>  // NOLINT
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "grpcpp/server.h"
#include "grpcpp/server_context.h"
#include "grpcpp/support/status.h"
#include "xls/common/file/filesystem.h"
#include "xls/common/file/temp_directory.h"
#include "xls/common/logging/logging.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/common/subprocess.h"
#include "xls/synthesis/synthesis.pb.h"
#include "xls/synthesis/yosys/yosys_util.h"

namespace xls {
namespace synthesis {

::grpc::Status YosysSynthesisServiceImpl::Compile(
    ::grpc::ServerContext* server_context, const CompileRequest* request,
    CompileResponse* result) {
  auto start = absl::Now();

  absl::Status synthesis_status = RunSynthesis(request, result);
  if (!synthesis_status.ok()) {
    return ::grpc::Status(grpc::StatusCode::INTERNAL,
                          std::string(synthesis_status.message()));
  }

  result->set_elapsed_runtime_ms(
      absl::ToInt64Milliseconds(absl::Now() - start));

  return ::grpc::Status::OK;
}

// Run the given arguments as a subprocess with InvokeSubprocess.
// InvokeSubprocess is wrapped because the error message can be very large (it
// includes both stdout and stderr) which breaks propagation of the error via
// GRPC because GRPC instead gives an error about trailing metadata being too
// large.
absl::StatusOr<std::pair<std::string, std::string>>
YosysSynthesisServiceImpl::RunSubprocess(
    absl::Span<const std::string> args) const {
  absl::StatusOr<std::pair<std::string, std::string>> stdout_stderr_status =
      SubprocessResultToStrings(
          SubprocessErrorAsStatus(InvokeSubprocess(args)));
  if (!stdout_stderr_status.ok()) {
    LOG(ERROR) << stdout_stderr_status.status();
    const int64_t kMaxMessageSize = 1024;
    auto prune_error_message = [](std::string_view message) -> std::string {
      if (message.size() >= kMaxMessageSize) {
        return absl::StrFormat(
            "%s\n...\n%s", message.substr(0, kMaxMessageSize),
            message.substr(message.size() - kMaxMessageSize));
      }
      return std::string(message);
    };
    return absl::InternalError(absl::StrFormat(
        "Failed to execute subprocess: %s. Error: %s", absl::StrJoin(args, " "),
        prune_error_message(stdout_stderr_status.status().message())));
  }
  return stdout_stderr_status;
}

// Build yosys synthesis script contents for stdcell backend
std::string YosysSynthesisServiceImpl::BuildYosysTcl(
    const CompileRequest* request, const std::filesystem::path& verilog_path,
    const std::filesystem::path& json_path,
    const std::filesystem::path& netlist_path) const {
  std::vector<std::string> yosys_tcl_vec;
  std::string yosys_tcl;

  // Input in hz, adjust for scale ps
  double clock_period_ps =
      1e12 / static_cast<double>(request->target_frequency_hz());
  std::string delay_target = absl::StrCat(clock_period_ps);

  // Define yosys commands
  const std::string yosys_import = "yosys -import";
  const std::string show_dont_use =
      "yosys log\n"
      "yosys log -n {DONT_USE_ARGS: }\n"
      "yosys log {*}$::env(DONT_USE_ARGS)";
  const std::string read_verilog_rtl =
      absl::StrFormat("read_verilog %s", verilog_path.string());
  const std::string delete_print_cells =
      "delete {*/t:$print}";

  const std::string perform_generic_synthesis =
      absl::StrFormat("synth -top %s", request->top_module_name());
  const std::string perform_ff_mapping =
      absl::StrFormat("dfflibmap -liberty %s; opt ;", synthesis_libraries_);
  const std::string perform_abc_mapping = absl::StrFormat(
      "abc -D %s -liberty %s -showtmp -script "
      "\"+strash;fraig;scorr;retime,%s;strash;dch,-f;map,-M,1,%s;"
      "topo;stime;buffer;topo;stime;minsize;"
      "stime;upsize;stime;dnsize;stime\""
      " {*}$::env(DONT_USE_ARGS)",
      delay_target, synthesis_libraries_, delay_target, delay_target);
  const std::string perform_cleanup =
      "setundef -zero\n" "splitnets";
  const std::string perform_optimizations =
      "opt\n" "clean\n" "yosys rename -enumerate";

  const std::string write_json_netlist =
      absl::StrFormat("write_json %s", json_path.string());
  const std::string write_verilog_netlist =
      absl::StrFormat("write_verilog -noattr -noexpr -nohex -nodec %s",
                      netlist_path.string());

  // Build yosys commandfile
  yosys_tcl_vec.push_back(yosys_import);
  yosys_tcl_vec.push_back(show_dont_use);
  yosys_tcl_vec.push_back(read_verilog_rtl);
  yosys_tcl_vec.push_back(delete_print_cells);

  yosys_tcl_vec.push_back(perform_generic_synthesis);
  yosys_tcl_vec.push_back(perform_ff_mapping);
  yosys_tcl_vec.push_back(perform_abc_mapping);
  yosys_tcl_vec.push_back(perform_cleanup);
  yosys_tcl_vec.push_back(perform_optimizations);

  yosys_tcl_vec.push_back(write_json_netlist);
  yosys_tcl_vec.push_back(write_verilog_netlist);

  yosys_tcl = absl::StrJoin(yosys_tcl_vec, "\n");

  VLOG(1) << "about to start, yosys tcl: " << yosys_tcl;
  return yosys_tcl;
}

// Invokes yosys and nextpnr to synthesis the verilog given in the
// CompileRequest.
absl::Status YosysSynthesisServiceImpl::RunSynthesis(
    const CompileRequest* request, CompileResponse* result) const {

  // This tells the timing client that the achieved implementation freq
  //  doesn't depend on the requested frequency, which is true
  //  for Yosys with the current script.
  result->set_insensitive_to_target_freq(true);

  if (request->top_module_name().empty()) {
    return absl::InvalidArgumentError("Must specify top module name.");
  }

  XLS_ASSIGN_OR_RETURN(TempDirectory temp_dir, TempDirectory::Create());
  const std::filesystem::path temp_dir_path = temp_dir.path();
  if (save_temps_) {
    std::move(temp_dir).Release();
  }
  std::filesystem::path verilog_path = temp_dir_path / "input.v";
  XLS_RETURN_IF_ERROR(SetFileContents(verilog_path, request->module_text()));

  // Invoke yosys to generate netlist.
  std::filesystem::path synth_json_path = temp_dir_path / "netlist.json";
  std::filesystem::path synth_verilog_path = temp_dir_path / "output.v";
  std::pair<std::string, std::string> string_pair;

  if (!synthesis_target_.empty()) {
    // Yosys for Nextpnr backend
    std::string yosys_cmd =
        absl::StrFormat("synth_%s -top %s -json %s", synthesis_target_,
                        request->top_module_name(), synth_json_path.string());
    LOG(INFO) << "yosys cmd: " << yosys_cmd;
    XLS_ASSIGN_OR_RETURN(
        string_pair,
        RunSubprocess({yosys_path_, "-p", yosys_cmd, verilog_path.string()}));
  } else {
    // Yosys for stdcell backend
    std::filesystem::path yosys_tcl_path = temp_dir_path / "yosys.tcl";
    std::string yosys_tcl = BuildYosysTcl(request, verilog_path,
                                           synth_json_path, synth_verilog_path);
    XLS_RETURN_IF_ERROR(SetFileContents(yosys_tcl_path, yosys_tcl));
    LOG(INFO) << "Running Yosys:  command file: " << yosys_tcl_path;
    XLS_ASSIGN_OR_RETURN(string_pair,
                         RunSubprocess({yosys_path_, "-c", yosys_tcl_path}));
  }

  auto [yosys_stdout, yosys_stderr] = string_pair;
  if (save_temps_) {
    XLS_RETURN_IF_ERROR(
        SetFileContents(temp_dir_path / "yosys.stdout", yosys_stdout));
    XLS_RETURN_IF_ERROR(
        SetFileContents(temp_dir_path / "yosys.stderr", yosys_stderr));
  }
  XLS_ASSIGN_OR_RETURN(std::string netlist, GetFileContents(synth_json_path));
  if (return_netlist_) {
    result->set_netlist(netlist);
  }

  // Add stats in response.
  XLS_ASSIGN_OR_RETURN(YosysSynthesisStatistics parse_stats,
                       ParseYosysOutput(yosys_stdout));
  XLS_RET_CHECK(!result->has_instance_count());
  for (const auto& name_count : parse_stats.cell_histogram) {
    (*result->mutable_instance_count()
          ->mutable_cell_histogram())[name_count.first] = name_count.second;
  }

  // If only synthesis requested, done.
  if (synthesis_only_) {
    return absl::OkStatus();
  }

  // Run the desired backend
  if (!synthesis_target_.empty()) {
    return RunNextPNR(request, result, temp_dir_path, synth_json_path);
  }

  return RunSTA(request, result, temp_dir_path, synth_verilog_path);
}

absl::Status YosysSynthesisServiceImpl::RunNextPNR(
    const CompileRequest* request, CompileResponse* result,
    const std::filesystem::path& temp_dir_path,
    const std::filesystem::path& synth_json_path) const {
  // Invoke nextpnr for place and route.
  std::optional<std::filesystem::path> pnr_path;
  std::vector<std::string> nextpnr_args = {nextpnr_path_, "--json",
                                           synth_json_path.string()};

  if (synthesis_target_ == "ecp5") {
    nextpnr_args.push_back("--45k");
    nextpnr_args.push_back("--textcfg");
    pnr_path = temp_dir_path / "pnr.cfg";
    nextpnr_args.push_back(pnr_path->string());
  } else if (synthesis_target_ == "ice40") {
    nextpnr_args.push_back("--hx8k");
  }

  if (request->has_target_frequency_hz()) {
    nextpnr_args.push_back("--freq");
    nextpnr_args.push_back(
        absl::StrCat(request->target_frequency_hz() / 1000000));
  }
  std::pair<std::string, std::string> string_pair;
  XLS_ASSIGN_OR_RETURN(string_pair, RunSubprocess(nextpnr_args));
  auto [nextpnr_stdout, nextpnr_stderr] = string_pair;
  if (save_temps_) {
    XLS_RETURN_IF_ERROR(
        SetFileContents(temp_dir_path / "nextpnr.stdout", nextpnr_stdout));
    XLS_RETURN_IF_ERROR(
        SetFileContents(temp_dir_path / "nextpnr.stderr", nextpnr_stderr));
  }

  if (pnr_path.has_value()) {
    XLS_ASSIGN_OR_RETURN(std::string pnr_result, GetFileContents(*pnr_path));
    result->set_place_and_route_result(pnr_result);
  }

  // Parse the stderr from nextpnr to get the maximum frequency.
  XLS_ASSIGN_OR_RETURN(int64_t max_frequency_hz,
                       ParseNextpnrOutput(nextpnr_stderr));
  result->set_max_frequency_hz(max_frequency_hz);
  LOG(INFO) << "max_frequency_mhz: "
            << (static_cast<double>(max_frequency_hz) / 1e6);

  return absl::OkStatus();
}

std::string YosysSynthesisServiceImpl::BuildSTACmds(
    const CompileRequest* request,
    const std::filesystem::path& netlist_path) const {
  // Invoke STA for timing and max freq analysis.
  std::vector<std::string> sta_cmd_vec;
  std::string sta_cmd;

  // Input in hz, adjust for scale ps
  double clock_period_ps =
      1e12 / static_cast<double>(request->target_frequency_hz());
  std::string delay_target = absl::StrCat(clock_period_ps);

  const std::string setup_libraries =
      absl::StrFormat("set LIB_FILES { %s }", sta_libraries_);
  const std::string read_libraries =
      absl::StrFormat("foreach libFile $LIB_FILES { read_liberty $libFile }");
  const std::string read_verilog_netlist =
      absl::StrFormat("read_verilog %s ", netlist_path.string());
  const std::string perform_elaboratation =
      absl::StrFormat("link_design %s  ", request->top_module_name());

  const std::string setup_units = absl::StrFormat("set_cmd_units -time ps");
  const std::string setup_clk_period =
      absl::StrFormat("set clk_period %s", delay_target);
  const std::string setup_clk_port =
      absl::StrFormat("set clk_port [get_ports -quiet clk]");
  const std::string setup_delay = absl::StrFormat("set clk_io_pct 0.001");

  const std::string setup_constraints = absl::StrFormat(
      R"(if { [string length [get_ports -quiet clk]] > 1 } {
  create_clock -name op_clk  -period $clk_period  $clk_port ;
  set non_clock_inputs [lsearch -inline -all -not -exact [all_inputs] $clk_port ] ;
  set_input_delay  [expr $clk_period * $clk_io_pct] -clock op_clk $non_clock_inputs ;
  set_output_delay [expr $clk_period * $clk_io_pct] -clock op_clk [all_outputs] ;
 })");

  const std::string perform_report_clk =
      absl::StrFormat("report_clock_min_period");
  const std::string perform_worst_slack =
      absl::StrFormat("report_worst_slack -max");
  const std::string perform_report_negative_slacks =
      absl::StrFormat("report_tns report_wns");

  const std::string perform_report_checks = absl::StrFormat(
      "report_checks -path_delay min_max -fields {slew cap input nets"
      "fanout} -format full_clock_expanded");

  const std::string perform_exit = absl::StrFormat("exit");

  sta_cmd_vec.push_back(setup_libraries);
  sta_cmd_vec.push_back(read_libraries);
  sta_cmd_vec.push_back(read_verilog_netlist);
  sta_cmd_vec.push_back(perform_elaboratation);

  sta_cmd_vec.push_back(setup_units);
  sta_cmd_vec.push_back(setup_clk_period);
  sta_cmd_vec.push_back(setup_clk_port);
  sta_cmd_vec.push_back(setup_delay);

  sta_cmd_vec.push_back(setup_constraints);

  sta_cmd_vec.push_back(perform_report_clk);
  sta_cmd_vec.push_back(perform_worst_slack);
  sta_cmd_vec.push_back(perform_report_negative_slacks);
  sta_cmd_vec.push_back(perform_report_checks);

  sta_cmd_vec.push_back(perform_exit);

  sta_cmd = absl::StrJoin(sta_cmd_vec, "\n");

  VLOG(1) << "about to start, sta cmd: " << sta_cmd;
  return sta_cmd;
}

absl::Status YosysSynthesisServiceImpl::RunSTA(
    const CompileRequest* request, CompileResponse* result,
    const std::filesystem::path& temp_dir_path,
    const std::filesystem::path& netlist_path) const {
  std::string sta_cmd;
  std::filesystem::path sta_cmd_path = temp_dir_path / "sta.tcl";
  sta_cmd = BuildSTACmds(request, netlist_path);
  XLS_RETURN_IF_ERROR(SetFileContents(sta_cmd_path, sta_cmd));

  LOG(INFO) << "Running OpenSTA: command file: " << sta_cmd_path;

  std::pair<std::string, std::string> string_pair;

  XLS_ASSIGN_OR_RETURN(string_pair, RunSubprocess({sta_path_, "-no_splash",
                                                   "-exit", sta_cmd_path}));

  auto [sta_stdout, sta_stderr] = string_pair;
  if (save_temps_) {
    XLS_RETURN_IF_ERROR(
        SetFileContents(temp_dir_path / "sta.stdout", sta_stdout));
    XLS_RETURN_IF_ERROR(
        SetFileContents(temp_dir_path / "sta.stderr", sta_stderr));
  }

  XLS_ASSIGN_OR_RETURN(STAStatistics sta_stats, ParseOpenSTAOutput(sta_stdout));

  result->set_max_frequency_hz(sta_stats.max_frequency_hz);
  result->set_slack_ps(sta_stats.slack_ps);

  return absl::OkStatus();
}

}  // namespace synthesis
}  // namespace xls
