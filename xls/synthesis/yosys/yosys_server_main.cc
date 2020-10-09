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

#include "grpcpp/grpcpp.h"
#include "grpcpp/security/server_credentials.h"
#include "grpcpp/server.h"
#include "grpcpp/server_builder.h"
#include "grpcpp/server_context.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/time/time.h"
#include "xls/common/file/filesystem.h"
#include "xls/common/file/temp_directory.h"
#include "xls/common/file/temp_file.h"
#include "xls/common/init_xls.h"
#include "xls/common/logging/logging.h"
#include "xls/common/status/status_macros.h"
#include "xls/common/subprocess.h"
#include "xls/synthesis/server_credentials.h"
#include "xls/synthesis/synthesis.pb.h"
#include "xls/synthesis/synthesis_service.grpc.pb.h"
#include "xls/synthesis/yosys/yosys_util.h"

const char kUsage[] =
    R"( Launches a XLS synthesis server which uses yosys and nextpnr for synthesis and
place-and-route.

Invocation:

  yosys_server --yosys_path=PATH --nextpnr_path=PATH
)";

ABSL_FLAG(int32, port, 10000, "Port to listen on.");
ABSL_FLAG(std::string, yosys_path, "", "The path to the yosys binary.");
ABSL_FLAG(std::string, nextpnr_path, "", "The path to the nextpnr binary.");
ABSL_FLAG(bool, save_temps, false, "Do not delete temporary files.");

namespace xls {
namespace synthesis {
namespace {

class YosysSynthesisServiceImpl : public SynthesisService::Service {
 public:
  explicit YosysSynthesisServiceImpl(absl::string_view yosys_path,
                                     absl::string_view nextpnr_path)
      : yosys_path_(yosys_path), nextpnr_path_(nextpnr_path) {}

  ::grpc::Status Compile(::grpc::ServerContext* server_context,
                         const CompileRequest* request,
                         CompileResponse* result) override {
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
  absl::StatusOr<std::pair<std::string, std::string>> RunSubprocess(
      absl::Span<const std::string> args) {
    absl::StatusOr<std::pair<std::string, std::string>> stdout_stderr_status =
        InvokeSubprocess(args);
    if (!stdout_stderr_status.ok()) {
      XLS_LOG(ERROR) << stdout_stderr_status.status();
      const int64 kMaxMessageSize = 1024;
      auto prune_error_message = [](absl::string_view message) -> std::string {
        if (message.size() >= kMaxMessageSize) {
          return absl::StrFormat(
              "%s\n...\n%s", message.substr(0, kMaxMessageSize),
              message.substr(message.size() - kMaxMessageSize));
        }
        return std::string(message);
      };
      return absl::InternalError(absl::StrFormat(
          "Failed to execute subprocess: %s. Error: %s",
          absl::StrJoin(args, " "),
          prune_error_message(stdout_stderr_status.status().message())));
    }
    return stdout_stderr_status;
  }

  // Invokes yosys and nextpnr to synthesis the verilog given in the
  // CompileRequest.
  absl::Status RunSynthesis(const CompileRequest* request,
                            CompileResponse* result) {
    if (request->top_module_name().empty()) {
      return absl::InvalidArgumentError("Must specify top module name.");
    }

    XLS_ASSIGN_OR_RETURN(TempDirectory temp_dir, TempDirectory::Create());
    std::filesystem::path temp_dir_path = temp_dir.path();
    if (absl::GetFlag(FLAGS_save_temps)) {
      std::move(temp_dir).Release();
    }
    std::filesystem::path verilog_path = temp_dir_path / "input.v";
    XLS_RETURN_IF_ERROR(SetFileContents(verilog_path, request->module_text()));

    // Invoke yosys to generate netlist.
    // TODO(meheff): Allow selecting synthesis targets (e.g., ecp5, ice40,
    // etc.).
    std::filesystem::path netlist_path = temp_dir_path / "netlist.json";
    std::pair<std::string, std::string> string_pair;
    XLS_ASSIGN_OR_RETURN(
        string_pair,
        RunSubprocess(
            {yosys_path_, "-p",
             absl::StrFormat("synth_ecp5 -top %s -json %s",
                             request->top_module_name(), netlist_path.string()),
             verilog_path.string()}));
    auto [yosys_stdout, yosys_stderr] = string_pair;
    if (absl::GetFlag(FLAGS_save_temps)) {
      XLS_RETURN_IF_ERROR(
          SetFileContents(temp_dir_path / "yosys.stdout", yosys_stdout));
      XLS_RETURN_IF_ERROR(
          SetFileContents(temp_dir_path / "yosys.stderr", yosys_stderr));
    }
    XLS_ASSIGN_OR_RETURN(std::string netlist, GetFileContents(netlist_path));
    result->set_netlist(netlist);

    // Invoke nextpnr for place and route.
    // TODO(meheff): Allow selecting different targets.
    std::filesystem::path pnr_path = temp_dir_path / "pnr.cfg";
    std::vector<std::string> nextpnr_args = {
        nextpnr_path_,         "--45k",     "--json",
        netlist_path.string(), "--textcfg", pnr_path.string()};
    if (request->has_target_frequency_hz()) {
      nextpnr_args.push_back("--freq");
      nextpnr_args.push_back(
          absl::StrCat(request->has_target_frequency_hz() / 1000000));
    }
    XLS_ASSIGN_OR_RETURN(string_pair, RunSubprocess(nextpnr_args));
    auto [nextpnr_stdout, nextpnr_stderr] = string_pair;
    if (absl::GetFlag(FLAGS_save_temps)) {
      XLS_RETURN_IF_ERROR(
          SetFileContents(temp_dir_path / "nextpnr.stdout", nextpnr_stdout));
      XLS_RETURN_IF_ERROR(
          SetFileContents(temp_dir_path / "nextpnr.stderr", nextpnr_stderr));
    }

    XLS_ASSIGN_OR_RETURN(std::string pnr_result, GetFileContents(pnr_path));
    result->set_place_and_route_result(pnr_result);

    // Parse the stderr from nextpnr to get the maximum frequency.
    XLS_ASSIGN_OR_RETURN(int64 max_frequency_hz,
                         ParseNextpnrOutput(nextpnr_stderr));
    result->set_max_frequency_hz(max_frequency_hz);

    return absl::OkStatus();
  }

 private:
  std::string yosys_path_;
  std::string nextpnr_path_;
};

void RealMain() {
  int port = absl::GetFlag(FLAGS_port);
  std::string server_address = absl::StrCat("0.0.0.0:", port);
  std::string yosys_path = absl::GetFlag(FLAGS_yosys_path);
  XLS_QCHECK_OK(FileExists(yosys_path));
  std::string nextpnr_path = absl::GetFlag(FLAGS_nextpnr_path);
  XLS_QCHECK_OK(FileExists(nextpnr_path));
  YosysSynthesisServiceImpl service(yosys_path, nextpnr_path);

  ::grpc::ServerBuilder builder;
  std::shared_ptr<::grpc::ServerCredentials> creds = GetServerCredentials();
  builder.AddListeningPort(server_address, creds);
  builder.RegisterService(&service);
  std::unique_ptr<::grpc::Server> server(builder.BuildAndStart());
  XLS_LOG(INFO) << "Serving on port: " << port;
  server->Wait();
}

}  // namespace
}  // namespace synthesis
}  // namespace xls

int main(int argc, char** argv) {
  xls::InitXls(kUsage, argc, argv);

  xls::synthesis::RealMain();

  return EXIT_SUCCESS;
}
