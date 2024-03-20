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

#include <cmath>
#include <cstdlib>
#include <memory>
#include <string>
#include <string_view>
#include <utility>

#include "absl/flags/flag.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/time/time.h"
#include "grpcpp/grpcpp.h"
#include "grpcpp/security/server_credentials.h"
#include "grpcpp/server.h"
#include "grpcpp/server_builder.h"
#include "grpcpp/server_context.h"
#include "libs/json11/json11.hpp"
#include "xls/common/file/filesystem.h"
#include "xls/common/file/temp_directory.h"
#include "xls/common/init_xls.h"
#include "xls/common/logging/logging.h"
#include "xls/common/status/status_macros.h"
#include "xls/synthesis/credentials.h"
#include "xls/synthesis/synthesis.pb.h"
#include "xls/synthesis/synthesis_service.grpc.pb.h"

const char kUsage[] = R"(
Launches a XLS synthesis server which generates OpenROAD metrics JSON via a
provided command and parses that JSON to create synthesis responses.

Invocation:

  json_metrics_server_main --metrics_command="bazel run :dummy_metrics_main"
)";

ABSL_FLAG(int32_t, port, 10000, "Port to listen on.");
ABSL_FLAG(std::string, metrics_command, "bazel run :dummy_metrics_main",
          "Command to run to generate JSON synthesis metrics.");
ABSL_FLAG(bool, save_temps, false, "Do not delete temporary files.");

namespace xls {
namespace synthesis {
namespace {

// Service implementation that dispatches compile requests.
class JsonMetricsSynthesisServiceImpl : public SynthesisService::Service {
 public:
  explicit JsonMetricsSynthesisServiceImpl(std::string_view metrics_command)
      : metrics_command_(metrics_command) {}

  ::grpc::Status Compile(::grpc::ServerContext* server_context,
                         const CompileRequest* request,
                         CompileResponse* result) override {
    auto start = absl::Now();

    absl::Status metrics_status = RunMetrics(request, result);
    if (!metrics_status.ok()) {
      return ::grpc::Status(grpc::StatusCode::INTERNAL,
                            std::string(metrics_status.message()));
    }

    result->set_elapsed_runtime_ms(
        absl::ToInt64Milliseconds(absl::Now() - start));

    return ::grpc::Status::OK;
  }

  absl::Status RunMetrics(const CompileRequest* request,
                          CompileResponse* result) {
    XLS_ASSIGN_OR_RETURN(TempDirectory temp_dir, TempDirectory::Create());

    std::filesystem::path temp_dir_path = temp_dir.path();
    if (absl::GetFlag(FLAGS_save_temps)) {
      std::move(temp_dir).Release();
    }
    std::filesystem::path verilog_path = temp_dir_path / "input.v";
    XLS_RETURN_IF_ERROR(SetFileContents(verilog_path, request->module_text()));

    double clock_period_ps = 1e12 / request->target_frequency_hz();
    setenv("CONSTANT_CLOCK_PORT", "clk", 1);
    setenv("CONSTANT_CLOCK_PERIOD_PS", absl::StrCat(clock_period_ps).c_str(),
           1);
    setenv("CONSTANT_TOP", request->top_module_name().c_str(), 1);
    setenv("INPUT_RTL", verilog_path.c_str(), 1);

    std::filesystem::path netlist_path = temp_dir_path / "netlist.v";
    setenv("OUTPUT_NETLIST", netlist_path.c_str(), 1);

    std::filesystem::path metrics_path = temp_dir_path / "metrics.json";
    setenv("OUTPUT_METRICS", metrics_path.c_str(), 1);

    if (EXIT_SUCCESS != system(metrics_command_.c_str())) {
      return absl::InternalError(absl::StrCat(
          "Metrics command \"", metrics_command_, "\" execution failed"));
    }

    XLS_ASSIGN_OR_RETURN(std::string metrics_str,
                         GetFileContents(metrics_path));

    std::string err;
    json11::Json metrics_json = json11::Json::parse(metrics_str, err);

    if (metrics_json.is_null()) {
      return absl::InternalError(
          absl::StrCat("Metrics json parsing failed with: ", err));
    }

    json11::Json slack_ps_json = metrics_json["slack_ps"];
    if (slack_ps_json.is_null()) {
      return absl::InternalError(
          absl::StrCat(metrics_str, " does not include slack_ps."));
    }

    double slack_ps_value = 0;
    // Use string_value here because the OpenROAD Tcl metrics logging logs
    // numbers as strings.
    if (!absl::SimpleAtod(slack_ps_json.string_value(), &slack_ps_value)) {
      return absl::InternalError(absl::StrCat(
          "slack_ps value ", slack_ps_json.dump(), " is not a double."));
    }

    result->set_slack_ps(std::lround(slack_ps_value));

    return absl::OkStatus();
  }

 private:
  std::string metrics_command_;
};

void RealMain() {
  int port = absl::GetFlag(FLAGS_port);
  std::string server_address = absl::StrCat("0.0.0.0:", port);
  std::string metrics_command = absl::GetFlag(FLAGS_metrics_command);
  JsonMetricsSynthesisServiceImpl service(absl::GetFlag(FLAGS_metrics_command));

  ::grpc::ServerBuilder builder;
  std::shared_ptr<::grpc::ServerCredentials> creds = GetServerCredentials();
  builder.AddListeningPort(server_address, creds);
  builder.RegisterService(&service);
  std::unique_ptr<::grpc::Server> server(builder.BuildAndStart());
  LOG(INFO) << "Serving on port: " << port;
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
