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

#include <cstdint>
#include <iostream>
#include <string>
#include <string_view>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "google/protobuf/text_format.h"
#include "xls/common/exit_status.h"
#include "xls/common/file/filesystem.h"
#include "xls/common/init_xls.h"
#include "xls/common/logging/logging.h"
#include "xls/synthesis/synthesis.pb.h"
#include "xls/synthesis/synthesis_client.h"

ABSL_FLAG(std::string, server, "localhost", "Server to connect to");
ABSL_FLAG(int, port, 10000, "Server port to connect to");
ABSL_FLAG(double, ghz, 1.0, "The target frequency for synthesis (GHz)");
ABSL_FLAG(std::string, top, "main", "Name of the top module to synthesize");

static constexpr char kUsage[] = R"(
A test client in C++ for using the synthesis server.
The default port matches that used by the Yosys synth server.

  synthesis_client_main \
       [--ghz=1.0] \
       [--port=10000] \
       [--server=localhost] \
       [--top="main"] \
       <path_to_verilog>
)";

int main(int argc, char** argv) {
  std::vector<std::string_view> positional_arguments =
      xls::InitXls(kUsage, argc, argv);

  const std::string server =
      absl::StrCat(absl::GetFlag(FLAGS_server), ":", absl::GetFlag(FLAGS_port));

  // Data we are sending to the server.
  xls::synthesis::CompileRequest request;
  request.set_top_module_name(absl::GetFlag(FLAGS_top));
  request.set_target_frequency_hz(
      static_cast<int64_t>(absl::GetFlag(FLAGS_ghz) * 1e9));

  // Check that input Verilog is provided, and get it
  if (positional_arguments.size() != 1) {
    XLS_LOG(QFATAL) << absl::StrCat("Expected invocation: ", argv[0],
                                    " [flags] VERILOG_FILE\n");
  }
  std::string_view vpath = positional_arguments[0];
  absl::StatusOr<std::string> verilog_contents = xls::GetFileContents(vpath);
  QCHECK_OK(verilog_contents.status());
  request.set_module_text(verilog_contents.value());

  // Use the client to perform the RPC
  absl::StatusOr<xls::synthesis::CompileResponse> compile_response_status =
      xls::synthesis::SynthesizeViaClient(server, request);

  // Examine the response
  if (compile_response_status.ok()) {
    std::string compile_response_text;
    google::protobuf::TextFormat::PrintToString(compile_response_status.value(),
                                      &compile_response_text);
    std::cout << compile_response_text << '\n';
  }
  return xls::ExitStatus(compile_response_status.status());
}
