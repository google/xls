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

#ifndef XLS_SYNTHESIS_YOSYS_YOSYS_SYNTHESIS_SERVICE_H_
#define XLS_SYNTHESIS_YOSYS_YOSYS_SYNTHESIS_SERVICE_H_

#include <filesystem>  // NOLINT
#include <string>
#include <string_view>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "grpcpp/server.h"
#include "grpcpp/server_context.h"
#include "grpcpp/support/status.h"
#include "xls/synthesis/synthesis.pb.h"
#include "xls/synthesis/synthesis_service.grpc.pb.h"

namespace xls {
namespace synthesis {

class YosysSynthesisServiceImpl : public SynthesisService::Service {
 public:
  explicit YosysSynthesisServiceImpl(
      std::string_view yosys_path, std::string_view nextpnr_path,
      std::string_view synthesis_target, std::string_view sta_path,
      std::string_view synthesis_libraries, std::string_view sta_libraries,
      std::string_view default_driver_cell, std::string_view default_load,
      bool save_temps, bool return_netlist, bool synthesis_only)
      : yosys_path_(yosys_path),
        nextpnr_path_(nextpnr_path),
        synthesis_target_(synthesis_target),
        sta_path_(sta_path),
        synthesis_libraries_(synthesis_libraries),
        sta_libraries_(sta_libraries),
        default_driver_cell_(default_driver_cell),
        default_load_(default_load),
        save_temps_(save_temps),
        return_netlist_(return_netlist),
        synthesis_only_(synthesis_only) {}

  ::grpc::Status Compile(::grpc::ServerContext* server_context,
                         const CompileRequest* request,
                         CompileResponse* result) override;

  // Run the given arguments as a subprocess with InvokeSubprocess.
  // InvokeSubprocess is wrapped because the error message can be very large (it
  // includes both stdout and stderr) which breaks propagation of the error via
  // GRPC because GRPC instead gives an error about trailing metadata being too
  // large.
  absl::StatusOr<std::pair<std::string, std::string>> RunSubprocess(
      absl::Span<const std::string> args) const;

  // Build ABC constraints file contents for stdcell backend
  std::string BuildAbcConstraints(const CompileRequest* request) const;

  // Build yosys synthesis script contents for stdcell backend
  std::string BuildYosysTcl(const CompileRequest* request,
                            const std::filesystem::path& abc_constr_path,
                            const std::filesystem::path& verilog_path,
                            const std::filesystem::path& json_path,
                            const std::filesystem::path& netlist_path) const;

  // Invokes yosys and nextpnr to synthesis the verilog given in the
  // CompileRequest.
  absl::Status RunSynthesis(const CompileRequest* request,
                            CompileResponse* result) const;

  absl::Status RunNextPNR(const CompileRequest* request,
                          CompileResponse* result,
                          const std::filesystem::path& temp_dir_path,
                          const std::filesystem::path& synth_json_path) const;

  std::string BuildSTACmds(const CompileRequest* request,
                           const std::filesystem::path& netlist_path) const;

  absl::Status RunSTA(const CompileRequest* request, CompileResponse* result,
                      const std::filesystem::path& temp_dir_path,
                      const std::filesystem::path& netlist_path) const;

 private:
  std::string yosys_path_;
  std::string nextpnr_path_;
  std::string synthesis_target_;
  std::string sta_path_;
  std::string synthesis_libraries_;
  std::string sta_libraries_;
  std::string default_driver_cell_;
  std::string default_load_;
  bool save_temps_;
  bool return_netlist_;
  bool synthesis_only_;
};

}  // namespace synthesis
}  // namespace xls

#endif  // XLS_SYNTHESIS_YOSYS_YOSYS_SYNTHESIS_SERVICE_H_
