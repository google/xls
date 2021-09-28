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

#include <string>
#include <utility>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/strings/strip.h"
#include "xls/common/file/get_runfile_path.h"
#include "xls/common/file/temp_file.h"
#include "xls/common/module_initializer.h"
#include "xls/common/status/status_macros.h"
#include "xls/common/subprocess.h"
#include "xls/simulation/verilog_simulator.h"
#include "xls/tools/verilog_include.h"

namespace xls {
namespace verilog {
namespace {

absl::StatusOr<std::pair<std::string, std::string>> InvokeIverilog(
    absl::Span<const std::string> args) {
  std::vector<std::string> args_vec;
  XLS_ASSIGN_OR_RETURN(std::filesystem::path iverilog_path,
                       GetXlsRunfilePath("external/com_icarus_iverilog/iverilog-bin"));
  args_vec.push_back(iverilog_path.string());
  args_vec.push_back("-B");
  args_vec.push_back(
      std::string(absl::StripSuffix(iverilog_path.string(), "iverilog-bin")));
  args_vec.insert(args_vec.end(), args.begin(), args.end());
  return InvokeSubprocess(args_vec);
}

absl::StatusOr<std::pair<std::string, std::string>> InvokeVvp(
    absl::Span<const std::string> args) {
  std::vector<std::string> args_vec;
  XLS_ASSIGN_OR_RETURN(std::filesystem::path iverilog_path,
                       GetXlsRunfilePath("external/com_icarus_iverilog/vvp-bin"));
  args_vec.push_back(iverilog_path.string());
  args_vec.push_back("-M");
  args_vec.push_back(
      std::string(absl::StripSuffix(iverilog_path.string(), "vvp-bin")));
  args_vec.insert(args_vec.end(), args.begin(), args.end());
  return InvokeSubprocess(args_vec);
}

class IcarusVerilogSimulator : public VerilogSimulator {
 public:
  absl::StatusOr<std::pair<std::string, std::string>> Run(
      absl::string_view text,
      absl::Span<const VerilogInclude> includes) const override {
    XLS_ASSIGN_OR_RETURN(TempFile temp, TempFile::CreateWithContent(text));
    XLS_ASSIGN_OR_RETURN(TempFile temp_out, TempFile::Create());
    XLS_RETURN_IF_ERROR(
        InvokeIverilog({temp.path().string(), "-o", temp_out.path().string()})
            .status());

    return InvokeVvp({temp_out.path().string()});
  }

  absl::Status RunSyntaxChecking(
      absl::string_view text,
      absl::Span<const VerilogInclude> includes) const override {
    XLS_ASSIGN_OR_RETURN(TempFile temp, TempFile::CreateWithContent(text));
    XLS_ASSIGN_OR_RETURN(TempFile temp_out, TempFile::Create());

    XLS_RETURN_IF_ERROR(
        InvokeIverilog({temp.path().string(), "-o", temp_out.path().string()})
            .status());

    return absl::OkStatus();
  }
};

XLS_REGISTER_MODULE_INITIALIZER(iverilog_simulator, {
  XLS_CHECK_OK(GetVerilogSimulatorManagerSingleton().RegisterVerilogSimulator(
      "iverilog", std::make_unique<IcarusVerilogSimulator>()));
});

}  // namespace
}  // namespace verilog
}  // namespace xls
