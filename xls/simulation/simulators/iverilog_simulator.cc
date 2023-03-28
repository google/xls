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
#include "xls/common/file/filesystem.h"
#include "xls/common/file/get_runfile_path.h"
#include "xls/common/file/temp_directory.h"
#include "xls/common/file/temp_file.h"
#include "xls/common/module_initializer.h"
#include "xls/common/status/status_macros.h"
#include "xls/common/subprocess.h"
#include "xls/simulation/verilog_simulator.h"
#include "xls/tools/verilog_include.h"

namespace xls {
namespace verilog {
namespace {

static absl::Status SetUpIncludes(const std::filesystem::path& temp_dir,
                                  absl::Span<const VerilogInclude> includes) {
  for (const VerilogInclude& include : includes) {
    std::filesystem::path path = temp_dir / include.relative_path;
    XLS_RETURN_IF_ERROR(RecursivelyCreateDir(path.parent_path()));
    XLS_RETURN_IF_ERROR(SetFileContents(path, include.verilog_text));
  }
  return absl::OkStatus();
}

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
  return SubprocessResultToStrings(
      SubprocessErrorAsStatus(InvokeSubprocess(args_vec)));
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
  return SubprocessResultToStrings(
      SubprocessErrorAsStatus(InvokeSubprocess(args_vec)));
}

class IcarusVerilogSimulator : public VerilogSimulator {
 public:
  absl::StatusOr<std::pair<std::string, std::string>> Run(
      std::string_view text, FileType file_type,
      absl::Span<const VerilogInclude> includes) const override {
    if (file_type == FileType::kSystemVerilog) {
      return absl::UnimplementedError(
          "iverilog does not support SystemVerilog");
    }
    XLS_ASSIGN_OR_RETURN(TempDirectory temp_top, TempDirectory::Create());
    XLS_RETURN_IF_ERROR(RecursivelyCreateDir(temp_top.path()));
    std::filesystem::path temp_dir = temp_top.path();

    std::string top_v_path = temp_dir / GetTopFileName(file_type);
    XLS_RETURN_IF_ERROR(SetFileContents(top_v_path, text));

    XLS_ASSIGN_OR_RETURN(TempFile temp_out, TempFile::Create(".out"));

    XLS_CHECK_OK(SetUpIncludes(temp_dir, includes));
    XLS_RETURN_IF_ERROR(
        InvokeIverilog({top_v_path, "-o", temp_out.path().string(), "-I",
                        temp_dir.string()})
            .status());

    return InvokeVvp({temp_out.path().string()});
  }

  absl::Status RunSyntaxChecking(
      std::string_view text, FileType file_type,
      absl::Span<const VerilogInclude> includes) const override {
    if (file_type == FileType::kSystemVerilog) {
      return absl::UnimplementedError(
          "iverilog does not support SystemVerilog");
    }
    XLS_ASSIGN_OR_RETURN(TempDirectory temp_top, TempDirectory::Create());
    XLS_RETURN_IF_ERROR(RecursivelyCreateDir(temp_top.path()));
    std::filesystem::path temp_dir = temp_top.path();

    std::string top_v_path = temp_dir / GetTopFileName(file_type);
    XLS_RETURN_IF_ERROR(SetFileContents(top_v_path, text));

    XLS_ASSIGN_OR_RETURN(TempFile temp_out, TempFile::Create(".out"));

    XLS_CHECK_OK(SetUpIncludes(temp_dir, includes));
    XLS_RETURN_IF_ERROR(
        InvokeIverilog({top_v_path, "-o", temp_out.path().string(), "-I",
                        temp_dir.string()})
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
