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

// Driver function for JIT wrapper generator.

#include <filesystem>

#include "absl/flags/flag.h"
#include "absl/status/status.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"
#include "absl/strings/strip.h"
#include "xls/common/file/filesystem.h"
#include "xls/common/init_xls.h"
#include "xls/common/logging/logging.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/ir_parser.h"
#include "xls/ir/jit_wrapper_generator.h"

ABSL_FLAG(std::string, class_name, "",
          "Name of the generated class. "
          "If unspecified, the camelized wrapped function name will be used.");
ABSL_FLAG(std::string, function, "",
          "Function to wrap. "
          "If unspecified, the package entry function will be used - "
          "in that case, the package-scoping mangling will be removed.");
ABSL_FLAG(std::string, ir_path, "", "Path to the IR to wrap.");
ABSL_FLAG(std::string, output_name, "",
          "Name of the generated files, foo.h and foo.c. "
          "If unspecified, the wrapped function name will be used.");
ABSL_FLAG(std::string, output_dir, "",
          "Directory into which to write the output. "
          "Files will be named <function>.h and <function>.cc");

namespace xls {
namespace {

std::string Camelize(absl::string_view input) {
  std::vector<std::string> pieces =
      absl::StrSplit(input, absl::ByAnyChar("-_"));
  for (std::string& piece : pieces) {
    piece[0] = toupper(piece[0]);
  }
  return absl::StrJoin(pieces, "");
}

}  // namespace

absl::Status RealMain(const std::filesystem::path& ir_path,
                      const std::filesystem::path& output_path,
                      std::string class_name, std::string output_name,
                      std::string function_name) {
  XLS_ASSIGN_OR_RETURN(std::string ir_text, GetFileContents(ir_path));
  XLS_ASSIGN_OR_RETURN(auto package, Parser::ParsePackage(ir_text));

  Function* function;
  std::string package_prefix = absl::StrCat("__", package->name(), "__");
  if (function_name.empty()) {
    XLS_ASSIGN_OR_RETURN(function, package->EntryFunction());
    function_name = absl::StripPrefix(function->name(), package_prefix);
  } else {
    // Apply the package prefix if not already there.
    if (!absl::StartsWith(function_name, package_prefix)) {
      function_name = absl::StrCat(package_prefix, function_name);
    }
    XLS_ASSIGN_OR_RETURN(function, package->GetFunction(function_name));
    function_name = absl::StripPrefix(function_name, package_prefix);
  }

  if (class_name.empty()) {
    class_name = function_name;
  }
  class_name = Camelize(class_name);

  std::filesystem::path header_path = output_path;
  if (output_name.empty()) {
    output_name = function_name;
  }
  header_path.append(absl::StrCat(output_name, ".h"));
  GeneratedJitWrapper wrapper =
      GenerateJitWrapper(*function, class_name, header_path);

  XLS_RETURN_IF_ERROR(SetFileContents(header_path, wrapper.header));

  std::filesystem::path source_path = output_path;
  source_path.append(absl::StrCat(output_name, ".cc"));
  XLS_RETURN_IF_ERROR(SetFileContents(source_path, wrapper.source));

  return absl::OkStatus();
}

}  // namespace xls

int main(int argc, char* argv[]) {
  xls::InitXls(argv[0], argc, argv);

  std::string ir_path = absl::GetFlag(FLAGS_ir_path);
  XLS_QCHECK(!ir_path.empty()) << "-ir_path must be specified!";

  std::string output_dir = absl::GetFlag(FLAGS_output_dir);
  XLS_QCHECK(!output_dir.empty()) << "-output_dir must be specified!";

  XLS_QCHECK_OK(xls::RealMain(
      ir_path, output_dir, absl::GetFlag(FLAGS_class_name),
      absl::GetFlag(FLAGS_output_name), absl::GetFlag(FLAGS_function)));

  return 0;
}
