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

// Converts a protobuf schema and instantiating message into DSLX structs and
// constant data.
#include <string>

#include "absl/flags/flag.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "xls/common/exit_status.h"
#include "xls/common/file/filesystem.h"
#include "xls/common/init_xls.h"
#include "xls/common/status/status_macros.h"
#include "xls/dslx/frontend/module.h"
#include "xls/tools/proto_to_dslx.h"

ABSL_FLAG(std::string, proto_def_path, "",
          "Path to the [structure] definition of the proto "
          "(i.e., the '.proto' file) to parse.");
ABSL_FLAG(std::string, output_path, "", "Path to which to write the output.");
ABSL_FLAG(std::string, proto_name, "",
          "Fully-qualified name of the proto message (i.e., the schema) "
          "to parse.");
ABSL_FLAG(std::string, source_root_path, ".",
          "Path to the root of the source tree, i.e., the directory in which "
          "xls can be found. Defaults to ${CWD}. "
          "(Needed for locating transitive proto dependencies.)");
ABSL_FLAG(std::string, textproto_path, "",
          "Path to the textproto to translate into DSLX.");
ABSL_FLAG(std::string, var_name, "",
          "The name of the DSLX variable to instantiate.");

namespace xls {

static absl::Status RealMain(const std::string& source_root_path,
                             const std::string& proto_def_path,
                             const std::string& proto_name,
                             const std::string& textproto_path,
                             const std::string& var_name,
                             const std::string& output_path) {
  XLS_ASSIGN_OR_RETURN(std::string textproto, GetFileContents(textproto_path));
  dslx::FileTable file_table;
  XLS_ASSIGN_OR_RETURN(auto module,
                       ProtoToDslx(source_root_path, proto_def_path, proto_name,
                                   textproto, var_name, file_table));
  return SetFileContents(output_path, module->ToString());
}

}  // namespace xls

int main(int argc, char* argv[]) {
  xls::InitXls(argv[0], argc, argv);

  std::string proto_def_path = absl::GetFlag(FLAGS_proto_def_path);
  QCHECK(!proto_def_path.empty()) << "--proto_def_path must be specified.";

  std::string source_root_path = absl::GetFlag(FLAGS_source_root_path);
  QCHECK(!source_root_path.empty()) << "--source_root_path must be specified.";

  std::string output_path = absl::GetFlag(FLAGS_output_path);
  QCHECK(!output_path.empty()) << "--output_path must be specified.";

  std::string proto_name = absl::GetFlag(FLAGS_proto_name);
  QCHECK(!proto_name.empty()) << "--proto_name must be specified.";

  std::string textproto_path = absl::GetFlag(FLAGS_textproto_path);
  QCHECK(!textproto_path.empty()) << "--textproto_path must be specified.";

  std::string var_name = absl::GetFlag(FLAGS_var_name);
  QCHECK(!var_name.empty()) << "--var_name must be specified.";
  return xls::ExitStatus(xls::RealMain(source_root_path, proto_def_path,
                                       proto_name, textproto_path, var_name,
                                       output_path));
}
