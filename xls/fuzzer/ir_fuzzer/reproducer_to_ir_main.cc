// Copyright 2025 The XLS Authors
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
#include <cstdint>
#include <memory>
#include <optional>
#include <string_view>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_join.h"
#include "xls/common/exit_status.h"
#include "xls/common/file/filesystem.h"
#include "xls/common/init_xls.h"
#include "xls/common/status/status_macros.h"
#include "xls/fuzzer/ir_fuzzer/fuzzer_arg_list.pb.h"
#include "xls/fuzzer/ir_fuzzer/reproducer_repo_files.h"
#include "xls/fuzzer/ir_fuzzer/reproducer_to_ir.h"
#include "xls/ir/function.h"
#include "xls/ir/value.h"
#include "xls/tests/testvector.pb.h"

constexpr static std::string_view kUsage = R"(
Parse a fuzztest reproducer which is only a IrFuzzDomain() or
IrFuzzDomainWithArgs(<fuzztest_args>) and print the package.
)";

ABSL_FLAG(std::optional<int64_t>, fuzztest_args, 10,
          "How many args the domain has");
ABSL_FLAG(std::string, ir_out, "/dev/stdout", "file to dump the ir to.");
ABSL_FLAG(std::string, args_out, "/dev/null",
          "file to dump the args as binary proto to.");
ABSL_FLAG(std::string, args_textproto_out, "/dev/null",
          "file to dump the args as text proto to.");
ABSL_FLAG(std::string, args_testvector_out, "/dev/null",
          "file to dump the args as testvector textproto to.");

namespace xls {
namespace {

absl::Status RealMain(std::string_view file_v) {
  std::string file = std::string(file_v);
  if (IsFuzztestReproPath(file)) {
    XLS_ASSIGN_OR_RETURN(
        file, FuzztestRepoToFilePath(file),
        _ << "Unable to find file for fuzztest repo target: " << file);
  }
  XLS_ASSIGN_OR_RETURN(std::string repro, GetFileContents(file), _ << file);
  XLS_ASSIGN_OR_RETURN(
      auto pkg, FuzzerReproToIr(repro, absl::GetFlag(FLAGS_fuzztest_args)));
  Function* the_function = pkg->functions().front().get();
  XLS_RETURN_IF_ERROR(pkg->SetTop(the_function));
  XLS_ASSIGN_OR_RETURN(
      auto args,
      FuzzerReproToValues(repro, absl::GetFlag(FLAGS_fuzztest_args)));
  XLS_RETURN_IF_ERROR(
      SetFileContents(absl::GetFlag(FLAGS_ir_out), pkg->DumpIr()));
  FuzzerArgListProto proto;
  testvector::SampleInputsProto testvector;
  for (const auto& lst : args) {
    auto* proto_list = proto.add_set();
    for (const Value& v : lst) {
      XLS_ASSIGN_OR_RETURN(*proto_list->add_arg(), v.AsProto());
    }
    testvector.mutable_function_args()->add_args(absl::StrJoin(
        lst, "; ",
        [](std::string* out, const Value& v) { out->append(v.ToString()); }));
  }
  XLS_RETURN_IF_ERROR(
      SetTextProtoFile(absl::GetFlag(FLAGS_args_textproto_out), proto));
  XLS_RETURN_IF_ERROR(SetFileContents(absl::GetFlag(FLAGS_args_out),
                                      proto.SerializeAsString()));
  XLS_RETURN_IF_ERROR(
      SetTextProtoFile(absl::GetFlag(FLAGS_args_testvector_out), testvector));
  return absl::OkStatus();
}

}  // namespace
}  // namespace xls

int main(int argc, char** argv) {
  std::vector<std::string_view> positional_arguments =
      xls::InitXls(kUsage, argc, argv);

  if (positional_arguments.size() != 1) {
    LOG(QFATAL) << "Expected invocation: " << argv[0] << " <repro_file>";
  }

  return xls::ExitStatus(xls::RealMain(positional_arguments[0]));
}
