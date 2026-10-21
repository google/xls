// Copyright 2026 The XLS Authors
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

#include "xls/tools/test_utils/test_codegen_flag_handler.h"

#include <memory>
#include <string>
#include <utility>

#include "absl/flags/flag.h"
#include "absl/status/status.h"
#include "xls/codegen/codegen_options.h"
#include "xls/tools/codegen_flags.pb.h"
#include "xls/tools/codegen_flags_handler_registry.h"
#include "xls/tools/test_utils/test_codegen_flags.pb.h"

ABSL_FLAG(std::string, test_codegen_flag, "default_val",
          "Test codegen flag for CodegenFlagsHandlerRegistry.");

namespace xls {
namespace {

absl::Status TestFlagHandler(const CodegenFlagsProto& proto,
                             verilog::CodegenOptions& options) {
  if (proto.HasExtension(test_codegen_flag)) {
    std::string val = proto.GetExtension(test_codegen_flag);
    if (val == "invalid") {
      return absl::InvalidArgumentError("Invalid test_codegen_flag value");
    }
    options.add_extension(
        std::make_unique<TestCodegenOptionExtension>(std::move(val)));
  }
  return absl::OkStatus();
}

absl::Status TestFlagParser(CodegenFlagsProto& proto) {
  std::string val = absl::GetFlag(FLAGS_test_codegen_flag);
  if (val == "parse_error") {
    return absl::InvalidArgumentError("Failed to parse test_codegen_flag");
  }
  if (val != "default_val") {
    proto.SetExtension(test_codegen_flag, std::move(val));
  }
  return absl::OkStatus();
}

XLS_REGISTER_CODEGEN_FLAGS_HANDLER(test_codegen_flag, TestFlagHandler,
                                   TestFlagParser);

}  // namespace
}  // namespace xls
