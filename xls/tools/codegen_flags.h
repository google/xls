// Copyright 2022 The XLS Authors
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

#ifndef XLS_TOOLS_CODEGEN_FLAGS_H_
#define XLS_TOOLS_CODEGEN_FLAGS_H_

#include <string>

#include "absl/flags/declare.h"
#include "absl/status/statusor.h"
#include "xls/tools/codegen_flags.pb.h"

ABSL_DECLARE_FLAG(std::string, output_verilog_path);
ABSL_DECLARE_FLAG(std::string, output_schedule_path);
ABSL_DECLARE_FLAG(std::string, output_schedule_ir_path);
ABSL_DECLARE_FLAG(std::string, output_block_ir_path);
ABSL_DECLARE_FLAG(std::string, output_signature_path);
ABSL_DECLARE_FLAG(std::string, output_verilog_line_map_path);
ABSL_DECLARE_FLAG(std::string, top);
ABSL_DECLARE_FLAG(std::optional<std::string>,
                  codegen_options_used_textproto_file);

namespace xls {

// Populates the codegen flags proto from the ABSL flags library values.
absl::StatusOr<CodegenFlagsProto> GetCodegenFlags();

}  // namespace xls

#endif  // XLS_TOOLS_CODEGEN_FLAGS_H_
