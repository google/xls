// Copyright 2021 The XLS Authors
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

#ifndef XLS_CODEGEN_BLOCK_GENERATOR_H_
#define XLS_CODEGEN_BLOCK_GENERATOR_H_

#include <string>

#include "absl/status/statusor.h"
#include "xls/codegen/codegen_options.h"
#include "xls/codegen/verilog_line_map.pb.h"
#include "xls/ir/block.h"

namespace xls {
namespace verilog {

// Generates and returns (System)Verilog text implementing the given top-level
// block. The text will include a Verilog module corresponding to the given
// block as well as module definitions for any instantiated blocks.
absl::StatusOr<std::string> GenerateVerilog(
    Block* top, const CodegenOptions& options,
    VerilogLineMap* verilog_line_map = nullptr);

}  // namespace verilog
}  // namespace xls

#endif  // XLS_CODEGEN_BLOCK_GENERATOR_H_
