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

#include "xls/codegen/combinational_generator.h"

#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"
#include "xls/codegen/block_conversion.h"
#include "xls/codegen/block_generator.h"
#include "xls/codegen/codegen_options.h"
#include "xls/codegen/codegen_pass.h"
#include "xls/codegen/codegen_pass_pipeline.h"
#include "xls/codegen/flattening.h"
#include "xls/codegen/module_builder.h"
#include "xls/codegen/node_expressions.h"
#include "xls/codegen/signature_generator.h"
#include "xls/codegen/vast.h"
#include "xls/common/logging/log_lines.h"
#include "xls/common/logging/logging.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/bits.h"
#include "xls/ir/dfs_visitor.h"
#include "xls/ir/function.h"
#include "xls/ir/node.h"
#include "xls/ir/node_iterator.h"
#include "xls/ir/node_util.h"
#include "xls/ir/nodes.h"
#include "xls/ir/type.h"

namespace xls {
namespace verilog {

absl::StatusOr<ModuleGeneratorResult> GenerateCombinationalModule(
    Function* func, bool use_system_verilog, absl::string_view module_name) {
  XLS_ASSIGN_OR_RETURN(
      Block * block,
      FunctionToBlock(func, module_name.empty()
                                ? SanitizeIdentifier(func->name())
                                : module_name));
  CodegenPassUnit unit(func->package(), block);
  CodegenPassOptions pass_options;
  pass_options.codegen_options.entry(block->name())
      .use_system_verilog(use_system_verilog);
  PassResults results;
  XLS_RETURN_IF_ERROR(
      CreateCodegenPassPipeline()->Run(&unit, pass_options, &results).status());
  XLS_RET_CHECK(unit.signature.has_value());
  XLS_ASSIGN_OR_RETURN(std::string verilog,
                       GenerateVerilog(block, pass_options.codegen_options));

  return ModuleGeneratorResult{verilog, unit.signature.value()};
}

}  // namespace verilog
}  // namespace xls
