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
    FunctionBase* module, const CodegenOptions& options) {
  Block* block = nullptr;

  XLS_RET_CHECK(module->IsProc() || module->IsFunction());
  if (module->IsFunction()) {
    XLS_ASSIGN_OR_RETURN(block, FunctionToCombinationalBlock(
                                    dynamic_cast<Function*>(module), options));
  } else {
    XLS_ASSIGN_OR_RETURN(
        block, ProcToCombinationalBlock(dynamic_cast<Proc*>(module), options));
  }

  CodegenPassUnit unit(module->package(), block);

  CodegenPassOptions codegen_pass_options;
  codegen_pass_options.codegen_options = options;

  PassResults results;
  XLS_RETURN_IF_ERROR(CreateCodegenPassPipeline()
                          ->Run(&unit, codegen_pass_options, &results)
                          .status());
  XLS_RET_CHECK(unit.signature.has_value());
  VerilogLineMap verilog_line_map;
  XLS_ASSIGN_OR_RETURN(std::string verilog,
                       GenerateVerilog(block, options, &verilog_line_map));

  return ModuleGeneratorResult{verilog, verilog_line_map,
                               unit.signature.value()};
}

}  // namespace verilog
}  // namespace xls
