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

#include "xls/codegen/codegen_checker.h"

#include <optional>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "xls/codegen/codegen_pass.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/node.h"
#include "xls/ir/verifier.h"
#include "xls/passes/pass_base.h"

namespace xls::verilog {

namespace {

absl::Status CheckNodeToStageMap(const CodegenPassUnit& unit) {
  XLS_RET_CHECK_EQ(
      unit.streaming_io_and_pipeline.node_to_stage_map.size(),
      absl::c_count_if(
          unit.block->nodes(),
          [&](Node* n) {
            return unit.streaming_io_and_pipeline.node_to_stage_map.contains(n);
          }))
      << "Dangling pointers present in node-id-to-stage map\n";
  return absl::OkStatus();
}

absl::Status CheckRegisterLists(const CodegenPassUnit& unit) {
  absl::flat_hash_set<Node*> nodes(unit.block->nodes().begin(),
                                   unit.block->nodes().end());
  for (const std::optional<StateRegister>& reg :
       unit.streaming_io_and_pipeline.state_registers) {
    if (reg) {
      XLS_RET_CHECK(nodes.contains(reg->reg_read)) << "read of " << reg->name;
      XLS_RET_CHECK(nodes.contains(reg->reg_write)) << "write of " << reg->name;
    }
  }
  for (const PipelineStageRegisters& stage :
       unit.streaming_io_and_pipeline.pipeline_registers) {
    for (const PipelineRegister& reg : stage) {
      XLS_RET_CHECK(nodes.contains(reg.reg_read));
      XLS_RET_CHECK(nodes.contains(reg.reg_write));
    }
  }
  return absl::OkStatus();
}

}  // namespace
absl::Status CodegenChecker::Run(CodegenPassUnit* unit,
                                 const CodegenPassOptions& options,
                                 PassResults* results) const {
  XLS_RETURN_IF_ERROR(CheckNodeToStageMap(*unit)) << unit->block->DumpIr();
  XLS_RETURN_IF_ERROR(CheckRegisterLists(*unit)) << unit->block->DumpIr();
  return VerifyPackage(unit->package);
}

}  // namespace xls::verilog
