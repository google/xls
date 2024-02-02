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

#include "absl/algorithm/container.h"
#include "absl/status/status.h"
#include "xls/codegen/codegen_pass.h"
#include "xls/common/status/ret_check.h"
#include "xls/ir/node.h"
#include "xls/ir/verifier.h"
#include "xls/passes/pass_base.h"

namespace xls::verilog {

absl::Status CodegenChecker::Run(CodegenPassUnit* unit,
                                 const CodegenPassOptions& options,
                                 PassResults* results) const {
  XLS_RET_CHECK_EQ(unit->streaming_io_and_pipeline.node_to_stage_map.size(),
                   absl::c_count_if(unit->block->nodes(),
                                    [&](Node* n) {
                                      return unit->streaming_io_and_pipeline
                                          .node_to_stage_map.contains(n);
                                    }))
      << "Dangling pointers present in node-id-to-stage map:\n"
      << unit->block->DumpIr();

  return VerifyPackage(unit->package);
}

}  // namespace xls::verilog
