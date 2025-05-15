// Copyright 2024 The XLS Authors
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

#include "xls/codegen/trace_verbosity_pass.h"

#include <cstdint>
#include <memory>
#include <vector>

#include "absl/status/statusor.h"
#include "xls/codegen/codegen_pass.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/block.h"
#include "xls/ir/node.h"
#include "xls/ir/nodes.h"
#include "xls/ir/topo_sort.h"
#include "xls/passes/pass_base.h"

namespace xls::verilog {
namespace {
absl::StatusOr<bool> FilterVerboseTraces(Block* block, int64_t verbosity) {
  bool changed = false;
  std::vector<Trace*> filtered_traces;
  for (Node* node : TopoSort(block)) {
    if (!node->Is<Trace>()) {
      continue;
    }
    Trace* trace = node->As<Trace>();
    if (trace->verbosity() <= verbosity) {
      continue;
    }
    filtered_traces.push_back(trace);
    XLS_RETURN_IF_ERROR(trace->ReplaceUsesWith(trace->token()));
    changed = true;
  }
  for (Trace* trace : filtered_traces) {
    XLS_RETURN_IF_ERROR(block->RemoveNode(trace));
  }
  return changed;
}
}  // namespace

absl::StatusOr<bool> TraceVerbosityPass::RunInternal(
    Package* package, const CodegenPassOptions& options, PassResults* results,
    CodegenContext& context) const {
  bool changed = false;
  for (const std::unique_ptr<Block>& block : package->blocks()) {
    XLS_ASSIGN_OR_RETURN(
        bool block_changed,
        FilterVerboseTraces(block.get(),
                            options.codegen_options.max_trace_verbosity()));
    changed = changed || block_changed;
  }

  if (changed) {
    context.GcMetadata();
  }

  return changed;
}

}  // namespace xls::verilog
