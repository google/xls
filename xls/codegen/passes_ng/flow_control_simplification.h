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

#ifndef XLS_CODEGEN_PASSES_NG_FLOW_CONTROL_SIMPLIFICATION_H_
#define XLS_CODEGEN_PASSES_NG_FLOW_CONTROL_SIMPLIFICATION_H_

#include "absl/base/nullability.h"
#include "absl/status/status.h"
#include "xls/codegen/passes_ng/stage_to_block_conversion_metadata.h"
#include "xls/ir/block.h"

namespace xls::verilog {

// Removes ready backpressure from the top-level block by ignoring any
// input ready signal and setting all output ready signals to 1.
//
// The top-level proc must be a proc created by stage conversion.
absl::Status RemoveReadyBackpressure(BlockMetadata& top_block_metadata);

// Removes valid signals from the top-level block by ignoring any
// input valid signal and setting all output valid signals to 1.
//
// The top-level proc must be a proc created by stage conversion.
absl::Status RemoveValidSignals(BlockMetadata& top_block_metadata);

// Removes unused input ports from the top-level block.
absl::Status RemoveUnusedInputPorts(Block* absl_nonnull top_block);

// Removes constant output ports from the top-level block.
//
// Note that this function only removed ports that are immediately connected
// to a constant value and no analysis or constant propagation is performed.
absl::Status RemoveConstantOutputPorts(Block* absl_nonnull top_block);

}  // namespace xls::verilog

#endif  // XLS_CODEGEN_PASSES_NG_BLOCK_PIPELINE_INSERTER_H_
