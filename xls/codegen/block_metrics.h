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

#ifndef XLS_CODEGEN_BLOCK_METRICS_H_
#define XLS_CODEGEN_BLOCK_METRICS_H_


#include "absl/status/statusor.h"
#include "xls/codegen/xls_metrics.pb.h"
#include "xls/estimators/delay_model/delay_estimator.h"
#include "xls/ir/block.h"

namespace xls::verilog {

// Collects and generate metrics related to the contents of the block.
// (ex. flop count, number of operations, etc...).
//
// TODO(tedhong): 2022-01-28 Add a class around the proto.
absl::StatusOr<BlockMetricsProto> GenerateBlockMetrics(
    Block* block, const DelayEstimator* delay_estimator = nullptr);

}  // namespace xls::verilog

#endif  // XLS_CODEGEN_BLOCK_METRICS_H_
