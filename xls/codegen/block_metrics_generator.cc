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

#include "xls/codegen/block_metrics_generator.h"

#include "absl/strings/str_format.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/block.h"

namespace xls::verilog {
namespace {

int64_t GenerateFlopCount(Block* block) {
  int64_t count = 0;

  for (Register* reg : block->GetRegisters()) {
    Type* reg_type = reg->type();
    count += reg_type->GetFlatBitCount();
  }

  return count;
}

}  // namespace

absl::StatusOr<BlockMetricsProto> GenerateBlockMetrics(Block* block) {
  BlockMetricsProto proto;
  proto.set_flop_count(GenerateFlopCount(block));
  return proto;
}

}  // namespace xls::verilog
