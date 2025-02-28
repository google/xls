// Copyright 2023 The XLS Authors
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

#ifndef XLS_PASSES_RAM_REWRITE_PASS_H_
#define XLS_PASSES_RAM_REWRITE_PASS_H_

#include <cstdint>
#include <optional>
#include <string_view>

#include "absl/status/statusor.h"
#include "xls/ir/channel.h"
#include "xls/ir/package.h"
#include "xls/ir/type.h"
#include "xls/passes/optimization_pass.h"
#include "xls/passes/pass_base.h"

namespace xls {

// Each kind of ram has logical names for each port. Internally, these are
// represented as these enums.
enum class RamLogicalChannel {
  // Abstract RAM
  kAbstractReadReq,
  kAbstractReadResp,
  kAbstractWriteReq,
  // 1RW RAM
  k1RWReq,
  k1RWResp,
  // 1R1W RAM
  k1R1WReadReq,
  k1R1WReadResp,
  k1R1WWriteReq,

  // Special: used to signal write completion, is empty and should not be
  // codegen'd as a real channel.
  kWriteCompletion,
};

absl::StatusOr<RamLogicalChannel> RamLogicalChannelFromName(
    std::string_view name);
std::string_view RamLogicalChannelName(RamLogicalChannel logical_channel);

ChannelDirection GetRamLogicalChannelDirection(
    RamLogicalChannel logical_channel);

// If mask_width is defined, return bits[mask_width]. Otherwise, there are no
// masks, which we represent with an empty tuple. The empty tuple will be
// removed in codegen.
Type* GetMaskType(Package* package, std::optional<int64_t> mask_width);

// Pass that rewrites RAMs of one type to a new type. Generally this will be
// some kind of lowering from more abstract to concrete RAMs.
class RamRewritePass : public OptimizationPass {
 public:
  static constexpr std::string_view kName = "ram_rewrite";
  explicit RamRewritePass() : OptimizationPass(kName, "RAM Rewrite") {}

  ~RamRewritePass() override = default;

 protected:
  absl::StatusOr<bool> RunInternal(Package* p,
                                   const OptimizationPassOptions& options,
                                   PassResults* results,
                                   OptimizationContext& context) const override;
};

}  // namespace xls

#endif  // XLS_PASSES_RAM_REWRITE_PASS_H_
