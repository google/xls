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

#ifndef XLS_PASSES_CHANNEL_LEGALIZATION_PASS_H_
#define XLS_PASSES_CHANNEL_LEGALIZATION_PASS_H_

#include <string_view>

#include "absl/status/statusor.h"
#include "xls/ir/package.h"
#include "xls/passes/optimization_pass.h"
#include "xls/passes/pass_base.h"

namespace xls {

// Pass that legalizes multiple send/receive operations per channel.
//
// This pass adds cross-activation tokens to guarantee that later activations of
// a proc cannot send or receive on a channel until all previous activations
// have completed working with that channel.
class ChannelLegalizationPass : public OptimizationPass {
 public:
  static constexpr std::string_view kName = "channel_legalization";
  ChannelLegalizationPass()
      : OptimizationPass(kName, "Legalize multiple send/recvs per channel") {}
  ~ChannelLegalizationPass() override = default;

 protected:
  absl::StatusOr<bool> RunInternal(Package* p,
                                   const OptimizationPassOptions& options,
                                   PassResults* results,
                                   OptimizationContext& context) const override;
};

}  // namespace xls

#endif  // XLS_PASSES_CHANNEL_LEGALIZATION_PASS_H_
