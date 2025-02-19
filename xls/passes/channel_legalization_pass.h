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
// This pass creates new channels and an adapter proc to replace multiple
// sends/receives with a send/receive from the new adapter proc. There are three
// new channels per original channel operation:
// 1) A send from the original proc to the adapter with the predicate of the
//    operation to the adapter.
// 2) The original operation on a new internal channel with the adapter on the
//    other side.
// 3) A send from the adapter proc to the original proc indicating the channel
//    operation has completed. There is no payload so the type of the channel is
//    the empty tuple type.
//
// For example, the original design:
//
// ┌───────────┐
// │ ┌──────┐  │
// │ │send.0├──┼────────┐
// │ └──────┘  │        │
// │ ┌──────┐  │      ┌─▼───┐
// │ │send.1├──┼─────►│chan0│
// │ └──────┘  │      └─▲───┘
// │ ┌──────┐  │        │
// │ │send.2├──┼────────┘
// │ └──────┘  │
// │           │
// │  proc_0   │
// └───────────┘
//
// becomes
//
// ┌─────────┐              ┌──────────────────┐               ┌─────────┐
// │ ┌────┐  │   ┌───────┐  │  ┌────┐  ┌────┐  │   ┌────────┐  │   ┌────┐│
// │ │send├──┼──►│chan0_0├──┼─►│recv│  │send├──┼──►│compl0_0│──┼──►│recv││
// │ └────┘  │   └───────┘  │  └────┘  └────┘  │   └────────┘  │   └────┘│
// │ ┌────┐  │   ┌───────┐  │  ┌────┐  ┌────┐  │   ┌────────┐  │   ┌────┐│
// │ │send├──┼──►│pred0_0├──┼─►│recv│  │send├──┼──►│compl0_1│──┼──►│recv││
// │ └────┘  │   └───────┘  │  └────┘  └────┘  │   └────────┘  │   └────┘│
// │ ┌────┐  │   ┌───────┐  │  ┌────┐  ┌────┐  │   ┌────────┐  │   ┌────┐│
// │ │send├──┼──►│chan0_1├──┼─►│recv│  │send├──┼──►│compl0_2│──┼──►│recv││
// │ └────┘  │   └───────┘  │  └────┘  └────┘  │   └────────┘  │   └────┘│
// │ ┌────┐  │   ┌───────┐  │  ┌────┐          │               │         │
// │ │send├──┼──►│pred0_1├──┼─►│recv│          │               │ proc_0  │
// │ └────┘  │   └───────┘  │  └────┘          │               └─────────┘
// │ ┌────┐  │   ┌───────┐  │  ┌────┐          │
// │ │send├──┼──►│chan0_2├──┼─►│recv│          │
// │ └────┘  │   └───────┘  │  └────┘          │
// │ ┌────┐  │   ┌───────┐  │  ┌────┐ ┌────┐   │  ┌─────┐
// │ │send├──┼──►│pred0_2├──┼─►│recv│ │send├───┼─►│chan0│
// │ └────┘  │   └───────┘  │  └────┘ └────┘   │  └─────┘
// │         │              │                  │
// │  proc_0 │              │  adapter_proc    │
// └─────────┘              └──────────────────┘
//
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
                                   OptimizationContext* context) const override;
};

}  // namespace xls

#endif  // XLS_PASSES_CHANNEL_LEGALIZATION_PASS_H_
