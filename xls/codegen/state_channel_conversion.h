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

#ifndef XLS_CODEGEN_STATE_CHANNEL_CONVERSION_H_
#define XLS_CODEGEN_STATE_CHANNEL_CONVERSION_H_

#include "absl/status/statusor.h"
#include "xls/ir/channel.h"
#include "xls/ir/proc.h"

namespace xls::verilog {

// Convert state to loop-back channel, reads to receive(), next to send().
// Use given ChannelConfig go configure channel.
// Return 'true' if any change was made.
absl::StatusOr<bool> State2ChannelConversionPass(Proc* proc,
                                                 const ChannelConfig& config);
}  // namespace xls::verilog

#endif  // XLS_CODEGEN_STATE_CHANNEL_CONVERSION_H_
