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

#include "xls/codegen/mark_channel_fifos_pass.h"

#include "absl/status/statusor.h"
#include "xls/codegen/codegen_options.h"
#include "xls/codegen/codegen_pass.h"
#include "xls/common/casts.h"
#include "xls/ir/channel.h"

namespace xls::verilog {
namespace {
FlopKind GetRealFlopKind(bool enabled, CodegenOptions::IOKind kind) {
  if (!enabled) {
    return FlopKind::kNone;
  }
  return CodegenOptions::IOKindToFlopKind(kind);
}
}  // namespace

absl::StatusOr<bool> MarkChannelFifosPass::RunInternal(
    CodegenPassUnit* unit, const CodegenPassOptions& options,
    CodegenPassResults* results) const {
  if (unit->package->ChannelsAreProcScoped()) {
    // Nothing to do for proc-scoped channels. ChannelInterfaces have a FlopKind
    // not a std::optional<FlopKind>.
  }
  bool changed = false;
  for (Channel* chan : unit->package->channels()) {
    if (chan->kind() != ChannelKind::kStreaming) {
      continue;
    }
    StreamingChannel* schan = down_cast<StreamingChannel*>(chan);
    if (!schan->channel_config().input_flop_kind()) {
      schan->channel_config(schan->channel_config().WithInputFlopKind(
          GetRealFlopKind(options.codegen_options.flop_inputs(),
                          options.codegen_options.flop_inputs_kind())));
      changed = true;
    }
    if (!schan->channel_config().output_flop_kind()) {
      schan->channel_config(schan->channel_config().WithOutputFlopKind(
          GetRealFlopKind(options.codegen_options.flop_outputs(),
                          options.codegen_options.flop_outputs_kind())));
      changed = true;
    }
  }
  return changed;
}

}  // namespace xls::verilog
