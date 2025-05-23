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

#include "xls/codegen/passes_ng/state_to_channel_conversion_pass.h"

#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "xls/codegen/codegen_pass.h"
#include "xls/codegen/passes_ng/stage_conversion.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/channel.h"
#include "xls/ir/channel_ops.h"
#include "xls/ir/function_base.h"
#include "xls/ir/node.h"
#include "xls/ir/nodes.h"
#include "xls/ir/package.h"
#include "xls/ir/state_element.h"
#include "xls/ir/type.h"
#include "xls/passes/pass_base.h"

namespace xls::verilog {

absl::StatusOr<bool> StateToChannelConversionPass::RunOnProcInternal(
    Proc* proc, const CodegenPassOptions& options, PassResults* results,
    CodegenContext& context) const {
  // Preprocess: find all state-elements and create a channel.
  struct StateChannelInfo {
    AfterAll* token;
    StreamingChannel* channel;
  };
  absl::flat_hash_map<Node*, StateChannelInfo> state_read2channel;

  // Be compatible with old and new-style proc.
  // With old-style procs (global names), we need to have a unique prefix.
  // Next two lines can melt away once we use new-style proc everywhere.
  Proc* const channel_proc = proc->is_new_style_proc() ? proc : nullptr;
  const char* prefix = proc->is_new_style_proc() ? "" : proc->name().c_str();

  // Set up channels first to create channel and corresponding token together.
  // There is already a convenient access to next values, use that to extract.
  for (Next* next : proc->next_values()) {
    StateRead* const state_read = next->state_read()->As<StateRead>();
    StateElement* const state = state_read->state_element();
    if (state_read2channel.contains(state_read)) {
      continue;  // Already handled.
    }
    Type* const state_type = state->type();
    std::string channel_name = absl::StrCat(prefix, state->name(), "_channel");
    XLS_ASSIGN_OR_RETURN(
        StreamingChannel * channel,
        proc->package()->CreateStreamingChannelInProc(
            channel_name, ChannelOps::kSendReceive, state_type, channel_proc,
            {state->initial_value()}, options.state_channel_config,
            FlowControl::kReadyValid,
            ChannelStrictness::kProvenMutuallyExclusive, /*id=*/std::nullopt));

    context.stage_conversion_metadata().AddStateLoopbackChannelName(
        proc, channel_name);

    // Token: here per channel, but possibly only one needed for all of state.
    absl::Span<Node*> no_dependencies;
    XLS_ASSIGN_OR_RETURN(AfterAll * token, proc->MakeNode<AfterAll>(
                                               next->loc(), no_dependencies));
    StateChannelInfo info{.token = token, .channel = channel};
    state_read2channel.emplace(state_read, info);
  }

  std::vector<Next*> to_remove;
  std::vector<std::pair<Node*, Node*>> state_read_replacements;
  // Wire up channels receive/send where we used state read/write before.
  for (Node* node : proc->nodes()) {
    if (node->Is<Next>()) {
      Next* const next = node->As<Next>();
      auto found = state_read2channel.find(next->state_read());
      XLS_RET_CHECK(found != state_read2channel.end());
      const StateChannelInfo& info = found->second;
      XLS_RET_CHECK_OK(proc->MakeNode<Send>(
          node->loc(), info.token, next->value(),
          /*predicate=*/next->predicate(), info.channel->name()));
      to_remove.push_back(next);
    } else if (node->Is<StateRead>()) {
      StateRead* state_read = node->As<StateRead>();
      auto found = state_read2channel.find(state_read);
      XLS_RET_CHECK(found != state_read2channel.end());
      const StateChannelInfo& info = found->second;

      XLS_ASSIGN_OR_RETURN(
          Receive * receive,
          proc->MakeNode<Receive>(node->loc(), info.token,
                                  /*predicate=*/state_read->predicate(),
                                  info.channel->name(),
                                  /*is_blocking=*/true));
      XLS_ASSIGN_OR_RETURN(Node * receive_value,
                           proc->MakeNode<TupleIndex>(node->loc(), receive, 1));
      state_read_replacements.emplace_back(node, receive_value);
    }
  }

  // Unlink old nodes and state elements.
  for (auto [from, to] : state_read_replacements) {
    XLS_RET_CHECK_OK(from->ReplaceUsesWith(to));
  }

  // State elements not needed anymore.
  for (int64_t i = proc->GetStateElementCount() - 1; i >= 0; --i) {
    XLS_RETURN_IF_ERROR(proc->RemoveStateElement(i));
  }

  for (Node* n : to_remove) {
    XLS_RETURN_IF_ERROR(proc->RemoveNode(n));
  }

  return !to_remove.empty();
}
}  // namespace xls::verilog
