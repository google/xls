// Copyright 2020 The XLS Authors
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

#include "xls/codegen/state_removal_pass.h"

#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "xls/ir/channel.h"
#include "xls/ir/nodes.h"
#include "xls/ir/package.h"

namespace xls {

absl::StatusOr<bool> StateRemovalPass::RunOnProcInternal(
    Proc* proc, const PassOptions& options, PassResults* results) const {
  // If state is empty then just return.
  if (proc->StateType() == proc->package()->GetTupleType({})) {
    return false;
  }

  if (proc->StateType()->GetFlatBitCount() == 0) {
    return absl::UnimplementedError(absl::StrFormat(
        "Unsupported state type: %s", proc->StateType()->ToString()));
  }

  // Replace state param with streaming channel.
  XLS_ASSIGN_OR_RETURN(
      Channel * state_channel,
      proc->package()->CreateStreamingChannel(
          proc->StateParam()->GetName(), ChannelOps::kSendReceive,
          proc->StateType(), {proc->InitValue()}));

  // Create a receive of the recurrent state and replace current uses of the
  // state param with the received state data.
  XLS_ASSIGN_OR_RETURN(
      Receive * state_receive,
      proc->MakeNode<Receive>(proc->StateParam()->loc(), proc->TokenParam(),
                              /*predicate=*/absl::nullopt,
                              /*channel_id=*/state_channel->id()));
  XLS_RETURN_IF_ERROR(
      proc->StateParam()
          ->ReplaceUsesWithNew<TupleIndex>(state_receive, /*index=*/1)
          .status());

  // Thread the receive token to the send.
  XLS_ASSIGN_OR_RETURN(Node * receive_token,
                       proc->MakeNode<TupleIndex>(/*loc=*/absl::nullopt,
                                                  state_receive, /*index=*/0));
  // Create a send of the next state.
  XLS_ASSIGN_OR_RETURN(
      Send * state_send,
      proc->MakeNode<Send>(/*loc=*/absl::nullopt, receive_token,
                           /*data=*/proc->NextState(),
                           /*predicate=*/absl::nullopt,
                           /*channel_id=*/state_channel->id()));

  // Join the token from the send with the existing next token value.
  XLS_ASSIGN_OR_RETURN(
      Node * after_all,
      proc->MakeNode<AfterAll>(
          /*loc=*/absl::nullopt,
          /*args=*/std::vector<Node*>({proc->NextToken(), state_send})));
  XLS_RETURN_IF_ERROR(proc->SetNextToken(after_all));

  // Replace the state with an empty tuple.
  XLS_ASSIGN_OR_RETURN(
      Node * nil_state,
      proc->MakeNode<Tuple>(/*loc=*/absl::nullopt, std::vector<Node*>()));
  XLS_RETURN_IF_ERROR(proc->ReplaceState(/*state_param_name=*/"nil_state",
                                         nil_state, Value::Tuple({})));

  return true;
}

}  // namespace xls
