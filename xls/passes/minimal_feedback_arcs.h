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

#ifndef XLS_PASSES_MINIMAL_FEEDBACK_ARCS_H_
#define XLS_PASSES_MINIMAL_FEEDBACK_ARCS_H_

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "xls/ir/channel.h"
#include "xls/ir/package.h"

namespace xls {
// Identifies the approximately minimal set of channels where combinational
// paths must be broken to prevent cycles.
//
// Proc networks can have cycles in their channel operations. For example, two
// procs that both receive on a channel and send the result back on another
// channel can be cross-coupled with some initialization to prime the loop.
// These networks are valid, but realizing them in hardware requires care: you
// need to avoid the channel cycle becoming a combinational loop in hardware.
// Somewhere, the cycle needs to be broken by a state element. In practice, this
// might mean realizing a FIFO without bypass.
//
// This function computes the approximately minimal set of channels with the
// property that removing combinational paths between the send and receive on
// each individual channel will ensure that there are no combinational loops in
// the entire network.
//
// Note that this function expects at most one send and at most one receive per
// channel and should therefore be performed after channel legalization.
absl::StatusOr<absl::flat_hash_set<Channel*>> MinimalFeedbackArcs(
    const Package* p);
}  // namespace xls

#endif  // XLS_PASSES_MINIMAL_FEEDBACK_ARCS_H_
