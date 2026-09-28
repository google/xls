// Copyright 2026 The XLS Authors
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

// SPIN <-> DSLX channel-event trace parsing and comparison utilities.

#ifndef XLS_SPIN_TRACE_COMPARE_H_
#define XLS_SPIN_TRACE_COMPARE_H_

#include <cstdint>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/container/btree_map.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xls/ir/package.h"

namespace xls::dslx {
class Module;
}  // namespace xls::dslx

namespace xls::spin {

enum class Direction { kSend, kRecv };

// (channel_name, direction) -> ordered value sequence.
// btree_map for deterministic iteration in error messages.
using TraceMap =
    absl::btree_map<std::pair<std::string, Direction>, std::vector<int64_t>>;

// Maps each IR proc name to the ordered list of DSLX-style proc paths for its
// instantiations. Index k is the path of the k-th spawn (matching DSLX
// instance numbering, e.g. "CounterTest->Counter#0").
using ProcInstPaths =
    absl::flat_hash_map<std::string, std::vector<std::string>>;

// Maps (proc_type, instance_index, variable_name) -> ChannelDecl string
// literal. Used by ParseDslxTrace to rewrite DSLX variable names to physical
// channel names.
using DslxChannelNameMap =
    absl::flat_hash_map<std::tuple<std::string, int64_t, std::string>,
                        std::string>;

// Builds a DslxChannelNameMap from a DSLX module. Maps each proc instance's
// channel variable names to the corresponding ChannelDecl string literals.
DslxChannelNameMap BuildDslxChannelNameMap(const dslx::Module& module);

// Builds proc-path lookup used by ParseSpinTrace; BFS maps mangled IR proc names
// to instantiation paths (e.g. "__foo__Child_0_next" -> "Parent->Child#0").
absl::StatusOr<ProcInstPaths> BuildProcInstPathsForSpin(Package* package);

// Returns true if the SPIN JSON trace contains a SEND on terminator_channel.
bool SpinTraceHasTerminator(std::string_view json,
                            std::string_view terminator_channel);

// Parses newline-delimited JSON from `spin -Q`; values reinterpreted as uint32.
// Truncates at the first SEND on terminator_channel; key format is proc-path::channel.
absl::StatusOr<TraceMap> ParseSpinTrace(
    std::string_view json, const ProcInstPaths& proc_paths,
    std::string_view terminator_channel = "");

// Parses an EvaluatorResultsProto textproto; channel_name_map rewrites bare names.
// Truncates at the first SEND on terminator_channel.
absl::StatusOr<TraceMap> ParseDslxTrace(
    std::string_view textproto, std::string_view terminator_channel = "",
    const DslxChannelNameMap& channel_name_map = {});

// Returns FailedPrecondition with a per-channel diff if the maps differ.
absl::Status CompareTraces(const TraceMap& spin, const TraceMap& dslx);

}  // namespace xls::spin

#endif  // XLS_SPIN_TRACE_COMPARE_H_
