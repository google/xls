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

#ifndef XLS_FDO_EXTRACT_NODES_H_
#define XLS_FDO_EXTRACT_NODES_H_

#include <string>
#include <string_view>

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "xls/ir/node.h"

namespace xls {

// Extract the given set of nodes from a function and return the verilog text of
// them. Flip-flops can be inserted to the live-ins and live-outs optionally.
absl::StatusOr<std::optional<std::string>> ExtractNodesAndGetVerilog(
    const absl::flat_hash_set<Node*>& nodes, std::string_view top_module_name,
    bool flop_inputs_outputs = false, bool return_all_liveouts = false);

}  // namespace xls

#endif  // XLS_FDO_EXTRACT_NODES_H_
