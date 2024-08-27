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

#ifndef XLS_DEV_TOOLS_EXTRACT_SEGMENT_H_
#define XLS_DEV_TOOLS_EXTRACT_SEGMENT_H_

#include <memory>
#include <string_view>

#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xls/ir/function.h"
#include "xls/ir/function_base.h"
#include "xls/ir/node.h"
#include "xls/ir/package.h"

namespace xls {

// Create a new package with a single Function which contains nodes in 'full'
// which are dominated by some 'source_nodes' and dominate some 'sink_nodes'. If
// no source or sink nodes are present all nodes are considered to be in the
// set. If the 'next_nodes_are_tuples' is true then 'next' nodes are considered
// to return a tuple containing the arguments.
//
// The function returns the values of all 'sink_nodes' in a tuple.
absl::StatusOr<std::unique_ptr<Package>> ExtractSegmentInNewPackage(
    FunctionBase* full, absl::Span<Node* const> source_nodes,
    absl::Span<Node* const> sink_nodes, std::string_view extracted_package_name,
    std::string_view extracted_function_name,
    bool next_nodes_are_tuples = true);

// Create a new function in the current package which contains nodes in 'full'
// which are dominated by some 'source_nodes' and dominate some 'sink_nodes'. If
// no source or sink nodes are present all nodes are considered to be in the
// set. If the 'next_nodes_are_tuples' is true then 'next' nodes are considered
// to return a tuple containing the arguments.
//
// The function returns the values of all 'sink_nodes' in a tuple.
absl::StatusOr<Function*> ExtractSegment(
    FunctionBase* full, absl::Span<Node* const> source_nodes,
    absl::Span<Node* const> sink_nodes,
    std::string_view extracted_function_name,
    bool next_nodes_are_tuples = true);
}  // namespace xls

#endif  // XLS_DEV_TOOLS_EXTRACT_SEGMENT_H_
