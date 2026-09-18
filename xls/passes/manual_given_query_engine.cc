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

#include "xls/passes/manual_given_query_engine.h"

#include "absl/status/status.h"
#include "absl/types/span.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/node.h"
#include "xls/passes/query_engine.h"

namespace xls {

absl::Status ManualGivenQueryEngine::SetGivens(
    absl::Span<Node* const> nodes, const QueryEngine& information_source) {
  XLS_ASSIGN_OR_RETURN(specialized_union_, base_union_.SpecializeOnNodes(
                                               nodes, information_source));
  return absl::OkStatus();
}

}  // namespace xls
