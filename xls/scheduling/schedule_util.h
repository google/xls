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

#ifndef XLS_SCHEDULING_SCHEDULE_UTIL_H_
#define XLS_SCHEDULING_SCHEDULE_UTIL_H_

#include "absl/container/flat_hash_set.h"
#include "xls/ir/function_base.h"
#include "xls/ir/node.h"

namespace xls {

// Returns the set of nodes in `f` that will have no use when the entity is
// synthesized; that is, the only effect of these nodes is to compute a value
// that is only used pre-synthesis (e.g., asserts, covers, & traces).
absl::flat_hash_set<Node*> GetDeadAfterSynthesisNodes(FunctionBase* f);

}  // namespace xls

#endif  // XLS_SCHEDULING_SCHEDULE_UTIL_H_
