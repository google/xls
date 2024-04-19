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

#ifndef XLS_IR_VERIFY_NODE_H_
#define XLS_IR_VERIFY_NODE_H_

#include "absl/status/status.h"
#include "xls/ir/node.h"

namespace xls {
// Verifies numerous invariants of the IR for the given node. Returns an error
// status if a violation is found.
// This is split out of verifier.h because that needs to be part of the
// monolithic "ir" compilation unit. Splitting this out lets most of the
// verifier be in its own compilation unit.
absl::Status VerifyNode(Node* Node, bool codegen = false);
}  // namespace xls

#endif  // XLS_IR_VERIFY_NODE_H_
