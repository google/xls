// Copyright 2020 Google LLC
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

#ifndef THIRD_PARTY_XLS_IR_VERIFIER_H_
#define THIRD_PARTY_XLS_IR_VERIFIER_H_

#include "absl/status/status.h"
#include "xls/ir/function.h"
#include "xls/ir/node.h"
#include "xls/ir/package.h"

namespace xls {

class Node;
class Function;
class Package;

// Verifies numerous invariants of the IR for the given package. Returns a
// error status if a violation is found.
absl::Status Verify(Package* package);

// Overload for functions.
absl::Status Verify(Function* function);

// Overload for nodes.
absl::Status Verify(Node* Node);

}  // namespace xls

#endif  // THIRD_PARTY_XLS_IR_VERIFIER_H_
