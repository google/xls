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

#ifndef XLS_IR_VERIFIER_H_
#define XLS_IR_VERIFIER_H_

#include <functional>
#include <vector>

#include "absl/status/status.h"
#include "xls/ir/topo_sort.h"

namespace xls {

class Node;
class Function;
class Proc;
class Block;
class Package;
class FunctionBase;

// Verifies numerous invariants of the IR for the given IR construct. Returns a
// error status if a violation is found.
absl::Status VerifyPackage(
    Package* package, bool codegen = false,
    std::function<std::vector<Node*>(FunctionBase*)> topo_sort =
        [](FunctionBase* fb) { return TopoSort(fb); });
absl::Status VerifyFunction(Function* function, bool codegen = false);
absl::Status VerifyProc(Proc* Proc, bool codegen = false);
absl::Status VerifyBlock(Block* Block, bool codegen = false);

}  // namespace xls

#endif  // XLS_IR_VERIFIER_H_
