// Copyright 2022 The XLS Authors
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

#ifndef XLS_CODEGEN_NODE_REPRESENTATION_H_
#define XLS_CODEGEN_NODE_REPRESENTATION_H_

#include <variant>

#include "xls/codegen/vast/vast.h"

namespace xls {

// Not all nodes have direct representations in the Verilog. To handle these
// cases, use an std::variant type which holds one of the two possible
// representations for a Node:
//
//  1) Expression* : node is represented directly by a Verilog expression. This
//     is the common case.
//
//  2) UnrepresentedSentinel : node has no representation in the Verilog. For
//     example, the node emits a token type.
struct UnrepresentedSentinel {};
using NodeRepresentation =
    std::variant<UnrepresentedSentinel, xls::verilog::Statement*,
                 xls::verilog::Expression*>;
}  // namespace xls

#endif  // XLS_CODEGEN_NODE_REPRESENTATION_H_
