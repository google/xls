// Copyright 2025 The XLS Authors
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

#ifndef XLS_CONTRIB_XLSCC_NODE_MANIPULATION_H_
#define XLS_CONTRIB_XLSCC_NODE_MANIPULATION_H_

#include "xls/ir/node.h"

namespace xlscc {

const xls::Node* RemoveIdentities(const xls::Node* node);

// This function checks, in the presence of continuations, that two nodes are
// equivalent. Simply checking that the node pointers are equal produces
// false negatives with continuations.
// TODO(seanhaskell): Is this necessary anymore once function slices are
// implemented?
bool NodesEquivalentWithContinuations(const xls::Node* a, const xls::Node* b);

}  // namespace xlscc

#endif  // XLS_CONTRIB_XLSCC_NODE_MANIPULATION_H_
