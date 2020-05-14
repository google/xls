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

#ifndef THIRD_PARTY_XLS_IR_NODE_ITERATOR_H_
#define THIRD_PARTY_XLS_IR_NODE_ITERATOR_H_

#include "xls/ir/function.h"
#include "xls/ir/node.h"

namespace xls {

// A type that orders the reachable nodes in a function into a usable traversal
// order. Currently just does a stable topological ordering.
//
// Note that this container value must outlive any iterators derived from it
// (via begin()/end()).
class NodeIterator {
 public:
  static NodeIterator Create(Function* f) {
    NodeIterator it(f);
    it.Initialize();
    return it;
  }

  static NodeIterator CreateReverse(Function* f) {
    NodeIterator it(f);
    it.Initialize();
    std::reverse(it.ordered_->begin(), it.ordered_->end());
    return it;
  }

  std::vector<Node*>::iterator begin() { return ordered_->begin(); }
  std::vector<Node*>::iterator end() { return ordered_->end(); }

 private:
  explicit NodeIterator(Function* f) : f_(f) {}

  void Initialize();

  // The vector of nodes is wrapped in a unique_ptr so that the
  // NodeIterator may be movable but the iterators returned to the
  // caller of begin()/end() are not invalidated by those moves.
  std::unique_ptr<std::vector<Node*>> ordered_;
  Function* f_;
};

// Convenience function for concise use in foreach constructs; e.g.:
//
//  for (Node* n : TopoSort(f)) {
//    ...
//  }
//
// Yields nodes in a stable topological traversal order (dependency ordering is
// satisfied).
//
// Note that the ordering for all nodes is computed up front, *not*
// incrementally as iteration proceeds.
inline NodeIterator TopoSort(Function* f) { return NodeIterator::Create(f); }

// As above, but returns a reverse topo order.
inline NodeIterator ReverseTopoSort(Function* f) {
  return NodeIterator::CreateReverse(f);
}

}  // namespace xls

#endif  // THIRD_PARTY_XLS_IR_NODE_ITERATOR_H_
