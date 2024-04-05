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

#ifndef XLS_IR_ELABORATION_H_
#define XLS_IR_ELABORATION_H_

#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "xls/ir/block.h"
#include "xls/ir/instantiation.h"
#include "xls/ir/node.h"
#include "xls/ir/package.h"
#include "xls/ir/proc.h"
#include "xls/ir/proc_instantiation.h"

namespace xls {

// Library for elaborating a proc or block hierarchy.
//
// A hierarchy is a directed acyclic graph of proc/blocks connected via
// instantiation. An elaboration flattens the hierarchy into a tree by walking
// all paths in the hierarchy starting at a `top` proc/block where a path is a
// chain of instantiations.
//
// The elaboration creates an "instance" object for each path through the
// hierarchy from the top proc/block to each IR construct (channel or
// instantiation).
//
// Example proc hierarchy:
//
//   proc leaf_proc<ch0: ... in, ch1: .... out>(...) { }
//
//   proc other_proc<x: ... in, y: .... out>(...) {
//     chan z(...)
//     proc_instantiation other_inst0(x, z, proc=leaf_proc)
//     proc_instantiation other_inst1(z, y, proc=leaf_proc)
//   }
//
//   proc my_top<a: ... in, b: ... out>(...) {
//     chan c(...)
//     chan d(...)
//     proc_instantiation my_inst0(a, b, proc=other_proc)
//     proc_instantiation my_inst1(c, c, proc=other_proc)
//     proc_instantiation my_inst2(d, d, proc=leaf_proc)
//   }
//
// Elaborating this hierarchy from `my_top` yields the following elaboration
// tree. Each line is a instance of either a proc or a channel.
//
//  <a, b>my_top
//    chan c
//    chan d
//    other_proc<x=a, y=b> [my_inst0]
//      chan z
//      leaf_proc<ch0=x, ch1=z> [other_inst0]
//      leaf_proc<ch0=z, ch1=y> [other_inst1]
//    other_proc<x=c, y=c> [my_inst1]
//      chan z
//      leaf_proc<ch0=x, ch1=z> [other_inst0]
//      leaf_proc<ch0=z, ch1=y> [other_inst1]
//    leaf_proc<ch0=d, ch1=d> [my_inst2]
//
// There are five instances of `leaf_proc` as there are five paths from
// `top_proc` to `leaf_proc` in the proc hierarchy.

// A path of instantiations. An instance of a proc/channel/block is uniquely
// identified by its InstantiationPath.
// This struct should be specialized for each type that is elaborated.
template <typename FunctionBaseType, typename InstType>
  requires(std::is_base_of_v<FunctionBase, FunctionBaseType>)
struct InstantiationPath {
  FunctionBaseType* top;
  std::vector<InstType*> path;

  template <typename H>
  friend H AbslHashValue(H h, const InstantiationPath& p) {
    H state = H::combine(std::move(h), p.top->name());
    for (const InstType* element : p.path) {
      state = H::combine(std::move(state), element->name());
    }
    return state;
  }
  bool operator==(const InstantiationPath& other) const {
    return top == other.top && path == other.path;
  }
  bool operator!=(const InstantiationPath& other) const {
    return !(*this == other);
  }

  std::string ToString() const {
    if (path.empty()) {
      return top->name();
    }
    return absl::StrFormat(
        "%s::%s", top->name(),
        absl::StrJoin(path, "::", [](std::string* s, const InstType* i) {
          absl::StrAppendFormat(s, "%s->%s", i->name(), InstantiatedName(*i));
        }));
  }

 private:
  // Returns the name of the entity instantiated by `inst`.
  // Note that this is not necessarily a `FunctionBase`, e.g. in the case of
  // fifo instantiations there is no underlying block with a name.
  static std::string_view InstantiatedName(const InstType& inst);
  // Returns the `FunctionBaseType` instantiated by `inst` if it exists.
  // Note that there may not be an underlying `FunctionBaseType` as in the case
  // of fifo instantiations.
  static std::optional<FunctionBaseType*> Instantiated(const InstType& inst);
  friend class BlockElaboration;
};

using ProcInstantiationPath = InstantiationPath<Proc, ProcInstantiation>;

// Note: this is a path of instantiations within `Block`s, which may be other
// kinds of instantiations besides `BlockInstantiation` (e.g. fifo or extern).
// This is in contrast to `ProcInstantiationPath`.
using BlockInstantiationPath = InstantiationPath<Block, Instantiation>;

}  // namespace xls

#endif  // XLS_IR_ELABORATION_H_
