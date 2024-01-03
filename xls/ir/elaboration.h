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

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xls/ir/channel.h"
#include "xls/ir/proc.h"
#include "xls/ir/proc_instantiation.h"

namespace xls {

// Library for elaborating a proc hierarchy. A proc hierarchy is a directed
// acyclic graph of procs connected via proc instantiation. An elaboration
// flattens the proc hierarchy into a tree by walking all paths in the hierarchy
// starting at a `top` proc where a path is a chain of proc instantiations. For
// each IR construct (proc or channel), The elaboration creates a separate
// "instance" object for each path through the hierarchy from the top proc to
// the IR construct.
//
// Example proc hierarchy:
//
//   proc leaf_proc<ch0: ... in, ch0: .... out>(...) { }
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
//  my_top
//    chan c
//    chan d
//    other_proc<a, b> [my_inst0]
//      chan z
//      leaf_proc<x, z> [other_inst0]
//      leaf_proc<z, y> [other_inst1]
//    other_proc<c, c> [my_inst1]
//      chan z
//      leaf_proc<x, z> [other_inst0]
//      leaf_proc<z, y> [other_inst1]
//    leaf_proc<d, d> [my_inst2]
//
// There are five instances of `leaf_proc` as there are five paths from
// `top_proc` to `leaf_proc` in the proc hierarchy.

// A path of proc instantiations. An instance of a proc or channel is uniquely
// identified by its InstantiationPath.
struct InstantiationPath {
  Proc* top;
  std::vector<ProcInstantiation*> path;

  template <typename H>
  friend H AbslHashValue(H h, const InstantiationPath& p) {
    H state = H::combine(std::move(h), p.top->name());
    for (const ProcInstantiation* element : p.path) {
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

  std::string ToString() const;
};

struct ChannelInstance {
  Channel* channel;
  InstantiationPath path;
};

// Representation of an instance of a proc. This is a recursive data structure
// which also holds all channel and proc instances instantiated by this proc
// instance including recursively.
class ProcInstance {
 public:
  // Creates and returns a ProcInstance for the given proc. Walks and constructs
  // the tree of proc instances beneath this one.
  static absl::StatusOr<std::unique_ptr<ProcInstance>> Create(
      Proc* proc, std::optional<ProcInstantiation*> proc_instantiation,
      const InstantiationPath& path,
      absl::Span<ChannelInstance* const> interface);

  Proc* proc() const { return proc_; }

  // The ProcInstantiation IR construct which instantiates this proc
  // instance. This is nullopt if the proc corresponding to this ProcInstance
  // is the top proc.
  std::optional<ProcInstantiation*> proc_instantiation() const {
    return proc_instantiation_;
  }

  // The path to this proc instance through the proc hierarchy/
  const InstantiationPath& path() const { return path_; }

  // The ChannelInstances comprising the interface of this proc instance.
  absl::Span<ChannelInstance* const> interface() const { return interface_; }

  // The ChannelInstances corresponding to the channels declared in the proc
  // associated with this proc instance.
  absl::Span<const std::unique_ptr<ChannelInstance>> channels() const {
    return channels_;
  }

  // The ProcInstances instantiated by this proc instance.
  absl::Span<const std::unique_ptr<ProcInstance>> instantiated_procs() const {
    return instantiated_procs_;
  }

  // Return a nested representation of the proc instance.
  std::string ToString(int64_t indent_amount = 0) const;

 private:
  ProcInstance(Proc* proc, std::optional<ProcInstantiation*> proc_instantiation,
               const InstantiationPath& path,
               absl::Span<ChannelInstance* const> interface)
      : proc_(proc),
        proc_instantiation_(proc_instantiation),
        path_(path),
        interface_(interface.begin(), interface.end()) {}

  Proc* proc_;
  std::optional<ProcInstantiation*> proc_instantiation_;
  InstantiationPath path_;
  std::vector<ChannelInstance*> interface_;

  // Channel and proc instances in this proc instance. Unique pointers are used
  // for pointer stability as pointers to these objects are handed out.
  std::vector<std::unique_ptr<ChannelInstance>> channels_;
  std::vector<std::unique_ptr<ProcInstance>> instantiated_procs_;
};

// Data structure representing the elaboration tree.
class Elaboration {
 public:
  static absl::StatusOr<Elaboration> Elaborate(Proc* top);

  const ProcInstance& top() const { return *top_; }

  std::string ToString() const;

  // Returns the proc/channel instance at the given path.
  absl::StatusOr<ProcInstance*> GetProcInstance(
      const InstantiationPath& path) const;
  absl::StatusOr<ChannelInstance*> GetChannelInstance(
      std::string_view channel_name, const InstantiationPath& path) const;

 private:
  std::unique_ptr<ProcInstance> top_;

  // Channel object for the interface of the top-level proc. This is necessary
  // as there are no associated Channel objects in the IR.
  // TODO(https://github.com/google/xls/issues/869): An IR object should
  // probably not live outside the IR. Distill the necessary information from
  // Channel and use that instead.
  std::vector<std::unique_ptr<Channel>> interface_channels_;

  // Channel instances for the interface channels.
  std::vector<std::unique_ptr<ChannelInstance>> interface_channel_instances_;

  // All proc instances in the elaboration indexed by instantiation path.
  absl::flat_hash_map<InstantiationPath, ProcInstance*> proc_instances_by_path_;

  // All channel instances in the elaboration indexed by channel name and
  // instantiation path.
  absl::flat_hash_map<std::pair<std::string, InstantiationPath>,
                      ChannelInstance*>
      channel_instances_by_path_;
};

}  // namespace xls

#endif  // XLS_IR_ELABORATION_H_
