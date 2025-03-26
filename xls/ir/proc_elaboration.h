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

#ifndef XLS_IR_PROC_ELABORATION_H_
#define XLS_IR_PROC_ELABORATION_H_

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xls/ir/channel.h"
#include "xls/ir/elaboration.h"
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

struct ChannelInstance {
  Channel* channel;

  // Instantiation path of the proc instance in which this channel is
  // defined. Is nullopt for old-style channels.
  std::optional<ProcInstantiationPath> path;

  std::string ToString() const;
};

// Data structure describing the binding of a channel interface (or channel in
// old-style procs) to a channel instance. Each channel interface has a unique
// binding. Channel interfaces on the interfaces of procs are bound to the
// respective channel interface of the proc instantiation argument. Channel
// interfaces to channels defined in the proc are bound directly to the channel.
struct ChannelBinding {
  // The channel instance the channel interface is bound to.
  ChannelInstance* instance;

  // The channel interface in the parent proc in the elaboration to which the
  // channel interface is bound. This value is std::nullopt for top-level
  // channel instances (and old-style procs) and for channel interfaces which
  // refer to channels declared in the proc itself.
  std::optional<ChannelInterface*> parent_interface;
};

// Representation of an instance of a proc. This is a recursive data structure
// which also holds all channel and proc instances instantiated by this proc
// instance including recursively.
class ProcInstance {
 public:
  ProcInstance(
      Proc* proc, std::optional<ProcInstantiation*> proc_instantiation,
      std::optional<ProcInstantiationPath> path,
      std::vector<std::unique_ptr<ChannelInstance>> channel_instances,
      std::vector<std::unique_ptr<ProcInstance>> instantiated_procs,
      absl::flat_hash_map<ChannelRef, ChannelBinding> channel_bindings);

  Proc* proc() const { return proc_; }

  // The ProcInstantiation IR construct which instantiates this proc
  // instance. This is std::nullopt if the proc corresponding to this
  // ProcInstance is the top proc.
  std::optional<ProcInstantiation*> proc_instantiation() const {
    return proc_instantiation_;
  }

  // The path to this proc instance through the proc hierarchy. This is
  // std::nullopt for old-style procs.
  const std::optional<ProcInstantiationPath>& path() const { return path_; }

  // The ChannelInstances corresponding to the channels declared in the proc
  // associated with this proc instance.
  absl::Span<const std::unique_ptr<ChannelInstance>> channels() const {
    return channel_instances_;
  }

  // The ProcInstances instantiated by this proc instance.
  absl::Span<const std::unique_ptr<ProcInstance>> instantiated_procs() const {
    return instantiated_procs_;
  }

  // Returns the ChannelInstance with the given name in this proc instance. The
  // channel instance can refer to an interface channel or a channel defined in
  // the proc. The name of the channel interface (ChannelInterface::name) may
  // differ than than name of the channel it is bound to
  // (ChannelInstance::channel.name).
  absl::StatusOr<ChannelInstance*> GetChannelInstance(
      std::string_view channel_interface_name) const;

  // Return the binding for the given channel interface. For new-style procs
  // only.
  ChannelBinding GetChannelBinding(ChannelInterface* channel_interface) const {
    CHECK(proc()->is_new_style_proc());
    return channel_bindings_.at(channel_interface);
  }

  // Return the binding for the given channel. For old-style procs only.
  ChannelBinding GetChannelBinding(Channel* channel) const {
    CHECK(!proc()->is_new_style_proc());
    return channel_bindings_.at(channel);
  }

  // Return the binding for the given ChannelRef.
  ChannelBinding GetChannelBinding(ChannelRef channel_ref) const {
    return channel_bindings_.at(channel_ref);
  }

  // Returns a unique name for this proc instantiation. For new-style procs this
  // includes the proc name and the instantiation path. For old-style procs this
  // is simply the proc name.
  std::string GetName() const;

  // Return a nested representation of the proc instance.
  std::string ToString(int64_t indent_amount = 0) const;

 private:
  Proc* proc_;
  std::optional<ProcInstantiation*> proc_instantiation_;
  std::optional<ProcInstantiationPath> path_;

  // Channel and proc instances in this proc instance. Unique pointers are used
  // for pointer stability as pointers to these objects are handed out.
  std::vector<std::unique_ptr<ChannelInstance>> channel_instances_;
  std::vector<std::unique_ptr<ProcInstance>> instantiated_procs_;

  // Map from ChannelRef (variant of ChannelInterface and Channel) to the
  // channel binding. For old-style procs this contains *all* channels as all
  // channels are referenceable in all procs. For new-style procs this contains
  // only the channel interfaces in this proc.
  absl::flat_hash_map<ChannelRef, ChannelBinding> channel_bindings_;

  // Map from channel interface name to channel instance for all channel
  // interfaces in the proc.
  absl::flat_hash_map<std::string, ChannelInstance*> channel_name_map_;
};

// Data structure representing the elaboration tree.
class ProcElaboration {
 public:
  static absl::StatusOr<ProcElaboration> Elaborate(Proc* top);

  // Elaborate the package of old style procs. This generates a single instance
  // for each proc and channel in the package. The instance paths of each object
  // are std::nullopt.

  // TODO(https://github.com/google/xls/issues/869): Remove when all procs are
  // new style.
  static absl::StatusOr<ProcElaboration> ElaborateOldStylePackage(
      Package* package);

  ProcInstance* top() const { return top_.get(); }

  std::string ToString() const;

  // Returns the proc/channel instance at the given path.
  absl::StatusOr<ProcInstance*> GetProcInstance(
      const ProcInstantiationPath& path) const;
  absl::StatusOr<ChannelInstance*> GetChannelInstance(
      std::string_view channel_name, const ProcInstantiationPath& path) const;

  // Returns the proc/channel instance at the given path where the path is given
  // as a serialization (e.g., `top_proc::inst->other_proc`).
  absl::StatusOr<ProcInstance*> GetProcInstance(
      std::string_view path_str) const;
  absl::StatusOr<ChannelInstance*> GetChannelInstance(
      std::string_view channel_name, std::string_view path_str) const;

  // Return a vector of all proc or channel instances in the elaboration.
  absl::Span<ProcInstance* const> proc_instances() const {
    return proc_instance_ptrs_;
  }
  absl::Span<ChannelInstance* const> channel_instances() const {
    return channel_instance_ptrs_;
  }

  // Returns the procs in this elaboration. The order of the procs is a
  // topological sort of the hierarchy with top first.
  absl::Span<Proc* const> procs() const { return procs_; }

  // Return all instances of a particular channel/proc.
  absl::Span<ProcInstance* const> GetInstances(Proc* proc) const;
  absl::Span<ChannelInstance* const> GetInstances(Channel* channel) const;

  // Return all channel instances which the given channel interface is bound to
  // in the elaboration.
  absl::Span<ChannelInstance* const> GetInstancesOfChannelInterface(
      ChannelInterface* channel_interface) const;

  // Returns whether the given channel interface binds to a channel on the top
  // interface.
  bool IsTopInterfaceChannel(ChannelInstance* channel) const;

  // Return the unique instance of the given proc/channel. Returns an error if
  // there is not exactly one instance associated with the IR object.
  absl::StatusOr<ProcInstance*> GetUniqueInstance(Proc* proc) const;
  absl::StatusOr<ChannelInstance*> GetUniqueInstance(Channel* channel) const;

  Package* package() const { return package_; }

  // Create path from the given path string serialization. Example input:
  //
  //    top_proc::inst1->other_proc::inst2->that_proc
  //
  // The return path will have the Proc pointer to `top_proc` as the top of the
  // path, with an instantiation path containing the ProcInstantiation pointers:
  // {inst1, inst2}.
  //
  // Returns an error if the path does not exist in the elaboration.
  absl::StatusOr<ProcInstantiationPath> CreatePath(
      std::string_view path_str) const;

 private:
  // Walks the hierarchy and builds the data member maps of instances.  Only
  // should be called for new-style procs.
  absl::Status BuildInstanceMaps(ProcInstance* proc_instance);

  Package* package_;

  // For a new-style proc, this is the top-level instantiation. All other
  // ProcInstances are contained within this instance.
  std::unique_ptr<ProcInstance> top_;

  // For non-new-style procs, this is the list of proc/channel instantiations,
  // one per proc in the package.
  std::vector<std::unique_ptr<ProcInstance>> proc_instances_;
  std::vector<std::unique_ptr<ChannelInstance>> channel_instances_;

  // Vectors of all proc/channel instances in the elaboration.
  std::vector<ProcInstance*> proc_instance_ptrs_;
  std::vector<ChannelInstance*> channel_instance_ptrs_;

  // Procs in this elaboration.
  std::vector<Proc*> procs_;

  // Channel object for the interface of the top-level proc. This is necessary
  // as there are no associated Channel objects in the IR.
  // TODO(https://github.com/google/xls/issues/869): An IR object should
  // probably not live outside the IR. Distill the necessary information from
  // Channel and use that instead.
  std::vector<std::unique_ptr<Channel>> interface_channels_;

  // Channel instances for the interface channels.
  std::vector<std::unique_ptr<ChannelInstance>> interface_channel_instances_;
  absl::flat_hash_set<ChannelInstance*> interface_channel_instance_set_;

  // All proc instances in the elaboration indexed by instantiation path.
  absl::flat_hash_map<ProcInstantiationPath, ProcInstance*>
      proc_instances_by_path_;

  // All channel instances in the elaboration indexed by channel interface name
  // (new style) or channel name (old style) and instantiation path.
  absl::flat_hash_map<std::pair<std::string, ProcInstantiationPath>,
                      ChannelInstance*>
      channel_instances_by_path_;

  // List of instances of each Proc/Channel.
  absl::flat_hash_map<Proc*, std::vector<ProcInstance*>> instances_of_proc_;
  absl::flat_hash_map<Channel*, std::vector<ChannelInstance*>>
      instances_of_channel_;

  // List of channel instances for each channel interface.
  absl::flat_hash_map<ChannelInterface*, std::vector<ChannelInstance*>>
      instances_of_channel_interface_;
};

}  // namespace xls

#endif  // XLS_IR_PROC_ELABORATION_H_
