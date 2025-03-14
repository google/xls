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

#include "xls/ir/proc_elaboration.h"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"
#include "absl/types/span.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/channel.h"
#include "xls/ir/channel_ops.h"
#include "xls/ir/elaboration.h"
#include "xls/ir/node.h"
#include "xls/ir/package.h"
#include "xls/ir/proc.h"
#include "xls/ir/proc_instantiation.h"
#include "xls/ir/value.h"

namespace xls {
namespace {

absl::StatusOr<std::unique_ptr<ProcInstance>> CreateNewStyleProcInstance(
    Proc* proc, std::optional<ProcInstantiation*> proc_instantiation,
    const ProcInstantiationPath& path,
    absl::Span<const ChannelBinding> interface_bindings) {
  XLS_RET_CHECK(proc->is_new_style_proc());

  // Map from channel reference name to ChannelInstance.
  absl::flat_hash_map<ChannelRef, ChannelBinding> channel_bindings;
  XLS_RET_CHECK_EQ(interface_bindings.size(), proc->interface().size());
  for (int64_t i = 0; i < interface_bindings.size(); ++i) {
    channel_bindings[proc->interface()[i]] = interface_bindings[i];
  }
  std::vector<std::unique_ptr<ChannelInstance>> declared_channels;
  for (Channel* channel : proc->channels()) {
    declared_channels.push_back(std::make_unique<ChannelInstance>(
        ChannelInstance{.channel = channel, .path = path}));
    ChannelInstance* channel_instance = declared_channels.back().get();
    XLS_ASSIGN_OR_RETURN(ChannelInterface * send_interface,
                         proc->GetSendChannelInterface(channel->name()));
    XLS_ASSIGN_OR_RETURN(ChannelInterface * receive_interface,
                         proc->GetReceiveChannelInterface(channel->name()));
    // Channel bindings for channels declared in this proc do not themselves
    // bind to another reference, so the parent reference field is empty.
    channel_bindings[send_interface] = ChannelBinding{
        .instance = channel_instance, .parent_interface = std::nullopt};
    channel_bindings[receive_interface] = ChannelBinding{
        .instance = channel_instance, .parent_interface = std::nullopt};
  }

  std::vector<std::unique_ptr<ProcInstance>> instantiated_procs;
  for (const std::unique_ptr<ProcInstantiation>& instantiation :
       proc->proc_instantiations()) {
    ProcInstantiationPath instantiation_path = path;
    instantiation_path.path.push_back(instantiation.get());

    // Check for circular dependencies. Walk the original path and see if
    // `instantiation->proc()` appears any where.
    if (instantiation->proc() == path.top ||
        std::find_if(path.path.begin(), path.path.end(),
                     [&](ProcInstantiation* pi) {
                       return pi->proc() == instantiation->proc();
                     }) != path.path.end()) {
      return absl::InternalError(
          absl::StrFormat("Circular dependency in proc instantiations: %s",
                          instantiation_path.ToString()));
    }

    std::vector<ChannelBinding> subproc_interface_bindings;
    for (ChannelInterface* channel_ref : instantiation->channel_args()) {
      subproc_interface_bindings.push_back(
          ChannelBinding{.instance = channel_bindings.at(channel_ref).instance,
                         .parent_interface = channel_ref});
    }
    XLS_ASSIGN_OR_RETURN(std::unique_ptr<ProcInstance> instantiation_instance,
                         CreateNewStyleProcInstance(
                             instantiation->proc(), instantiation.get(),
                             instantiation_path, subproc_interface_bindings));
    instantiated_procs.push_back(std::move(instantiation_instance));
  }

  return std::make_unique<ProcInstance>(
      proc, proc_instantiation, path, std::move(declared_channels),
      std::move(instantiated_procs), std::move(channel_bindings));
}

}  // namespace

std::string ChannelInstance::ToString() const {
  if (path.has_value()) {
    return absl::StrFormat("%s [%s]", channel->name(), path->ToString());
  }
  return std::string{channel->name()};
}

ProcInstance::ProcInstance(
    Proc* proc, std::optional<ProcInstantiation*> proc_instantiation,
    std::optional<ProcInstantiationPath> path,
    std::vector<std::unique_ptr<ChannelInstance>> channel_instances,
    std::vector<std::unique_ptr<ProcInstance>> instantiated_procs,
    absl::flat_hash_map<ChannelRef, ChannelBinding> channel_bindings)
    : proc_(proc),
      proc_instantiation_(proc_instantiation),
      path_(std::move(path)),
      channel_instances_(std::move(channel_instances)),
      instantiated_procs_(std::move(instantiated_procs)),
      channel_bindings_(std::move(channel_bindings)) {
  if (proc->is_new_style_proc()) {
    for (const std::unique_ptr<ChannelInterface>& channel_interface :
         proc->channel_interfaces()) {
      channel_name_map_[channel_interface->name()] =
          channel_bindings_.at(channel_interface.get()).instance;
    }
  } else {
    for (Channel* channel : proc->package()->channels()) {
      channel_name_map_[channel->name()] =
          channel_bindings_.at(channel).instance;
    }
  }
}

absl::StatusOr<ChannelInstance*> ProcInstance::GetChannelInstance(
    std::string_view channel_interface_name) const {
  if (channel_name_map_.contains(channel_interface_name)) {
    return channel_name_map_.at(channel_interface_name);
  }
  return absl::NotFoundError(
      absl::StrFormat("No channel reference named `%s` in proc `%s`",
                      channel_interface_name, proc()->name()));
}

std::string ProcInstance::GetName() const {
  if (!path().has_value()) {
    return proc()->name();
  }
  return absl::StrFormat("%s [%s]", proc()->name(), path()->ToString());
}

std::string ProcInstance::ToString(int64_t indent_amount) const {
  if (!proc()->is_new_style_proc()) {
    return proc()->name();
  }

  auto indent = [&](std::string_view s, int64_t additional_indent = 0) {
    std::string indent_str(2 * (indent_amount + additional_indent), ' ');
    return absl::StrFormat("%s%s", indent_str, s);
  };
  std::vector<std::string> pieces;
  std::string suffix =
      proc_instantiation().has_value()
          ? absl::StrFormat(" [%s]", (*proc_instantiation())->name())
          : "";
  pieces.push_back(indent(absl::StrFormat(
      "%s<%s>%s", proc()->name(),
      absl::StrJoin(
          proc()->interface(), ", ",
          [&](std::string* s, ChannelInterface* c) {
            absl::StrAppendFormat(
                s, "%s%s", c->name(),
                GetChannelBinding(c).parent_interface.has_value()
                    ? absl::StrCat(
                          "=",
                          GetChannelBinding(c).parent_interface.value()->name())
                    : "");
          }),
      suffix)));
  for (Channel* channel : proc()->channels()) {
    pieces.push_back(indent(absl::StrFormat("chan %s", channel->name()), 1));
  }
  for (const std::unique_ptr<ProcInstance>& instance : instantiated_procs_) {
    pieces.push_back(instance->ToString(indent_amount + 1));
  }
  return absl::StrJoin(pieces, "\n");
}

absl::Status ProcElaboration::BuildInstanceMaps(ProcInstance* proc_instance) {
  XLS_RET_CHECK(proc_instance->path().has_value());

  proc_instance_ptrs_.push_back(proc_instance);
  instances_of_proc_[proc_instance->proc()].push_back(proc_instance);

  proc_instances_by_path_[proc_instance->path().value()] = proc_instance;
  for (const std::unique_ptr<ChannelInstance>& channel_instance :
       proc_instance->channels()) {
    instances_of_channel_[channel_instance->channel].push_back(
        channel_instance.get());
    channel_instance_ptrs_.push_back(channel_instance.get());
  }

  for (const std::unique_ptr<ChannelInterface>& channel_interface :
       proc_instance->proc()->channel_interfaces()) {
    XLS_ASSIGN_OR_RETURN(
        ChannelInstance * channel_instance,
        proc_instance->GetChannelInstance(channel_interface->name()));
    channel_instances_by_path_[{std::string{channel_interface->name()},
                                proc_instance->path().value()}] =
        channel_instance;
    instances_of_channel_interface_[channel_interface.get()].push_back(
        channel_instance);
  }

  for (const std::unique_ptr<ProcInstance>& subinstance :
       proc_instance->instantiated_procs()) {
    XLS_RETURN_IF_ERROR(BuildInstanceMaps(subinstance.get()));
  }

  return absl::OkStatus();
}

/* static */ absl::StatusOr<ProcElaboration> ProcElaboration::Elaborate(
    Proc* top) {
  if (!top->is_new_style_proc()) {
    return absl::UnimplementedError(
        absl::StrFormat("Cannot elaborate old-style proc `%s`", top->name()));
  }

  ProcElaboration elaboration;
  elaboration.package_ = top->package();

  // Create top-level channels. These are required because there are no
  // xls::Channels in the IR corresponding to the ChannelInterfaces forming the
  // interface of `top`.
  int64_t channel_id = 0;
  std::vector<ChannelBinding> interface_bindings;
  ProcInstantiationPath path;
  path.top = top;
  for (ChannelInterface* channel_ref : top->interface()) {
    // TODO(https://github.com/google/xls/issues/869): Add options for
    // fifo-config, strictness, etc.
    ChannelOps ops = channel_ref->direction() == ChannelDirection::kSend
                         ? ChannelOps::kSendOnly
                         : ChannelOps::kReceiveOnly;
    if (channel_ref->kind() == ChannelKind::kStreaming) {
      elaboration.interface_channels_.push_back(
          std::make_unique<StreamingChannel>(
              channel_ref->name(), channel_id, ops, channel_ref->type(),
              /*intial_values=*/absl::Span<const Value>(),
              /*channel_config=*/ChannelConfig(),
              /*flow_control=*/FlowControl::kReadyValid,
              /*strictness=*/
              channel_ref->strictness().value_or(kDefaultChannelStrictness)));
    } else {
      XLS_RET_CHECK_EQ(channel_ref->kind(), ChannelKind::kSingleValue);
      elaboration.interface_channels_.push_back(
          std::make_unique<SingleValueChannel>(channel_ref->name(), channel_id,
                                               ops, channel_ref->type()));
    }
    ++channel_id;
    elaboration.interface_channel_instances_.push_back(
        std::make_unique<ChannelInstance>(ChannelInstance{
            .channel = elaboration.interface_channels_.back().get(),
            .path = std::nullopt}));
    ChannelInstance* channel_instance =
        elaboration.interface_channel_instances_.back().get();
    elaboration.interface_channel_instance_set_.insert(channel_instance);
    interface_bindings.push_back(ChannelBinding{
        .instance = channel_instance, .parent_interface = std::nullopt});
  }
  XLS_ASSIGN_OR_RETURN(
      elaboration.top_,
      CreateNewStyleProcInstance(top, /*proc_instantiation=*/std::nullopt, path,
                                 interface_bindings));

  for (const std::unique_ptr<ChannelInstance>& channel_instance :
       elaboration.interface_channel_instances_) {
    elaboration.channel_instance_ptrs_.push_back(channel_instance.get());
  }
  XLS_RETURN_IF_ERROR(elaboration.BuildInstanceMaps(elaboration.top_.get()));

  // Create the vector of procs which appear in this elaboration.
  absl::flat_hash_set<Proc*> proc_set;
  for (ProcInstance* proc_instance : elaboration.proc_instance_ptrs_) {
    if (!proc_set.contains(proc_instance->proc())) {
      proc_set.insert(proc_instance->proc());
      elaboration.procs_.push_back(proc_instance->proc());
    }
  }

  return elaboration;
}

std::string ProcElaboration::ToString() const {
  if (top_ != nullptr) {
    // New-style procs.
    return top()->ToString();
  }
  // Old-style procs.
  return absl::StrJoin(procs(), "\n", [](std::string* s, Proc* p) {
    absl::StrAppend(s, p->name());
  });
}

absl::StatusOr<ProcInstance*> ProcElaboration::GetProcInstance(
    const ProcInstantiationPath& path) const {
  auto it = proc_instances_by_path_.find(path);
  if (it == proc_instances_by_path_.end()) {
    return absl::NotFoundError(absl::StrFormat(
        "Instantiation path `%s` does not exist in elaboration from proc `%s`",
        path.ToString(), top()->proc()->name()));
  }
  return it->second;
}

absl::StatusOr<ProcInstance*> ProcElaboration::GetProcInstance(
    std::string_view path_str) const {
  XLS_ASSIGN_OR_RETURN(ProcInstantiationPath path, CreatePath(path_str));
  if (path.path.empty()) {
    return top();
  }
  return GetProcInstance(path);
}

absl::StatusOr<ChannelInstance*> ProcElaboration::GetChannelInstance(
    std::string_view channel_name, const ProcInstantiationPath& path) const {
  auto it = channel_instances_by_path_.find({std::string{channel_name}, path});
  if (it == channel_instances_by_path_.end()) {
    return absl::NotFoundError(
        absl::StrFormat("No channel `%s` at instantiation path `%s` in "
                        "elaboration from proc `%s`",
                        channel_name, path.ToString(), top()->proc()->name()));
  }
  return it->second;
}

absl::StatusOr<ChannelInstance*> ProcElaboration::GetChannelInstance(
    std::string_view channel_name, std::string_view path_str) const {
  XLS_ASSIGN_OR_RETURN(ProcInstantiationPath path, CreatePath(path_str));
  XLS_ASSIGN_OR_RETURN(ProcInstance * proc_instance, GetProcInstance(path));
  if (proc_instance->proc()->HasChannelInterface(channel_name,
                                                 ChannelDirection::kReceive)) {
    XLS_RETURN_IF_ERROR(
        proc_instance->proc()
            ->GetChannelInterface(channel_name, ChannelDirection::kReceive)
            .status());
  } else {
    XLS_RETURN_IF_ERROR(
        proc_instance->proc()
            ->GetChannelInterface(channel_name, ChannelDirection::kSend)
            .status());
  }
  return proc_instance->GetChannelInstance(channel_name);
}

/* static */ absl::StatusOr<ProcElaboration>
ProcElaboration::ElaborateOldStylePackage(Package* package) {
  // Iterate through every proc and channel and create a single instance for
  // each.
  ProcElaboration elaboration;
  elaboration.package_ = package;

  // All channels are available in all procs. Create a global map from channel
  // name to channel instance and pass it to the constructor of every proc
  // instance.
  absl::flat_hash_map<ChannelRef, ChannelBinding> channel_bindings;
  for (Channel* channel : package->channels()) {
    elaboration.channel_instances_.push_back(std::make_unique<ChannelInstance>(
        ChannelInstance{.channel = channel, .path = std::nullopt}));
    ChannelInstance* channel_instance =
        elaboration.channel_instances_.back().get();

    elaboration.channel_instance_ptrs_.push_back(channel_instance);
    elaboration.instances_of_channel_[channel] = {channel_instance};
    channel_bindings[channel] = ChannelBinding{
        .instance = channel_instance, .parent_interface = std::nullopt};
  }

  for (const std::unique_ptr<Proc>& proc : package->procs()) {
    XLS_RET_CHECK(!proc->is_new_style_proc());
    elaboration.proc_instances_.push_back(std::make_unique<ProcInstance>(
        proc.get(), /*proc_instantiation=*/std::nullopt,
        /*path=*/std::nullopt,
        /*channel_instances=*/std::vector<std::unique_ptr<ChannelInstance>>(),
        /*instantiated_procs=*/std::vector<std::unique_ptr<ProcInstance>>(),
        channel_bindings));
    elaboration.proc_instance_ptrs_.push_back(
        elaboration.proc_instances_.back().get());

    elaboration.instances_of_proc_[proc.get()] = {
        elaboration.proc_instance_ptrs_.back()};

    elaboration.procs_.push_back(proc.get());
  }

  return std::move(elaboration);
}

absl::Span<ProcInstance* const> ProcElaboration::GetInstances(
    Proc* proc) const {
  if (!instances_of_proc_.contains(proc)) {
    return {};
  }
  return instances_of_proc_.at(proc);
}

absl::Span<ChannelInstance* const> ProcElaboration::GetInstances(
    Channel* channel) const {
  if (!instances_of_channel_.contains(channel)) {
    return {};
  }
  return instances_of_channel_.at(channel);
}

absl::Span<ChannelInstance* const>
ProcElaboration::GetInstancesOfChannelInterface(
    ChannelInterface* channel_interface) const {
  if (!instances_of_channel_interface_.contains(channel_interface)) {
    return {};
  }
  return instances_of_channel_interface_.at(channel_interface);
}

bool ProcElaboration::IsTopInterfaceChannel(ChannelInstance* channel) const {
  return interface_channel_instance_set_.contains(channel);
}

absl::StatusOr<ProcInstance*> ProcElaboration::GetUniqueInstance(
    Proc* proc) const {
  absl::Span<ProcInstance* const> instances = GetInstances(proc);
  if (instances.size() != 1) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "There is not exactly 1 instance of proc `%s`, instance count: %d",
        proc->name(), instances.size()));
  }
  return instances.front();
}

absl::StatusOr<ChannelInstance*> ProcElaboration::GetUniqueInstance(
    Channel* channel) const {
  absl::Span<ChannelInstance* const> instances = GetInstances(channel);
  if (instances.size() != 1) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "There is not exactly 1 instance of channel `%s`, instance count: %d",
        channel->name(), instances.size()));
  }
  return instances.front();
}

absl::StatusOr<ProcInstantiationPath> ProcElaboration::CreatePath(
    std::string_view path_str) const {
  std::vector<std::string_view> pieces = absl::StrSplit(path_str, "::");
  if (pieces.front() != top()->proc()->name()) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Path top `%s` does not match name of top proc `%s`",
                        pieces.front(), top()->proc()->name()));
  }
  ProcInstantiationPath path;
  path.top = top()->proc();
  Proc* proc = path.top;
  for (std::string_view piece : absl::MakeSpan(pieces).subspan(1)) {
    std::vector<std::string_view> parts = absl::StrSplit(piece, "->");
    if (parts.size() != 2) {
      return absl::InvalidArgumentError(
          absl::StrFormat("Invalid component of path `%s`. Expected form: "
                          "`instantiation->proc`.",
                          piece));
    }
    absl::StatusOr<ProcInstantiation*> instantiation =
        proc->GetProcInstantiation(parts[0]);
    if (!instantiation.ok()) {
      return absl::InvalidArgumentError(
          absl::StrFormat("Proc `%s` does not have an instantiation named `%s`",
                          proc->name(), parts[0]));
    }
    if (parts[1] != (*instantiation)->proc()->name()) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "Instantiation `%s` in proc `%s` instantiates proc `%s`, but path "
          "element is `%s`",
          parts[0], proc->name(), (*instantiation)->proc()->name(), parts[1]));
    }
    path.path.push_back(*instantiation);
    proc = (*instantiation)->proc();
  }
  return path;
}

}  // namespace xls
