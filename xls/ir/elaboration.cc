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

#include "xls/ir/elaboration.h"

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
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"
#include "absl/types/span.h"
#include "xls/common/logging/logging.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/channel.h"
#include "xls/ir/channel_ops.h"
#include "xls/ir/package.h"
#include "xls/ir/proc.h"
#include "xls/ir/proc_instantiation.h"
#include "xls/ir/value.h"

namespace xls {

namespace {

absl::StatusOr<std::unique_ptr<ProcInstance>> CreateNewStyleProcInstance(
    Proc* proc, std::optional<ProcInstantiation*> proc_instantiation,
    const InstantiationPath& path,
    absl::Span<ChannelInstance* const> interface) {
  XLS_RET_CHECK(proc->is_new_style_proc());

  // Map from channel reference name to ChannelInstance.
  absl::flat_hash_map<std::string, ChannelInstance*> channel_instances;
  XLS_RET_CHECK_EQ(interface.size(), proc->interface().size());
  for (int64_t i = 0; i < interface.size(); ++i) {
    channel_instances[proc->interface()[i]->name()] = interface[i];
  }
  std::vector<std::unique_ptr<ChannelInstance>> declared_channels;
  for (Channel* channel : proc->channels()) {
    declared_channels.push_back(std::make_unique<ChannelInstance>(
        ChannelInstance{.channel = channel, .path = path}));
    channel_instances[channel->name()] = declared_channels.back().get();
  }

  std::vector<std::unique_ptr<ProcInstance>> instantiated_procs;
  for (const std::unique_ptr<ProcInstantiation>& instantiation :
       proc->proc_instantiations()) {
    InstantiationPath instantiation_path = path;
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

    std::vector<ChannelInstance*> instantiation_interface;
    for (ChannelReference* channel_ref : instantiation->channel_args()) {
      instantiation_interface.push_back(
          channel_instances.at(channel_ref->name()));
    }
    XLS_ASSIGN_OR_RETURN(std::unique_ptr<ProcInstance> instantiation_instance,
                         CreateNewStyleProcInstance(
                             instantiation->proc(), instantiation.get(),
                             instantiation_path, instantiation_interface));
    instantiated_procs.push_back(std::move(instantiation_instance));
  }
  return std::make_unique<ProcInstance>(proc, proc_instantiation, path,
                                        interface, std::move(declared_channels),
                                        std::move(instantiated_procs));
}

}  // namespace

std::string ChannelInstance::ToString() const {
  if (path.has_value()) {
    return absl::StrFormat("%s [%s]", channel->name(), path->ToString());
  }
  return std::string{channel->name()};
}

absl::StatusOr<ChannelInstance*> ProcInstance::GetChannelInstance(
    std::string_view channel_name) const {
  for (const std::unique_ptr<ChannelInstance>& channel : channels_) {
    if (channel->channel->name() == channel_name) {
      return channel.get();
    }
  }
  for (ChannelInstance* channel : interface_) {
    if (channel->channel->name() == channel_name) {
      return channel;
    }
  }
  return absl::NotFoundError(absl::StrFormat(
      "No channel instance with name `%s` in instance of proc `%s`",
      channel_name, proc()->name()));
}

std::string ProcInstance::GetName() const {
  if (!path().has_value()) {
    return proc()->name();
  }
  return absl::StrFormat("%s [%s]", proc()->name(), path()->ToString());
}

std::string ProcInstance::ToString(int64_t indent_amount) const {
  auto indent = [&](std::string_view s, int64_t additional_indent = 0) {
    std::string indent_str(2 * (indent_amount + additional_indent), ' ');
    return absl::StrFormat("%s%s", indent_str, s);
  };
  std::vector<std::string> pieces;
  std::string suffix =
      proc_instantiation().has_value()
          ? absl::StrFormat(" [%s]", (*proc_instantiation())->name())
          : "";
  pieces.push_back(indent(
      absl::StrFormat("%s<%s>%s", proc()->name(),
                      absl::StrJoin(proc()->interface(), ", ",
                                    [](std::string* s, ChannelReference* c) {
                                      absl::StrAppendFormat(s, "%s", c->name());
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

void Elaboration::BuildInstanceMaps(ProcInstance* proc_instance) {
  XLS_CHECK(proc_instance->path().has_value());

  proc_instance_ptrs_.push_back(proc_instance);
  instances_of_proc_[proc_instance->proc()].push_back(proc_instance);

  proc_instances_by_path_[proc_instance->path().value()] = proc_instance;
  for (const std::unique_ptr<ChannelInstance>& channel_instance :
       proc_instance->channels()) {
    channel_instances_by_path_[{std::string{channel_instance->channel->name()},
                                proc_instance->path().value()}] =
        channel_instance.get();
    instances_of_channel_[channel_instance->channel].push_back(
        channel_instance.get());
    channel_instance_ptrs_.push_back(channel_instance.get());
  }

  for (const std::unique_ptr<ProcInstance>& subinstance :
       proc_instance->instantiated_procs()) {
    BuildInstanceMaps(subinstance.get());
  }

  XLS_CHECK_EQ(proc_instance->interface().size(),
               proc_instance->proc()->interface().size());
  for (int64_t i = 0; i < proc_instance->interface().size(); ++i) {
    instances_of_channel_reference_[proc_instance->proc()->interface()[i]]
        .push_back(proc_instance->interface()[i]);
  }
}

/*static*/
absl::StatusOr<Elaboration> Elaboration::Elaborate(Proc* top) {
  if (!top->is_new_style_proc()) {
    return absl::UnimplementedError(
        absl::StrFormat("Cannot elaborate old-style proc `%s`", top->name()));
  }

  Elaboration elaboration;
  elaboration.package_ = top->package();

  // Create top-level channels. These are required because there are no
  // xls::Channels in the IR corresponding to the ChannelReferences forming the
  // interface of `top`.
  int64_t channel_id = 0;
  std::vector<ChannelInstance*> channel_instance_ptrs;
  InstantiationPath path;
  path.top = top;
  for (ChannelReference* channel_ref : top->interface()) {
    // TODO(https://github.com/google/xls/issues/869): Add options for
    // fifo-config, strictness, etc.
    elaboration.interface_channels_.push_back(
        std::make_unique<StreamingChannel>(
            channel_ref->name(), channel_id,
            channel_ref->direction() == Direction::kSend
                ? ChannelOps::kSendOnly
                : ChannelOps::kReceiveOnly,
            channel_ref->type(), /*intial_values=*/absl::Span<const Value>(),
            /*fifo_config=*/std::nullopt,
            /*flow_control=*/FlowControl::kReadyValid,
            /*strictness=*/ChannelStrictness::kProvenMutuallyExclusive,
            ChannelMetadataProto()));
    ++channel_id;
    elaboration.interface_channel_instances_.push_back(
        std::make_unique<ChannelInstance>(ChannelInstance{
            .channel = elaboration.interface_channels_.back().get(),
            .path = std::nullopt}));
    channel_instance_ptrs.push_back(
        elaboration.interface_channel_instances_.back().get());
  }
  XLS_ASSIGN_OR_RETURN(
      elaboration.top_,
      CreateNewStyleProcInstance(top, /*proc_instantiation=*/std::nullopt, path,
                                 channel_instance_ptrs));

  for (const std::unique_ptr<ChannelInstance>& channel_instance :
       elaboration.interface_channel_instances_) {
    elaboration.channel_instance_ptrs_.push_back(channel_instance.get());
  }
  elaboration.BuildInstanceMaps(elaboration.top_.get());

  return elaboration;
}

std::string Elaboration::ToString() const { return top()->ToString(); }

absl::StatusOr<ProcInstance*> Elaboration::GetProcInstance(
    const InstantiationPath& path) const {
  auto it = proc_instances_by_path_.find(path);
  if (it == proc_instances_by_path_.end()) {
    return absl::NotFoundError(absl::StrFormat(
        "Instantiation path `%s` does not exist in elaboration from proc `%s`",
        path.ToString(), top()->proc()->name()));
  }
  return it->second;
}

absl::StatusOr<ProcInstance*> Elaboration::GetProcInstance(
    std::string_view path_str) const {
  XLS_ASSIGN_OR_RETURN(InstantiationPath path, CreatePath(path_str));
  if (path.path.empty()) {
    return top();
  }
  return GetProcInstance(path);
}

absl::StatusOr<ChannelInstance*> Elaboration::GetChannelInstance(
    std::string_view channel_name, const InstantiationPath& path) const {
  auto it = channel_instances_by_path_.find({std::string{channel_name}, path});
  if (it == channel_instances_by_path_.end()) {
    return absl::NotFoundError(
        absl::StrFormat("No channel `%s` at instantiation path `%s` in "
                        "elaboration from proc `%s`",
                        channel_name, path.ToString(), top()->proc()->name()));
  }
  return it->second;
}

absl::StatusOr<ChannelInstance*> Elaboration::GetChannelInstance(
    std::string_view channel_name, std::string_view path_str) const {
  XLS_ASSIGN_OR_RETURN(InstantiationPath path, CreatePath(path_str));
  XLS_ASSIGN_OR_RETURN(ProcInstance * proc_instance, GetProcInstance(path));
  return proc_instance->GetChannelInstance(channel_name);
}

std::string InstantiationPath::ToString() const {
  if (path.empty()) {
    return top->name();
  }
  return absl::StrFormat(
      "%s::%s", top->name(),
      absl::StrJoin(
          path, "::", [](std::string* s, const ProcInstantiation* pi) {
            absl::StrAppendFormat(s, "%s->%s", pi->name(), pi->proc()->name());
          }));
}

/*static*/ absl::StatusOr<Elaboration> Elaboration::ElaborateOldStylePackage(
    Package* package) {
  // Iterate through every proc and channel and create a single instance for
  // each.
  Elaboration elaboration;
  elaboration.package_ = package;
  for (const std::unique_ptr<Proc>& proc : package->procs()) {
    XLS_RET_CHECK(!proc->is_new_style_proc());
    elaboration.proc_instances_.push_back(std::make_unique<ProcInstance>(
        proc.get(), /*proc_instantiation=*/std::nullopt,
        /*path=*/std::nullopt,
        /*interface=*/absl::Span<ChannelInstance* const>(),
        /*channel_instances=*/std::vector<std::unique_ptr<ChannelInstance>>(),
        /*instantiated_procs=*/std::vector<std::unique_ptr<ProcInstance>>()));
    elaboration.proc_instance_ptrs_.push_back(
        elaboration.proc_instances_.back().get());

    elaboration.instances_of_proc_[proc.get()] = {
        elaboration.proc_instance_ptrs_.back()};
  }
  for (Channel* channel : package->channels()) {
    elaboration.channel_instances_.push_back(std::make_unique<ChannelInstance>(
        ChannelInstance{.channel = channel, .path = std::nullopt}));
    elaboration.channel_instance_ptrs_.push_back(
        elaboration.channel_instances_.back().get());

    elaboration.instances_of_channel_[channel] = {
        elaboration.channel_instance_ptrs_.back()};
  }
  return std::move(elaboration);
}

absl::Span<ProcInstance* const> Elaboration::GetInstances(Proc* proc) const {
  if (!instances_of_proc_.contains(proc)) {
    return {};
  }
  return instances_of_proc_.at(proc);
}

absl::Span<ChannelInstance* const> Elaboration::GetInstances(
    Channel* channel) const {
  if (!instances_of_channel_.contains(channel)) {
    return {};
  }
  return instances_of_channel_.at(channel);
}

absl::Span<ChannelInstance* const> Elaboration::GetInstancesOfChannelReference(
    ChannelReference* channel_reference) const {
  if (!instances_of_channel_reference_.contains(channel_reference)) {
    return {};
  }
  return instances_of_channel_reference_.at(channel_reference);
}

absl::StatusOr<ProcInstance*> Elaboration::GetUniqueInstance(Proc* proc) const {
  absl::Span<ProcInstance* const> instances = GetInstances(proc);
  if (instances.size() != 1) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "There is not exactly 1 instance of proc `%s`, instance count: %d",
        proc->name(), instances.size()));
  }
  return instances.front();
}

absl::StatusOr<ChannelInstance*> Elaboration::GetUniqueInstance(
    Channel* channel) const {
  absl::Span<ChannelInstance* const> instances = GetInstances(channel);
  if (instances.size() != 1) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "There is not exactly 1 instance of channel `%s`, instance count: %d",
        channel->name(), instances.size()));
  }
  return instances.front();
}

absl::StatusOr<InstantiationPath> Elaboration::CreatePath(
    std::string_view path_str) const {
  std::vector<std::string_view> pieces = absl::StrSplit(path_str, "::");
  if (pieces.front() != top()->proc()->name()) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Path top `%s` does not match name of top proc `%s`",
                        pieces.front(), top()->proc()->name()));
  }
  InstantiationPath path;
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
