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
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/channel.h"
#include "xls/ir/channel_ops.h"
#include "xls/ir/proc.h"
#include "xls/ir/proc_instantiation.h"
#include "xls/ir/value.h"

namespace xls {

/*static*/
absl::StatusOr<std::unique_ptr<ProcInstance>> ProcInstance::Create(
    Proc* proc, std::optional<ProcInstantiation*> proc_instantiation,
    const InstantiationPath& path,
    absl::Span<ChannelInstance* const> interface) {
  XLS_RET_CHECK(proc->is_new_style_proc());
  auto instance = absl::WrapUnique(
      new ProcInstance(proc, proc_instantiation, path, interface));

  // Map from channel reference name to ChannelInstance.
  absl::flat_hash_map<std::string, ChannelInstance*> channel_instances;
  XLS_RET_CHECK_EQ(interface.size(), proc->interface().size());
  for (int64_t i = 0; i < interface.size(); ++i) {
    channel_instances[proc->interface()[i]->name()] = interface[i];
  }
  for (Channel* channel : proc->channels()) {
    instance->channels_.push_back(std::make_unique<ChannelInstance>(
        ChannelInstance{.channel = channel, .path = path}));
    channel_instances[channel->name()] = instance->channels_.back().get();
  }
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
                         Create(instantiation->proc(), instantiation.get(),
                                instantiation_path, instantiation_interface));
    instance->instantiated_procs_.push_back(std::move(instantiation_instance));
  }

  return std::move(instance);
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

static void BuildInstanceMaps(
    ProcInstance* proc_instance,
    absl::flat_hash_map<InstantiationPath, ProcInstance*>& proc_map,
    absl::flat_hash_map<std::pair<std::string, InstantiationPath>,
                        ChannelInstance*>& channel_map) {
  proc_map[proc_instance->path()] = proc_instance;
  for (const std::unique_ptr<ChannelInstance>& channel_instance :
       proc_instance->channels()) {
    channel_map[{std::string{channel_instance->channel->name()},
                 proc_instance->path()}] = channel_instance.get();
  }
  for (const std::unique_ptr<ProcInstance>& subinstance :
       proc_instance->instantiated_procs()) {
    BuildInstanceMaps(subinstance.get(), proc_map, channel_map);
  }
}

/*static*/
absl::StatusOr<Elaboration> Elaboration::Elaborate(Proc* top) {
  if (!top->is_new_style_proc()) {
    return absl::UnimplementedError(
        absl::StrFormat("Cannot elaborate old-style proc `%s`", top->name()));
  }

  Elaboration elaboration;
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
            .path = path}));
    channel_instance_ptrs.push_back(
        elaboration.interface_channel_instances_.back().get());
  }
  XLS_ASSIGN_OR_RETURN(
      elaboration.top_,
      ProcInstance::Create(top, /*proc_instantiation=*/std::nullopt, path,
                           channel_instance_ptrs));
  BuildInstanceMaps(elaboration.top_.get(), elaboration.proc_instances_by_path_,
                    elaboration.channel_instances_by_path_);
  return elaboration;
}

std::string Elaboration::ToString() const { return top().ToString(); }

absl::StatusOr<ProcInstance*> Elaboration::GetProcInstance(
    const InstantiationPath& path) const {
  auto it = proc_instances_by_path_.find(path);
  if (it == proc_instances_by_path_.end()) {
    return absl::NotFoundError(absl::StrFormat(
        "Instantiation path `%s` does not exist in elaboration from proc `%s`",
        path.ToString(), top().proc()->name()));
  }
  return it->second;
}

absl::StatusOr<ChannelInstance*> Elaboration::GetChannelInstance(
    std::string_view channel_name, const InstantiationPath& path) const {
  auto it = channel_instances_by_path_.find({std::string{channel_name}, path});
  if (it == channel_instances_by_path_.end()) {
    return absl::NotFoundError(
        absl::StrFormat("No channel `%s` at instantiation path `%s` in "
                        "elaboration from proc `%s`",
                        channel_name, path.ToString(), top().proc()->name()));
  }
  return it->second;
}

std::string InstantiationPath::ToString() const {
  if (path.empty()) {
    return top->name();
  }
  return absl::StrFormat(
      "%s::%s", top->name(),
      absl::StrJoin(
          path, "->", [](std::string* s, const ProcInstantiation* pi) {
            absl::StrAppendFormat(s, "%s::%s", pi->name(), pi->proc()->name());
          }));
}

}  // namespace xls
