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

#include "xls/ir/verifier.h"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "xls/common/casts.h"
#include "xls/common/logging/log_lines.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/block.h"
#include "xls/ir/block_elaboration.h"
#include "xls/ir/channel.h"
#include "xls/ir/code_template.h"
#include "xls/ir/dfs_visitor.h"
#include "xls/ir/elaborated_block_dfs_visitor.h"
#include "xls/ir/function.h"
#include "xls/ir/function_base.h"
#include "xls/ir/instantiation.h"
#include "xls/ir/ir_scanner.h"
#include "xls/ir/node.h"
#include "xls/ir/node_util.h"
#include "xls/ir/nodes.h"
#include "xls/ir/op.h"
#include "xls/ir/package.h"
#include "xls/ir/proc.h"
#include "xls/ir/proc_elaboration.h"
#include "xls/ir/proc_instantiation.h"
#include "xls/ir/register.h"
#include "xls/ir/source_location.h"
#include "xls/ir/type.h"
#include "xls/ir/verify_node.h"
#include "re2/re2.h"

namespace xls {
namespace {

using ::absl::StrFormat;

absl::Status VerifyNodeIdUnique(Node* node, std::vector<bool>* ids_seen,
                                int64_t* max_id_seen) {
  const int64_t id = node->id();
  if (id >= ids_seen->size()) {
    return absl::InternalError(
        absl::StrFormat("ID %d larger than expected max id in package", id));
  }
  *max_id_seen = std::max(id, *max_id_seen);

  if ((*ids_seen)[id]) {
    // Find locations of all nodes in the package with this node ID for error
    // message.
    std::vector<std::string> location_strings;
    for (FunctionBase* f : node->package()->GetFunctionBases()) {
      for (Node* n : f->nodes()) {
        if (n->id() == node->id()) {
          location_strings.push_back(n->loc().ToString());
        }
      }
    }
    return absl::InternalError(absl::StrFormat(
        "ID %d is not unique; source locations of nodes with same id:\n%s",
        node->id(), absl::StrJoin(location_strings, ", ")));
  }
  (*ids_seen)[id] = true;
  return absl::OkStatus();
}

absl::Status VerifyName(FunctionBase* function_base) {
  if (Token::GetKeywords().contains(function_base->name())) {
    return absl::InternalError(absl::StrFormat(
        "Function/proc/block name '%s' is a keyword", function_base->name()));
  }
  return absl::OkStatus();
}

// Verify common invariants to function-level constructs.
absl::Status VerifyFunctionBase(FunctionBase* function) {
  VLOG(2) << absl::StreamFormat("Verifying function %s:", function->name());
  XLS_VLOG_LINES(4, function->DumpIr());

  XLS_RETURN_IF_ERROR(VerifyName(function));

  // Verify all types are owned by package.
  for (Node* node : function->nodes()) {
    XLS_RET_CHECK(node->package()->IsOwnedType(node->GetType()));
    XLS_RET_CHECK(node->package() == function->package());
  }

  // Verify ids are unique within the function.
  std::vector<bool> ids_seen(function->package()->next_node_id());
  int64_t max_id_seen = -1;
  for (Node* node : function->nodes()) {
    XLS_RETURN_IF_ERROR(VerifyNodeIdUnique(node, &ids_seen, &max_id_seen));
  }

  // Verify that there are no cycles in the node graph.
  class CycleChecker : public DfsVisitorWithDefault {
    absl::Status DefaultHandler(Node* node) override {
      return absl::OkStatus();
    }
  };
  CycleChecker cycle_checker;
  XLS_RETURN_IF_ERROR(function->Accept(&cycle_checker));

  // Verify consistency of node::users() and node::operands().
  for (Node* node : function->nodes()) {
    XLS_RETURN_IF_ERROR(VerifyNode(node));
  }

  // Verify the set of parameter nodes is exactly Function::params(), and that
  // the parameter names are unique.
  absl::flat_hash_set<std::string> param_names;
  absl::flat_hash_set<Node*> param_set;
  for (Node* param : function->params()) {
    XLS_RET_CHECK(param_set.insert(param).second)
        << "Param appears more than once in Function::params()";
    XLS_RET_CHECK(param_names.insert(param->GetName()).second)
        << "Param name " << param->GetName()
        << " is duplicated in Function::params()";
  }
  int64_t param_node_count = 0;
  for (Node* node : function->nodes()) {
    if (node->Is<Param>()) {
      XLS_RET_CHECK(param_set.contains(node))
          << "Param " << node->GetName() << " is not in Function::params()";
      param_node_count++;
    }
  }
  XLS_RET_CHECK_EQ(param_set.size(), param_node_count)
      << "Number of param nodes not equal to Function::params() size for "
         "function "
      << function->name();

  return absl::OkStatus();
}

// Verify various invariants about the channels owned by the given package.
absl::Status VerifyChannels(
    Package* package, bool codegen,
    std::function<std::vector<Node*>(FunctionBase*)> topo_sort) {
  // All channels must be proc-scoped or no channels should be proc scoped.
  bool has_global_channels = !package->channels().empty();
  bool has_new_style_procs = false;
  bool has_old_style_procs = false;
  for (const std::unique_ptr<Proc>& proc : package->procs()) {
    if (proc->is_new_style_proc()) {
      has_new_style_procs = true;
    } else {
      has_old_style_procs = true;
    }
  }
  if (has_global_channels && has_new_style_procs) {
    return absl::InternalError(
        "Package has global channels and procs with proc-scoped channels.");
  }
  if (has_new_style_procs && has_old_style_procs) {
    return absl::InternalError(
        "Package has both new style procs (proc-scoped channels) and old-style "
        "procs.");
  }

  // Verify unique ids.
  absl::flat_hash_map<int64_t, Channel*> channels_by_id;
  for (Channel* channel : package->channels()) {
    XLS_RET_CHECK(!channels_by_id.contains(channel->id()))
        << absl::StreamFormat("More than one channel has id %d: '%s' and '%s'",
                              channel->id(), channel->name(),
                              channels_by_id.at(channel->id())->name());
    channels_by_id[channel->id()] = channel;
  }

  // Verify unique names.
  absl::flat_hash_map<std::string, Channel*> channels_by_name;
  for (Channel* channel : package->channels()) {
    XLS_RET_CHECK(!channels_by_name.contains(channel->name()))
        << absl::StreamFormat(
               "More than one channel has name '%s'. IDs of channels: %d and "
               "%d",
               channel->name(), channel->id(),
               channels_by_name.at(channel->name())->id());
    channels_by_name[channel->name()] = channel;
  }

  // Verify each package-scoped channel has the appropriate send/receive node.
  absl::flat_hash_map<Channel*, std::vector<Node*>> send_nodes;
  absl::flat_hash_map<Channel*, std::vector<Node*>> receive_nodes;
  for (auto& proc : package->procs()) {
    if (proc->is_new_style_proc()) {
      continue;
    }
    for (Node* node : topo_sort(proc.get())) {
      if (node->Is<Send>()) {
        XLS_ASSIGN_OR_RETURN(Channel * channel, GetChannelUsedByNode(node));
        send_nodes[channel].push_back(node);
      }
      if (node->Is<Receive>()) {
        XLS_ASSIGN_OR_RETURN(Channel * channel, GetChannelUsedByNode(node));
        receive_nodes[channel].push_back(node);
      }
    }
  }

  // Verify that each channel has the appropriate number of send and receive
  // nodes (one or zero).
  for (Channel* channel : package->channels()) {
    if (channel->CanSend()) {
      XLS_RET_CHECK(send_nodes.contains(channel)) << absl::StreamFormat(
          "Channel '%s' (id %d) has no associated send node", channel->name(),
          channel->id());
    } else {
      XLS_RET_CHECK(!send_nodes.contains(channel)) << absl::StreamFormat(
          "Channel '%s' (id %d) cannot send but has send node(s): %s",
          channel->name(), channel->id(),
          absl::StrJoin(send_nodes.at(channel), ", "));
    }
    if (channel->CanReceive()) {
      XLS_RET_CHECK(receive_nodes.contains(channel)) << absl::StreamFormat(
          "Channel '%s' (id %d) has no associated receive node",
          channel->name(), channel->id());
    } else {
      XLS_RET_CHECK(!receive_nodes.contains(channel)) << absl::StreamFormat(
          "Channel '%s' (id %d) cannot receive but has a receive node(s): %s",
          channel->name(), channel->id(),
          absl::StrJoin(receive_nodes.at(channel), ", "));
    }

    // Verify type-specific invariants of each channel.
    if (channel->kind() == ChannelKind::kSingleValue) {
      // Single-value channels cannot have initial values.
      XLS_RET_CHECK_EQ(channel->initial_values().size(), 0);
      // TODO(meheff): 2021/06/24 Single-value channels should not support
      // Send and Receive with predicates. Add check when such uses are removed.
    }
  }

  return absl::OkStatus();
}

// Verify various global properties about the proc elaboration (for new-style
// procs).
absl::Status VerifyElaboration(Package* package) {
  std::optional<FunctionBase*> top = package->GetTop();
  if (!top.has_value() || !(*top)->IsProc()) {
    return absl::OkStatus();
  }
  Proc* top_proc = (*top)->AsProcOrDie();
  if (!top_proc->is_new_style_proc()) {
    return absl::OkStatus();
  }
  XLS_ASSIGN_OR_RETURN(ProcElaboration elab,
                       ProcElaboration::Elaborate(top_proc));

  for (Proc* proc : elab.procs()) {
    for (const std::unique_ptr<ChannelInterface>& channel_interface :
         proc->channel_interfaces()) {
      std::optional<Channel*> channel;
      for (ProcInstance* proc_instance : elab.GetInstances(proc)) {
        ChannelBinding binding =
            proc_instance->GetChannelBinding(channel_interface.get());
        Channel* bound_channel = binding.instance->channel;

        // Verify ChannelInterface properties match respective Channel
        // properties.
        XLS_RET_CHECK_EQ(channel_interface->type(), bound_channel->type())
            << absl::StreamFormat(
                   "Type of ChannelInterface `%s` in proc `%s` does not match "
                   "the type of channel `%s` to which it is bound: %s vs %s",
                   channel_interface->name(), proc->name(),
                   bound_channel->name(), channel_interface->type()->ToString(),
                   bound_channel->type()->ToString());
        XLS_RET_CHECK_EQ(channel_interface->kind(), bound_channel->kind())
            << absl::StreamFormat(
                   "Kind of ChannelInterface `%s` in proc `%s` does not match "
                   "the kind of channel `%s` to which it is bound: %s vs %s",
                   channel_interface->name(), proc->name(),
                   bound_channel->name(),
                   ChannelKindToString(channel_interface->kind()),
                   ChannelKindToString(bound_channel->kind()));
        if (auto streaming_bound_channel =
                dynamic_cast<StreamingChannel*>(bound_channel)) {
          XLS_RET_CHECK(channel_interface->strictness().has_value())
              << absl::StreamFormat(
                     "ChannelInterface `%s` in proc `%s` is streaming but has "
                     "no strictness value",
                     channel_interface->name(), proc->name());
          XLS_RET_CHECK_EQ(channel_interface->strictness().value(),
                           streaming_bound_channel->GetStrictness())
              << absl::StreamFormat(
                     "Strictness of ChannelInterface `%s` in proc `%s` does "
                     "not match  the strictness of channel `%s` to which it is "
                     "bound: %s vs %s",
                     channel_interface->name(), proc->name(),
                     bound_channel->name(),
                     ChannelStrictnessToString(
                         channel_interface->strictness().value()),
                     ChannelStrictnessToString(
                         streaming_bound_channel->GetStrictness()));

        } else {
          XLS_RET_CHECK(!channel_interface->strictness().has_value())
              << absl::StreamFormat(
                     "ChannelInterface `%s` in proc `%s` is not streaming but "
                     "has  a strictness value",
                     channel_interface->name(), proc->name());
        }

        XLS_RET_CHECK_EQ(channel_interface->kind(), bound_channel->kind())
            << absl::StreamFormat(
                   "Kind of ChannelInterface `%s` in proc `%s` does not match "
                   "the kind of channel `%s` to which it is bound: %s vs %s",
                   channel_interface->name(), proc->name(),
                   bound_channel->name(),
                   ChannelKindToString(channel_interface->kind()),
                   ChannelKindToString(bound_channel->kind()));

        if (!channel.has_value()) {
          channel = bound_channel;
          continue;
        }
        // Verify that all Channels this reference is bound to have matching
        // properties.
        XLS_RET_CHECK_EQ(bound_channel->type(), (*channel)->type())
            << absl::StreamFormat(
                   "ChannelInterface `%s` in proc `%s` is bound to a channel "
                   "with a different type: %s vs %s",
                   channel_interface->name(), proc->name(),
                   channel_interface->type()->ToString(),
                   bound_channel->type()->ToString());
        XLS_RET_CHECK_EQ(bound_channel->kind(), (*channel)->kind())
            << absl::StreamFormat(
                   "ChannelInterface `%s` in proc `%s` is bound to a channel "
                   "with a different kind: %s vs %s",
                   channel_interface->name(), proc->name(),
                   ChannelKindToString(channel_interface->kind()),
                   ChannelKindToString(bound_channel->kind()));

        if ((*channel)->kind() == ChannelKind::kStreaming) {
          StreamingChannel* streaming_channel =
              dynamic_cast<StreamingChannel*>(*channel);
          StreamingChannel* streaming_bound_channel =
              dynamic_cast<StreamingChannel*>(bound_channel);
          XLS_RET_CHECK_EQ(streaming_channel->GetFlowControl(),
                           streaming_bound_channel->GetFlowControl())
              << absl::StreamFormat(
                     "ChannelInterface `%s` in proc `%s` bound to channels "
                     "with different flow control",
                     channel_interface->name(), proc->name());

          XLS_RET_CHECK_EQ(streaming_channel->GetStrictness(),
                           streaming_bound_channel->GetStrictness())
              << absl::StreamFormat(
                     "ChannelInterface `%s` in proc `%s` bound to channels "
                     "with different strictness",
                     channel_interface->name(), proc->name());
        }
      }
    }
  }

  return absl::OkStatus();
}

absl::Status VerifyBlockChannelMetadata(Block* block) {
  for (const auto& [channel, direction] : block->GetChannelsWithMappedPorts()) {
    XLS_ASSIGN_OR_RETURN(ChannelPortMetadata metadata,
                         block->GetChannelPortMetadata(channel, direction));
    if (metadata.direction == ChannelDirection::kReceive) {
      if (metadata.data_port.has_value()) {
        XLS_RET_CHECK(block->HasInputPort(*metadata.data_port))
            << absl::StrFormat(
                   "Data port `%s` for channel `%s` not found on block `%s`",
                   *metadata.data_port, metadata.channel_name, block->name());
        XLS_ASSIGN_OR_RETURN(InputPort * input_port,
                             block->GetInputPort(*metadata.data_port));
        XLS_ASSIGN_OR_RETURN(std::optional<PortNode*> data_port,
                             block->GetDataPortForChannel(metadata.channel_name,
                                                          metadata.direction));
        XLS_RET_CHECK(data_port.has_value());
        XLS_RET_CHECK_EQ(input_port, *data_port);
      }
      if (metadata.valid_port.has_value()) {
        XLS_RET_CHECK(block->HasInputPort(*metadata.valid_port))
            << absl::StrFormat(
                   "Valid port `%s` for channel `%s` not found on block `%s`",
                   *metadata.valid_port, metadata.channel_name, block->name());
        XLS_ASSIGN_OR_RETURN(InputPort * input_port,
                             block->GetInputPort(*metadata.valid_port));
        XLS_ASSIGN_OR_RETURN(std::optional<PortNode*> valid_port,
                             block->GetValidPortForChannel(
                                 metadata.channel_name, metadata.direction));
        XLS_RET_CHECK(valid_port.has_value());
        XLS_RET_CHECK_EQ(input_port, *valid_port);
      }
      if (metadata.ready_port.has_value()) {
        XLS_RET_CHECK(block->HasOutputPort(*metadata.ready_port))
            << absl::StrFormat(
                   "Ready port `%s` for channel `%s` not found on block `%s`",
                   *metadata.ready_port, metadata.channel_name, block->name());
        XLS_ASSIGN_OR_RETURN(OutputPort * output_port,
                             block->GetOutputPort(*metadata.ready_port));
        XLS_ASSIGN_OR_RETURN(std::optional<PortNode*> ready_port,
                             block->GetReadyPortForChannel(
                                 metadata.channel_name, metadata.direction));
        XLS_RET_CHECK(ready_port.has_value());
        XLS_RET_CHECK_EQ(output_port, *ready_port);
      }

    } else {
      XLS_RET_CHECK_EQ(metadata.direction, ChannelDirection::kSend);
      if (metadata.data_port.has_value()) {
        XLS_RET_CHECK(block->HasOutputPort(*metadata.data_port))
            << absl::StrFormat(
                   "Data port `%s` for channel `%s` not found on block `%s`",
                   *metadata.data_port, metadata.channel_name, block->name());
        XLS_ASSIGN_OR_RETURN(OutputPort * output_port,
                             block->GetOutputPort(*metadata.data_port));
        XLS_ASSIGN_OR_RETURN(std::optional<PortNode*> data_port,
                             block->GetDataPortForChannel(metadata.channel_name,
                                                          metadata.direction));
        XLS_RET_CHECK(data_port.has_value());
        XLS_RET_CHECK_EQ(output_port, *data_port);
      }
      if (metadata.valid_port.has_value()) {
        XLS_RET_CHECK(block->HasOutputPort(*metadata.valid_port))
            << absl::StrFormat(
                   "Valid port `%s` for channel `%s` not found on block `%s`",
                   *metadata.valid_port, metadata.channel_name, block->name());
        XLS_ASSIGN_OR_RETURN(OutputPort * output_port,
                             block->GetOutputPort(*metadata.valid_port));
        XLS_ASSIGN_OR_RETURN(std::optional<PortNode*> valid_port,
                             block->GetValidPortForChannel(
                                 metadata.channel_name, metadata.direction));
        XLS_RET_CHECK(valid_port.has_value());
        XLS_RET_CHECK_EQ(output_port, *valid_port);
      }
      if (metadata.ready_port.has_value()) {
        XLS_RET_CHECK(block->HasInputPort(*metadata.ready_port))
            << absl::StrFormat(
                   "Ready port `%s` for channel `%s` not found on block `%s`",
                   *metadata.ready_port, metadata.channel_name, block->name());
        XLS_ASSIGN_OR_RETURN(InputPort * input_port,
                             block->GetInputPort(*metadata.ready_port));
        XLS_ASSIGN_OR_RETURN(std::optional<PortNode*> ready_port,
                             block->GetReadyPortForChannel(
                                 metadata.channel_name, metadata.direction));
        XLS_RET_CHECK(ready_port.has_value());
        XLS_RET_CHECK_EQ(input_port, *ready_port);
      }
    }
  }
  return absl::OkStatus();
}

// Verify that no two blocks have the same provenance.
absl::Status VerifyBlockProvenance(Package* package) {
  absl::flat_hash_set<BlockProvenance> provenances;
  for (const auto& block : package->blocks()) {
    if (block->GetProvenance().has_value()) {
      auto [_, inserted] = provenances.insert(*block->GetProvenance());
      if (!inserted) {
        XLS_RET_CHECK(inserted) << "Multiple blocks have the same provenance: "
                                << absl::StrCat(*block->GetProvenance());
      }
    }
  }
  return absl::OkStatus();
}

}  // namespace

absl::Status VerifyPackage(
    Package* package, bool codegen,
    std::function<std::vector<Node*>(FunctionBase*)> topo_sort) {
  VLOG(4) << absl::StreamFormat("Verifying package %s:\n", package->name());
  XLS_VLOG_LINES(4, package->DumpIr());

  for (auto& function : package->functions()) {
    XLS_RETURN_IF_ERROR(VerifyFunction(function.get(), codegen));
  }

  for (auto& proc : package->procs()) {
    XLS_RETURN_IF_ERROR(VerifyProc(proc.get(), codegen));
  }

  for (auto& block : package->blocks()) {
    XLS_RETURN_IF_ERROR(VerifyBlock(block.get(), codegen));
  }

  // Verify node IDs are unique within the package and uplinks point to this
  // package.
  std::vector<bool> ids_seen(package->next_node_id());
  int64_t max_id_seen = -1;
  for (FunctionBase* function : package->GetFunctionBases()) {
    XLS_RET_CHECK(function->package() == package);
    for (Node* node : function->nodes()) {
      XLS_RETURN_IF_ERROR(VerifyNodeIdUnique(node, &ids_seen, &max_id_seen));
      XLS_RET_CHECK(node->package() == package);
    }
  }

  // Ensure that the package's "next ID" is not in the space of IDs currently
  // occupied by the package's nodes.
  XLS_RET_CHECK_GT(package->next_node_id(), max_id_seen);

  // Verify function, proc, block names are unique among functions/procs/blocks.
  absl::flat_hash_set<FunctionBase*> function_bases;
  absl::flat_hash_set<std::string> function_names;
  absl::flat_hash_set<std::string> proc_names;
  absl::flat_hash_set<std::string> block_names;
  for (FunctionBase* function_base : package->GetFunctionBases()) {
    absl::flat_hash_set<std::string>* name_set;
    if (function_base->IsFunction()) {
      name_set = &function_names;
    } else if (function_base->IsProc()) {
      name_set = &proc_names;
    } else {
      XLS_RET_CHECK(function_base->IsBlock());
      name_set = &block_names;
    }
    XLS_RET_CHECK(!name_set->contains(function_base->name()))
        << "Function/proc/block with name " << function_base->name()
        << " is not unique within package " << package->name();
    name_set->insert(function_base->name());

    XLS_RET_CHECK(!function_bases.contains(function_base))
        << "Function or proc with name " << function_base->name()
        << " appears more than once in within package" << package->name();
    function_bases.insert(function_base);
  }

  XLS_RETURN_IF_ERROR(VerifyChannels(package, codegen, topo_sort));
  XLS_RETURN_IF_ERROR(VerifyElaboration(package));
  XLS_RETURN_IF_ERROR(VerifyBlockProvenance(package));

  // TODO(meheff): Verify main entry point is one of the functions.
  // TODO(meheff): Verify functions called by any node are in the set of
  //   functions owned by the package.
  // TODO(meheff): Verify that there is no recursion.

  return absl::OkStatus();
}

absl::Status VerifyFunction(Function* function, bool codegen) {
  VLOG(4) << "Verifying function:\n";
  XLS_VLOG_LINES(4, function->DumpIr());

  XLS_RETURN_IF_ERROR(VerifyFunctionBase(function));

  for (Node* node : function->nodes()) {
    if (node->Is<Send>() || node->Is<Receive>()) {
      return absl::InternalError(absl::StrFormat(
          "Send and receive nodes can only be in procs, not functions (%s)",
          node->GetName()));
    }
  }

  return absl::OkStatus();
}

static absl::Status VerifyProcScopedChannels(Proc* proc) {
  // Verify channel references contains exactly the set expected from the
  // interface and channel definitions. Map value is used to track how many
  // times channel reference appears in the interface and channel definitions
  // (should always be one).
  absl::flat_hash_map<std::pair<std::string_view, ChannelDirection>, int>
      channel_interfaces;
  for (const std::unique_ptr<ChannelInterface>& channel_ref :
       proc->channel_interfaces()) {
    if (!channel_interfaces
             .insert({{channel_ref->name(), channel_ref->direction()}, 1})
             .second) {
      return absl::InternalError(absl::StrFormat(
          "Duplicate channel reference, name `%s` and direction `%s`",
          channel_ref->name(),
          ChannelDirectionToString(channel_ref->direction())));
    }
  }

  // Verifies that the channel reference with the given name and direction exist
  // and is unique.
  auto check_channel_ref_unique =
      [&](std::string_view name, ChannelDirection direction) -> absl::Status {
    if (!channel_interfaces.contains({name, direction})) {
      return absl::InternalError(
          absl::StrFormat("Channel reference with name `%s` and direction `%s` "
                          "does not exist in list of channel references",
                          name, ChannelDirectionToString(direction)));
    }
    if (--channel_interfaces[{name, direction}] != 0) {
      return absl::InternalError(absl::StrFormat(
          "Duplicate channel reference, name `%s` and direction `%s`", name,
          ChannelDirectionToString(direction)));
    }
    return absl::OkStatus();
  };

  // Verify no duplicate channel names.
  absl::flat_hash_set<std::string_view> channel_names;
  for (ChannelInterface* channel_ref : proc->interface()) {
    XLS_RETURN_IF_ERROR(check_channel_ref_unique(channel_ref->name(),
                                                 channel_ref->direction()));
  }
  for (Channel* channel : proc->channels()) {
    if (!channel_names.insert(channel->name()).second) {
      return absl::InternalError(
          absl::StrFormat("Duplicate channel name `%s` in proc `%s`",
                          channel->name(), proc->name()));
    }
    XLS_RETURN_IF_ERROR(
        check_channel_ref_unique(channel->name(), ChannelDirection::kSend));
    XLS_RETURN_IF_ERROR(
        check_channel_ref_unique(channel->name(), ChannelDirection::kReceive));
  }

  // All channel interfaces returned by Proc::GetChannelInterfaces should be
  // accounted for by the interface and channel declarations.
  for (const std::unique_ptr<ChannelInterface>& channel_ref :
       proc->channel_interfaces()) {
    if (channel_interfaces[{channel_ref->name(), channel_ref->direction()}] !=
        0) {
      return absl::InternalError(
          absl::StrFormat("%s channel reference `%s` appears in "
                          "Proc::GetChannelInterfaces() for proc `%s` "
                          "but not in the interface or declared channels",
                          ChannelDirectionToString(channel_ref->direction()),
                          channel_ref->name(), proc->name()));
    }
  }

  for (Node* node : proc->nodes()) {
    if (node->Is<Send>()) {
      if (!proc->HasChannelInterface(node->As<Send>()->channel_name(),
                                     ChannelDirection::kSend)) {
        return absl::InternalError(absl::StrFormat(
            "No send channel reference `%s` in proc `%s`, used by node `%s`",
            node->As<Send>()->channel_name(), proc->name(), node->GetName()));
      }
    }
    if (node->Is<Receive>()) {
      if (!proc->HasChannelInterface(node->As<Receive>()->channel_name(),
                                     ChannelDirection::kReceive)) {
        return absl::InternalError(absl::StrFormat(
            "No receive channel reference `%s` in proc `%s`, used by node `%s`",
            node->As<Receive>()->channel_name(), proc->name(),
            node->GetName()));
      }
    }
  }

  return absl::OkStatus();
}

static absl::Status VerifyProcInstantiations(Proc* proc) {
  absl::flat_hash_set<std::string_view> instantiation_names;
  for (const std::unique_ptr<ProcInstantiation>& instantiation :
       proc->proc_instantiations()) {
    auto [_, inserted] = instantiation_names.insert(instantiation->name());
    if (!inserted) {
      return absl::InternalError(
          absl::StrFormat("Multiple instantiations named `%s` in proc `%s`",
                          instantiation->name(), proc->name()));
    }

    bool found_proc = false;
    for (const std::unique_ptr<Proc>& package_proc : proc->package()->procs()) {
      if (instantiation->proc() == package_proc.get()) {
        found_proc = true;
        break;
      }
    }
    if (!found_proc) {
      return absl::InternalError(
          absl::StrFormat("Proc instantiation `%s` in proc `%s` does not refer "
                          "to proc in package",
                          instantiation->name(), proc->name()));
    }
    XLS_RET_CHECK(instantiation->proc()->is_new_style_proc());

    // Verify types and direction match for each channel argument.
    XLS_RET_CHECK_EQ(instantiation->channel_args().size(),
                     instantiation->proc()->interface().size())
        << absl::StrFormat("instantiation `%s` in proc `%s`",
                           instantiation->name(), proc->name());
    for (int64_t i = 0; i < instantiation->channel_args().size(); ++i) {
      if (instantiation->channel_args()[i]->direction() !=
          instantiation->proc()->interface()[i]->direction()) {
        return absl::InternalError(absl::StrFormat(
            "In proc instantiation `%s` in proc `%s`, expected direction of "
            "channel argument %d (`%s`) to be %s, got %s",
            instantiation->name(), proc->name(), i,
            instantiation->channel_args()[i]->name(),
            ChannelDirectionToString(
                instantiation->proc()->interface()[i]->direction()),
            ChannelDirectionToString(
                instantiation->channel_args()[i]->direction())));
      }
      if (instantiation->channel_args()[i]->type() !=
          instantiation->proc()->interface()[i]->type()) {
        return absl::InternalError(absl::StrFormat(
            "In proc instantiation `%s` in proc `%s`, expected type of "
            "channel argument %d (`%s`) to be %s, got %s",
            instantiation->name(), proc->name(), i,
            instantiation->channel_args()[i]->name(),
            instantiation->proc()->interface()[i]->type()->ToString(),
            instantiation->channel_args()[i]->type()->ToString()));
      }
      if (instantiation->channel_args()[i]->kind() !=
          instantiation->proc()->interface()[i]->kind()) {
        return absl::InternalError(absl::StrFormat(
            "In proc instantiation `%s` in proc `%s`, expected kind of "
            "channel argument %d (`%s`) to be %s, got %s",
            instantiation->name(), proc->name(), i,
            instantiation->channel_args()[i]->name(),
            ChannelKindToString(instantiation->proc()->interface()[i]->kind()),
            ChannelKindToString(instantiation->channel_args()[i]->kind())));
      }
    }
  }
  return absl::OkStatus();
}

absl::Status VerifyProc(Proc* proc, bool codegen) {
  VLOG(4) << "Verifying proc:\n";
  XLS_VLOG_LINES(4, proc->DumpIr());

  XLS_RETURN_IF_ERROR(VerifyFunctionBase(proc));

  if (proc->is_new_style_proc()) {
    XLS_RETURN_IF_ERROR(VerifyProcScopedChannels(proc));
    XLS_RETURN_IF_ERROR(VerifyProcInstantiations(proc));
  }

  // A Proc cannot have any parameters.
  XLS_RET_CHECK(proc->params().empty());

  return absl::OkStatus();
}

// Verify that the given set of port nodes on the instantiated block match
// one-to-one with the instantiation input/output nodes in the instantiating
// block.
template <typename PortNodeT, typename InstantiationNodeT>
static absl::Status VerifyPortsMatch(
    absl::Span<PortNodeT* const> port_nodes,
    absl::Span<InstantiationNodeT* const> instantiation_nodes,
    BlockInstantiation* instantiation) {
  std::vector<std::string> block_port_names;
  for (PortNodeT* port_node : port_nodes) {
    block_port_names.push_back(port_node->GetName());
  }
  std::vector<std::string> instantiation_port_names;
  for (InstantiationNodeT* instantiation_node : instantiation_nodes) {
    instantiation_port_names.push_back(instantiation_node->port_name());
  }
  for (const std::string& name : block_port_names) {
    if (std::find(instantiation_port_names.begin(),
                  instantiation_port_names.end(),
                  name) == instantiation_port_names.end()) {
      return absl::InternalError(
          absl::StrFormat("Instantiation `%s` of block `%s` is missing "
                          "instantation input/output node for port `%s`",
                          instantiation->name(),
                          instantiation->instantiated_block()->name(), name));
    }
  }
  for (const std::string& name : instantiation_port_names) {
    if (std::find(block_port_names.begin(), block_port_names.end(), name) ==
        block_port_names.end()) {
      return absl::InternalError(absl::StrFormat(
          "No port `%s` on instantiated block `%s` for instantiation `%s`",
          name, instantiation->instantiated_block()->name(),
          instantiation->name()));
    }
  }
  absl::flat_hash_set<std::string_view> name_set;
  for (const std::string& name : instantiation_port_names) {
    if (!name_set.insert(name).second) {
      return absl::InternalError(
          absl::StrFormat("Duplicate instantiation input/output nodes for port "
                          "`%s` in instantiation `%s` of block `%s`",
                          name, instantiation->name(),
                          instantiation->instantiated_block()->name()));
    }
  }

  return absl::OkStatus();
}

// Verifies invariants of the given block instantiation.
static absl::Status VerifyBlockInstantiation(BlockInstantiation* instantiation,
                                             Block* instantiating_block) {
  Block* instantiated_block = instantiation->instantiated_block();
  Package* package = instantiating_block->package();
  auto block_in_package = [](Package* p, Block* b) {
    for (const std::unique_ptr<Block>& block : p->blocks()) {
      if (block.get() == b) {
        return true;
      }
    }
    return false;
  };
  if (!block_in_package(package, instantiated_block)) {
    return absl::InternalError(absl::StrFormat(
        "Instantiated block `%s` (%p) is not owned by package `%s`",
        instantiated_block->name(), instantiated_block, package->name()));
  }

  // Verify a one-to-one correspondence between the following sets:
  // (1) InstantiationInput nodes returned by Block::GetInstantiationInputs.
  // (2) InputPorts on the instantiated Block.
  XLS_RETURN_IF_ERROR(VerifyPortsMatch(
      instantiated_block->GetInputPorts(),
      instantiating_block->GetInstantiationInputs(instantiation),
      instantiation));

  // Verify a one-to-one correspondence between the following sets:
  // (1) InstantiationOutput nodes returned by Block::GetInstantiationOutputs.
  // (2) OutputPorts on the instantiated Block.
  XLS_RETURN_IF_ERROR(VerifyPortsMatch(
      instantiated_block->GetOutputPorts(),
      instantiating_block->GetInstantiationOutputs(instantiation),
      instantiation));

  return absl::OkStatus();
}

// TODO(hzeller): 2023-06-28 This is only needing a foreign function as
// input so this test can be moved to earlier pase steps so that it can be
//   (a) independently and easily tested
//   (b) could be used to surface issues right in the language server.
static absl::Status VerifyForeignFunctionTemplate(Function* fun) {
  auto err_msg = [fun](std::string_view msg) -> std::string {
    return absl::StrCat("In FFI template for ", fun->name(), "(): ", msg);
  };

  XLS_ASSIGN_OR_RETURN(
      const CodeTemplate& code_template,
      CodeTemplate::Create(fun->ForeignFunctionData()->code_template()));
  int64_t instance_name_parameter_count = 0;
  std::vector<std::string> replacements;
  Type* const return_type = fun->GetType()->return_type();
  for (const std::string_view original : code_template.Expressions()) {
    if (original == "fn") {
      ++instance_name_parameter_count;
      continue;
    }
    if (original == "return") {
      continue;
    }

    static const LazyRE2 kReMatchTupleId{R"([^.]*\.([0-9]+)(\.([0-9]+))*)"};
    if (absl::StartsWith(original, "return.")) {
      if (!return_type->IsTuple()) {
        return absl::InvalidArgumentError(err_msg(
            "Dot-access `return.<idx>` in template, but function does not "
            "return a tuple."));
      }
      int64_t tuple_idx;
      if (!RE2::FullMatch(original, *kReMatchTupleId, &tuple_idx)) {
        return absl::InvalidArgumentError(
            err_msg(absl::StrCat("tuple index expected in `", original, "`")));
      }
      const int64_t expeced_max_idx = return_type->AsTupleOrDie()->size() - 1;
      if (tuple_idx < 0 || tuple_idx > expeced_max_idx) {
        return absl::InvalidArgumentError(
            err_msg(absl::StrFormat("Expected tuple index 0..%d, got `%s`",
                                    expeced_max_idx, original)));
      }
      continue;
    }

    // Any remaining template parameters must be function parameters.
    std::string_view::size_type dot_pos = original.find_first_of('.');
    std::string_view param_name = original.substr(0, dot_pos);
    auto found =
        std::find_if(fun->params().begin(), fun->params().end(),
                     [&](const Param* p) { return p->name() == param_name; });
    if (found == fun->params().end()) {
      return absl::NotFoundError(err_msg(
          absl::StrCat(" template wants '", param_name,
                       "', but that is not a parameter of the function")));
    }
    // If there is an tuple access, make sure this parameter is a tuple
    if (dot_pos != std::string_view::npos && !(*found)->GetType()->IsTuple()) {
      return absl::InvalidArgumentError(
          err_msg(absl::StrCat("Dot-access on `", param_name,
                               ".<idx>`, but parameter is not a tuple")));
    }
  }
  if (instance_name_parameter_count != 1) {
    return absl::NotFoundError(
        err_msg("Expected one {fn} template parameter for the instance name"));
  }

  return absl::OkStatus();
}

static absl::Status VerifyExternInstantiation(
    ExternInstantiation* instantiation) {
  Function* const fun = instantiation->function();
  if (!fun->ForeignFunctionData().has_value()) {
    return absl::NotFoundError(
        "Extern function instantation expects ffi template information");
  }
  return VerifyForeignFunctionTemplate(fun);
}

static absl::Status VerifyFifoInstantiation(Package* package,
                                            FifoInstantiation* instantiation) {
  if (instantiation->fifo_config().depth() < 0) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Expected fifo depth >= 0, got %d",
                        instantiation->fifo_config().depth()));
  }
  if (instantiation->channel_name().has_value() &&
      package->HasChannelWithName(instantiation->channel_name().value())) {
    XLS_ASSIGN_OR_RETURN(Channel * channel,
                         package->GetChannel(*instantiation->channel_name()));
    if (channel->kind() != ChannelKind::kStreaming) {
      return absl::InvalidArgumentError(
          absl::StrFormat("Expected channel %s (with FIFO instantiation %s) to "
                          "be streaming, got %s",
                          channel->name(), instantiation->name(),
                          ChannelKindToString(channel->kind())));
    }
    StreamingChannel* streaming_channel = down_cast<StreamingChannel*>(channel);
    if (!streaming_channel->channel_config().fifo_config().has_value()) {
      return absl::InvalidArgumentError(
          absl::StrFormat("Expected channel %s with fifo instantiation %s to "
                          "have a fifo config",
                          channel->name(), instantiation->name()));
    }
    // TODO(google/xls#1173): don't replicate fifo configs in the signature.
    if (streaming_channel->channel_config().fifo_config() !=
        instantiation->fifo_config()) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "Expected channel %s with fifo instantiation %s to have the same "
          "fifo config (%s != %s)",
          channel->name(), instantiation->name(),
          streaming_channel->channel_config().fifo_config()->ToString(),
          instantiation->fifo_config().ToString()));
    }
  }
  return absl::OkStatus();
}

absl::Status VerifyBlock(Block* block, bool codegen) {
  VLOG(4) << "Verifying block:\n";
  XLS_VLOG_LINES(4, block->DumpIr());

  XLS_RETURN_IF_ERROR(VerifyFunctionBase(block));

  // The block should have both reset port and reset behavior defined, or
  // neither.
  if (block->GetResetPort().has_value() &&
      !block->GetResetBehavior().has_value()) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Block `%s` has a reset port but no defined reset behavior",
        block->name()));
  }
  if (!block->GetResetPort().has_value() &&
      block->GetResetBehavior().has_value()) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Block `%s` has a defined reset behavior but no reset port",
        block->name()));
  }

  // If the block does not have a reset port, then none of the registers should
  // have reset values.
  if (!block->GetResetPort().has_value()) {
    for (Register* reg : block->GetRegisters()) {
      if (reg->reset_value().has_value()) {
        return absl::InternalError(
            absl::StrFormat("Register `%s` of block `%s` has a reset value but "
                            "the  block has no reset port",
                            reg->name(), block->name()));
      }
    }
  }

  // Verify that there are no cycles in the node graph.
  // The previous check in VerifyFunctionBase looks locally, but does not look
  // for cycles through instantiations. We elaborate and use a visitor on the
  // elaboration to look for more cycles through the hierarchy.
  class CycleChecker : public ElaboratedBlockDfsVisitorWithDefault {
    absl::Status DefaultHandler(const ElaboratedNode& node) override {
      return absl::OkStatus();
    }
  };
  CycleChecker cycle_checker;
  if (!block->GetInstantiations().empty()) {
    XLS_ASSIGN_OR_RETURN(BlockElaboration elab,
                         BlockElaboration::Elaborate(block));
    XLS_RETURN_IF_ERROR(elab.Accept(cycle_checker));
  }

  // Verify the nodes returned by Block::Get*Port methods are consistent.
  absl::flat_hash_set<Node*> all_data_ports;
  for (const Block::Port& port : block->GetPorts()) {
    if (std::holds_alternative<InputPort*>(port)) {
      all_data_ports.insert(std::get<InputPort*>(port));
    } else if (std::holds_alternative<OutputPort*>(port)) {
      all_data_ports.insert(std::get<OutputPort*>(port));
    }
  }
  absl::flat_hash_set<Node*> input_data_ports(block->GetInputPorts().begin(),
                                              block->GetInputPorts().end());
  absl::flat_hash_set<Node*> output_data_ports(block->GetOutputPorts().begin(),
                                               block->GetOutputPorts().end());

  // All the pointers returned by the GetPort methods should be unique.
  XLS_RET_CHECK_EQ(block->GetInputPorts().size(), input_data_ports.size());
  XLS_RET_CHECK_EQ(block->GetOutputPorts().size(), output_data_ports.size());
  XLS_RET_CHECK_EQ(
      block->GetInputPorts().size() + block->GetOutputPorts().size(),
      all_data_ports.size());

  int64_t input_port_count = 0;
  int64_t output_port_count = 0;
  for (Node* node : block->nodes()) {
    if (node->Is<InputPort>()) {
      XLS_RET_CHECK(all_data_ports.contains(node)) << node->GetName();
      XLS_RET_CHECK(input_data_ports.contains(node)) << node->GetName();
      XLS_RET_CHECK(block->HasInputPort(node->GetName()));
      XLS_RET_CHECK(!block->HasOutputPort(node->GetName()));
      input_port_count++;
    } else if (node->Is<OutputPort>()) {
      XLS_RET_CHECK(all_data_ports.contains(node)) << node->GetName();
      XLS_RET_CHECK(output_data_ports.contains(node)) << node->GetName();
      XLS_RET_CHECK(!block->HasInputPort(node->GetName()));
      XLS_RET_CHECK(block->HasOutputPort(node->GetName()));
      output_port_count++;
    } else {
      XLS_RET_CHECK(!block->HasInputPort(node->GetName()));
      XLS_RET_CHECK(!block->HasOutputPort(node->GetName()));
    }
  }
  XLS_RET_CHECK_EQ(input_port_count, input_data_ports.size());
  XLS_RET_CHECK_EQ(output_port_count, output_data_ports.size());

  // Blocks should have no parameters.
  XLS_RET_CHECK(block->params().empty());

  // The block must have a clock port if it has any registers.
  if (!block->GetRegisters().empty() && !block->GetClockPort().has_value()) {
    return absl::InternalError(
        StrFormat("Block has registers but no clock port"));
  }

  // Verify all registers have exactly one read and that operation is the one
  // returned by GetRegisterRead. A register may have more than one
  // RegisterWrite, but in that case all must have load enables and matching
  // reset (if any).
  absl::flat_hash_map<Register*, RegisterRead*> reg_reads;
  absl::flat_hash_map<Register*, std::vector<RegisterWrite*>> reg_writes;
  for (Node* node : block->nodes()) {
    if (node->Is<RegisterRead>()) {
      RegisterRead* reg_read = node->As<RegisterRead>();
      Register* reg = reg_read->GetRegister();
      if (reg_reads.contains(reg)) {
        return absl::InternalError(
            StrFormat("Register %s has multiple reads", reg->name()));
      }
      XLS_RET_CHECK_EQ(reg->type(), node->GetType());
      reg_reads[reg] = reg_read;
    } else if (node->Is<RegisterWrite>()) {
      RegisterWrite* reg_write = node->As<RegisterWrite>();
      Register* reg = reg_write->GetRegister();
      XLS_RET_CHECK_EQ(reg->type(), reg_write->data()->GetType());
      if (reg_write->load_enable().has_value()) {
        XLS_RET_CHECK_EQ(reg_write->load_enable().value()->GetType(),
                         block->package()->GetBitsType(1));
      }
      reg_writes[reg].push_back(reg_write);
    }
  }
  for (Register* reg : block->GetRegisters()) {
    if (!reg_reads.contains(reg)) {
      return absl::InternalError(
          StrFormat("Register %s has no read", reg->name()));
    }
    XLS_ASSIGN_OR_RETURN(RegisterRead * reg_read, block->GetRegisterRead(reg));
    XLS_RET_CHECK_EQ(reg_read, reg_reads.at(reg));

    auto reg_write_it = reg_writes.find(reg);
    if (reg_write_it == reg_writes.end()) {
      return absl::InternalError(
          StrFormat("Register %s has no write", reg->name()));
    }
    if (reg_write_it->second.size() == 1) {
      XLS_ASSIGN_OR_RETURN(RegisterWrite * reg_write,
                           block->GetUniqueRegisterWrite(reg));
      XLS_RET_CHECK_EQ(reg_write, reg_write_it->second.front());
    } else {
      for (RegisterWrite* reg_write : reg_write_it->second) {
        if (!reg_write->load_enable().has_value()) {
          return absl::InternalError(
              StrFormat("Register %s has multiple writes but not all writes "
                        "have load enables: %s",
                        reg->name(), reg_write->GetName()));
        }
        RegisterWrite* first_reg_write = reg_write_it->second.front();
        if (reg_write->reset() != first_reg_write->reset()) {
          return absl::InternalError(StrFormat(
              "Register writes for register %s have different reset operands: "
              "%s and %s",
              reg->name(), reg_write->GetName(), first_reg_write->GetName()));
        }
      }
    }
  }

  for (Instantiation* instantiation : block->GetInstantiations()) {
    switch (instantiation->kind()) {
      case InstantiationKind::kBlock:
        // Verify each instantiation is a block instantiation and the block is
        // owned the package.
        XLS_RETURN_IF_ERROR(VerifyBlockInstantiation(
            down_cast<BlockInstantiation*>(instantiation), block));
        break;
      case InstantiationKind::kExtern:
        XLS_RETURN_IF_ERROR(VerifyExternInstantiation(
            down_cast<ExternInstantiation*>(instantiation)));
        break;
      case InstantiationKind::kFifo:
        XLS_RETURN_IF_ERROR(VerifyFifoInstantiation(
            block->package(), down_cast<FifoInstantiation*>(instantiation)));
        break;
      default:
        XLS_RET_CHECK_FAIL()
            << "Only block, ffi, and fifo instantiations are supported: "
            << instantiation->ToString();
    }
  }

  XLS_RETURN_IF_ERROR(VerifyBlockChannelMetadata(block));

  return absl::OkStatus();
}

}  // namespace xls
