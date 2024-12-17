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

#include "xls/ir/proc.h"

#include <algorithm>
#include <cstdint>
#include <iterator>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/btree_set.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "xls/common/casts.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/channel.h"
#include "xls/ir/function.h"
#include "xls/ir/name_uniquer.h"
#include "xls/ir/node.h"
#include "xls/ir/nodes.h"
#include "xls/ir/op.h"
#include "xls/ir/package.h"
#include "xls/ir/proc_instantiation.h"
#include "xls/ir/source_location.h"
#include "xls/ir/state_element.h"
#include "xls/ir/topo_sort.h"
#include "xls/ir/type.h"
#include "xls/ir/value.h"
#include "xls/ir/value_utils.h"

namespace xls {

std::string Proc::DumpIr() const {
  std::string res = absl::StrFormat("proc %s", name());
  if (is_new_style_proc()) {
    absl::StrAppendFormat(
        &res, "<%s>",
        absl::StrJoin(interface_, ", ",
                      [](std::string* s, const ChannelReference* channel_ref) {
                        absl::StrAppend(s, channel_ref->ToString());
                      }));
  }
  auto state_formatter = [](std::string* s, StateElement* state) {
    absl::StrAppend(s, state->name(), ": ", state->type()->ToString());
  };
  absl::StrAppend(&res, "(",
                  absl::StrJoin(StateElements(), ", ", state_formatter));
  auto initial_value_formatter = [](std::string* s, StateElement* state) {
    UntypedValueFormatter(s, state->initial_value());
  };
  if (!StateElements().empty()) {
    absl::StrAppendFormat(
        &res, ", init={%s}",
        absl::StrJoin(StateElements(), ", ", initial_value_formatter));
  }
  absl::StrAppend(&res, ") {\n");

  if (is_new_style_proc()) {
    for (Channel* channel : channels()) {
      absl::StrAppendFormat(&res, "  %s\n", channel->ToString());
    }
    for (const std::unique_ptr<ProcInstantiation>& instantiation :
         proc_instantiations()) {
      absl::StrAppendFormat(&res, "  %s\n", instantiation->ToString());
    }
  }
  for (Node* node : TopoSort(const_cast<Proc*>(this))) {
    if (node->op() == Op::kParam) {
      continue;
    }
    absl::StrAppend(&res, "  ", node->ToString(), "\n");
  }

  // TODO: google/xls#1520 - remove this once fully transitioned over to
  // `next_value` nodes.
  if (next_values_.empty() && !state_elements_.empty()) {
    auto node_formatter = [](std::string* s, Node* node) {
      absl::StrAppend(s, node->GetName());
    };
    absl::StrAppend(&res, "  next (",
                    absl::StrJoin(next_state_, ", ", node_formatter), ")\n");
  }
  absl::StrAppend(&res, "}\n");
  return res;
}

int64_t Proc::GetStateFlatBitCount() const {
  int64_t total = 0;
  for (StateElement* state_element : StateElements()) {
    total += state_element->type()->GetFlatBitCount();
  }
  return total;
}

absl::StatusOr<StateElement*> Proc::GetStateElement(
    std::string_view name) const {
  if (std::optional<StateElement*> state_element = MaybeGetStateElement(name);
      state_element.has_value()) {
    return *state_element;
  }
  return absl::NotFoundError(absl::StrFormat(
      "Proc %s has no state element named %s", this->name(), name));
}

std::optional<StateElement*> Proc::MaybeGetStateElement(
    std::string_view name) const {
  auto it = state_elements_.find(name);
  if (it == state_elements_.end()) {
    return std::nullopt;
  }
  return it->second.get();
}

absl::StatusOr<int64_t> Proc::GetStateElementIndex(
    StateElement* state_element) const {
  if (std::optional<int64_t> index = MaybeGetStateElementIndex(state_element);
      index.has_value()) {
    return *index;
  }
  return absl::InvalidArgumentError(
      absl::StrCat("Given state element is not a state element of this proc: ",
                   state_element->ToString()));
}
std::optional<int64_t> Proc::MaybeGetStateElementIndex(
    StateElement* state_element) const {
  if (auto it = absl::c_find(state_vec_, state_element);
      it != state_vec_.end()) {
    return std::distance(state_vec_.begin(), it);
  }
  return std::nullopt;
}

absl::btree_set<int64_t> Proc::GetNextStateIndices(Node* node) const {
  auto it = next_state_indices_.find(node);
  if (it == next_state_indices_.end()) {
    return absl::btree_set<int64_t>();
  }
  return it->second;
}

absl::Status Proc::SetNextStateElement(int64_t index, Node* next) {
  if (next->GetType() != GetStateElementType(index)) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Cannot set next state element %d to \"%s\"; type %s does not match "
        "proc state element type %s",
        index, next->GetName(), next->GetType()->ToString(),
        GetStateElementType(index)->ToString()));
  }
  next_state_indices_[next_state_[index]].erase(index);
  next_state_[index] = next;
  next_state_indices_[next].insert(index);
  return absl::OkStatus();
}

absl::Status Proc::ReplaceState(
    absl::Span<const std::string> requested_state_names,
    absl::Span<const Value> init_values) {
  for (int64_t i = GetStateElementCount() - 1; i >= 0; --i) {
    XLS_RETURN_IF_ERROR(RemoveStateElement(i));
  }
  state_name_uniquer_.Reset();
  if (requested_state_names.size() != init_values.size()) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Must specify equal number of state names (%d) and initial values (%d)",
        requested_state_names.size(), init_values.size()));
  }
  for (int64_t i = 0; i < requested_state_names.size(); ++i) {
    XLS_RETURN_IF_ERROR(
        AppendStateElement(requested_state_names[i], init_values[i]).status());
  }
  return absl::OkStatus();
}

absl::Status Proc::ReplaceState(
    absl::Span<const std::string> requested_state_names,
    absl::Span<const Value> init_values, absl::Span<Node* const> next_state) {
  for (int64_t i = GetStateElementCount() - 1; i >= 0; --i) {
    XLS_RETURN_IF_ERROR(RemoveStateElement(i));
  }
  state_name_uniquer_.Reset();
  // Verify next values match the type of the initial values.
  if (requested_state_names.size() != next_state.size() ||
      requested_state_names.size() != init_values.size()) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Must specify equal number of state names (%d), next state values (%d) "
        "and initial values (%d)",
        requested_state_names.size(), next_state.size(), init_values.size()));
  }
  for (int64_t i = 0; i < requested_state_names.size(); ++i) {
    XLS_RETURN_IF_ERROR(AppendStateElement(requested_state_names[i],
                                           init_values[i], next_state[i])
                            .status());
  }
  return absl::OkStatus();
}

absl::StatusOr<StateRead*> Proc::ReplaceStateElement(
    int64_t index, std::string_view requested_state_name,
    const Value& init_value, std::optional<Node*> next_state) {
  XLS_RET_CHECK_LT(index, GetStateElementCount());

  // Check that it's safe to remove the current state read node.
  StateElement* old_state_element = GetStateElement(index);
  StateRead* old_state_read = state_reads_.at(old_state_element);
  if (!old_state_read->users().empty()) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Cannot remove state element %d of proc %s, existing "
                        "state read %s has uses",
                        index, name(), old_state_read->GetName()));
  }

  // Unless we're directly reusing the previous name, it needs to be uniqued.
  // Also, copy name to a local variable to avoid the use-after-free footgun of
  // `requested_state_name` referring to the existing to-be-removed state
  // element.
  std::string state_name;
  if (requested_state_name == old_state_element->name()) {
    state_name = requested_state_name;
  } else {
    state_name = UniquifyStateName(requested_state_name);
  }

  // TODO: google/xls#1520 - remove this once fully transitioned over to
  // `next_value` nodes.
  next_state_indices_[next_state_[index]].erase(index);
  next_state_[index] = nullptr;

  XLS_RETURN_IF_ERROR(RemoveNode(old_state_read));

  state_elements_[state_name] = std::make_unique<StateElement>(
      state_name, package()->GetTypeForValue(init_value), init_value);
  StateElement* new_state_element = state_elements_.at(state_name).get();
  state_vec_[index] = new_state_element;

  // Construct the new state-read node, and update all trackers.
  XLS_ASSIGN_OR_RETURN(
      StateRead * state_read,
      MakeNodeWithName<StateRead>(SourceInfo(), new_state_element, state_name));
  state_reads_[new_state_element] = state_read;
  if (next_state.has_value() &&
      !ValueConformsToType(init_value, next_state.value()->GetType())) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Cannot add state element at %d, next state value %s (type %s) does "
        "not match type of initial value: %s",
        index, next_state.value()->GetName(),
        next_state.value()->GetType()->ToString(), init_value.ToString()));
  }

  // TODO: google/xls#1520 - remove this once fully transitioned over to
  // `next_value` nodes.
  next_state_[index] = next_state.value_or(state_read);
  next_state_indices_[next_state_[index]].insert(index);

  return state_read;
}

absl::Status Proc::RemoveStateElement(int64_t index) {
  XLS_RET_CHECK_LT(index, GetStateElementCount());
  for (auto& [_, state_element] : state_elements_) {
    CHECK(absl::c_contains(state_vec_, state_element.get()));
  }
  for (StateElement* state_element : state_vec_) {
    CHECK(absl::c_any_of(state_elements_, [&](const auto& entry) {
      return entry.second.get() == state_element;
    }));
  }

  // TODO: google/xls#1520 - remove this once fully transitioned over to
  // `next_value` nodes.
  if (index < GetStateElementCount() - 1) {
    for (auto& [_, indices] : next_state_indices_) {
      // Relabel all indices > `index`.
      auto it = indices.upper_bound(index);
      absl::btree_set<int64_t> relabeled_indices(indices.begin(), it);
      for (; it != indices.end(); ++it) {
        relabeled_indices.insert((*it) - 1);
      }
      indices = std::move(relabeled_indices);
    }
  }
  next_state_indices_[next_state_[index]].erase(index);
  next_state_.erase(next_state_.begin() + index);

  StateElement* old_state_element = GetStateElement(index);
  StateRead* old_state_read = state_reads_.at(old_state_element);
  if (!old_state_read->users().empty()) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Cannot remove state element %d of proc %s, existing "
                        "state read %s has uses",
                        index, name(), old_state_read->GetNameView()));
  }
  XLS_RETURN_IF_ERROR(RemoveNode(old_state_read));

  state_elements_.erase(old_state_element->name());
  state_vec_.erase(state_vec_.begin() + index);
  return absl::OkStatus();
}

absl::StatusOr<StateRead*> Proc::AppendStateElement(
    std::string_view requested_state_name, const Value& init_value,
    std::optional<Node*> next_state) {
  return InsertStateElement(GetStateElementCount(), requested_state_name,
                            init_value, next_state);
}

absl::StatusOr<StateRead*> Proc::InsertStateElement(
    int64_t index, std::string_view requested_state_name,
    const Value& init_value, std::optional<Node*> next_state) {
  XLS_RET_CHECK_LE(index, GetStateElementCount());
  const bool is_append = (index == GetStateElementCount());
  std::string state_name = UniquifyStateName(requested_state_name);
  state_elements_[state_name] = std::make_unique<StateElement>(
      state_name, package()->GetTypeForValue(init_value), init_value);
  StateElement* state_element = state_elements_.at(state_name).get();
  state_vec_.insert(state_vec_.begin() + index, state_element);
  XLS_ASSIGN_OR_RETURN(
      StateRead * state_read,
      MakeNodeWithName<StateRead>(SourceInfo(), state_element, state_name));
  state_reads_[state_element] = state_read;

  // TODO: google/xls#1520 - remove this once fully transitioned over to
  // `next_value` nodes.
  if (next_state.has_value()) {
    if (!ValueConformsToType(init_value, next_state.value()->GetType())) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "Cannot add state element at %d, next state value %s (type %s) does "
          "not match type of initial value: %s",
          index, next_state.value()->GetName(),
          next_state.value()->GetType()->ToString(), init_value.ToString()));
    }
  }
  if (!is_append) {
    for (auto& [_, indices] : next_state_indices_) {
      // Relabel all indices >= `index`.
      auto it = indices.lower_bound(index);
      absl::btree_set<int64_t> relabeled_indices(indices.begin(), it);
      for (; it != indices.end(); ++it) {
        relabeled_indices.insert((*it) + 1);
      }
      indices = std::move(relabeled_indices);
    }
  }
  next_state_.insert(next_state_.begin() + index,
                     next_state.value_or(state_read));
  next_state_indices_[next_state_[index]].insert(index);

  return state_read;
}

bool Proc::HasImplicitUse(Node* node) const {
  // TODO: google/xls#1520 - remove this once fully transitioned over to
  // `next_value` nodes.
  if (auto it = next_state_indices_.find(node);
      it != next_state_indices_.end() && !it->second.empty()) {
    return true;
  }

  return false;
}

absl::StatusOr<Proc*> Proc::Clone(
    std::string_view new_name, Package* target_package,
    const absl::flat_hash_map<std::string, std::string>& channel_remapping,
    const absl::flat_hash_map<const FunctionBase*, FunctionBase*>&
        call_remapping,
    const absl::flat_hash_map<std::string, std::string>& state_name_remapping)
    const {
  auto new_chan_name = [&](std::string_view n) -> std::string_view {
    if (channel_remapping.contains(n)) {
      return channel_remapping.at(n);
    }
    return n;
  };

  absl::flat_hash_map<Node*, Node*> original_to_clone;
  if (target_package == nullptr) {
    target_package = package();
  }
  Proc* cloned_proc;
  if (is_new_style_proc()) {
    cloned_proc = target_package->AddProc(std::make_unique<Proc>(
        new_name, /*interface=*/absl::Span<std::unique_ptr<ChannelReference>>(),
        target_package));
  } else {
    cloned_proc = target_package->AddProc(
        std::make_unique<Proc>(new_name, target_package));
  }
  auto remap_state_name = [&](std::string_view orig) -> std::string_view {
    if (!state_name_remapping.contains(orig)) {
      return orig;
    }
    return state_name_remapping.at(orig);
  };
  for (int64_t i = 0; i < GetStateElementCount(); ++i) {
    XLS_ASSIGN_OR_RETURN(StateRead * cloned_state_read,
                         cloned_proc->AppendStateElement(
                             remap_state_name(GetStateElement(i)->name()),
                             GetStateElement(i)->initial_value()));
    original_to_clone[state_reads_.at(GetStateElement(i))] = cloned_state_read;
  }
  if (is_new_style_proc()) {
    for (ChannelReference* channel_ref : interface()) {
      if (channel_ref->direction() == Direction::kSend) {
        XLS_RETURN_IF_ERROR(
            cloned_proc
                ->AddOutputChannelReference(
                    std::make_unique<SendChannelReference>(
                        new_chan_name(channel_ref->name()), channel_ref->type(),
                        channel_ref->kind(), channel_ref->strictness()))
                .status());
      } else {
        XLS_RETURN_IF_ERROR(
            cloned_proc
                ->AddInputChannelReference(
                    std::make_unique<ReceiveChannelReference>(
                        new_chan_name(channel_ref->name()), channel_ref->type(),
                        channel_ref->kind(), channel_ref->strictness()))
                .status());
      }
    }
    for (Channel* channel : channels()) {
      std::unique_ptr<Channel> new_channel;
      XLS_ASSIGN_OR_RETURN(
          Type * chan_type,
          target_package->MapTypeFromOtherPackage(channel->type()));
      if (channel->kind() == ChannelKind::kStreaming) {
        StreamingChannel* streaming_channel =
            down_cast<StreamingChannel*>(channel);
        new_channel = std::make_unique<StreamingChannel>(
            new_chan_name(channel->name()), channel->id(),
            channel->supported_ops(), chan_type, channel->initial_values(),
            streaming_channel->channel_config(),
            streaming_channel->GetFlowControl(),
            streaming_channel->GetStrictness(), channel->metadata());
      } else {
        new_channel = std::make_unique<SingleValueChannel>(
            new_chan_name(channel->name()), channel->id(),
            channel->supported_ops(), chan_type, channel->metadata());
      }
      XLS_RETURN_IF_ERROR(
          cloned_proc->AddChannel(std::move(new_channel)).status());
    }
  }

  for (Node* node : TopoSort(const_cast<Proc*>(this))) {
    std::vector<Node*> cloned_operands;
    for (Node* operand : node->operands()) {
      cloned_operands.push_back(original_to_clone.at(operand));
    }

    switch (node->op()) {
      case Op::kStateRead: {
        continue;
      }
      case Op::kReceive: {
        Receive* src = node->As<Receive>();
        if (is_new_style_proc()) {
          XLS_ASSIGN_OR_RETURN(
              original_to_clone[node],
              cloned_proc->MakeNodeWithName<Receive>(
                  src->loc(), cloned_operands[0],
                  cloned_operands.size() == 2
                      ? std::optional<Node*>(cloned_operands[1])
                      : std::nullopt,
                  new_chan_name(src->channel_name()), src->is_blocking(),
                  src->GetName()));
        } else {
          std::string_view channel = new_chan_name(src->channel_name());
          XLS_ASSIGN_OR_RETURN(
              original_to_clone[node],
              cloned_proc->MakeNodeWithName<Receive>(
                  src->loc(), cloned_operands[0],
                  cloned_operands.size() == 2
                      ? std::optional<Node*>(cloned_operands[1])
                      : std::nullopt,
                  channel, src->is_blocking(), src->GetName()));
        }
        break;
      }
      case Op::kSend: {
        Send* src = node->As<Send>();
        if (is_new_style_proc()) {
          XLS_ASSIGN_OR_RETURN(
              original_to_clone[node],
              cloned_proc->MakeNodeWithName<Send>(
                  src->loc(), cloned_operands[0], cloned_operands[1],
                  cloned_operands.size() == 3
                      ? std::optional<Node*>(cloned_operands[2])
                      : std::nullopt,
                  new_chan_name(src->channel_name()), src->GetName()));
        } else {
          std::string_view channel = new_chan_name(src->channel_name());
          XLS_ASSIGN_OR_RETURN(
              original_to_clone[node],
              cloned_proc->MakeNodeWithName<Send>(
                  src->loc(), cloned_operands[0], cloned_operands[1],
                  cloned_operands.size() == 3
                      ? std::optional<Node*>(cloned_operands[2])
                      : std::nullopt,
                  channel, src->GetName()));
        }
        break;
      }
      // Remap CountedFor body.
      case Op::kCountedFor: {
        CountedFor* src = node->As<CountedFor>();
        auto remapped_call = call_remapping.find(src->body());
        if (remapped_call == call_remapping.end()) {
          return absl::InvalidArgumentError(absl::StrFormat(
              "Could not find mapping for CountedFor target %s.",
              src->GetName()));
        }
        Function* body = dynamic_cast<Function*>(remapped_call->second);
        if (body == nullptr) {
          return absl::InvalidArgumentError(absl::StrFormat(
              "CountedFor target was not mapped to a function."));
        }
        XLS_ASSIGN_OR_RETURN(
            original_to_clone[node],
            cloned_proc->MakeNodeWithName<CountedFor>(
                src->loc(), cloned_operands[0],
                absl::Span<Node*>(cloned_operands).subspan(1),
                src->trip_count(), src->stride(), body, src->GetName()));
        break;
      }
      // Remap Map to_apply.
      case Op::kMap: {
        Map* src = node->As<Map>();
        auto remapped_call = call_remapping.find(src->to_apply());
        if (remapped_call == call_remapping.end()) {
          return absl::InvalidArgumentError(absl::StrFormat(
              "Could not find mapping for Map target %s.", src->GetName()));
        }
        Function* to_apply = dynamic_cast<Function*>(remapped_call->second);
        if (to_apply == nullptr) {
          return absl::InvalidArgumentError(
              absl::StrFormat("Map target was not mapped to a function."));
        }
        XLS_ASSIGN_OR_RETURN(
            original_to_clone[node],
            cloned_proc->MakeNodeWithName<Map>(src->loc(), cloned_operands[0],
                                               to_apply, src->GetName()));
        break;
      }
      // Remap Invoke to_apply.
      case Op::kInvoke: {
        Invoke* src = node->As<Invoke>();
        auto remapped_call = call_remapping.find(src->to_apply());
        if (remapped_call == call_remapping.end()) {
          return absl::InvalidArgumentError(absl::StrFormat(
              "Could not find mapping for Invoke target %s.", src->GetName()));
        }
        Function* to_apply = dynamic_cast<Function*>(remapped_call->second);
        if (to_apply == nullptr) {
          return absl::InvalidArgumentError(
              absl::StrFormat("Invoke target was not mapped to a function."));
        }
        XLS_ASSIGN_OR_RETURN(
            original_to_clone[node],
            cloned_proc->MakeNodeWithName<Invoke>(src->loc(), cloned_operands,
                                                  to_apply, src->GetName()));
        break;
      }
      // Default clone.
      default: {
        XLS_ASSIGN_OR_RETURN(
            original_to_clone[node],
            node->CloneInNewFunction(cloned_operands, cloned_proc));
        break;
      }
    }
  }

  // TODO: google/xls#1520 - remove this once fully transitioned over to
  // `next_value` nodes.
  for (int64_t i = 0; i < GetStateElementCount(); ++i) {
    XLS_RETURN_IF_ERROR(cloned_proc->SetNextStateElement(
        i, original_to_clone.at(GetNextStateElement(i))));
  }

  return cloned_proc;
}

absl::StatusOr<Type*> Proc::GetChannelReferenceType(
    std::string_view name) const {
  if (is_new_style_proc()) {
    for (const std::unique_ptr<ChannelReference>& channel_ref :
         channel_references_) {
      if (name == channel_ref->name()) {
        return channel_ref->type();
      }
    }
    return absl::NotFoundError(absl::StrFormat(
        "No channel reference `%s` in proc `%s`", name, this->name()));
  }
  XLS_ASSIGN_OR_RETURN(Channel * channel, package()->GetChannel(name));
  return channel->type();
}

absl::StatusOr<ChannelReferences> Proc::AddChannel(
    std::unique_ptr<Channel> channel) {
  XLS_RET_CHECK(is_new_style_proc());
  std::string channel_name{channel->name()};
  auto [channel_it, inserted] =
      channels_.insert({channel_name, std::move(channel)});
  if (!inserted) {
    return absl::InternalError(
        absl::StrFormat("Channel already exists with name `%s` on proc `%s`.",
                        channel_name, name()));
  }
  Channel* channel_ptr = channel_it->second.get();

  // Verify the channel id is unique.
  for (Channel* ch : channel_vec_) {
    if (ch->id() == channel_ptr->id()) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "Channel already exists with id %d on proc `%s`", ch->id(), name()));
    }
  }

  // The channel name must be a valid identifier.
  if (!NameUniquer::IsValidIdentifier(channel_ptr->name())) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Invalid channel name: \"%s\"", channel_ptr->name()));
  }

  channel_vec_.push_back(channel_ptr);
  std::optional<ChannelStrictness> strictness;
  if (StreamingChannel* streaming_channel =
          dynamic_cast<StreamingChannel*>(channel_ptr)) {
    strictness = streaming_channel->GetStrictness();
  }

  auto send_channel_ref = std::make_unique<SendChannelReference>(
      channel_ptr->name(), channel_ptr->type(), channel_ptr->kind(),
      strictness);
  auto receive_channel_ref = std::make_unique<ReceiveChannelReference>(
      channel_ptr->name(), channel_ptr->type(), channel_ptr->kind(),
      strictness);

  ChannelReferences channel_refs{.channel = channel_ptr,
                                 .send_ref = send_channel_ref.get(),
                                 .receive_ref = receive_channel_ref.get()};
  channel_references_.push_back(std::move(send_channel_ref));
  channel_references_.push_back(std::move(receive_channel_ref));
  return channel_refs;
}

absl::StatusOr<Channel*> Proc::GetChannel(std::string_view name) {
  XLS_RET_CHECK(is_new_style_proc());
  auto it = channels_.find(name);
  if (it != channels_.end()) {
    return it->second.get();
  }
  return absl::NotFoundError(absl::StrFormat(
      "No channel with name `%s` in proc `%s`", name, this->name()));
}

absl::StatusOr<ChannelRef> Proc::GetChannelRef(std::string_view name,
                                               Direction direction) {
  if (is_new_style_proc()) {
    return GetChannelReference(name, direction);
  }
  return package()->GetChannel(name);
}

bool Proc::ChannelIsOwnedByProc(Channel* channel) {
  CHECK(is_new_style_proc());
  auto it = channels_.find(channel->name());
  if (it != channels_.end()) {
    return it->second.get() == channel;
  }
  return false;
}

absl::StatusOr<ReceiveChannelReference*> Proc::AddInputChannelReference(
    std::unique_ptr<ReceiveChannelReference> channel_ref) {
  XLS_ASSIGN_OR_RETURN(ChannelReference * channel_ref_ptr,
                       AddInterfaceChannelReference(std::move(channel_ref)));
  return down_cast<ReceiveChannelReference*>(channel_ref_ptr);
}

absl::StatusOr<SendChannelReference*> Proc::AddOutputChannelReference(
    std::unique_ptr<SendChannelReference> channel_ref) {
  XLS_ASSIGN_OR_RETURN(ChannelReference * channel_ref_ptr,
                       AddInterfaceChannelReference(std::move(channel_ref)));
  return down_cast<SendChannelReference*>(channel_ref_ptr);
}

absl::StatusOr<ChannelReference*> Proc::AddInterfaceChannelReference(
    std::unique_ptr<ChannelReference> channel_ref) {
  XLS_RET_CHECK(is_new_style_proc());
  if (channels_.contains(channel_ref->name())) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Cannot add channel `%s` to proc `%s`. Already a "
                        "channel of same name defined in the proc.",
                        channel_ref->name(), name()));
  }
  for (const std::unique_ptr<ChannelReference>& other_channel_ref :
       channel_references_) {
    if (other_channel_ref->name() == channel_ref->name()) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "Cannot add channel `%s` to proc `%s`. Already an "
          "%s channel of same name on the proc.",
          channel_ref->name(), name(),
          other_channel_ref->direction() == Direction::kReceive ? "input"
                                                                : "output"));
    }
  }
  channel_references_.push_back(std::move(channel_ref));
  interface_.push_back(channel_references_.back().get());
  return interface_.back();
}

absl::StatusOr<ReceiveChannelReference*> Proc::AddInputChannel(
    std::string_view name, Type* type, ChannelKind kind,
    std::optional<ChannelStrictness> strictness) {
  return AddInputChannelReference(
      std::make_unique<ReceiveChannelReference>(name, type, kind, strictness));
}

absl::StatusOr<SendChannelReference*> Proc::AddOutputChannel(
    std::string_view name, Type* type, ChannelKind kind,
    std::optional<ChannelStrictness> strictness) {
  return AddOutputChannelReference(
      std::make_unique<SendChannelReference>(name, type, kind, strictness));
}

absl::StatusOr<ChannelReference*> Proc::AddInterfaceChannel(
    std::string_view name, Direction direction, Type* type, ChannelKind kind,
    std::optional<ChannelStrictness> strictness) {
  if (direction == Direction::kSend) {
    return AddOutputChannel(name, type, kind, strictness);
  }
  return AddInputChannel(name, type, kind, strictness);
}

absl::Status Proc::RemoveInterfaceChannel(ChannelReference* channel_ref) {
  auto interface_it =
      std::find(interface_.begin(), interface_.end(), channel_ref);
  if (interface_it == interface_.end()) {
    return absl::NotFoundError(absl::StrFormat(
        "Channel reference `%s` (%p) is not on the interface of proc `%s`",
        channel_ref->name(), channel_ref, name()));
  }
  interface_.erase(interface_it);

  std::erase_if(channel_references_,
                [&](const std::unique_ptr<ChannelReference>& ref) {
                  return ref.get() == channel_ref;
                });
  return absl::OkStatus();
}

absl::StatusOr<ProcInstantiation*> Proc::AddProcInstantiation(
    std::string_view name, absl::Span<ChannelReference* const> channel_args,
    Proc* proc) {
  XLS_RET_CHECK(is_new_style_proc());
  proc_instantiations_.push_back(
      std::make_unique<ProcInstantiation>(name, channel_args, proc));
  return proc_instantiations_.back().get();
}

bool Proc::HasChannelReference(std::string_view name,
                               Direction direction) const {
  CHECK(is_new_style_proc());
  for (const std::unique_ptr<ChannelReference>& channel_ref :
       channel_references_) {
    if (name == channel_ref->name() && direction == channel_ref->direction()) {
      return true;
    }
  }
  return false;
}

absl::StatusOr<ChannelReference*> Proc::GetChannelReference(
    std::string_view name, Direction direction) const {
  XLS_RET_CHECK(is_new_style_proc());
  for (const std::unique_ptr<ChannelReference>& channel_ref :
       channel_references_) {
    if (channel_ref->name() == name && channel_ref->direction() == direction) {
      return channel_ref.get();
    }
  }
  return absl::NotFoundError(
      absl::StrFormat("No %s channel reference `%s` in proc `%s`",
                      DirectionToString(direction), name, this->name()));
}

absl::StatusOr<SendChannelReference*> Proc::GetSendChannelReference(
    std::string_view name) const {
  XLS_RET_CHECK(is_new_style_proc());
  XLS_ASSIGN_OR_RETURN(ChannelReference * channel_ref,
                       GetChannelReference(name, Direction::kSend));
  return down_cast<SendChannelReference*>(channel_ref);
}

absl::StatusOr<ReceiveChannelReference*> Proc::GetReceiveChannelReference(
    std::string_view name) const {
  XLS_RET_CHECK(is_new_style_proc());
  XLS_ASSIGN_OR_RETURN(ChannelReference * channel_ref,
                       GetChannelReference(name, Direction::kReceive));
  return down_cast<ReceiveChannelReference*>(channel_ref);
}

absl::StatusOr<ProcInstantiation*> Proc::GetProcInstantiation(
    std::string_view instantiation_name) const {
  for (const std::unique_ptr<ProcInstantiation>& instantiation :
       proc_instantiations()) {
    if (instantiation->name() == instantiation_name) {
      return instantiation.get();
    }
  }
  return absl::NotFoundError(
      absl::StrFormat("No proc instantiation named `%s` in proc `%s`",
                      instantiation_name, name()));
}

absl::Status Proc::ConvertToNewStyle() {
  if (is_new_style_proc()) {
    return absl::InternalError("Proc is already new style.");
  }
  is_new_style_proc_ = true;
  return absl::OkStatus();
}

absl::StatusOr<StateRead*> Proc::TransformStateElement(
    StateRead* old_state_read, const Value& init_value,
    Proc::StateElementTransformer& transform) {
  StateElement* old_state_element = old_state_read->state_element();
  std::string orig_name(old_state_element->name());
  std::string orig_read_name(old_state_read->GetNameView());
  XLS_ASSIGN_OR_RETURN(
      StateRead * new_state_read,
      AppendStateElement(absl::StrFormat("TEMP_NAME__%s__", orig_name),
                         init_value));
  new_state_read->SetLoc(old_state_read->loc());
  StateElement* new_state_element = new_state_read->state_element();
  std::string temp_name = new_state_element->name();

  XLS_ASSIGN_OR_RETURN(
      Node * new_state_value,
      transform.TransformStateRead(this, new_state_read, old_state_read));
  std::vector<std::pair<Node*, Node*>> to_replace{
      {old_state_read, new_state_value}};
  struct NextTransformation {
    Next* old_next;
    Node* new_value;
    std::optional<Node*> new_predicate;
  };
  std::vector<NextTransformation> transforms;
  for (Next* nxt : next_values(old_state_read)) {
    NextTransformation& new_next = transforms.emplace_back();
    new_next.old_next = nxt;
    XLS_ASSIGN_OR_RETURN(new_next.new_value, transform.TransformNextValue(
                                                 this, new_state_read, nxt));
    XLS_RET_CHECK(new_next.new_value->GetType() == new_state_read->GetType())
        << "New value is not compatible type. Expected: "
        << new_state_read->GetType() << " got " << new_next.new_value;
    XLS_ASSIGN_OR_RETURN(
        new_next.new_predicate,
        transform.TransformNextPredicate(this, new_state_read, nxt));
  }

  // We've transformed all the graph elements. Start replacing them.

  // Switch old_state_read's name to a temporary to-remove name
  std::string to_remove_name = UniquifyStateName(
      absl::StrFormat("TO_REMOVE_TRANSFORMED_STATE__%s__", orig_name));
  auto orig_storage = state_elements_.extract(orig_name);
  orig_storage.key() = to_remove_name;
  old_state_element->SetName(to_remove_name);
  old_state_read->SetName(to_remove_name);
  CHECK(state_elements_.insert(std::move(orig_storage)).inserted);

  // Take over the old state element & read names.
  auto new_storage = state_elements_.extract(temp_name);
  new_storage.key() = orig_name;
  new_state_element->SetName(orig_name);
  new_state_read->SetNameDirectly(orig_read_name);
  CHECK(state_elements_.insert(std::move(new_storage)).inserted);

  // Identity-ify the old next nodes and create new ones.
  for (const NextTransformation& nt : transforms) {
    // Make the next
    XLS_ASSIGN_OR_RETURN(
        Next * nxt,
        MakeNodeWithName<Next>(nt.old_next->loc(), new_state_read, nt.new_value,
                               nt.new_predicate, nt.old_next->GetName()));
    to_replace.push_back({nt.old_next, nxt});
    // Identity-ify the old next.
    XLS_RETURN_IF_ERROR(nt.old_next->ReplaceOperandNumber(
        Next::kValueOperand, nt.old_next->state_read()));
  }
  for (const auto& [old_n, new_n] : to_replace) {
    XLS_RETURN_IF_ERROR(old_n->ReplaceUsesWith(
        new_n,
        [&](Node* n) {
          if (n->Is<Next>() && n->As<Next>()->state_read() == old_n) {
            return false;
          }
          return true;
        },
        /*replace_implicit_uses=*/false));
  }
  return new_state_read;
}

}  // namespace xls
