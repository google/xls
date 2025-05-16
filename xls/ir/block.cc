// Copyright 2021 The XLS Authors
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

#include "xls/ir/block.h"

#include <algorithm>
#include <cstdint>
#include <deque>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <variant>
#include <vector>

#include "absl/base/casts.h"
#include "absl/container/btree_set.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "absl/types/variant.h"
#include "xls/common/casts.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/common/visitor.h"
#include "xls/ir/channel.h"
#include "xls/ir/instantiation.h"
#include "xls/ir/node.h"
#include "xls/ir/nodes.h"
#include "xls/ir/register.h"
#include "xls/ir/source_location.h"
#include "xls/ir/topo_sort.h"
#include "xls/ir/type.h"
#include "xls/ir/value.h"
#include "xls/ir/value_utils.h"

namespace xls {

namespace {

std::string BoolToString(bool b) { return b ? "true" : "false"; };

}  // namespace

std::string ChannelPortMetadata::ToString() const {
  std::string result =
      absl::StrFormat("channel_ports(name=%s, type=%s, direction=%s, kind=%s",
                      channel_name, type->ToString(),
                      direction == ChannelDirection::kSend ? "send" : "receive",
                      ChannelKindToString(channel_kind));
  if (flop_kind != FlopKind::kNone) {
    absl::StrAppendFormat(&result, ", flop=%s", FlopKindToString(flop_kind));
  }
  if (data_port.has_value()) {
    absl::StrAppendFormat(&result, ", data_port=%s", *data_port);
  }
  if (ready_port.has_value()) {
    absl::StrAppendFormat(&result, ", ready_port=%s", *ready_port);
  }
  if (valid_port.has_value()) {
    absl::StrAppendFormat(&result, ", valid_port=%s", *valid_port);
  }
  return result + ")";
}

// For each node in the block, compute the fewest number of registers
// (RegisterWrite/RegisterRead pair) in a path from the node to a user-less node
// (e.g., OutputPort) where path length is measured by the number of registers
// in the path. The graph used to compute distances is the data-flow graph of
// the block augmented with edges extending from register writes to the
// corresponding register read. `register_writes` is a map containing the
// RegisterWrite operation(s) for each register.
static absl::flat_hash_map<Node*, int64_t> ComputeRegisterDepthToLeaves(
    const Block* block,
    const absl::flat_hash_map<Register*, std::vector<RegisterWrite*>>&
        register_writes) {
  absl::flat_hash_map<Node*, int64_t> distances;
  std::deque<Node*> worklist;
  for (Node* node : block->nodes()) {
    if (node->users().empty() && !node->Is<RegisterWrite>()) {
      worklist.push_back(node);
      distances[node] = 0;
    } else {
      distances[node] = std::numeric_limits<int64_t>::max();
    }
  }

  while (!worklist.empty()) {
    Node* node = worklist.front();
    worklist.pop_front();

    auto maybe_update_distance = [&](Node* n, int64_t d) {
      if (d < distances.at(n)) {
        distances[n] = d;
        worklist.push_back(n);
      }
    };
    for (Node* operand : node->operands()) {
      maybe_update_distance(operand, distances.at(node));
    }
    if (RegisterRead* reg_read = dynamic_cast<RegisterRead*>(node)) {
      for (RegisterWrite* reg_write :
           register_writes.at(reg_read->GetRegister())) {
        maybe_update_distance(reg_write, distances.at(node) + 1);
      }
    }
  }
  return distances;
}

// Return the position of port node `n` in the ordered list of ports in the
// block.
static int64_t GetPortPosition(Node* n, const Block* block) {
  int64_t i = 0;
  for (const Block::Port& port : block->GetPorts()) {
    if ((std::holds_alternative<InputPort*>(port) &&
         n == std::get<InputPort*>(port)) ||
        (std::holds_alternative<OutputPort*>(port) &&
         n == std::get<OutputPort*>(port))) {
      return i;
    }
    i++;
  }
  LOG(FATAL) << absl::StreamFormat("Node %s is not a port node", n->ToString());
}

// Return the priority of a node for the purposes of dump order. Nodes with
// lower-valued priorities will be emitted earlier under the constraint that the
// order is a topological sort. The priority scheme results in the IR being
// dumped such that logical pipeline stages tend to get emitted together. A
// std::tuple is returned as the scheme involves tie breakers and std::tuple has
// a lexigraphcially defined less than operator.
using NodePriority = std::tuple<int64_t, int64_t, int64_t, int64_t>;
static NodePriority DumpOrderPriority(
    Node* n, const Block* block,
    const absl::flat_hash_map<Node*, int64_t>& reg_depth) {
  // Priority scheme (highest priority to lowest priority):
  //
  // (0) Input ports. Tie-breaker is port order in the block.
  //
  // (1) Non-port nodes ordered by register depth with tie breaking order:
  //
  //     (i) Register reads
  //
  //     (ii) Non-register nodes
  //
  //     (iii) Register writes
  //
  // (2) Output ports. Tie-breaker is port order in the block.
  //
  // TODO(meheff): 2021/11/03 Add instantiation input/output to priority scheme
  // so operations associated with the same instantiation are grouped together.
  if (n->Is<InputPort>()) {
    return {0, 0, GetPortPosition(n, block), n->id()};
  }
  if (n->Is<OutputPort>()) {
    return {2, 0, GetPortPosition(n, block), n->id()};
  }
  int64_t secondary_priority;
  if (n->Is<RegisterRead>()) {
    secondary_priority = 0;
  } else if (n->Is<RegisterWrite>()) {
    secondary_priority = 2;
  } else {
    secondary_priority = 1;
  }
  return {1, -reg_depth.at(n), secondary_priority, n->id()};
}

std::vector<Node*> Block::DumpOrder() const {
  // Compute the dump order using list scheduling where the priority is computed
  // by DumpOrderPriority. The scheme is chose to improve readability of the IR
  // such that logical pipeline stages tend to be emitted together.

  absl::flat_hash_map<Node*, NodePriority> priorities;
  absl::flat_hash_map<Node*, int64_t> reg_depth =
      ComputeRegisterDepthToLeaves(this, register_writes_);

  VLOG(4) << "Node dump order priorities:";
  for (Node* node : nodes()) {
    priorities[node] = DumpOrderPriority(node, this, reg_depth);
    VLOG(4) << absl::StreamFormat(
        "%s: (%d, %d, %d)", node->GetName(), std::get<0>(priorities.at(node)),
        std::get<1>(priorities.at(node)), std::get<2>(priorities.at(node)));
  }

  auto cmp_function = [&priorities](Node* a, Node* b) {
    return priorities.at(a) < priorities.at(b);
  };

  // The set of nodes ready to be emitted (all operands emitted) is kept in this
  // set, ordered by DumpOrderPriority.
  absl::btree_set<Node*, decltype(cmp_function)> ready_nodes(cmp_function);

  // The count of the unemitted operands of each node. If this value is
  // zero for a node then the node is ready to be emitted.
  absl::flat_hash_map<Node*, int64_t> unemitted_operand_count;
  for (Node* node : nodes()) {
    // To avoid double counting operands, set the unemitted operand count to the
    // number of *unique* operands of the node.
    unemitted_operand_count[node] =
        absl::flat_hash_set<Node*>(node->operands().begin(),
                                   node->operands().end())
            .size();
    if (node->operand_count() == 0) {
      ready_nodes.insert(node);
    }
  }

  std::vector<Node*> order;
  while (!ready_nodes.empty()) {
    Node* node = *ready_nodes.begin();
    ready_nodes.erase(ready_nodes.begin());

    auto place_node = [&](Node* n) {
      order.push_back(n);
      for (Node* user : n->users()) {
        if (--unemitted_operand_count[user] == 0) {
          ready_nodes.insert(user);
        }
      }
    };

    place_node(node);
  }

  CHECK_EQ(order.size(), node_count());
  return order;
}

std::string Block::DumpIr() const {
  std::vector<std::string> port_strings;
  for (const Port& port : GetPorts()) {
    if (std::holds_alternative<ClockPort*>(port)) {
      port_strings.push_back(
          absl::StrFormat("%s: clock", std::get<ClockPort*>(port)->name));
    } else if (std::holds_alternative<InputPort*>(port)) {
      port_strings.push_back(
          absl::StrFormat("%s: %s", std::get<InputPort*>(port)->GetName(),
                          std::get<InputPort*>(port)->GetType()->ToString()));
    } else {
      port_strings.push_back(absl::StrFormat(
          "%s: %s", std::get<OutputPort*>(port)->GetName(),
          std::get<OutputPort*>(port)->operand(0)->GetType()->ToString()));
    }
  }
  std::string res = absl::StrFormat("block %s(%s) {\n", name(),
                                    absl::StrJoin(port_strings, ", "));

  if (provenance_.has_value()) {
    absl::StrAppendFormat(&res, "  #![%v]\n", *provenance_);
  }
  if (reset_behavior_.has_value()) {
    absl::StrAppendFormat(
        &res, "  #![reset(port=\"%s\", asynchronous=%s, active_low=%s)]\n",
        reset_port_.value()->name(),
        BoolToString(reset_behavior_->asynchronous),
        BoolToString(reset_behavior_->active_low));
  }
  for (const std::pair<std::string, ChannelDirection>& channel_direction :
       GetChannelsWithMappedPorts()) {
    absl::StrAppendFormat(
        &res, "  #![%s]\n",
        channel_port_metadata_.at(channel_direction).ToString());
  }
  for (Instantiation* instantiation : GetInstantiations()) {
    absl::StrAppendFormat(&res, "  %s\n", instantiation->ToString());
  }

  for (Register* reg : GetRegisters()) {
    absl::StrAppendFormat(&res, "  %s\n", reg->ToString());
  }

  for (Node* node : DumpOrder()) {
    absl::StrAppend(&res, "  ", node->ToString(), "\n");
  }
  absl::StrAppend(&res, "}\n");
  return res;
}

absl::Status Block::SetPortNameExactly(std::string_view name, Node* node) {
  // TODO(https://github.com/google/xls/issues/477): If this name is an invalid
  // Verilog identifier then an error should be returned.
  XLS_RET_CHECK(node->Is<InputPort>() || node->Is<OutputPort>());

  if (node->GetName() == name) {
    return absl::OkStatus();
  }

  XLS_RET_CHECK(node->function_base() == this);
  for (Node* n : nodes()) {
    if (n->GetName() == name) {
      if (n->Is<InputPort>() || n->Is<OutputPort>()) {
        return absl::InvalidArgumentError(
            absl::StrFormat("Cannot name port `%s` because a port "
                            "already exists with this name",
                            name));
      }
      // Pick a new name for n.
      n->SetNameDirectly(UniquifyNodeName(name));
      XLS_RET_CHECK_NE(n->GetName(), name);
      node->SetNameDirectly(name);
      return absl::OkStatus();
    }
  }
  // Ensure the name is known by the uniquer.
  UniquifyNodeName(name);
  node->SetNameDirectly(name);
  return absl::OkStatus();
}

bool Block::HasInputPort(std::string_view name) const {
  return ports_by_name_.contains(name) &&
         std::holds_alternative<InputPort*>(ports_by_name_.at(name));
}
bool Block::HasOutputPort(std::string_view name) const {
  return ports_by_name_.contains(name) &&
         std::holds_alternative<OutputPort*>(ports_by_name_.at(name));
}

absl::StatusOr<InputPort*> Block::GetInputPort(std::string_view name) const {
  auto port_iter = ports_by_name_.find(name);
  if (port_iter == ports_by_name_.end()) {
    return absl::NotFoundError(
        absl::StrFormat("Input port %s not found", name));
  }

  Port port = port_iter->second;
  if (std::holds_alternative<InputPort*>(port)) {
    return std::get<InputPort*>(port);
  }

  return absl::NotFoundError(
      absl::StrFormat("Port %s is not an input port", name));
}

absl::StatusOr<OutputPort*> Block::GetOutputPort(std::string_view name) const {
  auto port_iter = ports_by_name_.find(name);
  if (port_iter == ports_by_name_.end()) {
    return absl::NotFoundError(
        absl::StrFormat("Output port %s not found", name));
  }

  Port port = port_iter->second;
  if (std::holds_alternative<OutputPort*>(port)) {
    return std::get<OutputPort*>(port);
  }

  return absl::NotFoundError(
      absl::StrFormat("Port %s is not an output port", name));
}

absl::StatusOr<InputPort*> Block::AddInputPort(std::string_view name,
                                               Type* type,
                                               const SourceInfo& loc) {
  if (ports_by_name_.contains(name)) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Block %s already contains a port named %s", this->name(), name));
  }
  InputPort* port = AddNode(std::make_unique<InputPort>(loc, name, type, this));
  if (name != port->GetName()) {
    // The name uniquer changed the given name of the input port to preserve
    // name uniqueness which means another node with this name may already
    // exist.  Force the `port` to have this name potentially be renaming the
    // colliding node.
    XLS_RETURN_IF_ERROR(SetPortNameExactly(name, port));
  }

  ports_by_name_[name] = port;
  ports_.push_back(port);
  input_ports_.push_back(port);
  return port;
}

absl::StatusOr<OutputPort*> Block::AddOutputPort(std::string_view name,
                                                 Node* operand,
                                                 const SourceInfo& loc) {
  if (ports_by_name_.contains(name)) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Block %s already contains a port named %s", this->name(), name));
  }
  OutputPort* port =
      AddNode(std::make_unique<OutputPort>(loc, operand, name, this));

  if (name != port->GetName()) {
    // The name uniquer changed the given name of the output port to preserve
    // name uniqueness which means another node with this name may already
    // exist.  Force the `port` to have this name potentially be renaming the
    // colliding node.
    XLS_RETURN_IF_ERROR(SetPortNameExactly(name, port));
  }
  ports_by_name_[name] = port;
  ports_.push_back(port);
  output_ports_.push_back(port);
  return port;
}

absl::StatusOr<Register*> Block::AddRegister(std::string_view requested_name,
                                             Type* type,
                                             std::optional<Value> reset_value) {
  std::string name =
      register_name_uniquer_.GetSanitizedUniqueName(requested_name);
  if (name != requested_name) {
    VLOG(2) << "Multiple registers with name `" << requested_name
            << "` requested. Using name `" << name << "`.";
  }
  if (reset_value.has_value()) {
    if (type != package()->GetTypeForValue(*reset_value)) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "Reset value `%s` for register `%s` is not of type `%s`",
          reset_value->ToString(), name, type->ToString()));
    }
  }
  registers_[name] =
      std::make_unique<Register>(std::string(name), type, reset_value);
  register_vec_.push_back(registers_[name].get());
  Register* reg = register_vec_.back();
  register_reads_[reg] = {};
  register_writes_[reg] = {};

  return register_vec_.back();
}

absl::StatusOr<Register*> Block::AddRegisterWithZeroResetValue(
    std::string_view requested_name, Type* type) {
  return AddRegister(requested_name, type, ZeroOfType(type));
}

absl::Status Block::RemoveRegister(Register* reg) {
  if (!IsOwned(reg)) {
    return absl::InvalidArgumentError("Register is not owned by block.");
  }

  if (!register_reads_.at(reg).empty() || !register_writes_.at(reg).empty()) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Register %s can't be removed because a register read "
                        "or write operation for this register still exists",
                        reg->name()));
  }
  register_reads_.erase(reg);
  register_writes_.erase(reg);

  auto it = std::find(register_vec_.begin(), register_vec_.end(), reg);
  XLS_RET_CHECK(it != register_vec_.end());
  register_vec_.erase(it);
  registers_.erase(reg->name());
  return absl::OkStatus();
}

absl::StatusOr<Register*> Block::GetRegister(std::string_view name) const {
  if (!registers_.contains(name)) {
    return absl::NotFoundError(absl::StrFormat(
        "Block %s has no register named %s", this->name(), name));
  }
  return registers_.at(name).get();
}

absl::Status Block::AddClockPort(std::string_view name) {
  if (clock_port_.has_value()) {
    return absl::InternalError("Block already has clock");
  }
  if (ports_by_name_.contains(name)) {
    return absl::InternalError(
        absl::StrFormat("Block already has a port named %s", name));
  }
  clock_port_ = ClockPort{std::string(name)};
  ports_.push_back(&clock_port_.value());
  return absl::OkStatus();
}

absl::StatusOr<InputPort*> Block::AddResetPort(std::string_view port_name,
                                               ResetBehavior behavior) {
  if (reset_port_.has_value()) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Block `%s` already has a reset port", name()));
  }
  XLS_ASSIGN_OR_RETURN(InputPort * port,
                       AddInputPort(port_name, package()->GetBitsType(1)));
  XLS_RETURN_IF_ERROR(SetResetPort(port, behavior));
  return port;
}

absl::Status Block::SetResetPort(InputPort* input_port,
                                 ResetBehavior behavior) {
  XLS_RET_CHECK(input_port->function_base() == this);
  if (reset_port_.has_value()) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Block `%s` already has a reset port", name()));
  }
  if (input_port->GetType() != package()->GetBitsType(1)) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Reset port must be single bit type, port `%s` type is %s",
        input_port->name(), input_port->port_type()->ToString()));
  }
  reset_port_ = input_port;
  reset_behavior_ = behavior;
  return absl::OkStatus();
}

absl::Status Block::OverrideResetBehavior(ResetBehavior behavior) {
  if (!reset_port_.has_value()) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Cannot override reset behavior of block `%s`. Block "
                        "does not have a reset port set",
                        name()));
  }
  reset_behavior_ = behavior;
  return absl::OkStatus();
}

// Removes the element `node` from the vector element in the given map at the
// given key. Used for updated register_read_ and register_write_ data members
// of Block.
template <typename KeyT, typename NodeT>
static absl::Status RemoveFromMapOfNodeVectors(
    KeyT key, NodeT* node,
    absl::flat_hash_map<KeyT, std::vector<NodeT*>>* map) {
  XLS_RET_CHECK(map->contains(key)) << node->GetName();
  std::vector<NodeT*>& vector = map->at(key);
  auto it = std::find(vector.begin(), vector.end(), node);
  XLS_RET_CHECK(it != vector.end()) << node->GetName();
  vector.erase(it);
  return absl::OkStatus();
}

// Adds the element `node` to the vector element in the given map at the given
// key. Used for updated register_read_ and register_write_ data members of
// Block.
template <typename KeyT, typename NodeT>
static absl::Status AddToMapOfNodeVectors(
    KeyT key, NodeT* node,
    absl::flat_hash_map<KeyT, std::vector<NodeT*>>* map) {
  XLS_RET_CHECK(map->contains(key)) << node->GetName();
  map->at(key).push_back(node);
  return absl::OkStatus();
}

Node* Block::AddNodeInternal(std::unique_ptr<Node> node) {
  Node* ptr = FunctionBase::AddNodeInternal(std::move(node));
  if (RegisterRead* reg_read = dynamic_cast<RegisterRead*>(ptr)) {
    CHECK_OK(AddToMapOfNodeVectors(reg_read->GetRegister(), reg_read,
                                   &register_reads_));
  } else if (RegisterWrite* reg_write = dynamic_cast<RegisterWrite*>(ptr)) {
    CHECK_OK(AddToMapOfNodeVectors(reg_write->GetRegister(), reg_write,
                                   &register_writes_));
  } else if (InstantiationInput* inst_input =
                 dynamic_cast<InstantiationInput*>(ptr)) {
    instantiation_inputs_[inst_input->instantiation()].push_back(inst_input);
  } else if (InstantiationOutput* inst_output =
                 dynamic_cast<InstantiationOutput*>(ptr)) {
    instantiation_outputs_[inst_output->instantiation()].push_back(inst_output);
  }

  return ptr;
}

absl::Status Block::RemoveNode(Node* n) {
  // Similar to parameters in xls::Functions, input and output ports are also
  // also stored separately as vectors for easy access and to indicate ordering.
  // Fix up these vectors prior to removing the node.
  if (n->Is<InputPort>() || n->Is<OutputPort>()) {
    Port port;
    if (n->Is<InputPort>()) {
      port = n->As<InputPort>();
      auto it = std::find(input_ports_.begin(), input_ports_.end(),
                          n->As<InputPort>());
      XLS_RET_CHECK(it != input_ports_.end()) << absl::StrFormat(
          "input port node %s is not in the vector of input ports",
          n->GetName());
      input_ports_.erase(it);
    } else if (n->Is<OutputPort>()) {
      port = n->As<OutputPort>();
      auto it = std::find(output_ports_.begin(), output_ports_.end(),
                          n->As<OutputPort>());
      XLS_RET_CHECK(it != output_ports_.end()) << absl::StrFormat(
          "output port node %s is not in the vector of output ports",
          n->GetName());
      output_ports_.erase(it);
    }
    std::string port_name = n->GetName();
    auto ports_by_name_it = ports_by_name_.find(port_name);
    XLS_RET_CHECK(ports_by_name_it != ports_by_name_.end());
    ports_by_name_.erase(ports_by_name_it);
    auto port_it = std::find(ports_.begin(), ports_.end(), port);
    XLS_RET_CHECK(port_it != ports_.end()) << absl::StrFormat(
        "port node %s is not in the vector of ports", port_name);
    ports_.erase(port_it);

    if (GetResetPort().has_value() && n == GetResetPort().value()) {
      reset_port_ = std::nullopt;
      reset_behavior_ = std::nullopt;
    }

    // If the port appears in the channel metadata mapping, remove it. This can
    // occur, for example, when removing zero-width ports (e.g., from empty
    // tuple typed channels).
    for (auto& [channel_name, metadata] : channel_port_metadata_) {
      if (metadata.data_port.has_value() && port_name == *metadata.data_port) {
        VLOG(3) << absl::StreamFormat(
            "Data port for channel `%s` in block `%s` removed: %s",
            metadata.channel_name, name(), metadata.data_port.value());
        metadata.data_port = std::nullopt;
        break;
      } else if (metadata.ready_port.has_value() &&
                 port_name == *metadata.ready_port) {
        VLOG(3) << absl::StreamFormat(
            "Ready port for channel `%s` in block `%s` removed: %s",
            metadata.channel_name, name(), metadata.ready_port.value());
        metadata.ready_port = std::nullopt;
        break;
      } else if (metadata.valid_port.has_value() &&
                 port_name == *metadata.valid_port) {
        VLOG(3) << absl::StreamFormat(
            "Valid port for channel `%s` in block `%s` removed: %s",
            metadata.channel_name, name(), metadata.valid_port.value());
        metadata.valid_port = std::nullopt;
        break;
      }
    }
  } else if (RegisterRead* reg_read = dynamic_cast<RegisterRead*>(n)) {
    XLS_RETURN_IF_ERROR(RemoveFromMapOfNodeVectors(reg_read->GetRegister(),
                                                   reg_read, &register_reads_));
  } else if (RegisterWrite* reg_write = dynamic_cast<RegisterWrite*>(n)) {
    XLS_RETURN_IF_ERROR(RemoveFromMapOfNodeVectors(
        reg_write->GetRegister(), reg_write, &register_writes_));
  } else if (InstantiationInput* inst_input =
                 dynamic_cast<InstantiationInput*>(n)) {
    XLS_RETURN_IF_ERROR(RemoveFromMapOfNodeVectors(
        inst_input->instantiation(), inst_input, &instantiation_inputs_));
  } else if (InstantiationOutput* inst_output =
                 dynamic_cast<InstantiationOutput*>(n)) {
    XLS_RETURN_IF_ERROR(RemoveFromMapOfNodeVectors(
        inst_output->instantiation(), inst_output, &instantiation_outputs_));
  }

  return FunctionBase::RemoveNode(n);
}

absl::StatusOr<RegisterRead*> Block::GetRegisterRead(Register* reg) const {
  XLS_RET_CHECK(IsOwned(reg)) << absl::StreamFormat(
      "Block %s does not own register %s (%p)", name(), reg->name(), reg);
  const std::vector<RegisterRead*>& reads = register_reads_.at(reg);
  if (reads.size() == 1) {
    return reads.front();
  }
  if (reads.empty()) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Block %s has no read operation for register %s", name(), reg->name()));
  }
  return absl::InvalidArgumentError(
      absl::StrFormat("Block %s has multiple read operation for register %s",
                      name(), reg->name()));
}

absl::StatusOr<RegisterWrite*> Block::GetRegisterWrite(Register* reg) const {
  XLS_RET_CHECK(register_writes_.contains(reg)) << absl::StreamFormat(
      "Block %s does not have register %s (%p)", name(), reg->name(), reg);
  const std::vector<RegisterWrite*>& writes = register_writes_.at(reg);
  if (writes.size() == 1) {
    return writes.front();
  }
  if (writes.empty()) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Block %s has no write operation for register %s",
                        name(), reg->name()));
  }
  return absl::InvalidArgumentError(
      absl::StrFormat("Block %s has multiple write operation for register %s",
                      name(), reg->name()));
}

absl::Status Block::ReorderPorts(absl::Span<const std::string> port_names) {
  absl::flat_hash_map<std::string, int64_t> port_order;
  for (int64_t i = 0; i < port_names.size(); ++i) {
    port_order[port_names[i]] = i;
  }
  XLS_RET_CHECK_EQ(port_order.size(), port_names.size())
      << "Port order has duplicate names";
  for (const Port& port : GetPorts()) {
    XLS_RET_CHECK(port_order.contains(PortName(port)))
        << absl::StreamFormat("Port order missing port \"%s\"", PortName(port));
  }
  XLS_RET_CHECK_EQ(port_order.size(), GetPorts().size())
      << "Port order includes invalid port names";
  std::sort(ports_.begin(), ports_.end(), [&](const Port& a, const Port& b) {
    return port_order.at(PortName(a)) < port_order.at(PortName(b));
  });
  return absl::OkStatus();
}

/* static */ std::string Block::PortName(const Port& port) {
  return absl::visit(Visitor{
                         [](ClockPort* p) { return p->name; },
                         [](InputPort* p) { return p->GetName(); },
                         [](OutputPort* p) { return p->GetName(); },
                     },
                     port);
}

absl::StatusOr<BlockInstantiation*> Block::AddBlockInstantiation(
    std::string_view name, Block* instantiated_block) {
  XLS_ASSIGN_OR_RETURN(
      absl::StatusOr<Instantiation*> instantiation,
      AddInstantiation(name, std::make_unique<BlockInstantiation>(
                                 name, instantiated_block)));
  return down_cast<BlockInstantiation*>(instantiation.value());
}

absl::StatusOr<FifoInstantiation*> Block::AddFifoInstantiation(
    std::string_view name, FifoConfig fifo_config, Type* data_type,
    std::optional<std::string_view> channel) {
  XLS_RET_CHECK(package()->IsOwnedType(data_type));
  XLS_ASSIGN_OR_RETURN(
      absl::StatusOr<Instantiation*> instantiation,
      AddInstantiation(name,
                       std::make_unique<FifoInstantiation>(
                           name, fifo_config, data_type, channel, package())));
  return down_cast<FifoInstantiation*>(instantiation.value());
}

absl::StatusOr<Instantiation*> Block::AddInstantiation(
    std::string_view name, std::unique_ptr<Instantiation> instantiation) {
  if (instantiations_.contains(name)) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Instantiation already exists with name %s", name));
  }

  Instantiation* instantiation_ptr = instantiation.get();
  instantiations_[name] = std::move(instantiation);

  instantiation_vec_.push_back(instantiation_ptr);
  instantiation_inputs_[instantiation_ptr] = {};
  instantiation_outputs_[instantiation_ptr] = {};

  return instantiation_ptr;
}

absl::Status Block::ReplaceInstantiationWith(
    Instantiation* old_inst, Instantiation* new_inst,
    absl::flat_hash_map<std::string, std::string> port_renaming) {
  XLS_RET_CHECK(IsOwned(old_inst));
  XLS_RET_CHECK(IsOwned(new_inst)) << "must add instantiation to this block "
                                      "before replacing uses of another.";
  std::vector<InstantiationInput*> inps(instantiation_inputs_.at(old_inst));
  std::vector<InstantiationOutput*> outs(instantiation_outputs_.at(old_inst));

  XLS_ASSIGN_OR_RETURN(InstantiationType old_type_orig, old_inst->type());
  XLS_ASSIGN_OR_RETURN(InstantiationType new_type, new_inst->type());

  // Validate renaming scheme:
  for (auto& [orig_port_name, new_port_name] : port_renaming) {
    XLS_RET_CHECK(old_type_orig.input_types().contains(orig_port_name) ||
                  old_type_orig.output_types().contains(orig_port_name))
        << "attempting to rename port that does not exists";

    // Note: It would be possible to rename a port to the name of an
    // already existing port, if that port is also getting renamed. This,
    // however, makes analysis significantly more complex, particularly
    // because it would have to deal with "rename loops":
    //
    // port_renaming = { {"a", "b"}, {"b", "a"} }; // !!
    //
    // For now, we simply don't allow it:
    XLS_RET_CHECK((!old_type_orig.input_types().contains(new_port_name)) &&
                  (!old_type_orig.output_types().contains(new_port_name)))
        << "new port name already exists";
  }

  // Construct old type with renamed ports:
  auto input_types = old_type_orig.input_types();
  auto output_types = old_type_orig.output_types();
  for (auto& [orig_port_name, new_port_name] : port_renaming) {
    if (input_types.contains(orig_port_name)) {
      auto node = input_types.extract(orig_port_name);
      node.key() = new_port_name;
      input_types.insert(std::move(node));
    }
    if (output_types.contains(orig_port_name)) {
      auto node = output_types.extract(orig_port_name);
      node.key() = new_port_name;
      output_types.insert(std::move(node));
    }
  }
  InstantiationType old_type_renamed =
      InstantiationType(input_types, output_types);

  // Validate that signature matches after renaming:
  XLS_RET_CHECK(old_type_renamed == new_type)
      << "Type mismatch of instantiations";

  // Replace instantiation:
  for (InstantiationInput* inp : inps) {
    auto renaming_rule = port_renaming.find(inp->port_name());
    auto new_name = renaming_rule == port_renaming.end()
                        ? inp->port_name()
                        : renaming_rule->second;
    XLS_RETURN_IF_ERROR(inp->ReplaceUsesWithNew<InstantiationInput>(
                               inp->data(), new_inst, new_name)
                            .status());
    XLS_RETURN_IF_ERROR(RemoveNode(inp));
  }
  for (InstantiationOutput* out : outs) {
    auto renaming_rule = port_renaming.find(out->port_name());
    auto new_name = renaming_rule == port_renaming.end()
                        ? out->port_name()
                        : renaming_rule->second;
    XLS_RETURN_IF_ERROR(
        out->ReplaceUsesWithNew<InstantiationOutput>(new_inst, new_name)
            .status());
    XLS_RETURN_IF_ERROR(RemoveNode(out));
  }
  return RemoveInstantiation(old_inst);
}

absl::Status Block::RemoveInstantiation(Instantiation* instantiation) {
  if (!IsOwned(instantiation)) {
    return absl::InvalidArgumentError("Instantiation is not owned by block.");
  }
  if (!instantiation_inputs_.at(instantiation).empty() ||
      !instantiation_outputs_.at(instantiation).empty()) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Instantiation %s can't be removed because an input or "
                        "output operation for this instantiation still exists",
                        instantiation->name()));
  }
  instantiation_inputs_.erase(instantiation);
  instantiation_outputs_.erase(instantiation);

  auto it = std::find(instantiation_vec_.begin(), instantiation_vec_.end(),
                      instantiation);
  XLS_RET_CHECK(it != instantiation_vec_.end());
  instantiation_vec_.erase(it);
  instantiations_.erase(instantiation->name());
  return absl::OkStatus();
}

absl::StatusOr<Instantiation*> Block::GetInstantiation(
    std::string_view name) const {
  if (!instantiations_.contains(name)) {
    return absl::NotFoundError(absl::StrFormat(
        "Block %s has no instantiation named %s", this->name(), name));
  }
  return instantiations_.at(name).get();
}

absl::Span<InstantiationInput* const> Block::GetInstantiationInputs(
    Instantiation* instantiation) const {
  CHECK(IsOwned(instantiation))
      << absl::StreamFormat("Block %s does not have instantiation %s (%p)",
                            name(), instantiation->name(), instantiation);
  return instantiation_inputs_.at(instantiation);
}

absl::Span<InstantiationOutput* const> Block::GetInstantiationOutputs(
    Instantiation* instantiation) const {
  CHECK(IsOwned(instantiation))
      << absl::StreamFormat("Block %s does not have instantiation %s (%p)",
                            name(), instantiation->name(), instantiation);
  return instantiation_outputs_.at(instantiation);
}

absl::StatusOr<Block*> Block::Clone(
    std::string_view new_name, Package* target_package,
    const absl::flat_hash_map<std::string, std::string>& reg_name_map,
    const absl::flat_hash_map<const Block*, Block*>& block_instantiation_map)
    const {
  absl::flat_hash_map<Node*, Node*> original_to_clone;
  absl::flat_hash_map<Register*, Register*> register_map;
  absl::flat_hash_map<Instantiation*, Instantiation*> instantiation_map;

  if (target_package == nullptr) {
    target_package = package();
  }

  Block* cloned_block = target_package->AddBlock(
      std::make_unique<Block>(new_name, target_package));

  std::optional<std::string> clk_port_name;
  for (const Port& port : GetPorts()) {
    if (std::holds_alternative<ClockPort*>(port)) {
      auto old_name = std::get<ClockPort*>(port)->name;
      clk_port_name = reg_name_map.contains(old_name)
                          ? reg_name_map.at(old_name)
                          : old_name;
      XLS_RETURN_IF_ERROR(cloned_block->AddClockPort(*clk_port_name));
    }
  }
  auto to_new_name = [&](Register* r) {
    auto it = reg_name_map.find(r->name());
    if (it == reg_name_map.end()) {
      return r->name();
    }
    return it->second;
  };
  for (Register* reg : GetRegisters()) {
    XLS_ASSIGN_OR_RETURN(Type * mapped_type,
                         target_package->MapTypeFromOtherPackage(reg->type()));
    XLS_ASSIGN_OR_RETURN(register_map[reg], cloned_block->AddRegister(
                                                to_new_name(reg), mapped_type,
                                                reg->reset_value()));
  }

  for (Instantiation* inst : GetInstantiations()) {
    if (inst->kind() == InstantiationKind::kBlock) {
      XLS_ASSIGN_OR_RETURN(BlockInstantiation * block_inst,
                           inst->AsBlockInstantiation());
      Block* new_inst;
      if (!block_instantiation_map.contains(block_inst->instantiated_block())) {
        XLS_RET_CHECK_EQ(target_package, package())
            << "No definition of block " << block_inst->name() << "("
            << block_inst->instantiated_block()->name() << ") provided.";
        new_inst = block_inst->instantiated_block();
      } else {
        new_inst = block_instantiation_map.at(block_inst->instantiated_block());
      }
      XLS_ASSIGN_OR_RETURN(
          instantiation_map[inst],
          cloned_block->AddBlockInstantiation(block_inst->name(), new_inst));
    } else if (inst->kind() == InstantiationKind::kExtern) {
      XLS_ASSIGN_OR_RETURN(ExternInstantiation * extern_inst,
                           inst->AsExternInstantiation());
      XLS_ASSIGN_OR_RETURN(
          instantiation_map[inst],
          cloned_block->AddInstantiation(
              inst->name(), std::make_unique<ExternInstantiation>(
                                inst->name(), extern_inst->function())));
    } else {
      XLS_RET_CHECK_EQ(inst->kind(), InstantiationKind::kFifo)
          << "Unknown instantiation kind";
      XLS_ASSIGN_OR_RETURN(FifoInstantiation * fifo_inst,
                           inst->AsFifoInstantiation());
      XLS_ASSIGN_OR_RETURN(
          Type * data_type,
          target_package->MapTypeFromOtherPackage(fifo_inst->data_type()));
      XLS_ASSIGN_OR_RETURN(instantiation_map[inst],
                           cloned_block->AddFifoInstantiation(
                               inst->name(), fifo_inst->fifo_config(),
                               data_type, fifo_inst->channel_name()));
    }
  }

  for (Node* node : TopoSort(const_cast<Block*>(this))) {
    std::vector<Node*> cloned_operands;
    for (Node* operand : node->operands()) {
      cloned_operands.push_back(original_to_clone.at(operand));
    }

    if (node->Is<InputPort>()) {
      InputPort* src = node->As<InputPort>();
      XLS_ASSIGN_OR_RETURN(
          Type * mapped_type,
          target_package->MapTypeFromOtherPackage(src->GetType()));
      XLS_ASSIGN_OR_RETURN(
          original_to_clone[node],
          cloned_block->AddInputPort(src->name(), mapped_type, src->loc()));
    } else if (node->Is<OutputPort>()) {
      OutputPort* src = node->As<OutputPort>();
      XLS_ASSIGN_OR_RETURN(original_to_clone[node],
                           cloned_block->AddOutputPort(
                               src->name(), cloned_operands[0], src->loc()));
    } else if (node->Is<RegisterRead>()) {
      RegisterRead* src = node->As<RegisterRead>();
      XLS_ASSIGN_OR_RETURN(
          original_to_clone[node],
          cloned_block->MakeNodeWithName<RegisterRead>(
              src->loc(), register_map.at(src->GetRegister()), src->GetName()));
    } else if (node->Is<RegisterWrite>()) {
      RegisterWrite* src = node->As<RegisterWrite>();
      XLS_ASSIGN_OR_RETURN(
          original_to_clone[node],
          cloned_block->MakeNodeWithName<RegisterWrite>(
              src->loc(), cloned_operands[0],
              src->load_enable().has_value()
                  ? std::optional<Node*>(
                        original_to_clone.at(*src->load_enable()))
                  : std::nullopt,
              src->reset().has_value()
                  ? std::optional<Node*>(original_to_clone.at(*src->reset()))
                  : std::nullopt,
              register_map.at(src->GetRegister()), src->GetName()));
    } else if (node->Is<InstantiationInput>()) {
      InstantiationInput* src = node->As<InstantiationInput>();
      XLS_ASSIGN_OR_RETURN(original_to_clone[node],
                           cloned_block->MakeNodeWithName<InstantiationInput>(
                               src->loc(), cloned_operands[0],
                               instantiation_map.at(src->instantiation()),
                               src->port_name(), src->GetName()));
    } else if (node->Is<InstantiationOutput>()) {
      InstantiationOutput* src = node->As<InstantiationOutput>();
      XLS_ASSIGN_OR_RETURN(
          original_to_clone[node],
          cloned_block->MakeNodeWithName<InstantiationOutput>(
              src->loc(), instantiation_map.at(src->instantiation()),
              src->port_name(), src->GetName()));
    } else {
      XLS_ASSIGN_OR_RETURN(
          original_to_clone[node],
          node->CloneInNewFunction(cloned_operands, cloned_block));
    }
  }

  // Copy over reset and channel port metadata.
  if (GetResetPort().has_value()) {
    XLS_RETURN_IF_ERROR(cloned_block->SetResetPort(
        original_to_clone[*GetResetPort()]->As<InputPort>(),
        *GetResetBehavior()));
  }
  cloned_block->channel_port_metadata_ = channel_port_metadata_;

  {
    std::vector<std::string> correct_ordering;
    for (const Port& port : GetPorts()) {
      if (std::holds_alternative<InputPort*>(port)) {
        std::string_view view = std::get<InputPort*>(port)->name();
        correct_ordering.push_back(std::string(view.begin(), view.end()));
      } else if (std::holds_alternative<OutputPort*>(port)) {
        std::string_view view = std::get<OutputPort*>(port)->name();
        correct_ordering.push_back(std::string(view.begin(), view.end()));
      } else if (std::holds_alternative<ClockPort*>(port)) {
        correct_ordering.push_back(*clk_port_name);
      }
    }
    XLS_RETURN_IF_ERROR(cloned_block->ReorderPorts(correct_ordering));
  }

  return cloned_block;
}

absl::Status Block::AddChannelPortMetadata(ChannelPortMetadata metadata) {
  // Verify that the specified ports exist.
  if (metadata.direction == ChannelDirection::kReceive) {
    if (metadata.data_port.has_value()) {
      XLS_RET_CHECK(HasInputPort(metadata.data_port.value()))
          << absl::StrFormat(
                 "No input port `%s` on block `%s` (data port for channel "
                 "`%s`)",
                 metadata.data_port.value(), name(), metadata.channel_name);
    }
    if (metadata.valid_port.has_value()) {
      XLS_RET_CHECK(HasInputPort(metadata.valid_port.value()))
          << absl::StrFormat(
                 "No input port `%s` on block `%s` (valid port for channel "
                 "`%s`)",
                 metadata.valid_port.value(), name(), metadata.channel_name);
    }
    if (metadata.ready_port.has_value()) {
      XLS_RET_CHECK(HasOutputPort(metadata.ready_port.value()))
          << absl::StrFormat(
                 "No output port `%s` on block `%s` (ready port for channel "
                 "`%s`)",
                 metadata.ready_port.value(), name(), metadata.channel_name);
    }
  } else {
    CHECK_EQ(metadata.direction, ChannelDirection::kSend);
    if (metadata.data_port.has_value()) {
      XLS_RET_CHECK(HasOutputPort(metadata.data_port.value()))
          << absl::StrFormat(
                 "No output port `%s` on block `%s` (data port for channel "
                 "`%s`)",
                 metadata.data_port.value(), name(), metadata.channel_name);
    }
    if (metadata.valid_port.has_value()) {
      XLS_RET_CHECK(HasOutputPort(metadata.valid_port.value()))
          << absl::StrFormat(
                 "No output port `%s` on block `%s` (valid port for channel "
                 "`%s`)",
                 metadata.data_port.value(), name(), metadata.channel_name);
    }
    if (metadata.ready_port.has_value()) {
      XLS_RET_CHECK(HasInputPort(metadata.ready_port.value()))
          << absl::StrFormat(
                 "No input port `%s` on block `%s` (ready port for channel "
                 "`%s`)",
                 metadata.data_port.value(), name(), metadata.channel_name);
    }
  }
  VLOG(3) << "AddChannelPortMetadata for block `" << name()
          << "`: " << metadata.ToString();
  channel_port_metadata_[{metadata.channel_name, metadata.direction}] =
      std::move(metadata);
  return absl::OkStatus();
}

absl::Status Block::AddChannelPortMetadata(
    ChannelRef channel, ChannelDirection direction,
    std::optional<std::string> data_port, std::optional<std::string> valid_port,
    std::optional<std::string> ready_port) {
  FlopKind flop_kind;
  if (std::holds_alternative<Channel*>(channel)) {
    if (StreamingChannel* streaming_channel =
            dynamic_cast<StreamingChannel*>(std::get<Channel*>(channel));
        streaming_channel != nullptr) {
      flop_kind =
          direction == ChannelDirection::kReceive
              ? streaming_channel->channel_config().input_flop_kind().value_or(
                    FlopKind::kNone)
              : streaming_channel->channel_config().output_flop_kind().value_or(
                    FlopKind::kNone);
    } else {
      flop_kind = FlopKind::kNone;
    }
  } else {
    flop_kind = std::get<ChannelInterface*>(channel)->flop_kind();
  }
  return AddChannelPortMetadata(
      ChannelPortMetadata{.channel_name = std::string{ChannelRefName(channel)},
                          .type = ChannelRefType(channel),
                          .direction = direction,
                          .channel_kind = ChannelRefKind(channel),
                          .flop_kind = flop_kind,
                          .data_port = data_port,
                          .valid_port = valid_port,
                          .ready_port = ready_port});
}

absl::StatusOr<ChannelPortMetadata> Block::GetChannelPortMetadata(
    std::string_view channel_name, ChannelDirection direction) const {
  XLS_ASSIGN_OR_RETURN(const ChannelPortMetadata* metadata,
                       GetChannelPortMetadataInternal(channel_name, direction));
  return *metadata;
}

absl::StatusOr<const ChannelPortMetadata*>
Block::GetChannelPortMetadataInternal(std::string_view channel_name,
                                      ChannelDirection direction) const {
  auto it =
      channel_port_metadata_.find(std::pair<std::string, ChannelDirection>(
          std::string{channel_name}, direction));
  if (it == channel_port_metadata_.end()) {
    return absl::NotFoundError(absl::StrFormat(
        "Block `%s` has no port metadata for channel `%s` in the %s direction",
        name(), channel_name,
        direction == ChannelDirection::kSend ? "send" : "receive"));
  }
  return &it->second;
}

absl::StatusOr<PortNode*> Block::GetPortNode(std::string_view name) const {
  auto it = ports_by_name_.find(name);
  if (it == ports_by_name_.end()) {
    return absl::NotFoundError(absl::StrFormat(
        "Block `%s` has no port node named  `%s`", this->name(), name));
  }
  Port port = it->second;
  if (std::holds_alternative<InputPort*>(port)) {
    return absl::implicit_cast<PortNode*>(std::get<InputPort*>(port));
  }
  if (std::holds_alternative<OutputPort*>(port)) {
    return absl::implicit_cast<PortNode*>(std::get<OutputPort*>(port));
  }
  return absl::InvalidArgumentError("Cannot call GetPortNode on clock port");
}

absl::StatusOr<std::optional<PortNode*>> Block::GetReadyPortForChannel(
    std::string_view channel_name, ChannelDirection direction) {
  XLS_ASSIGN_OR_RETURN(const ChannelPortMetadata* metadata,
                       GetChannelPortMetadataInternal(channel_name, direction));
  if (!metadata->ready_port.has_value()) {
    return std::nullopt;
  }
  return GetPortNode(*metadata->ready_port);
}

absl::StatusOr<std::optional<PortNode*>> Block::GetValidPortForChannel(
    std::string_view channel_name, ChannelDirection direction) {
  XLS_ASSIGN_OR_RETURN(const ChannelPortMetadata* metadata,
                       GetChannelPortMetadataInternal(channel_name, direction));
  if (!metadata->valid_port.has_value()) {
    return std::nullopt;
  }
  return GetPortNode(*metadata->valid_port);
}

absl::StatusOr<std::optional<PortNode*>> Block::GetDataPortForChannel(
    std::string_view channel_name, ChannelDirection direction) {
  XLS_ASSIGN_OR_RETURN(const ChannelPortMetadata* metadata,
                       GetChannelPortMetadataInternal(channel_name, direction));
  if (!metadata->data_port.has_value()) {
    return std::nullopt;
  }
  return GetPortNode(*metadata->data_port);
}

std::vector<std::pair<std::string, ChannelDirection>>
Block::GetChannelsWithMappedPorts() const {
  std::vector<std::pair<std::string, ChannelDirection>> result;
  for (const auto& [key, _] : channel_port_metadata_) {
    result.push_back(key);
  }
  std::sort(result.begin(), result.end());
  return result;
}

}  // namespace xls
