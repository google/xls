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

#include "xls/ir/block_elaboration.h"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <optional>
#include <ostream>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/base/optimization.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_split.h"
#include "absl/types/span.h"
#include "xls/common/casts.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/elaboration.h"
#include "xls/ir/instantiation.h"
#include "xls/ir/node.h"
#include "xls/ir/nodes.h"

namespace xls {

std::ostream& operator<<(std::ostream& os,
                         const ElaboratedNode& node_and_instance) {
  os << absl::StreamFormat("%v (in inst %s)", *node_and_instance.node,
                           node_and_instance.instance->ToString());
  return os;
}

struct InstAndPort {
  Instantiation* inst;
  std::string name;

  bool operator==(const InstAndPort& other) const {
    return inst == other.inst && name == other.name;
  }
};

template <typename H>
H AbslHashValue(H h, const InstAndPort& inst_and_port) {
  return H::combine(std::move(h), inst_and_port.inst->name(),
                    inst_and_port.inst->name());
}

BlockInstance::BlockInstance(
    std::optional<Block*> block, std::optional<Instantiation*> instantiation,
    BlockInstantiationPath&& path,
    std::vector<std::unique_ptr<BlockInstance>> instantiated_blocks)
    : block_(block),
      instantiation_(instantiation),
      path_(std::move(path)),
      child_instances_(std::move(instantiated_blocks)) {
  if (!block.has_value()) {
    return;
  }
  absl::flat_hash_map<Instantiation*, BlockInstance*> instantiation_to_instance;
  instantiation_to_instance.reserve(child_instances_.size());
  for (const std::unique_ptr<BlockInstance>& child_instance :
       child_instances_) {
    if (!child_instance->instantiation().has_value()) {
      continue;
    }
    instantiation_to_instance.insert(
        {*child_instance->instantiation(), child_instance.get()});
  }
  absl::flat_hash_map<InstAndPort, Node*> inst_ports;
  for (Instantiation* inst : block.value()->GetInstantiations()) {
    if (inst->kind() != InstantiationKind::kBlock) {
      continue;
    }
    Block* inst_block =
        down_cast<BlockInstantiation*>(inst)->instantiated_block();
    for (const Block::Port& port : inst_block->GetPorts()) {
      if (std::holds_alternative<Block::ClockPort*>(port)) {
        continue;
      }
      Node* node;
      if (std::holds_alternative<InputPort*>(port)) {
        node = std::get<InputPort*>(port);
      } else if (std::holds_alternative<OutputPort*>(port)) {
        node = std::get<OutputPort*>(port);
      } else {
        ABSL_UNREACHABLE();
      }
      inst_ports.insert(
          {InstAndPort{.inst = inst, .name = Block::PortName(port)}, node});
    }
  }

  parent_to_child_ports_.reserve(inst_ports.size());
  child_to_parent_ports_.reserve(inst_ports.size());
  for (Node* node : block.value()->nodes()) {
    std::string port_name;
    Instantiation* inst;
    if (node->Is<InstantiationInput>()) {
      port_name = node->As<InstantiationInput>()->port_name();
      inst = node->As<InstantiationInput>()->instantiation();
    } else if (node->Is<InstantiationOutput>()) {
      port_name = node->As<InstantiationOutput>()->port_name();
      inst = node->As<InstantiationOutput>()->instantiation();
    } else {
      continue;
    }
    auto inst_itr =
        inst_ports.find(InstAndPort{.inst = inst, .name = port_name});
    if (inst_itr == inst_ports.end()) {
      continue;
    }
    BlockInstance* child_instance = instantiation_to_instance.at(inst);
    parent_to_child_ports_.insert(
        {node,
         ElaboratedNode{.node = inst_itr->second, .instance = child_instance}});
    child_instance->child_to_parent_ports_.insert(
        {inst_itr->second, ElaboratedNode{.node = node, .instance = this}});
  }
}

std::string BlockInstance::ToString() const {
  std::string_view block_name = "<no block>";
  if (block_.has_value()) {
    block_name = block_.value()->name();
  }

  return absl::StrFormat("%s [%s]", block_name, path_.ToString());
}

static absl::StatusOr<std::unique_ptr<BlockInstance>> ElaborateBlock(
    Block* block, std::optional<Instantiation*> instantiation,
    BlockInstantiationPath&& path) {
  std::vector<std::unique_ptr<BlockInstance>> instantiated_blocks;
  for (Instantiation* inst : block->GetInstantiations()) {
    BlockInstantiationPath instantiation_path = path;
    instantiation_path.path.push_back(inst);

    absl::Nullable<Block*> inst_block = nullptr;
    if (inst->kind() == InstantiationKind::kBlock) {
      inst_block = down_cast<BlockInstantiation*>(inst)->instantiated_block();
    }

    // Check for circular dependencies. Walk the original path and see if
    // `instantiation->proc()` appears any where.
    if (inst_block != nullptr &&
        (inst_block == path.top ||
         std::find_if(path.path.begin(), path.path.end(),
                      [&](Instantiation* bi) {
                        return bi->kind() == InstantiationKind::kBlock &&
                               down_cast<BlockInstantiation*>(bi)
                                       ->instantiated_block() == inst_block;
                      }) != path.path.end())) {
      return absl::InternalError(
          absl::StrFormat("Circular dependency in block instantiations: %s",
                          instantiation_path.ToString()));
    }
    if (inst_block != nullptr) {
      XLS_ASSIGN_OR_RETURN(
          std::unique_ptr<BlockInstance> subblock_instance,
          ElaborateBlock(inst_block, inst, std::move(instantiation_path)));
      instantiated_blocks.push_back(std::move(subblock_instance));
    } else {
      // No need to elaborate further, this is a non-block instance.
      // Just add an instance and continue.
      instantiated_blocks.push_back(std::make_unique<BlockInstance>(
          std::nullopt, inst, std::move(instantiation_path),
          std::vector<std::unique_ptr<BlockInstance>>{}));
    }
  }

  return std::make_unique<BlockInstance>(
      block == nullptr ? std::nullopt : std::make_optional(block),
      instantiation, std::move(path), std::move(instantiated_blocks));
}

/* static */ absl::StatusOr<BlockElaboration> BlockElaboration::Elaborate(
    Block* top) {
  BlockElaboration elaboration;
  elaboration.package_ = top->package();
  BlockInstantiationPath path;
  path.top = top;
  XLS_ASSIGN_OR_RETURN(elaboration.top_,
                       ElaborateBlock(top, std::nullopt, std::move(path)));
  elaboration.instance_ptrs_.push_back(elaboration.top_.get());
  elaboration.instances_by_path_[elaboration.top_->path()] =
      elaboration.top_.get();
  elaboration.instances_of_function_[elaboration.top_->block().value()] = {
      elaboration.top_.get()};

  int64_t idx = 0;
  while (idx < elaboration.instance_ptrs_.size()) {
    for (const std::unique_ptr<BlockInstance>& inst :
         elaboration.instance_ptrs_[idx]->child_instances()) {
      elaboration.instance_ptrs_.push_back(inst.get());
      elaboration.instances_by_path_[inst->path()] = inst.get();
      if (inst->block().has_value()) {
        elaboration.instances_of_function_[inst->block().value()].push_back(
            inst.get());
      }
    }
    idx++;
  }
  absl::flat_hash_set<Block*> block_set;
  block_set.reserve(elaboration.instance_ptrs_.size());
  for (BlockInstance* block_instance : elaboration.instance_ptrs_) {
    if (block_instance->block().has_value()) {
      auto [_, inserted] = block_set.insert(*block_instance->block());
      if (inserted) {
        elaboration.blocks_.push_back(*block_instance->block());
      }
    }
  }
  return elaboration;
}

absl::StatusOr<BlockInstance*> BlockElaboration::GetInstance(
    const BlockInstantiationPath& path) const {
  auto it = instances_by_path_.find(path);
  if (it == instances_by_path_.end()) {
    return absl::NotFoundError(
        absl::StrFormat("Instantiation path `%s` does not exist in "
                        "elaboration from proc `%s`",
                        path.ToString(), top()->block().value()->name()));
  }
  return it->second;
}

absl::StatusOr<BlockInstance*> BlockElaboration::GetInstance(
    std::string_view path_str) const {
  XLS_ASSIGN_OR_RETURN(BlockInstantiationPath path, CreatePath(path_str));
  if (path.path.empty()) {
    return top();
  }
  return GetInstance(path);
}

absl::Span<BlockInstance* const> BlockElaboration::GetInstances(
    Block* block) const {
  auto itr = instances_of_function_.find(block);
  if (itr == instances_of_function_.end()) {
    return {};
  }
  return itr->second;
}

absl::StatusOr<BlockInstance*> BlockElaboration::GetUniqueInstance(
    Block* function) const {
  absl::Span<BlockInstance* const> instances = GetInstances(function);
  if (instances.size() != 1) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "There is not exactly 1 instance of `%s`, instance count: %d",
        function->name(), instances.size()));
  }
  return instances.front();
}

absl::StatusOr<BlockInstantiationPath> BlockElaboration::CreatePath(
    std::string_view path_str) const {
  std::vector<std::string_view> pieces = absl::StrSplit(path_str, "::");
  if (pieces.front() != top()->block().value()->name()) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Path top `%s` does not match name of top proc `%s`",
                        pieces.front(), top()->block().value()->name()));
  }
  BlockInstantiationPath path;
  XLS_RET_CHECK(top()->block().has_value())
      << "Elaboration top must be a block!";
  path.top = *top()->block();
  std::optional<Block*> block = path.top;
  for (std::string_view piece : absl::MakeSpan(pieces).subspan(1)) {
    std::vector<std::string_view> parts = absl::StrSplit(piece, "->");
    if (parts.size() != 2) {
      return absl::InvalidArgumentError(
          absl::StrFormat("Invalid component of path `%s`. Expected form: "
                          "`instantiation->proc`.",
                          piece));
    }
    if (!block.has_value()) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "Element `%s` of path is not a block and has no instantiation `%s`.",
          path.path.back()->ToString(), parts[0]));
    }
    absl::StatusOr<Instantiation*> instantiation =
        (*block)->GetInstantiation(parts[0]);
    if (!instantiation.ok()) {
      return absl::InvalidArgumentError(
          absl::StrFormat("`%s` does not have an instantiation named `%s`",
                          (*block)->name(), parts[0]));
    }
    if (parts[1] != BlockInstantiationPath::InstantiatedName(**instantiation)) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "Instantiation `%s` in `%s` instantiates `%s`, but "
          "path element is `%s`",
          parts[0], (*block)->name(),
          BlockInstantiationPath::InstantiatedName(**instantiation), parts[1]));
    }
    path.path.push_back(*instantiation);
    block = BlockInstantiationPath::Instantiated(**instantiation);
  }
  return path;
}

}  // namespace xls
