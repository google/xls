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
#include <deque>
#include <memory>
#include <optional>
#include <ostream>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/base/nullability.h"
#include "absl/base/optimization.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"
#include "absl/types/span.h"
#include "xls/common/casts.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/elaborated_block_dfs_visitor.h"
#include "xls/ir/elaboration.h"
#include "xls/ir/instantiation.h"
#include "xls/ir/node.h"
#include "xls/ir/nodes.h"
#include "xls/ir/op.h"

namespace xls {
namespace {
// Returns the predecessor of `node_and_instance` that exists in a different
// instance, if it exists.
//
// If the input node is an InputPort, this returns the InstantiationInput in the
// parent instance. If the input node is an InstantiationOutput, this returns
// the OutputPort in the child instance.
std::optional<ElaboratedNode> InterInstancePredecessor(
    const ElaboratedNode& node_and_instance) {
  Node* node = node_and_instance.node;
  if (node->Is<InputPort>()) {
    auto itr = node_and_instance.instance->child_to_parent_ports().find(
        node_and_instance.node);
    if (itr != node_and_instance.instance->child_to_parent_ports().end()) {
      return itr->second;
    }
  }
  if (node->Is<InstantiationOutput>()) {
    auto itr = node_and_instance.instance->parent_to_child_ports().find(
        node_and_instance.node);
    if (itr != node_and_instance.instance->parent_to_child_ports().end()) {
      return itr->second;
    }
  }

  return std::nullopt;
}

// Returns the successor of `node_and_instance` that exists in a different
// instance, if it exists.
//
// If the input node is an OutputPort, this returns the InstantiationOutput in
// the parent instance. If the input node is an InstantiationInput, this returns
// the InputPort in the child instance.
std::optional<ElaboratedNode> InterInstanceSuccessor(
    const ElaboratedNode& node_and_instance) {
  Node* node = node_and_instance.node;
  if (node->Is<OutputPort>()) {
    auto itr = node_and_instance.instance->child_to_parent_ports().find(
        node_and_instance.node);
    if (itr != node_and_instance.instance->child_to_parent_ports().end()) {
      return itr->second;
    }
  }
  if (node->Is<InstantiationInput>()) {
    auto itr = node_and_instance.instance->parent_to_child_ports().find(
        node_and_instance.node);
    if (itr != node_and_instance.instance->parent_to_child_ports().end()) {
      return itr->second;
    }
  }
  return std::nullopt;
}

std::string MakeRegisterPrefix(const BlockInstantiationPath& path) {
  if (path.path.empty()) {
    return "";
  }
  return absl::StrCat(
      absl::StrJoin(path.path, "::",
                    [](std::string* out, const Instantiation* inst) {
                      out->append(inst->name());
                    }),
      "::");
}

}  // namespace

std::string ElaboratedNode::ToString() const {
  return absl::StrFormat("%s (%s)", node->ToString(), instance->ToString());
}

absl::Status ElaboratedNode::Accept(ElaboratedBlockDfsVisitor& visitor) {
  if (visitor.IsVisited(*this)) {
    return absl::OkStatus();
  }
  if (visitor.IsTraversing(*this)) {
    std::vector<std::string> cycle_names = {ToString()};
    auto first_traversing_node =
        [&](ElaboratedNode node) -> std::optional<ElaboratedNode> {
      ElaboratedNode to_check;
      for (Node* operand : node.node->operands()) {
        to_check.node = operand;
        to_check.instance = node.instance;
        if (visitor.IsTraversing(to_check)) {
          return to_check;
        }
      }
      std::optional<ElaboratedNode> predecessor =
          InterInstancePredecessor(node);
      if (predecessor.has_value() && visitor.IsTraversing(*predecessor)) {
        return predecessor;
      }
      return std::nullopt;
    };
    ElaboratedNode current_node = *this;
    do {
      std::optional<ElaboratedNode> first_traversing =
          first_traversing_node(current_node);
      bool broke = first_traversing.has_value();
      CHECK(broke);
      current_node = *first_traversing;
      cycle_names.push_back(current_node.ToString());
    } while (current_node != *this);
    return absl::InternalError(absl::StrFormat(
        "Cycle detected: [\n%s\n]", absl::StrJoin(cycle_names, " ->\n")));
  }
  VLOG(5) << "Traversing node: " << ToString();
  visitor.SetTraversing(*this);
  for (Node* operand : node->operands()) {
    ElaboratedNode operand_node{.node = operand, .instance = instance};
    XLS_RETURN_IF_ERROR(operand_node.Accept(visitor));
  }
  std::optional<ElaboratedNode> predecessor = InterInstancePredecessor(*this);
  if (predecessor.has_value()) {
    XLS_RETURN_IF_ERROR(predecessor->Accept(visitor));
  }

  visitor.UnsetTraversing(*this);
  visitor.MarkVisited(*this);
  return VisitSingleNode(visitor);
  VLOG(5) << "Traversed node: " << ToString();
  return absl::OkStatus();
}

absl::Status ElaboratedNode::VisitSingleNode(
    ElaboratedBlockDfsVisitor& visitor) {
  VLOG(5) << "Visiting elaborated node: " << ToString() << "\n";
  switch (node->op()) {
    case Op::kAdd:
      return visitor.HandleAdd(down_cast<BinOp*>(node), instance);

    case Op::kAnd:
      return visitor.HandleNaryAnd(down_cast<NaryOp*>(node), instance);

    case Op::kAndReduce:
      return visitor.HandleAndReduce(down_cast<BitwiseReductionOp*>(node),
                                     instance);

    case Op::kAssert:
      return visitor.HandleAssert(down_cast<Assert*>(node), instance);

    case Op::kCover:
      return visitor.HandleCover(down_cast<Cover*>(node), instance);

    case Op::kTrace:
      return visitor.HandleTrace(down_cast<Trace*>(node), instance);

    case Op::kReceive:
      return visitor.HandleReceive(down_cast<Receive*>(node), instance);

    case Op::kSend:
      return visitor.HandleSend(down_cast<Send*>(node), instance);

    case Op::kNand:
      return visitor.HandleNaryNand(down_cast<NaryOp*>(node), instance);

    case Op::kNor:
      return visitor.HandleNaryNor(down_cast<NaryOp*>(node), instance);

    case Op::kAfterAll:
      return visitor.HandleAfterAll(down_cast<AfterAll*>(node), instance);

    case Op::kMinDelay:
      return visitor.HandleMinDelay(down_cast<MinDelay*>(node), instance);

    case Op::kArray:
      return visitor.HandleArray(down_cast<Array*>(node), instance);

    case Op::kBitSlice:
      return visitor.HandleBitSlice(down_cast<BitSlice*>(node), instance);

    case Op::kDynamicBitSlice:
      return visitor.HandleDynamicBitSlice(down_cast<DynamicBitSlice*>(node),
                                           instance);

    case Op::kBitSliceUpdate:
      return visitor.HandleBitSliceUpdate(down_cast<BitSliceUpdate*>(node),
                                          instance);

    case Op::kConcat:
      return visitor.HandleConcat(down_cast<Concat*>(node), instance);

    case Op::kDecode:
      return visitor.HandleDecode(down_cast<Decode*>(node), instance);

    case Op::kEncode:
      return visitor.HandleEncode(down_cast<Encode*>(node), instance);

    case Op::kEq:
      return visitor.HandleEq(down_cast<CompareOp*>(node), instance);

    case Op::kIdentity:
      return visitor.HandleIdentity(down_cast<UnOp*>(node), instance);

    case Op::kArrayIndex:
      return visitor.HandleArrayIndex(down_cast<ArrayIndex*>(node), instance);

    case Op::kArrayUpdate:
      return visitor.HandleArrayUpdate(down_cast<ArrayUpdate*>(node), instance);

    case Op::kArrayConcat:
      return visitor.HandleArrayConcat(down_cast<ArrayConcat*>(node), instance);

    case Op::kArraySlice:
      return visitor.HandleArraySlice(down_cast<ArraySlice*>(node), instance);

    case Op::kInvoke:
      return visitor.HandleInvoke(down_cast<Invoke*>(node), instance);

    case Op::kCountedFor:
      return visitor.HandleCountedFor(down_cast<CountedFor*>(node), instance);

    case Op::kDynamicCountedFor:
      return visitor.HandleDynamicCountedFor(
          down_cast<DynamicCountedFor*>(node), instance);

    case Op::kLiteral:
      return visitor.HandleLiteral(down_cast<Literal*>(node), instance);

    case Op::kMap:
      return visitor.HandleMap(down_cast<Map*>(node), instance);

    case Op::kNe:
      return visitor.HandleNe(down_cast<CompareOp*>(node), instance);

    case Op::kNeg:
      return visitor.HandleNeg(down_cast<UnOp*>(node), instance);

    case Op::kNot:
      return visitor.HandleNot(down_cast<UnOp*>(node), instance);

    case Op::kOneHot:
      return visitor.HandleOneHot(down_cast<OneHot*>(node), instance);

    case Op::kOneHotSel:
      return visitor.HandleOneHotSel(down_cast<OneHotSelect*>(node), instance);

    case Op::kPrioritySel:
      return visitor.HandlePrioritySel(down_cast<PrioritySelect*>(node),
                                       instance);

    case Op::kOr:
      return visitor.HandleNaryOr(down_cast<NaryOp*>(node), instance);

    case Op::kOrReduce:
      return visitor.HandleOrReduce(down_cast<BitwiseReductionOp*>(node),
                                    instance);

    case Op::kParam:
      return visitor.HandleParam(down_cast<Param*>(node), instance);

    case Op::kNext:
      return visitor.HandleNext(down_cast<Next*>(node), instance);

    case Op::kRegisterRead:
      return visitor.HandleRegisterRead(down_cast<RegisterRead*>(node),
                                        instance);

    case Op::kRegisterWrite:
      return visitor.HandleRegisterWrite(down_cast<RegisterWrite*>(node),
                                         instance);

    case Op::kReverse:
      return visitor.HandleReverse(down_cast<UnOp*>(node), instance);

    case Op::kSDiv:
      return visitor.HandleSDiv(down_cast<BinOp*>(node), instance);

    case Op::kSel:
      return visitor.HandleSel(down_cast<Select*>(node), instance);

    case Op::kSGt:
      return visitor.HandleSGt(down_cast<CompareOp*>(node), instance);

    case Op::kSGe:
      return visitor.HandleSGe(down_cast<CompareOp*>(node), instance);

    case Op::kShll:
      return visitor.HandleShll(down_cast<BinOp*>(node), instance);

    case Op::kShra:
      return visitor.HandleShra(down_cast<BinOp*>(node), instance);

    case Op::kShrl:
      return visitor.HandleShrl(down_cast<BinOp*>(node), instance);

    case Op::kSLe:
      return visitor.HandleSLe(down_cast<CompareOp*>(node), instance);

    case Op::kSLt:
      return visitor.HandleSLt(down_cast<CompareOp*>(node), instance);

    case Op::kSMod:
      return visitor.HandleSMod(down_cast<BinOp*>(node), instance);

    case Op::kSMul:
      return visitor.HandleSMul(down_cast<ArithOp*>(node), instance);

    case Op::kSMulp:
      return visitor.HandleSMulp(down_cast<PartialProductOp*>(node), instance);

    case Op::kSub:
      return visitor.HandleSub(down_cast<BinOp*>(node), instance);

    case Op::kTupleIndex:
      return visitor.HandleTupleIndex(down_cast<TupleIndex*>(node), instance);

    case Op::kTuple:
      return visitor.HandleTuple(down_cast<Tuple*>(node), instance);

    case Op::kUDiv:
      return visitor.HandleUDiv(down_cast<BinOp*>(node), instance);

    case Op::kUGe:
      return visitor.HandleUGe(down_cast<CompareOp*>(node), instance);

    case Op::kUGt:
      return visitor.HandleUGt(down_cast<CompareOp*>(node), instance);

    case Op::kULe:
      return visitor.HandleULe(down_cast<CompareOp*>(node), instance);

    case Op::kULt:
      return visitor.HandleULt(down_cast<CompareOp*>(node), instance);

    case Op::kUMod:
      return visitor.HandleUMod(down_cast<BinOp*>(node), instance);

    case Op::kUMul:
      return visitor.HandleUMul(down_cast<ArithOp*>(node), instance);

    case Op::kUMulp:
      return visitor.HandleUMulp(down_cast<PartialProductOp*>(node), instance);

    case Op::kXor:
      return visitor.HandleNaryXor(down_cast<NaryOp*>(node), instance);

    case Op::kXorReduce:
      return visitor.HandleXorReduce(down_cast<BitwiseReductionOp*>(node),
                                     instance);

    case Op::kSignExt:
      return visitor.HandleSignExtend(down_cast<ExtendOp*>(node), instance);

    case Op::kZeroExt:
      return visitor.HandleZeroExtend(down_cast<ExtendOp*>(node), instance);

    case Op::kInputPort:
      return visitor.HandleInputPort(down_cast<InputPort*>(node), instance);

    case Op::kOutputPort:
      return visitor.HandleOutputPort(down_cast<OutputPort*>(node), instance);

    case Op::kGate:
      return visitor.HandleGate(down_cast<Gate*>(node), instance);

    case Op::kInstantiationInput:
      return visitor.HandleInstantiationInput(
          down_cast<InstantiationInput*>(node), instance);

    case Op::kInstantiationOutput:
      return visitor.HandleInstantiationOutput(
          down_cast<InstantiationOutput*>(node), instance);
  }
}

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
      register_prefix_(MakeRegisterPrefix(path_)),
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

absl::Status BlockElaboration::Accept(
    ElaboratedBlockDfsVisitor& visitor) const {
  int64_t node_count = 0;
  for (BlockInstance* instance : instance_ptrs_) {
    if (!instance->block().has_value()) {
      continue;
    }
    Block* block = *instance->block();
    node_count += block->node_count();
    for (Node* node : block->nodes()) {
      ElaboratedNode en = {.node = node, .instance = instance};
      if (node->users().empty() && !InterInstanceSuccessor(en).has_value()) {
        XLS_RETURN_IF_ERROR(en.Accept(visitor));
      }
    }
  }
  if (visitor.GetVisitedCount() < node_count) {
    // Not all nodes were visited. This indicates a cycle, for example consider
    // a pair of identity ops with each other as their operands. They will never
    // be visited above.
    //
    // Create a separate trivial DFS visitor to find the cycle.
    class CycleChecker : public ElaboratedBlockDfsVisitorWithDefault {
      absl::Status DefaultHandler(const ElaboratedNode& node) override {
        return absl::OkStatus();
      }
    };
    CycleChecker cycle_checker;
    for (BlockInstance* instance : instance_ptrs_) {
      if (!instance->block().has_value()) {
        continue;
      }
      Block* block = *instance->block();
      for (Node* node : block->nodes()) {
        ElaboratedNode en = {.node = node, .instance = instance};
        if (!cycle_checker.IsVisited(en)) {
          XLS_RETURN_IF_ERROR(en.Accept(cycle_checker));
        }
      }
    }
    return absl::InternalError(absl::StrFormat(
        "Expected to find cycle in elaboration %s, but none was found.",
        ToString()));
  }
  return absl::OkStatus();
}

// Returns a list of every (Node, BlockInstance) in the elaboration in topo
// order.
//
// Based on implementation for unelaborated `FunctionBase`s in
// xls/ir/topo_sort.cc. Note that that implementation discusses 'users' and
// 'operands', whereas this one discusses 'successors' and 'predecessors'. In an
// elaboration, 'successors' include 'users' and 'predecessors' include
// 'operands', but both can potentially also include ElaboratedNodes from a
// different instance, e.g. an InstantiationOutput is a successor of an
// instantiated subblock's OutputPort.
std::vector<ElaboratedNode> ElaboratedReverseTopoSort(
    const BlockElaboration& elaboration) {
  // For topological traversal we only add nodes to the order when all of its
  // successors have been scheduled.
  //
  //       o    node, now ready, can be added to order!
  //      /|\
  //     v v v
  //     o o o  (successors, all present in order)
  //
  // When a node is placed into the ordering, we place all of its predecessors
  // into the "pending_to_remaining_users" mapping if it is not yet present --
  // this keeps track of how many more successors must be seen (before that node
  // is ready to place into the ordering).
  //
  // NOTE: sorts reverse-topologically.  To sort topologically, reverse the
  // result.
  absl::flat_hash_map<ElaboratedNode, int64_t> pending_to_remaining_successors;
  std::vector<ElaboratedNode> ordered;
  std::deque<ElaboratedNode> ready;

  auto seed_ready = [&](ElaboratedNode n) {
    ready.push_front(n);
    CHECK(pending_to_remaining_successors.insert({n, -1}).second);
  };

  int64_t node_count = 0;
  for (BlockInstance* inst : elaboration.instances()) {
    if (!inst->block().has_value()) {
      continue;
    }
    Block* block = *inst->block();
    for (Node* node : block->nodes()) {
      ++node_count;
      ElaboratedNode elaborated_node{.node = node, .instance = inst};
      std::optional<ElaboratedNode> inter_instance_user =
          InterInstanceSuccessor(elaborated_node);

      if (!inter_instance_user.has_value() && node->users().empty()) {
        VLOG(5) << "At start node was ready: " << node;
        seed_ready(elaborated_node);
      }
    }
  }

  ordered.reserve(node_count);

  auto all_successors_scheduled = [&](const ElaboratedNode& n) {
    if (std::optional<ElaboratedNode> inter_instance_user =
            InterInstanceSuccessor(n);
        inter_instance_user.has_value()) {
      auto it = pending_to_remaining_successors.find(*inter_instance_user);
      if (it == pending_to_remaining_successors.end() || it->second >= 0) {
        return false;
      }
    }
    return absl::c_all_of(n.node->users(), [&](Node* user) {
      auto it = pending_to_remaining_successors.find(
          ElaboratedNode{.node = user, .instance = n.instance});
      if (it == pending_to_remaining_successors.end()) {
        return false;
      }
      return it->second < 0;
    });
  };
  auto bump_down_remaining_successors = [&](const ElaboratedNode& n) {
    CHECK(!n.node->users().empty() || InterInstanceSuccessor(n).has_value());
    auto [it, inserted] =
        pending_to_remaining_successors.insert({n, n.node->users().size()});
    int64_t& remaining_successors = it->second;
    // If we inserted, check if there's an inter-instance user.
    if (inserted && InterInstanceSuccessor(n).has_value()) {
      ++remaining_successors;
    }
    CHECK_GT(remaining_successors, 0);
    remaining_successors -= 1;
    VLOG(5) << "Bumped down remaining successors for: " << n
            << "; now: " << remaining_successors;
    if (remaining_successors == 0) {
      ready.push_back(it->first);
      remaining_successors -= 1;
    }
  };

  absl::flat_hash_set<Node*> seen_operands;
  auto add_to_order = [&](ElaboratedNode r) {
    VLOG(5) << "Adding node to order: " << r;
    DCHECK(all_successors_scheduled(r)) << r;
    ordered.push_back(r);

    if (std::optional<ElaboratedNode> inter_instance_predecessor =
            InterInstancePredecessor(r);
        inter_instance_predecessor.has_value()) {
      // No need to put in seen_operands, we won't see this again.
      bump_down_remaining_successors(*inter_instance_predecessor);
    }
    // We want to be careful to only bump down our operands once, since we're a
    // single user, even though we may refer to them multiple times in our
    // operands sequence.
    // We share seen_operands across invocations of add_to_order to reduce
    // overhead of constructing/allocating a set each time. Clear it before
    // using it.
    seen_operands.clear();
    for (auto it = r.node->operands().rbegin(); it != r.node->operands().rend();
         ++it) {
      Node* operand = *it;
      if (auto [_, inserted] = seen_operands.insert(operand); inserted) {
        bump_down_remaining_successors(
            ElaboratedNode{.node = operand, .instance = r.instance});
      }
    }
  };

  while (!ready.empty()) {
    ElaboratedNode r = ready.front();
    ready.pop_front();
    add_to_order(r);
  }

  if (ordered.size() < node_count) {
    // Not all nodes have been placed indicating a cycle in the graph. Run a
    // trivial DFS visitor which will emit an error message displaying the
    // cycle.
    class CycleChecker : public ElaboratedBlockDfsVisitorWithDefault {
      absl::Status DefaultHandler(const ElaboratedNode& node) override {
        return absl::OkStatus();
      }
    };
    CycleChecker cycle_checker;
    CHECK_OK(elaboration.Accept(cycle_checker));
    LOG(FATAL) << "Expected to find a cycle in the elaboration.";
  }

  return ordered;
}

std::vector<ElaboratedNode> ElaboratedTopoSort(
    const BlockElaboration& elaboration) {
  std::vector<ElaboratedNode> ordered = ElaboratedReverseTopoSort(elaboration);
  std::reverse(ordered.begin(), ordered.end());
  return ordered;
}

}  // namespace xls
