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

#include "xls/passes/cse_pass.h"

#include <algorithm>
#include <array>
#include <bit>
#include <compare>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iterator>
#include <memory>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/function_base.h"
#include "xls/ir/lsb_or_msb.h"
#include "xls/ir/nodes.h"
#include "xls/ir/op.h"
#include "xls/ir/type.h"
#include "xls/passes/optimization_pass.h"
#include "xls/passes/pass_base.h"

namespace xls {

namespace {

// Returns 'true' if 'l' is a better combined name than r.
bool CompareName(Node* l, Node* r) {
  if (l->HasAssignedName() && r->HasAssignedName()) {
    auto cmp = (l->GetNameView().size() <=> r->GetNameView().size());
    if (cmp == std::strong_ordering::equal) {
      return l->id() < r->id();
    } else {
      return cmp == std::strong_ordering::less;
    }
  }
  if (l->HasAssignedName()) {
    return true;
  }
  if (r->HasAssignedName()) {
    return false;
  }
  return l->id() < r->id();
}

// Helper to represent a node that can be cse-d together.
class CseNode {
 public:
  CseNode(CseNode&&) = default;
  CseNode(const CseNode&) = default;
  CseNode& operator=(CseNode&&) = default;
  CseNode& operator=(const CseNode&) = default;
  CseNode(Op op, std::vector<CseNode*> operands,
          std::variant<int64_t, std::vector<uint8_t>> misc_data, Type* type)
      : op_(op),
        operands_(std::move(operands)),
        misc_data_(std::move(misc_data)),
        type_(type),
        id_(-1) {}
  CseNode(Node* n, std::vector<CseNode*> operands,
          std::variant<int64_t, std::vector<uint8_t>> misc_data)
      : CseNode(n->op(), std::move(operands), std::move(misc_data),
                n->GetType()) {}

  bool is_forced_unique() const {
    return std::holds_alternative<int64_t>(misc_data_);
  }
  // 'id' isn't used for equivalence and only used to provide a consistent order
  // for operands which are commutative.
  int64_t id() const { return id_; }
  void set_id(int64_t id) { id_ = id; }

  // Equality operator for use in hash maps.
  bool operator==(const CseNode& other) const {
    if (op_ != other.op_) {
      return false;
    }
    if (misc_data_ != other.misc_data_) {
      return false;
    }
    if (type_ != other.type_) {
      return false;
    }
    if (operands_.size() != other.operands_.size()) {
      return false;
    }
    // NB No need to deref since the pointers are canonical.
    for (size_t i = 0; i < operands_.size(); ++i) {
      if (operands_[i] != other.operands_[i]) {
        return false;
      }
    }
    return true;
  }

  // Required for absl::Hash.
  template <typename H>
  friend H AbslHashValue(H h, const CseNode& c) {
    h = H::combine(std::move(h), c.op_, c.misc_data_, c.type_);
    for (const CseNode* operand : c.operands_) {
      h = H::combine(std::move(h), operand);
    }
    return h;
  }

 private:
  Op op_;
  std::vector<CseNode*> operands_;
  // Some operations have config that isn't embedded in the (op, operands,
  // result-ty) tuple (eg the start bit of a bitslice, the exact element of a
  // tuple-index etc). These are needed to identify which values can be
  // combined. We simply serialize the data to the vector here and use it ensure
  // comparison works. Since we don't actually care about what the data says we
  // don't do any sort of padding.
  //
  // NB A bunch of operations do include extra data that is duplicitive of the
  // result type. We don't bother to include that here (though we could without
  // issue).
  //
  // For some nodes which should never be combined we just put a unique number
  // in this field to ensure the CseNode evaluates not-equal to every other
  // CseNode.
  std::variant<int64_t, std::vector<uint8_t>> misc_data_;
  Type* type_;
  // id used for sorting only. Not considered part of the struct for hash or eq.
  int64_t id_;
};
class CseNodeArena {
 public:
  explicit CseNodeArena(bool common_literals, FunctionBase* f)
      : common_literals_(common_literals) {
    node_map_.reserve(f->node_count());
    nodes_.reserve(f->node_count());
    arena_.reserve(f->node_count());
  }
  absl::StatusOr<CseNode*> GetOrCreate(Node* n) {
    if (auto it = node_map_.find(n); it != node_map_.end()) {
      return it->second;
    }
    XLS_ASSIGN_OR_RETURN(std::unique_ptr<CseNode> node, Create(n));
    // Make sure that side-effecting ops (other than gate) are marked as
    // force-unique.
    XLS_RET_CHECK(!OpIsSideEffecting(n->op()) || n->op() == Op::kGate ||
                  node->is_forced_unique())
        << n << " is not forced unique while side-effecting";
    auto it = nodes_.find(CseNodeHolder{node.get()});
    if (it != nodes_.end()) {
      node_map_[n] = it->get();
      return it->get();
    }
    node_map_[n] = node.get();
    node->set_id(next_id_++);
    nodes_.emplace(node.get());
    CseNode* res = node.get();
    arena_.push_back(std::move(node));
    return res;
  }
  absl::StatusOr<CseNode*> Get(Node* n) {
    auto it = node_map_.find(n);
    if (it == node_map_.end()) {
      return absl::NotFoundError(
          absl::StrFormat("Node %s not found in CSE node arena", n->GetName()));
    }
    return it->second;
  }

 private:
  absl::StatusOr<std::unique_ptr<CseNode>> Create(Node* n) {
    std::vector<CseNode*> operands;
    operands.reserve(n->operands().size());
    for (Node* operand : n->operands()) {
      // Built in TopoSort order so the operands should have already been
      // created.
      XLS_ASSIGN_OR_RETURN(
          std::back_inserter(operands), Get(operand),
          _ << "Unable to cse " << n << " due to operand being unvisited");
    }
    if (OpIsCommutative(n->op())) {
      absl::c_sort(operands,
                   [](CseNode* a, CseNode* b) { return a->id() < b->id(); });
    }
    auto bytes_data = [](int64_t v) {
      std::vector<uint8_t> data;
      data.reserve(8);
      std::array<uint8_t, 8> arr = std::bit_cast<std::array<uint8_t, 8>>(v);
      data.insert(data.end(), arr.begin(), arr.end());
      return data;
    };
    auto add_bytes = [](std::vector<uint8_t>& data, int64_t v) {
      std::array<uint8_t, 8> arr = std::bit_cast<std::array<uint8_t, 8>>(v);
      data.insert(data.end(), arr.begin(), arr.end());
    };
    // Add in any extra data that is can distinguish one node of a particular
    // type, operation and operands with another.
    switch (n->op()) {
      // assumed_in_bounds_
      case Op::kArrayIndex: {
        return std::make_unique<CseNode>(
            n, std::move(operands),
            std::vector<uint8_t>{n->As<ArrayIndex>()->assumed_in_bounds()});
      }
      // assumed_in_bounds_
      case Op::kArrayUpdate: {
        return std::make_unique<CseNode>(
            n, std::move(operands),
            std::vector<uint8_t>{n->As<ArrayUpdate>()->assumed_in_bounds()});
      }
      // start
      case Op::kBitSlice: {
        return std::make_unique<CseNode>(
            n, std::move(operands), bytes_data(n->As<BitSlice>()->start()));
      }
      // priority
      case Op::kOneHot: {
        return std::make_unique<CseNode>(
            n, std::move(operands),
            std::vector<uint8_t>{n->As<OneHot>()->priority() == LsbOrMsb::kLsb
                                     ? uint8_t{1}
                                     : uint8_t{0}});
      }
      // index
      case Op::kTupleIndex: {
        return std::make_unique<CseNode>(
            n, std::move(operands), bytes_data(n->As<TupleIndex>()->index()));
      }
      // trip-count, stride, body-function
      case Op::kCountedFor: {
        std::vector<uint8_t> data;
        add_bytes(data, n->As<CountedFor>()->trip_count());
        add_bytes(data, n->As<CountedFor>()->stride());
        add_bytes(data, static_cast<int64_t>(std::bit_cast<intptr_t>(
                            n->As<CountedFor>()->body())));
        return std::make_unique<CseNode>(n, std::move(operands),
                                         std::move(data));
      }
      // function
      case Op::kDynamicCountedFor: {
        return std::make_unique<CseNode>(
            n, std::move(operands),
            bytes_data(static_cast<int64_t>(
                std::bit_cast<intptr_t>(n->As<DynamicCountedFor>()->body()))));
      }
      // function
      case Op::kInvoke: {
        return std::make_unique<CseNode>(
            n, std::move(operands),
            bytes_data(static_cast<int64_t>(
                std::bit_cast<intptr_t>(n->As<Invoke>()->to_apply()))));
      }
      // function
      case Op::kMap: {
        return std::make_unique<CseNode>(
            n, std::move(operands),
            bytes_data(static_cast<int64_t>(
                std::bit_cast<intptr_t>(n->As<Map>()->to_apply()))));
      }
      // The actual literal value (or 'do not cse' if !common_literals_)
      case Op::kLiteral: {
        if (common_literals_) {
          // Flatten data. Just use the proto format.
          std::vector<uint8_t> data;
          XLS_ASSIGN_OR_RETURN(auto proto, n->As<Literal>()->value().AsProto());
          data.resize(proto.ByteSizeLong());
          XLS_RET_CHECK(proto.SerializeToArray(data.data(), data.size()))
              << "Failed to serialize literal for " << n;
          return std::make_unique<CseNode>(n, std::move(operands),
                                           std::move(data));
        }
        // We don't want to merge literals so just make each of them unique.
        return std::make_unique<CseNode>(n, std::move(operands), non_cse_id_++);
      }
      // do not cse. All of these should never be merged since they are
      // side-effecting.
      case Op::kAssert:
      case Op::kCover:
      case Op::kInputPort:
      case Op::kInstantiationInput:
      case Op::kInstantiationOutput:
      case Op::kMinDelay:
      case Op::kNewChannel:
      case Op::kNext:
      case Op::kOutputPort:
      case Op::kParam:
      case Op::kReceive:
      case Op::kRecvChannelEnd:
      case Op::kRegisterRead:
      case Op::kRegisterWrite:
      case Op::kSend:
      case Op::kSendChannelEnd:
      case Op::kStateRead:
      case Op::kTrace: {
        return std::make_unique<CseNode>(n, std::move(operands), non_cse_id_++);
      }
      // everything below here does not need any additional information.
      case Op::kAdd:
      case Op::kAfterAll:
      case Op::kAnd:
      case Op::kAndReduce:
      case Op::kArray:
      case Op::kArrayConcat:
      case Op::kArraySlice:
      case Op::kBitSliceUpdate:
      case Op::kConcat:
      case Op::kDecode:
      case Op::kDynamicBitSlice:
      case Op::kEncode:
      case Op::kEq:
      case Op::kGate:
      case Op::kIdentity:
      case Op::kNand:
      case Op::kNe:
      case Op::kNeg:
      case Op::kNor:
      case Op::kNot:
      case Op::kOneHotSel:
      case Op::kOr:
      case Op::kOrReduce:
      case Op::kPrioritySel:
      case Op::kReverse:
      case Op::kSDiv:
      case Op::kSGe:
      case Op::kSGt:
      case Op::kSLe:
      case Op::kSLt:
      case Op::kSMod:
      case Op::kSMul:
      case Op::kSMulp:
      case Op::kSel:
      case Op::kShll:
      case Op::kShra:
      case Op::kShrl:
      case Op::kSignExt:
      case Op::kSub:
      case Op::kTuple:
      case Op::kUDiv:
      case Op::kUGe:
      case Op::kUGt:
      case Op::kULe:
      case Op::kULt:
      case Op::kUMod:
      case Op::kUMul:
      case Op::kUMulp:
      case Op::kXor:
      case Op::kXorReduce:
      case Op::kZeroExt: {
        return std::make_unique<CseNode>(n, std::move(operands),
                                         std::vector<uint8_t>{});
      }
    }
  }
  struct CseNodeHolder {
    CseNode* node_;

    CseNode* get() const { return node_; }
    template <typename H>
    friend H AbslHashValue(H h, const CseNodeHolder& c) {
      return AbslHashValue(std::move(h), *c.node_);
    }
    bool operator==(const CseNodeHolder& other) const {
      return *node_ == *other.node_;
    }
  };

  bool common_literals_;
  // Id used for sorting commutative operations based on the order they are
  // encountered.
  int64_t next_id_ = 1;
  // Id used to ensure that all non-cse-eligible nodes are considered globally
  // unique. Has no semantic meaning.
  int64_t non_cse_id_ = 1;
  absl::flat_hash_map<Node*, CseNode*> node_map_;
  absl::flat_hash_set<CseNodeHolder> nodes_;
  std::vector<std::unique_ptr<CseNode>> arena_;
};

}  // namespace

absl::StatusOr<bool> RunCse(FunctionBase* f, OptimizationContext& context,
                            absl::flat_hash_map<Node*, Node*>* replacements,
                            bool common_literals) {
  CseNodeArena arena(common_literals, f);
  // All the nodes in an equivalence set that we've found so far.
  absl::flat_hash_map<CseNode*, std::vector<Node*>> node_buckets;

  int64_t dead_nodes = 0;
  node_buckets.reserve(f->node_count());
  // Identify all the equivalence groups.
  for (Node* node : context.TopoSort(f)) {
    // Normally, dead nodes are removed by the DCE pass. However, if the node is
    // (e.g.) an invoke, DCE won't touch it, waiting for inlining to remove
    // it... and if we try to replace it, we'll think we changed the IR when we
    // actually didn't.
    if (node->IsDead()) {
      ++dead_nodes;
      continue;
    }
    // NB arena determines what nodes are eligible for cse.
    XLS_ASSIGN_OR_RETURN(CseNode * cse, arena.GetOrCreate(node));
    node_buckets[cse].push_back(node);
  }

  // Short circuit. Pigeonhole principle. All groups must have only a single
  // element.
  if (node_buckets.size() + dead_nodes == f->node_count()) {
    return false;
  }
  // Actually do the replacement.
  for (const auto& [_, bucket] : node_buckets) {
    if (bucket.size() == 1) {
      continue;
    }
    CHECK(!bucket.empty()) << "How did an empty bucket end up here?";
    // Pick the shortest name. This ensures that we deterministically
    // choose the "canonical" node in case there are multiple options.
    Node* representative = *absl::c_min_element(bucket, CompareName);
    for (Node* node : bucket) {
      if (node != representative) {
        if (replacements != nullptr) {
          replacements->insert({node, representative});
        }
        VLOG(3) << absl::StreamFormat("Replacing %s with equivalent node %s",
                                      node->GetName(),
                                      representative->GetName());

        // TODO(allight): Merge the location information too.
        XLS_RETURN_IF_ERROR(node->ReplaceUsesWith(representative));
      }
    }
  }

  return true;
}

absl::StatusOr<bool> CsePass::RunOnFunctionBaseInternal(
    FunctionBase* f, const OptimizationPassOptions& options,
    PassResults* results, OptimizationContext& context) const {
  return RunCse(f, context, nullptr, common_literals_);
}

}  // namespace xls
