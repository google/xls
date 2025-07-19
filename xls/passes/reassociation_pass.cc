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

#include "xls/passes/reassociation_pass.h"

#include <algorithm>
#include <compare>
#include <cstdint>
#include <functional>
#include <ios>
#include <iterator>
#include <optional>
#include <ostream>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "cppitertools/chain.hpp"
#include "cppitertools/chunked.hpp"
#include "cppitertools/filter.hpp"
#include "cppitertools/sliding_window.hpp"
#include "xls/common/math_util.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/data_structures/leaf_type_tree.h"
#include "xls/ir/bits.h"
#include "xls/ir/change_listener.h"
#include "xls/ir/dfs_visitor.h"
#include "xls/ir/function_base.h"
#include "xls/ir/node.h"
#include "xls/ir/nodes.h"
#include "xls/ir/op.h"
#include "xls/ir/ternary.h"
#include "xls/ir/type.h"
#include "xls/ir/value.h"
#include "xls/passes/bit_count_query_engine.h"
#include "xls/passes/lazy_node_info.h"
#include "xls/passes/lazy_ternary_query_engine.h"
#include "xls/passes/optimization_pass.h"
#include "xls/passes/pass_base.h"
#include "xls/passes/query_engine.h"
#include "xls/passes/stateless_query_engine.h"
#include "xls/passes/ternary_evaluator.h"
#include "xls/passes/union_query_engine.h"

namespace xls {

namespace {

// A helper to allow one to treat a node as if it was a zero-ext.
class ZeroExtLike {
 public:
  static std::optional<ZeroExtLike> Make(Node* n, const QueryEngine& qe) {
    switch (n->op()) {
      case Op::kZeroExt:
        return ZeroExtLike(n, n->operand(0), /*op_num=*/0);
      case Op::kSignExt:
        if (qe.KnownLeadingZeros(n->operand(0)).value_or(0) >= 1) {
          return ZeroExtLike(n, n->operand(0), /*op_num=*/0);
        }
        return std::nullopt;
      case Op::kConcat:
        if (absl::c_all_of(n->operands().subspan(0, n->operand_count() - 1),
                           [&](Node* op) { return qe.IsAllZeros(op); })) {
          return ZeroExtLike(n, n->operands().back(),
                             /*op_num=*/n->operand_count() - 1);
        }
        return std::nullopt;
      default:
        return std::nullopt;
    }
  }

  Node* real_node() const { return src_; }
  Node* to_extend() const { return arg_; }
  int64_t op_num() const { return op_num_; }

 private:
  ZeroExtLike(Node* src, Node* arg, int64_t op_num)
      : src_(src), arg_(arg), op_num_(op_num) {}
  Node* src_;
  Node* arg_;
  int64_t op_num_;
};

// Is this an op we want to reassociate.
bool IsAssociativeOp(Node* n) {
  return n->OpIn({Op::kAdd, Op::kSub, Op::kSMul, Op::kUMul});
}

// For addition elements, nodes that represent variables that are the same type
// and next to each other will be combined. For example, (b + a + a) becomes
// (b + (a + a)) instead of ((b + a) + a). After, create a sum/product as a
// minimum depth tree. The tree is biased to the left side. This is to ensure
// that as long as the input is sorted by bit-count then the deeper sections
// will be on smaller values.
// TODO(allight): Consider if/when a perfectly balanced tree could be better. I
// think the answer is never but it is hard to say.
template <typename MakeOp, typename T>
  requires(std::is_invocable_r_v<absl::StatusOr<T>, MakeOp, const T&, const T&>)
absl::StatusOr<T> CreateTreeSum(MakeOp make_op, absl::Span<T const> nodes,
                                std::optional<Op> op) {
  XLS_RET_CHECK(!nodes.empty());
  std::vector<T> prev;
  prev.reserve(nodes.size());
  std::vector<T> cur(nodes.begin(), nodes.end());
  // Use of a 2 pointer list traversal algorithm to combine same variable nodes
  // for addition elements only.
  if (op == Op::kAdd) {
    std::swap(prev, cur);
    cur.clear();
    cur.reserve(prev.size() / 2 + prev.size() % 2);
    int64_t left_idx = 0;
    int64_t right_idx = 1;
    while (left_idx < prev.size()) {
      // In the case of (a), no pair is needed so just transfer the node to the
      // other list.
      if (right_idx >= prev.size()) {
        cur.push_back(prev[left_idx]);
        break;
        // In the case of (a + b + b), pair the second two nodes.
      } else if (right_idx + 1 < prev.size() &&
                 prev[right_idx].node == prev[right_idx + 1].node) {
        XLS_ASSIGN_OR_RETURN(std::back_inserter(cur),
                             make_op(prev[right_idx], prev[right_idx + 1]));
        right_idx += 2;
        // In the case of (a + a + b) or (a + b + c) or (a + b), pair the first
        // two nodes.
      } else {
        XLS_ASSIGN_OR_RETURN(std::back_inserter(cur),
                             make_op(prev[left_idx], prev[right_idx]));
        left_idx = right_idx + 1;
        right_idx += 2;
      }
    }
  }
  // Create a minimum depth tree.
  while (cur.size() > 1) {
    std::swap(prev, cur);
    cur.clear();
    cur.reserve(prev.size() / 2 + prev.size() % 2);
    // Bias toward the left side.
    for (auto&& elements : iter::chunked(prev, 2)) {
      if (elements.size() == 1) {
        cur.push_back(elements[0]);
      } else {
        XLS_RET_CHECK_EQ(elements.size(), 2);
        XLS_ASSIGN_OR_RETURN(std::back_inserter(cur),
                             make_op(elements[0], elements[1]));
      }
    }
  }
  return std::move(cur.front());
}

// A set of leaf values that can be combined together in any order (at the
// bit-width of 'node()') to generate the value of node()
class AssociativeElements {
 public:
  struct NodeData {
    Node* node;
    bool needs_negate;

    absl::StatusOr<Node*> ToNode() const {
      if (!needs_negate) {
        return node;
      }
      return node->function_base()->MakeNodeWithName<UnOp>(
          node->loc(), node, Op::kNeg,
          node->HasAssignedName()
              ? absl::StrCat(node->GetNameView(), "_negated")
              : "");
    }
    bool operator==(const NodeData& nd) const {
      return node == nd.node && needs_negate == nd.needs_negate;
    }
    bool operator!=(const NodeData& nd) const = default;

    template <typename Sink>
    friend void AbslStringify(Sink& sink,
                              const AssociativeElements::NodeData& e) {
      if (e.needs_negate) {
        absl::Format(&sink, "negate(%s)", e.node->ToString());
      } else {
        absl::Format(&sink, "%s", e.node->ToString());
      }
    }
    std::string DumpShort() const {
      std::string res =
          absl::StrCat(node->GetName(), ":", node->GetType()->ToString());
      if (needs_negate) {
        return absl::StrCat("negate(", res, ")");
      }
      return res;
    }
  };
  static AssociativeElements MakeLeaf(Node* n) {
    return AssociativeElements(
        n, /*leaf=*/NodeData{.node = n, .needs_negate = false},
        /*op=*/std::nullopt, /*variables=*/{},
        /*constants=*/{},
        /*overflows=*/false,
        /*depth=*/0,
        // NB identities don't have variables.
        /*variable_cnt=*/0);
  }
  void SetOverflow() { this->overflows_ = true; }
  AssociativeElements WithOverflow() const {
    AssociativeElements res = *this;
    res.SetOverflow();
    return res;
  }
  AssociativeElements Negative() const {
    return AssociativeElements::Negative(*this);
  }
  AssociativeElements WithOwner(Node* n) const {
    // Leaf values just become new leafs with a new owner.
    AssociativeElements res = *this;
    res.node_ = n;
    return res;
  }

  // Combine the elements in l and r using 'cur_op'.
  //
  // overflows is if the entire operation is possible to overflow.
  static AssociativeElements Combine(const QueryEngine& qe, Node* n, Op cur_op,
                                     const AssociativeElements& l,
                                     const AssociativeElements& r,
                                     bool overflows) {
    CHECK(IsAssociativeOp(n)) << n << " cannot build associative tree";
    CHECK(cur_op == n->op() || (cur_op == Op::kAdd && n->op() == Op::kSub))
        << n << " with op " << cur_op;

    auto can_combine_element = [&](const AssociativeElements& e) -> bool {
      if (e.is_leaf()) {
        // Leafs aren't really combinable in a real sense.. They just get added.
        // We can return false here to get the addition behavior.
        return false;
      }
      if (e.op() != cur_op) {
        // the l/r element is from a different operation. Can't combine.
        return false;
      }
      if (!e.overflows() && !overflows) {
        // No overflow is happening anywhere so you can merge in this element.
        return true;
      }
      // If the bit-width changed we can't combine.
      return absl::c_all_of(e.all_elements(), [&](const NodeData& nd) {
        return nd.node->BitCountOrDie() == n->BitCountOrDie();
      });
    };
    bool can_combine_l = can_combine_element(l);
    bool can_combine_r = can_combine_element(r);
    // This overall operation overflows if either (1) it trivially overflows or
    // (2) the embedded operands overflow themselves.
    bool combined_overflow = overflows || (l.overflows() && can_combine_l) ||
                             (r.overflows() && can_combine_r);
    std::vector<NodeData> datas;
    std::vector<NodeData> consts;
    auto expected_size = [&](int64_t l_size, int64_t r_size) {
      return (can_combine_l ? l_size : 1) + (can_combine_r ? r_size : 1);
    };
    datas.reserve(expected_size(l.variables().size(), r.variables().size()));
    consts.reserve(expected_size(l.constants().size(), r.constants().size()));
    auto combine_one = [&](bool can_combine,
                           const AssociativeElements& elements,
                           bool elements_is_negated) -> int64_t {
      if (can_combine) {
        absl::c_copy(elements.variables(), std::back_inserter(datas));
        absl::c_copy(elements.constants(), std::back_inserter(consts));
        return *elements.full_variable_count();
      } else if (qe.IsFullyKnown(elements.node())) {
        if (elements.leaf_ && !elements.overflows()) {
          consts.push_back(*elements.leaf_);
        } else {
          consts.push_back(
              {.node = elements.node(), .needs_negate = elements_is_negated});
        }
        return 0;
      } else if (elements.leaf_ && !elements.overflows()) {
        datas.push_back(*elements.leaf_);
        return 1;
      } else {
        // Can't combine. This becomes a leaf, potentially hiding any other
        // computation.
        datas.push_back(
            {.node = elements.node(), .needs_negate = elements_is_negated});
        return 1;
      }
    };
    int64_t l_var_cnt =
        combine_one(can_combine_l, l, /*elements_is_negated=*/false);
    // If the real operation we're doing is a sub and we need to turn the rhs
    // into a leaf we need to know to make the right-side a negated leaf.
    int64_t r_var_cnt = combine_one(
        can_combine_r, r, /*elements_is_negated=*/n->op() == Op::kSub);
    if (cur_op == Op::kAdd) {
      datas = RemoveAdditiveInverses(std::move(datas), n->BitCountOrDie());
      // NB This might be empty. Our translation already handles this however.
    }
    int64_t new_depth = 1 + std::max(can_combine_l ? l.depth() : 0,
                                     can_combine_r ? r.depth() : 0);
    int64_t new_full_var_count = l_var_cnt + r_var_cnt;
    return AssociativeElements(
        n, /*leaf=*/std::nullopt, cur_op, std::move(datas), std::move(consts),
        combined_overflow, new_depth, new_full_var_count);
  }

  Node* node() const { return node_; }
  std::optional<Op> op() const { return op_; }
  absl::Span<const NodeData> variables() const { return variables_; }
  absl::Span<const NodeData> constants() const { return constants_; }
  decltype(iter::chain(std::declval<absl::Span<NodeData const>>(),
                       std::declval<absl::Span<NodeData const>>()))
  all_elements() const {
    return iter::chain(variables(), constants());
  }
  int64_t element_count() const {
    return std::max<int64_t>(variables_.size() + constants_.size(), 1);
  }
  // How many leafs the whole tree currently contains. This is full variable
  // count and constants combined.
  int64_t full_leaf_count() const {
    return std::max<int64_t>(
        full_variable_count().value_or(0) + constants_.size(), 1);
  }
  bool is_leaf() const { return leaf_.has_value(); }
  bool overflows() const { return overflows_; }
  int64_t depth() const { return depth_; }
  // For non-leaf elements the number of variable nodes in the tree with the
  // head at the current node.
  std::optional<int64_t> full_variable_count() const {
    return leaf_ ? std::nullopt : std::make_optional(variable_cnt_);
  }
  // How many elements are there if we combine all constants into a single
  // element.
  int64_t ElementCountWithMaxOneConstant() const {
    if (constants_.empty()) {
      return element_count();
    }
    return variables_.size() + 1;
  }

  std::string ElementsToString() const {
    return absl::StrFormat(
        "[%s]", absl::StrJoin(
                    all_elements(), ", ",
                    [](std::string* s, const AssociativeElements::NodeData& n) {
                      absl::StrAppend(s, n.DumpShort());
                    }));
  }

  AssociativeElements(AssociativeElements&&) = default;
  AssociativeElements& operator=(AssociativeElements&&) = default;
  AssociativeElements(const AssociativeElements&) = default;
  AssociativeElements& operator=(const AssociativeElements&) = default;
  // Let flat-hash-set work easily.
  AssociativeElements()
      : AssociativeElements(nullptr, /*leaf=*/std::nullopt, std::nullopt, {},
                            {},
                            /*overflows=*/false,
                            /*depth=*/0, /*variable_cnt=*/0) {}

  bool operator==(const AssociativeElements& rhs) const {
    return node_ == rhs.node_ && op_ == rhs.op_ && leaf_ == rhs.leaf_ &&
           absl::c_equal(variables_, rhs.variables_) &&
           absl::c_equal(constants_, rhs.constants_) &&
           overflows_ == rhs.overflows_ && depth_ == rhs.depth_ &&
           variable_cnt_ == rhs.variable_cnt_;
  }
  bool operator!=(const AssociativeElements& rhs) const = default;

  std::string DumpShort() const {
    auto node_formatter = [](std::string* s, const NodeData& n) {
      absl::StrAppend(s, n.DumpShort());
    };
    if (leaf_) {
      return absl::StrFormat(
          "%sleaf(%s)%s", leaf_->needs_negate ? "negative(" : "",
          leaf_->node->GetName(), leaf_->needs_negate ? ")" : "");
    }
    return absl::StrFormat(
        "{node: %v, op: %s, nodes(%d): [%s], constants(%d): [%s], "
        "overflows: %v, depth: %d, variable_cnt: %d}",
        node_->GetName(), OpToString(*op_), variables_.size(),
        absl::StrJoin(variables_, ", ", node_formatter), constants_.size(),
        absl::StrJoin(constants_, ", ", node_formatter), overflows_, depth_,
        variable_cnt_);
  }
  template <typename Sink>
  friend void AbslStringify(Sink& sink, const AssociativeElements& e) {
    auto node_formatter = [](std::string* s, const NodeData& n) {
      absl::StrAppend(s, n);
    };
    absl::Format(
        &sink,
        "{node: %v, op: %s, variables(%d): [%s], constants(%d): [%s], "
        "overflows: %v, depth: %d, variable_cnt: %d}",
        e.node_->GetName(),
        e.op_.has_value() ? OpToString(*e.op_)
                          : absl::StrFormat("<leaf(%v)>", *e.leaf_),
        e.variables_.size(), absl::StrJoin(e.variables_, ", ", node_formatter),
        e.constants_.size(), absl::StrJoin(e.constants_, ", ", node_formatter),
        e.overflows_, e.depth_, e.variable_cnt_);
  }

 private:
  AssociativeElements(Node* node, std::optional<NodeData> leaf,
                      std::optional<Op> op, std::vector<NodeData> variables,
                      std::vector<NodeData> constants, bool overflows,
                      int64_t depth, int64_t variable_cnt)
      : node_(node),
        leaf_(leaf),
        op_(op),
        variables_(std::move(variables)),
        constants_(std::move(constants)),
        overflows_(overflows),
        depth_(depth),
        variable_cnt_(variable_cnt) {
    CHECK(leaf_ || op_);
  }

  static AssociativeElements Negative(const AssociativeElements& e) {
    if (e.op() == Op::kAdd || e.is_leaf()) {
      std::vector<NodeData> datas;
      std::vector<NodeData> consts;
      datas.reserve(e.variables().size());
      consts.reserve(e.constants().size());
      for (NodeData d : e.variables()) {
        d.needs_negate = !d.needs_negate;
        datas.push_back(d);
      }
      for (NodeData d : e.constants()) {
        d.needs_negate = !d.needs_negate;
        consts.push_back(d);
      }
      std::optional<NodeData> new_leaf;
      if (e.leaf_) {
        new_leaf = {.node = e.leaf_->node,
                    .needs_negate = !e.leaf_->needs_negate};
      }
      return AssociativeElements(e.node(), new_leaf, e.op(), std::move(datas),
                                 std::move(consts), e.overflows(), e.depth(),
                                 e.full_variable_count().value_or(0));
    } else {
      // SMul/UMul
      std::vector<NodeData> datas(e.variables().begin(), e.variables().end());
      std::vector<NodeData> consts(e.constants().begin(), e.constants().end());
      std::optional<NodeData> leaf = e.leaf_;
      if (leaf) {
        leaf->needs_negate = !leaf->needs_negate;
      } else if (!datas.empty()) {
        datas.front().needs_negate = !datas.front().needs_negate;
      } else {
        CHECK(!consts.empty());
        consts.front().needs_negate = !consts.front().needs_negate;
      }
      return AssociativeElements(e.node(), leaf, e.op(), std::move(datas),
                                 std::move(consts), e.overflows(), e.depth(),
                                 e.full_variable_count().value_or(0));
    }
  }

  static std::vector<NodeData> RemoveAdditiveInverses(
      std::vector<NodeData> data, int64_t top_width) {
    // Ensure the elements of the same node are adjacent.
    if (data.empty()) {
      return data;
    }
    absl::c_sort(data, [](const NodeData& lhs, const NodeData& rhs) {
      return lhs.node->id() < rhs.node->id();
    });
    std::vector<AssociativeElements::NodeData> res;
    res.reserve(data.size());
    auto it_range = iter::sliding_window(data, 2);
    for (auto it = it_range.begin(); it != it_range.end(); ++it) {
      auto chunk = *it;
      AssociativeElements::NodeData first = chunk[0];
      AssociativeElements::NodeData second = chunk[1];
      if (first.node == second.node &&
          first.needs_negate != second.needs_negate &&
          // If the width is smaller than this operations width then we'd
          // erroneously get rid of the carry bit.
          // TODO(allight): We could inject a synthetic constant which is just
          // the carry bit.
          first.node->BitCountOrDie() == top_width) {
        // these cancel.
        if (++it == it_range.end()) {
          // second was the last element anyway.
          return res;
        }
        continue;
      }
      res.push_back(first);
    }
    // If we canceled the last element we would have already returned. Otherwise
    // we need to add the last element.
    res.push_back(data.back());
    return res;
  }
  Node* node_;
  // The leaf node data (if this is a leaf node).
  std::optional<NodeData> leaf_;
  // The operation to combine the variables_ to get the value of 'node_'. If
  // nullopt then the node_ is a leaf.
  std::optional<Op> op_;
  // Non-constant leaf elements.
  std::vector<NodeData> variables_;
  // Constant leaf elements.
  std::vector<NodeData> constants_;
  // Does the full computation potentially overflow at 'node_' bit-width
  bool overflows_;
  // How deep is the deepest leaf node.
  int64_t depth_;

  // How many variables (not counting removed ones) were part of this transform.
  int64_t variable_cnt_;
};

std::ostream& operator<<(std::ostream& os, const AssociativeElements& e) {
  return os << absl::StrCat(e);
}

class ReassociationCache;

// A pair of associative elements for the signed and unsigned domains.
struct SignednessPair {
  AssociativeElements signed_values;
  AssociativeElements unsigned_values;

  bool operator==(const SignednessPair& rhs) const {
    return signed_values == rhs.signed_values &&
           unsigned_values == rhs.unsigned_values;
  }
  std::string DumpShort() const {
    return absl::StrFormat("{signed: %s, Unsigned: %s}",
                           signed_values.DumpShort(),
                           unsigned_values.DumpShort());
  }
  template <typename Sink>
  friend void AbslStringify(Sink& sink, const SignednessPair& p) {
    absl::Format(&sink, "{signed: %v, unsigned: %v}", p.signed_values,
                 p.unsigned_values);
  }
};

std::ostream& operator<<(std::ostream& os, const SignednessPair& p) {
  return os << absl::StrCat(p);
}
std::ostream& operator<<(std::ostream& os,
                         const std::optional<SignednessPair>& p) {
  if (p) {
    return os << absl::StrCat(*p);
  }
  return os << "<NONE>";
}

// Visitor to collect the AssociativeElements for each node in the tree.
//
// We could make this a dataflow visitor but since there's no real way to create
// composite elements the result of going through any array would be a leaf node
// anyway meaning there's little point. In cases where the value can be tracked
// precisely enough to gain some information dataflow-simp will remove the
// compound data type anyway giving us another chance.
class OneShotReassociationVisitor : public DfsVisitorWithDefault {
 public:
  explicit OneShotReassociationVisitor(
      const QueryEngine& query_engine,
      absl::Span<const LeafTypeTree<std::optional<SignednessPair>>* const>
          operand_infos)
      : query_engine_(query_engine), operand_infos_(operand_infos) {}

  absl::Status DefaultHandler(Node* n) override {
    XLS_RET_CHECK(n->GetType()->IsBits()) << n;
    // If we reach here we aren't associative.
    unsigned_result_ = AssociativeElements::MakeLeaf(n);
    signed_result_ = AssociativeElements::MakeLeaf(n);
    return absl::OkStatus();
  }

  absl::Status HandleAdd(BinOp* op) override { return HandleAssociativeOp(op); }
  absl::Status HandleSub(BinOp* op) override { return HandleAssociativeOp(op); }
  absl::Status HandleSMul(ArithOp* op) override {
    unsigned_result_ = AssociativeElements::MakeLeaf(op);
    XLS_ASSIGN_OR_RETURN(signed_result_,
                         CalculateAssociativeOp(op, /*is_signed=*/true));
    return absl::OkStatus();
  }
  absl::Status HandleUMul(ArithOp* op) override {
    signed_result_ = AssociativeElements::MakeLeaf(op);
    XLS_ASSIGN_OR_RETURN(unsigned_result_,
                         CalculateAssociativeOp(op, /*is_signed=*/false));
    return absl::OkStatus();
  }
  absl::Status HandleNeg(UnOp* op) override {
    const AssociativeElements& base_signed =
        GetOperandInfo(0, /*is_signed=*/true);
    if (base_signed.op() == Op::kUMul) {
      signed_result_ = AssociativeElements::MakeLeaf(op);
    } else {
      signed_result_ = base_signed.Negative().WithOwner(op);
      // The negation of the min signed value overflows & ends up back at the
      // min signed value; it no longer commutes with (e.g.) sign extension.
      Node* operand = op->operand(0);
      if (query_engine_.Covers(operand,
                               Bits::MinSigned(operand->BitCountOrDie()))) {
        signed_result_->SetOverflow();
      }
    }
    const AssociativeElements& base_unsigned =
        GetOperandInfo(0, /*is_signed=*/false);
    if (base_unsigned.op() == Op::kSMul) {
      unsigned_result_ = AssociativeElements::MakeLeaf(op);
    } else {
      unsigned_result_ = base_unsigned.Negative().WithOverflow().WithOwner(op);
    }
    return absl::OkStatus();
  }
  absl::Status HandleConcat(Concat* cc) override {
    std::optional<ZeroExtLike> ext_like = ZeroExtLike::Make(cc, query_engine_);
    if (!ext_like) {
      return DefaultHandler(cc);
    }
    return HandleZeroExtLike(*ext_like);
  }

  absl::Status HandleZeroExtend(ExtendOp* ext) override {
    return HandleZeroExtLike(*ZeroExtLike::Make(ext, query_engine_));
  }

  absl::Status HandleSignExtend(ExtendOp* ext) override {
    const AssociativeElements& inner_unsigned =
        GetOperandInfo(0, /*is_signed=*/false);
    std::optional<ZeroExtLike> zero_ext_like =
        ZeroExtLike::Make(ext, query_engine_);
    if (zero_ext_like && !inner_unsigned.overflows()) {
      unsigned_result_ = inner_unsigned.WithOwner(ext);
    } else {
      unsigned_result_ = AssociativeElements::MakeLeaf(ext);
    }
    const AssociativeElements& inner_signed =
        GetOperandInfo(0, /*is_signed=*/true);
    if (!inner_signed.overflows()) {
      signed_result_ = inner_signed.WithOwner(ext);
    } else {
      signed_result_ = AssociativeElements::MakeLeaf(ext);
    }
    return absl::OkStatus();
  }

  absl::StatusOr<SignednessPair> Take() && {
    XLS_RET_CHECK(unsigned_result_.has_value());
    XLS_RET_CHECK(signed_result_.has_value());
    return SignednessPair{.signed_values = *std::move(signed_result_),
                          .unsigned_values = *std::move(unsigned_result_)};
  }

 private:
  const AssociativeElements& GetOperandInfo(int64_t idx, bool is_signed) const {
    return is_signed ? operand_infos_[idx]->Get({})->signed_values
                     : operand_infos_[idx]->Get({})->unsigned_values;
  }
  absl::Status HandleZeroExtLike(const ZeroExtLike& ext_like) {
    const AssociativeElements& inner_unsigned =
        GetOperandInfo(ext_like.op_num(), /*is_signed=*/false);
    if (!inner_unsigned.overflows()) {
      unsigned_result_ = inner_unsigned.WithOwner(ext_like.real_node());
    } else {
      unsigned_result_ = AssociativeElements::MakeLeaf(ext_like.real_node());
    }
    const AssociativeElements& inner_signed =
        GetOperandInfo(ext_like.op_num(), /*is_signed=*/true);
    int64_t known_leading_zeros =
        query_engine_.KnownLeadingZeros(ext_like.to_extend()).value_or(0);
    // Loses signedness so can't pull internals up.
    if (!inner_signed.overflows() && known_leading_zeros >= 1) {
      signed_result_ = GetOperandInfo(ext_like.op_num(), /*is_signed=*/true)
                           .WithOwner(ext_like.real_node());
    } else {
      signed_result_ = AssociativeElements::MakeLeaf(ext_like.real_node());
    }
    return absl::OkStatus();
  }
  absl::Status HandleAssociativeOp(Node* op) {
    XLS_RET_CHECK(op->OpIn({Op::kAdd, Op::kSub}));
    XLS_ASSIGN_OR_RETURN(unsigned_result_,
                         CalculateAssociativeOp(op, /*is_signed=*/false));
    XLS_ASSIGN_OR_RETURN(signed_result_,
                         CalculateAssociativeOp(op, /*is_signed=*/true));
    return absl::OkStatus();
  }
  absl::StatusOr<AssociativeElements> CalculateAssociativeOp(
      Node* op, bool is_signed) const {
    XLS_RET_CHECK(IsAssociativeOp(op)) << op;
    XLS_ASSIGN_OR_RETURN(bool overflow, IsOverflowPossible(op, is_signed));

    const AssociativeElements& lhs_elements = GetOperandInfo(0, is_signed);
    std::optional<AssociativeElements> neg_rhs;
    Op real_op = op->op();
    const AssociativeElements& rhs_elements =
        [&]() -> const AssociativeElements& {
      if (op->op() == Op::kSub) {
        // Force to addition.
        real_op = Op::kAdd;
        neg_rhs = GetOperandInfo(1, is_signed).Negative();
        if (!is_signed) {
          // Unsigned negate is overflow.
          neg_rhs = neg_rhs->WithOverflow();
        }
        return *neg_rhs;
      } else {
        return GetOperandInfo(1, is_signed);
      }
    }();
    return AssociativeElements::Combine(query_engine_, op, real_op,
                                        lhs_elements, rhs_elements, overflow);
  }

  // Find if the node can cause overflow according to 2s complement if is_signed
  // or unsigned value otherwise.
  absl::StatusOr<bool> IsOverflowPossible(Node* node, bool is_signed) const {
    TernaryEvaluator eval;
    TernaryVector lv = GetExplicitTernary(node->operand(0));
    TernaryVector rv = GetExplicitTernary(node->operand(1));
    int64_t l_lead_signs =
        *query_engine_.KnownLeadingSignBits(node->operand(0));
    int64_t r_lead_signs =
        *query_engine_.KnownLeadingSignBits(node->operand(1));
    XLS_RET_CHECK(IsAssociativeOp(node)) << node;
    switch (node->op()) {
      case Op::kAdd:
        if (is_signed) {
          // Signed overflow is impossible if more than one sign bit is present.
          return eval.AddWithSignedOverflow(lv, rv).overflow !=
                     TernaryValue::kKnownZero &&
                 !(l_lead_signs > 1 && r_lead_signs > 1);
        } else {
          // NB Unsigned overflow requires known zeros at high bit. The ternary
          // contains this information.
          return eval.AddWithCarry(lv, rv).overflow != TernaryValue::kKnownZero;
        }
      case Op::kSub:
        if (is_signed) {
          return eval.SubWithSignedUnderflow(lv, rv).overflow !=
                     TernaryValue::kKnownZero &&
                 !(l_lead_signs > 1 && r_lead_signs > 1);
        } else {
          return eval.SubWithUnsignedUnderflow(lv, rv).overflow !=
                 TernaryValue::kKnownZero;
        }
      case Op::kUMul:
        // TODO(allight): We can use leading bit counts here to get better
        // knowledge.
        XLS_RET_CHECK(!is_signed) << "signed umul request: " << node;
        return eval.UMulWithOverflow(lv, rv, node->BitCountOrDie()).overflow !=
               TernaryValue::kKnownZero;
      case Op::kSMul:
        XLS_RET_CHECK(is_signed) << "Unsigned smul request: " << node;
        return eval.SMulWithOverflow(lv, rv, node->BitCountOrDie()).overflow !=
               TernaryValue::kKnownZero;
      default:
        return absl::InvalidArgumentError(
            absl::StrFormat("%v is not an associative op", node->ToString()));
    }
  }

  TernaryVector GetExplicitTernary(Node* node) const {
    if (auto tree = query_engine_.GetTernary(node)) {
      return tree->Get({});
    }
    return TernaryVector(node->BitCountOrDie(), TernaryValue::kUnknown);
  }
  const QueryEngine& query_engine_;
  absl::Span<const LeafTypeTree<std::optional<SignednessPair>>* const>
      operand_infos_;
  // The result of invoking this visitor.
  std::optional<AssociativeElements> unsigned_result_;
  std::optional<AssociativeElements> signed_result_;
};

// Helper to lazily calculate the reassociation state.
//
// TODO(allight): We probably could use something like the SharedQueryEngine to
// keep this alive between runs. Needing to clear out forced values makes it a
// little less useful/more complicated but nothing all that difficult.
class ReassociationCache : public LazyNodeInfo<std::optional<SignednessPair>> {
 public:
  explicit ReassociationCache(const QueryEngine& qe) : qe_(qe) {}
  const QueryEngine& query_engine() const { return qe_; }

 protected:
  LeafTypeTree<std::optional<SignednessPair>> ComputeInfo(
      Node* node,
      absl::Span<const LeafTypeTree<std::optional<SignednessPair>>* const>
          operand_infos) const final {
    if (!node->GetType()->IsBits()) {
      return LeafTypeTree<std::optional<SignednessPair>>::CreateFromFunction(
                 node->GetType(), [](auto v) { return std::nullopt; })
          .value();
    }
    OneShotReassociationVisitor vis(qe_, operand_infos);
    CHECK_OK(node->VisitSingleNode(&vis)) << "Generating for " << node;

    auto result_status = std::move(vis).Take();
    CHECK_OK(result_status) << "For " << node;
    VLOG(5) << "reassociation info for " << node << " is "
            << result_status.value();
    return LeafTypeTree<std::optional<SignednessPair>>::CreateSingleElementTree(
        node->GetType(), std::move(result_status).value());
  }

  absl::Status MergeWithGiven(
      std::optional<SignednessPair>& info,
      const std::optional<SignednessPair>& given) const final {
    return absl::InternalError("Cannot merge reassociation information!");
  }

 private:
  const QueryEngine& qe_;
};

class Reassociation {
 public:
  Reassociation(FunctionBase* fb, const QueryEngine& qe,
                OptimizationContext& context)
      : fb_(fb), cache_(qe), context_(context) {}

  absl::StatusOr<bool> Reassociate() {
    bool changed = false;
    XLS_RETURN_IF_ERROR(cache_.Attach(fb_).status());
    if (VLOG_IS_ON(5) && fb_->IsFunction()) {
      VLOG(5) << "Initial reassociations:\n"
              << fb_->AsFunctionOrDie()->DumpIrWithAnnotations(
                     [&](Node* n) -> std::optional<std::string> {
                       auto info = *cache_.GetInfo(n);
                       if (!info.type()->IsBits()) {
                         return std::nullopt;
                       }
                       return info.Get({})->DumpShort();
                     });
    }
    // Set of all nodes which were used as leafs in some reassociation.
    absl::flat_hash_set<Node*> used_as_leaf;
    absl::flat_hash_set<Node*> ever_used_as_leaf;
    bool first = true;
    // We need to run to fixedpoint trying to reassociate nodes that we discover
    // are used in leaf positions of earlier reassociations.
    while (first || !used_as_leaf.empty()) {
      bool was_first = first;
      absl::flat_hash_set<Node*> current_worklist = std::move(used_as_leaf);
      used_as_leaf.clear();
      first = false;
      // We always need to run through the candidates in topo order to ensure we
      // don't duplicate nodes.
      for (Node* node : iter::filter(
               [&](Node* n) -> bool {
                 return was_first || current_worklist.contains(n);
               },
               context_.TopoSort(fb_))) {
        if (!node->GetType()->IsBits()) {
          VLOG(5) << "Skipping " << node << " due to non-bits types";
          continue;
        }

        // NB Intentional copy to avoid reference invalidation.
        SignednessPair elements = *cache_.GetInfo(node)->Get({});
        VLOG(3) << "Examining " << node << " with " << elements;

        const AssociativeElements& unsigned_elements = elements.unsigned_values;
        const AssociativeElements& signed_elements = elements.signed_values;
        if (unsigned_elements.is_leaf() && signed_elements.is_leaf()) {
          // Nothing we can reassociate.
          VLOG(4)
              << "    - Skipping " << node
              << " since it is a leaf node for both signed and unsigned use.";
          continue;
        }
        if (unsigned_elements.full_variable_count() == 0 ||
            signed_elements.full_variable_count() == 0) {
          VLOG(4) << "    - Skipping " << node
                  << " since it is entierly made up of constants so later "
                     "constant-prop will simplify it.";
          continue;
        }

        bool needs_materialization =
            used_as_leaf.contains(node) || fb_->HasImplicitUse(node) ||
            IsUserNotPartOfSameReassociationSet(node, elements) ||
            IsPartOfMultipleReassociationSets(node, elements);
        if (!needs_materialization) {
          VLOG(4) << "    - Skipping " << node
                  << " since no user requires materialization.";
          continue;
        }

        // We've decided that reassociation will happen or be determined to have
        // already happened/be optimal.
        VLOG(4) << "    - Node " << node
                << " is a candidate for reassociation.";

        // First try to materialize without affecting any other users. Possible
        // if the elements can be expressed as a single variable and a constant.
        if (elements.unsigned_values.variables().size() == 1 ||
            elements.signed_values.variables().size() == 1) {
          XLS_ASSIGN_OR_RETURN(
              std::optional<Node*> replaced_with_op_with_constant,
              MaybeReplaceWithSingleConstantOp(node, unsigned_elements,
                                               signed_elements));
          if (replaced_with_op_with_constant) {
            changed = true;
            VLOG(3) << "    - Reassociated " << node
                    << " with only one non-constant leaf node into "
                    << *replaced_with_op_with_constant;
            // We've done a replacement but its with a constant value so we can
            // leave the elements of users unchanged.
            LeafTypeTree<std::optional<SignednessPair>> given =
                LeafTypeTree<std::optional<SignednessPair>>::
                    CreateSingleElementTree(node->GetType(), elements);
            // Mark that nothing actually changed.
            XLS_RETURN_IF_ERROR(
                cache_.SetForced(*replaced_with_op_with_constant, given)
                    .status());
            XLS_RETURN_IF_ERROR(cache_.SetForced(node, given).status());
            XLS_RETURN_IF_ERROR(
                cache_.SetForced(elements.signed_values.node(), given)
                    .status());
            XLS_RETURN_IF_ERROR(
                cache_.SetForced(elements.unsigned_values.node(), given)
                    .status());
          }
          continue;
        }

        auto is_eligible =
            [&](const AssociativeElements& elements) -> absl::StatusOr<bool> {
          VLOG(4) << "    - elements (" << elements.element_count()
                  << "): " << elements.ElementsToString();
          VLOG(4) << "    - depth: " << elements.depth();
          if (elements.is_leaf()) {
            VLOG(4) << "    - " << elements.node()
                    << " not eligible due to being a leaf";
            return false;
          }
          XLS_RET_CHECK_GE(
              elements.depth(),
              CeilOfLog2(elements.ElementCountWithMaxOneConstant()))
              << elements << " counted depth is smaller than minimum depth of "
              << CeilOfLog2(elements.ElementCountWithMaxOneConstant());
          if (elements.depth() ==
                  CeilOfLog2(elements.ElementCountWithMaxOneConstant()) &&
              elements.constants().size() <= 1) {
            VLOG(4) << "    - " << elements.node()
                    << " not eligible current depth being same as minimum and "
                       "only a single/zero constants";
            // We are at the exact min depth and we can't combine any constants.
            return false;
          }
          return true;
        };
        VLOG(4) << "  - Checking eligibility of signed transform:";
        XLS_ASSIGN_OR_RETURN(bool signed_eligible,
                             is_eligible(signed_elements));
        VLOG(4) << "  - Checking eligibility of unsigned transform:";
        XLS_ASSIGN_OR_RETURN(bool unsigned_eligible,
                             is_eligible(unsigned_elements));
        VLOG(4) << "    - For node " << node
                << " can perform signed: " << std::boolalpha << signed_eligible
                << ", unsigned: " << unsigned_eligible;
        if (!signed_eligible && !unsigned_eligible) {
          VLOG(4) << "    - Node " << node
                  << " not being reassociated because current computation is "
                     "optimal.";
          // If this is optimal later users shouldn't mess with us. This is
          // needed to prevent later users from reasociating with this ones
          // leafs. NB To get here each is either already optimal or leaf
          // already.
          SignednessPair leaf{
              .signed_values = AssociativeElements::MakeLeaf(node),
              .unsigned_values = AssociativeElements::MakeLeaf(node),
          };
          XLS_RETURN_IF_ERROR(
              cache_
                  .SetForced(node, LeafTypeTree<std::optional<SignednessPair>>::
                                       CreateSingleElementTree(node->GetType(),
                                                               std::move(leaf)))
                  .status());
          continue;
        }
        bool is_signed =
            signed_eligible &&
            (!unsigned_eligible ||
             signed_elements.ElementCountWithMaxOneConstant() >
                 unsigned_elements.ElementCountWithMaxOneConstant());
        VLOG(3) << "  - Selected " << (is_signed ? "signed" : "unsigned")
                << " as reassociation data.";
        const AssociativeElements& selected =
            is_signed ? signed_elements : unsigned_elements;
        // Mark all leafs for re-examination since we know they weren't
        // reassociated through anymore. Only retry each node once.
        for (const auto& nd : selected.variables()) {
          if (ever_used_as_leaf.insert(nd.node).second) {
            used_as_leaf.insert(nd.node);
          }
        }
        XLS_ASSIGN_OR_RETURN(Node * replacement,
                             ReassociateOneOperation(selected, is_signed));

        LeafTypeTree<std::optional<SignednessPair>> leaf =
            LeafTypeTree<std::optional<SignednessPair>>::
                CreateSingleElementTree(
                    node->GetType(),
                    SignednessPair{
                        .signed_values =
                            AssociativeElements::MakeLeaf(replacement),
                        .unsigned_values =
                            AssociativeElements::MakeLeaf(replacement)});
        // Make the cache consider the replaced node an leaf for anything that
        // comes after.
        XLS_RETURN_IF_ERROR(cache_.SetForced(node, leaf).status());
        XLS_RETURN_IF_ERROR(
            cache_.SetForced(elements.signed_values.node(), leaf).status());
        XLS_RETURN_IF_ERROR(
            cache_.SetForced(elements.unsigned_values.node(), leaf).status());
        XLS_RETURN_IF_ERROR(cache_.SetForced(replacement, leaf).status());
        changed = true;
      }
    }
    VLOG(3) << fb_->name() << (changed ? " changed" : " did not change");
    return changed;
  }

 private:
  // Check if any user is part of a different reassociation, either an identity
  // itself or using this node as a leaf.
  bool IsUserNotPartOfSameReassociationSet(Node* node,
                                           const SignednessPair& elements) {
    return absl::c_any_of(node->users(), [&](Node* user) -> bool {
      if (!user->GetType()->IsBits()) {
        // Non-bits users can't be reassociated.
        VLOG(4) << "      - " << node << " has non-bits user: " << user;
        return true;
      }
      auto user_elements = *cache_.GetInfo(user)->Get({});
      // If the user can't be reassociated itself or has a different op.
      auto is_leaf_user = [&](const AssociativeElements& user_elements,
                              const AssociativeElements& elements) {
        return user_elements.is_leaf() || user_elements.op() != elements.op() ||
               absl::c_any_of(user_elements.variables(),
                              [&](const AssociativeElements::NodeData& data) {
                                return data.node == elements.node();
                              });
      };
      bool unsigned_res =
          is_leaf_user(user_elements.unsigned_values, elements.unsigned_values);
      bool signed_res =
          is_leaf_user(user_elements.signed_values, elements.signed_values);
      bool res = unsigned_res && signed_res;
      if (res) {
        VLOG(4) << "      - " << node << " has identity/leaf user: " << user;
      }
      return res;
    });
  }

  // Check if this single node is used in two different reassociation sets.
  // TODO(allight): Technically reassociating this node might not be needed. Eg
  // if you have something like
  //
  // x = sum(...)
  // l = sum(..., x)
  // r = sum(..., x)
  // y = l + r
  // send(y)
  //
  // it looks like x is used in both the l and r reassociation sets but since
  // the actual user is only y which dominates both we could avoid doing
  // reassociation here. Determining this fact is difficult however. For now we
  // won't bother.
  bool IsPartOfMultipleReassociationSets(Node* node,
                                         const SignednessPair& elements) {
    return absl::c_count_if(node->users(), [&](Node* user) {
             if (!user->GetType()->IsBits()) {
               return true;
             }
             auto user_elements = *cache_.GetInfo(user)->Get({});
             bool unsigned_constant_fn =
                 user_elements.unsigned_values.variables().size() == 1;
             bool signed_constant_fn =
                 user_elements.signed_values.variables().size() == 1;
             return !unsigned_constant_fn && !signed_constant_fn;
           }) > 1;
  }

  absl::StatusOr<Node*> ReassociateOneOperation(
      const AssociativeElements& elements, bool is_signed) {
    using NodeData = AssociativeElements::NodeData;
    // Split all constant operations into a single element.
    XLS_RET_CHECK(!elements.is_leaf())
        << "Reassociate leaf node: " << elements.node();
    XLS_RET_CHECK(elements.op())
        << "not associative operation" << elements.node();
    VLOG(2) << "Reassociating operation " << elements.node()
            << " with elements: [" << elements.ElementsToString() << "]";
    std::vector<NodeData> variable_elements(elements.variables().begin(),
                                            elements.variables().end());
    // Make sure that variable elements are sorted consistently to ensure that
    // CSE will be able to merge them. Sort by bit-count, non-reassociativity,
    // negatedness then id.
    auto is_basic_candidate = [&](Node* n) -> bool {
      auto elem = cache_.GetInfo(n)->Get({});
      return elem && (!elem->signed_values.is_leaf() ||
                      !elem->unsigned_values.is_leaf());
    };
    auto elements_cmp = [&](const NodeData& a, const NodeData& b) {
      auto bit_count_comp = a.node->BitCountOrDie() <=> b.node->BitCountOrDie();
      if (bit_count_comp != std::strong_ordering::equal) {
        return bit_count_comp == std::strong_ordering::less;
      }
      // Try to push nodes which we can feasibly reassociate again to the right
      // since we build the tree left biased further right can be at lower
      // depths which is good if we end up being able to reassociate again.
      bool lhs_is_reassoc_candidate = is_basic_candidate(a.node);
      bool rhs_is_reassoc_candidate = is_basic_candidate(b.node);
      if (lhs_is_reassoc_candidate != rhs_is_reassoc_candidate) {
        return !lhs_is_reassoc_candidate;
      }
      auto id_cmp = a.node->id() <=> b.node->id();
      if (id_cmp != std::strong_ordering::equal) {
        return id_cmp == std::strong_ordering::less;
      }
      return (a.needs_negate != b.needs_negate) ? a.needs_negate : false;
    };
    absl::c_sort(variable_elements, elements_cmp);
    std::string associative_sum_name =
        elements.node()->HasAssignedName()
            ? absl::StrCat(elements.node()->GetNameView(),
                           "_associative_element")
            : "";
    auto sum_width = [&](int64_t lhs_width, int64_t rhs_width) -> int64_t {
      int64_t max = elements.node()->BitCountOrDie();
      switch (*elements.op()) {
        case Op::kAdd:
        case Op::kSub: {
          return std::min(std::max(lhs_width, rhs_width) + 1, max);
        }
        case Op::kSMul:
        case Op::kUMul: {
          return std::min(lhs_width + rhs_width, max);
        }
        default:
          LOG(FATAL) << "Unexpected op: " << *elements.op();
          break;
      }
    };
    auto make_sum = [&](NodeData lhs,
                        NodeData rhs) -> absl::StatusOr<NodeData> {
      int64_t width =
          sum_width(lhs.node->BitCountOrDie(), rhs.node->BitCountOrDie());
      switch (*elements.op()) {
        case Op::kAdd: {
          // Determine how to materialize this operation avoiding neg's as much
          // as possible. If we need to zero-extend a negative number we can't
          // transform to a simple sub.
          // l_compat = is_signed || width(l) == sum_width(r, l)
          // r_compat = is_signed || width(r) == sum_width(r, l)
          // both_compat = l_compat && r_compat
          // lhs_ext = ext(lhs)
          // rhs_ext = ext(rhs)
          // cases (rhs_ext, lhs_ext):
          //   (+L, +R) => +Add(L, R)
          //   (-L, +R) => +Sub(R, L) if l_compat    else Add(-L, R)
          //   (+L, -R) => +Sub(L, R) if r_compat    else Add(L, -R)
          //   (-L, -R) => -Add(R, L) if both_compat else Add(-L, -R)
          bool l_width_compat = is_signed || lhs.node->BitCountOrDie() == width;
          bool r_width_compat = is_signed || rhs.node->BitCountOrDie() == width;
          bool both_width_compat = l_width_compat && r_width_compat;

          // Force extend to output width. Narrowing will crush these down
          // later if possible.
          if (lhs.node->BitCountOrDie() != width) {
            XLS_ASSIGN_OR_RETURN(lhs.node,
                                 fb_->MakeNode<ExtendOp>(
                                     elements.node()->loc(), lhs.node, width,
                                     is_signed ? Op::kSignExt : Op::kZeroExt));
          }
          if (rhs.node->BitCountOrDie() != width) {
            XLS_ASSIGN_OR_RETURN(rhs.node,
                                 fb_->MakeNode<ExtendOp>(
                                     elements.node()->loc(), rhs.node, width,
                                     is_signed ? Op::kSignExt : Op::kZeroExt));
          }
          Op real_op;
          bool res_needs_negate;
          if (!lhs.needs_negate && !rhs.needs_negate) {
            real_op = Op::kAdd;
            res_needs_negate = false;
          } else if (lhs.needs_negate && !rhs.needs_negate) {
            res_needs_negate = false;
            if (l_width_compat) {
              real_op = Op::kSub;
              std::swap(lhs.node, rhs.node);
            } else {
              real_op = Op::kAdd;
              XLS_ASSIGN_OR_RETURN(lhs.node, lhs.ToNode());
              lhs.needs_negate = false;
            }
          } else if (!lhs.needs_negate && rhs.needs_negate) {
            res_needs_negate = false;
            if (r_width_compat) {
              real_op = Op::kSub;
            } else {
              real_op = Op::kAdd;
              XLS_ASSIGN_OR_RETURN(rhs.node, rhs.ToNode());
              rhs.needs_negate = false;
            }
          } else {
            XLS_RET_CHECK(lhs.needs_negate && rhs.needs_negate);
            real_op = Op::kAdd;
            if (both_width_compat) {
              res_needs_negate = true;
            } else {
              res_needs_negate = false;
              XLS_ASSIGN_OR_RETURN(rhs.node, rhs.ToNode());
              XLS_ASSIGN_OR_RETURN(lhs.node, lhs.ToNode());
              lhs.needs_negate = false;
              rhs.needs_negate = false;
            }
          }
          // LHS and RHS are now in the correct order and real_op and
          // res_needs_negate say how to combine them.

          // Either a sum or the negative of a sum.
          XLS_ASSIGN_OR_RETURN(Node * sum,
                               fb_->MakeNodeWithName<BinOp>(
                                   elements.node()->loc(), lhs.node, rhs.node,
                                   real_op, associative_sum_name));
          return NodeData{.node = sum, .needs_negate = res_needs_negate};
        }
        case Op::kSMul:
        case Op::kUMul: {
          // cases (lhs, rhs):
          //  (+L, +R) => +Mul(L, R)
          //  (-L, +R) => -Mul(L, R)
          //  (+L, -R) => -Mul(L, R)
          //  (-L, -R) => +Mul(L, R)
          bool res_needs_negate = lhs.needs_negate != rhs.needs_negate;
          XLS_ASSIGN_OR_RETURN(
              Node * prod, fb_->MakeNodeWithName<ArithOp>(
                               elements.node()->loc(), lhs.node, rhs.node,
                               width, *elements.op(), associative_sum_name));
          return NodeData{.node = prod, .needs_negate = res_needs_negate};
        }
        case Op::kSub:
          return absl::InternalError("Sub should have been normalized to add.");
        default:
          return absl::InternalError(absl::StrFormat(
              "%v is not a supported associative operation", *elements.op()));
      }
    };
    if (!elements.constants().empty()) {
      // Add the constants to the final result. Let ConstProp deal with actually
      // reducing the constant to a single value.
      XLS_ASSIGN_OR_RETURN(
          std::back_inserter(variable_elements),
          CreateTreeSum(make_sum, elements.constants(), elements.op()));
      // Resort the variable elements to avoid issues where larger adds then
      // needed are consistently created.
      absl::c_sort(variable_elements, elements_cmp);
    }

    NodeData replacement;
    if (variable_elements.empty()) {
      // Only possible if all the elements canceled. Weird but whatever.
      XLS_RET_CHECK_EQ(elements.op().value(), Op::kAdd)
          << "Only an add should be able to cancel to empty sum: " << elements;
      replacement.needs_negate = false;
      XLS_ASSIGN_OR_RETURN(
          replacement.node,
          fb_->MakeNode<Literal>(
              elements.node()->loc(),
              Value(UBits(0, elements.node()->BitCountOrDie()))));
    } else {
      XLS_ASSIGN_OR_RETURN(
          replacement,
          CreateTreeSum(make_sum, absl::MakeConstSpan(variable_elements),
                        elements.op()));
    }
    // give the replacement the original name.
    std::string original_name =
        elements.node()->HasAssignedName() ? elements.node()->GetName() : "";
    if (replacement.node->BitCountOrDie() != elements.node()->BitCountOrDie()) {
      XLS_ASSIGN_OR_RETURN(
          replacement.node,
          fb_->MakeNode<ExtendOp>(elements.node()->loc(), replacement.node,
                                  elements.node()->BitCountOrDie(),
                                  is_signed ? Op::kSignExt : Op::kZeroExt));
    }
    XLS_ASSIGN_OR_RETURN(Node * result, replacement.ToNode());
    XLS_RETURN_IF_ERROR(elements.node()->ReplaceUsesWith(result));
    if (elements.node()->HasAssignedName()) {
      elements.node()->ClearName();
      result->SetNameDirectly(original_name);
    }
    return result;
  }

  absl::StatusOr<std::optional<Node*>> MaybeReplaceWithSingleConstantOp(
      Node* node, const AssociativeElements& unsigned_elements,
      const AssociativeElements& signed_elements) {
    // See if we can turn this into a single expression with a constant on one
    // side.
    int64_t unsigned_not_fully_known = unsigned_elements.variables().size();
    int64_t signed_not_fully_known = signed_elements.variables().size();
    if (signed_not_fully_known == 0 || unsigned_not_fully_known == 0) {
      VLOG(4) << "  - Node " << node << " is fully known along some path.";
      // already a constant expression. Don't bother doing anything.
      return std::nullopt;
    }
    // Don't mess with operations which are just a single binary operation.
    bool unsigned_eligible = unsigned_elements.full_leaf_count() > 2 &&
                             unsigned_not_fully_known == 1;
    bool signed_eligible =
        signed_elements.full_leaf_count() > 2 && signed_not_fully_known == 1;
    VLOG(4) << "   - Node " << node
            << " signed-constant eligible: " << std::boolalpha
            << signed_eligible
            << ", unsigned-constant eligible: " << unsigned_eligible;
    // Note that by comparing the element_count we ensure that we take the
    // bigger tree.
    if (unsigned_eligible &&
        (!signed_eligible || signed_elements.element_count() <=
                                 unsigned_elements.element_count())) {
      XLS_ASSIGN_OR_RETURN(
          Node * replacement,
          ReassociateOneOperation(unsigned_elements, /*is_signed=*/false));
      return replacement;
    }
    if (signed_eligible) {
      // NB If unsigned was better that branch would have already been taken.
      XLS_ASSIGN_OR_RETURN(
          Node * replacement,
          ReassociateOneOperation(signed_elements, /*is_signed=*/true));
      return replacement;
    }
    return std::nullopt;
  }
  FunctionBase* fb_;
  ReassociationCache cache_;
  OptimizationContext& context_;
};

}  // namespace

absl::StatusOr<bool> ReassociationPass::RunOnFunctionBaseInternal(
    FunctionBase* f, const OptimizationPassOptions& options,
    PassResults* results, OptimizationContext& context) const {
  UnionQueryEngine query_engine = UnionQueryEngine::Of(
      StatelessQueryEngine(),
      GetSharedQueryEngine<BitCountQueryEngine>(context, f),
      GetSharedQueryEngine<LazyTernaryQueryEngine>(context, f));
  XLS_RETURN_IF_ERROR(query_engine.Populate(f).status());
  Reassociation reassoc(f, query_engine, context);
  return reassoc.Reassociate();
}

}  // namespace xls
